# -*- encoding: utf-8 -*-
"""
sft_train_weighted.py  (v2: completion-only masking + heavier defaults)
======================================================================

跟原 sft_train.py 的区别：

1. **Completion-only loss masking**：手动预 tokenize，把 system+user 段的 labels
   设为 -100。这样 cross-entropy 只在答案 token (`full`/`partial`/`filter`) 上生效，
   类权重不会被几百个 user/system token 稀释。**这是 v2 的核心修复**。

2. **少数类过采样** `--oversample {none|inverse|sqrt}`：
   - inverse : 复制次数 = round(max_count / class_count)；纯反频次
   - sqrt    : 复制次数 = round(sqrt(max_count / class_count))；缓和
   - none    : 不过采样

3. **类权重 CE** `--label_weights "full=1.0,partial=8.0,filter=15.0"`：
   答案 token 处的 cross-entropy 乘以对应类权重。默认值已调激进（v1 是 3/5，
   完全压不住；v2 默认 8/15 起步）。

4. **per-class accuracy on val** 训练结束自动打印，用来判断是否还在退化。

推荐起点（如果 v1 实验里 partial/filter acc=0.000）：

    python sft_train_weighted.py \
        --base_model /path/to/Qwen3-0.6B \
        --train_file data/routing_train.jsonl \
        --val_file   data/routing_val.jsonl \
        --output_dir checkpoints/router_v2 \
        --oversample inverse \
        --label_weights "full=1.0,partial=8.0,filter=15.0" \
        --epochs 5 --bs 16 --lr 2e-4 --lora_r 32

若 partial/filter acc 反过来跑到 0.9+ 而 full < 0.5（过头了），就把权重调小
（partial=3, filter=5）或 oversample 退回 sqrt。
"""
from __future__ import annotations
import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


LABELS = ("full", "partial", "filter")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_label_weights(s: str) -> Dict[str, float]:
    if not s:
        return {l: 1.0 for l in LABELS}
    out = {l: 1.0 for l in LABELS}
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip().lower()
        if k not in LABELS:
            raise ValueError(f"unknown label key: {k}")
        out[k] = float(v)
    return out


def get_label_of(example: Dict) -> str:
    return example["messages"][-1]["content"].strip().lower()


def oversample(rows: List[Dict], mode: str) -> List[Dict]:
    cnt = Counter(get_label_of(r) for r in rows)
    print(f"[oversample] original distribution: {dict(cnt)}")
    if mode == "none":
        return list(rows)
    max_n = max(cnt.values())
    rep: Dict[str, int] = {}
    for k, v in cnt.items():
        if v == 0:
            rep[k] = 0
            continue
        if mode == "inverse":
            rep[k] = max(1, round(max_n / v))
        elif mode == "sqrt":
            rep[k] = max(1, round(math.sqrt(max_n / v)))
        else:
            raise ValueError(f"oversample must be none|inverse|sqrt, got {mode!r}")
    print(f"[oversample] repeat counts: {rep}")
    out = []
    for r in rows:
        out.extend([r] * rep[get_label_of(r)])
    cnt2 = Counter(get_label_of(r) for r in out)
    print(f"[oversample] new distribution: {dict(cnt2)} (size {len(out)})")
    return out


# ─────────────────────────────────────────────────────────────────────
# 预 tokenize：完全 mask 掉 prefix（system+user+chat-template framing）
# ─────────────────────────────────────────────────────────────────────

class RoutingDataset(TorchDataset):
    """每条样本被 tokenize 成 input_ids/labels，labels 的 prefix 部分全设 -100。"""

    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: List[Dict] = []
        skipped = 0
        for r in rows:
            ex = self._build(r["messages"])
            if ex is None:
                skipped += 1
                continue
            self.examples.append(ex)
        if skipped:
            print(f"[dataset] skipped {skipped} examples (couldn't align prefix/full)")
        print(f"[dataset] kept {len(self.examples)} examples")

    def _build(self, messages: List[Dict]) -> Dict:
        tok = self.tokenizer
        # 不带 assistant 答案的 prefix，要 add_generation_prompt，让 model 看到的就是 prefix 末尾
        prefix_text = tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        # 完整对话（含 assistant 答案）
        full_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        if not full_text.startswith(prefix_text):
            # chat template 可能在 assistant 头加额外 tag，掉头退回保守做法：
            # 找到 prefix_text 的最长前缀
            common = 0
            for i in range(min(len(prefix_text), len(full_text))):
                if prefix_text[i] != full_text[i]:
                    break
                common += 1
            if common < len(prefix_text) * 0.5:
                return None
            prefix_text = prefix_text[:common]

        prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        # 把 eos 也放进答案部分，让模型学会停
        if tok.eos_token_id is not None and (not full_ids or full_ids[-1] != tok.eos_token_id):
            full_ids = full_ids + [tok.eos_token_id]

        # 截断
        full_ids = full_ids[: self.max_len]
        if len(prefix_ids) >= len(full_ids):
            return None  # 答案被截掉了

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
        labels = labels[: len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class PadCollator:
    """简单左对齐 padding collator。pad_token_id 给 input_ids，-100 给 labels。"""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = max_len - n
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────
# 自定义 Trainer：在答案 token 处按类加权 CE
# ─────────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, *args, label_token_ids: Dict[str, List[int]] = None,
                 label_weight_values: Dict[str, float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # label_token_ids: 一个 label → 多个可能的 token id（带空格 / 不带空格 / 大小写等）
        self.label_token_ids = label_token_ids or {}
        self.label_weight_values = label_weight_values or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce = CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        per_token = ce(flat_logits, flat_labels)

        weights = torch.ones_like(per_token)
        for label_name, tids in self.label_token_ids.items():
            w = self.label_weight_values.get(label_name, 1.0)
            if w == 1.0:
                continue
            for tid in tids:
                if tid is None:
                    continue
                weights[flat_labels == tid] = w

        mask = (flat_labels != -100).float()
        total = (mask * weights).sum().clamp(min=1.0)
        loss = (per_token * mask * weights).sum() / total
        return (loss, outputs) if return_outputs else loss


def resolve_label_token_ids(tokenizer) -> Dict[str, List[int]]:
    """
    返回每个 label 所有可能的单 token id：
      - 无前缀（chat template 在 ...</think>\\n\\n 之后直接塞 'filter'）：主要走这条
      - 带空格前缀（fallback：万一未来 chat template 改成空格分隔）
    Qwen3-0.6B 实测：'full'/'partial'/'filter' 都是单 token，无歧义。
    """
    out: Dict[str, List[int]] = {}
    for label in LABELS:
        ids: List[int] = []
        for variant in (label, " " + label):
            v = tokenizer.encode(variant, add_special_tokens=False)
            if len(v) == 1 and v[0] not in ids:
                ids.append(v[0])
        if not ids:
            # 兜底：万一两种 variant 都不是单 token，取无前缀的最后一个
            v = tokenizer.encode(label, add_special_tokens=False)
            ids.append(v[-1])
        out[label] = ids
    return out


# ─────────────────────────────────────────────────────────────────────
# 验证集 per-class accuracy
# ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def per_class_eval(model, tokenizer, val_rows: List[Dict], device: str = "cuda") -> Dict[str, float]:
    """
    评测时要注意 Qwen3 默认开 thinking mode，answer 在 <think>...\\n\\n</think>\\n\\nLABEL 之后，
    至少要 max_new_tokens=20+ 才能看到 LABEL。我们解析整段生成文本里 </think> 之后的 LABEL。
    """
    model.eval()
    import re
    label_re = re.compile(r"\b(full|partial|filter)\b", re.IGNORECASE)
    by_class = {l: [0, 0] for l in LABELS}
    for r in val_rows:
        msgs = r["messages"][:2]
        gold = get_label_of(r)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=32,  # 容纳 <think>\n\n</think>\n\nLABEL<|im_end|>，至少 6-8 token
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False).lower()
        # 优先从 </think> 之后找
        after = text.split("</think>", 1)[-1] if "</think>" in text else text
        m = label_re.search(after) or label_re.search(text)
        pred = m.group(1).lower() if m else "full"
        by_class[gold][1] += 1
        if pred == gold:
            by_class[gold][0] += 1
    return {l: (c / max(t, 1)) for l, (c, t) in by_class.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="梯度累积步数；显存紧张时 bs=4, grad_accum=4 等效 bs=16")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="开启 gradient checkpointing 进一步省显存（训练慢约 30%）")
    ap.add_argument("--oversample", choices=["none", "inverse", "sqrt"], default="inverse",
                    help="少数类过采样策略；inverse 默认（fully balance）；如果训练失稳改 sqrt")
    ap.add_argument("--label_weights", default="full=1.0,partial=8.0,filter=15.0",
                    help="形如 full=1.0,partial=8.0,filter=15.0；CE 答案 token 的乘性权重")
    ap.add_argument("--no_per_class_eval", action="store_true")
    args = ap.parse_args()

    train_raw = load_jsonl(args.train_file)
    val_raw = load_jsonl(args.val_file)
    print(f"raw train {len(train_raw)} / val {len(val_raw)}")

    train_raw = oversample(train_raw, args.oversample)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_weights = parse_label_weights(args.label_weights)
    label_token_ids = resolve_label_token_ids(tokenizer)
    print(f"[weights] label_weights = {label_weights}")
    print(f"[weights] label_token_ids = {label_token_ids}")

    train_ds = RoutingDataset(train_raw, tokenizer, max_len=args.max_seq_length)
    val_ds = RoutingDataset(val_raw, tokenizer, max_len=args.max_seq_length)
    collator = PadCollator(tokenizer)

    # 检查：随机一个 train 样本，确认只有答案部分有非 -100 标签
    sample = train_ds[0]
    n_active = sum(1 for x in sample["labels"] if x != -100)
    print(f"[sanity] first sample has {len(sample['input_ids'])} tokens, "
          f"{n_active} non-masked labels (should be small, e.g. 2-4)")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True, device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=max(1, args.bs // 2),
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        label_token_ids=label_token_ids,
        label_weight_values=label_weights,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved to {args.output_dir}")

    if not args.no_per_class_eval:
        print("\n=== Per-class accuracy on val ===")
        try:
            device = next(model.parameters()).device.type
            acc = per_class_eval(model, tokenizer, val_raw, device=device)
            for l in LABELS:
                cnt = sum(1 for r in val_raw if get_label_of(r) == l)
                print(f"  {l:8s}: acc={acc[l]:.3f}  (n={cnt})")
            macro = sum(acc.values()) / len(acc)
            print(f"  macro acc = {macro:.3f}")
        except Exception as e:
            print(f"per-class eval failed: {e}")


if __name__ == "__main__":
    main()
