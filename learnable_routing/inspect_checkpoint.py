# -*- encoding: utf-8 -*-
"""
inspect_checkpoint.py
=====================

用现有 LoRA checkpoint 在 val 集上跑 per-class accuracy，独立于训练脚本，方便快速判断
"训练效果到底好不好" vs "评测 bug 让结果看起来差"。

用法（在 STR/learnable_routing 里）：

    python3 inspect_checkpoint.py \
        --base_model /amax/.../Qwen3-0.6B \
        --lora_ckpt checkpoints/router_v2 \
        --val_file data/routing_val.jsonl \
        --max_new_tokens 32 \
        --print_samples 10
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ("full", "partial", "filter")
LABEL_RE = re.compile(r"\b(full|partial|filter)\b", re.IGNORECASE)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gold_of(r):
    return r["messages"][-1]["content"].strip().lower()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_ckpt", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--print_samples", type=int, default=10,
                    help="额外打印多少条 raw generation 用于人眼检查")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    print(f"[load] base = {args.base_model}")
    print(f"[load] lora = {args.lora_ckpt}")

    tok = AutoTokenizer.from_pretrained(args.lora_ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.lora_ckpt).to(device)
    model.eval()

    val = load_jsonl(args.val_file)
    print(f"[load] val = {len(val)} samples")
    gold_dist = Counter(gold_of(r) for r in val)
    print(f"[load] val gold distribution: {dict(gold_dist)}")

    by_class = {l: [0, 0] for l in LABELS}  # [correct, total]
    pred_dist = Counter()
    raw_samples_printed = 0

    for i, r in enumerate(val):
        msgs = r["messages"][:2]
        gold = gold_of(r)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new = out[0, ids["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=False).lower()
        after = text.split("</think>", 1)[-1] if "</think>" in text else text
        m = LABEL_RE.search(after) or LABEL_RE.search(text)
        pred = m.group(1).lower() if m else "NONE"
        pred_dist[pred] += 1

        if pred in LABELS:
            by_class[gold][1] += 1
            if pred == gold:
                by_class[gold][0] += 1
        else:
            by_class[gold][1] += 1  # 视为答错

        if raw_samples_printed < args.print_samples:
            print(f"  [{i:4d}] gold={gold:8s} pred={pred:8s}  raw={text[:120]!r}")
            raw_samples_printed += 1

    print()
    print("=== Per-class accuracy ===")
    for l in LABELS:
        c, t = by_class[l]
        print(f"  {l:8s}: acc={c/max(t,1):.3f}  (correct {c}/{t})")
    macro = sum(c / max(t, 1) for c, t in by_class.values()) / len(LABELS)
    print(f"  macro acc = {macro:.3f}")
    print(f"  pred distribution = {dict(pred_dist)}")


if __name__ == "__main__":
    main()
