# -*- encoding: utf-8 -*-
"""
sft_train.py
============

用 HuggingFace TRL + LoRA 把 Qwen3-0.6B 微调为路由分类器。
输入数据格式见 merge_labels.py（每行 {"messages": [...]}）。

最小化训练脚本，依赖：
    pip install torch transformers>=4.45 trl>=0.11 peft>=0.13 datasets accelerate

典型用法：
    python sft_train.py \
        --base_model Qwen/Qwen3-0.6B \
        --train_file data/routing_train.jsonl \
        --val_file   data/routing_val.jsonl \
        --output_dir checkpoints/router_v1 \
        --epochs 3 --bs 16 --lr 1e-4 --lora_r 16

单卡 A100/3090/4090 上约 0.5–2 小时跑完（取决于数据规模）。
"""
from __future__ import annotations
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_total_limit", type=int, default=2)
    args = ap.parse_args()

    train_raw = load_jsonl(args.train_file)
    val_raw = load_jsonl(args.val_file)
    print(f"train {len(train_raw)} / val {len(val_raw)}")

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_raw])
    val_ds = Dataset.from_list([{"messages": r["messages"]} for r in val_raw])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 常见 attention/MLP 投影名称，Qwen 用这些命名
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=max(1, args.bs // 2),
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_seq_length=args.max_seq_length,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
