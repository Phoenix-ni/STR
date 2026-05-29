# -*- encoding: utf-8 -*-
"""
合并三个 benchmark 的 oracle 标签 → 训练集 / 验证集。

输入：build_oracle_labels.py 输出的若干 jsonl
输出：train.jsonl / val.jsonl，每行：
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "filter" | "partial" | "full"},
        ],
        "meta": {
            "sample_id": "...", "benchmark": "...", "label": "..."
        }
    }

用法：
    python merge_labels.py \
        --inputs out/routing_labels_tableeval.jsonl out/routing_labels_wtq.jsonl out/routing_labels_tablebench.jsonl \
        --out_train data/routing_train.jsonl \
        --out_val data/routing_val.jsonl \
        --val_ratio 0.1
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List


SYSTEM_PROMPT = (
    "你是表格路由分类器。给定用户的查询和表格元信息，预测最适合的路由强度：\n"
    "- filter  : 强筛选，只保留命中的实体（适合稀疏证据检索）\n"
    "- partial : 保守过滤 + 覆盖率熔断（适合通用问答）\n"
    "- full    : 全量表格喂给模型（适合全局聚合 / 开放式分析 / 代码生成）\n"
    "只输出 filter / partial / full 三个标签之一，不输出任何解释。"
)


def render_user(sample: Dict) -> str:
    meta = sample.get("table_meta", {})
    return (
        f"benchmark: {meta.get('benchmark','?')}\n"
        f"sub_task : {meta.get('sub_task_name','?')}\n"
        f"language : {meta.get('lang','?')}\n"
        f"table    : {meta.get('rows',0)} rows x {meta.get('cols',0)} cols ({meta.get('cells',0)} cells)\n"
        f"query    : {sample.get('question','')}"
    )


def to_train_example(sample: Dict) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user(sample)},
            {"role": "assistant", "content": sample["label"]},
        ],
        "meta": {
            "sample_id": sample["sample_id"],
            "benchmark": sample["benchmark"],
            "label": sample["label"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_samples: List[Dict] = []
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_samples.append(json.loads(line))

    # 标签分布统计
    bench_label = Counter()
    for s in all_samples:
        bench_label[(s["benchmark"], s["label"])] += 1
    print("=== label distribution ===")
    for (b, l), n in sorted(bench_label.items()):
        print(f"  {b:12s} {l:8s} {n}")
    print(f"  total: {len(all_samples)}")

    # 按 (benchmark, label) 分层 split
    by_strata: Dict = {}
    for s in all_samples:
        key = (s["benchmark"], s["label"])
        by_strata.setdefault(key, []).append(s)

    rng = random.Random(args.seed)
    train, val = [], []
    for k, group in by_strata.items():
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * args.val_ratio))) if len(group) >= 5 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_train)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_val)) or ".", exist_ok=True)

    with open(args.out_train, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(to_train_example(s), ensure_ascii=False) + "\n")
    with open(args.out_val, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(to_train_example(s), ensure_ascii=False) + "\n")

    print(f"wrote {len(train)} train / {len(val)} val")


if __name__ == "__main__":
    main()
