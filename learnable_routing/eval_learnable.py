# -*- encoding: utf-8 -*-
"""
eval_learnable.py
=================

在某一个 benchmark 上对比 Fixed Router (route_mode=auto) 与 Learnable Router
(route_mode=learnable)。输出：
    - 整体精度 (TableEval=loose F1, WTQ/TableBench=strict accuracy)
    - 平均 input token
    - 每条样本的 jsonl（含 route 决策）

典型用法：
    python eval_learnable.py \
        --benchmark wtq \
        --test_jsonl /path/to/wtq-test.jsonl \
        --model_name longcat-flash-lite \
        --config_file /path/to/TableEval/config/api.yaml \
        --qwen_url http://localhost:8001/v1 \
        --router fixed \
        --output_jsonl out/wtq_fixed.jsonl

    python eval_learnable.py \
        --benchmark wtq \
        --test_jsonl /path/to/wtq-test.jsonl \
        --model_name longcat-flash-lite \
        --config_file /path/to/TableEval/config/api.yaml \
        --qwen_url http://localhost:8001/v1 \
        --router learnable \
        --lora_ckpt checkpoints/router_v1 \
        --base_model Qwen/Qwen3-0.6B \
        --output_jsonl out/wtq_learnable.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
STR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if STR_ROOT not in sys.path:
    sys.path.insert(0, STR_ROOT)

from triplet_agent.agent import TripletAgent  # noqa: E402

from benchmark_adapters import BENCHMARK_ITERATORS, OracleSample  # noqa: E402
from answer_scorer import score  # noqa: E402


def build_client(model_name: str, config_file: str | None = None):
    if config_file:
        tableeval_root = os.path.abspath(os.path.join(STR_ROOT, "..", "TableEval"))
        if tableeval_root not in sys.path:
            sys.path.insert(0, tableeval_root)
        try:
            from openai_client import OpenAIClient

            return OpenAIClient(model=model_name, config_file=config_file)
        except Exception as exc:
            logging.warning("Falling back to STR_LLM_* environment client: %s", exc)
    from str_core.llm import OpenAICompatibleClient

    return OpenAICompatibleClient(model=model_name)


def build_router(args):
    if args.router == "fixed":
        return None, "auto"
    if args.router == "stub":
        from learnable_router import StubLearnableRouter
        return StubLearnableRouter(), "learnable"
    # learnable
    from learnable_router import LearnableRouter
    lr = LearnableRouter(model_path=args.lora_ckpt, base_model=args.base_model)
    return lr, "learnable"


def run_sample(agent: TripletAgent, sample: OracleSample, route_mode: str, learnable_router) -> Dict[str, Any]:
    out = agent.process_triplet_query(
        triplet_data=sample.triplet_data,
        question_list=sample.question_list,
        instruction_template=sample.instruction,
        system_message=sample.system_message,
        pre_context=sample.pre_context,
        post_context=sample.post_context,
        route_mode=route_mode,
        learnable_router=learnable_router,
    )
    preds = out["prediction_list"]
    tokens = out.get("input_tokens_per_turn") or [0] * len(preds)
    route_seq = out.get("route_mode_per_turn") or []
    correct = []
    for pred, gold in zip(preds, sample.golden_answers):
        ok, _ = score(pred or "", gold or "", sample.benchmark)
        correct.append(bool(ok))
    return {
        "sample_id": sample.sample_id,
        "benchmark": sample.benchmark,
        "sub_task_name": sample.sub_task_name,
        "lang": sample.lang,
        "predictions": preds,
        "golden_answers": sample.golden_answers,
        "correct": correct,
        "input_tokens": [int(t) for t in tokens],
        "route_per_turn": route_seq,
        "sum_input_tokens": int(sum(tokens)),
        "num_turns": len(preds),
        "all_correct": bool(correct) and all(correct),
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    total_turns = sum(r["num_turns"] for r in records)
    total_correct = sum(sum(r["correct"]) for r in records)
    total_tokens = sum(r["sum_input_tokens"] for r in records)
    return {
        "n_samples": len(records),
        "n_turns": total_turns,
        "turn_level_accuracy": total_correct / max(total_turns, 1),
        "sample_level_accuracy": sum(1 for r in records if r["all_correct"]) / len(records),
        "avg_input_tokens_per_turn": total_tokens / max(total_turns, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BENCHMARK_ITERATORS.keys()))
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--triplet_jsonl", default=None)
    ap.add_argument("--model_name", default=os.getenv("STR_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--config_file", default=None)
    ap.add_argument("--qwen_url", default=os.getenv("STR_QWEN_URL"))
    ap.add_argument("--router", choices=["fixed", "learnable", "stub"], default="fixed")
    ap.add_argument("--lora_ckpt", default=None, help="learnable 模式必填")
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.router == "learnable" and not args.lora_ckpt:
        ap.error("--lora_ckpt is required when --router=learnable")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    iter_fn = BENCHMARK_ITERATORS[args.benchmark]
    if args.benchmark == "tableeval":
        if not args.triplet_jsonl:
            ap.error("--triplet_jsonl is required for TableEval")
        samples = list(iter_fn(args.test_jsonl, args.triplet_jsonl, limit=args.limit))
    else:
        samples = list(iter_fn(args.test_jsonl, limit=args.limit))
    logging.info(f"loaded {len(samples)} samples from {args.benchmark}")

    learnable_router, route_mode = build_router(args)
    client = build_client(args.model_name, args.config_file)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    fout = open(args.output_jsonl, "w", encoding="utf-8")

    def _agent_pool(n):
        return [TripletAgent(client, qwen_url=args.qwen_url) for _ in range(n)]
    agents = _agent_pool(args.max_workers)

    records: List[Dict[str, Any]] = []
    start = time.time()
    n_turns_total = 0
    n_correct_total = 0
    tok_sum_total = 0

    pbar = tqdm(
        total=len(samples),
        desc=f"eval({args.benchmark}/{args.router})",
        unit="smp",
        dynamic_ncols=True,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for i, s in enumerate(samples):
            a = agents[i % args.max_workers]
            futures[pool.submit(run_sample, a, s, route_mode, learnable_router)] = s.sample_id

        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                rec = fut.result()
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_turns_total += rec["num_turns"]
                n_correct_total += sum(rec["correct"])
                tok_sum_total += rec["sum_input_tokens"]
                acc = n_correct_total / max(n_turns_total, 1)
                avg_tok = tok_sum_total / max(n_turns_total, 1)
                pbar.update(1)
                pbar.set_postfix(acc=f"{acc:.3f}", avg_tok=f"{avg_tok:.0f}")
            except Exception as e:
                pbar.update(1)
                logging.error(f"sample {sid} failed: {e}")
    pbar.close()
    fout.close()

    summary = summarize(records)
    summary["router"] = args.router
    summary["benchmark"] = args.benchmark
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = os.path.splitext(args.output_jsonl)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
