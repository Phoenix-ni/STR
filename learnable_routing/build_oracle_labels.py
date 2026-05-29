# -*- encoding: utf-8 -*-
"""
build_oracle_labels.py
======================

对一个 benchmark，对每条样本依次跑 route_mode ∈ {full, partial, filter}，
记录每档下的 (correct, input_tokens)，按规则选 oracle 标签：

    oracle =
        argmin_{mode ∈ correct_modes} input_tokens(mode)        if 至少一档对了
        "full"                                                  otherwise

输出 jsonl，每行：
    {
        "sample_id": "...",
        "benchmark": "...",
        "sub_task_name": "...",
        "lang": "zh"|"en",
        "question": "...",         (第一个问题；多轮场景里 oracle 标签基于整轮通过率)
        "table_meta": {rows, cols, cells, lang, benchmark, sub_task_name},
        "scores": {
            "full":     {"correct": [bool, ...], "input_tokens": [int, ...], "all_correct": bool, "sum_input": int},
            "partial":  {...},
            "filter":   {...},
        },
        "label": "full"|"partial"|"filter"
    }

用法（举例，每个 benchmark 200 条做 smoke test）：

    python build_oracle_labels.py \
        --benchmark tableeval \
        --test_jsonl /path/to/TableEval-test.jsonl \
        --triplet_jsonl /path/to/TableEval-triplet.jsonl \
        --model_name longcat-flash-lite \
        --config_file /path/to/TableEval/config/api.yaml \
        --qwen_url http://localhost:8001/v1 \
        --output_jsonl out/routing_labels_tableeval.jsonl \
        --max_workers 4 \
        --limit 200

跑完三个 benchmark 之后用 merge_labels.py 合到一个文件做训练。
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

# 让 STR/triplet_agent 可被导入
HERE = os.path.dirname(os.path.abspath(__file__))
STR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if STR_ROOT not in sys.path:
    sys.path.insert(0, STR_ROOT)

from triplet_agent.agent import TripletAgent  # noqa: E402

from benchmark_adapters import BENCHMARK_ITERATORS, OracleSample  # noqa: E402
from answer_scorer import score  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# LLM client：复用 TableEval/openai_client.py 的 OpenAIClient
# ─────────────────────────────────────────────────────────────────────

def build_client(model_name: str, config_file: str | None = None):
    """Build an LLM client.

    Public STR uses environment variables through `str_core.llm`. For backward
    compatibility with the original experiment tree, a TableEval-style
    `openai_client.py` is used when an explicit config file is supplied.
    """
    if config_file:
        tableeval_root = os.path.abspath(os.path.join(STR_ROOT, "..", "TableEval"))
        if tableeval_root not in sys.path:
            sys.path.insert(0, tableeval_root)
        try:
            from openai_client import OpenAIClient  # noqa: WPS433

            return OpenAIClient(model=model_name, config_file=config_file)
        except Exception as exc:
            logging.warning("Falling back to STR_LLM_* environment client: %s", exc)
    from str_core.llm import OpenAICompatibleClient  # noqa: WPS433

    return OpenAICompatibleClient(model=model_name)


ROUTE_MODES = ("full", "partial", "filter")


def run_one_mode(agent: TripletAgent, sample: OracleSample, mode: str) -> Dict[str, Any]:
    """跑一条样本在指定 route_mode 下的所有轮次。"""
    out = agent.process_triplet_query(
        triplet_data=sample.triplet_data,
        question_list=sample.question_list,
        instruction_template=sample.instruction,
        system_message=sample.system_message,
        pre_context=sample.pre_context,
        post_context=sample.post_context,
        route_mode=mode,
    )
    preds = out["prediction_list"]
    tokens = out.get("input_tokens_per_turn") or [0] * len(preds)
    correct = []
    for pred, gold in zip(preds, sample.golden_answers):
        ok, _ = score(pred or "", gold or "", sample.benchmark)
        correct.append(bool(ok))
    return {
        "correct": correct,
        "input_tokens": [int(t) for t in tokens],
        "all_correct": bool(correct) and all(correct),
        "sum_input": int(sum(tokens)),
        "preds": preds,
    }


def oracle_label(scores: Dict[str, Dict[str, Any]]) -> str:
    """all_correct 优先；同等正确情况下 sum_input 最小者；全错回退 full。"""
    correct_modes = [m for m in ROUTE_MODES if scores[m]["all_correct"]]
    if correct_modes:
        return min(correct_modes, key=lambda m: scores[m]["sum_input"])
    return "full"


def process_sample(agent: TripletAgent, sample: OracleSample) -> Dict[str, Any]:
    scores = {}
    for mode in ROUTE_MODES:
        try:
            scores[mode] = run_one_mode(agent, sample, mode)
        except Exception as e:
            logging.error(f"[{sample.sample_id}] mode={mode} failed: {e}")
            scores[mode] = {
                "correct": [False] * len(sample.question_list),
                "input_tokens": [0] * len(sample.question_list),
                "all_correct": False,
                "sum_input": 0,
                "preds": [],
                "error": str(e),
            }
    label = oracle_label(scores)
    return {
        "sample_id": sample.sample_id,
        "benchmark": sample.benchmark,
        "sub_task_name": sample.sub_task_name,
        "lang": sample.lang,
        "question": sample.question_list[0] if sample.question_list else "",
        "table_meta": sample.to_meta(),
        "scores": {m: {k: v for k, v in s.items() if k != "preds"} for m, s in scores.items()},
        "label": label,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=list(BENCHMARK_ITERATORS.keys()))
    p.add_argument("--test_jsonl", required=True, help="benchmark test split jsonl")
    p.add_argument("--triplet_jsonl", default=None, help="TableEval triplet jsonl (TableEval only)")
    p.add_argument("--model_name", default=os.getenv("STR_LLM_MODEL", "gpt-4o-mini"))
    p.add_argument("--config_file", default=None, help="Optional legacy TableEval OpenAIClient config")
    p.add_argument("--qwen_url", default=os.getenv("STR_QWEN_URL"))
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    iter_fn = BENCHMARK_ITERATORS[args.benchmark]
    if args.benchmark == "tableeval":
        if not args.triplet_jsonl:
            p.error("--triplet_jsonl is required for TableEval")
        samples = list(iter_fn(args.test_jsonl, args.triplet_jsonl, limit=args.limit))
    else:
        samples = list(iter_fn(args.test_jsonl, limit=args.limit))

    logging.info(f"loaded {len(samples)} samples from {args.benchmark}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    done_ids = set()
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        logging.info(f"resume: {len(done_ids)} already done")
        samples = [s for s in samples if s.sample_id not in done_ids]

    client = build_client(args.model_name, args.config_file)

    def make_agent():
        return TripletAgent(client, qwen_url=args.qwen_url)

    fout = open(args.output_jsonl, "a", encoding="utf-8")
    start = time.time()
    n_done = 0
    label_counter: Counter = Counter()

    pbar = tqdm(total=len(samples), desc=f"oracle({args.benchmark})", unit="smp", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        agents = [make_agent() for _ in range(args.max_workers)]
        futures = {}
        for i, s in enumerate(samples):
            a = agents[i % args.max_workers]
            futures[pool.submit(process_sample, a, s)] = s.sample_id

        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                rec = fut.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                label_counter[rec["label"]] += 1
                pbar.update(1)
                pbar.set_postfix(
                    label=rec["label"],
                    F=label_counter.get("full", 0),
                    P=label_counter.get("partial", 0),
                    f=label_counter.get("filter", 0),
                )
            except Exception as e:
                pbar.update(1)
                logging.error(f"sample {sid} failed: {e}")

    pbar.close()
    fout.close()
    logging.info(f"done. wrote {n_done} records to {args.output_jsonl}; label dist: {dict(label_counter)}")


if __name__ == "__main__":
    main()
