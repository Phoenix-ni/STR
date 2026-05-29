# -*- encoding: utf-8 -*-
"""
Adapters: 把三个 benchmark 的原始样本翻译成统一的 OracleSample 格式，
便于 build_oracle_labels.py 与 eval_learnable.py 共用。

OracleSample 字段：
    sample_id        : str, benchmark 内唯一 id
    benchmark        : "tableeval" | "wtq" | "tablebench"
    sub_task_name    : str
    lang             : "zh" | "en"
    question_list    : List[str]
    golden_answers   : List[str]   长度与 question_list 对齐
    triplet_data     : Dict, 喂给 TripletAgent.process_triplet_query
    instruction      : str
    system_message   : str
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class OracleSample:
    sample_id: str
    benchmark: str
    sub_task_name: str
    lang: str
    question_list: List[str]
    golden_answers: List[str]
    triplet_data: Dict[str, Any]
    instruction: str
    system_message: str = "You are a helpful assistant."
    pre_context: str = ""
    post_context: str = ""

    def to_meta(self) -> Dict[str, Any]:
        """喂给 learnable router 用的 table metadata。"""
        items = self.triplet_data.get("items") or []
        feats = self.triplet_data.get("features") or []
        return {
            "rows": len(items),
            "cols": len(feats),
            "cells": len(items) * len(feats),
            "lang": self.lang,
            "benchmark": self.benchmark,
            "sub_task_name": self.sub_task_name,
        }


# ─────────────────────────────────────────────────────────────────────
# TableEval
# ─────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _index_triplets(triplet_jsonl_path: str, key: str = "table_id") -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in _load_jsonl(triplet_jsonl_path):
        out[str(row[key])] = row
    return out


def iter_tableeval(
    test_jsonl: str,
    triplet_jsonl: str,
    limit: Optional[int] = None,
) -> Iterator[OracleSample]:
    triplet_idx = _index_triplets(triplet_jsonl)
    rows = _load_jsonl(test_jsonl)
    if limit:
        rows = rows[:limit]
    for r in rows:
        tid = str(r.get("table_id"))
        triplet = triplet_idx.get(tid)
        if triplet is None:
            continue
        golden = []
        for ga in r.get("golden_answer_list") or []:
            for qa in ga.get("问题列表") or []:
                ans = qa.get("最终答案")
                if isinstance(ans, list):
                    ans = " | ".join(str(x) for x in ans)
                golden.append(str(ans) if ans is not None else "")
        yield OracleSample(
            sample_id=f"tableeval-{r['id']}",
            benchmark="tableeval",
            sub_task_name=str(r.get("sub_task_name") or ""),
            lang="zh",
            question_list=list(r.get("question_list") or []),
            golden_answers=golden,
            triplet_data=triplet,
            instruction=r.get("instruction") or "",
            system_message=r.get("system_message") or "You are a helpful assistant.",
        )


# ─────────────────────────────────────────────────────────────────────
# WikiTableQuestions
# ─────────────────────────────────────────────────────────────────────

def iter_wtq(test_jsonl: str, limit: Optional[int] = None) -> Iterator[OracleSample]:
    rows = _load_jsonl(test_jsonl)
    if limit:
        rows = rows[:limit]
    for r in rows:
        triplet = (r.get("context") or {}).get("triplet")
        if not triplet:
            continue
        golden = []
        for ga in r.get("golden_answer_list") or []:
            for qa in ga.get("问题列表") or []:
                ans = qa.get("最终答案")
                if isinstance(ans, list):
                    ans = " | ".join(str(x) for x in ans)
                golden.append(str(ans) if ans is not None else "")
        yield OracleSample(
            sample_id=f"wtq-{r.get('wtq_id') or r.get('id')}",
            benchmark="wtq",
            sub_task_name=str(r.get("sub_task_name") or "QA"),
            lang="en",
            question_list=list(r.get("question_list") or []),
            golden_answers=golden,
            triplet_data=triplet,
            instruction=r.get("instruction") or "",
            system_message=r.get("system_message") or "You are a helpful assistant.",
        )


# ─────────────────────────────────────────────────────────────────────
# TableBench
# ─────────────────────────────────────────────────────────────────────

# 默认 TableBench 没有自带 instruction 模板，这里用一个最简单的
TABLEBENCH_DEFAULT_INSTRUCTION = (
    "Read the table represented as semantic triplets and answer the question. "
    "Output ONLY the final answer with no explanation.\n\n"
    "Table:\n{context}\n\nQuestion: {question}"
)


def iter_tablebench(test_jsonl: str, limit: Optional[int] = None) -> Iterator[OracleSample]:
    rows = _load_jsonl(test_jsonl)
    if limit:
        rows = rows[:limit]
    for r in rows:
        # TableBench triplet 直接 inline 在样本里
        triplet = {
            "shape": r.get("shape"),
            "span": r.get("span") or {},
            "items": r.get("items") or [],
            "features": r.get("features") or [],
            "group": r.get("group") or [],
            "remark": r.get("remark"),
        }
        if not triplet["group"]:
            continue
        ans = r.get("answer")
        if isinstance(ans, list):
            ans = " | ".join(str(x) for x in ans)
        yield OracleSample(
            sample_id=f"tablebench-{r['id']}",
            benchmark="tablebench",
            sub_task_name=str(r.get("qsubtype") or r.get("qtype") or ""),
            lang="en",
            question_list=[r.get("question") or ""],
            golden_answers=[str(ans) if ans is not None else ""],
            triplet_data=triplet,
            instruction=TABLEBENCH_DEFAULT_INSTRUCTION,
        )


BENCHMARK_ITERATORS = {
    "tableeval": iter_tableeval,
    "wtq": iter_wtq,
    "tablebench": iter_tablebench,
}
