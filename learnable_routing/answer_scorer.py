# -*- encoding: utf-8 -*-
"""
oracle correctness scorer：把模型输出 vs gold answer 翻译成 0/1 标签。

不追求复刻 TableEval / WTQ / TableBench 各自的官方评测脚本，
只要"oracle 标签足够区分 full/partial/filter 哪种最好"即可。

策略：
  - WTQ：strict exact match (官方约束)
  - TableBench：strict exact match（answer 通常是单值或 list）
  - TableEval：loose match —— 双向 substring 命中，或数字归一化后相等

返回 (is_correct: bool, normalized_pair: tuple) 用于 debug。
"""
from __future__ import annotations
import re
import unicodedata
from typing import Tuple


_NUM_RE = re.compile(r"-?\d+(?:[\.,]\d+)?")


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("，", ",").replace("。", ".")
    return s


def _norm_number(s: str) -> str:
    s = _norm(s).replace(",", "").replace(" ", "")
    return s


def _strict_match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if not g:
        return False
    if p == g:
        return True
    # 数字归一化
    pn, gn = _norm_number(pred), _norm_number(gold)
    if pn and gn and pn == gn:
        return True
    return False


def _loose_match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if not g:
        return False
    if _strict_match(pred, gold):
        return True
    # 双向 substring
    if g in p or p in g:
        return True
    # 数字集合匹配（多答案情形）
    pn = set(_NUM_RE.findall(p.replace(",", "")))
    gn = set(_NUM_RE.findall(g.replace(",", "")))
    if gn and gn.issubset(pn):
        return True
    return False


def score(pred: str, gold: str, benchmark: str) -> Tuple[bool, Tuple[str, str]]:
    if benchmark in {"wtq", "tablebench"}:
        ok = _strict_match(pred, gold)
    else:
        ok = _loose_match(pred, gold)
    return ok, (_norm(pred), _norm(gold))


if __name__ == "__main__":
    cases = [
        ("Italy", "italy", "wtq", True),
        ("Italy ", "italy", "wtq", True),
        ("1,000", "1000", "wtq", True),
        ("The answer is Italy.", "italy", "tableeval", True),
        ("31.49", "31.49", "tablebench", True),
        ("apples and oranges", "bananas", "tableeval", False),
    ]
    for pred, gold, bench, expect in cases:
        ok, _ = score(pred, gold, bench)
        flag = "OK" if ok == expect else "FAIL"
        print(f"[{flag}] bench={bench:10s} pred={pred!r:30s} gold={gold!r:15s} -> {ok} (expect {expect})")
