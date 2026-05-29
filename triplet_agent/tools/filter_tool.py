# -*- encoding: utf-8 -*-
from typing import List, Optional, Tuple, Dict, Set
import json

def match_question_entities(question: str, candidates: List[str], limit: int = 8) -> List[str]:
    matched: List[str] = []
    q_lower = question.lower()
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        text_lower = text.lower()
        if text_lower in q_lower or q_lower in text_lower:
            matched.append(text)
            if len(matched) >= limit:
                break
    return list(dict.fromkeys(matched))


def merge_unique(base: List[str], extra: List[str]) -> List[str]:
    out = list(base)
    seen = set(out)
    for x in extra:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def expand_neighbors(selected: List[str], full_list: List[str], window: int = 1) -> List[str]:
    """将选中的 item 扩展其在 full_list 中的前后邻居，应对条件查询。"""
    if not selected or not full_list:
        return selected
    idx_set = set()
    full_idx = {v: i for i, v in enumerate(full_list)}
    for s in selected:
        i = full_idx.get(s)
        if i is not None:
            for offset in range(-window, window + 1):
                ni = i + offset
                if 0 <= ni < len(full_list):
                    idx_set.add(ni)
    return [full_list[i] for i in sorted(idx_set)]


def _fuzzy_match(candidate: str, valid_set: set, valid_list: list) -> Optional[str]:
    if candidate in valid_set:
        return candidate
    cl = candidate.lower().strip()
    for v in valid_list:
        vl = v.lower().strip()
        if cl in vl or vl in cl:
            return v
    return None


def extract_json_obj(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


class TripletFilter:
    """Agent 工具层：用于精准过滤三元组并提取关键行列。"""

    def __init__(self, client):
        self.client = client

    def llm_filter_with_state(
        self,
        question: str,
        items: List[str],
        features: List[str],
        state_summary: str = "",
    ) -> Tuple[str, List[str], List[str], dict]:
        state_block = f"【对话状态】\n{state_summary}\n\n" if state_summary else ""
        prompt = (
            "你是表格三元组筛选助手。给定问题、items、features，以及可能的对话状态，请只做高召回筛选。\n\n"
            "规则：\n"
            "- 如果问题需要遍历、统计、汇总、排序、比较大量数据，请返回 full。\n"
            "- 如果问题只是定位少数特定实体或特征，请返回 filter。\n"
            "- 对省略或指代问题，可参考对话状态理解当前实体，但仍应宁可多选、绝不遗漏。\n"
            "- 返回的 item/feature 必须是下方数组中的精确字符串。\n\n"
            f"{state_block}"
            f"【items】\n{json.dumps(items, ensure_ascii=False)}\n\n"
            f"【features】\n{json.dumps(features, ensure_ascii=False)}\n\n"
            f"【问题】{question}\n\n"
            "只输出合法 JSON，不要输出其它文本：\n"
            '策略1: {"decision": "full"}\n'
            '策略2: {"decision": "filter", "selected_items": ["..."], "selected_features": ["..."]}'
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        response, usage = self.client.generate_response(messages)
        parsed = extract_json_obj(response or "")

        decision = "full"
        selected_items: List[str] = []
        selected_features: List[str] = []

        if isinstance(parsed, dict) and parsed.get("decision") == "filter":
            items_set, features_set = set(items), set(features)
            raw_items = parsed.get("selected_items") or []
            raw_features = parsed.get("selected_features") or []
            if isinstance(raw_items, list):
                for ri in raw_items:
                    m = _fuzzy_match(str(ri), items_set, items)
                    if m:
                        selected_items.append(m)
            if isinstance(raw_features, list):
                for rf in raw_features:
                    m = _fuzzy_match(str(rf), features_set, features)
                    if m:
                        selected_features.append(m)
            selected_items = list(dict.fromkeys(selected_items))
            selected_features = list(dict.fromkeys(selected_features))
            # V3.1 改进 1C：只要 items 非空即可判定 filter，features 将全保留
            if selected_items:
                decision = "filter"

        return decision, selected_items, selected_features, usage
