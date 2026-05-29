from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from .heuristic import heuristic_restore_triplets
from .json_utils import extract_json_object
from .llm import OpenAICompatibleClient
from .prompts import TRIPLET_SYSTEM_PROMPT, build_row_column_descriptions, build_triplet_prompt
from .types import GridTable

UseLLMMode = Literal["auto", "always", "never"]


class TripletConverter:
    """Convert parsed table grids into STR triplet JSON."""

    def __init__(self, llm_client: Optional[OpenAICompatibleClient] = None, use_llm: UseLLMMode = "auto"):
        self.llm_client = llm_client
        self.use_llm = use_llm

    def from_grid(self, table: GridTable) -> Dict[str, Any]:
        if self._should_use_llm():
            restored = self._restore_with_llm(table)
            if restored:
                return self._wrap(table, restored, method="llm")
            if self.use_llm == "always":
                raise RuntimeError("LLM triplet restoration failed to return valid JSON.")
        fallback = heuristic_restore_triplets(table)
        fallback["restoration_method"] = "heuristic"
        return fallback

    def _should_use_llm(self) -> bool:
        if self.use_llm == "never":
            return False
        if self.use_llm == "always":
            return True
        return bool(self.llm_client and self.llm_client.available)

    def _restore_with_llm(self, table: GridTable) -> Dict[str, Any]:
        if self.llm_client is None:
            self.llm_client = OpenAICompatibleClient.from_env()
        desc = build_row_column_descriptions(table.grid)
        prompt = build_triplet_prompt(desc, table.span)
        content, usage = self.llm_client.generate_response(
            [
                {"role": "system", "content": TRIPLET_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = extract_json_object(content)
        if parsed:
            parsed["_usage"] = usage
        return parsed

    def _wrap(self, table: GridTable, restored: Dict[str, Any], *, method: str) -> Dict[str, Any]:
        group = restored.get("group") or []
        items = restored.get("items") or _extract_unique(group, "item")
        features = restored.get("features") or _extract_unique(group, "feature")
        out = {
            "table_id": table.table_id,
            "name": table.name,
            "source_type": table.source_type,
            "shape": table.shape_text,
            "span": table.span,
            "items": items,
            "features": features,
            "group": group,
            "remark": restored.get("remark", ""),
            "metadata": table.metadata,
            "restoration_method": method,
        }
        if restored.get("_usage"):
            out["usage"] = restored["_usage"]
        return out


def _extract_unique(group, key: str):
    seen = set()
    out = []
    for row in group:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
