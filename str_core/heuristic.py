from __future__ import annotations

from typing import Any, Dict, List

from .types import GridTable


def heuristic_restore_triplets(table: GridTable, *, header_rows: int = 1, label_cols: int = 1) -> Dict[str, Any]:
    """Deterministic fallback for simple rectangular tables.

    The paper path uses the LLM prompt in `prompts.py`; this fallback keeps the
    CLI and tests usable without network credentials.
    """

    rows, cols = table.shape
    if rows == 0 or cols == 0:
        return _wrap(table, [], [], [], "")

    header_rows = min(max(header_rows, 1), rows)
    label_cols = min(max(label_cols, 1), cols)
    data_rows = range(header_rows, rows)
    data_cols = range(label_cols, cols)

    features = [_join_path([table.grid[r][c] for r in range(header_rows)]) or f"feature_{c}" for c in data_cols]
    features = _dedupe(features)

    items: List[str] = []
    group: List[Dict[str, Any]] = []
    for r in data_rows:
        item = _join_path([table.grid[r][c] for c in range(label_cols)]) or f"item_{r}"
        items.append(item)
    items = _dedupe(items)

    for item_pos, r in enumerate(data_rows):
        for feature_pos, c in enumerate(data_cols):
            value = table.grid[r][c]
            if value == "":
                continue
            group.append({"item": items[item_pos], "feature": features[feature_pos], "value": value})

    return _wrap(table, items, features, group, "")


def _join_path(parts: List[str]) -> str:
    cleaned = []
    for part in parts:
        value = str(part).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return ".".join(cleaned)


def _dedupe(values: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    out = []
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        out.append(value if counts[value] == 1 else f"{value}.{counts[value]}")
    return out


def _wrap(table: GridTable, items: List[str], features: List[str], group: List[Dict], remark: str) -> Dict[str, Any]:
    return {
        "table_id": table.table_id,
        "name": table.name,
        "source_type": table.source_type,
        "shape": table.shape_text,
        "span": table.span,
        "items": items,
        "features": features,
        "group": group,
        "remark": remark,
        "metadata": table.metadata,
    }
