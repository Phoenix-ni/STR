from __future__ import annotations

import json
from typing import Dict, List

from .types import SpanInfo


TRIPLET_SYSTEM_PROMPT = (
    "You are a table semantic triplet restoration expert. Restore explicit "
    "semantic item paths and feature paths from table rows, columns, merged "
    "cells, and hierarchical headers. Return only valid JSON."
)


def build_row_column_descriptions(grid: List[List[str]]) -> Dict[str, List[Dict]]:
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    return {
        "rows": [{"row_index": r, "cells": [grid[r][c] for c in range(cols)]} for r in range(rows)],
        "columns": [{"col_index": c, "cells": [grid[r][c] for r in range(rows)]} for c in range(cols)],
    }


def build_triplet_prompt(row_col_desc: Dict, span_info: SpanInfo | None = None) -> str:
    rows = row_col_desc["rows"]
    columns = row_col_desc["columns"]
    lines = [
        "# Task: Semantic Triplet Restoration (row=item, column=feature)",
        "",
        "You are given a 2D table grid whose merged cells have already been broadcast into every covered cell.",
        "Assign each data row a semantic item path and each data column a semantic feature path.",
        "Do not use placeholder labels such as row_0, row_1, col_0, or col_1.",
        "",
        "Item rules:",
        "- Use table semantics to build a unique row-wise item path.",
        "- For hierarchical row headers, concatenate levels with a period '.', e.g. Investment rating.Stock rating.Buy.",
        "- A merged full-row title may become a top-level path component.",
        "- A rowspan group label should be used as a parent path component for rows under it.",
        "",
        "Feature rules:",
        "- Use column semantics to build a unique feature path.",
        "- For hierarchical column headers, concatenate levels with '.', e.g. EPS.2024E.",
        "- A colspan group header should be used as a parent path component for child columns.",
        "",
        "Group rules:",
        "- Output only data-cell triplets in group.",
        "- If a cell was selected as the row item or column feature, do not repeat it as a value.",
        "- Remarks, notes, and explanatory full-row text should be moved to remark, not group.",
    ]

    if span_info:
        lines.extend(["", "## Merged cells"])
        for content, span in span_info.items():
            parts = []
            if "rowspan" in span:
                parts.append(f"rowspan={span['rowspan']}")
            if "colspan" in span:
                parts.append(f"colspan={span['colspan']}")
            lines.append(f"- {content}: {', '.join(parts)}")

    lines.extend(["", "## Rows"])
    for row in rows:
        lines.append(f"- row {row['row_index']}: {json.dumps(row['cells'], ensure_ascii=False)}")

    lines.extend(["", "## Columns"])
    for col in columns:
        lines.append(f"- column {col['col_index']}: {json.dumps(col['cells'], ensure_ascii=False)}")

    lines.extend(
        [
            "",
            "## Output",
            "Return one valid JSON object and no markdown fences.",
            "Schema:",
            '{"items": ["semantic item path", "..."], '
            '"features": ["semantic feature path", "..."], '
            '"group": [{"item": "...", "feature": "...", "value": "..."}], '
            '"remark": ""}',
            "Every item and every feature should be unique. Use original table semantics rather than physical coordinates.",
        ]
    )
    return "\n".join(lines)
