from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


SpanInfo = Dict[str, Dict[str, int]]
TripletGroup = List[Dict[str, Any]]


@dataclass
class GridTable:
    """A parsed table grid with merged-cell metadata already broadcast."""

    grid: List[List[str]]
    shape: Tuple[int, int]
    span: SpanInfo = field(default_factory=dict)
    table_id: str = ""
    name: str = ""
    source_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_grid(
        cls,
        grid: List[List[str]],
        *,
        span: SpanInfo | None = None,
        table_id: str = "",
        name: str = "",
        source_type: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> "GridTable":
        rows = len(grid)
        cols = max((len(row) for row in grid), default=0)
        normalized = [list(row) + [""] * (cols - len(row)) for row in grid]
        return cls(
            grid=normalized,
            shape=(rows, cols),
            span=span or {},
            table_id=table_id,
            name=name,
            source_type=source_type,
            metadata=metadata or {},
        )

    @property
    def shape_text(self) -> str:
        return f"{self.shape[0]}*{self.shape[1]}"


def default_name(path: str | Path) -> str:
    return Path(path).stem
