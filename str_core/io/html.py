from __future__ import annotations

import re
from pathlib import Path
from typing import List

from ..types import GridTable, SpanInfo, default_name


def read_html_table(path: str | Path, *, table_id: str = "") -> GridTable:
    path = Path(path)
    return parse_table_markup(path.read_text(encoding="utf-8"), table_id=table_id, name=default_name(path))


def parse_table_markup(markup: str, *, table_id: str = "", name: str = "") -> GridTable:
    if "<table" not in (markup or "").lower():
        raise ValueError("HTML input must contain a <table> element.")
    return parse_html_table(markup, table_id=table_id, name=name)


def parse_html_table(html: str, *, table_id: str = "", name: str = "") -> GridTable:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_clean_html(html), "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> element found in HTML input.")

    grid: List[List[str]] = []
    occupied = set()
    span_info: SpanInfo = {}
    seen = set()

    trs = table.find_all("tr")
    for r, tr in enumerate(trs):
        _ensure_size(grid, r + 1, 0)
        c = 0
        for td in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in occupied:
                c += 1
            rowspan = _safe_span(td.get("rowspan", 1))
            colspan = _safe_span(td.get("colspan", 1))
            text = td.get_text(" ", strip=True)
            _ensure_size(grid, r + rowspan, c + colspan)
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    grid[rr][cc] = text
                    occupied.add((rr, cc))

            if rowspan > 1 or colspan > 1:
                key = text if text else f"(empty@R{r + 1}C{c + 1})"
                if key in seen:
                    key = f"{key}@R{r + 1}C{c + 1}"
                seen.add(key)
                span_info[key] = {}
                if rowspan > 1:
                    span_info[key]["rowspan"] = rowspan
                if colspan > 1:
                    span_info[key]["colspan"] = colspan
            c += colspan

    return GridTable.from_grid(grid, span=span_info, table_id=table_id, name=name, source_type="html")


def _clean_html(raw_html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "html.parser")
    for selector in ["img", "style", "script"]:
        for tag in soup.find_all(selector):
            tag.decompose()
    for tag_name in ["a", "abbr", "b", "i", "em", "strong", "small", "font", "span"]:
        for tag in soup.find_all(tag_name):
            tag.unwrap()
    return str(soup)


def _safe_span(value, default: int = 1) -> int:
    match = re.match(r"\d+", str(value or "").strip())
    return int(match.group()) if match else default


def _ensure_size(grid: List[List[str]], rows: int, cols: int) -> None:
    current_cols = max([len(row) for row in grid], default=0)
    target_cols = max(cols, current_cols)
    while len(grid) < rows:
        grid.append([""] * target_cols)
    if target_cols > current_cols:
        for row in grid:
            row.extend([""] * (target_cols - len(row)))
