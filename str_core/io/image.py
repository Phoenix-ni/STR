from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..types import GridTable, default_name
from .html import parse_table_markup


def read_image_table(
    image_path: str | Path,
    *,
    split_checkpoint: str | Path,
    merge_checkpoint: str | Path,
    ocr_words_path: str | Path | None = None,
    device: str = "cuda",
    img_size: int = 960,
    table_id: str = "",
) -> GridTable:
    """Run TABLET structure inference and parse generated HTML into STR input."""

    html, meta = image_to_html_with_tablet(
        image_path,
        split_checkpoint=split_checkpoint,
        merge_checkpoint=merge_checkpoint,
        ocr_words_path=ocr_words_path,
        device=device,
        img_size=img_size,
    )
    grid = parse_table_markup(html, table_id=table_id, name=default_name(image_path))
    grid.source_type = "image"
    grid.metadata["tablet"] = meta
    return grid


def image_to_html_with_tablet(
    image_path: str | Path,
    *,
    split_checkpoint: str | Path,
    merge_checkpoint: str | Path,
    ocr_words_path: str | Path | None = None,
    device: str = "cuda",
    img_size: int = 960,
) -> Tuple[str, Dict[str, Any]]:
    image_path = Path(image_path)
    split_checkpoint = Path(split_checkpoint)
    merge_checkpoint = Path(merge_checkpoint)
    if not split_checkpoint.is_file() or not merge_checkpoint.is_file():
        raise FileNotFoundError(
            "Image conversion needs local TABLET checkpoints. Weights are not "
            "included in the public repository."
        )

    tablet_dir = Path(__file__).resolve().parents[2] / "tablet"
    sys.path.insert(0, str(tablet_dir))
    try:
        from inference import TABLETInference
    finally:
        try:
            sys.path.remove(str(tablet_dir))
        except ValueError:
            pass

    words = _load_word_boxes(ocr_words_path) if ocr_words_path else None
    pipeline = TABLETInference(
        split_checkpoint=str(split_checkpoint),
        merge_checkpoint=str(merge_checkpoint),
        device=device,
        img_size=img_size,
    )
    result = pipeline.process_single_image(str(image_path), word_bboxes=words)
    html = result.get("html") or "<table></table>"
    meta = {
        "grid_shape": result.get("grid_shape"),
        "processing_time": result.get("processing_time"),
        "has_ocr_words": bool(words),
    }
    return html, meta


def _load_word_boxes(path: str | Path) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("words", "tokens", "word_bboxes"):
            if isinstance(data.get(key), list):
                return data[key]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported OCR word-box JSON format: {path}")
