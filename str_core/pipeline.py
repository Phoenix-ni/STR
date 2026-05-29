from __future__ import annotations

from pathlib import Path
from typing import Optional

from .converter import TripletConverter, UseLLMMode
from .io.excel import read_excel_table
from .io.html import read_html_table
from .io.image import read_image_table
from .llm import OpenAICompatibleClient
from .types import GridTable

EXCEL_EXTS = {".xlsx", ".xlsm", ".xltx", ".xltm"}
HTML_EXTS = {".html", ".htm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def convert_path(
    path: str | Path,
    *,
    input_type: Optional[str] = None,
    use_llm: UseLLMMode = "auto",
    llm_client: Optional[OpenAICompatibleClient] = None,
    table_id: str = "",
    sheet: Optional[str] = None,
    split_checkpoint: Optional[str | Path] = None,
    merge_checkpoint: Optional[str | Path] = None,
    ocr_words_path: Optional[str | Path] = None,
    device: str = "cuda",
) -> dict:
    table = read_table(
        path,
        input_type=input_type,
        table_id=table_id,
        sheet=sheet,
        split_checkpoint=split_checkpoint,
        merge_checkpoint=merge_checkpoint,
        ocr_words_path=ocr_words_path,
        device=device,
    )
    return TripletConverter(llm_client=llm_client, use_llm=use_llm).from_grid(table)


def read_table(
    path: str | Path,
    *,
    input_type: Optional[str] = None,
    table_id: str = "",
    sheet: Optional[str] = None,
    split_checkpoint: Optional[str | Path] = None,
    merge_checkpoint: Optional[str | Path] = None,
    ocr_words_path: Optional[str | Path] = None,
    device: str = "cuda",
) -> GridTable:
    path = Path(path)
    kind = (input_type or _detect_type(path)).lower()
    if kind == "excel":
        return read_excel_table(path, sheet=sheet, table_id=table_id)
    if kind == "html":
        return read_html_table(path, table_id=table_id)
    if kind == "image":
        if split_checkpoint is None or merge_checkpoint is None:
            raise ValueError(
                "Image input requires local TABLET weights. Provide both "
                "--split-checkpoint and --merge-checkpoint; weights are not included "
                "in the public repository."
            )
        return read_image_table(
            path,
            split_checkpoint=split_checkpoint,
            merge_checkpoint=merge_checkpoint,
            ocr_words_path=ocr_words_path,
            device=device,
            table_id=table_id,
        )
    raise ValueError(f"Unsupported input type: {kind}")


def _detect_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in EXCEL_EXTS:
        return "excel"
    if suffix in HTML_EXTS:
        return "html"
    if suffix in IMAGE_EXTS:
        return "image"
    raise ValueError(f"Cannot infer input type from extension: {path}")
