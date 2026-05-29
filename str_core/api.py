from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .llm import OpenAICompatibleClient
from .pipeline import convert_path
from .qa import answer_triplet_questions

app = FastAPI(title="STR TripletQL API", version="0.1.0")


class QARequest(BaseModel):
    triplet_data: Dict[str, Any] = Field(..., description="STR triplet JSON object.")
    question: Optional[str] = None
    questions: Optional[List[str]] = None
    route_mode: str = "auto"
    pre_context: str = ""
    post_context: str = ""
    system_message: str = "You are a careful table question-answering assistant."
    instruction_template: str = (
        "Please answer the question based on the table represented below.\n\n"
        "Table:\n{context}\n\nQuestion: {question}"
    )
    qwen_url: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/convert")
async def convert_endpoint(
    file: UploadFile = File(...),
    input_type: Optional[str] = Form(None),
    use_llm: str = Form("auto"),
    table_id: str = Form(""),
    sheet: Optional[str] = Form(None),
    split_checkpoint: Optional[str] = Form(None),
    merge_checkpoint: Optional[str] = Form(None),
    ocr_words_json: Optional[str] = Form(None),
    device: str = Form("cuda"),
) -> Dict[str, Any]:
    suffix = Path(file.filename or "table.bin").suffix
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        input_path.write_bytes(await file.read())
        words_path = None
        if ocr_words_json:
            words_path = Path(tmpdir) / "ocr_words.json"
            words_path.write_text(ocr_words_json, encoding="utf-8")
            json.loads(ocr_words_json)
        try:
            return convert_path(
                input_path,
                input_type=input_type,
                use_llm=_normalize_use_llm(use_llm),
                llm_client=OpenAICompatibleClient.from_env(),
                table_id=table_id,
                sheet=sheet,
                split_checkpoint=split_checkpoint,
                merge_checkpoint=merge_checkpoint,
                ocr_words_path=words_path,
                device=device,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/qa")
def qa_endpoint(request: QARequest) -> Dict[str, Any]:
    questions = request.questions or ([request.question] if request.question else [])
    if not questions:
        raise HTTPException(status_code=400, detail="Provide `question` or `questions`.")
    return answer_triplet_questions(
        request.triplet_data,
        questions,
        client=OpenAICompatibleClient.from_env(),
        route_mode=request.route_mode,
        pre_context=request.pre_context,
        post_context=request.post_context,
        system_message=request.system_message,
        instruction_template=request.instruction_template,
        qwen_url=request.qwen_url,
    )


def _normalize_use_llm(value: str):
    normalized = (value or "auto").lower()
    if normalized in {"true", "yes", "1"}:
        return "always"
    if normalized in {"false", "no", "0"}:
        return "never"
    if normalized not in {"auto", "always", "never"}:
        raise HTTPException(status_code=400, detail="use_llm must be auto, always, or never.")
    return normalized
