from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from triplet_agent.agent import TripletAgent

from .llm import OpenAICompatibleClient

DEFAULT_INSTRUCTION = (
    "Please answer the question based on the table represented below.\n\n"
    "Table:\n{context}\n\n"
    "Question: {question}"
)


def answer_triplet_questions(
    triplet_data: Dict[str, Any],
    questions: str | List[str],
    *,
    client: Optional[OpenAICompatibleClient] = None,
    instruction_template: str = DEFAULT_INSTRUCTION,
    system_message: str = "You are a careful table question-answering assistant.",
    pre_context: str = "",
    post_context: str = "",
    route_mode: str = "auto",
    qwen_url: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(questions, str):
        questions = [questions]
    client = client or OpenAICompatibleClient.from_env()
    agent = TripletAgent(client, qwen_url=qwen_url or os.getenv("STR_QWEN_URL"))
    return agent.process_triplet_query(
        triplet_data=triplet_data,
        question_list=questions,
        instruction_template=instruction_template,
        system_message=system_message,
        pre_context=pre_context,
        post_context=post_context,
        route_mode=route_mode,
    )
