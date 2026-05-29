from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


class OpenAICompatibleClient:
    """Small adapter used by both STR conversion and TripletQL QA."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("STR_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("STR_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("STR_LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        return cls()

    def _get_client(self):
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key configured. Set STR_LLM_API_KEY or OPENAI_API_KEY, "
                "or run conversion with use_llm='never'."
            )
        if self._client is None:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate_response(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = (response.choices[0].message.content or "").strip()
        usage_obj = getattr(response, "usage", None)
        usage = {
            "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        }
        return content, usage
