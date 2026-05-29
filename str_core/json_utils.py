from __future__ import annotations

import json
import re
from typing import Any, Dict

_BAD_UNICODE_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])")


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating markdown fences."""

    raw = (text or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    elif raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError as exc:
            if "Invalid \\uXXXX" in str(exc) or "uXXXX" in str(exc):
                cleaned = _BAD_UNICODE_ESCAPE.sub("", candidate)
                try:
                    parsed = json.loads(cleaned)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    pass
    return {}
