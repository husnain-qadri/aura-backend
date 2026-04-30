import json
import re
from typing import Any

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def chat(system: str, user: str, *, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
    )
    return resp.choices[0].message.content or ''


def chat_json(system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 4096) -> Any:
    """Call the LLM and parse the response as JSON."""
    system_with_json = system.rstrip() + '\n\nIMPORTANT: Your response must be valid JSON only. No markdown fences, no extra text.'
    client = _get_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {'role': 'system', 'content': system_with_json},
            {'role': 'user', 'content': user},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
    )
    raw = resp.choices[0].message.content or '{}'
    raw = raw.strip()
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if fence_match:
        raw = fence_match.group(1).strip()
    return json.loads(raw)
