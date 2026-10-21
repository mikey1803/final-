"""Minimal Groq chat client for the recovery executors.

A separate client (rather than ``src.agent.llm_client``, Layer 1, read-only)
because executors need what that wrapper does not return: provider-reported
token usage (for the budget and ``cost_tokens``) and the exact model id echoed
back by the API (recorded on every RecoveryAttempt). Same raw-HTTP pattern as
``experiments/llm_designers.py``. Model: ``openai/gpt-oss-20b`` (Task 21 D6,
the model the live agent's tool selector already uses).

The key is read from the environment (``GROQ_API_KEY``), loading
``backend/.env`` if it is not already set; it is never logged or returned.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import requests

from recovery.live.executors.base import LLMError, LLMReply

__all__ = ["GroqChatClient", "DEFAULT_MODEL"]

DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def _api_key() -> Optional[str]:
    if not os.environ.get("GROQ_API_KEY") and _ENV_PATH.exists():
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    return os.environ.get("GROQ_API_KEY") or None


class GroqChatClient:
    def __init__(self, model: str = DEFAULT_MODEL, reasoning_effort: str = "low",
                 base_url: str = GROQ_BASE_URL) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url

    def complete(self, prompt: str, max_tokens: int, timeout_s: float) -> LLMReply:
        key = _api_key()
        if not key:
            raise LLMError("GROQ_API_KEY not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "reasoning_effort": self.reasoning_effort,
        }
        start = time.monotonic()
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=timeout_s,
                                 headers={"Authorization": f"Bearer {key}"})
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from None
        latency_ms = (time.monotonic() - start) * 1000.0
        if not resp.ok:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            body = resp.json()
            text = body["choices"][0]["message"].get("content") or ""
            usage = body["usage"]
            return LLMReply(text=text.strip(), model_id=body["model"],
                            prompt_tokens=int(usage["prompt_tokens"]),
                            completion_tokens=int(usage["completion_tokens"]), latency_ms=latency_ms)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed response envelope: {exc}") from None
