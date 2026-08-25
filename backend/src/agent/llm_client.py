"""Groq-backed LLM client for zero-shot recovery-strategy selection.

Thin wrapper around ChatGroq (via langchain-groq) so any part of the
project needing a real zero-shot LLM call can reuse one function —
currently ``OneShotLLMBaseline`` in the evaluation harness.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load GROQ_API_KEY from .env (falling back to .env.example) once."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv(project_root / ".env.example")
    _ENV_LOADED = True


def get_groq_api_key() -> Optional[str]:
    """Return GROQ_API_KEY if configured, else None."""
    _ensure_env_loaded()
    return os.environ.get("GROQ_API_KEY") or None


@lru_cache(maxsize=1)
def _get_chat_model():
    from langchain_groq import ChatGroq

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in a .env file (see .env.example) "
            "to enable real LLM calls."
        )
    return ChatGroq(model="openai/gpt-oss-20b", api_key=api_key, temperature=0.0, max_tokens=20)


@lru_cache(maxsize=1)
def _get_agent_chat_model():
    """A second, separately-cached ChatGroq instance with a larger token
    budget than ``_get_chat_model()`` — that one is deliberately capped at
    max_tokens=20 for the one-shot strategy-name use case, too small for a
    tool-call JSON response."""
    from langchain_groq import ChatGroq

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in a .env file (see .env.example) "
            "to enable real LLM calls."
        )
    return ChatGroq(model="openai/gpt-oss-20b", api_key=api_key, temperature=0.0, max_tokens=200)


def parse_tool_request(user_request: str, tool_descriptions: list[dict]) -> dict:
    """Ask the LLM to pick a tool and fill in its arguments for a natural-
    language request. Returns {"tool": str, "args": dict} parsed from the
    model's JSON response.

    Raises ValueError if the model's response isn't parseable JSON — callers
    should catch this and fall back rather than crash the request.
    """
    model = _get_agent_chat_model()
    tools_desc = "\n".join(
        f"- {t['name']}: {t['description']} (args: {t['args']})" for t in tool_descriptions
    )
    prompt = (
        "You are an AI agent that fulfils a user's request by calling exactly one tool.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        f"User request: {user_request}\n\n"
        'Respond with ONLY a JSON object of the form {"tool": "<tool_name>", "args": {...}}, '
        "choosing the single best tool and filling in its arguments from the request. "
        "No other text, no markdown formatting."
    )
    response = model.invoke(prompt)
    text = response.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    import json

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text!r}") from exc
    if "tool" not in parsed or "args" not in parsed:
        raise ValueError(f"LLM JSON missing 'tool'/'args' keys: {parsed!r}")
    return parsed


def classify_recovery_strategy(
    raw_error: str,
    taxonomy_description: str,
    strategy_options: list[str],
) -> str:
    """Ask the LLM to pick exactly one recovery-strategy name.

    Returns the model's raw text response (trimmed) — callers parse it
    against their own list of valid strategy names, since the model isn't
    guaranteed to answer in exactly the requested format.
    """
    model = _get_chat_model()
    prompt = (
        "You are an AI agent's recovery-strategy selector.\n\n"
        "Failure taxonomy:\n" + taxonomy_description + "\n\n"
        f"Observed error:\n{raw_error}\n\n"
        "Available recovery strategies:\n"
        + "\n".join(f"- {s}" for s in strategy_options)
        + "\n\nPick exactly one recovery strategy name from the list above "
        "that best addresses this error. Respond with ONLY the strategy "
        "name, nothing else."
    )
    response = model.invoke(prompt)
    return response.content.strip()
