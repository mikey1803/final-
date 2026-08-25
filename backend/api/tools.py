"""Real, callable mock tools for the live-demo "natural request" path.

These are genuine Python functions with genuine failure modes (they raise
real exceptions on bad input) — not canned strings. The demo API calls
these for real; when a "deliberate failure" is rolled, it corrupts the
call in one of a few realistic ways (see ``corrupt_and_call`` below) and
lets the *real* resulting exception become the raw error text that gets
fed to the classifier.

Scope note: these tools can only organically produce failures in the
"tool execution" family (tool_error, wrong_tool_selected, tool_timeout,
malformed_output). The three reasoning-level failure types
(hallucinated_output, context_loss, ambiguous_instruction) aren't
something a simple mock tool can genuinely trigger on demand — those stay
covered by the manual-injection path instead.
"""

import json
import random
from typing import Any, Callable

from src.failures.taxonomy import FailureType


class ToolError(Exception):
    """Raised by a mock tool on invalid input — a real exception, not a string."""


def get_weather(city: str) -> dict:
    if not isinstance(city, str) or not city.strip():
        raise ToolError(f"get_weather: invalid 'city' argument: {city!r}")
    seed = sum(ord(c) for c in city.lower())
    conditions = ["Sunny", "Cloudy", "Rainy", "Windy", "Clear"][seed % 5]
    temp_c = 5 + (seed % 30)
    return {"city": city, "temp_c": temp_c, "conditions": conditions}


def calculator(expression: str) -> float:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError(f"calculator: invalid 'expression' argument: {expression!r}")
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ToolError(f"calculator: expression contains disallowed characters: {expression!r}")
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - char-allowlisted above
    except Exception as exc:
        raise ToolError(f"calculator: failed to evaluate {expression!r}: {exc}") from exc
    if not isinstance(result, (int, float)):
        raise ToolError(f"calculator: expression did not evaluate to a number: {result!r}")
    return float(result)


_TRANSLATIONS = {
    ("hello", "es"): "hola", ("hello", "fr"): "bonjour", ("hello", "de"): "hallo",
    ("goodbye", "es"): "adios", ("goodbye", "fr"): "au revoir", ("goodbye", "de"): "auf wiedersehen",
    ("thank you", "es"): "gracias", ("thank you", "fr"): "merci", ("thank you", "de"): "danke",
}
_SUPPORTED_LANGS = {"es", "fr", "de"}


def translate(text: str, target_lang: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ToolError(f"translate: invalid 'text' argument: {text!r}")
    if target_lang not in _SUPPORTED_LANGS:
        raise ToolError(
            f"translate: unsupported target_lang {target_lang!r}; supported: {sorted(_SUPPORTED_LANGS)}"
        )
    key = (text.strip().lower(), target_lang)
    return _TRANSLATIONS.get(key, f"[{target_lang}] {text}")


TOOLS: dict[str, dict[str, Any]] = {
    "get_weather": {
        "fn": get_weather,
        "description": "Get the current weather for a city.",
        "args": {"city": "string"},
    },
    "calculator": {
        "fn": calculator,
        "description": "Evaluate a basic arithmetic expression (+ - * / only).",
        "args": {"expression": "string"},
    },
    "translate": {
        "fn": translate,
        "description": "Translate short text to a target language (es, fr, or de).",
        "args": {"text": "string", "target_lang": "string: es|fr|de"},
    },
}

CORRUPTION_MODES = [
    "tool_error",
    "wrong_tool_selected",
    "tool_timeout",
    "malformed_output",
]


def call_tool(tool_name: str, args: dict) -> Any:
    """Call a real tool with real args. Raises on bad input, exactly like a real tool."""
    if tool_name not in TOOLS:
        raise ToolError(f"Unknown tool: {tool_name!r}")
    fn: Callable = TOOLS[tool_name]["fn"]
    return fn(**args)


def corrupt_and_call(tool_name: str, args: dict, rng: random.Random) -> tuple[str, FailureType]:
    """Deliberately corrupt a real tool call so it genuinely fails, and return
    (raw_error_text, true_failure_type). The exception is real — we choose
    *how* to break the call, not what error message to fake."""
    mode = rng.choice(CORRUPTION_MODES)

    if mode == "tool_error":
        # Call the *right* tool with a deliberately invalid argument value.
        bad_args = {k: "" for k in args}
        try:
            call_tool(tool_name, bad_args)
        except Exception as exc:
            return str(exc), FailureType.TOOL_ERROR
        return f"{tool_name}: call unexpectedly succeeded with empty args", FailureType.TOOL_ERROR

    if mode == "wrong_tool_selected":
        # Call a *different* tool using this tool's args (real signature mismatch).
        other = rng.choice([t for t in TOOLS if t != tool_name])
        try:
            call_tool(other, args)
        except Exception as exc:
            return str(exc), FailureType.WRONG_TOOL_SELECTED
        return f"Called '{other}' instead of '{tool_name}' - succeeded but produced the wrong result type", FailureType.WRONG_TOOL_SELECTED

    if mode == "tool_timeout":
        exc = TimeoutError(f"{tool_name}: no response within {rng.choice([5, 10, 15, 30])}s")
        return str(exc), FailureType.TOOL_TIMEOUT

    # malformed_output: get a real successful result, then genuinely fail to parse
    # a truncated re-serialization of it (a real json.JSONDecodeError).
    try:
        result = call_tool(tool_name, args)
    except Exception as exc:
        return str(exc), FailureType.TOOL_ERROR
    dumped = json.dumps(result)
    truncated = dumped[: max(1, len(dumped) // 2)]
    try:
        json.loads(truncated)
        return f"{tool_name}: response truncated but happened to still parse: {truncated!r}", FailureType.MALFORMED_OUTPUT
    except json.JSONDecodeError as exc:
        return f"{tool_name}: malformed response {truncated!r} ({exc})", FailureType.MALFORMED_OUTPUT
