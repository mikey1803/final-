"""Failure classification for the wrapper: rules first, trained model second.

Transport-level failures are classified by fixed rules:
- a timeout exception, or error text starting ``TimeoutError:`` / containing
  "timed out" -> ``tool_timeout``;
- an HTTP 4xx/5xx status (or text starting ``HTTP nnn``, optionally after an
  exception-type prefix), a connection error, or an open circuit -> ``tool_error``.
A rule match gets confidence 1.0 -- a design choice (the rule is
deterministic), not a measured accuracy.

Anything else goes to the existing trained Layer-1 ``FailureClassifier``
(``backend/data/classifier_model.joblib``, loaded once, never retrained);
confidence is its top ``predict_proba``. On real HTTP error text that model
gave top confidences of 0.16-0.42 on six hand-written examples (2026-10-22,
not a measurement on a sample), which is why transport failures are not left
to it. If the model file is missing the result is ``unknown`` with
confidence 0.0, which escalates.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["Classification", "RuleThenTrainedClassifier", "DEFAULT_MODEL_PATH", "CIRCUIT_OPEN_PREFIX"]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "classifier_model.joblib"
CIRCUIT_OPEN_PREFIX = "CircuitOpen:"
RULE_CONFIDENCE = 1.0   # design choice, see module docstring

_HTTP = re.compile(r"^(?:\w+: )?HTTP (\d{3})\b")   # "HTTP 503: ..." or "RuntimeError: HTTP 503 ..."


@dataclass(frozen=True)
class Classification:
    failure_type: str
    confidence: float
    source: str                     # "rule:<name>" | "trained" | "none"
    proba: Optional[dict] = None


def _is_timeout_exc(exc: Optional[BaseException]) -> bool:
    if exc is None:
        return False
    if isinstance(exc, TimeoutError):
        return True
    try:
        import requests
        return isinstance(exc, requests.exceptions.Timeout)
    except ImportError:  # pragma: no cover
        return False


def _is_connection_exc(exc: Optional[BaseException]) -> bool:
    if exc is None:
        return False
    if isinstance(exc, ConnectionError):
        return True
    try:
        import requests
        return isinstance(exc, requests.exceptions.ConnectionError) and not _is_timeout_exc(exc)
    except ImportError:  # pragma: no cover
        return False


class RuleThenTrainedClassifier:
    def __init__(self, model_path: Optional[Path] = DEFAULT_MODEL_PATH, trained=None) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._trained = trained
        self._load_failed = False

    def _model(self):
        if self._trained is None and not self._load_failed:
            if self._model_path is None or not self._model_path.exists():
                self._load_failed = True
                return None
            from src.failures.classifier import FailureClassifier
            clf = FailureClassifier()
            with warnings.catch_warnings():   # sklearn version-mismatch notice on load
                warnings.simplefilter("ignore")
                clf.load(self._model_path)
            self._trained = clf
        return self._trained

    def rule(self, error_text: Optional[str], http_status: Optional[int],
             exc: Optional[BaseException]) -> Optional[Classification]:
        text = error_text or ""
        # the text itself, and the text after one "Name: " prefix (a wrapper exception around the real error)
        heads = [text] + ([text.split(": ", 1)[1]] if re.match(r"^\w+: ", text) else [])
        if any(h.startswith(CIRCUIT_OPEN_PREFIX) for h in heads):
            return Classification("tool_error", RULE_CONFIDENCE, "rule:circuit_open")
        if (_is_timeout_exc(exc) or any(h.startswith("TimeoutError:") for h in heads)
                or "timed out" in text.lower()):
            return Classification("tool_timeout", RULE_CONFIDENCE, "rule:timeout")
        m = _HTTP.match(text)
        status = http_status if http_status is not None else (int(m.group(1)) if m else None)
        if status is not None and 400 <= status <= 599:
            return Classification("tool_error", RULE_CONFIDENCE, f"rule:http_{status // 100}xx")
        if _is_connection_exc(exc) or any(h.startswith("ConnectionError:") for h in heads):
            return Classification("tool_error", RULE_CONFIDENCE, "rule:connection_error")
        return None

    def classify(self, error_text: Optional[str], tool_name: str, http_status: Optional[int] = None,
                 exc: Optional[BaseException] = None) -> Classification:
        hit = self.rule(error_text, http_status, exc)
        if hit is not None:
            return hit
        model = self._model()
        if model is None:
            return Classification("unknown", 0.0, "none")
        proba = model.predict_proba(error_text or "", {"tool_name": tool_name})
        top = max(proba, key=proba.get)
        return Classification(top.value, float(proba[top]), "trained",
                              {k.value: round(float(v), 4) for k, v in proba.items()})
