"""Classifier-in-the-loop context resolution for the evaluation harness.

The core harness always conditions selectors on the *true* failure type
(the "oracle" context_mode). This module supplies the machinery for two
more realistic alternatives:

  - "predicted": the selector only sees whatever the trained
    ``FailureClassifier`` actually predicts for a freshly-simulated error
    of the true type.
  - "synthetic(beta)": the selector sees a type drawn from a noise-
    interpolated confusion matrix ``C(beta) = beta * C_emp + (1-beta) * U``,
    letting classification accuracy be swept as a controlled variable
    instead of depending on whatever the classifier happens to score.

In both cases the *actual* recovery outcome is still sampled from
``SUCCESS_MATRIX[true_type]`` — only the selector's belief about the
failure type is degraded, not reality.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.failures.classifier import FailureClassifier
from src.failures.injector import FailureInjector
from src.failures.taxonomy import FailureType

TYPES: list[FailureType] = list(FailureType)
LABELS: list[str] = [ft.value for ft in TYPES]


def _make_step_context(i: int, tag: str = "ctx") -> dict:
    return {
        "step_index": i,
        "tool_name": ["web_search", "calculator", "db_query", "code_executor"][i % 4],
        "tool_args": {"query": f"{tag}_{i}"},
        "goal": f"Perform subtask {i}",
        "tool_output": f"Output for step {i}",
    }


def load_or_train_classifier(data_dir: Optional[Path] = None) -> FailureClassifier:
    """Load the persisted classifier from data/, training a fresh one if absent."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    clf = FailureClassifier()
    joblib_path = data_dir / "classifier_model.joblib"
    if joblib_path.exists():
        clf.load(joblib_path)
        return clf

    injector = FailureInjector(injection_rate=1.0, seed=99)
    for i in range(300):
        injector.inject(_make_step_context(i, tag="train"))
    clf.train(injector.get_injection_log())
    return clf


def compute_empirical_confusion_matrix(
    clf: FailureClassifier,
    per_type: int = 150,
    seed: int = 4242,
) -> np.ndarray:
    """Compute C_emp[true_idx, predicted_idx] as row-stochastic probabilities.

    Generates fresh labelled errors of every type (independent of whatever
    the classifier was trained on) so each row gets even coverage, then
    row-normalizes sklearn's confusion matrix into P(predicted | true).
    """
    injector = FailureInjector(injection_rate=1.0, seed=seed)
    y_true: list[str] = []
    y_pred: list[str] = []

    for ft in TYPES:
        for i in range(per_type):
            ctx = _make_step_context(i, tag=f"cm_{ft.value}")
            event = injector._build_event(ft, ctx)
            predicted = clf.predict(event.raw_error, event.step_context)
            y_true.append(ft.value)
            y_pred.append(predicted.value)

    cm = sk_confusion_matrix(y_true, y_pred, labels=LABELS).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def interpolated_confusion_matrix(c_emp: np.ndarray, beta: float) -> np.ndarray:
    """C(beta) = beta * C_emp + (1 - beta) * Uniform.

    beta=1.0 reproduces the classifier's empirical confusion matrix;
    beta=0.0 is uniform-random guessing among all failure types.
    """
    n = c_emp.shape[0]
    uniform = np.full((n, n), 1.0 / n)
    return beta * c_emp + (1.0 - beta) * uniform


def effective_accuracy(c_beta: np.ndarray) -> float:
    """Mean diagonal probability of C(beta) — the sweep's x-axis metric."""
    return float(np.mean(np.diag(c_beta)))


def sample_synthetic_context(
    true_type: FailureType,
    c_beta: np.ndarray,
    rng: random.Random,
) -> FailureType:
    """Draw an observed FailureType from row C(beta)[true_type]."""
    row = c_beta[TYPES.index(true_type)]
    r = rng.random()
    cum = 0.0
    for j, p in enumerate(row):
        cum += p
        if r < cum:
            return TYPES[j]
    return TYPES[-1]


def resolve_observed_type(
    mode: str,
    true_type: FailureType,
    rng: random.Random,
    clf: Optional[FailureClassifier] = None,
    injector: Optional[FailureInjector] = None,
    step_context: Optional[dict] = None,
    c_beta: Optional[np.ndarray] = None,
) -> FailureType:
    """Resolve the "observed" FailureType a selector should condition on.

    - "oracle": returns *true_type* unchanged (perfect information).
    - "predicted": simulates a fresh error of *true_type* and runs it
      through *clf* — requires ``clf`` and ``injector``.
    - "synthetic": draws from ``c_beta[true_type]`` — requires ``c_beta``
      (see ``interpolated_confusion_matrix``).
    """
    if mode == "oracle":
        return true_type
    if mode == "predicted":
        if clf is None or injector is None:
            raise ValueError("context_mode='predicted' requires clf and injector")
        event = injector._build_event(true_type, step_context or _make_step_context(0))
        return clf.predict(event.raw_error, event.step_context)
    if mode == "synthetic":
        if c_beta is None:
            raise ValueError("context_mode='synthetic' requires c_beta")
        return sample_synthetic_context(true_type, c_beta, rng)
    raise ValueError(f"Unknown context_mode '{mode}'; use 'oracle', 'predicted', or 'synthetic'.")
