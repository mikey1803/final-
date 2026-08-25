"""Adapter wrapping the existing trained ``FailureClassifier`` behind the
``Classifier`` protocol.

Does not retrain, modify, or replace ``src.failures.classifier.FailureClassifier``
in any way -- this module only wraps an already-constructed instance.

Limitation: the real classifier predicts from raw error TEXT (plus a small
step-context dict), not from an integer type index (see
``src.failures.classifier.FailureClassifier.predict``). There is therefore
no direct function ``int -> int`` to wrap: ``predict(true_type, rng)`` must
first synthesize a *representative* error string for ``true_type`` (via the
existing ``src.failures.injector.FailureInjector``), then run that text
through the real classifier, then map its ``FailureType`` output back to an
index. This synthesis step is itself a source of variance -- which
representative text gets generated for a given true type affects what the
classifier predicts, which is exactly how the "empirical confusion matrix"
in ``src.evaluation.classifier_context.compute_empirical_confusion_matrix``
arises. That behavior is preserved here via an injectable ``text_sampler``
hook (defaulting to a reasonable synthesis using ``FailureInjector``), so
callers who need the exact same sampling distribution used elsewhere in the
repo can supply their own sampler instead of relying on the default.
"""

from typing import Callable, Optional

import numpy as np

from src.failures.injector import FailureInjector
from src.failures.taxonomy import FailureType

__all__ = ["TrainedClassifierAdapter"]

_TYPES: list[FailureType] = list(FailureType)

# (true_type_idx, rng) -> (raw_error_text, step_context)
TextSampler = Callable[[int, np.random.Generator], tuple[str, dict]]

_KNOWN_TOOLS: list[str] = [
    "web_search", "calculator", "file_reader", "api_caller",
    "database_query", "code_executor",
]


def _default_text_sampler(true_type_idx: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Synthesize one representative (raw_error, step_context) pair for a type.

    Draws exactly one integer from ``rng`` to seed a fresh, disposable
    ``FailureInjector`` and pick a tool/step index -- this is what
    produces different (and therefore sometimes differently-classified)
    error text across repeated calls for the same true type, mirroring how
    the real deployment pipeline sees varied error text per failure type
    rather than one fixed string.
    """
    failure_type = _TYPES[true_type_idx]
    draw = int(rng.integers(0, 1_000_000))
    injector = FailureInjector(injection_rate=1.0, seed=draw)
    step_context = {
        "step_index": draw,
        "tool_name": _KNOWN_TOOLS[draw % len(_KNOWN_TOOLS)],
        "tool_args": {"query": f"kernel_sample_{draw}"},
        "goal": f"Perform subtask {draw}",
        "tool_output": f"Output for step {draw}",
    }
    event = injector.generate_event(failure_type, step_context)
    return event.raw_error, event.step_context


class TrainedClassifierAdapter:
    """Wraps a trained ``FailureClassifier`` behind the ``Classifier`` protocol.

    Parameters
    ----------
    clf : src.failures.classifier.FailureClassifier
        An already-trained (or already-loaded) classifier instance. Not
        retrained, modified, or copied.
    text_sampler : callable, optional
        ``(true_type_idx, rng) -> (raw_error, step_context)``. Defaults to
        ``_default_text_sampler``, which synthesizes a fresh representative
        error via ``FailureInjector``. Supply a custom sampler to reuse a
        fixed corpus of held-out labelled errors instead.

    Notes
    -----
    Every ``predict`` call constructs a disposable ``FailureInjector``
    (when using the default sampler) and runs TF-IDF + linear-model
    inference -- roughly 2ms/call, several orders of magnitude slower than
    ``PerfectClassifier``/``ConfusionClassifier``. Do not use this adapter
    for 200k-episode runs; a few thousand episodes is a reasonable budget.
    """

    def __init__(self, clf, text_sampler: Optional[TextSampler] = None) -> None:
        self._clf = clf
        self._text_sampler = text_sampler or _default_text_sampler

    def predict(self, true_type: int, rng: np.random.Generator) -> int:
        """Predict a type index for ``true_type`` via the real classifier.

        Parameters
        ----------
        true_type : int
            The true type index.
        rng : numpy.random.Generator
            Passed to the text sampler to synthesize representative error
            text (see module docstring).

        Returns
        -------
        int
            The predicted type index.
        """
        raw_error, step_context = self._text_sampler(true_type, rng)
        predicted_type = self._clf.predict(raw_error, step_context)
        return _TYPES.index(predicted_type)
