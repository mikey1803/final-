#!/usr/bin/env python
"""Dashboard content generator: real confusion matrix + per-class
precision/recall/F1 for the actual trained failure classifier, on a fresh
held-out synthetic test set.

NOT part of the reproduce.py pipeline (Task 16) -- this produces dashboard
UI content (frontend/dashboard-react/public/data/classifier_diagnostics.json),
not a paper figure/table, so it isn't in reproduce.py's scope or validated by
its Stage 7 comparison. Run manually to regenerate:

    cd backend && python experiments/classifier_diagnostics.py

Reuses the same real trained classifier
(``src.evaluation.classifier_context.load_or_train_classifier``) and the
same balanced fresh-test-set generation approach as
``compute_empirical_confusion_matrix`` in that module (same seed, 4242, for
consistency with the Classifier-Noise Sweep tab's confusion matrix) --
except this also keeps raw (non-row-normalized) counts, since per-class
precision needs column totals that a row-stochastic matrix already threw
away.
"""

import json
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
)

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from src.evaluation.classifier_context import _make_step_context, load_or_train_classifier  # noqa: E402
from src.failures.injector import FailureInjector  # noqa: E402
from src.failures.taxonomy import FailureType  # noqa: E402

TYPES: list[FailureType] = list(FailureType)
LABELS: list[str] = [ft.value for ft in TYPES]
PER_TYPE = 150
SEED = 4242  # matches compute_empirical_confusion_matrix's default, for consistency

_OUTPUT_PATH = (
    _BACKEND_DIR.parent / "frontend" / "dashboard-react" / "public" / "data" / "classifier_diagnostics.json"
)


def main() -> None:
    clf = load_or_train_classifier()

    injector = FailureInjector(injection_rate=1.0, seed=SEED)
    y_true: list[str] = []
    y_pred: list[str] = []
    for ft in TYPES:
        for i in range(PER_TYPE):
            ctx = _make_step_context(i, tag=f"diag_{ft.value}")
            event = injector._build_event(ft, ctx)
            predicted = clf.predict(event.raw_error, event.step_context)
            y_true.append(ft.value)
            y_pred.append(predicted.value)

    cm_counts = sk_confusion_matrix(y_true, y_pred, labels=LABELS).astype(float)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_normalized = cm_counts / row_sums

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )

    per_class = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }

    macro = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision.mean()), 4),
        "recall": round(float(recall.mean()), 4),
        "f1": round(float(f1.mean()), 4),
    }

    payload = {
        "labels": LABELS,
        "confusion_matrix": [[round(float(v), 4) for v in row] for row in cm_normalized],
        "per_class": per_class,
        "macro": macro,
        "n_per_type": PER_TYPE,
        "seed": SEED,
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {_OUTPUT_PATH} -- macro accuracy {macro['accuracy']:.1%}")


if __name__ == "__main__":
    main()
