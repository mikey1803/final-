"""Integration test for the FailureClassifier.

Generates 300 labelled FailureEvent examples using the FailureInjector,
trains the classifier on 80 % of them, evaluates on the remaining 20 %,
and prints metrics plus a handful of example predictions for sanity-checking.
Also generates a confusion matrix heatmap saved to data/confusion_matrix.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.failures.classifier import FailureClassifier
from src.failures.injector import FailureInjector
from src.failures.taxonomy import FailureType


def generate_events(n: int = 300, seed: int = 99):
    """Create n labelled FailureEvents via the injector."""
    injector = FailureInjector(injection_rate=1.0, seed=seed)
    for i in range(n):
        ctx = {
            "step_index": i,
            "tool_name": ["web_search", "calculator", "db_query", "code_executor"][i % 4],
            "tool_args": {"query": f"test_{i}"},
            "goal": f"Perform subtask {i}",
            "tool_output": f"Output for step {i}",
        }
        injector.inject(ctx)
    return injector.get_injection_log()


def print_examples(clf, events, n=8):
    """Show a few raw_error -> predicted vs actual comparisons."""
    print(f"\n  {'='*70}")
    print(f"  Sample predictions")
    print(f"  {'='*70}")
    shown = 0
    seen_types = set()
    for ev in events:
        if shown >= n:
            break
        # Spread across types so we see variety
        if ev.failure_type in seen_types and shown > 2:
            continue
        predicted = clf.predict(ev.raw_error, ev.step_context)
        match = "OK" if predicted == ev.failure_type else "MISS"
        snippet = ev.raw_error[:60].replace("\n", " ").encode("ascii", "replace").decode()
        print(
            f"  [{match:4s}]  actual={ev.failure_type.value:28s}  "
            f"predicted={predicted.value:28s}"
        )
        print(f"           text: {snippet}...")
        seen_types.add(ev.failure_type)
        shown += 1
    print()


def generate_confusion_matrix(clf, test_events, save_dir: Path):
    """Compute, print, and plot the confusion matrix for the classifier.

    Args:
        clf: A trained FailureClassifier.
        test_events: Held-out FailureEvent objects with known labels.
        save_dir: Directory to save the heatmap image to.
    """
    labels = sorted([ft.value for ft in FailureType])
    y_true = [e.failure_type.value for e in test_events]
    y_pred = [
        clf.predict(e.raw_error, e.step_context).value for e in test_events
    ]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- console table ---
    abbrevs = [v[:10] for v in labels]
    header = "".join(f"{a:>12s}" for a in abbrevs)
    print(f"\n  Confusion matrix (rows=actual, cols=predicted)")
    print(f"  {'':12s}{header}")
    print(f"  {'-' * (12 + 12 * len(labels))}")
    for i, row_label in enumerate(labels):
        row_str = "".join(f"{cm[i, j]:12d}" for j in range(len(labels)))
        print(f"  {abbrevs[i]:12s}{row_str}")
    print()

    # --- heatmap ---
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Failure Classifier — Confusion Matrix")
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "confusion_matrix.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {out_path}\n")


def main():
    events = generate_events(300)
    print(f"Generated {len(events)} labelled FailureEvent examples.")

    split = int(len(events) * 0.8)
    train_events = events[:split]
    test_events = events[split:]
    print(f"Train: {len(train_events)}  |  Test: {len(test_events)}")

    clf = FailureClassifier()
    train_info = clf.train(train_events)
    print(f"Training complete — train accuracy: {train_info['train_accuracy']:.3f}")

    metrics = clf.evaluate(test_events)
    print(f"\n  Test set metrics (macro-averaged):")
    print(f"    Accuracy  : {metrics['accuracy']:.3f}")
    print(f"    Precision : {metrics['precision']:.3f}")
    print(f"    Recall    : {metrics['recall']:.3f}")
    print(f"    F1        : {metrics['f1']:.3f}")

    print_examples(clf, test_events, n=10)

    # predict_proba check
    sample = test_events[0]
    probs = clf.predict_proba(sample.raw_error, sample.step_context)
    print(f"  predict_proba for first test sample ({sample.failure_type.value}):")
    for ftype, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 40)
        print(f"    {ftype.value:28s}  {p:.3f}  {bar}")
    print()

    # Save / load round-trip
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / "classifier_test.joblib"
    clf.save(save_path)
    print(f"  Saved classifier to {save_path}")

    clf2 = FailureClassifier()
    clf2.load(save_path)
    metrics2 = clf2.evaluate(test_events)
    assert metrics == metrics2, "Loaded classifier metrics must match original"
    print(f"  Loaded classifier — metrics match. Round-trip OK.\n")
    save_path.unlink(missing_ok=True)

    # Confusion matrix
    generate_confusion_matrix(clf, test_events, data_dir)


if __name__ == "__main__":
    main()
