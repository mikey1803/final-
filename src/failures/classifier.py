"""sklearn-based failure classifier for categorising agent failures.

Uses TF-IDF features over the raw error text (supplemented with sparse
one-hot features from step_context fields like ``tool_name``) fed into a
LinearSVC.  In production the classifier does NOT see the ground-truth
failure type — it must predict it solely from the error string and context.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

from src.failures.taxonomy import FailureEvent, FailureType


class FailureClassifier:
    """Predicts the FailureType from raw error text and step_context.

    The pipeline is:
        1. ``TfidfVectorizer`` transforms ``raw_error`` into a sparse
           feature matrix.
        2. Sparse one-hot columns are appended for categorical step_context
           fields (currently ``tool_name``).
        3. ``LinearSVC`` is trained on the combined feature matrix.

    The ``predict_proba`` method uses ``LogisticRegression`` (with ``warm_start``)
    so that class-level probabilities are available for the dashboard.

    Example::

        clf = FailureClassifier()
        clf.train(training_events)
        predicted = clf.predict(raw_error_text, step_context_dict)
        probs    = clf.predict_proba(raw_error_text, step_context_dict)
    """

    # Known tool names — used for one-hot encoding of the tool_name context field.
    KNOWN_TOOLS: list[str] = [
        "web_search", "calculator", "db_query", "code_executor",
        "translator", "summariser", "correct_tool", "unknown_tool",
    ]

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
        )
        self.svc = LinearSVC(max_iter=10000, class_weight="balanced")
        self.prob_model = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs",
        )
        self.label_encoder = LabelEncoder()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_features(
        self,
        raw_errors: list[str],
        step_contexts: list[dict],
        fit: bool = False,
    ):
        """Build the combined TF-IDF + one-hot feature matrix.

        Args:
            raw_errors: List of error strings to vectorise.
            step_contexts: Corresponding step_context dicts.
            fit: If ``True`` the vectoriser and label encoder are fitted
                on the supplied data (training mode).

        Returns:
            A sparse scipy CSR matrix of shape ``(n_samples, n_features)``.
        """
        if fit:
            tfidf_matrix = self.vectorizer.fit_transform(raw_errors)
        else:
            tfidf_matrix = self.vectorizer.transform(raw_errors)

        tool_names = [ctx.get("tool_name", "unknown_tool") for ctx in step_contexts]
        tool_onehot = self._encode_tool_names(tool_names, fit=fit)

        return hstack([tfidf_matrix, tool_onehot])

    def _encode_tool_names(
        self, tool_names: list[str], fit: bool = False,
    ) -> csr_matrix:
        """One-hot encode tool names into a sparse matrix."""
        rows = []
        indices = []
        for i, name in enumerate(tool_names):
            if name in self.KNOWN_TOOLS:
                idx = self.KNOWN_TOOLS.index(name)
                rows.append(i)
                indices.append(idx)
        data = np.ones(len(rows), dtype=np.float64)
        return csr_matrix(
            (data, (rows, indices)),
            shape=(len(tool_names), len(self.KNOWN_TOOLS)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, events: list[FailureEvent]) -> dict:
        """Fit the vectoriser and classifiers on a list of labelled events.

        Args:
            events: Labeled ``FailureEvent`` objects whose ``failure_type``
                fields serve as ground-truth labels.

        Returns:
            A dict with training accuracy and the number of samples used.
        """
        raw_errors = [e.raw_error for e in events]
        step_contexts = [e.step_context for e in events]
        labels = self.label_encoder.fit_transform(
            [e.failure_type.value for e in events]
        )

        X = self._build_features(raw_errors, step_contexts, fit=True)
        self.svc.fit(X, labels)
        self.prob_model.fit(X, labels)
        self._fitted = True

        train_preds = self.svc.predict(X)
        train_acc = accuracy_score(labels, train_preds)

        return {"train_accuracy": float(train_acc), "n_samples": len(events)}

    def predict(self, raw_error: str, step_context: dict) -> FailureType:
        """Predict the failure type for a single unseen error.

        Args:
            raw_error: The error string or malformed output to classify.
            step_context: Context dict from the agent step.

        Returns:
            The predicted ``FailureType`` enum value.
        """
        self._ensure_fitted()
        X = self._build_features([raw_error], [step_context])
        label_idx = self.svc.predict(X)[0]
        return FailureType(self.label_encoder.inverse_transform([label_idx])[0])

    def predict_proba(self, raw_error: str, step_context: dict) -> dict[FailureType, float]:
        """Return a confidence score for every FailureType.

        Uses the ``LogisticRegression`` model so that real probabilities
        are returned rather than SVM decision-function values.

        Args:
            raw_error: The error string or malformed output to classify.
            step_context: Context dict from the agent step.

        Returns:
            A dict mapping each ``FailureType`` to its predicted probability.
        """
        self._ensure_fitted()
        X = self._build_features([raw_error], [step_context])
        proba = self.prob_model.predict_proba(X)[0]
        classes = self.label_encoder.inverse_transform(self.prob_model.classes_)
        return {FailureType(c): float(p) for c, p in zip(classes, proba)}

    def evaluate(self, test_events: list[FailureEvent]) -> dict:
        """Compute accuracy, precision, recall, and F1 on a test set.

        All metrics are macro-averaged so that each class contributes
        equally regardless of frequency.

        Args:
            test_events: Held-out ``FailureEvent`` objects with known labels.

        Returns:
            Dict with keys ``accuracy``, ``precision``, ``recall``, ``f1``
            (all floats).
        """
        self._ensure_fitted()
        raw_errors = [e.raw_error for e in test_events]
        step_contexts = [e.step_context for e in test_events]
        y_true = [e.failure_type.value for e in test_events]

        X = self._build_features(raw_errors, step_contexts)
        y_pred = self.label_encoder.inverse_transform(self.svc.predict(X))

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def save(self, path: str | Path) -> None:
        """Persist the trained classifier to disk via joblib.

        Saves the vectoriser, SVC, logistic-regression model, and label
        encoder in a single bundle.
        """
        self._ensure_fitted()
        bundle = {
            "vectorizer": self.vectorizer,
            "svc": self.svc,
            "prob_model": self.prob_model,
            "label_encoder": self.label_encoder,
        }
        joblib.dump(bundle, str(path))

    def load(self, path: str | Path) -> None:
        """Restore a previously saved classifier from disk.

        Overwrites any currently loaded models.
        """
        bundle = joblib.load(str(path))
        self.vectorizer = bundle["vectorizer"]
        self.svc = bundle["svc"]
        self.prob_model = bundle["prob_model"]
        self.label_encoder = bundle["label_encoder"]
        self._fitted = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_fitted(self) -> None:
        """Raise if the classifier has not been trained yet."""
        if not self._fitted:
            raise RuntimeError("Classifier has not been trained yet. Call train() first.")
