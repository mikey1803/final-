"""Generate the real, data-backed figures used in paper/main.tex.

Every number plotted here is read from files already on disk in
backend/results/ (the pre-registered held-out kappa run and the earlier
crossover-theory validation run), or recomputed the same deterministic way
crossover_theory_validation.py does (same seeds, same persisted classifier
model), never re-simulated or re-fit. Run from the backend/ directory so the
project imports resolve:

    cd backend && ../.venv/Scripts/python.exe ../paper/scripts/make_paper_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "src"))

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


def fig_confusion_matrix():
    """Fig: empirical confusion matrix of the trained FailureClassifier.

    Reproduces the exact array crossover_theory_validation.py uses for the
    paper's 0.939 mean-diagonal-accuracy figure: same persisted model
    (backend/data/classifier_model.joblib), same compute_empirical_confusion_matrix
    call with its default seed.
    """
    from src.evaluation.classifier_context import (
        compute_empirical_confusion_matrix,
        load_or_train_classifier,
        LABELS,
    )

    clf = load_or_train_classifier()
    c_emp = compute_empirical_confusion_matrix(clf)
    acc = float(np.diag(c_emp).mean())
    print(f"confusion matrix mean diagonal accuracy = {acc:.4f}")

    short_labels = [l.replace("_", "\n") for l in LABELS]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(c_emp, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(short_labels, fontsize=5.5, rotation=90)
    ax.set_yticklabels(short_labels, fontsize=5.5)
    ax.set_xlabel("Predicted type", fontsize=8)
    ax.set_ylabel("True type", fontsize=8)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = c_emp[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=4.8, color="white" if v > 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_confusion_matrix.png", dpi=300)
    plt.close(fig)


def fig_confusion_and_channels_combined():
    """Combined two-panel figure: confusion matrix (left) + channel decomposition (right).

    Same source data as fig_confusion_matrix() and fig_channel_decomposition();
    merged into one figure to save a float's worth of caption/spacing overhead
    in the final PDF.
    """
    from src.evaluation.classifier_context import (
        compute_empirical_confusion_matrix,
        load_or_train_classifier,
        LABELS,
    )

    clf = load_or_train_classifier()
    c_emp = compute_empirical_confusion_matrix(clf)
    acc = float(np.diag(c_emp).mean())

    ct_dir = BACKEND / "results" / "phase10" / "crossover_theory"
    bucket_df = pd.read_parquet(ct_dir / "gap_shrinkage_per_bucket.parquet")
    summary_df = pd.read_parquet(ct_dir / "channel_summary.parquet")
    ceiling = float(summary_df["ceiling_reduction"].iloc[0])
    buckets = bucket_df["bucket"].astype(int).tolist()
    gap_shrinkage = bucket_df["gap_shrinkage"].tolist()

    print(f"[combined] confusion acc={acc:.4f}  ceiling={ceiling:.4f}")

    short_labels = [l.replace("_", "\n") for l in LABELS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                    gridspec_kw={"width_ratios": [1.15, 1]})

    im = ax1.imshow(c_emp, cmap="Greys", vmin=0, vmax=1)
    ax1.set_xticks(range(len(LABELS)))
    ax1.set_yticks(range(len(LABELS)))
    ax1.set_xticklabels(short_labels, fontsize=4.6, rotation=90)
    ax1.set_yticklabels(short_labels, fontsize=4.6)
    ax1.set_xlabel("Predicted type", fontsize=7.5)
    ax1.set_ylabel("True type", fontsize=7.5)
    ax1.set_title("(a) Confusion matrix", fontsize=8)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = c_emp[i, j]
            if v >= 0.01:
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=4.0, color="white" if v > 0.5 else "black")

    x = np.arange(len(buckets))
    ax2.bar(x, gap_shrinkage, color="#555555", width=0.6,
            label="Gap shrinkage\n(learnability)")
    ax2.axhline(ceiling, color="black", linestyle="--", linewidth=1.1,
                label="Ceiling reduction\n(0.0120, all buckets)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$t'_{{{b}}}$" for b in buckets], fontsize=7)
    ax2.set_xlabel("Predicted bucket", fontsize=7.5)
    ax2.set_ylabel("Loss magnitude", fontsize=7.5)
    ax2.set_title("(b) Ceiling vs. learnability loss", fontsize=8)
    ax2.legend(fontsize=5.2, loc="upper right", frameon=False)
    ax2.tick_params(labelsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_confusion_and_channels.png", dpi=300)
    plt.close(fig)


def fig_channel_decomposition():
    """Fig: ceiling loss vs. gap-shrinkage (learnability) loss per predicted bucket.

    Direct read of the Aug-18 crossover_theory_validation.py output
    (gap_shrinkage_per_bucket.parquet, channel_summary.parquet) — the same
    run behind Table tab:classification and Section VII-A's 0.0120 / 0.0020
    figures. No recomputation.
    """
    ct_dir = BACKEND / "results" / "phase10" / "crossover_theory"
    bucket_df = pd.read_parquet(ct_dir / "gap_shrinkage_per_bucket.parquet")
    summary_df = pd.read_parquet(ct_dir / "channel_summary.parquet")
    ceiling = float(summary_df["ceiling_reduction"].iloc[0])
    mean_gap = float(summary_df["mean_gap_shrinkage"].iloc[0])
    print(f"ceiling_reduction={ceiling:.4f}  mean_gap_shrinkage={mean_gap:.4f}")

    buckets = bucket_df["bucket"].astype(int).tolist()
    gap_shrinkage = bucket_df["gap_shrinkage"].tolist()

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(buckets))
    bars = ax.bar(x, gap_shrinkage, color="#555555", width=0.6,
                   label="Gap shrinkage (learnability)")
    ax.axhline(ceiling, color="black", linestyle="--", linewidth=1.1,
               label="Ceiling reduction (0.0120, all buckets)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$t'_{{{b}}}$" for b in buckets], fontsize=7)
    ax.set_xlabel("Predicted bucket", fontsize=8)
    ax.set_ylabel("Loss magnitude", fontsize=8)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.28),
              ncol=1, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_channel_decomposition.png", dpi=300)
    plt.close(fig)


def fig_percell_heldout():
    """Fig: Branch A per-cell held-out ratio, the same 8 rows as Table tab:percell.

    Direct read of kappa_preregistered_branchA_test.parquet, the pre-registered
    Aug-18 held-out run — exact same numbers already printed in the table,
    just plotted.
    """
    ct_dir = BACKEND / "results" / "phase10" / "crossover_theory"
    df = pd.read_parquet(ct_dir / "kappa_preregistered_branchA_test.parquet")
    df = df.sort_values("env_id")
    envs = [e.replace("gap0.20_idx", "idx") for e in df["env_id"]]
    ratios = df["ratio"].tolist()

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = ["#888888" if r >= 1.0 else "#222222" for r in ratios]
    x = np.arange(len(envs))
    ax.bar(x, ratios, color=colors, width=0.6)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("Predicted / observed $T$ (ratio)", fontsize=8)
    ax.set_xlabel("Held-out TEST environment", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_percell_heldout.png", dpi=300)
    plt.close(fig)


def fig_drift_attempts():
    """Fig: R^2 of the three pre-registered drift sign-rule attempts.

    Values are hardcoded from Table tab:drift in main.tex itself (0.012,
    0.579-0.735 retracted, 0.676) plus the sign-accuracy-vs-majority-baseline
    comparison (79.7% vs 79.7%) from Section VII-F's own text. This is a
    plot of already-published, already-verified numbers, not a new
    computation.
    """
    attempts = ["Attempt 1\n(survival,\ngap-change)", "Attempt 2\n(magnitude,\nrecovery-ratio)\n[retracted]", "Attempt 3\n(env. regret,\nreversion)"]
    r2_low = [0.012, 0.579, 0.676]
    r2_high = [0.012, 0.735, 0.676]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    x = np.arange(3)
    colors = ["#666666", "#bbbbbb", "#222222"]
    for i in x:
        ax.bar(i, r2_high[i], color=colors[i], width=0.55)
        if r2_high[i] != r2_low[i]:
            ax.plot([i, i], [r2_low[i], r2_high[i]], color="black", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(attempts, fontsize=5.8)
    ax.set_ylabel("$R^2$", fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.tick_params(labelsize=7)
    ax.set_title("Sign-rule attempts", fontsize=8)

    ax2 = axes[1]
    ax2.bar(["Attempt 3", "Majority\nbaseline"], [79.7, 79.7],
            color=["#222222", "#bbbbbb"], width=0.55)
    ax2.set_ylabel("Sign accuracy (\\%)", fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.tick_params(labelsize=7)
    ax2.set_title("Held-out sign accuracy", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_drift_attempts.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_confusion_and_channels_combined()
    fig_percell_heldout()
    fig_drift_attempts()
    print("Done. Figures written to", OUT_DIR)
