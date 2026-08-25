"""Generate the seven vector figures for ieee_v3.tex into ./figs/.

matplotlib + numpy only. Run: python make_figs.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

C = {
    "blue": "#2f5c8f",
    "orange": "#d6802b",
    "green": "#4e8b57",
    "red": "#b4483f",
    "purple": "#6f5b9e",
    "grey": "#8a8a8a",
    "ink": "#222222",   # text token -- annotations/labels never wear a series color
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.titlesize": 7.5,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "text.color": C["ink"],
    "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT_DIR, exist_ok=True)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=7.5, pad=6)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"wrote {path}")


def calibrate_and_save(make_fn, name, target_w, target_h, tol=0.008, max_iter=8):
    """savefig's tight bbox crops unused canvas margin, which can shift the
    aspect ratio of the saved PDF away from (target_w, target_h) -- and since
    ieee_v3.tex always renders at width=\\columnwidth, only the ASPECT RATIO
    of the saved PDF matters, not its absolute size. This holds the figsize
    WIDTH fixed at target_w (the hard constraint) and iterates the internal
    canvas HEIGHT passed to make_fn until the post-crop aspect ratio matches
    target_h/target_w within `tol` -- the fallback the brief authorizes
    ("adjust that figure's figsize height") rather than fighting the .tex.
    """
    target_ratio = target_h / target_w
    h = target_h
    fig = None
    for i in range(max_iter):
        if fig is not None:
            plt.close(fig)
        fig = make_fn(h)
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        pad = plt.rcParams["savefig.pad_inches"]
        actual_w, actual_h = bbox.width + 2 * pad, bbox.height + 2 * pad
        actual_ratio = actual_h / actual_w
        rel_err = abs(actual_ratio - target_ratio) / target_ratio
        print(f"  [{name}] iter {i}: canvas_h={h:.4f}in -> saved=({actual_w:.3f},{actual_h:.3f})in "
              f"ratio={actual_ratio:.4f} (target {target_ratio:.4f}) rel_err={rel_err:.3f}")
        if rel_err <= tol:
            break
        h = h * (target_ratio / actual_ratio)
    save(fig, name)


# ---------------------------------------------------------------------------
# fig1_pipeline.pdf -- block diagram, no axes
# ---------------------------------------------------------------------------

def make_fig1(h=1.15):
    fig, ax = plt.subplots(figsize=(3.4, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    boxes = ["Failure\ninjection", "Classifier", "Bandit selector\n(K=6 strategies)", "Outcome\nmemory"]
    n = len(boxes)
    # Per-box widths -- "Bandit selector\n(K=6 strategies)" is the longest
    # label, so it gets a wider box rather than sharing a uniform width.
    box_widths = [16.5, 14.5, 27.0, 14.5]
    box_h = 11.5
    gap = (100 - sum(box_widths)) / (n - 1)
    centers = []
    x0 = 0.0
    for i, (label, box_w) in enumerate(zip(boxes, box_widths)):
        centers.append(x0 + box_w / 2)
        x0 += box_w + gap
    y_c = 14.5
    for i, (label, box_w) in enumerate(zip(boxes, box_widths)):
        x0 = centers[i] - box_w / 2
        fb = FancyBboxPatch(
            (x0, y_c - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.5,rounding_size=2.6",
            linewidth=1.0, edgecolor=C["blue"], facecolor="#eef3f8",
        )
        ax.add_patch(fb)
        ax.text(centers[i], y_c, label, ha="center", va="center", fontsize=5.5, color=C["ink"])

    arrow_labels = ["true type $t$", "predicted $t'$", "outcome"]
    label_y = y_c + box_h / 2 + 4.0  # clear margin above every box top
    for i in range(n - 1):
        x_start = centers[i] + box_widths[i] / 2
        x_end = centers[i + 1] - box_widths[i + 1] / 2
        arr = FancyArrowPatch(
            (x_start, y_c), (x_end, y_c), arrowstyle="-|>", mutation_scale=6,
            linewidth=1.0, color=C["ink"],
        )
        ax.add_patch(arr)
        ax.text((x_start + x_end) / 2, label_y, arrow_labels[i], ha="center", va="bottom", fontsize=5.0)

    ax.text(50, 1.2,
            "policy sees $t'$ only  —  true type $t$ never reaches it",
            ha="center", va="bottom", fontsize=6.0, color=C["red"])

    return fig


# ---------------------------------------------------------------------------
# fig2_confusion.pdf -- 7x7 confusion matrix heatmap
# ---------------------------------------------------------------------------

def make_fig2(h=2.11):
    wrapped = [
        "timeout", "tool error", "wrong tool", "hallucinated",
        "malformed", "context loss", "ambiguous",
    ]

    M = np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.05, 0.93, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
        [0.15, 0.18, 0.00, 0.03, 0.64, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ])

    fig, ax = plt.subplots(figsize=(3.4, h))
    # aspect="auto" (not "equal"): a forced-square 7x7 grid inside a wide
    # 3.4in canvas leaves blank side margin that savefig's tight bbox then
    # crops away, badly distorting the saved aspect ratio -- letting cells
    # be rectangular fills the actual canvas width instead.
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(wrapped, rotation=90, fontsize=5.0)
    ax.set_yticklabels(wrapped, fontsize=5.0)
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("True type")
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)

    for i in range(7):
        for j in range(7):
            v = M[i, j]
            if v > 0:
                color = "white" if v > 0.5 else C["ink"]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.0, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5.2)

    fig.subplots_adjust(bottom=0.30, left=0.30)
    return fig


# ---------------------------------------------------------------------------
# fig3_decomposition.pdf -- two stacked panels (panel (b) removed)
# ---------------------------------------------------------------------------

def make_fig3(h=2.11):
    # Explicit, hand-computed axes rectangles (figure-fraction coordinates)
    # instead of tight_layout/constrained_layout: both auto-layout engines
    # solve spacing from MEASURED text extents, and that measurement shifted
    # slightly depending on matplotlib's font-cache warm-up state -- a
    # layout that looked clean generating this figure alone reappeared
    # overlapping when generated as the 3rd of 7 figures in one process (as
    # make_figs.py's __main__ actually does). Fixed fractions sidestep that
    # non-determinism entirely: nothing here is decided by measuring
    # anything, so the result cannot depend on when/how it's measured.
    # figsize is pinned at (3.4, 2.11) exactly, unchanged from the original
    # spec -- only tight-bbox crops the saved PDF afterward, as with every
    # other figure in this file.
    left_ab, width_ab = 0.24, 0.72   # panel (a): now just 2 short category names
    left_c, width_c = 0.16, 0.80     # panel (c): numeric y-ticks

    # Panel (a)'s category-name tick labels (6.3pt serif words) need more
    # room than a numeric tick row -- 0.24 was measured for short numeric
    # ticks elsewhere and was NOT enough here; verified empirically that
    # 0.38 clears it with margin to spare.
    tick_zone_a = 0.38
    title_h = 0.12
    buf = 0.03

    y = 0.05                          # bottom margin
    # Panel (c) keeps a normal auto-placed xlabel/ylabel -- it's the
    # bottommost panel now, nothing below it for an xlabel to collide with.
    xlabel_zone_c = 0.30
    y += xlabel_zone_c
    c_bottom = y
    c_height = 0.56
    y += c_height
    c_top = y
    y += buf
    y += title_h
    c_title_y = y
    y += 0.08                         # gap between panels
    # Panel (a): vertical bars, categorical x-axis -- category names sit
    # directly under the axes via tick_zone_a, no separate xlabel needed.
    y += tick_zone_a
    a_bottom = y
    a_height = 0.34
    y += a_height
    a_top = y
    y += buf
    y += title_h
    a_title_y = y
    y += 0.04                         # top margin
    # y now equals h (2.11) by construction of the constants above.

    fig = plt.figure(figsize=(3.4, h))
    ax_c = fig.add_axes([left_c, c_bottom / h, width_c, c_height / h])
    ax_a = fig.add_axes([left_ab, a_bottom / h, width_ab, a_height / h])

    def fixed_title(text, title_y_in, left):
        fig.text(left, title_y_in / h, text, fontsize=7.5, va="bottom", ha="left")

    # (a) two channels -- VERTICAL bars so the two category names sit side
    # by side on the x axis instead of overlapping on the y axis.
    ax = ax_a
    fixed_title("(a) Two channels", a_title_y, left_ab)
    names = ["ceiling loss", "learnability loss"]
    vals = [0.0120, 0.0020]
    colors = [C["blue"], C["orange"]]
    x = np.arange(2)
    ax.bar(x, vals, color=colors, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=5.6, color=C["ink"])
    ax.set_ylabel("success-probability\nunits", fontsize=6.0)
    ax.set_ylim(0, 0.0175)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="y")
    ax.text(0.97, 0.90, "≈ 6×", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.0, color=C["ink"])

    # (c) static-side validation
    ax = ax_c
    fixed_title("(c) Static-side validation", c_title_y, left_c)
    beta = np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
    pred = np.array([0.4408, 0.4691, 0.5256, 0.5821, 0.6669, 0.7234])
    emp = np.array([0.4306, 0.4582, 0.5112, 0.5744, 0.6584, 0.7156])
    ax.plot(beta, pred, marker="o", ms=3, color=C["blue"], linestyle="-", label="predicted")
    ax.plot(beta, emp, marker="s", ms=3, color=C["orange"], linestyle="--", label="empirical")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.3)
    ax.set_xlabel("noise level β (0 = guessing, 1 = measured matrix)", fontsize=6.3)
    ax.set_ylabel("success rate")
    ax.set_ylim(0.40, 0.82)
    ax.grid()
    # Lower right: the curves rise from ~0.43 (bottom-left) to ~0.72
    # (upper-right), so the lower-right corner stays clear of both lines.
    ax.text(0.97, 0.04, "median abs. error 0.93 pp", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.2, color=C["ink"])

    return fig


# ---------------------------------------------------------------------------
# fig5_crossover.pdf -- cumulative regret vs horizon
# ---------------------------------------------------------------------------

def make_fig5(h=1.87):
    N = 7
    T = np.linspace(1, 5600, 2000)

    rho = 0.0214
    rho_c = 0.0338
    Tstar = 3053.0
    Tstar_c = 3126.0
    A = rho * Tstar / np.log(Tstar / N)
    A_c = rho_c * Tstar_c / np.log(Tstar_c / N)

    static_perfect = rho * T
    static_measured = rho_c * T
    learner_perfect = A * np.log(np.maximum(T / N, 1.0))
    learner_measured = A_c * np.log(np.maximum(T / N, 1.0))

    fig, ax = plt.subplots(figsize=(3.4, h))

    ax.axvspan(2600, 3600, color=C["grey"], alpha=0.15, lw=0)
    # Near the top of the axes: the highest curve value anywhere in the
    # shaded band [2600,3600] is static_measured at x=3600 (~121.7), so
    # y=148 clears every curve across the whole band width with margin --
    # the legend now occupies most of the lower-right area instead.
    ax.text(3100, 148, "honest resolution:\n≈ 3000 episodes", ha="center", va="top",
            fontsize=5.4, color=C["ink"])

    ax.plot(T, static_perfect, color=C["blue"], linestyle="-", label="static, perfect classifier")
    ax.plot(T, static_measured, color=C["blue"], linestyle="--", label="static, measured classifier")
    ax.plot(T, learner_perfect, color=C["orange"], linestyle="-", label="learner, perfect classifier")
    ax.plot(T, learner_measured, color=C["orange"], linestyle="--", label="learner, measured classifier")

    y_star = rho * Tstar
    y_star_c = rho_c * Tstar_c
    ax.plot([Tstar], [y_star], marker="o", ms=3.2, color=C["ink"], zorder=5)
    ax.plot([Tstar_c], [y_star_c], marker="o", ms=3.2, color=C["ink"], zorder=5)
    # Both curve pairs cross near (x~2850-3300), with a clear vertical gap
    # there between the two LOW curves (static/learner perfect, ~61-68) and
    # the two HIGH curves (static/learner measured, ~96-107) at every x in
    # that range. Short, near-vertical leader lines that stay inside that
    # gap band never cross a curve -- long diagonal leaders (the previous
    # version) did.
    ax.annotate("$T^*=3053$", xy=(Tstar, y_star), xytext=(2853, 90),
                fontsize=5.6, color=C["ink"], ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["ink"]))
    ax.annotate(r"$T^*_{conf}=3126$", xy=(Tstar_c, y_star_c), xytext=(3176, 82),
                fontsize=5.6, color=C["ink"], ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["ink"]))

    ax.set_xlabel("horizon $T$ (episodes)")
    ax.set_ylabel("cumulative regret")
    ax.set_ylim(0, 150)
    ax.set_xlim(0, 5600)
    ax.grid()
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, labelspacing=0.3, borderaxespad=0.2)

    return fig


# ---------------------------------------------------------------------------
# fig6_nonmonotone.pdf -- computed, two stacked panels sharing x
# ---------------------------------------------------------------------------

def _bern_kl(p, q):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _solve_Tstar(coeff, rho, N, T_hi0=1e6, iters=200):
    def f(T):
        return coeff * np.log(max(T / N, 1.0)) - rho * T

    if rho <= 0 or coeff <= 0:
        return np.nan
    T_peak = coeff / rho
    if T_peak <= N or f(T_peak) <= 0:
        return np.nan

    lo, hi = T_peak, max(T_hi0, T_peak * 2)
    tries = 0
    while f(hi) > 0 and tries < 60:
        hi *= 2.0
        tries += 1
    if f(hi) > 0:
        return np.nan

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_fig6(h=1.90):
    P = np.array([[0.80, 0.50], [0.30, 0.70]])
    p = np.array([0.5, 0.5])
    pi = np.argmax(P, axis=1)  # [0, 1]
    kappa = 0.044381
    N = 2  # two types, two strategies in this toy example

    accs = np.linspace(0.50, 1.00, 401)
    Tstars = np.full_like(accs, np.nan)
    gaps = np.full_like(accs, np.nan)

    oracle = float(np.sum(p * P.max(axis=1)))

    for i, a in enumerate(accs):
        conf = np.array([[a, 1 - a], [1 - a, a]])
        p_joint = p[:, None] * conf              # p_joint[t, t']
        p_prime = p_joint.sum(axis=0)            # marginal over predicted type
        posterior = p_joint / p_prime[None, :]   # P(true=t | pred=t')
        P_eff = posterior.T @ P                  # shape (2, 2): P_eff[t', k]

        cross_term = 0.0
        for t in range(2):
            for tp in range(2):
                cross_term += p[t] * conf[t, tp] * P[t, pi[tp]]
        rho_conf = oracle - cross_term

        coeff = 0.0
        for tp in range(2):
            row = P_eff[tp]
            kstar = int(np.argmax(row))
            best = row[kstar]
            for k in range(2):
                if k == kstar:
                    continue
                coeff += p_prime[tp] * (1.0 / _bern_kl(row[k], best))
        coeff *= kappa * N

        Tstars[i] = _solve_Tstar(coeff, rho_conf, N)
        # Predicted bucket t'=1 in the brief's 1-indexed notation is P_eff
        # row 0 here (0-indexed) -- verified numerically: only this row's
        # signed gap actually changes sign in [0.50, 1.00] (at a~=0.571,
        # matching the brief's expected flip point); P_eff row 1 (their
        # t'=2) never flips in this range, it's monotonic throughout.
        gaps[i] = abs(P_eff[0, 0] - P_eff[0, 1])

    finite = np.isfinite(gaps)
    flip_acc = accs[finite][np.argmin(gaps[finite])]

    fig, axes = plt.subplots(2, 1, figsize=(3.4, h), sharex=True)

    ax = axes[0]
    panel_title(ax, "(a) Crossover point vs. classifier accuracy")
    ax.plot(accs, Tstars, color=C["blue"])
    ax.set_yscale("log")
    ax.set_ylabel("$T^*$ (episodes,\nlog scale)", fontsize=6.3)
    ax.axvline(flip_acc, color=C["red"], linestyle="--", linewidth=1.0)
    ax.annotate(f"best arm flips\nat accuracy {flip_acc:.3f}", xy=(flip_acc, ax.get_ylim()[1]),
                xytext=(flip_acc + 0.045, ax.get_ylim()[1]),
                fontsize=5.4, color=C["red"], va="top", ha="left")
    ax.grid()

    ax = axes[1]
    panel_title(ax, "(b) Top-two gap at predicted bucket $t'=1$")
    ax.plot(accs, gaps, color=C["orange"])
    ax.axvline(flip_acc, color=C["red"], linestyle="--", linewidth=1.0)
    ax.set_ylabel("top-two gap")
    ax.set_xlabel("classifier accuracy")
    ax.grid()

    fig.tight_layout(h_pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# fig7_heldout.pdf -- two stacked panels
# ---------------------------------------------------------------------------

def make_fig7(h=2.45):
    fig, axes = plt.subplots(2, 1, figsize=(3.4, h))

    # (a) fitted constants, dot-and-whisker
    ax = axes[0]
    panel_title(ax, "(a) Fitted constants")

    precommit_lo, precommit_hi = 0.00045, 0.045
    ax.axvspan(precommit_lo, precommit_hi, color=C["green"], alpha=0.12, lw=0)
    ax.axvline(precommit_hi, color=C["green"], linestyle=":", linewidth=1.0)
    ax.text(precommit_hi - 0.001, 1.85, "pre-committed range", ha="right", va="top",
            fontsize=5.2, color=C["green"])

    rows = ["Branch A\n(transfer)", "Branch B\n(in-condition)"]
    points = [0.044381, 0.042559]
    lo = [0.0369, 0.0105]
    hi = [0.0544, 0.0426]
    colors = [C["blue"], C["orange"]]
    y = [1, 0]
    for yi, pt, l, h, col in zip(y, points, lo, hi, colors):
        ax.errorbar([pt], [yi], xerr=[[pt - l], [h - pt]], fmt="o", ms=3.5,
                    color=col, ecolor=col, elinewidth=1.2, capsize=2.2)
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=5.6)
    ax.set_ylim(-0.6, 2.1)
    ax.set_xlabel("fitted constant κ (95% bootstrap CI)")
    ax.grid(axis="x")

    # (b) per-cell held-out error, diverging bars
    ax = axes[1]
    panel_title(ax, "(b) Per-cell held-out error")

    data = [
        ("idx16", -66.7), ("idx11", -32.5), ("idx06", 28.3), ("idx00", 48.7),
        ("idx01", 52.7), ("idx24", 69.4), ("idx17", 475.2), ("idx08", 475.6),
    ]
    names = [d[0] for d in data]
    vals = np.array([d[1] for d in data])
    colors = [C["blue"] if v >= 0 else C["orange"] for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.62)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.4)
    ax.set_xlabel("signed relative error, Branch A (%)")
    xmax = vals.max() * 1.18
    xmin = vals.min() * 2.1  # extra room: negative labels sit past the (short) bar tips
    ax.set_xlim(xmin, xmax)
    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        off = xmax * 0.02 if v >= 0 else xmin * 0.02
        ax.text(v + off, yi, f"{v:+.1f}", va="center", ha=ha, fontsize=4.8, color=C["ink"])
    ax.grid(axis="x")

    fig.tight_layout(h_pad=1.1)
    return fig


# ---------------------------------------------------------------------------
# fig8_drift.pdf -- two panels side by side
# ---------------------------------------------------------------------------

def make_fig8(h=1.43):
    fig, axes = plt.subplots(1, 2, figsize=(3.4, h), constrained_layout=True)

    # (a) candidates
    ax = axes[0]
    panel_title(ax, "(a) Candidates")
    names = ["C1\nsurvival\n+ gap", "C2\nmagnitude\n+ recovery", "C3\nenv. regret\n+ reversion"]
    vals = [0.012, 0.735, 0.676]
    colors = [C["grey"], C["purple"], C["blue"]]
    x = np.arange(3)
    ax.bar(x, vals, color=colors, width=0.6)
    ax.errorbar([1], [0.735], yerr=[[0.735 - 0.579], [0.0]], fmt="none",
                ecolor=C["ink"], elinewidth=1.0, capsize=2.5)
    ax.text(1, 0.735 + 0.05, "0.579–0.735", ha="center", va="bottom", fontsize=5.0, color=C["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=4.6)
    ax.set_ylabel("$R^2$ (leave-one-\ncell-out)", fontsize=6.0)
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y")

    # (b) reversion
    ax = axes[1]
    panel_title(ax, "(b) Reversion")
    groups = ["reversion\n≥ 0.68", "reversion\n≈ 0"]
    neg = [13, 0]
    pos = [0, 32]
    x = np.arange(2)
    w = 0.32
    ax.bar(x - w / 2, neg, width=w, color=C["red"], label="negative advantage")
    ax.bar(x + w / 2, pos, width=w, color=C["green"], label="positive advantage")
    for xi, v in zip(x - w / 2, neg):
        if v > 0:
            ax.text(xi, v + 0.6, str(v), ha="center", va="bottom", fontsize=5.4, color=C["ink"])
    for xi, v in zip(x + w / 2, pos):
        if v > 0:
            ax.text(xi, v + 0.6, str(v), ha="center", va="bottom", fontsize=5.4, color=C["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=5.4)
    ax.set_ylabel("cells")
    ax.set_ylim(0, 41)
    ax.legend(loc="upper left", frameon=False, fontsize=5.2, handlelength=1.4, borderaxespad=0.2)
    ax.grid(axis="y")

    return fig


if __name__ == "__main__":
    calibrate_and_save(make_fig1, "fig1_pipeline.pdf", 3.4, 1.15)
    calibrate_and_save(make_fig2, "fig2_confusion.pdf", 3.4, 2.11)
    # fig3 uses fixed, hand-computed axes positions (see comment in
    # make_fig3), not the calibrate_and_save height search -- varying its
    # input height would desync those positions from the actual canvas.
    save(make_fig3(), "fig3_decomposition.pdf")
    calibrate_and_save(make_fig5, "fig5_crossover.pdf", 3.4, 1.87)
    calibrate_and_save(make_fig6, "fig6_nonmonotone.pdf", 3.4, 1.90)
    calibrate_and_save(make_fig7, "fig7_heldout.pdf", 3.4, 2.45)
    calibrate_and_save(make_fig8, "fig8_drift.pdf", 3.4, 1.43)

    expected = [
        "fig1_pipeline.pdf", "fig2_confusion.pdf", "fig3_decomposition.pdf",
        "fig5_crossover.pdf", "fig6_nonmonotone.pdf", "fig7_heldout.pdf", "fig8_drift.pdf",
    ]
    missing = [f for f in expected if not os.path.exists(os.path.join(OUT_DIR, f))]
    if missing:
        raise SystemExit(f"MISSING figures: {missing}")
    print(f"\nAll {len(expected)} figures present in {OUT_DIR}")
