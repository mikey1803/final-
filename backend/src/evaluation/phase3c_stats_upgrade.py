"""Phase 3C: statistical rigor upgrades, applied to the existing Phase 1
and Phase 2 experiments.

This module introduces no new simulation logic of its own — the upgrades
(Holm-Bonferroni correction, bootstrap 95% CIs, Cliff's delta, and the
num_trials=30->50 bump) all live in ``harness.py`` and ``stats_utils.py``
as additive changes. This driver just re-runs ``run_and_report()``
(Phase 1) and ``run_phase2_context_sweep()`` (Phase 2) so their saved
tables in results/phase1/ and results/phase2/ pick up the new columns,
then writes a short changelog to results/phase3c/.
"""

from pathlib import Path
from typing import Optional

from src.evaluation.harness import run_and_report, run_phase2_context_sweep
from src.evaluation.stats_utils import MC_VARIATION_NOTE


def run_phase3c(results_dir: Optional[Path] = None) -> None:
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase3c"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("  Phase 3C — 1/2: re-running Phase 1 main comparison at num_trials=50")
    print("=" * 100)
    run_and_report(num_episodes=500, num_trials=50)

    print("\n" + "=" * 100)
    print("  Phase 3C — 2/2: re-running Phase 2 context-noise sweep at num_trials=50")
    print("=" * 100)
    run_phase2_context_sweep(num_episodes=500, num_trials=50)

    notes_path = results_dir / "NOTES.md"
    notes_path.write_text(
        "# Phase 3C: Statistics Fixes\n\n"
        "Applied to `src/evaluation/harness.py` (+ new `stats_utils.py`) and "
        "re-run for Phase 1 (`results/phase1/`) and Phase 2 "
        "(`results/phase2/`), both now at **num_trials=50** (up from 30):\n\n"
        "- **Holm-Bonferroni correction**: every significance/effect-size "
        "table now has `p_value_holm` and `significant_holm` columns "
        "alongside the raw `p_value`/`significant`, correcting for the "
        "family of pairwise comparisons reported in that table.\n"
        "- **Bootstrap 95% CIs (10,000 resamples)**: added on late-stage "
        "success rate and cumulative regret in Phase 1's summary table "
        "(`bootstrap_ci95_*` columns), and on late-stage rate + normalized "
        "score in Phase 2's sweep table.\n"
        "- **Cliff's delta**: added (`cliffs_delta`, `cliffs_delta_magnitude`) "
        "alongside Cohen's d in every effect-size table (Thompson-vs-method "
        "and cost-effect-size tables).\n\n"
        f"> {MC_VARIATION_NOTE}\n\n"
        "Updated files: `results/phase1/comparison_summary.csv`, "
        "`results/phase1/late_stage_significance.csv`, "
        "`results/phase1/thompson_effect_sizes.csv`, "
        "`results/phase1/cost_effect_sizes.csv`, "
        "`results/phase2/context_sweep_results.csv`.\n\n"
        "Phase 3A (`results/phase3a/`) and Phase 3B (`results/phase3b/`) "
        "were built with bootstrap 95% CIs from the start (see their own "
        "NOTES.txt), since they were created in the same batch of work as "
        "this statistics upgrade.\n",
        encoding="utf-8",
    )
    print(f"\n  Phase 3C notes saved to {notes_path}")
