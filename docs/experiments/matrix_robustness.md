# SUCCESS_MATRIX Sensitivity and Robustness Analysis (Task 12)

This document explains Task 12's deliverable
(`backend/recovery/envs/matrix_perturbation.py`,
`backend/experiments/matrix_robustness.py`) -- an empirical test of
whether the paper's headline stationary-world claim ("a well-designed
blind static rule beats every tested adaptive method") describes the
*framework*, or merely describes the one hand-authored matrix every
other phase conditions on.

## SUCCESS_MATRIX provenance (stated explicitly, as required)

`recovery.envs.canonical.MATRIX` mirrors `backend/src/recovery/strategies.py`'s
`SUCCESS_MATRIX`, whose own comment block states it is **"the hidden
ground-truth success-rate matrix"** the adaptive selector must learn --
i.e. it was **authored by hand** for this project. It is **not** derived
from a literature survey and **not** measured from real agent
telemetry. This is a limitation the paper must state plainly (it already
does, see `docs/PAPER_PREP.md` §6, "Single blind-rule instance"); Task
12 exists specifically to characterize how much of the paper's
conclusion depends on this one authored instance.

## What "headline ranking" means here (a deliberate scope reduction)

The Phase 1/4 comparison spans 11 methods, several requiring live LLM
calls (`One-Shot LLM`) or otherwise outside this sweep's ~2,200-world
scale. Task 12 operationalizes the paper's headline claim as exactly 5
methods, chosen because they are the ones Phases 4-7's "static beats
adaptive in a stationary world" narrative is actually built on, and
because three of them have exact closed-form rates with zero simulation
cost:

- **Oracle**, **BlindRule** (canonical `BLIND_RULE`), **UniformRandom**
  -- computed exactly via `recovery.theory.analytic.{oracle_rate,
  policy_rate, random_rate}`. These are properties of a fixed policy
  under a known matrix; simulating them would only add Monte Carlo noise
  with zero benefit.
- **Thompson**, **LinUCB** -- the two adaptive methods with no closed
  form in this codebase, simulated for real via
  `recovery.core.episode.run_episodes` (the same kernel every other
  phase uses), 500 episodes/seed, late-stage rate = final 20% (100
  episodes), averaged over `n_seeds=30` independent seeds per world
  (raised from 10 by Task 13's seed-count audit, `docs/statistics_registry.md`).

**headline_preserved** := `BlindRule_rate > Thompson_rate AND
BlindRule_rate > LinUCB_rate`. **full_order_preserved** is the stricter
condition that all 5 methods land in the exact canonical order `Oracle >
BlindRule > LinUCB > Thompson > UniformRandom`.

## Two different, non-interchangeable senses of "hardest failure type"

Task 11's finding that `malformed_output` is hardest is about **blind
designers' accuracy** reading the plain-English failure descriptions (a
property of human/LLM behavior, not recomputable from a matrix alone --
testing a new matrix would require entirely new blind participants, who
by protocol must never see the matrix). Task 12 needs a **different,
purely matrix-derived** notion to be computable for ~2,200 perturbed
matrices with no new elicitation: the **ceiling-hardest type**,
`argmin_t max_k P[t, k]` -- the failure type whose best-case strategy is
weakest. `recovery.envs.matrix_perturbation.hardest_type_by_ceiling`
implements this and is tested (`test_matrix_perturbation.py`) to confirm
a non-obvious fact: **under this definition, the canonical `MATRIX`'s
hardest type is `tool_timeout` (ceiling 0.70), not `malformed_output`**
(whose ceiling is 0.75, via `verify`). The two findings answer different
questions and neither substitutes for the other -- see "Does the
malformed_output finding survive Scheme 3(c)?" below for how this
distinction plays out.

## The three perturbation schemes

**Scheme 1 -- probability noise** (`gaussian_perturb`): every matrix
entry independently perturbed by `N(0, sigma)`, clipped to `[0, 1]`.
Swept over `sigma in {0.05, 0.10, 0.15, 0.20, 0.30}`, 200 sampled
matrices per level (1,000 worlds total).

**Scheme 2 -- entry permutation** (`permute_row_entries`): a controlled
fraction of each row's entries permuted among themselves. Two modes,
reported separately per the task's requirement:
- `rank_preserving` -- the row's current argmax position is excluded
  from the eligible-to-permute set, so the optimal strategy for that row
  can never change, by construction.
- `rank_breaking` -- the argmax position is eligible, so the optimal
  strategy for a row *can* change.

Swept over `fraction in {0.10, 0.25, 0.50, 1.00}` x 2 modes, 150 sampled
matrices per (fraction, mode) cell (1,200 worlds total). At `K=6`
strategies, `fraction=0.10` selects `round(0.10*6)=1` position -- fewer
than the 2 needed to permute anything, so it is a documented, expected
no-op (visible below as `fraction=0.10`'s near-identical rates to the
canonical row for both modes).

**Scheme 3 -- structured alternatives** (three hand-authored matrices, 1
world each):
- `uniform_difficulty`: a fixed 6-value profile cyclically rotated
  across all 7 rows, giving every row an identical ceiling (no failure
  type is structurally hardest).
- `single_dominant_strategy`: strategy index 4 (`verify`) fixed at 0.85
  and optimal for every row.
- `malformed_not_hardest`: the canonical matrix with only two rows
  hand-edited -- `malformed_output`'s ceiling raised to 0.95,
  `ambiguous_instruction`'s lowered to 0.20 -- so `ambiguous_instruction`
  becomes the new ceiling-hardest type and `malformed_output` explicitly
  is not. All 5 other rows are byte-identical to `MATRIX`.

## Method: real Monte Carlo, real analytic layer, no refitting

`evaluate_world` (`experiments/matrix_robustness.py`) is the single
function computing every field for one (possibly perturbed) matrix. It
reuses, unmodified: the simulation kernel (`run_episodes`), the analytic
layer (`recovery.theory.analytic`, `recovery.theory.crossover`), Task
10's already-calibrated crossover constant `c` (`load_calibrated_c()`,
never refit), the metrics layer (`recovery.metrics.bootstrap.bootstrap_ci`,
used for every CI -- never a p-value, per `docs/statistics.md`), and the
guards layer (`recovery.guards.invariants.run_all_guards`, called after
every simulated rate to catch anomalies automatically; zero
`InvariantViolation`s were raised across the full run).

**Deliberate deviation from `RunStore`**: this script calls
`run_episodes` directly in a `joblib.Parallel(backend="loky")` loop
rather than going through `recovery.core.parallel.RunStore`/`run_grid`.
Justification: this experiment's output is one aggregated summary table
over ~2,200 worlds, not a resumable large-artifact grid; per-episode-log
persistence for the ~13 million simulated episodes in the full run would
be pure I/O overhead with no reuse benefit within a single sweep
execution. `experiments/stationary_comparison.py` already establishes
this same precedent (`run_episodes` called directly, not through
`RunStore`) for the same reason.

**Deterministic RNG discipline**: every world's matrix and every
simulation's seed derive from a fixed `BASE_SEED=120_026` via an
explicit offset formula (`build_world_specs`/`build_noise_floor_specs`)
-- no wall-clock, PID, or execution-order dependence anywhere, confirmed
by `test_matrix_robustness.py`'s determinism and no-duplicate-seed tests.

## A necessary addition: noise-floor calibration

Smoke-testing at `n_seeds=2` first showed `headline_preserved=False` on
the *unperturbed* canonical matrix -- contradicting the paper's own
established Phase 1 finding. Cross-checking against
`results/tables/table_v.parquet` (Task-1-era, 50 seeds/method) explained
why: per-seed late-stage-rate standard deviation is ~0.04-0.06 for
Thompson/LinUCB, while BlindRule's margin over them is only ~0.04 --
less than one standard deviation, so a handful of seeds can spuriously
flip the comparison from noise alone, with no bug and no
`InvariantViolation` involved.

Two fixes were applied. First, `n_seeds` was raised from an initially
planned 6 to 10 for the first full run. **Task 13's statistical audit
(`docs/statistics_registry.md`) subsequently required every CI in the
project to rest on >=30 seeds, so the full sweep was re-run a second time
at `n_seeds=30`** — the noise floor dropped to essentially zero at that
seed count (below), directly confirming the seed-count diagnosis. Second,
and more importantly, a **noise-floor calibration** was added and is not
optional: it replicates the *unperturbed* canonical matrix 200 times with
disjoint, non-overlapping seed blocks and measures how often
`headline_preserved` spuriously reads `False` at *zero* perturbation,
purely from simulation noise. Every perturbation-sweep result below must
be read relative to this baseline, not against a naive "100% at zero
perturbation" expectation.

**Measured noise floor (n=200 replicates, n_seeds=30, current): fraction
preserving headline ranking = 1.000 (95% bootstrap CI [1.000, 1.000]).**
At `n_seeds=10` (the original run) this was 0.975 (95% CI [0.950,
0.995]) — i.e. ~2.5% of simulation-only replicates spuriously showed
`BlindRule` losing to Thompson or LinUCB purely from Monte Carlo noise.
Tripling the seed count to 30 eliminated that spurious-breakdown rate
entirely across all 200 replicates tested — direct, clean confirmation
that the original 2.5% was exactly what it was diagnosed as (simulation
noise, not a real effect), and that every ranking-preservation number
below now reflects the perturbation itself, not residual seed-count noise.

## Real results (full run: 2,204 worlds, n_seeds=30, 500 episodes/seed, 3,669.8s wall time — re-run by Task 13, up from n_seeds=10)

### Ranking preservation

| scheme | magnitude | n | fraction headline preserved | 95% CI | fraction full order preserved |
|---|---|---|---|---|---|
| canonical | canonical | 1 | 1.000 | -- | 1.000 |
| gaussian | sigma=0.05 | 200 | 0.985 | [0.965, 1.000] | 0.985 |
| gaussian | sigma=0.10 | 200 | 0.755 | [0.695, 0.815] | 0.755 |
| gaussian | sigma=0.15 | 200 | 0.395 | [0.325, 0.460] | 0.395 |
| gaussian | sigma=0.20 | 200 | 0.205 | [0.150, 0.265] | 0.200 |
| gaussian | sigma=0.30 | 200 | 0.030 | [0.010, 0.055] | 0.025 |
| permute | fraction=0.10, rank_preserving | 150 | 1.000 | [1.000, 1.000] | 1.000 |
| permute | fraction=0.10, rank_breaking | 150 | 1.000 | [1.000, 1.000] | 1.000 |
| permute | fraction=0.25, rank_preserving | 150 | 0.907 | [0.860, 0.953] | 0.907 |
| permute | fraction=0.25, rank_breaking | 150 | 0.467 | [0.387, 0.547] | 0.467 |
| permute | fraction=0.50, rank_preserving | 150 | 0.847 | [0.787, 0.900] | 0.847 |
| permute | fraction=0.50, rank_breaking | 150 | 0.180 | [0.120, 0.247] | 0.180 |
| permute | fraction=1.00, rank_preserving | 150 | 0.773 | [0.707, 0.840] | 0.773 |
| permute | fraction=1.00, rank_breaking | 150 | 0.000 | [0.000, 0.000] | 0.000 |
| structured | uniform_difficulty | 1 | 0.000 | -- | 0.000 |
| structured | single_dominant_strategy | 1 | 0.000 | -- | 0.000 |
| structured | malformed_not_hardest | 1 | 1.000 | -- | 1.000 |

Full per-world data: `results/tables/matrix_robustness_raw.parquet`;
this summary: `matrix_robustness_ranking.parquet`. Note `fraction=0.10`
(both modes) now reads EXACTLY 1.000 with a zero-width CI, matching the
noise floor exactly — clean confirmation that this magnitude is a de
facto no-op at K=6 (documented above), not a robustness finding of its
own.

### Breakdown thresholds (first tested magnitude where fraction preserved < 0.5)

- **Scheme 1 (Gaussian noise): sigma=0.15.** At sigma=0.10 the ranking
  still holds 75.5% of the time; by sigma=0.15 it has fallen to 39.5%.
  Unchanged from the `n_seeds=10` run (sigma=0.15 was already the
  breakdown point there too) — the seed-count increase refined the
  numbers without changing this conclusion.
- **Scheme 2, rank_preserving: never (`None`)** across all tested
  fractions up to 1.0 -- consistent with the mode's construction, which
  by design never lets a row's optimal strategy change; the residual
  degradation (100% to 77%) comes entirely from Thompson/LinUCB's
  simulated rates shifting as non-argmax entries move, not from
  BlindRule's own optimality being challenged.
- **Scheme 2, rank_breaking: fraction=0.25.** Once the argmax value
  itself becomes eligible to move, preservation collapses far faster
  than Scheme 1 in relative terms: 100% at fraction=0.10 down to 46.7% by
  fraction=0.25, and to exactly 0% at fraction=1.00 (a full within-row
  shuffle essentially always changes at least one row's optimal
  strategy, sometimes in `BlindRule`'s favor and sometimes against it,
  and at fraction=1.0 it happens to always break the specific canonical
  ordering). Unchanged breakdown point from the `n_seeds=10` run.

### T* (crossover budget) distribution

| scheme | magnitude | n | n infinite | fraction infinite | finite mean | finite median | 95% CI (finite) |
|---|---|---|---|---|---|---|---|
| canonical | canonical | 1 | 0 | 0.000 | 160.3 | 160.3 | -- |
| gaussian | sigma=0.05 | 200 | 2 | 0.010 | 45,058 | 489.7 | [4,576, 122,357] |
| gaussian | sigma=0.10 | 200 | 12 | 0.060 | 330,307 | 1,406.5 | [72,424, 675,749] |
| gaussian | sigma=0.15 | 200 | 9 | 0.045 | 871,000 | 502.0 | [39,251, 2,425,019] |
| gaussian | sigma=0.20 | 200 | 1 | 0.005 | 1,329,888 | 199.6 | [69,538, 3,309,219] |
| gaussian | sigma=0.30 | 200 | 2 | 0.010 | 10,774 | 10.0 | [3,411, 21,162] |
| permute | fraction=0.10, rank_preserving | 150 | 0 | 0.000 | 160.3 | 160.3 | [160.3, 160.3] |
| permute | fraction=0.10, rank_breaking | 150 | 0 | 0.000 | 160.3 | 160.3 | [160.3, 160.3] |
| permute | fraction=0.25, rank_preserving | 150 | 0 | 0.000 | 131.0 | 160.3 | [121.5, 139.8] |
| permute | fraction=0.25, rank_breaking | 150 | 1 | 0.007 | 62.1 | 10.0 | [51.0, 73.2] |
| permute | fraction=0.50, rank_preserving | 150 | 0 | 0.000 | 110.5 | 160.3 | [99.6, 121.2] |
| permute | fraction=0.50, rank_breaking | 150 | 0 | 0.000 | 24.3 | 10.0 | [18.1, 31.4] |
| permute | fraction=1.00, rank_preserving | 150 | 0 | 0.000 | 63.6 | 10.0 | [53.4, 74.4] |
| permute | fraction=1.00, rank_breaking | 150 | 0 | 0.000 | 10.0 | 10.0 | [10.0, 10.0] |
| structured | uniform_difficulty | 1 | 0 | 0.000 | 10.0 | 10.0 | -- |
| structured | single_dominant_strategy | 1 | 0 | 0.000 | 10.0 | 10.0 | -- |
| structured | malformed_not_hardest | 1 | 0 | 0.000 | 326.2 | 326.2 | -- |

`T*` widens by orders of magnitude and grows extremely right-skewed as
Gaussian noise increases (mean >> median at every sigma level except
0.30) -- consistent with `predict_crossover`'s theory: as the matrix
gets noisier, `BlindRule`'s effective quality gap over a bandit shrinks
toward the point where a small number of perturbed worlds sit very close
to (but not at) the point where the crossover diverges, producing a few
very large finite draws that dominate the mean without moving the
median much. `fraction_infinite` never exceeds ~6% in any cell -- a
perfectly-scoring `BLIND_RULE` (T*=infinite) is rare but not absent
under perturbation. Full distribution: `results/figures/robustness_tstar_distribution.png/.pdf`.

### Ceiling-hardest-type distribution

| scheme | magnitude | n | tool_timeout | malformed_output | tool_error | hallucinated_output | wrong_tool_selected | context_loss | ambiguous_instruction |
|---|---|---|---|---|---|---|---|---|---|
| canonical | canonical | 1 | 1.000 | -- | -- | -- | -- | -- | -- |
| gaussian | sigma=0.05 | 200 | 0.475 | 0.135 | 0.130 | 0.120 | 0.105 | 0.025 | 0.010 |
| gaussian | sigma=0.10 | 200 | 0.265 | 0.175 | 0.090 | 0.180 | 0.140 | 0.050 | 0.100 |
| gaussian | sigma=0.15 | 200 | 0.180 | 0.115 | 0.130 | 0.200 | 0.140 | 0.085 | 0.150 |
| gaussian | sigma=0.20 | 200 | 0.180 | 0.160 | 0.100 | 0.165 | 0.110 | 0.095 | 0.190 |
| gaussian | sigma=0.30 | 200 | 0.155 | 0.140 | 0.120 | 0.145 | 0.110 | 0.120 | 0.210 |
| permute (all fractions/modes) | -- | 150 each | 1.000 | -- | -- | -- | -- | -- | -- |
| structured | uniform_difficulty | 1 | 1.000 | -- | -- | -- | -- | -- | -- |
| structured | single_dominant_strategy | 1 | 1.000 | -- | -- | -- | -- | -- | -- |
| structured | malformed_not_hardest | 1 | -- | -- | -- | -- | -- | -- | 1.000 |

Two findings worth stating precisely:

1. **Row permutation never changes the ceiling-hardest type, at any
   fraction or mode (always `tool_timeout`, 100%).** This is a
   structural fact, not a robustness result: permuting entries *within*
   a row cannot change that row's maximum value, only which strategy
   achieves it -- so `hardest_type_by_ceiling` is mathematically
   invariant under Scheme 2 by construction. It is reported for
   completeness, not as evidence of robustness.
2. **The ceiling-hardest identity is itself fragile under Gaussian
   noise**, and considerably more so than the headline ranking:
   `tool_timeout`'s share drops from 100% (unperturbed) to 47.5% already
   at sigma=0.05 (the *smallest* tested noise level), and keeps falling
   to 15.5% by sigma=0.30, with `ambiguous_instruction` and
   `hallucinated_output` becoming comparably frequent. **The specific
   claim "tool_timeout is the ceiling-hardest type" is a low-confidence,
   noise-sensitive artifact of the exact authored values, not a robust
   structural property of the canonical matrix** -- a materially
   different (more fragile) conclusion than the headline
   BlindRule-beats-bandits ranking, and the report is honest about that
   difference rather than glossing over it.

### Scheme 3: structured matrices

| matrix | BlindRule rate | Thompson rate | LinUCB rate | headline preserved | ceiling-hardest type | full ranking |
|---|---|---|---|---|---|---|
| uniform_difficulty | 0.379 | 0.641 | 0.676 | **False** | tool_timeout | Oracle > LinUCB > Thompson > UniformRandom > BlindRule |
| single_dominant_strategy | 0.629 | 0.796 | 0.832 | **False** | tool_timeout | Oracle > LinUCB > Thompson > BlindRule > UniformRandom |
| malformed_not_hardest | 0.679 | 0.621 | 0.651 | **True** | ambiguous_instruction | Oracle > BlindRule > LinUCB > Thompson > UniformRandom |

(BlindRule/Oracle/UniformRandom rates are exact analytic values, unchanged
by the seed-count bump; Thompson/LinUCB rates above are the `n_seeds=30`
re-run — refined slightly from the original `n_seeds=10` run, same
qualitative conclusions for all three matrices.)

**(a) uniform_difficulty and (b) single_dominant_strategy both break the
headline ranking outright** -- in both, the canonical `BLIND_RULE`
(which was hand-designed against the *canonical* matrix's specific
structure) is no longer competitive: by construction, when every row is
equally hard, or when one strategy is best everywhere, there is nothing
structurally advantageous left for a blind rule with the canonical
matrix's specific per-type mapping to exploit, and both adaptive methods
overtake it. This is an honest negative result under two of the three
structured alternatives, not something to minimize.

**(c) malformed_not_hardest preserves the full canonical ranking**
(`Oracle > BlindRule > LinUCB > Thompson > UniformRandom`, exactly).
See next section for what this does and does not establish.

## Does the malformed_output finding survive Scheme 3(c)?

This question has two different answers depending on which
"malformed_output finding" is meant, and both must be stated -- neither
can substitute for the other:

- **Task 11's finding** (blind LLM *designers* score lowest, 28.6-42.9%
  raw accuracy, on `malformed_output` of all three tested model
  families -- formally significant after Holm correction for 1 of the 3
  families, per Task 13's statistical audit, `docs/statistics_registry.md`)
  is a property of how designers read plain-English descriptions. It
  **cannot be re-tested here**: re-testing it on a new matrix would
  require entirely new blind elicitation sessions (human or LLM), since
  participants must never see the matrix by protocol. Task 12 does not
  attempt this and the report does not claim otherwise.
- **The ceiling-hardest-type finding** (this document's matrix-only,
  `argmin_t max_k P[t,k]` sense) already established, before any
  perturbation, that the canonical matrix's ceiling-hardest type is
  `tool_timeout`, **not** `malformed_output` -- so there was no
  ceiling-sense "malformed_output is hardest" claim to begin with, and
  nothing to survive or fail under Scheme 3(c). `malformed_not_hardest`
  was constructed specifically to test the ceiling metric's sensitivity
  to authored values (by explicitly raising `malformed_output`'s ceiling
  and lowering `ambiguous_instruction`'s) and, separately, to check
  whether the headline BlindRule-vs-adaptive ranking depends on which
  type is ceiling-hardest. It does not: the ranking is preserved exactly
  under this structured alternative.

**Bottom line, stated plainly**: the paper should describe Task 11's
`malformed_output` finding and this document's `tool_timeout`
ceiling-hardest finding as two separate, non-contradictory facts about
two different constructs -- never as the same claim measured two ways.

## Honest summary of the robustness result

The headline ranking is **not unconditionally robust** -- it degrades
smoothly and predictably as perturbation grows, and Gaussian noise
breaks it faster (below 50% by sigma=0.15) than a naive reading of "0%
noise-floor error at zero perturbation" would suggest. It is, however,
**robust under small, realistic perturbations and under a
permutation mode that cannot change which strategy is truly optimal**:
sigma=0.05 (98.5%) sits close to the measured 100% noise floor (at
`n_seeds=30`), and fraction=0.10 (100% exactly, a documented no-op at
K=6) matches it precisely -- small perturbations are close to
indistinguishable from pure simulation noise at this seed count.
Rank-preserving permutation stays above 77% even at fraction=1.0 (full
within-row shuffle, argmax excluded) -- direct evidence that most of the
ranking's persistence comes from `BlindRule`'s per-row optimal *choices*
being correct, not from the specific non-optimal probability values. Two
of three structured alternative matrices (uniform-difficulty,
single-dominant-strategy) break the ranking outright: this is the
sharpest, most honest finding of this study -- **the paper's headline
claim describes the canonical matrix's specific structure (heterogeneous
per-type optimal strategies), not an unconditional property of every
possible success-probability matrix.** The malformed-not-hardest
structured alternative, and rank-breaking permutation up to fraction=0.10,
both preserve it.

## Reproducibility

```bash
# From backend/, with the project venv active
python experiments/matrix_robustness.py --smoke     # ~seconds, correctness check only
python experiments/matrix_robustness.py              # full run: 2,204 worlds, n_seeds=30, ~3,670s (already executed, see above)

# Tests
pytest tests/test_matrix_perturbation.py tests/test_matrix_robustness.py -v
```

Outputs (all under `backend/results/`):
`tables/matrix_robustness_{raw,ranking,tstar,hardest_type,noise_floor}.parquet`,
`figures/robustness_ranking_preservation.{png,pdf}`,
`figures/robustness_tstar_distribution.{png,pdf}`.
