# Practitioner Guidelines: Using the Crossover Predictor Safely

Companion to [`crossover.md`](crossover.md) (the mathematical derivation
and calibration study) -- this document is written for someone who wants
to USE `results/tables/crossover_lookup.parquet` or
`recovery.theory.crossover.predict_crossover` to make a real
static-table-vs-online-learning decision, without re-deriving the theory
first. Read this before reading a single number out of the lookup table.

## The one-sentence version

**Trust `prediction_status == "validated"` rows. Treat everything else as
a rough, possibly-wrong-by-orders-of-magnitude directional signal, not a
number to plan a deployment around.**

## When to trust the predictor

Only when ALL of the following hold simultaneously:

1. **Taxonomy size is 7** (`N == 7`). This is the only taxonomy size the
   calibration grid ever measured -- everything else is a pure
   theoretical extrapolation of the formula's structure, tested only by
   unit-level monotonicity checks (`tests/test_crossover.py`), never
   against a real simulated outcome.
2. **The table's quality is very low** -- 0 or 1 correct rows out of 7.
   This is counterintuitive (you would expect a predictor to be MOST
   useful for a decent table, not a terrible one) but it is exactly what
   `results/figures/crossover_reliability_heatmap.png` shows: reliability
   is high across every gap at quality 0-1/7, and collapses to near-zero
   at quality >= 2/7 for every gap except 0.05.
3. **The optimal-arm gap is close to 0.05**, OR quality is 0-1/7 (gap
   matters less at very low quality -- see the heatmap; every gap column
   is green in the top two rows).
4. `crossover_lookup.parquet`'s `prediction_status` column for that
   exact (N, gap, quality) row reads `"validated"` (`confidence_level ==
   "High"`). Do not eyeball whether a row "seems close" to the validated
   region -- check the column.

In this regime: median prediction/observed ratio ~2.1x, 70% of
predictions land within 3x of the true crossover (see
`results/tables/crossover_calibration_summary.parquet`, gap=0.05 rows).
That is still NOT exact -- treat the prediction as "this many episodes,
give or take a factor of 2-3," not a precise number.

## When NOT to trust it

- **`prediction_status in {"unsupported", "extrapolated"}`.** These are
  not the same failure mode, and the difference matters for how wrong the
  number might be:
  - `"unsupported"`: this exact (gap, quality) combination WAS
    empirically tested, and the predictor was shown unreliable there
    (e.g. gap=0.20, quality=2/7: 0% within 3x on 9 real calibration
    cells). The number is not just uncertain -- it is measurably wrong,
    in a known direction (see "Failure modes" in `crossover.md`: it
    systematically UNDERpredicts for better tables, sometimes by 50x+).
  - `"extrapolated"`: this combination was never tested at all (a
    taxonomy size other than 7, or a (gap, quality) pair no calibration
    matrix happened to produce). The direction and size of the error are
    genuinely unknown -- there is no data to even describe the failure
    mode.
- **Any table with quality >= 2/7 at gap >= 0.10.** The heatmap's bottom
  three rows are red (0% within 3x) almost everywhere gap ISN'T 0.05.
  Concretely: the project's OWN hand-authored canonical table (quality
  6/7) is underpredicted by ~50x by the calibrated formula (predicted
  ~160 episodes; the true crossover, measured directly, is ~8,470).
- **Never use the UNCALIBRATED formula (`c=1.0`) for a real decision.**
  It is off by a median factor of ~111x on the calibration grid --
  useful only as a reference point for how much calibration mattered,
  never as a prediction.

## Minimum validated operating region

As of this calibration run (`results/tables/crossover_calibration_summary.parquet`,
56 cells, KL form, `c=0.00452`):

| Axis | Validated range |
|---|---|
| Taxonomy size `N` | Exactly 7 (never tested elsewhere) |
| Table quality | 0/7 - 1/7 correct rows |
| Optimal-arm gap | ~0.05 (the only gap bucket meeting the 50%-within-3x bar) |
| Classifier | Perfect (noiseless) only |
| Learner | Thompson Sampling only |
| Reward model | Bernoulli, stationary |

Any deployment question outside this box should be treated as
**directional, not quantitative** guidance at best.

## How to use the lookup table safely

1. Load `results/tables/crossover_lookup.parquet`.
2. Filter to your `(N, gap, quality)` -- if it isn't an exact row, do
   NOT interpolate between neighboring rows; the predictor's error is not
   smooth or well-behaved enough near the validated boundary to trust
   interpolation (see the scatter plot's floor-at-`t_min` cluster --
   many nearby points collapse to the same trivial answer for very
   different reasons).
3. Read `prediction_status` and `confidence_level` FIRST, before the
   `prediction` value itself.
   - `validated` / High: use the number, with the ~2-3x uncertainty
     noted above.
   - `partially_validated` / Medium: use only as a coarse "is this likely
     under a year, or likely over a decade" signal -- see the worked
     example below.
   - `unsupported` or `extrapolated` / Low: do not use the number for
     planning. Use the qualitative direction only if you also read
     "Failure modes" in `crossover.md` (the predictor's known bias is
     to UNDERpredict how long good tables take, so treat a `Low`-confidence
     "short" prediction with the most suspicion of all).
4. If your exact combination isn't `validated`, prefer running a short
   real pilot (Thompson Sampling in shadow mode against your actual
   failure distribution for a few weeks/months) over trusting
   interpolation or extrapolation from this table.

## Why extrapolation flags exist

The theory (`recovery.theory.crossover`) is a calibrated fit to ONE
environment family (`src.evaluation.matrix_family.sample_matrix`'s
gap-controlled random matrices), at ONE taxonomy size, against ONE
static table shape (`BLIND_RULE`), for ONE learner (Thompson Sampling).
Every deployment is asking the model to say something about a
combination it was never shown. A model that silently returns a number
regardless of how far outside its training data the query is invites the
single most common failure mode of "a model that looks precise, so it
must be right" -- flagging is the difference between "this number is
probably useful" and "this number was computed by the same formula, but
nothing says it's meaningful here." `is_extrapolation`/`prediction_status`/
`confidence_level` make that distinction explicit and machine-readable,
so a downstream consumer of the parquet file cannot accidentally treat
an extrapolated row the same as a validated one.

## Worked examples using realistic deployment budgets

### Example A: validated regime

A support-automation team logs 7 recurring failure types, roughly 500
failures/month total (about 71/type/month, 500 episodes/month). Their
current fixed runbook gets `1/7` types right, at an estimated gap of
`~0.05` on the rest.

```python
from recovery.theory.crossover import predict_crossover_grid
rows = predict_crossover_grid(quality_levels=[1], gaps=[0.05], K=6, N=7,
                               n_matrices=25, seed=1, form="kl", c=0.00452)
```

This lands in the VALIDATED region (`quality=1/7`, `gap=0.05`). Expect a
predicted `T*` on the order of ~100-230 episodes (see
`results/tables/crossover_calibration_summary.parquet`, gap=0.05,
quality=1 row) -- roughly **2-5 months** at this deployment's episode
rate, with the documented ~2-3x uncertainty band (so realistically
"somewhere between 1 and 12 months," not a precise date). Recommendation:
a live Thompson Sampling pilot is very likely worth setting up soon --
even the pessimistic end of the range is a short pilot.

### Example B: outside the validated regime (the Section 6 example from `crossover.md`)

The same team's runbook is actually much better than average -- `5/7`
correct, gap `~0.15`. This point is neither `validated` nor
`unsupported`: it is `extrapolated` (quality=5 was never observed in the
calibration grid at any gap). The calibrated and uncalibrated formulas
disagree by ~3,300x (~1 month vs. ~10 years -- see `crossover.md`,
Section 6). Recommendation: do NOT plan around either number. The
qualitative fact that `rho > 0` (a permanent gap exists) is trustworthy;
how many months that translates to is not knowable from this predictor
alone for a table this good. Pilot first.
