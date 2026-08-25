// Task 19 — Theory & Validation page data. STATIC, hardcoded, no API calls,
// no computation at render time. Every value here is copied verbatim from
// docs/theory/crossover_derivation.md, docs/theory/kappa_preregistration.md,
// and docs/theory/drift_crossover_negative.md — do not recompute or round
// anything in this file; if a source value changes, update it here by hand.

export type StatusColor = 'validated' | 'partial' | 'negative'

export interface StatusRow {
  component: string
  status: string
  statusColor: StatusColor
  justification: string
}

export const HEADER = {
  title: 'Crossover Theory Validation',
  subtitle:
    'Two extensions to the crossover predictor from earlier in this dashboard: classifier noise (19A) and environment drift (19B).',
}

export const STATUS_ROWS: StatusRow[] = [
  {
    component: 'effective_matrix_under_confusion (eq 1-2)',
    status: 'VALIDATED',
    statusColor: 'validated',
    justification: 'Exact Bayes math, proved, unit-tested',
  },
  {
    component: 'static_rate_confused (eq 3)',
    status: 'VALIDATED',
    statusColor: 'validated',
    justification: 'Exact math; median abs error 0.93pp (n=6 noise levels x 50 seeds)',
  },
  {
    component: 'ceiling_reduction (eq 6)',
    status: 'VALIDATED',
    statusColor: 'validated',
    justification: 'Exact math, proved, unit-tested',
  },
  {
    component: 'gap_shrinkage_report',
    status: 'VALIDATED',
    statusColor: 'validated',
    justification: 'Exact math, unit-tested',
  },
  {
    component: 'argmax_inversion_check',
    status: 'VALIDATED',
    statusColor: 'validated',
    justification: 'Exact math; constructed counter-example proves non-vacuous',
  },
  {
    component: 'T* non-monotonicity in accuracy',
    status: 'VALIDATED (shape finding)',
    statusColor: 'validated',
    justification: 'Proved via 2-type/2-strategy construction',
  },
  {
    component: 'ts_cumulative_regret_confused (eq 4)',
    status: 'PARTIALLY-SUPPORTED',
    statusColor: 'partial',
    justification: 'Inherits eq 5 status',
  },
  {
    component: 'crossover_horizon_confused / kappa (eq 5)',
    status: 'PARTIALLY-SUPPORTED',
    statusColor: 'partial',
    justification: 'Case 1 by point estimate; bootstrap CI does not confirm robustness',
  },
  {
    component: 'Drift-crossover unified law (19B)',
    status: 'NEGATIVE RESULT (documented)',
    statusColor: 'negative',
    justification: 'Three pre-registered attempts; no quantitative sign rule survives',
  },
]

export const CHANNELS = {
  meanDiagonalAccuracy: '0.939',
  channel1Label: 'Channel 1: ceiling reduction',
  channel1Value: '0.0120',
  channel2Label: 'Channel 2: mean gap shrinkage (across 7 buckets)',
  channel2Value: '0.0020',
  caption:
    'Ceiling reduction is ~6x larger than gap shrinkage. Most measured degradation is reward-ceiling loss from merging types, not learnability loss.',
}

export const KAPPA_BANNER =
  'Protocol committed BEFORE the 26-environment dataset existed. docs/theory/kappa_preregistration.md'

export const KAPPA_BRANCHES = {
  branchA: {
    label: 'Branch A (transfer)',
    kappa: '0.044381',
    nTrain: '18',
    heldOutError: '59.7%',
  },
  branchB: {
    label: 'Branch B (in-condition)',
    kappa: '0.042559',
    nTrain: '18',
    heldOutError: '40.5%',
  },
}

export const KAPPA_RESULT = {
  formula: 'R = E_A / E_B = 0.5970 / 0.4053 = 1.4729',
  threshold: 'Threshold: R <= 1.5 -> Case 1',
  decision: 'Decision: CASE 1, satisfied by 1.8% margin',
  bootstrap: 'Bootstrap 95% CI for R: [1.00, 9.55]  (2000 resamples, environment-level)',
  powerCheck: 'Power check: PASSED (18/18 TRAIN, 8/8 TEST cells finite; floor was 5)',
}

export const KAPPA_CALLOUT =
  "Satisfies the pre-registered criterion, but not robustly. The point estimate clears the bar; the same protocol's own bootstrap CI spans into Case-2 territory. Reported as a calibrated predictor with disclosed fragility, not a validated bound."

export interface PerCellRow {
  envId: string
  ratio: string
  signedRelError: string
}

export const PER_CELL_ROWS: PerCellRow[] = [
  { envId: 'gap0.20_idx00', ratio: '1.487', signedRelError: '+48.7%' },
  { envId: 'gap0.20_idx01', ratio: '1.527', signedRelError: '+52.7%' },
  { envId: 'gap0.20_idx06', ratio: '1.283', signedRelError: '+28.3%' },
  { envId: 'gap0.20_idx08', ratio: '5.756', signedRelError: '+475.6%' },
  { envId: 'gap0.20_idx11', ratio: '0.675', signedRelError: '-32.5%' },
  { envId: 'gap0.20_idx16', ratio: '0.333', signedRelError: '-66.7%' },
  { envId: 'gap0.20_idx17', ratio: '5.752', signedRelError: '+475.2%' },
  { envId: 'gap0.20_idx24', ratio: '1.694', signedRelError: '+69.4%' },
]

export const PER_CELL_CAPTION =
  'Highly heterogeneous. No single scalar kappa corrects this spread; no rescue attempted (pre-registration no-rescue commitment).'

export interface DriftAttemptRow {
  attempt: string
  statistic: string
  result: string
  verdict: string
}

export const DRIFT_ATTEMPTS: DriftAttemptRow[] = [
  {
    attempt: '1',
    statistic: '(s, g) survival, gap-change',
    result: 'R2 = 0.012',
    verdict: 'Insufficient',
  },
  {
    attempt: '2',
    statistic: '(d, tau) magnitude, recovery-ratio',
    result: 'R2 = 0.579-0.735',
    verdict: 'RETRACTED - conceptually circular',
  },
  {
    attempt: '3',
    statistic: '(D, R) env-level regret, reversion fraction',
    result: 'R2 = 0.676; sign acc 79.7% = majority baseline',
    verdict: 'No quantitative rule survives',
  },
]

export const DRIFT_CONFIRMED_FINDING = {
  title: 'CONFIRMED POSITIVE FINDING',
  headline: 'Reversion, not magnitude, governs the sign of adaptive_advantage under drift.',
  bullets: [
    '13/13 negative-advantage cells sit at R >= 0.68 (260 seed draws)',
    '32/32 R~0 cells are positive (640 seed draws)',
    'monotone_decay R ~ 0, 95% CI [0.0000, 0.0160], n = 236',
  ],
}

export const SCOPE_LIMITATION =
  'The empirical Tier 1 validation, as currently available, is confined to one matrix family at a single gap level (gap=0.20). Generalization across other gap structures and independently generated environment families is therefore untested.'

export const DATASET_NOTE =
  'Dataset: results/tables/horizon_sweep_full26.parquet (26 environments x 2 classifiers, 52 cells, 2184 rows across 7 horizons).'
