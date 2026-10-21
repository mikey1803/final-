import { FAILURE_TYPES } from './failureTypes'
import { RECOVERY_STRATEGIES } from './recoveryStrategies'

// Real matrices, not invented for the UI. Rows/columns follow the same
// TYPE_NAMES/STRATEGY_NAMES order as backend/recovery/envs/canonical.py.
// Canonical is the exact SUCCESS_MATRIX from backend/src/recovery/strategies.py
// (== canonical.py's MATRIX). The three "structured_*" alternatives are
// transcribed byte-for-byte from backend/recovery/envs/matrix_perturbation.py
// (Task 9's hand-constructed alternative-matrix robustness test) — two of
// these three are the ones that broke the "Fixed Rule wins" ranking outright.
const TYPE_IDS = FAILURE_TYPES.map((f) => f.id)
const STRATEGY_IDS = RECOVERY_STRATEGIES.map((s) => s.id)

function toMatrixObject(rows) {
  const obj = {}
  TYPE_IDS.forEach((typeId, i) => {
    obj[typeId] = {}
    STRATEGY_IDS.forEach((stratId, j) => {
      obj[typeId][stratId] = rows[i][j]
    })
  })
  return obj
}

export const MATRIX_PRESETS = [
  {
    id: 'canonical',
    label: 'Canonical (the paper’s matrix)',
    description: 'The real SUCCESS_MATRIX every other tab in this dashboard uses. Fixed Rule wins here.',
    rows: toMatrixObject([
      [0.70, 0.50, 0.60, 0.15, 0.30, 0.10],
      [0.65, 0.55, 0.75, 0.20, 0.35, 0.25],
      [0.15, 0.60, 0.75, 0.45, 0.40, 0.55],
      [0.10, 0.45, 0.30, 0.55, 0.75, 0.50],
      [0.40, 0.50, 0.60, 0.35, 0.75, 0.30],
      [0.10, 0.55, 0.25, 0.80, 0.40, 0.50],
      [0.10, 0.45, 0.20, 0.35, 0.30, 0.80],
    ]),
  },
  {
    id: 'uniform-difficulty',
    label: 'Uniform difficulty',
    description: 'Every failure type has an identical achievable ceiling (Task 9, scheme 3a) — one of the two matrices that broke the ranking.',
    rows: toMatrixObject([
      [0.30, 0.45, 0.55, 0.65, 0.75, 0.20],
      [0.20, 0.30, 0.45, 0.55, 0.65, 0.75],
      [0.75, 0.20, 0.30, 0.45, 0.55, 0.65],
      [0.65, 0.75, 0.20, 0.30, 0.45, 0.55],
      [0.55, 0.65, 0.75, 0.20, 0.30, 0.45],
      [0.45, 0.55, 0.65, 0.75, 0.20, 0.30],
      [0.30, 0.45, 0.55, 0.65, 0.75, 0.20],
    ]),
  },
  {
    id: 'single-dominant',
    label: 'Single dominant strategy',
    description: '"Verify" is optimal for every failure type (Task 9, scheme 3b) — the other matrix that broke the ranking.',
    rows: toMatrixObject([
      [0.40, 0.30, 0.55, 0.20, 0.85, 0.35],
      [0.45, 0.35, 0.50, 0.25, 0.85, 0.40],
      [0.20, 0.55, 0.60, 0.30, 0.85, 0.45],
      [0.15, 0.40, 0.35, 0.50, 0.85, 0.45],
      [0.35, 0.45, 0.50, 0.30, 0.85, 0.25],
      [0.10, 0.50, 0.25, 0.60, 0.85, 0.40],
      [0.15, 0.40, 0.20, 0.35, 0.85, 0.65],
    ]),
  },
  {
    id: 'malformed-not-hardest',
    label: 'Malformed output not hardest',
    description: 'Two rows hand-edited so ambiguous_instruction, not malformed_output, is the hardest type (Task 9, scheme 3c) — ranking held here.',
    rows: toMatrixObject([
      [0.70, 0.50, 0.60, 0.15, 0.30, 0.10],
      [0.65, 0.55, 0.75, 0.20, 0.35, 0.25],
      [0.15, 0.60, 0.75, 0.45, 0.40, 0.55],
      [0.10, 0.45, 0.30, 0.55, 0.75, 0.50],
      [0.50, 0.60, 0.70, 0.45, 0.95, 0.40],
      [0.10, 0.55, 0.25, 0.80, 0.40, 0.50],
      [0.05, 0.15, 0.10, 0.10, 0.05, 0.20],
    ]),
  },
]
