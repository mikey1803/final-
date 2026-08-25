// The 6 recovery strategies from backend/src/recovery/strategies.py (RecoveryStrategy enum).
export const RECOVERY_STRATEGIES = [
  { id: 'retry', label: 'Retry', description: 'Try the exact same action again' },
  { id: 'replan', label: 'Replan', description: 'Scrap the plan, start fresh' },
  { id: 'switch_tool', label: 'Switch Tool', description: 'Use a different tool for the job' },
  { id: 're_ground', label: 'Re-ground', description: 'Re-fetch context, re-anchor state' },
  { id: 'verify', label: 'Verify', description: 'Double-check the last output' },
  { id: 'clarify', label: 'Clarify', description: 'Ask a clarifying question' },
]

// A few rows lifted from the real hidden SUCCESS_MATRIX in strategies.py, used to
// drive the "belief" bar chart in the learning-loop diagram with real numbers
// rather than fabricated ones.
export const BELIEF_EXAMPLES = [
  {
    failureTypeId: 'hallucinated_output',
    winner: 'verify',
    beliefs: { retry: 0.10, replan: 0.45, switch_tool: 0.30, re_ground: 0.55, verify: 0.75, clarify: 0.50 },
  },
  {
    failureTypeId: 'context_loss',
    winner: 're_ground',
    beliefs: { retry: 0.10, replan: 0.55, switch_tool: 0.25, re_ground: 0.80, verify: 0.40, clarify: 0.50 },
  },
  {
    failureTypeId: 'ambiguous_instruction',
    winner: 'clarify',
    beliefs: { retry: 0.10, replan: 0.45, switch_tool: 0.20, re_ground: 0.35, verify: 0.30, clarify: 0.80 },
  },
]
