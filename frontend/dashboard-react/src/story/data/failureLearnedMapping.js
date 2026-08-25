// The full converged (failure_type -> best strategy) mapping, derived from the
// real SUCCESS_MATRIX in backend/src/recovery/strategies.py (argmax per row).
// `revealEpisode` is an illustrative ordering (not a per-type logged metric) —
// larger success gaps between the best and second-best strategy are modeled as
// learned sooner, spread across the real, reported fact that Thompson Sampling
// converges on the correct strategy for all 7 types by episode ~300.
export const FAILURE_LEARNED_MAPPING = [
  { failureTypeId: 'ambiguous_instruction', strategy: 'clarify', revealEpisode: 90 },
  { failureTypeId: 'context_loss', strategy: 're_ground', revealEpisode: 140 },
  { failureTypeId: 'hallucinated_output', strategy: 'verify', revealEpisode: 190 },
  { failureTypeId: 'malformed_output', strategy: 'verify', revealEpisode: 230 },
  { failureTypeId: 'wrong_tool_selected', strategy: 'switch_tool', revealEpisode: 260 },
  { failureTypeId: 'tool_error', strategy: 'switch_tool', revealEpisode: 285 },
  { failureTypeId: 'tool_timeout', strategy: 'retry', revealEpisode: 300 },
]

export const STATIONARY_ANNOTATIONS = [
  { maxEpisode: 50, text: 'Exploring: trying different strategies for each failure type.' },
  { maxEpisode: 150, text: 'Starting to learn patterns. Some strategies are pulling ahead.' },
  { maxEpisode: 300, text: 'Mostly converged. The top strategy per failure type is now clear.' },
  { maxEpisode: 500, text: 'Stable. Knows what to do for each failure type.' },
]

export const DISRUPTED_ANNOTATIONS = [
  { maxEpisode: 50, text: 'Exploring: trying different strategies for each failure type.' },
  { maxEpisode: 250, text: 'Converged and stable, for now.' },
  { maxEpisode: 280, text: 'Disruption hits at episode 250: the environment just changed underneath it.' },
  { maxEpisode: 400, text: 'Re-adapting: relearning which strategies work now.' },
  { maxEpisode: 500, text: 'Recovered. Back to a strong success rate despite the shift.' },
]
