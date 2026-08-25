// Per-method chart colors, tuned for contrast against the dark void
// background (#0E0B16 / #191228) — same per-method hue identity as the
// original light theme (Thompson/UCB1/e-greedy stay lavender-family,
// Fixed Rule stays orange, LinUCB stays teal), lightened/desaturated
// where the old value would have been too dark to read.
export const METHOD_COLORS = {
  'Adaptive (Thompson)': '#A78BFA',
  'Adaptive (UCB1)': '#C4B5FD',
  'Adaptive (e-greedy)': '#DDD6FE',
  'Adaptive (Decaying e)': '#8B5CF6',
  'Contextual Cost-Aware': '#7C3AED',
  'Naive Retry': '#6B6584',
  'Fixed Rule': '#F5A962',
  'Fixed Rule (blind)': '#F5A962',
  UniformRandom: '#4A4560',
  Random: '#4A4560',
  'One-Shot LLM': '#8B84A6',
  LinUCB: '#5EEAD4',
  Oracle: '#EDE9F7',
  'Stock TS': '#A78BFA',
  'Discounted TS (gamma=0.99)': '#F0A6A6',
}

export function colorFor(method, fallback = '#8B84A6') {
  return METHOD_COLORS[method] ?? fallback
}
