import {
  Clock,
  XCircle,
  Shuffle,
  Sparkles,
  FileWarning,
  BrainCircuit,
  HelpCircle,
} from 'lucide-react'

// The 7-type taxonomy from backend/src/failures/taxonomy.py (FailureType enum),
// re-described in plain English for a non-technical reader.
export const FAILURE_TYPES = [
  {
    id: 'tool_timeout',
    label: 'Timeout',
    icon: Clock,
    color: '#F2A65A',
    description: 'The agent calls a tool and just... never hears back. It hangs, waiting for a response that never comes.',
  },
  {
    id: 'tool_error',
    label: 'Tool Error',
    icon: XCircle,
    color: '#D88C9A',
    description: 'The tool runs, but comes back with an error instead of a result. The request itself was rejected.',
  },
  {
    id: 'wrong_tool_selected',
    label: 'Wrong Tool',
    icon: Shuffle,
    color: '#9B8FE8',
    description: 'The agent reaches for the wrong tool entirely, like asking a calculator to check the weather.',
  },
  {
    id: 'hallucinated_output',
    label: 'Hallucination',
    icon: Sparkles,
    color: '#4FB8A8',
    description: 'The agent confidently states something that simply isn’t true, with no indication it’s unsure.',
  },
  {
    id: 'malformed_output',
    label: 'Malformed Output',
    icon: FileWarning,
    color: '#B0A7F0',
    description: 'The agent’s response breaks the expected format, so nothing downstream can actually parse it.',
  },
  {
    id: 'context_loss',
    label: 'Context Loss',
    icon: BrainCircuit,
    color: '#6A5DC6',
    description: 'The agent forgets something important it was told earlier in the conversation.',
  },
  {
    id: 'ambiguous_instruction',
    label: 'Ambiguous Instruction',
    icon: HelpCircle,
    color: '#7FB88F',
    description: 'The instruction could reasonably mean two different things, and the agent guesses wrong.',
  },
]
