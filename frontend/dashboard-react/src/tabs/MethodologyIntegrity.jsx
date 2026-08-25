import { Search, Wrench, ScrollText } from 'lucide-react'
import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'

const BADGE = 'rounded-full border border-lav-amber/40 bg-lav-amber/10 px-3 py-1 font-sans text-xs font-semibold tracking-wide text-lav-amber'

const CASES = [
  {
    icon: Search,
    badge: 'Investigated',
    title: 'The 0.741 anomaly',
    body: 'One published number (a recovery success rate of 0.741) sat above the highest that specific rule could mathematically ever score (0.7357), given a perfect classifier. Three independent checks, a fresh run of the original code, a closed-form calculation, and a completely separate from-scratch reimplementation, landed in the same tight band (0.715 to 0.724), consistently below the ceiling and consistently not reproducing 0.741. The likely explanation: the original number came from one specific historical run whose exact conditions could not be recovered. No code was changed to force a match; instead a runtime check was added so a result like this fails loudly in the future instead of silently reaching a table again.',
  },
  {
    icon: Wrench,
    badge: 'Fixed',
    title: 'A real, non-deterministic seeding bug',
    body: "One experiment generated part of its random seed from Python's built-in hash() function, which randomizes string hashes per process by default. Running the identical code twice produced three different numbers across three separate runs. Traced and fixed (swapped the non-deterministic hash for a deterministic checksum), then every affected result was re-run. Consequence: an earlier document had built a multi-paragraph mechanistic explanation for why two independent methods disagreed by a large margin on one data point. That disagreement turned out to be this bug. After the fix the gap shrank to well within normal noise, and the incorrect explanation was corrected in place rather than deleted.",
  },
  {
    icon: ScrollText,
    badge: 'Corrected',
    title: "A claim that didn't survive stricter testing",
    body: 'An early finding claimed one failure type (malformed output) was the hardest across every AI model tested. Applying proper multiple-comparison statistical correction, that claim held up rigorously in only 1 of 3 models. A second model narrowly missed the bar after correction. A third model got every other failure type right, which makes its statistical test mathematically undefined rather than merely "not significant." The original, stronger claim was corrected in the project documentation once this was found.',
  },
]

export function MethodologyIntegrity() {
  return (
    <div>
      <PageHeader
        title="Methodology & Integrity"
        subtitle="Tasks 4/4A and 13: three real cases from this project's internal audits, a suspicious number traced rather than hidden, a bug caught and fixed, and a claim corrected after stricter testing."
      />

      <StaggerGroup className="flex flex-col gap-6">
        {CASES.map((c) => (
          <StaggerItem key={c.title}>
            <Card hover={false}>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2.5">
                  <c.icon size={17} className="shrink-0 text-lav-amber" />
                  <h3 className="font-sans text-base font-semibold text-lav-text">{c.title}</h3>
                </div>
                <span className={BADGE}>{c.badge}</span>
              </div>
              <p className="font-sans text-sm leading-relaxed text-lav-text">{c.body}</p>
            </Card>
          </StaggerItem>
        ))}

        <StaggerItem>
          <Card hover={false} className="border-lav-amber/25 bg-lav-amber/[0.03]">
            <p className="font-sans text-sm leading-relaxed text-lav-text">
              A related policy decision underlies all of this dashboard's statistics: this project
              deliberately does not compute p-values for its simulation-based comparisons.
              Re-running a deterministic simulator with more random seeds makes a p-value shrink
              toward zero for literally any nonzero difference, no matter how practically
              meaningless, so a p-value there would be a fake signal, not real evidence. Every
              simulation-based comparison in this dashboard instead reports an effect size plus a
              bootstrap confidence interval. p-values are computed for exactly one family of
              tests, the Blind-Design Study, because each of its samples is a real, independent
              call to a real external model, a genuine population to test against, and those get
              proper Holm-Bonferroni correction for testing 7 failure types at once.
            </p>
          </Card>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
