import { StoryNav } from './StoryNav'
import { HeroSection } from './sections/HeroSection'
import { ProblemSection } from './sections/ProblemSection'
import { SolutionSection } from './sections/SolutionSection'
import { DemoSection } from './sections/DemoSection'
import { ResultsSection } from './sections/ResultsSection'
import { useScrollGradient } from '../hooks/useScrollGradient'

export function StoryPage({ onOpenDashboard }) {
  const gradientRef = useScrollGradient()

  return (
    <div className="min-h-screen w-full">
      <div ref={gradientRef} className="scroll-gradient-bg" aria-hidden="true" />
      <StoryNav onOpenDashboard={onOpenDashboard} />
      <div className="mx-auto max-w-6xl px-6">
        <HeroSection />
        <ProblemSection />
        <SolutionSection />
        <DemoSection />
        <ResultsSection onOpenDashboard={onOpenDashboard} />
      </div>
    </div>
  )
}
