import type { ResolvedSection } from '@/data/posterData'
import { posterData } from '@/data/posterData'
import { SourceNote } from './SourceNote'

export function ScopeStrip({ section }: { section: ResolvedSection }) {
  return (
    <section className="scope-strip" data-section={section.id} aria-labelledby={`${section.id}-heading`}>
      <div>
        <h2 id={`${section.id}-heading`}>{section.heading}</h2>
      </div>
      <ul data-result-surface="scope">
        {posterData.scopeFacts.map((fact) => (
          <li key={`${fact.claimId}-${fact.label}`} data-claim-id={fact.claimId}>
            <strong>{fact.value}</strong>
            <span>{fact.label}</span>
          </li>
        ))}
      </ul>
      <SourceNote section={section} />
    </section>
  )
}
