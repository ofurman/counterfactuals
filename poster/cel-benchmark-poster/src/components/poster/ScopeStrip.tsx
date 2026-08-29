import type { ResolvedSection } from '@/data/posterData'

export function ScopeStrip({ section }: { section: ResolvedSection }) {
  return (
    <section className="scope-strip" data-section={section.id} aria-labelledby={`${section.id}-heading`}>
      <div>
        <p className="section-kicker">Benchmark scope</p>
        <h2 id={`${section.id}-heading`}>{section.heading}</h2>
      </div>
      <ul>
        {section.claims.map((claim) => <li key={claim.id}>{claim.posterWording}</li>)}
      </ul>
    </section>
  )
}
