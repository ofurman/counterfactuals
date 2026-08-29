import type { ReactNode } from 'react'
import type { ResolvedSection } from '@/data/posterData'

type Props = {
  section: ResolvedSection
  className?: string
  children?: ReactNode
  showClaims?: boolean
}

export function SectionBlock({ section, className = '', children, showClaims = true }: Props) {
  return (
    <section
      className={`section-block ${className}`.trim()}
      data-section={section.id}
      aria-labelledby={`${section.id}-heading`}
    >
      <p className="section-kicker">{section.owner}</p>
      <h2 id={`${section.id}-heading`}>{section.heading}</h2>
      {section.copy.map((paragraph) => <p className="section-copy" key={paragraph}>{paragraph}</p>)}
      {showClaims && section.claims.length > 0 && (
        <ul className="claim-list">
          {section.claims.map((claim) => <li key={claim.id}>{claim.posterWording}</li>)}
        </ul>
      )}
      {children}
    </section>
  )
}
