import type { ReactNode } from 'react'
import type { ResolvedSection } from '@/data/posterData'
import { SourceNote } from './SourceNote'

type Props = {
  section: ResolvedSection
  className?: string
  children?: ReactNode
  showClaims?: boolean
  as?: 'section' | 'article'
}

export function SectionBlock({ section, className = '', children, showClaims = true, as: Element = 'section' }: Props) {
  const Heading = Element === 'article' ? 'h3' : 'h2'
  return (
    <Element
      className={`section-block ${className}`.trim()}
      data-section={section.id}
      aria-labelledby={`${section.id}-heading`}
    >
      <Heading id={`${section.id}-heading`}>{section.heading}</Heading>
      {section.copy.map((paragraph) => <p className="section-copy" key={paragraph}>{paragraph}</p>)}
      {showClaims && section.claims.length > 0 && (
        <ul className="claim-list">
          {section.claims.map((claim) => <li key={claim.id}>{claim.posterWording}</li>)}
        </ul>
      )}
      {children}
      <SourceNote section={section} />
    </Element>
  )
}
