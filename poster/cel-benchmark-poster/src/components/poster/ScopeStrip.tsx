import type { ReactNode } from 'react'
import type { ResolvedSection } from '@/data/posterData'
import { posterData } from '@/data/posterData'
import { SourceNote } from './SourceNote'

// `children` renders in the middle column of the strip, flanked by the four tiles.
export function ScopeStrip({ section, children }: { section: ResolvedSection; children?: ReactNode }) {
  return (
    <section className="scope-strip" data-section={section.id}
      aria-labelledby={section.showHeading === false ? undefined : `${section.id}-heading`}
      aria-label={section.showHeading === false ? section.heading : undefined}>
      {section.showHeading !== false && <div>
        <h2 id={`${section.id}-heading`}>{section.heading}</h2>
      </div>}
      <ul className="scope-tiles" data-result-surface="scope">
        {posterData.scopeFacts.map((fact) => (
          <li className="scope-tile" key={fact.claimId} data-claim-id={fact.claimId}>
            <svg className="scope-tile__outline" aria-hidden="true" focusable="false">
              <rect x="1" y="1" rx="9" />
            </svg>
            <h3 className="scope-tile__heading"><strong>{fact.value}</strong><span>{fact.label}</span></h3>
            <div className="scope-tile__inventory">
              {fact.inventory.map((group) => (
                <p key={group.label} data-scope-group={group.label}>
                  {group.label && <b>{group.label}</b>}
                  {group.names.map((name, index) => (
                    <span key={name}>{index > 0 && ' · '}<span data-scope-name>{name}</span></span>
                  ))}
                </p>
              ))}
            </div>
          </li>
        ))}
      </ul>
      {children}
      <SourceNote section={section} />
    </section>
  )
}
