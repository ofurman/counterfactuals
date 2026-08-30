import { posterData, type ResolvedSection } from '@/data/posterData'
import { BrandStrip } from './BrandStrip'

export function PosterHeader({ section }: { section: ResolvedSection }) {
  const { identity } = posterData

  return (
    <header className="poster-header" data-section={section.id}>
      <BrandStrip side="left" />
      <div className="header-copy">
        <p className="eyebrow">{identity.venue}</p>
        <h1>{section.heading}</h1>
        <p className="authors">{identity.authors.map((author) => author.name).join(' · ')}</p>
        <p className="affiliation">{identity.affiliation}</p>
      </div>
      <BrandStrip side="right" />
    </header>
  )
}
