import { posterData, type ResolvedSection } from '@/data/posterData'

export function PosterHeader({ section }: { section: ResolvedSection }) {
  const { identity } = posterData

  return (
    <header className="poster-header" data-section={section.id}>
      <div className="header-copy">
        <p className="eyebrow">{identity.venue} · {identity.title}</p>
        <h1>{section.heading}</h1>
        <p className="poster-thesis">{section.copy[0]}</p>
        <p className="authors">{identity.authors.map((author) => author.name).join(' · ')}</p>
        <p className="affiliation">{identity.affiliation}</p>
      </div>
      <a className="project-mark" href={identity.links.repository} aria-label="Open the CEL project repository">
        <span className="project-mark__monogram" aria-hidden="true">CEL</span>
        <span>{identity.qr.label}</span>
      </a>
    </header>
  )
}
