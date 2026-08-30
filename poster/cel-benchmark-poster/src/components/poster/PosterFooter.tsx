import { posterData, type ResolvedSection } from '@/data/posterData'

export function PosterFooter({ section }: { section: ResolvedSection }) {
  return (
    <footer className="poster-footer" data-section={section.id}>
      <div>
        <strong>{section.heading}</strong>
        <code>{section.copy[0]}</code>
      </div>
      <span hidden>
        {posterData.precedents.map((precedent) => (
          <span data-source-citation={precedent.sourceCitation} key={precedent.citationKey} />
        ))}
      </span>
      <nav aria-label="Project links">
        <a href={posterData.identity.links.repository}>{posterData.links.repository.label}</a>
        <a href={posterData.identity.links.documentation}>{posterData.links.documentation.label}</a>
      </nav>
    </footer>
  )
}
