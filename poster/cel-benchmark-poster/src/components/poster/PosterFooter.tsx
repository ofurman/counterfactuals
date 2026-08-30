import { posterData, type ResolvedSection } from '@/data/posterData'

export function PosterFooter({ section }: { section: ResolvedSection }) {
  return (
    <footer className="poster-footer" data-section={section.id}>
      <div>
        <strong>{section.heading}</strong>
        <code>{section.copy[0]}</code>
      </div>
      <p className="precedent-citations">
        Benchmark context&nbsp;·&nbsp;
        {posterData.precedents.map((precedent, index) => (
          <span data-source-citation={precedent.sourceCitation} key={precedent.citationKey}>
            {index > 0 ? ' · ' : ''}{precedent.label} [{precedent.citationKey}]
          </span>
        ))}
      </p>
      <nav aria-label="Project links">
        <a href={posterData.identity.links.repository}>{posterData.links.repository.label}</a>
        <a href={posterData.identity.links.documentation}>{posterData.links.documentation.label}</a>
      </nav>
    </footer>
  )
}
