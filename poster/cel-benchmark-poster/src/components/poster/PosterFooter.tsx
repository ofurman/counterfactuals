import { posterData, type ResolvedSection } from '@/data/posterData'

export function PosterFooter({ section }: { section: ResolvedSection }) {
  return (
    <footer className="poster-footer" data-section={section.id}>
      <div>
        <strong>{section.heading}</strong>
        <span>{section.copy.join(' · ')}</span>
      </div>
      <nav aria-label="Project links">
        <a href={posterData.identity.links.repository}>{posterData.links.repository.label}</a>
        <a href={posterData.identity.links.documentation}>{posterData.links.documentation.label}</a>
      </nav>
    </footer>
  )
}
