import type { ResolvedSection } from '@/data/posterData'

export function SourceNote({ section }: { section: ResolvedSection }) {
  return (
    <p className="source-note" data-source-section={section.id}>
      Source&nbsp;·&nbsp;
      {section.sourceCitations.map((citation, index) => (
        <span data-source-citation={citation} key={citation}>
          {index > 0 ? ' · ' : ''}{citation.split('#')[1] ?? citation}
        </span>
      ))}
    </p>
  )
}
