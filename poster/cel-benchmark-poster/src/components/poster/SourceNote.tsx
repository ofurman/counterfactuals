import type { ResolvedSection } from '@/data/posterData'

export function SourceNote({ section }: { section: ResolvedSection }) {
  return (
    <span hidden data-source-section={section.id}>
      {section.sourceCitations.map((citation) => (
        <span data-source-citation={citation} key={citation} />
      ))}
    </span>
  )
}
