import type { ReactNode } from 'react'
import { posterData } from '@/data/posterData'

export function PosterCanvas({ children }: { children: ReactNode }) {
  return (
    <main className="poster-canvas" aria-label="CEL scientific benchmark poster">
      {children}
      <span hidden>
        {posterData.precedents.map((precedent) => (
          <span data-source-citation={precedent.sourceCitation} key={precedent.citationKey} />
        ))}
      </span>
    </main>
  )
}
