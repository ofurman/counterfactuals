import type { ReactNode } from 'react'

export function PosterCanvas({ children }: { children: ReactNode }) {
  return (
    <main className="poster-canvas" aria-label="CEL scientific benchmark poster">
      {children}
    </main>
  )
}
