import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'
import { posterData } from './data/posterData'

describe('poster scaffold', () => {
  it('uses semantic poster landmarks and screen-only controls', () => {
    render(<App />)
    expect(screen.getByRole('main', { name: /CEL scientific benchmark poster/i })).toBeInTheDocument()
    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.getByRole('contentinfo')).toBeInTheDocument()
    expect(screen.getByRole('toolbar', { name: /Poster controls/i })).toBeInTheDocument()
    expect(screen.getAllByRole('figure')).toHaveLength(5)
    expect(screen.getAllByRole('region')).toHaveLength(7)
    const { widthPx, heightPx } = posterData.visualSpec.canvas
    expect(screen.getByText(new RegExp(`${widthPx} × ${heightPx}`))).toBeInTheDocument()
  })

  it('renders every section owned by the frozen content brief', () => {
    const { container } = render(<App />)
    const sectionIds = Array.from(container.querySelectorAll('[data-section]'))
      .map((element) => element.getAttribute('data-section'))
    expect(sectionIds).toEqual(expect.arrayContaining([
      'header', 'problem', 'scope', 'protocol', 'local-tradeoff', 'group-tradeoff',
      'applicability', 'guidance-limitations', 'reproducibility',
    ]))
    expect(sectionIds).toHaveLength(9)
    expect(sectionIds).not.toContain('regression-tradeoff')
  })

  it('uses the exact manuscript title and keeps the poster copy concise', () => {
    const { container } = render(<App />)
    expect(container.querySelector('h1')?.textContent).toBe(posterData.identity.title)
    expect(container.querySelector('.eyebrow')).toBeNull()
    expect(container.querySelector('.poster-header')?.textContent).not.toContain(posterData.identity.venue)
    expect(container.querySelector('.poster-thesis')).toBeNull()
    expect(container.querySelector('.poster-header')?.textContent).not.toContain('One protocol. Multiple CE paradigms. Measurable trade-offs.')
    const headingsAndCopy = posterData.sections.flatMap((section) => [section.heading, ...section.copy]).join(' ')
    expect(headingsAndCopy.trim().split(/\s+/).length).toBeLessThanOrEqual(110)
    expect(container.querySelectorAll('.section-block .section-kicker')).toHaveLength(0)
  })

  it('places the concept and results in the requested columns', () => {
    const { container } = render(<App />)
    for (const [column, ids] of [
      ['left', ['problem', 'applicability']],
      ['center', ['protocol', 'local-tradeoff']],
      ['right', ['guidance-limitations', 'group-tradeoff']],
    ] as const) {
      expect(Array.from(container.querySelectorAll(`.poster-column--${column} > [data-section]`))
        .map((element) => element.getAttribute('data-section'))).toEqual(ids)
    }
    expect(container.querySelector('[data-finding="regression"]')).toBeNull()
  })
})
