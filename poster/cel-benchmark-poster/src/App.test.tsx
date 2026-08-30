import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'
import { posterData } from './data/posterData'

describe('poster scaffold', () => {
  it('uses semantic poster landmarks and screen-only controls', () => {
    render(<App />)
    expect(screen.getByRole('main', { name: /CEL scientific benchmark poster/i })).toBeInTheDocument()
    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.queryByRole('contentinfo')).not.toBeInTheDocument()
    expect(screen.getByRole('toolbar', { name: /Poster controls/i })).toBeInTheDocument()
    expect(screen.getAllByRole('figure')).toHaveLength(6)
    expect(screen.getAllByRole('region')).toHaveLength(5)
    const { widthPx, heightPx } = posterData.visualSpec.canvas
    expect(screen.getByText(new RegExp(`${widthPx} × ${heightPx}`))).toBeInTheDocument()
  })

  it('renders every section owned by the frozen content brief', () => {
    const { container } = render(<App />)
    const sectionIds = Array.from(container.querySelectorAll('[data-section]'))
      .map((element) => element.getAttribute('data-section'))
    expect(sectionIds).toEqual(expect.arrayContaining([
      'header', 'problem', 'scope', 'protocol', 'local-tradeoff', 'group-tradeoff',
      'applicability', 'results', 'guidance-limitations', 'regression-tradeoff',
    ]))
    expect(sectionIds).toHaveLength(10)
  })

  it('omits the bottom reproduction strip while retaining the contribution QR', () => {
    const { container } = render(<App />)
    const poster = within(container)
    expect(container.querySelector('.poster-footer, [data-section="reproducibility"]')).toBeNull()
    expect(poster.queryByText('Reproduce and extend')).not.toBeInTheDocument()
    expect(poster.queryByText('uv add ce-library')).not.toBeInTheDocument()
    expect(poster.queryByText('Documentation')).not.toBeInTheDocument()
    expect(poster.queryByRole('navigation', { name: 'Project links' })).not.toBeInTheDocument()
    expect(poster.getAllByText('Code & project')).toHaveLength(1)
    expect(container.querySelector('[data-section="guidance-limitations"] [data-qr-destination]')).not.toBeNull()
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
      ['left', ['problem']],
      ['center', ['protocol', 'scope', 'guidance-limitations']],
      ['right', ['results']],
    ] as const) {
      expect(Array.from(container.querySelectorAll(`.poster-column--${column} > [data-section]`))
        .map((element) => element.getAttribute('data-section'))).toEqual(ids)
    }
    expect(container.querySelector('[data-finding="regression"]')).toBeInTheDocument()
    expect(Array.from(container.querySelectorAll('[data-section="results"] article[data-section]')).map((element) => element.getAttribute('data-section'))).toEqual(['applicability', 'local-tradeoff', 'group-tradeoff', 'regression-tradeoff'])
    expect(container.querySelectorAll('.result-panel > .result-panel__outline[aria-hidden="true"]')).toHaveLength(4)
    expect(container.querySelectorAll('.scope-tile')).toHaveLength(4)
    expect(Array.from(container.querySelectorAll('.scope-tile__heading')).map((element) => element.textContent)).toEqual(['18Datasets', '14Methods', '2Backbones / Task', '9Classification Metrics'])
    expect(container.querySelector('.scope-strip [data-claim-id="scope.folds"]')).toBeNull()
    expect(container.querySelector('.scope-strip')?.textContent).not.toMatch(/5folds|3paradigms/)
  })

  it('omits center-column headings but keeps accessible section names', () => {
    const { container } = render(<App />)
    expect(container.querySelectorAll('[data-section="protocol"] h2, [data-section="scope"] h2')).toHaveLength(0)
    for (const name of ['One evaluation framework', 'Benchmark scope']) {
      expect(within(container).queryByText(name)).not.toBeInTheDocument()
      expect(within(container).getByRole('region', { name })).toBeInTheDocument()
    }
  })
})
