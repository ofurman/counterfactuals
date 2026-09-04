import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'
import { posterData } from './data/posterData'

const liveClaimIds = new Set(posterData.claims.map((claim) => claim.id))

describe('rendered evidence provenance', () => {
  it('uses poster typography derivatives without losing dataset identity or source graphics', () => {
    const { container } = render(<App />)
    expect(container.querySelectorAll('.result-manuscript-figure figcaption')).toHaveLength(0)
    const figures = Array.from(container.querySelectorAll('.result-manuscript-figure'))
    expect(figures).toHaveLength(4)
    for (const figure of figures) expect(figure.getAttribute('data-dataset')).toBe(figure.getAttribute('data-finding') === 'regression' ? 'Concrete' : 'Adult Census')
    for (const figure of figures) {
      expect(figure.getAttribute('data-typography-asset')).toBe(figure.getAttribute('data-finding'))
      expect(figure.querySelector('img')?.getAttribute('src')).toContain(`manuscript-${figure.getAttribute('data-finding')}.svg`)
    }
  })

  it('renders a poster-native architecture schematic without a repeated caption', () => {
    const { container } = render(<App />)
    const figure = container.querySelector('.architecture-figure')
    expect(figure?.querySelector('figcaption')).toBeNull()
    expect(figure?.textContent).not.toContain('Shared evaluation protocol')
    expect(figure?.getAttribute('data-typography-asset')).toBe('architecture')
    expect(figure?.getAttribute('data-manuscript-presentation')).toBe('native')
    // Recreated as vector shapes, not a shrunk manuscript raster.
    expect(figure?.querySelector('img')).toBeNull()
    const svg = figure?.querySelector('.architecture-image-window svg')
    expect(svg).not.toBeNull()
    expect(svg?.querySelector('[data-claim-id="scope.methods"]')).not.toBeNull()
  })

  it('binds every rendered claim marker to the live generated ledger', () => {
    const { container } = render(<App />)
    const markers = Array.from(container.querySelectorAll<HTMLElement>('[data-claim-id]'))
    expect(markers.length).toBeGreaterThan(12)
    for (const marker of markers) {
      expect(liveClaimIds.has(marker.dataset.claimId ?? ''), marker.outerHTML).toBe(true)
    }
    expect(container.querySelectorAll('[data-finding]')).toHaveLength(posterData.resultVisuals.length)
    const manuscriptSources = Array.from(container.querySelectorAll<HTMLElement>('[data-manuscript-source]'))
      .map((element) => element.dataset.manuscriptSource)
    expect(manuscriptSources).toEqual(expect.arrayContaining([
      'manuscript/figures/teaser.pdf',
      'manuscript/figures/metrics_boxplot_local.png',
      'manuscript/figures/metrics_boxplot_global.png',
      'manuscript/figures/metrics_boxplot_group_wise.png',
      'manuscript/figures/regression_metrics_boxplot.png',
    ]))
    expect(new Set(manuscriptSources).size).toBe(5)
    for (const image of container.querySelectorAll<HTMLImageElement>('.manuscript-figure img')) {
      expect(image.alt.length).toBeGreaterThan(30)
    }
  })

  it('does not render a visible result numeral outside a claim-owned element', () => {
    const { container } = render(<App />)
    for (const surface of container.querySelectorAll('[data-result-surface]')) {
      const walker = document.createTreeWalker(surface, NodeFilter.SHOW_TEXT)
      let node = walker.nextNode()
      while (node) {
        if (/\d/.test(node.textContent ?? '')) {
          const owner = node.parentElement?.closest('[data-claim-id]')
          expect(owner, `Unowned numeric text: ${node.textContent}`).not.toBeNull()
        }
        node = walker.nextNode()
      }
    }
  })

  it('binds project links, the QR, and non-printing source metadata to frozen inputs', () => {
    const { container } = render(<App />)
    const allowedLinks = new Set(Object.values(posterData.identity.links))
    for (const anchor of container.querySelectorAll<HTMLAnchorElement>('a[href]')) {
      expect(allowedLinks.has(anchor.href)).toBe(true)
    }
    const qr = container.querySelector<HTMLElement>('[data-qr-destination]')
    expect(container.querySelectorAll('[data-qr-destination]')).toHaveLength(1)
    expect(container.querySelector('.poster-header [data-qr-destination]')).toBeNull()
    expect(qr?.closest('article')?.dataset.claimId).toBe('contribution.library')
    expect(qr?.closest('[data-section]')?.getAttribute('data-section')).toBe('guidance-limitations')
    expect(qr?.closest('.poster-column--bottom')).not.toBeNull()
    expect(qr?.dataset.qrDestination).toBe(posterData.identity.qr.url)
    expect(qr?.getAttribute('href')).toBe(posterData.identity.links.repository)

    for (const note of container.querySelectorAll<HTMLElement>('[data-source-section]')) {
      expect(note).not.toBeVisible()
      const section = posterData.sections.find((item) => item.id === note.dataset.sourceSection)
      expect(section).toBeDefined()
      for (const citation of note.querySelectorAll<HTMLElement>('[data-source-citation]')) {
        expect(section?.sourceCitations).toContain(citation.dataset.sourceCitation)
      }
    }
    const renderedCitations = new Set(
      Array.from(container.querySelectorAll<HTMLElement>('[data-source-citation]'))
        .map((element) => element.dataset.sourceCitation),
    )
    for (const precedent of posterData.precedents) {
      expect(renderedCitations.has(precedent.sourceCitation)).toBe(true)
    }
    expect(container.querySelector('.source-note, .example-disclaimer, .example-rule')).toBeNull()
  })
})
