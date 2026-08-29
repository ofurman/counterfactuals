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
    expect(screen.getAllByRole('figure')).toHaveLength(6)
    expect(screen.getAllByRole('region')).toHaveLength(8)
    const { widthPx, heightPx } = posterData.visualSpec.canvas
    expect(screen.getByText(new RegExp(`${widthPx} × ${heightPx}`))).toBeInTheDocument()
  })

  it('renders every section owned by the frozen content brief', () => {
    const { container } = render(<App />)
    const sectionIds = Array.from(container.querySelectorAll('[data-section]'))
      .map((element) => element.getAttribute('data-section'))
    expect(sectionIds).toEqual(expect.arrayContaining([
      'header', 'problem', 'scope', 'protocol', 'local-tradeoff', 'group-tradeoff',
      'regression-tradeoff', 'applicability', 'guidance-limitations', 'reproducibility',
    ]))
  })
})
