import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import visualSpec from '../../research/visual-spec.json'

const css = fs.readFileSync(path.join(process.cwd(), 'src/index.css'), 'utf8')

function cssNumber(name: string): number {
  const value = css.match(new RegExp(`--${name}:\\s*([0-9.]+)`))?.[1]
  if (!value) throw new Error(`Missing CSS number token: ${name}`)
  return Number(value)
}

function cssToken(name: string): string {
  const value = css.match(new RegExp(`--${name}:\\s*([^;]+)`))?.[1]
  if (!value) throw new Error(`Missing CSS token: ${name}`)
  return value.trim()
}

describe('print contract', () => {
  it('enlarges PWr by fifteen percent and reduces partner logos by ten percent', () => {
    expect(css).toContain('.brand-logo--pwr { width: 52.9px; height: 85.1px; }')
    expect(css).toContain('.brand-logo--genwro { width: 99px; height: 25.2px; }')
    expect(css).toContain('.brand-logo--tooploox { width: 99px; height: 26.1px; }')
  })

  it('preserves the original manuscript glyphs by bypassing schema SVG optimization', () => {
    const parcel = JSON.parse(fs.readFileSync(path.join(process.cwd(), '.parcelrc'), 'utf8'))
    expect(parcel.optimizers).toEqual({ '**/manuscript-architecture*.svg': [] })
  })

  it('uses an explicit Georgia/Arial scale with an 80pt title and 28pt subheadings', () => {
    const pointsPerPixel = cssNumber('print-scale') * 0.75
    expect(cssNumber('title-size') * pointsPerPixel).toBeCloseTo(80, 6)
    expect(cssNumber('subheading-size') * pointsPerPixel).toBeCloseTo(28, 6)
    expect(cssNumber('results-heading-size') * pointsPerPixel).toBeCloseTo(32, 6)
    expect(cssNumber('contribution-detail-size') * pointsPerPixel).toBeCloseTo(18, 6)
    expect(css).toContain('font-family: Arial, sans-serif;')
    expect(css).not.toMatch(/Aptos|Segoe UI|font-weight: (550|750|800|900)/)
    expect(css).toMatch(/\.recourse-panel h3\s*\{[^}]*font-size: var\(--subheading-size\)/)
    expect(css).toMatch(/\.result-panel h3\s*\{[^}]*font-size: var\(--subheading-size\)/)
    expect(css).toMatch(/\.contribution-item h3\s*\{[^}]*font-family: Georgia, 'Times New Roman', serif;[^}]*font-size: var\(--subheading-size\);[^}]*font-weight: 700/)
    expect(css).toMatch(/\.benchmark-grid \.problem-block > h2,\s*\.benchmark-grid \.unified-results-block > h2,\s*\.benchmark-grid \.contributions-block > h2\s*\{[^}]*font-size: var\(--results-heading-size\);[^}]*font-weight: 700/)
  })
  it('aligns result frames and contribution headings while opening up scope inventories', () => {
    expect(css).toMatch(/\.result-panels\s*\{[^}]*align-items: stretch/)
    expect(css).toMatch(/\.benchmark-grid \.result-panel\s*\{[^}]*padding: 12px/)
    expect(css).toMatch(/\.scope-tile__inventory\s*\{[^}]*gap: 10px;[^}]*line-height: 1.4/)
    expect(css).toContain('.scope-tile__inventory [data-scope-name] { white-space: nowrap; }')
    expect(css).toMatch(/\.contribution-item\s*\{[^}]*align-items: start/)
    expect(css).toMatch(/\.contribution-item\s*\{[^}]*background: transparent;[^}]*border: 0/)
    expect(css).toMatch(/\.contribution-stack\s*\{[^}]*grid-template-columns: repeat\(3, minmax\(0, 1fr\)\)/)
    expect(css).toMatch(/\.contribution-item--extend \.project-mark\s*\{[^}]*position: absolute;[^}]*top: 0;[^}]*right: 0/)
  })
  it('fixes the native canvas and asymmetric macro grid', () => {
    expect(cssNumber('poster-width')).toBe(visualSpec.canvas.widthPx)
    expect(cssNumber('poster-height')).toBe(visualSpec.canvas.heightPx)
    expect(css).toMatch(/\.benchmark-grid\s*\{[^}]*row-gap: 12px/)
    expect(css).toContain(`grid-template-columns: ${visualSpec.macroGrid.columns.join(' ')}`)
    expect(cssNumber('safe-top')).toBe(visualSpec.safeArea.topPx)
    expect(cssNumber('safe-right')).toBe(visualSpec.safeArea.rightPx)
    expect(cssNumber('safe-bottom')).toBe(visualSpec.safeArea.bottomPx)
    expect(cssNumber('safe-left')).toBe(visualSpec.safeArea.leftPx)
    expect(cssToken('paper')).toBe(visualSpec.colors.paper)
    expect(cssToken('navy')).toBe(visualSpec.colors.navy)
    expect(cssToken('teal')).toBe(visualSpec.colors.teal)
    expect(cssToken('orange')).toBe(visualSpec.colors.orange)
  })

  it('declares exact A1 portrait output and removes screen controls', () => {
    expect(visualSpec.page).toMatchObject({ format: 'A1', orientation: 'portrait', widthMm: 594, heightMm: 841 })
    expect(css).toContain('.poster-column--right { grid-template-rows: auto; grid-column: 1 / -1; }')
    expect(css).toMatch(/\.result-panels\s*\{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/)
    expect(css).toContain(`@page { size: ${visualSpec.page.widthMm}mm ${visualSpec.page.heightMm}mm; margin: ${visualSpec.page.marginMm}; }`)
    expect(css).toContain('print-color-adjust: exact')
    expect(css).toMatch(/\.screen-only, \.print-toolbar\s*{\s*display: none !important;/)

    const cssPixelsPerMillimetre = 96 / 25.4
    const expectedScale = visualSpec.page.widthMm * cssPixelsPerMillimetre / visualSpec.canvas.widthPx
    expect(cssNumber('print-scale')).toBeCloseTo(expectedScale, 9)
    const printedHeightMm = visualSpec.canvas.heightPx * cssNumber('print-scale') / cssPixelsPerMillimetre
    expect(printedHeightMm).toBeLessThanOrEqual(visualSpec.page.heightMm)
    expect(visualSpec.page.heightMm - printedHeightMm).toBeLessThan(0.2)
  })
})
