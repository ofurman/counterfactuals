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
  it('preserves the original manuscript glyphs by bypassing schema SVG optimization', () => {
    const parcel = JSON.parse(fs.readFileSync(path.join(process.cwd(), '.parcelrc'), 'utf8'))
    expect(parcel.optimizers).toEqual({ '**/manuscript-architecture*.svg': [] })
  })

  it('uses an explicit Georgia/Arial scale with an 80pt title and 28pt subheadings', () => {
    const pointsPerPixel = cssNumber('print-scale') * 0.75
    expect(cssNumber('title-size') * pointsPerPixel).toBeCloseTo(80, 6)
    expect(cssNumber('subheading-size') * pointsPerPixel).toBeCloseTo(28, 6)
    expect(css).toContain('font-family: Arial, sans-serif;')
    expect(css).not.toMatch(/Aptos|Segoe UI|font-weight: (550|750|800|900)/)
    expect(css).toMatch(/\.recourse-panel h3\s*\{[^}]*font-size: var\(--subheading-size\)/)
    expect(css).toMatch(/\.result-panel h3\s*\{[^}]*font-size: var\(--subheading-size\)/)
  })
  it('fixes the native canvas and asymmetric macro grid', () => {
    expect(cssNumber('poster-width')).toBe(visualSpec.canvas.widthPx)
    expect(cssNumber('poster-height')).toBe(visualSpec.canvas.heightPx)
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
