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

  it('declares exact A0 landscape output and removes screen controls', () => {
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
