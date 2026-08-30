import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CounterfactualExample } from './components/figures/BenchmarkFraming'
import { counterfactualExample as example, changedFeatureCount, exampleFeatureRows, examplePrediction, exampleApplicants, exampleTransitions, examplePlotPoint, examplePlotBounds, type ExampleParadigm } from './data/counterfactualExample'

describe('illustrative CE example', () => {
  it('shows three features and changes only income and debt payments', () => {
    expect(example.kind).toBe('illustrative')
    expect(exampleFeatureRows).toHaveLength(3)
    expect(exampleFeatureRows.map((row) => row.key)).toEqual(['monthlyIncome', 'monthlyDebt', 'employment'])
    expect(changedFeatureCount).toBe(2)
    expect(exampleFeatureRows.filter((row) => row.changed).map((row) => row.key)).toEqual(['monthlyIncome', 'monthlyDebt'])
    expect(exampleFeatureRows.filter((row) => !row.actionable).every((row) => !row.changed)).toBe(true)
    expect(examplePrediction(example.original)).toBe('Declined')
    expect(examplePrediction(example.counterfactual)).toBe('Approved')
    expect(examplePrediction({ ...example.original, monthlyIncome: example.counterfactual.monthlyIncome })).toBe('Declined')
    expect(examplePrediction({ ...example.original, monthlyDebt: example.counterfactual.monthlyDebt })).toBe('Declined')
  })

  it('keeps synthetic inputs separate from manuscript result evidence', () => {
    const { container } = render(<CounterfactualExample />)
    expect(container.querySelectorAll('[data-example-kind="illustrative"]')).toHaveLength(1)
    expect(container.querySelector('[data-result-surface], [data-finding], [data-manuscript-source]')).toBeNull()
    expect(container.textContent).toContain(example.label)
    expect(container.textContent).not.toMatch(/toy|source|not benchmark|threshold/i)
    expect(example.provenance).toMatch(/Invented applicant profiles/)
    for (const feature of exampleFeatureRows) {
      const row = container.querySelector(`[data-example-feature="${feature.key}"]`)
      expect(row?.getAttribute('data-changed')).toBe(String(feature.changed))
      expect(row?.querySelector('[data-example-value="original"]')?.textContent).toBe(feature.original)
      expect(row?.querySelector('[data-example-value="counterfactual"]')?.textContent).toBe(feature.counterfactual)
      expect(row?.querySelector('.example-value--changed') !== null).toBe(feature.changed)
    }
  })

  it('starts every applicant declined and approves every endpoint without changing hidden features', () => {
    expect(exampleApplicants.map(({ id }) => id)).toEqual(['A', 'B', 'C', 'D'])
    for (const { profile } of exampleApplicants) expect(examplePrediction(profile)).toBe('Declined')
    for (const paradigm of ['local', 'global', 'group-wise'] as ExampleParadigm[]) {
      for (const { original, counterfactual } of exampleTransitions(paradigm)) {
        expect(examplePrediction(original)).toBe('Declined')
        expect(examplePrediction(counterfactual)).toBe('Approved')
        expect(counterfactual.monthlyIncome - counterfactual.monthlyDebt).toBeGreaterThan(example.model.minimumIncomeAfterDebt)
        for (const key of ['employment', 'age', 'creditHistoryYears', 'loanAmount'] as const) {
          expect(counterfactual[key]).toBe(original[key])
          expect(original[key]).toBe(example.original[key])
        }
      }
    }
  })

  it('reuses the same population and exactly one action globally or one action per group', () => {
    const global = exampleTransitions('global')
    const grouped = exampleTransitions('group-wise')
    expect(global.map(({ id, original }) => ({ id, original }))).toEqual(grouped.map(({ id, original }) => ({ id, original })))
    const delta = ({ original, counterfactual }: typeof global[number]) => ({ monthlyIncome: counterfactual.monthlyIncome - original.monthlyIncome, monthlyDebt: counterfactual.monthlyDebt - original.monthlyDebt })
    expect(global.map(delta)).toEqual(Array(4).fill({ monthlyIncome: 1300, monthlyDebt: 0 }))
    expect(grouped.map(delta)).toEqual([
      { monthlyIncome: 0, monthlyDebt: -1000 }, { monthlyIncome: 0, monthlyDebt: -1000 },
      { monthlyIncome: 1300, monthlyDebt: 0 }, { monthlyIncome: 1300, monthlyDebt: 0 },
    ])
  })

  it('renders three plots at the same scale with nine forward arrows and declined originals', () => {
    const { container } = render(<CounterfactualExample />)
    const plots = [...container.querySelectorAll('[data-example-plot]')]
    expect(plots.map((plot) => plot.getAttribute('data-example-plot'))).toEqual(['local', 'global', 'group-wise'])
    expect(new Set(plots.map((plot) => plot.getAttribute('viewBox'))).size).toBe(1)
    expect(new Set(plots.map((plot) => plot.querySelector('.recourse-boundary')?.getAttribute('d'))).size).toBe(1)
    expect(container.querySelectorAll('[data-example-transition]')).toHaveLength(9)
    for (const plot of plots) {
      const paradigm = plot.getAttribute('data-example-plot') as ExampleParadigm
      const transitions = exampleTransitions(paradigm)
      for (const transition of transitions) {
        const mark = plot.querySelector(`[data-example-transition="${transition.id}"]`)!
        expect(mark.getAttribute('data-from')).toBe('Declined')
        expect(mark.getAttribute('data-to')).toBe('Approved')
        const start = examplePlotPoint(transition.original)
        const end = examplePlotPoint(transition.counterfactual)
        const circle = mark.querySelector('.recourse-point--original')!
        expect(Number(circle.getAttribute('cx'))).toBe(start.x)
        expect(Number(circle.getAttribute('cy'))).toBe(start.y)
        const arrow = mark.querySelector('line')!
        expect(arrow.getAttribute('marker-end')).toBe(`url(#${paradigm}-arrow-${transition.group})`)
        expect(Number(arrow.getAttribute('x1'))).toBe(start.x)
        expect(Number(arrow.getAttribute('y1'))).toBe(start.y)
        const tip = { x: Number(arrow.getAttribute('x2')), y: Number(arrow.getAttribute('y2')) }
        expect((tip.x - start.x) * (end.x - start.x) + (tip.y - start.y) * (end.y - start.y)).toBeGreaterThan(0)
        const tipIncome = example.plot.minimumIncome + (tip.x - examplePlotBounds.left) / (examplePlotBounds.right - examplePlotBounds.left) * (example.plot.maximumIncome - example.plot.minimumIncome)
        const tipDebt = (examplePlotBounds.bottom - tip.y) / (examplePlotBounds.bottom - examplePlotBounds.top) * example.plot.maximumDebt
        expect(tipIncome - tipDebt).toBeGreaterThan(example.model.minimumIncomeAfterDebt)
        for (const point of [start, end]) {
          expect(point.x).toBeGreaterThanOrEqual(examplePlotBounds.left)
          expect(point.x).toBeLessThanOrEqual(examplePlotBounds.right)
          expect(point.y).toBeGreaterThanOrEqual(examplePlotBounds.top)
          expect(point.y).toBeLessThanOrEqual(examplePlotBounds.bottom)
        }
      }
    }
  })
})
