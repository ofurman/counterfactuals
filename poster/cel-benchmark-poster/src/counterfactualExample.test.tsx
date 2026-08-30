import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CounterfactualExample } from './components/figures/BenchmarkFraming'
import { counterfactualExample as example, changedFeatureCount, exampleFeatureRows, examplePrediction, exampleApplicants, exampleTransitions, type ExampleParadigm } from './data/counterfactualExample'

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

  it('imports the three Matplotlib SVG assets with accurate accessible descriptions', () => {
    const { container } = render(<CounterfactualExample />)
    const plots = [...container.querySelectorAll<HTMLImageElement>('[data-example-plot]')]
    expect(plots.map((plot) => plot.dataset.examplePlot)).toEqual(['local', 'global', 'group-wise'])
    expect(container.querySelector('svg.recourse-plot')).toBeNull()
    for (const plot of plots) {
      const paradigm = plot.dataset.examplePlot as ExampleParadigm
      expect(plot.dataset.exampleRenderer).toBe('matplotlib')
      expect(plot.dataset.exampleAsset).toBe(`poster/plots/generated/ce-example-${paradigm}.svg`)
      expect(plot.getAttribute('src')).toContain(`ce-example-${paradigm}.svg`)
      expect(plot.getAttribute('width')).toBe('640')
      expect(plot.getAttribute('height')).toBe(paradigm === 'group-wise' ? '298' : '265')
      expect(plot.alt.includes('Legend:')).toBe(paradigm === 'group-wise')
      for (const { id, original, counterfactual } of exampleTransitions(paradigm)) {
        expect(plot.alt).toContain(`${id}: Declined at income €${original.monthlyIncome} and debt payments €${original.monthlyDebt}; Approved at income €${counterfactual.monthlyIncome} and debt payments €${counterfactual.monthlyDebt}`)
      }
      expect(plot.alt).toContain('employment stays full-time')
    }
  })
})
