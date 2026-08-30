import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CounterfactualExample } from './components/figures/BenchmarkFraming'
import { counterfactualExample as example, changedFeatureCount, exampleFeatureRows, examplePrediction } from './data/counterfactualExample'

describe('illustrative CE example', () => {
  it('changes only two actionable features in a six-feature applicant profile', () => {
    expect(example.kind).toBe('illustrative')
    expect(exampleFeatureRows).toHaveLength(6)
    expect(exampleFeatureRows.map((row) => row.key).sort()).toEqual(Object.keys(example.original).sort())
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
    expect(container.textContent).toContain('Only 2 of 6 features change.')
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
})
