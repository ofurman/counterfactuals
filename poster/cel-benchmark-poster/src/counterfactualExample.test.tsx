import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CounterfactualExample } from './components/figures/BenchmarkFraming'
import { counterfactualExample as example, exampleIncome, examplePrediction } from './data/counterfactualExample'

describe('illustrative CE example', () => {
  it('flips the toy prediction by changing only income to the threshold', () => {
    expect(example.kind).toBe('illustrative')
    expect(example.originalIncome).toBeLessThan(example.approvalThreshold)
    expect(example.counterfactualIncome).toBe(example.approvalThreshold)
    expect(examplePrediction(example.originalIncome)).toBe('Declined')
    expect(examplePrediction(example.counterfactualIncome)).toBe('Approved')
  })

  it('keeps synthetic inputs separate from manuscript result evidence', () => {
    const { container } = render(<CounterfactualExample />)
    expect(container.querySelectorAll('[data-example-kind="illustrative"]')).toHaveLength(1)
    expect(container.querySelector('[data-result-surface], [data-finding], [data-manuscript-source]')).toBeNull()
    expect(container.textContent).toContain(example.disclaimer)
    expect(container.textContent).toContain(example.fixedFeatures)
    expect(container.querySelector('[data-example-value="original"]')?.textContent).toBe(exampleIncome(example.originalIncome))
    expect(container.querySelector('[data-example-value="counterfactual"]')?.textContent).toBe(exampleIncome(example.counterfactualIncome))
    expect(container.querySelector('[data-example-value="threshold"]')?.textContent).toBe(exampleIncome(example.approvalThreshold))
  })
})
