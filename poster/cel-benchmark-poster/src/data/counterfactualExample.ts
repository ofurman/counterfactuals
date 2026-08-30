import example from '../../../research/ce-example.json'

// These invented inputs explain the concept; they are not experimental evidence.
export const counterfactualExample = example

export function examplePrediction(monthlyIncome: number) {
  return monthlyIncome >= example.approvalThreshold ? 'Approved' : 'Declined'
}

export function exampleIncome(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: example.currency, maximumFractionDigits: 0,
  }).format(value)
}
