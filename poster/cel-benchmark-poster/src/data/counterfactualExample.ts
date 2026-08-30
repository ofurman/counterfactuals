import example from '../../../research/ce-example.json'

// These invented inputs explain the concept; they are not experimental evidence.
export const counterfactualExample = example
export type ExampleProfile = typeof example.original

export function examplePrediction(profile: ExampleProfile) {
  const affordable = profile.monthlyIncome - profile.monthlyDebt >= example.model.minimumIncomeAfterDebt
  const establishedCredit = profile.creditHistoryYears >= example.model.minimumCreditHistoryYears
  const boundedLoan = profile.loanAmount / (12 * profile.monthlyIncome) <= example.model.maximumLoanToAnnualIncomeRatio
  return affordable && establishedCredit && boundedLoan ? 'Approved' : 'Declined'
}

function formatFeature(value: string | number, format: string) {
  if (format === 'currency') return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: example.currency, maximumFractionDigits: 0,
  }).format(Number(value))
  return format === 'years' ? `${value} years` : String(value)
}

export const exampleFeatureRows = example.features.map((feature) => {
  const key = feature.key as keyof ExampleProfile
  return {
    ...feature,
    original: formatFeature(example.original[key], feature.format),
    counterfactual: formatFeature(example.counterfactual[key], feature.format),
    changed: example.original[key] !== example.counterfactual[key],
  }
})

export const changedFeatureCount = exampleFeatureRows.filter((feature) => feature.changed).length
