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

export type ExampleParadigm = 'local' | 'global' | 'group-wise'

export const exampleApplicants = example.applicants.map(({ id, ...values }) => ({
  id,
  profile: { ...example.original, ...values },
}))

export function exampleTransitions(paradigm: ExampleParadigm) {
  if (paradigm === 'local') return [{ id: 'A', group: 'individual', original: example.original, counterfactual: example.counterfactual }]
  return exampleApplicants.map(({ id, profile }) => {
    const group = example.groups.find((candidate) => candidate.applicants.includes(id))
    if (!group) throw new Error(`Applicant ${id} has no example group`)
    const change = paradigm === 'global' ? example.globalChange : group.change
    return {
      id,
      group: paradigm === 'global' ? 'shared' : group.id,
      original: profile,
      counterfactual: {
        ...profile,
        monthlyIncome: profile.monthlyIncome + change.monthlyIncome,
        monthlyDebt: profile.monthlyDebt + change.monthlyDebt,
      },
    }
  })
}
