import { ClaimWording } from '@/components/poster'
import { ArrowDown } from 'lucide-react'
import { counterfactualExample as example, exampleIncome, examplePrediction } from '@/data/counterfactualExample'

export function CounterfactualExample() {
  return (
    <figure className="ce-example" data-example-kind={example.kind} aria-label="Illustrative counterfactual explanation for a toy loan model">
      <figcaption>{example.label}</figcaption>
      <div className="example-case example-case--original" data-example-case="original">
        <div>
          <span className="example-case-label">Original input</span>
          <strong data-example-value="original">{exampleIncome(example.originalIncome)}</strong>
          <span className="example-feature">{example.feature}</span>
        </div>
        <span className="example-prediction">{examplePrediction(example.originalIncome)}</span>
      </div>
      <div className="example-change"><ArrowDown size={25} aria-hidden="true" /><span>Change income only</span></div>
      <div className="example-case example-case--counterfactual" data-example-case="counterfactual">
        <div>
          <span className="example-case-label">Counterfactual input</span>
          <strong data-example-value="counterfactual">{exampleIncome(example.counterfactualIncome)}</strong>
          <span className="example-feature">{example.feature}</span>
        </div>
        <span className="example-prediction">{examplePrediction(example.counterfactualIncome)}</span>
      </div>
      <p className="example-fixed">{example.fixedFeatures}</p>
      <p className="example-rule">Toy rule: approve at ≥ <span data-example-value="threshold">{exampleIncome(example.approvalThreshold)}</span>/month.</p>
      <p className="example-disclaimer">{example.disclaimer}</p>
    </figure>
  )
}

const contributions = [
  { eyebrow: 'CONTROL', claimId: 'contribution.protocol' },
  { eyebrow: 'COVER', claimId: 'contribution.benchmark' },
  { eyebrow: 'EXTEND', claimId: 'contribution.library' },
]

export function ContributionStack() {
  return (
    <div className="contribution-stack">
      {contributions.map((contribution) => (
        <article className="contribution-item" data-claim-id={contribution.claimId} key={contribution.claimId}>
          <span>{contribution.eyebrow}</span>
          <p><ClaimWording claimId={contribution.claimId} /></p>
        </article>
      ))}
    </div>
  )
}
