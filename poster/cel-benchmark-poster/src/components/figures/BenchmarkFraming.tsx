import { ClaimWording } from '@/components/poster'
import { ProjectQr } from '@/components/poster/ProjectQr'
import { counterfactualExample as example, exampleFeatureRows, changedFeatureCount, examplePrediction } from '@/data/counterfactualExample'

export function CounterfactualExample() {
  return (
    <figure className="ce-example" data-example-kind={example.kind} aria-label="Loan application example comparing original and counterfactual features">
      <figcaption>{example.label}</figcaption>
      <table className="example-profile" aria-label="Original and counterfactual applicant profile">
        <thead><tr><th scope="col">Feature</th><th scope="col">Original</th><th scope="col">Counterfactual</th></tr></thead>
        <tbody>
          {exampleFeatureRows.map((feature) => (
            <tr key={feature.key} data-example-feature={feature.key} data-changed={feature.changed}>
              <th scope="row">{feature.label}</th>
              <td data-example-value="original">{feature.original}</td>
              <td data-example-value="counterfactual" className={feature.changed ? 'example-value--changed' : ''}>{feature.counterfactual}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <th scope="row">Prediction</th>
            <td><span className="example-prediction example-prediction--declined">{examplePrediction(example.original)}</span></td>
            <td><span className="example-prediction example-prediction--approved">{examplePrediction(example.counterfactual)}</span></td>
          </tr>
        </tfoot>
      </table>
      <p className="example-summary"><span className="example-change-key" aria-hidden="true" />Only <strong>{changedFeatureCount} of {exampleFeatureRows.length}</strong> features change.</p>
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
        <article className={`contribution-item${contribution.claimId === 'contribution.library' ? ' contribution-item--extend' : ''}`} data-claim-id={contribution.claimId} key={contribution.claimId}>
          <span>{contribution.eyebrow}</span>
          <p><ClaimWording claimId={contribution.claimId} /></p>
          {contribution.claimId === 'contribution.library' && <ProjectQr />}
        </article>
      ))}
    </div>
  )
}
