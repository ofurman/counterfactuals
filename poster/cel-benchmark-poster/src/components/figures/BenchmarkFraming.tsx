import { ClaimWording } from '@/components/poster'
import { ProjectQr } from '@/components/poster/ProjectQr'
import { counterfactualExample as example, exampleFeatureRows, examplePrediction } from '@/data/counterfactualExample'
import { RecoursePlot } from './RecoursePlot'

export function CounterfactualExample() {
  return (
    <figure className="ce-example" data-example-kind={example.kind} aria-label="Loan application examples: local, global, and group-wise counterfactuals">
      <figcaption>{example.label}</figcaption>
      <div className="recourse-panel recourse-panel--local" data-example-paradigm="local">
        <h3>Local <span>One applicant</span></h3>
        <table className="example-profile" aria-label="Original and counterfactual applicant profile">
          <thead><tr><th scope="col">€/month</th><th scope="col">Original A</th><th scope="col">Counterfactual</th></tr></thead>
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
        <RecoursePlot paradigm="local" />
      </div>
      <div className="recourse-panel" data-example-paradigm="global">
        <h3>Global <span>One shared change</span></h3>
        <p className="recourse-action"><b>A-D</b> Income +€1,300; debt unchanged.</p>
        <RecoursePlot paradigm="global" />
      </div>
      <div className="recourse-panel" data-example-paradigm="group-wise">
        <h3>Group-wise <span>One change per group</span></h3>
        <p className="recourse-action recourse-action--higher-debt"><b>A-B</b> Debt −€1,000; income unchanged.</p>
        <p className="recourse-action recourse-action--lower-income"><b>C-D</b> Income +€1,300; debt unchanged.</p>
        <RecoursePlot paradigm="group-wise" />
      </div>
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
