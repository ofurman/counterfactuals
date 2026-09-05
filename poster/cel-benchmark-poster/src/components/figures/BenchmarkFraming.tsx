import { ClaimWording } from '@/components/poster'
import { ProjectQr } from '@/components/poster/ProjectQr'
import { resolveClaim } from '@/data/posterData'
import { counterfactualExample as example, exampleFeatureRows, examplePrediction } from '@/data/counterfactualExample'
import { RecoursePlot } from './RecoursePlot'

export function CounterfactualExample() {
  return (
    <figure className="ce-example" data-example-kind={example.kind} aria-label="Loan application examples: local, global, and group-wise counterfactuals">
      <p className="section-copy ce-definition">A <strong>counterfactual explanation</strong> describes the smallest, realistic change in input data needed to flip an algorithmic decision.</p>
      <blockquote className="ce-what-if">
        <p>It answers the “what-if” question by showing how a situation would look if specific facts were different. For example: “If your annual income was $500 higher, your loan would have been approved.”</p>
      </blockquote>
      <figcaption className="ce-example__accessible-label">{example.label}</figcaption>
      <div className="ce-example__panels">
        <div className="recourse-panel recourse-panel--local" data-example-paradigm="local">
          <h3>Local <span>One applicant</span></h3>
          <div className="recourse-actions">
            <p className="recourse-action"><b>A</b> Income +€500; debt −€500.</p>
          </div>
          <table className="example-profile example-profile--accessible" aria-label="Original and counterfactual applicant profile">
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
          <div className="recourse-actions">
            <p className="recourse-action"><b>A-D</b> Income +€1,300; debt unchanged.</p>
          </div>
          <RecoursePlot paradigm="global" />
        </div>
        <div className="recourse-panel" data-example-paradigm="group-wise">
          <h3>Group-wise <span>One change per group</span></h3>
          <div className="recourse-actions">
            <p className="recourse-action recourse-action--higher-debt"><b>A-B</b> Debt −€1,000; income unchanged.</p>
            <p className="recourse-action recourse-action--lower-income"><b>C-D</b> Income +€1,300; debt unchanged.</p>
          </div>
          <RecoursePlot paradigm="group-wise" />
        </div>
      </div>
    </figure>
  )
}

const contributions = [
  { claimId: 'contribution.protocol' },
  { claimId: 'contribution.benchmark' },
  { claimId: 'contribution.library' },
]

export function ContributionStack() {
  return (
    <div className="contribution-stack">
      {contributions.map((contribution, index) => (
        <article className={`contribution-item${contribution.claimId === 'contribution.library' ? ' contribution-item--extend' : ''}`} data-claim-id={contribution.claimId} key={contribution.claimId}>
          <span className="contribution-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
          <h3><ClaimWording claimId={contribution.claimId} /></h3>
          <p data-claim-detail={contribution.claimId}>{resolveClaim(contribution.claimId).posterDetail}</p>
          {contribution.claimId === 'contribution.library' && <ProjectQr />}
        </article>
      ))}
    </div>
  )
}
