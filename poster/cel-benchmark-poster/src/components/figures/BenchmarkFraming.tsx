import { ClaimVerdict } from '@/components/poster'

const confounds = ['Data splits', 'Preprocessing', 'Predictors', 'Constraints', 'Metrics']

export function BenchmarkMotivation() {
  return (
    <div className="benchmark-motivation" data-claim-id="scope.protocol">
      <p className="motivation-question">Are we measuring the CE method - or its setup?</p>
      <div className="confound-list" aria-label="Protocol choices controlled by CEL">
        {confounds.map((confound) => <span key={confound}>{confound}</span>)}
      </div>
    </div>
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
          <p><ClaimVerdict claimId={contribution.claimId} /></p>
        </article>
      ))}
    </div>
  )
}
