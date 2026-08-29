import { ClaimQualifier, ClaimVerdict } from '@/components/poster'

const priorities = [
  { label: 'Need reliable success?', claimId: 'global.adult.globe.validity', action: 'Inspect validity after coverage.' },
  { label: 'Need small change?', claimId: 'group.adult.tcrex.distance', action: 'Read distance with rounded-zero semantics.' },
  { label: 'Need plausibility?', claimId: 'local.blobs.ppcef.pp', action: 'Compare inside one dataset and model.' },
  { label: 'Missing output?', claimId: 'global.blobs.ares.missing', action: 'Keep missing distinct from zero or inapplicable.' },
]

export function SelectionGuide() {
  return (
    <div className="selection-guide">
      {priorities.map((priority) => (
        <div className="selection-row" data-claim-id={priority.claimId} key={priority.claimId}>
          <strong>{priority.label}</strong>
          <span>{priority.action}</span>
        </div>
      ))}
      <p className="selection-conclusion"><ClaimVerdict claimId="conclusion.tradeoffs" /></p>
      <p className="selection-caveat"><ClaimQualifier claimId="caveat.sparsity-direction" /></p>
    </div>
  )
}
