import { claimDisplay, resolveClaim } from '@/data/posterData'

type ClaimProps = { claimId: string; className?: string }

export function ClaimValue({ claimId, className = '' }: ClaimProps) {
  const claim = resolveClaim(claimId)
  return (
    <span
      className={className}
      data-claim-id={claim.id}
      data-claim-status={claim.status}
      title={claim.qualifier}
    >
      {claimDisplay(claimId)}
    </span>
  )
}

export function ClaimWording({ claimId, className = '' }: ClaimProps) {
  const claim = resolveClaim(claimId)
  return <span className={className} data-claim-id={claim.id}>{claim.posterWording}</span>
}

export function ClaimQualifier({ claimId, className = '' }: ClaimProps) {
  const claim = resolveClaim(claimId)
  return <span className={className} data-claim-id={claim.id}>{claim.qualifier}</span>
}

export function ClaimVerdict({ claimId, className = '' }: ClaimProps) {
  const claim = resolveClaim(claimId)
  return <span className={className} data-claim-id={claim.id}>{claim.verdict}</span>
}
