import { posterData, resolveClaim } from '@/data/posterData'

export function ConfoundDiagram() {
  const claim = resolveClaim('scope.protocol')
  const colors = posterData.visualSpec.colors
  return (
    <figure className="confound-figure" data-claim-id={claim.id}>
      <svg viewBox="0 0 310 210" role="img" aria-labelledby="confound-title confound-description">
        <title id="confound-title">Protocol variation obscures method comparison</title>
        <desc id="confound-description">Two method paths pass through different setup choices before CEL aligns them under shared controls.</desc>
        <path d="M22 38 C105 38 72 94 155 94 S210 38 288 38" fill="none" stroke={colors.orange} strokeWidth="6" />
        <path d="M22 74 C92 74 98 130 155 130 S226 74 288 74" fill="none" stroke={colors.navyMuted} strokeWidth="6" />
        <line x1="28" y1="166" x2="282" y2="166" stroke={colors.teal} strokeWidth="10" />
        <text x="22" y="24" fontSize="13" fontWeight="700">Different setups</text>
        <text x="155" y="195" textAnchor="middle" fontSize="13" fontWeight="800">Shared CEL controls</text>
      </svg>
      <figcaption>{claim.verdict}</figcaption>
    </figure>
  )
}
