import claimsJson from '../../../research/claims/claims.generated.json'
import identityJson from '../../../research/identity.json'
import methodNamesJson from '../../../research/method-names.json'
import contentJson from '../../../research/poster-content.json'
import visualSpecJson from '../../../research/visual-spec.json'

export type ClaimValue =
  | { kind: 'finite'; [key: string]: string | number }
  | { kind: 'missing'; token?: string; meaning?: string }
  | { kind: 'qualitative' }

export type Claim = {
  id: string
  claimKind: string
  posterWording: string
  value: ClaimValue
  unit: string | null
  verdict: string
  source: { file: string; anchor: string }
  extractionRule: string
  direction: string
  qualifier: string
  status: 'publishable' | 'qualified' | 'contradictory'
}

export type PosterSection = {
  id: string
  owner: 'header' | 'left' | 'center' | 'right' | 'bottom' | 'footer'
  order: number
  heading: string
  copy: string[]
  claimIds: string[]
  sourceCitations: string[]
  linkIds: string[]
  assetRoles: string[]
}

export type ResolvedSection = PosterSection & { claims: Claim[] }

const claims = claimsJson.claims as Claim[]
const sections = contentJson.sections as PosterSection[]
const claimsById = new Map(claims.map((claim) => [claim.id, claim]))

export function resolveClaim(claimId: string): Claim {
  const claim = claimsById.get(claimId)
  if (!claim) throw new Error(`Unknown frozen claim: ${claimId}`)
  return claim
}

export function resolveSection(sectionId: string): ResolvedSection {
  const section = sections.find((candidate) => candidate.id === sectionId)
  if (!section) throw new Error(`Unknown poster section: ${sectionId}`)
  return { ...section, claims: section.claimIds.map(resolveClaim) }
}

const placeholderPattern = /\b(?:todo|tbd|lorem ipsum|placeholder)\b/i
const allFrozenText = JSON.stringify({ claims, sections, identityJson, methodNamesJson })
if (placeholderPattern.test(allFrozenText)) {
  throw new Error('Frozen poster inputs contain placeholder text')
}

const argumentClaims = contentJson.argument.claimIds.map(resolveClaim)
const protocolClaim = resolveClaim('scope.protocol')
const protocolControls = protocolClaim.posterWording
  .replace(/^.*?:\s*/, '')
  .replace(/, and /, ', ')
  .split(', ')

export const posterData = {
  claims,
  sections: sections.map((section) => ({
    ...section,
    claims: section.claimIds.map(resolveClaim),
  })) as ResolvedSection[],
  identity: identityJson,
  methodNames: methodNamesJson,
  visualSpec: visualSpecJson,
  links: contentJson.links,
  resultVisuals: contentJson.resultVisuals,
  argument: argumentClaims
    .map((claim) => claim.verdict)
    .join(contentJson.argument.separator),
  protocol: {
    caption: protocolClaim.posterWording,
    controls: protocolControls,
  },
} as const
