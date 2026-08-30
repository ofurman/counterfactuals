import claimsJson from '../../../research/claims/claims.generated.json'
import identityJson from '../../../research/identity.json'
import methodNamesJson from '../../../research/method-names.json'
import contentJson from '../../../research/poster-content.json'
import visualSpecJson from '../../../research/visual-spec.json'
import precedentsJson from '../../../research/precedents.json'

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

export function claimDisplay(claimId: string): string {
  const claim = resolveClaim(claimId)
  if ('display' in claim.value && typeof claim.value.display === 'string') return claim.value.display
  return claim.posterWording
}

export function claimMean(claimId: string): number | null {
  const claim = resolveClaim(claimId)
  if ('mean' in claim.value && typeof claim.value.mean === 'number') return claim.value.mean
  return null
}

export function resultDescriptor(claimId: string) {
  const claim = resolveClaim(claimId)
  const display = claimDisplay(claimId)
  const wording = claim.posterWording.endsWith(display)
    ? claim.posterWording.slice(0, -display.length).trim()
    : claim.posterWording
  const separator = wording.indexOf(': ')
  return {
    context: separator >= 0 ? wording.slice(0, separator) : '',
    label: separator >= 0 ? wording.slice(separator + 2) : wording,
  }
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
const datasetScope = resolveClaim('scope.datasets').value as Extract<ClaimValue, { kind: 'finite' }>
const methodScope = resolveClaim('scope.methods').value as Extract<ClaimValue, { kind: 'finite' }>
const backboneScope = resolveClaim('scope.backbones').value as Extract<ClaimValue, { kind: 'finite' }>
const foldScope = resolveClaim('scope.folds').value as Extract<ClaimValue, { kind: 'finite' }>
const metricScope = resolveClaim('scope.metrics').value as Extract<ClaimValue, { kind: 'finite' }>

export const posterData = {
  claims,
  sections: sections.map((section) => ({
    ...section,
    claims: section.claimIds.map(resolveClaim),
  })) as ResolvedSection[],
  identity: identityJson,
  methodNames: methodNamesJson,
  visualSpec: visualSpecJson,
  precedents: precedentsJson.items,
  links: contentJson.links,
  resultVisuals: contentJson.resultVisuals,
  argument: argumentClaims
    .map((claim) => claim.verdict)
    .join(contentJson.argument.separator),
  protocol: {
    caption: protocolClaim.posterWording,
    controls: protocolControls,
  },
  scopeFacts: [
    { claimId: 'scope.datasets', value: datasetScope.total, label: 'datasets' },
    { claimId: 'scope.methods', value: methodScope.total, label: 'methods' },
    {
      claimId: 'scope.methods',
      value: ['local', 'global', 'groupWise'].filter((key) => typeof methodScope[key] === 'number').length,
      label: 'paradigms',
    },
    { claimId: 'scope.backbones', value: backboneScope.total, label: 'backbones / task' },
    { claimId: 'scope.folds', value: foldScope.total, label: 'folds' },
    { claimId: 'scope.metrics', value: metricScope.total, label: 'classification metrics' },
  ],
} as const
