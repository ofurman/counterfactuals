import { describe, expect, it } from 'vitest'
import { posterData, resolveClaim, resolveSection } from './posterData'

describe('poster data adapter', () => {
  it('resolves every section claim against the frozen registry', () => {
    for (const section of posterData.sections) {
      expect(resolveSection(section.id).claims).toHaveLength(section.claimIds.length)
      section.claimIds.forEach((claimId) => expect(resolveClaim(claimId).id).toBe(claimId))
    }
  })

  it('composes the argument from frozen claim verdicts', () => {
    expect(posterData.argument).toContain(resolveClaim('scope.protocol').verdict)
    expect(posterData.argument).toContain(resolveClaim('conclusion.tradeoffs').verdict)
  })

  it('contains no drafting placeholders', () => {
    expect(JSON.stringify(posterData)).not.toMatch(/\b(?:todo|tbd|lorem ipsum|placeholder)\b/i)
  })
})
