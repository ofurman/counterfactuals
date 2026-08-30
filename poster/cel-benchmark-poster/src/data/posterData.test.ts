import { describe, expect, it } from 'vitest'
import { posterData, resolveClaim, resolveSection } from './posterData'

describe('poster data adapter', () => {
  it('names every scope item and splits methods by their manuscript paradigm', () => {
    expect(posterData.scopeFacts.map((fact) => fact.claimId)).toEqual(['scope.datasets', 'scope.methods', 'scope.backbones', 'scope.metrics'])
    for (const fact of posterData.scopeFacts) {
      const names = fact.inventory.flatMap((group) => group.names)
      expect(new Set(names).size).toBe(names.length)
      if (fact.claimId === 'scope.backbones') {
        expect(fact.inventory.every((group) => group.names.length === fact.value)).toBe(true)
      } else expect(names).toHaveLength(Number(fact.value))
    }
    const groups = resolveClaim('scope.methods').inventory!
    expect(groups.map((group) => [group.label, group.names.length])).toEqual([['Local', 10], ['Global', 2], ['Group-wise', 2]])
    expect(groups.find((group) => group.label === 'Local')?.names).toContain('CEARM')
    expect(groups.find((group) => group.label === 'Group-wise')?.names).toEqual(['GLANCE', 'T-CREx'])
  })

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
