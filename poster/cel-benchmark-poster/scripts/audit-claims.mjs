import { readFile } from 'node:fs/promises'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { repositoryDir, withPosterPage } from './harness.mjs'

for (const validator of ['validate-claims.mjs', 'validate-brief.mjs']) {
  const result = spawnSync(process.execPath, [path.join(repositoryDir, 'poster/research/scripts', validator)], { cwd: repositoryDir, encoding: 'utf8' })
  if (result.status !== 0) throw new Error(`${validator} failed:\n${result.stdout}${result.stderr}`)
  process.stdout.write(result.stdout)
}

const claims = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/claims/claims.generated.json'), 'utf8'))
const identity = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/identity.json'), 'utf8'))
const content = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/poster-content.json'), 'utf8'))
const claimsById = new Map(claims.claims.map((claim) => [claim.id, claim]))

const rendered = await withPosterPage(async ({ page, failures }) => {
  if (failures.length) throw new Error(failures.join('\n'))
  return page.evaluate(() => ({
    claims: [...document.querySelectorAll('[data-claim-id]')].map((element) => ({
      id: element.getAttribute('data-claim-id'),
      text: element.textContent?.trim() ?? '',
      status: element.getAttribute('data-claim-status'),
      visible: element.getClientRects().length > 0,
    })),
    sources: [...document.querySelectorAll('[data-source-citation]')].map((element) => element.getAttribute('data-source-citation')),
    links: [...document.querySelectorAll('a[href]')].map((element) => element.href),
    qr: document.querySelector('[data-qr-destination]')?.getAttribute('data-qr-destination'),
    header: document.querySelector('.poster-header')?.textContent ?? '',
    scope: [...document.querySelectorAll('.scope-strip li')].map((element) => ({ id: element.getAttribute('data-claim-id'), text: element.textContent?.trim() ?? '' })),
  }))
})

for (const marker of rendered.claims) {
  const claim = claimsById.get(marker.id)
  if (!claim) throw new Error(`Rendered unknown claim: ${marker.id}`)
  if (!claim.qualifier || !claim.source?.file) throw new Error(`Rendered claim lacks qualifier/source: ${marker.id}`)
  await readFile(path.join(repositoryDir, claim.source.file), 'utf8')
  if (marker.status) {
    const expected = typeof claim.value?.display === 'string' ? claim.value.display : claim.posterWording
    if (marker.text !== expected) throw new Error(`Rendered value for ${marker.id} is ${JSON.stringify(marker.text)}, expected ${JSON.stringify(expected)}`)
  }
}
const expectedScope = [
  ['scope.datasets', claimsById.get('scope.datasets').value.total, 'datasets'],
  ['scope.methods', claimsById.get('scope.methods').value.total, 'methods'],
  ['scope.methods', ['local', 'global', 'groupWise'].filter((key) => Number.isFinite(claimsById.get('scope.methods').value[key])).length, 'paradigms'],
  ['scope.backbones', claimsById.get('scope.backbones').value.total, 'backbones / task'],
  ['scope.folds', claimsById.get('scope.folds').value.total, 'folds'],
]
if (JSON.stringify(rendered.scope) !== JSON.stringify(expectedScope.map(([id, value, label]) => ({ id, text: `${value}${label}` })))) throw new Error('Rendered scope facts do not match the generated ledger')
const allowedSources = new Set(content.sections.flatMap((section) => section.sourceCitations))
for (const source of rendered.sources) if (!allowedSources.has(source) && !source.endsWith('#Related Works')) throw new Error(`Rendered unknown source citation: ${source}`)
const allowedLinks = new Set(Object.values(identity.links))
for (const link of rendered.links) if (!allowedLinks.has(link)) throw new Error(`Rendered unknown link: ${link}`)
if (rendered.qr !== identity.qr.url) throw new Error('Rendered QR destination does not match identity')
for (const text of [identity.venue, identity.affiliation, ...identity.authors.map((author) => author.name)]) if (!rendered.header.includes(text)) throw new Error(`Header identity is missing: ${text}`)

console.log(`Claim audit passed: rendered markers=${rendered.claims.length}, visible=${rendered.claims.filter((claim) => claim.visible).length}, source notes=${rendered.sources.length}`)
