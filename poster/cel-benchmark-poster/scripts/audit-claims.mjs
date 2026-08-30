import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
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
const brands = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/brand-assets.json'), 'utf8'))
const example = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/ce-example.json'), 'utf8'))
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
    title: document.querySelector('.poster-header h1')?.textContent ?? '',
    visibleWordCount: document.querySelector('.poster-canvas')?.innerText.trim().split(/\s+/).length ?? 0,
    brands: [...document.querySelectorAll('[data-brand-id]')].map((image) => ({ id: image.dataset.brandId, alt: image.alt, loaded: image.complete && image.naturalWidth > 0 })),
    example: {
      kind: document.querySelector('[data-example-kind]')?.getAttribute('data-example-kind'),
      text: document.querySelector('.ce-example')?.textContent ?? '',
      values: [...document.querySelectorAll('[data-example-value]')].map((element) => [element.dataset.exampleValue, element.textContent]),
      isResult: Boolean(document.querySelector('.ce-example[data-result-surface], .ce-example[data-finding], .ce-example[data-manuscript-source]')),
    },
    hasRegression: Boolean(document.querySelector('[data-section="regression-tradeoff"], [data-finding="regression"]')),
    scope: [...document.querySelectorAll('.scope-strip li')].map((element) => ({ id: element.getAttribute('data-claim-id'), text: element.textContent?.trim() ?? '' })),
    manuscriptFigures: [...document.querySelectorAll('[data-manuscript-source]')].map((element) => ({
      source: element.getAttribute('data-manuscript-source'),
      alt: element.querySelector('img')?.getAttribute('alt') ?? '',
    })),
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
if (rendered.title !== identity.title) throw new Error('Poster title differs from the manuscript title')
if (!rendered.visibleWordCount || rendered.visibleWordCount > 260) throw new Error(`Poster exceeds the concise visible-text budget: ${rendered.visibleWordCount}/260`)
for (const text of [identity.venue, identity.affiliation, ...identity.authors.map((author) => author.name)]) if (!rendered.header.includes(text)) throw new Error(`Header identity is missing: ${text}`)

const expectedManuscriptFigures = [
  'manuscript/figures/teaser.pdf',
  'manuscript/figures/metrics_boxplot_local.png',
  'manuscript/figures/metrics_boxplot_global.png',
  'manuscript/figures/metrics_boxplot_group_wise.png',
]
if (JSON.stringify(rendered.manuscriptFigures.map((figure) => figure.source).sort()) !== JSON.stringify(expectedManuscriptFigures.sort())) throw new Error('Rendered manuscript figure inventory is incomplete or substituted')
for (const figure of rendered.manuscriptFigures) {
  await readFile(path.join(repositoryDir, figure.source))
  if (figure.alt.length < 30) throw new Error(`Manuscript figure lacks meaningful alt text: ${figure.source}`)
}

if (rendered.hasRegression) throw new Error('Removed regression results still appear on the poster')
if (rendered.brands.length !== 3) throw new Error('Expected the three requested reference-poster logos')
for (const brand of brands.assets) {
  const bytes = await readFile(path.join(repositoryDir, brand.localFile))
  if (createHash('sha256').update(bytes).digest('hex') !== brand.sha256) throw new Error(`Reference logo was modified: ${brand.id}`)
  const visible = rendered.brands.find((item) => item.id === brand.id)
  if (!visible?.loaded || visible.alt !== brand.label) throw new Error(`Missing or mislabelled reference logo: ${brand.id}`)
}
const money = (value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: example.currency, maximumFractionDigits: 0 }).format(value)
const expectedExampleValues = [['original', money(example.originalIncome)], ['counterfactual', money(example.counterfactualIncome)], ['threshold', money(example.approvalThreshold)]]
if (rendered.example.kind !== 'illustrative' || rendered.example.isResult || !rendered.example.text.includes(example.disclaimer) || !rendered.example.text.includes(example.fixedFeatures)) throw new Error('Toy example is missing its illustrative provenance or fixed-feature note')
if (JSON.stringify(rendered.example.values) !== JSON.stringify(expectedExampleValues)) throw new Error('Displayed toy example does not match its declared inputs')
if (!(example.originalIncome < example.approvalThreshold && example.counterfactualIncome === example.approvalThreshold) || !rendered.example.text.includes('Declined') || !rendered.example.text.includes('Approved')) throw new Error('Toy counterfactual does not flip its stated model prediction')

console.log(`Claim audit passed: rendered markers=${rendered.claims.length}, visible=${rendered.claims.filter((claim) => claim.visible).length}, source notes=${rendered.sources.length}, manuscript figures=${rendered.manuscriptFigures.length}, logos=${rendered.brands.length}, toy example=labelled, visible words=${rendered.visibleWordCount}/260, manuscript title=exact`)
