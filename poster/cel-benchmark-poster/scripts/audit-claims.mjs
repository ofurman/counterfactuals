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
    qrCount: document.querySelectorAll('[data-qr-destination]').length,
    qrOwner: document.querySelector('[data-qr-destination]')?.closest('article')?.dataset.claimId,
    qrSection: document.querySelector('[data-qr-destination]')?.closest('[data-section]')?.getAttribute('data-section'),
    header: document.querySelector('.poster-header')?.textContent ?? '',
    title: document.querySelector('.poster-header h1')?.textContent ?? '',
    visibleWordCount: document.querySelector('.poster-canvas')?.innerText.trim().split(/\s+/).length ?? 0,
    brands: [...document.querySelectorAll('[data-brand-id]')].map((image) => ({ id: image.dataset.brandId, alt: image.alt, loaded: image.complete && image.naturalWidth > 0 })),
    example: {
      kind: document.querySelector('[data-example-kind]')?.getAttribute('data-example-kind'),
      text: document.querySelector('.ce-example')?.textContent ?? '',
      features: [...document.querySelectorAll('[data-example-feature]')].map((element) => ({
        key: element.dataset.exampleFeature,
        changed: element.dataset.changed === 'true',
        original: element.querySelector('[data-example-value="original"]')?.textContent,
        counterfactual: element.querySelector('[data-example-value="counterfactual"]')?.textContent,
        highlighted: Boolean(element.querySelector('.example-value--changed')),
      })),
      summary: document.querySelector('.example-summary')?.textContent,
      predictions: [...document.querySelectorAll('.example-prediction')].map((element) => element.textContent),
      isResult: Boolean(document.querySelector('.ce-example[data-result-surface], .ce-example[data-finding], .ce-example[data-manuscript-source]')),
    },
    hasRegression: Boolean(document.querySelector('[data-section="regression-tradeoff"], [data-finding="regression"]')),
    scope: [...document.querySelectorAll('.scope-tile')].map((element) => ({
      id: element.getAttribute('data-claim-id'),
      text: element.querySelector('.scope-tile__heading')?.textContent?.trim() ?? '',
      inventory: [...element.querySelectorAll('[data-scope-group]')].map((group) => ({ label: group.dataset.scopeGroup, names: [...group.querySelectorAll('[data-scope-name]')].map((name) => name.textContent) })),
    })),
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
  ['scope.backbones', claimsById.get('scope.backbones').value.total, 'backbones / task'],
  ['scope.metrics', claimsById.get('scope.metrics').value.total, 'classification metrics'],
]
if (JSON.stringify(rendered.scope) !== JSON.stringify(expectedScope.map(([id, value, label]) => ({ id, text: `${value}${label}`, inventory: claimsById.get(id).inventory })))) throw new Error('Rendered scope tiles and named inventories do not match the generated ledger')
const allowedSources = new Set(content.sections.flatMap((section) => section.sourceCitations))
for (const source of rendered.sources) if (!allowedSources.has(source) && !source.endsWith('#Related Works')) throw new Error(`Rendered unknown source citation: ${source}`)
const allowedLinks = new Set(Object.values(identity.links))
for (const link of rendered.links) if (!allowedLinks.has(link)) throw new Error(`Rendered unknown link: ${link}`)
if (rendered.qr !== identity.qr.url) throw new Error('Rendered QR destination does not match identity')
if (rendered.qrCount !== 1 || rendered.qrOwner !== 'contribution.library' || rendered.qrSection !== 'guidance-limitations') throw new Error('The unique project QR must be inside the Extend contribution')
if (rendered.title !== identity.title) throw new Error('Poster title differs from the manuscript title')
if (!rendered.visibleWordCount || rendered.visibleWordCount > 260) throw new Error(`Poster exceeds the concise visible-text budget: ${rendered.visibleWordCount}/260`)
for (const text of [identity.affiliation, ...identity.authors.map((author) => author.name)]) if (!rendered.header.includes(text)) throw new Error(`Header identity is missing: ${text}`)
if (rendered.header.includes(identity.venue)) throw new Error('Removed venue marker remains in the header')

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
if (rendered.brands.length !== 5) throw new Error('Expected the three institutional logos and two conference logos')
for (const brand of brands.assets) {
  const bytes = await readFile(path.join(repositoryDir, brand.localFile))
  if (createHash('sha256').update(bytes).digest('hex') !== brand.sha256) throw new Error(`Reference logo was modified: ${brand.id}`)
  const visible = rendered.brands.find((item) => item.id === brand.id)
  if (!visible?.loaded || visible.alt !== brand.label) throw new Error(`Missing or mislabelled reference logo: ${brand.id}`)
}
const money = (value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: example.currency, maximumFractionDigits: 0 }).format(value)
const expectedFeatures = example.features.map((feature) => {
  const format = (value) => feature.format === 'currency' ? money(value) : feature.format === 'years' ? `${value} years` : String(value)
  const changed = example.original[feature.key] !== example.counterfactual[feature.key]
  if (changed && !feature.actionable) throw new Error(`Example changes a fixed feature: ${feature.key}`)
  return { key: feature.key, changed, original: format(example.original[feature.key]), counterfactual: format(example.counterfactual[feature.key]), highlighted: changed }
})
if (rendered.example.kind !== 'illustrative' || rendered.example.isResult || !rendered.example.text.includes(example.label) || !example.provenance) throw new Error('Example is not separated from benchmark evidence')
if (JSON.stringify(rendered.example.features) !== JSON.stringify(expectedFeatures)) throw new Error('Displayed example does not match its feature profiles and highlights')
const changedCount = expectedFeatures.filter((feature) => feature.changed).length
if (expectedFeatures.length !== 6 || changedCount !== 2 || rendered.example.summary !== `Only ${changedCount} of ${expectedFeatures.length} features change.`) throw new Error('Example must show two changed features and four unchanged features')
const predict = (profile) => profile.monthlyIncome - profile.monthlyDebt >= example.model.minimumIncomeAfterDebt && profile.creditHistoryYears >= example.model.minimumCreditHistoryYears && profile.loanAmount / (12 * profile.monthlyIncome) <= example.model.maximumLoanToAnnualIncomeRatio ? 'Approved' : 'Declined'
if (predict(example.original) !== 'Declined' || predict(example.counterfactual) !== 'Approved' || JSON.stringify(rendered.example.predictions) !== JSON.stringify(['Declined', 'Approved'])) throw new Error('Counterfactual does not flip the declared example prediction')
if (/toy|source|not benchmark|threshold/i.test(rendered.example.text)) throw new Error('Removed explanatory footnotes remain visible in the example')

console.log(`Claim audit passed: rendered markers=${rendered.claims.length}, visible=${rendered.claims.filter((claim) => claim.visible).length}, source metadata=${rendered.sources.length}, manuscript figures=${rendered.manuscriptFigures.length}, logos=${rendered.brands.length}, example features=${expectedFeatures.length} (${changedCount} changed), visible words=${rendered.visibleWordCount}/260, manuscript title=exact`)
