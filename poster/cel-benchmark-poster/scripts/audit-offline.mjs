import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { projectDir, withPosterPage } from './harness.mjs'

const bundlePath = path.join(projectDir, 'bundle.html')
const bundle = await readFile(bundlePath, 'utf8')
const bundleStat = await stat(bundlePath)
if (bundleStat.size === 0 || bundleStat.size > 3_000_000) throw new Error(`Bundle size is outside the 1–3,000,000 byte contract: ${bundleStat.size}`)
if (!bundle.includes('<style>') || !bundle.includes('<script>')) throw new Error('Bundle does not contain inline style and script assets')
if (/<script\b[^>]*\bsrc\s*=|<link\b[^>]*rel=["']?stylesheet/i.test(bundle)) throw new Error('Bundle contains a required external or local script or stylesheet')
for (const image of bundle.matchAll(/<img\b[^>]*\bsrc=["']([^"']+)["']/gi)) {
  if (!image[1].startsWith('data:')) throw new Error(`Bundle contains a non-inline image: ${image[1]}`)
}

const result = await withPosterPage(async ({ page, failures, remoteRequests, consoleErrors }) => {
  const rendered = await page.evaluate(() => ({
    canvas: Boolean(document.querySelector('.poster-canvas')),
    sections: document.querySelectorAll('[data-section]').length,
    claimMarkers: document.querySelectorAll('[data-claim-id]').length,
    manuscriptFigures: document.querySelectorAll('[data-manuscript-source]').length,
    brandLogos: document.querySelectorAll('[data-brand-id]').length,
    examplePlots: [...document.querySelectorAll('[data-example-plot]')].filter((image) => image.complete && image.naturalWidth > 0).length,
    example: Boolean(document.querySelector('[data-example-kind="illustrative"]')),
    regression: Boolean(document.querySelector('[data-finding="regression"]')),
    links: [...document.querySelectorAll('a[href]')].map((link) => link.href),
    readyState: document.readyState,
  }))
  const defects = [...failures, ...consoleErrors]
  if (remoteRequests.length) defects.push(`Required remote requests: ${remoteRequests.join(', ')}`)
  if (!rendered.canvas || rendered.sections !== 10 || rendered.claimMarkers < 13 || rendered.manuscriptFigures !== 5 || rendered.brandLogos !== 5 || !rendered.example || rendered.examplePlots !== 3 || !rendered.regression || rendered.readyState !== 'complete') defects.push(`Incomplete offline render: ${JSON.stringify(rendered)}`)
  if (defects.length) throw new Error(defects.join('\n'))
  return rendered
}, { entry: 'bundle', blockRemote: true })

console.log(`Offline bundle audit passed: bytes=${bundleStat.size}, sections=${result.sections}, claims=${result.claimMarkers}, manuscript figures=${result.manuscriptFigures}, required remote requests=0`)
