import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { repositoryDir } from './harness.mjs'

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex')

export async function loadManuscriptAssets() {
  const generatorSha256 = hash(await readFile(path.join(repositoryDir, 'poster/plots/manuscript_typography.py')))
  return Promise.all(['architecture', 'global', 'local', 'group', 'regression'].map(async (kind) => {
    const svg = await readFile(path.join(repositoryDir, `poster/plots/generated/manuscript-${kind}.svg`), 'utf8')
    const metadata = JSON.parse(svg.match(/<metadata>([\s\S]*?)<\/metadata>/)?.[1] ?? '{}')
    if (metadata.generatorSha256 !== generatorSha256 || metadata.sourceSha256 !== hash(await readFile(path.join(repositoryDir, metadata.source)))) throw new Error(`Stale manuscript typography: ${kind}`)
    if (kind === 'architecture') {
      const original = svg.replace(/^<\?xml[^>]*>\s*/, '').replace(/<metadata>[\s\S]*?<\/metadata>/, '')
      if (metadata.presentation !== 'original-manuscript' || !svg.includes('id="glyph-') || svg.includes('poster-typography') || hash(original) !== metadata.sourceSvgSha256) throw new Error('Architecture must preserve the complete original manuscript SVG conversion, including glyphs')
    } else {
      if (metadata.fontFamily !== 'Arial' || /<text\b/.test(svg)) throw new Error(`${kind} must have outlined Arial typography`)
      const labels = [...svg.matchAll(/data-font-size="([^"]+)"/g)].map((match) => Number(match[1]))
      if (!labels.length || Math.min(...labels) !== metadata.minimumFontSize) throw new Error(`Unverified font size: ${kind}`)
    }
    const viewBox = svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number)
    if (kind !== 'architecture') {
      if (kind === 'global' && (metadata.layout !== 'three-plus-two' || JSON.stringify(metadata.methodKey) !== JSON.stringify(metadata.methods.map((method, index) => ({ tick: String(index + 1), method }))))) throw new Error('Global numbered ticks must match the complete, ordered method key')
      const images = [...svg.matchAll(/href="data:image\/png;base64,([^"]+)"/g)]
      if (images.length !== 5 || metadata.crops.length !== 5) throw new Error(`${kind} must retain five source plot crops`)
      images.forEach((image, index) => {
        if (hash(Buffer.from(image[1], 'base64')) !== metadata.crops[index].pngSha256) throw new Error(`Changed plot pixels: ${kind}/${index}`)
      })
    }
    return { kind, svg, ...metadata, width: viewBox[2], height: viewBox[3] }
  }))
}

export async function auditManuscriptLabelBounds(page, assets) {
  const inspection = await page.context().newPage()
  try {
    for (const asset of assets) {
      // The original schema is validated as an intact source conversion above;
      // only result derivatives contain replacement-label bounding boxes.
      if (asset.kind === 'architecture') continue
      await inspection.setContent(asset.svg.replace(/<\?xml[^>]*>/, ''))
      const errors = await inspection.locator('svg').evaluate((svg) => {
        const canvas = svg.getBoundingClientRect()
        const errors = []
        for (const label of svg.querySelectorAll('[data-label]')) {
          const box = label.getBoundingClientRect()
          let container = canvas
          if (label.dataset.container) {
            const node = [...svg.querySelectorAll('use')].find((node) => node.getAttribute('xlink:href') === label.dataset.container)
            if (!node) { errors.push(`No schema container: ${label.dataset.label}`); continue }
            container = node.getBoundingClientRect()
          }
          if (box.left < container.left - 0.5 || box.right > container.right + 0.5 || box.top < container.top - 0.5 || box.bottom > container.bottom + 0.5) errors.push(`Clipped label: ${label.dataset.label}`)
        }
        return errors
      })
      if (errors.length) throw new Error(`${asset.kind}: ${errors.join('; ')}`)
    }
  } finally {
    await inspection.close()
  }
}
