import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { repositoryDir } from './harness.mjs'

// Tie the imported figures to the tested Matplotlib generator and current data.
export async function loadExampleAssets() {
  const digest = async (relative) => createHash('sha256').update(await readFile(path.join(repositoryDir, relative))).digest('hex')
  const expectedHashes = {
    dataSha256: await digest('poster/research/ce-example.json'),
    styleSha256: await digest('cel/plotting/plot_utils.py'),
    generatorSha256: await digest('poster/plots/plot_ce_examples.py'),
  }
  return Promise.all(['local', 'global', 'group-wise'].map(async (paradigm) => {
    const source = `poster/plots/generated/ce-example-${paradigm}.svg`
    const svg = await readFile(path.join(repositoryDir, source), 'utf8')
    const metadata = JSON.parse(svg.match(/<dc:description>([\s\S]*?)<\/dc:description>/)?.[1] ?? '{}')
    for (const [key, expected] of Object.entries(expectedHashes)) {
      if (metadata[key] !== expected) throw new Error(`${paradigm} Matplotlib asset is stale: ${key}`)
    }
    if (metadata.fontFamily !== 'Arial' || !svg.includes('ArialMT-') || !metadata.transparent || /<image\b|<text\b/i.test(svg)) throw new Error(`${paradigm} must be a transparent, Arial-outlined vector`)
    const legendLabels = paradigm === 'global' ? ['Original', 'Counterfactual', 'Decision boundary'] : []
    if (JSON.stringify(metadata.legendLabels) !== JSON.stringify(legendLabels) || svg.includes('id="global-legend"') !== (paradigm === 'global')) throw new Error('Only global must have the three-item legend')
    for (const label of legendLabels) if (!svg.includes(`<!-- ${label} -->`)) throw new Error(`Missing legend label: ${label}`)
    const transitions = metadata.transitions?.[paradigm]
    if (transitions?.length !== (paradigm === 'local' ? 1 : 4)) throw new Error(`${paradigm} has the wrong applicant count`)
    for (const transition of transitions) {
      if (!svg.includes(`id="${paradigm}-arrow-${transition.id}"`)) throw new Error(`${paradigm}/${transition.id} has no rendered arrow`)
    }
    for (const suffix of ['boundary', 'originals-declined', 'counterfactuals-approved']) {
      if (!svg.includes(`id="${paradigm}-${suffix}"`)) throw new Error(`${paradigm} is missing its ${suffix} layer`)
    }
    const viewBox = svg.match(/viewBox="([^"]+)"/)?.[1].split(/\s+/).map(Number)
    if (!viewBox || viewBox.length !== 4) throw new Error(`${paradigm} SVG lacks dimensions`)
    return { paradigm, source, transitions, width: viewBox[2], height: viewBox[3], minimumFontPt: metadata.minimumFontPt }
  }))
}
