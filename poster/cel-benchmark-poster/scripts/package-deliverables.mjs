import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { copyFile, readFile, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { deliverablesDir, projectDir, repositoryDir } from './harness.mjs'

const revisionResult = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: repositoryDir, encoding: 'utf8' })
if (revisionResult.status !== 0) throw new Error(`Could not resolve source revision: ${revisionResult.stderr}`)
const revision = revisionResult.stdout.trim()
const htmlName = 'cel-benchmark-poster.html'
const pdfName = 'cel-benchmark-poster-a0.pdf'
const previewName = 'cel-benchmark-poster-preview.png'
await copyFile(path.join(projectDir, 'bundle.html'), path.join(deliverablesDir, htmlName))

const finalNames = [htmlName, pdfName, previewName]
const lines = []
for (const name of finalNames) {
  const bytes = await readFile(path.join(deliverablesDir, name))
  if (bytes.length === 0) throw new Error(`Cannot package empty artifact: ${name}`)
  lines.push(`${createHash('sha256').update(bytes).digest('hex')}  ${name}`)
}
await writeFile(path.join(deliverablesDir, 'SHA256SUMS'), `${lines.join('\n')}\n`)

const readme = `# CEL scientific benchmark poster deliverables

Source revision: \`${revision}\`

## Contents

- \`${pdfName}\` — one-page A0 landscape PDF (1189 × 841 mm).
- \`${previewName}\` — 72 DPI PNG rendered from the final PDF with Poppler.
- \`${htmlName}\` — self-contained offline HTML poster; repository and documentation links remain clickable but are not required for rendering.
- \`SHA256SUMS\` — SHA-256 hashes for the three final artifacts above.
- \`audit-layout.json\` — measured native-canvas layout report.
- \`cel-benchmark-poster-print-review.png\` — retained Stage 6 print-review evidence.

## Print

Print the PDF at 100% / actual size on A0 landscape media. Do not use “fit to page.” The page is 1189 × 841 mm and includes the poster's configured safe area.

## Regenerate and verify

From \`poster/cel-benchmark-poster\`:

\`\`\`sh
pnpm install
pnpm run verify:all
cd deliverables
shasum -a 256 -c SHA256SUMS
\`\`\`

The full check validates manuscript-derived claims and graphics, the visual brief, TypeScript, tests, both Vite and single-file Parcel builds, offline rendering, native layout, PDF size/content, and package checksums.

Verified links: https://github.com/ofurman/counterfactuals and https://ofurman.github.io/counterfactuals/

## Known limitations

- The poster uses focused views of the manuscript result graphics; consult the paper and supplement for the full figure matrices and exact tables.
- Raster result panels preserve the manuscript's published plotting style and are intended for A0 print or zoomed digital inspection.
- Sparsity rankings are intentionally omitted until the manuscript's direction convention is reconciled; the source figure is retained without an added comparative claim.
`
await writeFile(path.join(deliverablesDir, 'README.md'), readme)

const sizes = await Promise.all(finalNames.map(async (name) => [name, (await stat(path.join(deliverablesDir, name))).size]))
console.log(`Packaged revision ${revision.slice(0, 12)}: ${sizes.map(([name, size]) => `${name}=${size}`).join(', ')}`)
