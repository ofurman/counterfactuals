import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { copyFile, readFile, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { deliverablesDir, projectDir, repositoryDir } from './harness.mjs'

const revisionResult = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: repositoryDir, encoding: 'utf8' })
if (revisionResult.status !== 0) throw new Error(`Could not resolve source revision: ${revisionResult.stderr}`)
const revision = revisionResult.stdout.trim()
const htmlName = 'cel-benchmark-poster.html'
const pdfName = 'cel-benchmark-poster-a1.pdf'
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

- \`${pdfName}\` — one-page A1 portrait PDF (594 × 841 mm).
- \`${previewName}\` — 72 DPI PNG rendered from the final PDF with Poppler.
- \`${htmlName}\` — self-contained offline HTML poster; the contribution QR links to the repository but is not required for rendering. No bottom reproduction strip is shown.
- \`SHA256SUMS\` — SHA-256 hashes for the three final artifacts above.
- \`audit-layout.json\` — measured native-canvas layout report.
- \`cel-benchmark-poster-print-review.png\` — retained Stage 6 print-review evidence.

## Print

Print the PDF at 100% / actual size on A1 portrait media. Do not use “fit to page.” The page is 594 × 841 mm with 13.5 mm safe margins. The previous A0 PDF is retained for reference only and is not the workshop print file.

## Regenerate and verify

From \`poster/cel-benchmark-poster\`:

\`\`\`sh
pnpm install
pnpm run verify:all
cd deliverables
shasum -a 256 -c SHA256SUMS
\`\`\`

The full check validates manuscript-derived claims and graphics, the three-feature CE example, regenerated plot-only Matplotlib SVGs, local/global/group-wise actions and declined-to-approved transitions, non-printing provenance, reference-logo hashes, column order, the visual brief, TypeScript, tests, both Vite and single-file Parcel builds, offline rendering, native layout, PDF size/content, and package checksums.

Verified links: https://github.com/ofurman/counterfactuals and https://ofurman.github.io/counterfactuals/

## Known limitations

- The loan application example uses invented profiles and an illustrative affordability rule, not benchmark data or lending advice. Its feature values, unchanged fields, and model assumptions are recorded in \`poster/research/ce-example.json\`; these provenance notes are not printed on the poster.
- PWr, genwro.AI, and Tooploox logos are unchanged copies from the user-supplied PUMAL reference poster; their provenance and hashes are recorded in \`poster/research/brand-assets.json\`.
- XKDD and ECML-PKDD logos use the supplied project assets unchanged; their original external sources are not recorded. Local file hashes are retained in the same brand manifest.
- The poster uses focused views of the manuscript result graphics; consult the paper and supplement for the full figure matrices and exact tables.
- The example plots are transparent, font-outlined SVGs generated with CEL plotting helpers. Their data, generator, and style hashes are validated before packaging.
- Result SVGs retain lossless manuscript plot crops with enlarged outlined Arial labels; schema boxes and connectors remain original vectors. Source hashes, crop pixels, and uniform scaling are checked. The source manuscript files are unchanged.
- Typography uses an 80pt Georgia title, 28pt Georgia subheadings, approximately 18pt Arial body text, and manuscript figure labels of at least 17pt at A1. Regression density ticks use k for thousands. The examples and framework occupy the upper two columns; a two-by-two results grid spans the lower page, followed by the full-width Three contributions section and project QR at the bottom.
- Sparsity rankings are intentionally omitted until the manuscript's direction convention is reconciled; the source figure is retained without an added comparative claim.
`
await writeFile(path.join(deliverablesDir, 'README.md'), readme)

const sizes = await Promise.all(finalNames.map(async (name) => [name, (await stat(path.join(deliverablesDir, name))).size]))
console.log(`Packaged revision ${revision.slice(0, 12)}: ${sizes.map(([name, size]) => `${name}=${size}`).join(', ')}`)
