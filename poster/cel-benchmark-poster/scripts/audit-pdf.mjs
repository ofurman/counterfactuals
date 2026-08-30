import { spawnSync } from 'node:child_process'
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { deliverablesDir, repositoryDir } from './harness.mjs'

const pdfPath = path.join(deliverablesDir, 'cel-benchmark-poster-a1.pdf')
const reviewPrefix = path.join(deliverablesDir, 'cel-benchmark-poster-preview')
const info = spawnSync('pdfinfo', [pdfPath], { encoding: 'utf8' })
if (info.status !== 0) throw new Error(`pdfinfo failed:\n${info.stdout}${info.stderr}`)
const pages = Number(info.stdout.match(/^Pages:\s+(\d+)/m)?.[1])
const dimensions = info.stdout.match(/^Page size:\s+([0-9.]+) x ([0-9.]+) pts/m)
if (!dimensions) throw new Error('pdfinfo did not report page dimensions')
const widthPt = Number(dimensions[1])
const heightPt = Number(dimensions[2])
const expectedWidthPt = 594 * 72 / 25.4
const expectedHeightPt = 841 * 72 / 25.4
if (pages !== 1) throw new Error(`Expected one page, found ${pages}`)
if (Math.abs(widthPt - expectedWidthPt) > 1 || Math.abs(heightPt - expectedHeightPt) > 1 || widthPt >= heightPt) throw new Error(`Wrong A1 portrait PDF page size: ${widthPt} × ${heightPt} pt`)
const render = spawnSync('pdftoppm', ['-png', '-singlefile', '-r', '72', pdfPath, reviewPrefix], { encoding: 'utf8' })
if (render.status !== 0) throw new Error(`pdftoppm failed:\n${render.stdout}${render.stderr}`)
const reviewPath = `${reviewPrefix}.png`
const review = await stat(reviewPath)
if (review.size === 0) throw new Error('PDF-derived review PNG is empty')
const text = spawnSync('pdftotext', [pdfPath, '-'], { encoding: 'utf8' })
if (text.status !== 0) throw new Error(`pdftotext failed:\n${text.stdout}${text.stderr}`)
const normalizedText = text.stdout.replace(/\s+/g, ' ').trim()
const content = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/poster-content.json'), 'utf8'))
const identity = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/identity.json'), 'utf8'))
const requiredSectionText = content.sections.filter((section) => section.showHeading !== false).map((section) => section.heading)
if (requiredSectionText[0] !== identity.title) throw new Error('PDF title contract differs from the manuscript')
for (const expected of requiredSectionText) if (!normalizedText.includes(expected)) throw new Error(`PDF text is missing required section content: ${expected}`)
if (normalizedText.includes('One protocol. Multiple CE paradigms. Measurable trade-offs.')) throw new Error('Removed header subtitle remains in the PDF')
if (normalizedText.includes(identity.venue)) throw new Error('Removed venue marker remains in the PDF')
if (/Reproduce and extend|uv add ce-library|Documentation/.test(normalizedText)) throw new Error('Removed footer text remains in the PDF')
if (/Code & project|Three contributions/.test(normalizedText)) throw new Error('Removed QR caption or old Contributions heading remains')
if (!normalizedText.includes('Counterfactual Explanations Metrics') || normalizedText.includes('Classification Metrics')) throw new Error('Metrics tile must use the updated Counterfactual Explanations Metrics label')
if (/shared evaluation protocol/i.test(normalizedText)) throw new Error('Removed architecture caption remains in the PDF')
if (/One evaluation framework|Benchmark scope/i.test(normalizedText)) throw new Error('Removed center-column heading remains in the PDF')
if (/Adult Census\s*·\s*(?:global|local|group-wise) methods/i.test(normalizedText)) throw new Error('Removed result caption remains in the PDF')
for (const expected of ['Loan application example', 'Employment', 'Monthly income', 'Debt payments', 'Original A', 'Counterfactual', 'Declined', 'Approved', 'One applicant', 'One shared change', 'One change per group', 'Income +€1,300', 'Debt −€1,000']) if (!normalizedText.toLowerCase().includes(expected.toLowerCase())) throw new Error(`PDF is missing the applicant comparison: ${expected}`)
if (/\bSource\s*[·:]|\btoy\b|not benchmark data|Sparsity direction unresolved|Benchmark context/i.test(normalizedText)) throw new Error('Removed source or disclaimer text remains in the PDF')
if (!normalizedText.includes('Regression: accuracy versus change')) throw new Error('Requested regression result heading is missing from the PDF')
if (/\b5\s+folds\b|\b3\s+paradigms\b/i.test(normalizedText)) throw new Error('Removed folds or paradigms tiles remain in the PDF')
const scopeClaims = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/claims/claims.generated.json'), 'utf8')).claims;
for (const claim of scopeClaims.filter((claim) => claim.id.startsWith('contribution.'))) {
  for (const wording of [claim.posterWording, claim.posterDetail]) if (!wording || !normalizedText.includes(wording)) throw new Error(`PDF is missing contribution copy: ${wording}`)
}
for (const claim of scopeClaims.filter((claim) => claim.inventory)) {
  for (const name of claim.inventory.flatMap((group) => group.names)) if (!normalizedText.includes(name)) throw new Error(`PDF is missing named scope item: ${name}`)
}
const pngSignature = await readFile(reviewPath).then((bytes) => bytes.subarray(0, 8).toString('hex'))
if (pngSignature !== '89504e470d0a1a0a') throw new Error('PDF-derived preview is not a PNG')
console.log(info.stdout.trim())
console.log(`PDF audit passed: pages=${pages}, size=${widthPt} × ${heightPt} pt, preview=${review.size} bytes, required sections=${requiredSectionText.length}`)
