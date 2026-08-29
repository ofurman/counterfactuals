import { spawnSync } from 'node:child_process'
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { deliverablesDir } from './harness.mjs'

const pdfPath = path.join(deliverablesDir, 'cel-benchmark-poster-a0.pdf')
const reviewPrefix = path.join(deliverablesDir, 'cel-benchmark-poster-preview')
const info = spawnSync('pdfinfo', [pdfPath], { encoding: 'utf8' })
if (info.status !== 0) throw new Error(`pdfinfo failed:\n${info.stdout}${info.stderr}`)
const pages = Number(info.stdout.match(/^Pages:\s+(\d+)/m)?.[1])
const dimensions = info.stdout.match(/^Page size:\s+([0-9.]+) x ([0-9.]+) pts/m)
if (!dimensions) throw new Error('pdfinfo did not report page dimensions')
const widthPt = Number(dimensions[1])
const heightPt = Number(dimensions[2])
const expectedWidthPt = 1189 * 72 / 25.4
const expectedHeightPt = 841 * 72 / 25.4
if (pages !== 1) throw new Error(`Expected one page, found ${pages}`)
if (Math.abs(widthPt - expectedWidthPt) > 1 || Math.abs(heightPt - expectedHeightPt) > 1 || widthPt <= heightPt) throw new Error(`Wrong PDF page size: ${widthPt} × ${heightPt} pt`)
const render = spawnSync('pdftoppm', ['-png', '-singlefile', '-r', '72', pdfPath, reviewPrefix], { encoding: 'utf8' })
if (render.status !== 0) throw new Error(`pdftoppm failed:\n${render.stdout}${render.stderr}`)
const reviewPath = `${reviewPrefix}.png`
const review = await stat(reviewPath)
if (review.size === 0) throw new Error('PDF-derived review PNG is empty')
const text = spawnSync('pdftotext', [pdfPath, '-'], { encoding: 'utf8' })
if (text.status !== 0) throw new Error(`pdftotext failed:\n${text.stdout}${text.stderr}`)
const normalizedText = text.stdout.replace(/\s+/g, ' ').trim()
const requiredSectionText = [
  'CEL: controlled comparisons, visible trade-offs',
  'One broad benchmark',
  'Protocol differences can confound comparisons',
  'CEL standardizes a shared protocol',
  'Plausibility can cost more change',
  'Shared shifts trade reach for size',
  'Similar targets, different perturbations',
  'Missing output is part of the evidence',
  'Choose by task, metric, and applicability',
  'Reproduce and extend',
]
for (const expected of requiredSectionText) if (!normalizedText.includes(expected)) throw new Error(`PDF text is missing required section content: ${expected}`)
const pngSignature = await readFile(reviewPath).then((bytes) => bytes.subarray(0, 8).toString('hex'))
if (pngSignature !== '89504e470d0a1a0a') throw new Error('PDF-derived preview is not a PNG')
console.log(info.stdout.trim())
console.log(`PDF audit passed: pages=${pages}, size=${widthPt} × ${heightPt} pt, preview=${review.size} bytes, required sections=${requiredSectionText.length}`)
