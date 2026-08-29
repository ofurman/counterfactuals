import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { deliverablesDir } from './harness.mjs'

const sumsPath = path.join(deliverablesDir, 'SHA256SUMS')
const sums = (await readFile(sumsPath, 'utf8')).trim().split('\n')
const expectedNames = ['cel-benchmark-poster.html', 'cel-benchmark-poster-a0.pdf', 'cel-benchmark-poster-preview.png']
if (sums.length !== expectedNames.length) throw new Error(`Checksum manifest has ${sums.length} entries; expected ${expectedNames.length}`)
for (const [index, line] of sums.entries()) {
  const match = line.match(/^([0-9a-f]{64})  (.+)$/)
  if (!match || match[2] !== expectedNames[index]) throw new Error(`Invalid checksum entry: ${line}`)
  const bytes = await readFile(path.join(deliverablesDir, match[2]))
  const actual = createHash('sha256').update(bytes).digest('hex')
  if (actual !== match[1]) throw new Error(`Checksum mismatch: ${match[2]}`)
}

const readme = await readFile(path.join(deliverablesDir, 'README.md'), 'utf8')
for (const required of ['1189 × 841 mm', 'Print', 'Source revision', 'pnpm run verify:all', 'Known limitations', ...expectedNames]) if (!readme.includes(required)) throw new Error(`Deliverables README is missing: ${required}`)

const temporaryDir = await mkdtemp(path.join(os.tmpdir(), 'cel-poster-audit-'))
try {
  const prefix = path.join(temporaryDir, 'preview')
  const pdfPath = path.join(deliverablesDir, 'cel-benchmark-poster-a0.pdf')
  const render = spawnSync('pdftoppm', ['-png', '-singlefile', '-r', '72', pdfPath, prefix], { encoding: 'utf8' })
  if (render.status !== 0) throw new Error(`Could not reproduce PDF preview:\n${render.stdout}${render.stderr}`)
  const [reproduced, packaged] = await Promise.all([
    readFile(`${prefix}.png`),
    readFile(path.join(deliverablesDir, 'cel-benchmark-poster-preview.png')),
  ])
  const reproducedHash = createHash('sha256').update(reproduced).digest('hex')
  const packagedHash = createHash('sha256').update(packaged).digest('hex')
  if (reproducedHash !== packagedHash) throw new Error('Packaged preview is not the current PDF-derived PNG')
} finally {
  await rm(temporaryDir, { recursive: true, force: true })
}

const sizes = await Promise.all(expectedNames.map(async (name) => (await stat(path.join(deliverablesDir, name))).size))
console.log(`Package audit passed: checksums=${sums.length}, PDF-derived preview verified, bytes=${sizes.join('/')}`)
