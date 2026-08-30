import { spawnSync } from 'node:child_process'
import { readdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { projectDir } from './harness.mjs'

const parcelCache = path.join(projectDir, '.parcel-cache')
await rm(parcelCache, { recursive: true, force: true })

const skillScript = '/Users/ofurman/.agents/skills/web-artifacts-builder/scripts/bundle-artifact.sh'
const bundle = spawnSync('bash', [skillScript], { cwd: projectDir, encoding: 'utf8' })
process.stdout.write(bundle.stdout)
process.stderr.write(bundle.stderr)
if (bundle.status !== 0) throw new Error(`Named artifact bundler failed with status ${bundle.status}`)

const bundlePath = path.join(projectDir, 'bundle.html')
const distDir = path.join(projectDir, 'dist')
let html = await readFile(bundlePath, 'utf8')
const imageFiles = (await readdir(distDir)).filter((file) => /\.(?:jpe?g|png|svg)$/i.test(file)).sort()
if (imageFiles.length !== 9) throw new Error(`Expected four manuscript graphics and five logos from Parcel, found ${imageFiles.length}`)

for (const file of imageFiles) {
  const bytes = await readFile(path.join(distDir, file))
  const extension = path.extname(file).toLowerCase()
  const mime = extension === '.svg' ? 'image/svg+xml' : extension === '.png' ? 'image/png' : 'image/jpeg'
  const dataUrl = `data:${mime};base64,${bytes.toString('base64')}`
  const occurrences = html.split(file).length - 1
  if (occurrences === 0) throw new Error(`Bundle does not reference emitted image asset: ${file}`)
  // Parcel emits root-relative image paths. Replace the complete path token so
  // browsers do not reinterpret the embedded data URL as `/data:image/...`.
  html = html.replaceAll(`./${file}`, dataUrl)
  html = html.replaceAll(`/${file}`, dataUrl)
  html = html.replaceAll(file, dataUrl)
}

for (const file of imageFiles) {
  if (html.includes(file)) throw new Error(`Emitted image reference remains after inlining: ${file}`)
}

await writeFile(bundlePath, html)
const bundled = await stat(bundlePath)
console.log(`Inlined ${imageFiles.length} image assets; self-contained bundle=${bundled.size} bytes`)
