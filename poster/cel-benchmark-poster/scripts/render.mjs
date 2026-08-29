import { mkdir, stat } from 'node:fs/promises'
import path from 'node:path'
import { deliverablesDir, withPosterPage } from './harness.mjs'

const previewPath = path.join(deliverablesDir, 'cel-benchmark-poster-preview.png')
const pdfPath = path.join(deliverablesDir, 'cel-benchmark-poster-a0.pdf')
await mkdir(deliverablesDir, { recursive: true })

await withPosterPage(async ({ page, failures }) => {
  const canvas = page.locator('.poster-canvas')
  await canvas.screenshot({ path: previewPath, animations: 'disabled' })
  await page.emulateMedia({ media: 'print' })
  const toolbarDisplay = await page.locator('.print-toolbar').evaluate((element) => getComputedStyle(element).display)
  if (toolbarDisplay !== 'none') throw new Error(`Print toolbar is visible: ${toolbarDisplay}`)
  await page.pdf({
    path: pdfPath,
    preferCSSPageSize: true,
    printBackground: true,
    tagged: true,
    outline: true,
  })
  if (failures.length) throw new Error(`Render requests failed:\n${failures.join('\n')}`)
})

const [preview, pdf] = await Promise.all([stat(previewPath), stat(pdfPath)])
if (preview.size === 0 || pdf.size === 0) throw new Error('Render produced an empty artifact')
console.log(`Rendered ${path.relative(deliverablesDir, previewPath)} (${preview.size} bytes)`)
console.log(`Rendered ${path.relative(deliverablesDir, pdfPath)} (${pdf.size} bytes)`)
