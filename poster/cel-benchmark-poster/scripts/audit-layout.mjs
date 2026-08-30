import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { deliverablesDir, repositoryDir, withPosterPage } from './harness.mjs'

const visualSpec = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/visual-spec.json'), 'utf8'))

const report = await withPosterPage(async ({ page, failures }) => {
  const screen = await page.evaluate(() => {
    const canvas = document.querySelector('.poster-canvas')
    if (!canvas) throw new Error('Poster canvas is missing')
    const canvasRect = canvas.getBoundingClientRect()
    const styles = getComputedStyle(document.documentElement)
    const numberToken = (name) => Number.parseFloat(styles.getPropertyValue(name))
    const safe = {
      top: numberToken('--safe-top'),
      right: numberToken('--safe-right'),
      bottom: numberToken('--safe-bottom'),
      left: numberToken('--safe-left'),
    }
    const safeBounds = {
      top: canvasRect.top + canvas.clientTop + safe.top,
      right: canvasRect.right - canvas.clientLeft - safe.right,
      bottom: canvasRect.bottom - safe.bottom,
      left: canvasRect.left + canvas.clientLeft + safe.left,
    }
    const rect = (element) => {
      const value = element.getBoundingClientRect()
      return { top: value.top, right: value.right, bottom: value.bottom, left: value.left, width: value.width, height: value.height }
    }
    const sections = [...document.querySelectorAll('[data-section]')].map((element) => ({
      id: element.getAttribute('data-section'),
      rect: rect(element),
      clientWidth: element.clientWidth,
      scrollWidth: element.scrollWidth,
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
    }))
    const figures = [...document.querySelectorAll('figure')].map((element) => ({
      clientWidth: element.clientWidth,
      scrollWidth: element.scrollWidth,
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
    }))
    const fontSizes = (selector) => [...document.querySelectorAll(selector)]
      .filter((element) => element.getClientRects().length > 0)
      .map((element) => Number.parseFloat(getComputedStyle(element).fontSize))
      .filter((value) => Number.isFinite(value) && value > 0)
    const minimum = (selector) => {
      const values = fontSizes(selector)
      return values.length ? Math.min(...values) : null
    }
    return {
      canvas: { ...rect(canvas), clientWidth: canvas.clientWidth, scrollWidth: canvas.scrollWidth, clientHeight: canvas.clientHeight, scrollHeight: canvas.scrollHeight },
      document: {
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
        clientHeight: document.documentElement.clientHeight,
        scrollHeight: document.documentElement.scrollHeight,
      },
      sections,
      figures,
      exampleBackground: getComputedStyle(document.querySelector('.problem-block')).backgroundColor,
      canvasTopBorderWidth: getComputedStyle(canvas).borderTopWidth,
      visibleProvenance: [...document.querySelectorAll('[data-source-section], [data-source-citation]')].filter((element) => element.getClientRects().length > 0).length,
      columns: Object.fromEntries(['left', 'center', 'right'].map((column) => [column, [...document.querySelectorAll(`.poster-column--${column} > [data-section]`)].map((element) => ({ id: element.dataset.section, ...rect(element) }))])),
      branding: {
        header: rect(document.querySelector('.poster-header')),
        title: rect(document.querySelector('.header-copy h1')),
        titleAlign: getComputedStyle(document.querySelector('.header-copy h1')).textAlign,
        logos: [...document.querySelectorAll('[data-brand-id]')].map((element) => ({ id: element.dataset.brandId, ...rect(element), objectFit: getComputedStyle(element).objectFit })),
        identityText: [...document.querySelectorAll('.header-copy h1, .authors, .affiliation')].map((element) => {
          const range = document.createRange()
          range.selectNodeContents(element)
          return rect(range)
        }),
      },
      safeBounds,
      images: [...document.images].map((image) => ({ src: image.currentSrc || image.src, complete: image.complete, naturalWidth: image.naturalWidth })),
      fonts: { status: document.fonts.status },
      minimumFontPx: {
        body: minimum('.section-copy'),
        caption: minimum('.source-note, figcaption'),
        chartLabel: minimum('.manuscript-figure figcaption'),
      },
    }
  })

  const failuresFound = [...failures]
  if (screen.canvasTopBorderWidth !== '0px') failuresFound.push('Removed top color line remains on the canvas')
  if (screen.exampleBackground !== 'rgba(0, 0, 0, 0)' && screen.exampleBackground !== 'rgb(255, 253, 248)') failuresFound.push('Example panel must use the light poster background')
  if (screen.visibleProvenance !== 0) failuresFound.push('Source metadata must not appear on the poster')
  if (Math.abs(screen.canvas.width - 1800) > 0.01 || Math.abs(screen.canvas.height - 1273) > 0.01) failuresFound.push(`Native canvas is ${screen.canvas.width} × ${screen.canvas.height}`)
  if (screen.canvas.scrollWidth > screen.canvas.clientWidth || screen.canvas.scrollHeight > screen.canvas.clientHeight) failuresFound.push('Poster canvas overflows')
  if (screen.document.scrollWidth > screen.document.clientWidth) failuresFound.push('Document has horizontal overflow')
  if (screen.sections.length !== 9 || new Set(screen.sections.map((section) => section.id)).size !== 9) failuresFound.push('Named section inventory is not exactly nine unique sections')
  for (const [column, expected] of Object.entries({ left: ['problem', 'applicability'], center: ['protocol', 'local-tradeoff'], right: ['guidance-limitations', 'group-tradeoff'] })) {
    if (JSON.stringify(screen.columns[column].map((section) => section.id)) !== JSON.stringify(expected)) failuresFound.push(`${column} column is not in the requested order`)
    if (Math.abs(screen.columns[column][1]?.top - screen.columns.center[1]?.top) > 1) failuresFound.push(`${column} bottom result is not aligned with the center result`)
  }
  const overlaps = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top
  for (const logo of screen.branding.logos) {
    if (logo.width <= 0 || logo.height <= 0 || logo.objectFit !== 'contain') failuresFound.push(`${logo.id} logo is missing or distorted`)
    if (logo.left < screen.branding.header.left || logo.right > screen.branding.header.right || logo.top < screen.branding.header.top || logo.bottom > screen.branding.header.bottom) failuresFound.push(`${logo.id} logo exceeds the header`)
  }
  if (screen.branding.identityText.some((text) => screen.branding.logos.some((logo) => overlaps(logo, text)))) failuresFound.push('Brand logos overlap header text')
  const centerX = (rect) => (rect.left + rect.right) / 2
  if (screen.branding.titleAlign !== 'center' || Math.abs(centerX(screen.branding.title) - centerX(screen.branding.header)) > 1) failuresFound.push('Manuscript title is not centered in the header')
  for (const [side, ids] of Object.entries({ left: ['xkdd', 'ecml-pkdd'], right: ['pwr', 'genwro', 'tooploox'] })) {
    const logos = ids.map((id) => screen.branding.logos.find((logo) => logo.id === id))
    if (logos.some((logo) => !logo)) {
      failuresFound.push(`Missing ${side} header logo`)
      continue
    }
    for (const [index, logo] of logos.entries()) {
      if (side === 'left' ? logo.right > screen.branding.title.left : logo.left < screen.branding.title.right) failuresFound.push(`${logo.id} is not on the ${side} of the title`)
      if (index > 0 && (logos[index - 1].bottom >= logo.top || Math.abs(centerX(logos[0]) - centerX(logo)) > 1)) failuresFound.push(`${side} logos must be vertically stacked in order: ${ids.join(', ')}`)
    }
  }
  for (const section of screen.sections) {
    if (section.scrollWidth > section.clientWidth + 1 || section.scrollHeight > section.clientHeight + 1) failuresFound.push(`${section.id} has clipped or overflowing content`)
    if (section.rect.left < screen.safeBounds.left - 1 || section.rect.right > screen.safeBounds.right + 1 || section.rect.top < screen.safeBounds.top - 1 || section.rect.bottom > screen.safeBounds.bottom + 1) failuresFound.push(`${section.id} breaches the safe area`)
  }
  for (const [index, figure] of screen.figures.entries()) {
    if (figure.scrollWidth > figure.clientWidth + 1 || figure.scrollHeight > figure.clientHeight + 1) failuresFound.push(`Figure ${index + 1} is clipped`)
  }
  if (screen.images.some((image) => !image.complete || image.naturalWidth === 0)) failuresFound.push('Poster has an unresolved image')
  if (screen.fonts.status !== 'loaded') failuresFound.push(`Fonts are not loaded: ${screen.fonts.status}`)
  for (const [role, value] of Object.entries(screen.minimumFontPx)) if (!value) failuresFound.push(`${role} font size is NOT MEASURED`)
  const requiredFontPx = {
    body: visualSpec.minimumPrintType.body.cssPx,
    caption: visualSpec.minimumPrintType.citation.cssPx,
    chartLabel: visualSpec.minimumPrintType.chartLabel.cssPx,
  }
  for (const [role, minimumPx] of Object.entries(requiredFontPx)) {
    if (screen.minimumFontPx[role] && screen.minimumFontPx[role] + 0.01 < minimumPx) failuresFound.push(`${role} font is ${screen.minimumFontPx[role]}px; minimum is ${minimumPx}px`)
  }

  await page.emulateMedia({ media: 'print' })
  const print = await page.evaluate(() => ({
    toolbarDisplay: getComputedStyle(document.querySelector('.print-toolbar')).display,
    colorAdjust: getComputedStyle(document.body).printColorAdjust,
  }))
  if (print.toolbarDisplay !== 'none') failuresFound.push(`Toolbar is visible in print: ${print.toolbarDisplay}`)
  if (print.colorAdjust !== 'exact') failuresFound.push(`Print color adjustment is ${print.colorAdjust}`)
  if (failuresFound.length) throw new Error(failuresFound.join('\n'))
  return { ...screen, print }
})

await mkdir(deliverablesDir, { recursive: true })
const reportPath = path.join(deliverablesDir, 'audit-layout.json')
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`)
console.log(`Layout audit passed: sections=${report.sections.length}, body=${report.minimumFontPx.body}px, caption=${report.minimumFontPx.caption}px, chart=${report.minimumFontPx.chartLabel}px`)
