import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { loadExampleAssets } from './example-assets.mjs'
import { deliverablesDir, repositoryDir, withPosterPage } from './harness.mjs'

const visualSpec = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/visual-spec.json'), 'utf8'))

const exampleAssets = await loadExampleAssets()

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
      sectionRules: [...document.querySelectorAll('[data-section]')].map((element) => {
        const style = getComputedStyle(element)
        return { id: element.dataset.section, top: parseFloat(style.borderTopWidth), bottom: parseFloat(style.borderBottomWidth) }
      }),
      horizontalRuleCount: document.querySelectorAll('.poster-canvas hr').length,
      figures,
      exampleBackground: getComputedStyle(document.querySelector('.problem-block')).backgroundColor,
      examplePlots: [...document.querySelectorAll('[data-example-plot]')].map((plot) => ({
        paradigm: plot.dataset.examplePlot, viewport: rect(plot),
        renderer: plot.dataset.exampleRenderer,
        loaded: plot.complete && plot.naturalWidth > 0,
      })),
      canvasTopBorderWidth: getComputedStyle(canvas).borderTopWidth,
      headerBodyGap: document.querySelector('.poster-grid').getBoundingClientRect().top - document.querySelector('.poster-header').getBoundingClientRect().bottom,
      headerBottomPadding: parseFloat(getComputedStyle(document.querySelector('.poster-header')).paddingBottom),
      footerCount: document.querySelectorAll('.poster-footer, [data-section="reproducibility"]').length,
      architecture: {
        viewport: rect(document.querySelector('.architecture-image-window')),
        image: rect(document.querySelector('.architecture-image-window img')),
        background: getComputedStyle(document.querySelector('.protocol-block')).backgroundColor,
        outline: getComputedStyle(document.querySelector('.protocol-block')).outlineStyle,
        blend: getComputedStyle(document.querySelector('.architecture-image-window img')).mixBlendMode,
        captionCount: document.querySelectorAll('.architecture-figure figcaption').length,
      },
      visibleProvenance: [...document.querySelectorAll('[data-source-section], [data-source-citation]')].filter((element) => element.getClientRects().length > 0).length,
      columns: Object.fromEntries(['left', 'center', 'right'].map((column) => [column, [...document.querySelectorAll(`.poster-column--${column} > [data-section]`)].map((element) => ({ id: element.dataset.section, ...rect(element) }))])),
      resultPanels: [...document.querySelectorAll('[data-section="results"] article[data-section]')].map((element) => ({ id: element.dataset.section, ...rect(element) })),
      resultFrames: [...document.querySelectorAll('.result-panel')].map((element) => {
        const outline = element.querySelector('.result-panel__outline rect')
        const style = getComputedStyle(outline)
        return { background: getComputedStyle(element).backgroundColor, radius: Number(outline.getAttribute('rx')), fill: style.fill, stroke: style.stroke, strokeWidth: parseFloat(style.strokeWidth), dashArray: style.strokeDasharray.split(/[ ,]+/).map(parseFloat), outline: rect(outline), panel: rect(element) }
      }),
      regression: {
        viewport: rect(document.querySelector('[data-finding="regression"] .manuscript-image-window')),
        image: rect(document.querySelector('[data-finding="regression"] img')),
      },
      resultCaptionCount: document.querySelectorAll('.result-manuscript-figure figcaption').length,
      localMetricPanels: [...document.querySelectorAll('.local-metric-window')].map((element) => ({ metric: element.dataset.metric, crop: element.dataset.crop.split(' ').map(Number), viewport: rect(element), image: rect(element.querySelector('img')) })),
      rasterChartLabels: 'Embedded manuscript labels; reviewed visually, not measured as DOM text.',
      scopeItems: [...document.querySelectorAll('.scope-tile')].map(rect),
      scopeTileStyles: [...document.querySelectorAll('.scope-tile')].map((element) => {
        const style = getComputedStyle(element)
        const heading = getComputedStyle(element.querySelector('.scope-tile__heading'))
        const outline = element.querySelector('.scope-tile__outline rect')
        const stroke = getComputedStyle(outline)
        return {
          radius: parseFloat(style.borderTopLeftRadius),
          outline: rect(outline),
          tile: rect(element),
          strokeWidth: parseFloat(stroke.strokeWidth),
          dashArray: stroke.strokeDasharray.split(/[ ,]+/).map(parseFloat),
          background: style.backgroundColor,
          headingRadius: parseFloat(heading.borderTopLeftRadius),
          headingBorder: heading.borderTopStyle,
          headingBackground: heading.backgroundColor,
          headingColor: heading.color,
          borderColor: stroke.stroke,
        }
      }),
      centerHeadingCount: document.querySelectorAll('[data-section="protocol"] h2, [data-section="scope"] h2').length,
      scopeTopBorderWidth: getComputedStyle(document.querySelector('.scope-strip')).borderTopWidth,
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
        body: minimum('.section-copy, .scope-tile__inventory'),
        caption: minimum('.source-note, figcaption'),
        resultHeading: minimum('.result-panel h3'),
      },
    }
  })

  const failuresFound = [...failures]
  if (Math.abs(screen.headerBodyGap - 12) > 0.1 || screen.headerBottomPadding !== 0) failuresFound.push('Header-to-body spacing must be compact: 12px gap and no bottom padding')
  if (screen.footerCount !== 0) failuresFound.push('Removed reproduction footer remains')
  if (screen.horizontalRuleCount !== 0 || screen.sectionRules.some((section) => section.top !== 0 || section.bottom !== 0)) failuresFound.push('Horizontal section dividers must be absent, including header and footer rules')
  if (screen.centerHeadingCount !== 0 || screen.scopeTopBorderWidth !== '0px') failuresFound.push('Removed center-column headings or divider remain')
  if (screen.scopeTileStyles.length !== 4 || screen.scopeTileStyles.some((tile) => tile.radius < 6 || tile.headingRadius < 4 || tile.strokeWidth !== 2 || JSON.stringify(tile.dashArray) !== '[10,5]' || tile.headingBorder !== 'solid' || tile.background !== 'rgb(230, 244, 252)' || tile.headingBackground !== 'rgb(252, 232, 198)' || tile.headingColor !== tile.borderColor)) failuresFound.push('Scope tiles must match the schema with 10px dashes, 5px gaps, and a 2px outline')
  for (const tile of screen.scopeTileStyles) {
    if (Math.abs(tile.outline.width - tile.tile.width + 2) > 1 || Math.abs(tile.outline.height - tile.tile.height + 2) > 1 || Math.abs(tile.outline.left - tile.tile.left - 1) > 1 || Math.abs(tile.outline.top - tile.tile.top - 1) > 1) failuresFound.push('Scope tile outline does not follow its container bounds')
  }
  if (screen.canvasTopBorderWidth !== '0px') failuresFound.push('Removed top color line remains on the canvas')
  if (screen.architecture.background !== 'rgba(0, 0, 0, 0)' || screen.architecture.outline !== 'none' || screen.architecture.blend !== 'multiply') failuresFound.push('Architecture section must have no panel or image background')
  if (screen.architecture.captionCount !== 0) failuresFound.push('Removed architecture caption remains')
  if (Math.abs(screen.architecture.viewport.width - screen.columns.center[0].width) > 1 || screen.architecture.viewport.width < 700) failuresFound.push('Architecture schema must fill the rebalanced center column')
  const cropScale = screen.architecture.viewport.width / 1220
  if (Math.abs(screen.architecture.image.width - 1400 * cropScale) > 1 || Math.abs(screen.architecture.viewport.left - screen.architecture.image.left - 90 * cropScale) > 1 || Math.abs(screen.architecture.viewport.top - screen.architecture.image.top - 28 * cropScale) > 1 || Math.abs(screen.architecture.viewport.height - 622 * cropScale) > 1) failuresFound.push('Architecture whitespace crop differs from its source-image bounds')
  if (screen.exampleBackground !== 'rgba(0, 0, 0, 0)' && screen.exampleBackground !== 'rgb(255, 253, 248)') failuresFound.push('Example panel must use the light poster background')
  if (JSON.stringify(screen.examplePlots.map((plot) => plot.paradigm)) !== JSON.stringify(['local', 'global', 'group-wise'])) failuresFound.push('All three Matplotlib examples must be visible')
  if (new Set(screen.examplePlots.map((plot) => plot.viewport.width)).size !== 1) failuresFound.push('Example plots must share identical widths')
  if (Math.abs(screen.examplePlots[0].viewport.height - screen.examplePlots[1].viewport.height) > 1 || screen.examplePlots[2].viewport.height <= screen.examplePlots[1].viewport.height) failuresFound.push('Only group-wise must reserve the extra legend row')
  for (const plot of screen.examplePlots) {
    const asset = exampleAssets.find((candidate) => candidate.paradigm === plot.paradigm)
    if (!plot.loaded || plot.renderer !== 'matplotlib') failuresFound.push(`${plot.paradigm} SVG failed to load`)
    if (Math.abs(plot.viewport.height - plot.viewport.width * asset.height / asset.width) > 1) failuresFound.push(`${plot.paradigm} SVG is stretched`)
    if (asset.minimumFontPt * plot.viewport.width / asset.width < 12) failuresFound.push(`${plot.paradigm} SVG labels are smaller than twelve native pixels`)
  }
  if (screen.visibleProvenance !== 0) failuresFound.push('Source metadata must not appear on the poster')
  if (Math.abs(screen.canvas.width - 1800) > 0.01 || Math.abs(screen.canvas.height - 1273) > 0.01) failuresFound.push(`Native canvas is ${screen.canvas.width} × ${screen.canvas.height}`)
  if (screen.canvas.scrollWidth > screen.canvas.clientWidth || screen.canvas.scrollHeight > screen.canvas.clientHeight) failuresFound.push('Poster canvas overflows')
  if (screen.document.scrollWidth > screen.document.clientWidth) failuresFound.push('Document has horizontal overflow')
  if (screen.sections.length !== 10 || new Set(screen.sections.map((section) => section.id)).size !== 10) failuresFound.push('Named inventory must include six top-level sections and four nested result panels')
  for (const [column, expected] of Object.entries({ left: ['problem'], center: ['protocol', 'scope', 'guidance-limitations'], right: ['results'] })) {
    if (JSON.stringify(screen.columns[column].map((section) => section.id)) !== JSON.stringify(expected)) failuresFound.push(`${column} column is not in the requested order`)
    if (screen.columns[column].some((section, index, sections) => index > 0 && sections[index - 1].bottom >= section.top)) failuresFound.push(`${column} sections overlap`)
  }
  if (JSON.stringify(screen.resultPanels.map((panel) => panel.id)) !== JSON.stringify(['applicability', 'local-tradeoff', 'group-tradeoff', 'regression-tradeoff'])) failuresFound.push('Global, local, group-wise, and regression results must be inside the unified Results section')
  if (screen.resultFrames.length !== 4 || screen.resultFrames.some((frame) => frame.background !== 'rgba(0, 0, 0, 0)' || frame.radius !== 9 || frame.fill !== 'none' || frame.stroke !== 'rgb(16, 56, 76)' || frame.strokeWidth !== 2 || JSON.stringify(frame.dashArray) !== '[10,5]')) failuresFound.push('Each result category must have a transparent rounded long-dash outline')
  for (const frame of screen.resultFrames) {
    if (Math.abs(frame.outline.width - frame.panel.width + 2) > 1 || Math.abs(frame.outline.height - frame.panel.height + 2) > 1 || Math.abs(frame.outline.left - frame.panel.left - 1) > 1 || Math.abs(frame.outline.top - frame.panel.top - 1) > 1) failuresFound.push('Result outline does not follow its category bounds')
  }
  const regressionScale = screen.regression.viewport.width / 1100
  if (Math.abs(screen.regression.image.width - 1100 * regressionScale) > 1 || Math.abs(screen.regression.image.height - 655 * regressionScale) > 1 || Math.abs(screen.regression.viewport.height - 225 * regressionScale) > 1 || Math.abs(screen.regression.image.top - screen.regression.viewport.top) > 1) failuresFound.push('Regression figure must preserve the complete Concrete row without stretching')
  if (screen.resultPanels.some((panel, index) => index > 0 && screen.resultPanels[index - 1].bottom >= panel.top)) failuresFound.push('Unified result panels overlap')
  if (screen.resultCaptionCount !== 0) failuresFound.push('Removed result captions remain')
  if (JSON.stringify(screen.localMetricPanels.map((panel) => panel.metric)) !== JSON.stringify(['Validity', 'L2-Hamming', 'Sparsity', 'Log-density', 'Runtime'])) failuresFound.push('All five local metric panels must remain visible')
  for (const panel of screen.localMetricPanels) {
    const [x, y, width, height] = panel.crop
    const scale = panel.viewport.width / width
    if (panel.viewport.width < 175 || panel.viewport.height < 100) failuresFound.push(`${panel.metric} is not enlarged to a readable metric panel`)
    if (Math.abs(panel.image.width - 1400 * scale) > 1 || Math.abs(panel.image.height - 1101 * scale) > 1 || Math.abs(panel.viewport.left - panel.image.left - x * scale) > 1 || Math.abs(panel.viewport.top - panel.image.top - y * scale) > 1 || Math.abs(panel.viewport.height - height * scale) > 1) failuresFound.push(`${panel.metric} crop is stretched or has incorrect source geometry`)
  }
  if (screen.localMetricPanels.slice(0, 3).some((panel) => Math.abs(panel.viewport.top - screen.localMetricPanels[0].viewport.top) > 1) || screen.localMetricPanels.slice(3).some((panel) => Math.abs(panel.viewport.top - screen.localMetricPanels[3].viewport.top) > 1) || screen.localMetricPanels[0].viewport.bottom >= screen.localMetricPanels[3].viewport.top) failuresFound.push('Local metric panels must occupy two rows without overlap')
  if (screen.scopeItems.length !== 4 || screen.scopeItems.slice(0, 2).some((item) => Math.abs(item.top - screen.scopeItems[0].top) > 1) || screen.scopeItems.slice(2).some((item) => Math.abs(item.top - screen.scopeItems[2].top) > 1) || screen.scopeItems[0].bottom >= screen.scopeItems[2].top) failuresFound.push('Benchmark scope must be a two-by-two named tile grid')
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
    resultHeading: visualSpec.minimumPrintType.body.cssPx,
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
console.log(`Layout audit passed: sections=${report.sections.length}, body=${report.minimumFontPx.body}px, caption=${report.minimumFontPx.caption}px, result headings=${report.minimumFontPx.resultHeading}px; raster labels require visual review`)
