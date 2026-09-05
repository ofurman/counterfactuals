import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { loadExampleAssets } from './example-assets.mjs'
import { loadManuscriptAssets, auditManuscriptLabelBounds } from './manuscript-assets.mjs'
import { deliverablesDir, repositoryDir, withPosterPage } from './harness.mjs'

const visualSpec = JSON.parse(await readFile(path.join(repositoryDir, 'poster/research/visual-spec.json'), 'utf8'))

const exampleAssets = await loadExampleAssets()
const manuscriptAssets = await loadManuscriptAssets()

const report = await withPosterPage(async ({ page, failures, url }) => {
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
      layoutContainers: [...document.querySelectorAll('.poster-grid, .poster-column, .scope-tile, .contribution-item')].map((element) => ({ name: element.className, clientWidth: element.clientWidth, scrollWidth: element.scrollWidth, clientHeight: element.clientHeight, scrollHeight: element.scrollHeight })),
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
        svgCount: document.querySelectorAll('.architecture-image-window svg').length,
        background: getComputedStyle(document.querySelector('.protocol-block')).backgroundColor,
        outline: getComputedStyle(document.querySelector('.protocol-block')).outlineStyle,
        captionCount: document.querySelectorAll('.architecture-figure figcaption').length,
      },
      visibleProvenance: [...document.querySelectorAll('[data-source-section], [data-source-citation]')].filter((element) => element.getClientRects().length > 0).length,
      columns: Object.fromEntries(['top', 'examples', 'right', 'bottom'].map((column) => [column, [...document.querySelectorAll(`.poster-column--${column} > [data-section]`)].map((element) => ({ id: element.dataset.section, ...rect(element) }))])),
      resultPanels: [...document.querySelectorAll('[data-section="results"] article[data-section]')].map((element) => ({ id: element.dataset.section, ...rect(element) })),
      contributionHeadings: [...document.querySelectorAll('.contribution-item h3')].map(rect),
      contributionDetails: [...document.querySelectorAll('.contribution-item p')].map(rect),
      contributionTitle: rect(document.querySelector('.contributions-block > h2')),
      contributionItems: [...document.querySelectorAll('.contribution-item')].map((element) => ({ ...rect(element), background: getComputedStyle(element).backgroundColor, borders: ['Top', 'Right', 'Bottom', 'Left'].map((side) => parseFloat(getComputedStyle(element)[`border${side}Width`])) })),
      contributionNumbers: [...document.querySelectorAll('.contribution-number')].map((element) => ({text: element.textContent, color: getComputedStyle(element).color})),
      contributionQr: rect(document.querySelector('.contribution-item .project-mark')),
      qrBranding: {
        symbol: rect(document.querySelector('.project-mark__qr svg')),
        logo: rect(document.querySelector('.project-mark__qr image')),
        logoSource: document.querySelector('.project-mark__qr image').getAttribute('href'),
        errorLevel: document.querySelector('.project-mark__qr svg').dataset.qrErrorLevel,
        margin: document.querySelector('.project-mark__qr svg').dataset.qrMargin,
        visibleText: document.querySelector('.project-mark').textContent.trim(),
      },
      resultFrames: [...document.querySelectorAll('.result-panel')].map((element) => {
        const outline = element.querySelector('.result-panel__outline rect')
        const style = getComputedStyle(outline)
        const panelStyle = getComputedStyle(element)
        return { background: panelStyle.backgroundColor, padding: ['Top', 'Right', 'Bottom', 'Left'].map((side) => parseFloat(panelStyle[`padding${side}`])), radius: Number(outline.getAttribute('rx')), fill: style.fill, stroke: style.stroke, strokeWidth: parseFloat(style.strokeWidth), dashArray: style.strokeDasharray.split(/[ ,]+/).map(parseFloat), outline: rect(outline), panel: rect(element) }
      }),
      regression: {
        viewport: rect(document.querySelector('[data-finding="regression"] .manuscript-image-window')),
        image: rect(document.querySelector('[data-finding="regression"] img')),
      },
      resultCaptionCount: document.querySelectorAll('.result-manuscript-figure figcaption').length,
      manuscriptAssets: [...document.querySelectorAll('[data-typography-asset]')].map((element) => { const img = element.querySelector('img'); return {kind: element.dataset.typographyAsset, image: img ? rect(img) : null} }),
      typography: {
        pointsPerPixel: numberToken('--print-scale') * 0.75,
        titlePt: parseFloat(getComputedStyle(document.querySelector('h1')).fontSize) * numberToken('--print-scale') * 0.75,
        resultsHeadingPt: parseFloat(getComputedStyle(document.querySelector('.unified-results-block > h2')).fontSize) * numberToken('--print-scale') * 0.75,
        resultsHeadingWeight: getComputedStyle(document.querySelector('.unified-results-block > h2')).fontWeight,
        contributionHeadingPt: fontSizes('.contribution-item h3').map((size) => size * numberToken('--print-scale') * 0.75),
        contributionHeadingStyles: [...document.querySelectorAll('.contribution-item h3')].map((element) => ({ family: getComputedStyle(element).fontFamily, weight: getComputedStyle(element).fontWeight })),
        sectionHeadingStyles: [...document.querySelectorAll('.unified-results-block > h2, .contributions-block > h2')].map((element) => { const style = getComputedStyle(element); return { family: style.fontFamily, weight: style.fontWeight, size: style.fontSize, lineHeight: style.lineHeight, color: style.color } }),
        contributionDetailPt: fontSizes('.contribution-item p').map((size) => size * numberToken('--print-scale') * 0.75),
        subheadingPt: fontSizes('.recourse-panel h3, .result-panel h3').map((size) => size * numberToken('--print-scale') * 0.75),
        bodyFamily: getComputedStyle(document.querySelector('.section-copy')).fontFamily,
      },
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
  if (screen.architecture.background !== 'rgba(0, 0, 0, 0)' || screen.architecture.outline !== 'none' || screen.architecture.svgCount !== 1) failuresFound.push('Architecture section must render a single native schematic with no panel background')
  if (screen.architecture.captionCount !== 0) failuresFound.push('Removed architecture caption remains')
  const archCenterX = (screen.architecture.viewport.left + screen.architecture.viewport.right) / 2
  const canvasCenterX = (screen.canvas.left + screen.canvas.right) / 2
  if (Math.abs(archCenterX - canvasCenterX) > 1 || screen.architecture.viewport.width < 560) failuresFound.push('Architecture schema must be centred at the top of the page')
  if (Math.abs(screen.typography.titlePt - 80) > 0.05 || screen.typography.subheadingPt.some((size) => Math.abs(size - 28) > 0.05) || screen.typography.bodyFamily !== 'Arial, sans-serif') failuresFound.push('Typography must use an 80pt title, 28pt subheadings, and Arial body')
  if (Math.abs(screen.typography.resultsHeadingPt - 32) > 0.05 || screen.typography.resultsHeadingWeight !== '700') failuresFound.push('Results heading must have stronger 32pt bold hierarchy')
  for (const asset of manuscriptAssets) {
    // The architecture is now a poster-native vector schematic, not an <img> crop.
    if (asset.kind === 'architecture') continue
    const rendered = screen.manuscriptAssets.find((item) => item.kind === asset.kind)
    if (!rendered || !rendered.image || Math.abs(rendered.image.height - rendered.image.width * asset.height / asset.width) > 1) failuresFound.push(`${asset.kind} typography asset is missing or stretched`)
    const printSize = asset.minimumFontSize * rendered.image.width / asset.width * screen.typography.pointsPerPixel
    if (printSize < 17) failuresFound.push(`${asset.kind} labels are below 17pt at A1: ${printSize}`)
  }
  if (screen.exampleBackground !== 'rgba(0, 0, 0, 0)' && screen.exampleBackground !== 'rgb(255, 253, 248)') failuresFound.push('Example panel must use the light poster background')
  if (JSON.stringify(screen.examplePlots.map((plot) => plot.paradigm)) !== JSON.stringify(['local', 'global', 'group-wise'])) failuresFound.push('All three Matplotlib examples must be visible')
  if (new Set(screen.examplePlots.map((plot) => plot.viewport.width)).size !== 1) failuresFound.push('Example plots must share identical widths')
  if (Math.abs(screen.examplePlots[0].viewport.height - screen.examplePlots[1].viewport.height) > 1 || screen.examplePlots[2].viewport.height <= screen.examplePlots[1].viewport.height) failuresFound.push('Only group-wise must reserve the extra legend row')
  for (const plot of screen.examplePlots) {
    const asset = exampleAssets.find((candidate) => candidate.paradigm === plot.paradigm)
    if (!plot.loaded || plot.renderer !== 'matplotlib') failuresFound.push(`${plot.paradigm} SVG failed to load`)
    if (Math.abs(plot.viewport.height - plot.viewport.width * asset.height / asset.width) > 1) failuresFound.push(`${plot.paradigm} SVG is stretched`)
    if (asset.minimumFontPt * plot.viewport.width / asset.width * screen.typography.pointsPerPixel < 17) failuresFound.push(`${plot.paradigm} SVG labels are smaller than 17pt at A1`)
  }
  if (screen.visibleProvenance !== 0) failuresFound.push('Source metadata must not appear on the poster')
  if (Math.abs(screen.canvas.width - visualSpec.canvas.widthPx) > 0.02 || Math.abs(screen.canvas.height - visualSpec.canvas.heightPx) > 0.02) failuresFound.push(`Native canvas is ${screen.canvas.width} × ${screen.canvas.height}`)
  if (screen.canvas.scrollWidth > screen.canvas.clientWidth || screen.canvas.scrollHeight > screen.canvas.clientHeight) failuresFound.push('Poster canvas overflows')
  for (const container of screen.layoutContainers) if (container.scrollWidth > container.clientWidth + 1 || container.scrollHeight > container.clientHeight + 1) failuresFound.push(`${container.name} overflows or clips content`)
  if (screen.document.scrollWidth > screen.document.clientWidth) failuresFound.push('Document has horizontal overflow')
  if (screen.sections.length !== 10 || new Set(screen.sections.map((section) => section.id)).size !== 10) failuresFound.push('Named inventory must include six top-level sections and four nested result panels')
  for (const [column, expected] of Object.entries({ top: ['scope'], examples: ['problem'], right: ['results'], bottom: ['guidance-limitations'] })) {
    if (JSON.stringify(screen.columns[column].map((section) => section.id)) !== JSON.stringify(expected)) failuresFound.push(`${column} column is not in the requested order`)
    if (screen.columns[column].some((section, index, sections) => index > 0 && sections[index - 1].bottom >= section.top)) failuresFound.push(`${column} sections overlap`)
  }
  if (JSON.stringify(screen.resultPanels.map((panel) => panel.id)) !== JSON.stringify(['applicability', 'local-tradeoff', 'group-tradeoff', 'regression-tradeoff'])) failuresFound.push('Global, local, group-wise, and regression results must be inside the unified Results section')
  if (screen.resultFrames.length !== 4 || screen.resultFrames.some((frame) => frame.background !== 'rgba(0, 0, 0, 0)' || frame.radius !== 9 || frame.fill !== 'none' || frame.stroke !== 'rgb(16, 56, 76)' || frame.strokeWidth !== 2 || JSON.stringify(frame.dashArray) !== '[10,5]')) failuresFound.push('Each result category must have a transparent rounded long-dash outline')
  for (const frame of screen.resultFrames) {
    if (frame.padding.some((padding) => padding < 12)) failuresFound.push('Result frames need at least 12px of internal padding')
    if (Math.abs(frame.outline.width - frame.panel.width + 2) > 1 || Math.abs(frame.outline.height - frame.panel.height + 2) > 1 || Math.abs(frame.outline.left - frame.panel.left - 1) > 1 || Math.abs(frame.outline.top - frame.panel.top - 1) > 1) failuresFound.push('Result outline does not follow its category bounds')
  }
  const results = screen.columns.right[0]
  const contributions = screen.columns.bottom[0]
  if (screen.contributionHeadings.length !== 3 || screen.contributionHeadings.some((heading) => Math.abs(heading.top - screen.contributionHeadings[0].top) > 1) || screen.contributionHeadings[2].right >= screen.contributionQr.left) failuresFound.push('Contribution statements must align at the top with a separate QR column')
  if (screen.contributionItems.some((item) => item.background !== 'rgba(0, 0, 0, 0)' || item.borders.some((width) => width !== 0) || Math.abs(item.width - screen.contributionItems[0].width) > 1)) failuresFound.push('Contributions must be equal-width unboxed messages without colored bars')
  if (screen.contributionTitle.bottom >= screen.contributionHeadings[0].top || Math.abs(screen.contributionQr.top - contributions.top) > 1 || screen.contributionQr.bottom > contributions.bottom + 1) failuresFound.push('Contribution heading must sit above the messages, with the QR anchored to the whole strip')
  if (screen.contributionDetails.length !== 3 || screen.contributionDetails.some((detail, index) => detail.top < screen.contributionHeadings[index].bottom || detail.bottom > contributions.bottom + 1 || Math.abs(detail.top - screen.contributionDetails[0].top) > 1)) failuresFound.push('Contribution supporting lines must align below their headings and stay inside the strip')
  if (JSON.stringify(screen.contributionNumbers.map((number) => number.text)) !== '["01","02","03"]' || screen.contributionNumbers.some((number) => number.color !== 'rgb(8, 127, 120)')) failuresFound.push('Contributions must use small ordered teal numerals')
  if (screen.typography.contributionHeadingPt.some((size) => Math.abs(size - 28) > 0.05) || screen.typography.contributionHeadingStyles.some((style) => !style.family.startsWith('Georgia') || style.weight !== '700') || screen.typography.contributionDetailPt.some((size) => Math.abs(size - 18) > 0.05)) failuresFound.push('Contribution headings must match result titles at 28pt Georgia, but bold; supporting lines stay 18pt')
  if (JSON.stringify(screen.typography.sectionHeadingStyles[0]) !== JSON.stringify(screen.typography.sectionHeadingStyles[1])) failuresFound.push('Contributions and Results section headings must have identical typography')
  const qr = screen.qrBranding
  const localGithubLogo = new URL(qr.logoSource).origin === new URL(url).origin && /\/GitHub_Invertocat_Black_Clearspace\.[a-zA-Z0-9_-]+\.svg$/.test(new URL(qr.logoSource).pathname)
  if (qr.symbol.width !== 96 || qr.symbol.height !== 96 || qr.visibleText !== '' || qr.errorLevel !== 'H' || qr.margin !== '4' || !localGithubLogo) failuresFound.push('QR must be 96px, caption-free, high-error-correction, and contain the local GitHub SVG with a four-module quiet zone')
  if (Math.abs(qr.logo.width - 22) > 0.1 || Math.abs(qr.logo.height - 22) > 0.1 || Math.abs(qr.logo.left + qr.logo.width / 2 - qr.symbol.left - qr.symbol.width / 2) > 0.1 || Math.abs(qr.logo.top + qr.logo.height / 2 - qr.symbol.top - qr.symbol.height / 2) > 0.1) failuresFound.push('GitHub logo must be centered and undistorted inside the QR')
  if (!contributions || contributions.top <= results.bottom || Math.abs(contributions.left - screen.safeBounds.left) > 1 || Math.abs(contributions.right - screen.safeBounds.right) > 1) failuresFound.push('Contributions and their QR must span the bottom below all results')
  if (results.top <= Math.max(...screen.columns.top.map((section) => section.bottom), ...screen.columns.examples.map((section) => section.bottom)) || Math.abs(results.left - screen.safeBounds.left) > 1 || Math.abs(results.right - screen.safeBounds.right) > 1) failuresFound.push('Results must span the lower page below the scope band and the example row')
  if (screen.resultPanels.length === 4) {
    const [global, local, group, regression] = screen.resultPanels
    if (Math.abs(global.top - local.top) > 1 || Math.abs(group.top - regression.top) > 1 || global.right >= local.left || group.right >= regression.left || Math.max(global.bottom, local.bottom) >= Math.min(group.top, regression.top)) failuresFound.push('Results must use a non-overlapping two-by-two grid')
    if (Math.abs(global.bottom - local.bottom) > 1 || Math.abs(group.bottom - regression.bottom) > 1) failuresFound.push('Result frames must align at the bottom of each row')
  }
  if (screen.resultCaptionCount !== 0) failuresFound.push('Removed result captions remain')
  for (const kind of ['local', 'global']) {
    const crops = manuscriptAssets.find((asset) => asset.kind === kind).crops
    if (crops.slice(0, 3).some((crop) => crop.display[1] !== crops[0].display[1]) || crops.slice(3).some((crop) => crop.display[1] <= crops[0].display[1] + crops[0].display[3])) failuresFound.push(`${kind} metric panels must use the three-plus-two arrangement`)
  }
  if (screen.columns.examples[0].top <= screen.columns.top[0].bottom) failuresFound.push('The example row must sit below the scope band')
  if (screen.scopeItems.length !== 4 || screen.scopeItems.slice(0, 2).some((item) => Math.abs(item.top - screen.scopeItems[0].top) > 1) || screen.scopeItems.slice(2).some((item) => Math.abs(item.top - screen.scopeItems[2].top) > 1) || screen.scopeItems[0].bottom >= screen.scopeItems[2].top) failuresFound.push('Benchmark scope must be a two-by-two named tile grid')
  // Two tiles flank each side of the architecture; all three columns share the same height.
  const arch = screen.architecture.viewport
  const band = screen.columns.top[0]
  if ([0, 2].some((index) => screen.scopeItems[index].right >= arch.left) || [1, 3].some((index) => screen.scopeItems[index].left <= arch.right)) failuresFound.push('Scope tiles must flank the architecture schematic on both sides')
  if (Math.abs(arch.top - band.top) > 1 || Math.abs(arch.bottom - band.bottom) > 1 || Math.abs(arch.height - band.height) > 1) failuresFound.push('Architecture schematic must match the height of the flanking scope sections')
  const overlaps = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top
  for (const [id, [width, height]] of Object.entries({ pwr: [52.9, 85.1], genwro: [99, 25.2], tooploox: [99, 26.1] })) {
    const logo = screen.branding.logos.find((item) => item.id === id)
    if (!logo || Math.abs(logo.width - width) > 0.1 || Math.abs(logo.height - height) > 0.1) failuresFound.push(`${id} does not match the requested header logo scale`)
  }
  for (const logo of screen.branding.logos) {
    if (logo.width <= 0 || logo.height <= 0 || logo.objectFit !== 'contain') failuresFound.push(`${logo.id} logo is missing or distorted`)
    if (logo.left < screen.branding.header.left || logo.right > screen.branding.header.right || logo.top < screen.branding.header.top || logo.bottom > screen.branding.header.bottom) failuresFound.push(`${logo.id} logo exceeds the header`)
  }
  if (screen.branding.identityText.some((text) => screen.branding.logos.some((logo) => overlaps(logo, text)))) failuresFound.push('Brand logos overlap header text')
  const centerX = (rect) => (rect.left + rect.right) / 2
  if (screen.branding.titleAlign !== 'center' || Math.abs(centerX(screen.branding.title) - centerX(screen.branding.header)) > 1) failuresFound.push('Manuscript title is not centered in the header')
  for (const [side, ids] of Object.entries({ left: ['ecml-pkdd', 'xkdd'], right: ['pwr', 'genwro', 'tooploox'] })) {
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
  for (const [role, value] of Object.entries(screen.minimumFontPx)) {
    if (!value && role !== 'caption') failuresFound.push(`${role} font size is NOT MEASURED`)
  }
  const requiredFontPx = {
    body: visualSpec.minimumPrintType.body.cssPx,
    caption: visualSpec.minimumPrintType.citation.cssPx,
    resultHeading: visualSpec.minimumPrintType.body.cssPx,
  }
  for (const [role, minimumPx] of Object.entries(requiredFontPx)) {
    if (screen.minimumFontPx[role] && screen.minimumFontPx[role] + 0.01 < minimumPx) failuresFound.push(`${role} font is ${screen.minimumFontPx[role]}px; minimum is ${minimumPx}px`)
  }

  await auditManuscriptLabelBounds(page, manuscriptAssets)
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
console.log(`A1 portrait layout audit passed: sections=${report.sections.length}, title=${report.typography.titlePt.toFixed(2)}pt, subheadings=28pt, result labels≥17pt; original manuscript schema preserved`)
