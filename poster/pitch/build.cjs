const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');
const sharp = require('sharp');
const PptxGenJS = require('pptxgenjs');
const JSZip = require('jszip');
const html2pptx = require(process.env.PPTX_HTML_CONVERTER || '/Users/ofurman/.agents/skills/pptx/scripts/html2pptx.js');

const root = path.resolve(__dirname, '../..');
const assets = path.join(__dirname, 'assets');
const output = path.join(__dirname, 'deliverables');
const hash = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

async function main() {
  await fs.mkdir(assets, { recursive: true });
  await fs.mkdir(output, { recursive: true });
  const identity = JSON.parse(await fs.readFile(path.join(root, 'poster/research/identity.json'), 'utf8'));
  const speech = JSON.parse(await fs.readFile(path.join(__dirname, 'speech.json'), 'utf8'));
  const manuscript = await fs.readFile(path.join(root, 'manuscript/main_lncs.tex'), 'utf8');
  for (const text of ['18 pre-configured datasets', '14 widely used', 'no single method uniformly dominates']) {
    if (!manuscript.includes(text)) throw new Error(`Missing manuscript support: ${text}`);
  }

  const browser = await chromium.launch({ channel: 'chrome' });
  const figures = [];
  try {
    const page = await browser.newPage();
    for (const paradigm of ['local', 'global', 'group']) {
      const sourcePath = `poster/plots/generated/manuscript-${paradigm}.svg`;
      const source = await fs.readFile(path.join(root, sourcePath), 'utf8');
      const height = { local: 142, global: 128, group: 152 }[paradigm];
      const extracted = await page.evaluate(({ svgText, height }) => {
        const doc = new DOMParser().parseFromString(svgText, 'image/svg+xml');
        const svg = doc.documentElement;
        const metadata = JSON.parse(svg.querySelector('metadata').textContent);
        for (const child of [...svg.children]) {
          if (!['metric-0', 'metric-1', 'metric-3'].includes(child.id)) child.remove();
        }
        // Translate whole groups only: preserve original pixels, axes and labels.
        [...svg.children].forEach((group, index) => {
          const image = group.querySelector('image');
          const center = Number(image.getAttribute('x')) + Number(image.getAttribute('width')) / 2;
          group.setAttribute('transform', `translate(${95 + 175 * index - center} ${24 - Number(image.getAttribute('y'))})`);
        });
        svg.setAttribute('viewBox', `0 0 525 ${height}`);
        svg.setAttribute('width', '1050');
        svg.setAttribute('height', String(height * 2));
        return { svg: new XMLSerializer().serializeToString(svg), metadata };
      }, { svgText: source, height });
      await page.setContent(extracted.svg);
      const clipped = await page.locator('svg [data-label], svg image').evaluateAll((elements) => elements.filter((el) => {
        const r = el.getBoundingClientRect();
        const s = el.ownerSVGElement.getBoundingClientRect();
        return r.left < s.left - 1 || r.top < s.top - 1 || r.right > s.right + 1 || r.bottom > s.bottom + 1;
      }).map((el) => el.getAttribute('data-label') || 'plot image'));
      if (clipped.length) throw new Error(`${paradigm}: clipped figure labels: ${clipped.join(', ')}`);
      await fs.writeFile(path.join(assets, `${paradigm}-results.svg`), extracted.svg);
      await sharp(Buffer.from(extracted.svg)).resize({ width: 2625 }).png().toFile(path.join(assets, `${paradigm}-results.png`));
      figures.push({ paradigm, source: sourcePath, sha256: hash(source), manuscriptSource: extracted.metadata.source,
        retainedMetrics: [0, 1, 3].map((index) => extracted.metadata.crops[index]), methodKey: extracted.metadata.methodKey, viewport: [0, 0, 525, height] });
    }

    await page.goto(pathToFileURL(path.join(root, 'poster/cel-benchmark-poster/bundle.html')).href);
    await page.waitForSelector('.project-mark__qr svg image');
    const qr = await page.locator('.project-mark__qr svg').evaluate((element) => new XMLSerializer().serializeToString(element));
    await sharp(Buffer.from(qr)).resize(580, 580).png().toFile(path.join(assets, 'github-qr.png'));

    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto(pathToFileURL(path.join(__dirname, 'slide.html')).href);
    await page.evaluate(() => document.fonts.ready);
    const audit = await page.evaluate(() => {
      const title = document.querySelector('h1').innerText.replace(/\s+/g, ' ').trim();
      const overflow = [...document.querySelectorAll('p,h1,h2,img')].filter((el) => {
        const r = el.getBoundingClientRect();
        return r.left < 0 || r.top < 0 || r.right > 1280 || r.bottom > 720 || el.scrollWidth > el.clientWidth + 1;
      }).map((el) => el.innerText || el.alt);
      const broken = [...document.images].filter((img) => !img.complete || !img.naturalWidth).map((img) => img.src);
      const tiles = [...document.querySelectorAll('.scope-tile')].map((el) => ({ id: el.dataset.scope, number: el.querySelector('.number').textContent, label: el.querySelector('.label').innerText.replace(/\s+/g, ' ') }));
      const tileOverflow = [...document.querySelectorAll('.scope-tile p')].filter((el) => {
        const r = el.getBoundingClientRect();
        const tile = el.closest('.scope-tile').getBoundingClientRect();
        return r.left < tile.left || r.right > tile.right || r.top < tile.top || r.bottom > tile.bottom;
      }).map((el) => el.innerText);
      const rect = (selector) => { const r = document.querySelector(selector).getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom }; };
      const titleRect = rect('h1');
      const authorsRect = rect('.authors');
      const titleStyle = getComputedStyle(document.querySelector('h1'));
      const results = [...document.querySelectorAll('.result-row')].map((el) => {
        const r = el.getBoundingClientRect();
        const plot = el.querySelector('.result').getBoundingClientRect();
        return { paradigm: el.dataset.paradigm, left: r.left, top: r.top, bottom: r.bottom, plotBottom: plot.bottom };
      });
      return { title, overflow: [...overflow, ...tileOverflow], broken, tiles, results, header: { titleLeft: titleRect.left, titleBottom: titleRect.bottom, titleLines: (titleRect.bottom - titleRect.top) / parseFloat(titleStyle.lineHeight), authorsLeft: authorsRect.left, authorsTop: authorsRect.top, hasAffiliation: !!document.querySelector('.affiliation') }, words: document.body.innerText.trim().split(/\s+/).length };
    });
    if (audit.title !== identity.title || audit.overflow.length || audit.broken.length) throw new Error(JSON.stringify(audit));
    await fs.writeFile(path.join(output, 'layout-audit.json'), JSON.stringify(audit, null, 2) + '\n');
  } finally {
    await browser.close();
  }

  const words = speech.sections.reduce((total, part) => total + part.text.split(/\s+/).length, 0);
  const notes = speech.sections.map((part) => `${part.time} | ${part.cue}\n${part.text}`).join('\n\n');
  const provenance = '\n\nEvidence: manuscript/main_lncs.tex (Introduction, Benchmark, Results, Conclusions); Adult Census validity, L2+Hamming and log-density plausibility panels from manuscript/figures/metrics_boxplot_local.png, metrics_boxplot_global.png and metrics_boxplot_group_wise.png. All three comparisons aggregate LR and MLP as described in the manuscript. Original plot pixels, axes, labels and aspect ratios are preserved via the poster typography derivatives; complete metric groups are translated only. Higher log-density indicates greater distributional plausibility under the fitted density model, not causal feasibility. Global method keys: 1 AReS, 2 GLOBE-CE, 3 GlobalGLANCE (GLANCE configured with one group). The four tiles summarize 18 datasets, 14 methods (10 local, 2 global, 2 group-wise), two backbones per task, and nine metrics reported in classification tables, not the whole library registry. These are representative within-paradigm comparisons, not an aggregate or cross-paradigm method ranking. Axes retain their original ranges.';
  const deck = new PptxGenJS();
  deck.defineLayout({ name: 'CEL_WIDE', width: 13.333333, height: 7.5 });
  deck.layout = 'CEL_WIDE';
  deck.author = identity.authors.map((author) => author.name).join(', ');
  deck.title = identity.title;
  deck.subject = 'Two-minute, one-slide poster pitch';
  deck.lang = 'en-GB';
  deck.theme = { headFontFace: 'Georgia', bodyFontFace: 'Arial', lang: 'en-GB' };
  const { slide } = await html2pptx(path.join(__dirname, 'slide.html'), deck);
  slide.addNotes(notes + provenance);
  const pptxPath = path.join(output, 'cel-poster-pitch.pptx');
  await deck.writeFile({ fileName: pptxPath });
  // The HTML converter drops dashed borders; restore them in the native tile shapes.
  const archive = await JSZip.loadAsync(await fs.readFile(pptxPath));
  const slidePath = 'ppt/slides/slide1.xml';
  let tileCount = 0;
  const slideXml = (await archive.file(slidePath).async('string')).replace(/<p:sp>.*?<\/p:sp>/g, (shape) => {
    if (!/val="E6F4FC"/i.test(shape)) return shape;
    tileCount += 1;
    return shape.replace(/<a:ln\b[^>]*>(.*?)<\/a:ln>/, '<a:ln w="15240">$1<a:prstDash val="lgDash"/></a:ln>');
  });
  if (tileCount !== 4) throw new Error(`Expected four native scope tiles, found ${tileCount}`);
  archive.file(slidePath, slideXml);
  await fs.writeFile(pptxPath, await archive.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
  const script = `# CEL: two-minute poster pitch\n\n${words} spoken words. Rehearse at approximately ${Math.round(words / 2)} words per minute; the timestamps are pacing targets, not an automatic timer. The same script is embedded in the single slide's speaker notes.\n\n` + speech.sections.map((part) => `## ${part.time}\n\n*${part.cue}*\n\n${part.text}\n`).join('\n');
  await fs.writeFile(path.join(output, 'two-minute-script.md'), script);
  await fs.writeFile(path.join(assets, 'provenance.json'), JSON.stringify({
    figures,
    changes: 'Select validity, L2+Hamming and log-density plausibility for local, global and group-wise results. Translate complete metric groups into aligned columns. Dataset label and global method key use native slide text. Original plot pixels, aspect ratios, ticks, directions and method labels are unchanged.',
    repository: identity.links.repository,
  }, null, 2) + '\n');
  const profile = await fs.mkdtemp(path.join(require('node:os').tmpdir(), 'cel-pitch-lo-'));
  try {
    execFileSync('soffice', [`-env:UserInstallation=${pathToFileURL(profile).href}`, '--headless', '--convert-to', 'pdf', '--outdir', output, pptxPath], { stdio: 'inherit', timeout: 60000 });
  } finally {
    await fs.rm(profile, { recursive: true, force: true });
  }
  execFileSync('pdftoppm', ['-png', '-singlefile', '-r', '144', path.join(output, 'cel-poster-pitch.pdf'), path.join(output, 'cel-poster-pitch')], { stdio: 'inherit' });
  console.log(`Created one slide with ${words} spoken words; PowerPoint, PDF, PNG and script are in ${output}`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
