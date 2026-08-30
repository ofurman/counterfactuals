#!/usr/bin/env node

import {readFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const researchDir = path.resolve(here, "..");
const repositoryDir = path.resolve(researchDir, "../..");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const fail = (message) => { throw new Error(message); };

const localReferencePath = "docs/plans/cel-scientific-benchmark-poster/resources/planning-findings.md";
const localReferenceHref = `../../${localReferencePath}`;
const [sources, notes, claims, identity, guidelines, visual, storyboard, content, precedents, localReference] = await Promise.all([
  readJson(path.join(researchDir, "neurips/sources.json")),
  readFile(path.join(researchDir, "neurips/notes.md"), "utf8"),
  readJson(path.join(researchDir, "claims/claims.generated.json")),
  readJson(path.join(researchDir, "identity.json")),
  readFile(path.join(researchDir, "design-guidelines.md"), "utf8"),
  readJson(path.join(researchDir, "visual-spec.json")),
  readFile(path.join(researchDir, "storyboard.md"), "utf8"),
  readJson(path.join(researchDir, "poster-content.json")),
  readJson(path.join(researchDir, "precedents.json")),
  readFile(path.join(repositoryDir, localReferencePath), "utf8")
]);

const sourceIds = new Set(sources.sources.map((source) => source.id));
const observedIds = new Set([...notes.matchAll(/\*\*Observation \[([^\]]+)\]/g)].map((match) => match[1]));
const claimsById = new Map(claims.claims.map((claim) => [claim.id, claim]));
const linkIds = new Set(Object.keys(content.links));

// local-reference is a checked repository source, not a magic source-ID exception.
const escapedLocalHref = localReferenceHref.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
if (!new RegExp(`\\[local-reference\\]\\(${escapedLocalHref}\\)`).test(guidelines)) {
  fail(`Design guidelines must bind local-reference to ${localReferencePath}`);
}
for (const [fact, pattern] of [
  ["native canvas", /Fixed `1800 × 1273` CSS-pixel canvas/],
  ["A0 page", /A0 landscape print target \(`1189 × 841 mm`\)/],
  ["macro-grid", /Asymmetric `1fr 2fr 1fr` body grid/]
]) {
  if (!pattern.test(localReference)) fail(`local-reference lacks the exact ${fact} fact`);
}

// A source list is a strict manifest, not a free-text place where unknown IDs can disappear.
const ruleBlocks = guidelines
  .split(/(?=\n- \*\*(?:Preserve|Adapt|Avoid)\b)/)
  .filter((block) => /^\n?- \*\*(?:Preserve|Adapt|Avoid)\b/.test(block));
if (ruleBlocks.length < 12) fail("Design guidelines contain too few explicit preserve/adapt/avoid decisions");
for (const block of ruleBlocks) {
  const sourceList = block.match(/Sources:\s*\[([\s\S]*?)\]\./);
  if (!sourceList) fail(`Guideline has no Sources list: ${block.split("\n")[0]}`);
  const entries = sourceList[1].split(",").map((entry) => entry.trim()).filter(Boolean);
  if (entries.length === 0) fail(`Guideline has an empty Sources list: ${block.split("\n")[0]}`);

  let observedNeuripsCount = 0;
  for (const entry of entries) {
    const parsed = entry.match(/^([a-z0-9]+(?:-[a-z0-9]+)*)(?:\s+\((local-only)\))?$/);
    if (!parsed) fail(`Malformed guideline source entry "${entry}": ${block.split("\n")[0]}`);
    const [, id, localOnly] = parsed;
    if (id === "local-reference") {
      if (!localOnly) fail(`local-reference must be explicitly marked local-only: ${block.split("\n")[0]}`);
      if (!localReference.includes("## Local reference poster")) fail("local-reference does not resolve to its named source section");
      continue;
    }
    if (localOnly) fail(`Only local-reference may be marked local-only: ${id}`);
    if (!sourceIds.has(id)) fail(`Guideline references unknown source ID: ${id}`);
    if (!observedIds.has(id)) fail(`Guideline cites a source without an observed note: ${id}`);
    observedNeuripsCount += 1;
  }
  if (observedNeuripsCount === 0) fail(`Guideline has no observed NeurIPS source: ${block.split("\n")[0]}`);
}
if (!/web-artifacts-builder constraints[\s\S]*not observations attributed to the NeurIPS posters/i.test(guidelines)) {
  fail("Web-artifacts-builder constraints are not separated from source-observation claims");
}

// Gate 2: geometry must be positive, contained, and physically coherent at A0 scale.
const expectedPage = visual.page;
if (expectedPage.format !== "A0" || expectedPage.orientation !== "landscape" || expectedPage.widthMm !== 1189 || expectedPage.heightMm !== 841 || expectedPage.marginMm !== 0) fail("Visual spec lacks exact A0 landscape page geometry");
if (visual.canvas.widthPx !== 1800 || visual.canvas.heightPx !== 1273) fail("Visual spec lacks the native canvas dimensions");
if (expectedPage.widthMm <= 0 || expectedPage.heightMm <= 0 || visual.canvas.widthPx <= 0 || visual.canvas.heightPx <= 0) fail("Page and canvas dimensions must be positive");
for (const edge of ["topPx", "rightPx", "bottomPx", "leftPx"]) {
  if (!Number.isFinite(visual.safeArea?.[edge]) || visual.safeArea[edge] <= 0) fail(`Invalid safe-area value: ${edge}`);
}
if (visual.safeArea.leftPx + visual.safeArea.rightPx >= visual.canvas.widthPx || visual.safeArea.topPx + visual.safeArea.bottomPx >= visual.canvas.heightPx) fail("Safe area is not contained by the canvas");
if (JSON.stringify(visual.macroGrid.columns) !== JSON.stringify(["1fr", "1.85fr", "1.45fr"])) fail("Visual spec lacks the rebalanced framework/results macro-grid");
if (!Number.isFinite(visual.macroGrid.gapPx) || visual.macroGrid.gapPx <= 0) fail("Macro-grid gap must be positive");
for (const family of ["title", "body", "mono"]) if (!visual.fonts?.[family]) fail(`Missing font stack: ${family}`);

const mmPerCanvasPxX = expectedPage.widthMm / visual.canvas.widthPx;
const mmPerCanvasPxY = expectedPage.heightMm / visual.canvas.heightPx;
if (Math.abs(mmPerCanvasPxX - mmPerCanvasPxY) / mmPerCanvasPxX > 0.005) fail("Canvas-to-A0 scaling is incoherent across axes");
for (const role of ["title", "sectionHeading", "body", "chartLabel", "citation"]) {
  const size = visual.minimumPrintType?.[role];
  if (!Number.isFinite(size?.cssPx) || size.cssPx <= 0 || !Number.isFinite(size?.approxPtAtA0) || size.approxPtAtA0 <= 0) fail(`Invalid minimum print type: ${role}`);
  const derivedPt = size.cssPx * mmPerCanvasPxX * 72 / 25.4;
  const tolerancePt = Math.max(1, derivedPt * 0.025);
  if (Math.abs(size.approxPtAtA0 - derivedPt) > tolerancePt) fail(`Incoherent CSS-pixel to A0-point conversion for ${role}: expected approximately ${derivedPt.toFixed(2)}pt`);
}
if (visual.print.pageCss !== "@page { size: 1189mm 841mm; margin: 0; }" || visual.print.colorAdjust !== "exact" || visual.print.singlePage !== true) fail("Visual spec lacks exact print-color/page handling");
if (!Number.isFinite(visual.expectedBundleAssetBudgetBytes) || visual.expectedBundleAssetBudgetBytes <= 0) fail("Missing expected bundle asset budget");

// Citations are repository-relative and their fragments must still exist when one is supplied.
const sourceTextCache = new Map();
const validateCitation = async (citation, owner) => {
  if (typeof citation !== "string" || citation.trim() === "") fail(`${owner} has an empty source citation`);
  const [relativeFile, fragment, ...extra] = citation.split("#");
  if (extra.length > 0 || !relativeFile) fail(`${owner} has a malformed source citation: ${citation}`);
  const absoluteFile = path.resolve(repositoryDir, relativeFile);
  if (!(absoluteFile === repositoryDir || absoluteFile.startsWith(`${repositoryDir}${path.sep}`))) fail(`${owner} source escapes the repository: ${citation}`);
  let sourceText = sourceTextCache.get(absoluteFile);
  if (sourceText === undefined) {
    try {
      sourceText = await readFile(absoluteFile, "utf8");
    } catch {
      fail(`${owner} source file does not exist: ${relativeFile}`);
    }
    sourceTextCache.set(absoluteFile, sourceText);
  }
  if (fragment) {
    const decoded = decodeURIComponent(fragment);
    const spaced = decoded.replaceAll("-", " ");
    if (!sourceText.includes(decoded) && !sourceText.toLowerCase().includes(spaced.toLowerCase())) fail(`${owner} source anchor does not resolve: ${citation}`);
  }
  return relativeFile;
};

const validateClaimMapping = async (ids, citations, owner, {required = true} = {}) => {
  if (!Array.isArray(ids) || (required && ids.length === 0)) fail(`${owner} lacks an explicit claim mapping`);
  if (!Array.isArray(citations) || (required && citations.length === 0)) fail(`${owner} lacks source citations`);
  const citedFiles = new Set();
  for (const citation of citations ?? []) citedFiles.add(await validateCitation(citation, owner));
  for (const id of ids ?? []) {
    const claim = claimsById.get(id);
    if (!claim) fail(`${owner} references unknown claim ID: ${id}`);
    if (required && !citedFiles.has(claim.source.file)) fail(`${owner} does not cite the source file for claim ${id}: ${claim.source.file}`);
  }
};

const precedentKeys = new Set();
for (const precedent of precedents.items ?? []) {
  if (!precedent.label || !precedent.citationKey || !precedent.sourceCitation) fail("Incomplete benchmark precedent citation");
  if (precedentKeys.has(precedent.citationKey)) fail(`Duplicate benchmark precedent: ${precedent.citationKey}`);
  precedentKeys.add(precedent.citationKey);
  const sourceFile = await validateCitation(precedent.sourceCitation, `benchmark precedent ${precedent.label}`);
  const sourceText = await readFile(path.join(repositoryDir, sourceFile), "utf8");
  if (!sourceText.includes(`${precedent.label} \\cite{${precedent.citationKey}}`)) fail(`Benchmark precedent is not source-derived: ${precedent.label}`);
}
if (precedentKeys.size < 2) fail("Poster lacks benchmark precedent citations");

// Gate 3: require actual narrative bodies, not headings or an appended paper/table dump.
const sectionBody = (heading) => {
  const escaped = heading.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = storyboard.match(new RegExp(`^## ${escaped}[ \\t]*\\r?\\n([\\s\\S]*?)(?=^## |(?![\\s\\S]))`, "m"));
  return match?.[1]?.trim() ?? "";
};
const proseWordCount = (text) => text.replace(/<!--[\s\S]*?-->/g, " ").trim().split(/\s+/).filter(Boolean).length;
for (const [heading, minimumWords] of [["Explicit reading order", 35], ["Thirty-second visitor narrative", 20], ["Two-minute visitor narrative", 30]]) {
  const body = sectionBody(heading);
  if (!body || proseWordCount(body) < minimumWords) fail(`${heading} must contain a substantive narrative body`);
}
for (const marker of ["Header → Left → Center → Right → Footer", "Thirty-second visitor narrative", "Two-minute visitor narrative"]) {
  if (!storyboard.includes(marker)) fail(`Storyboard lacks required narrative/order marker: ${marker}`);
}
if (/^#{2,6} (?:Abstract|Introduction|Related Works?|CEL(?::.*)?|Benchmark|Datasets?|Models?|Methods?|Metrics?|Experimental Setup|Results?|Full Results?|Conclusions?)\s*$/mi.test(storyboard) || /\\(?:section|subsection)\{|\\begin\{(?:table|tabular)\}/.test(storyboard)) fail("Storyboard reproduces a manuscript section dump");
const markdownTableLines = storyboard.split("\n").filter((line) => /^\s*\|.*\|\s*$/.test(line));
if (markdownTableLines.length >= 2) fail("Storyboard contains an appended Markdown table");
if (/<\s*table\b/i.test(storyboard)) fail("Storyboard contains an appended HTML table");
if (!/not a dump of manuscript sections or a full result\s+table/i.test(storyboard)) fail("Storyboard does not explicitly reject a full result table");

// Every storyboard claim tag is complete, current, and tied to its claim source file.
const storyboardTags = [...storyboard.matchAll(/<!--\s*claims:\s*([^|]+?)\s*\|\s*sources:\s*([^>]+?)\s*-->/g)];
if (storyboardTags.length < 12) fail("Storyboard has too few explicit claim/source mappings");
for (const [index, tag] of storyboardTags.entries()) {
  const ids = tag[1].split(",").map((value) => value.trim()).filter(Boolean);
  const citations = tag[2].split(",").map((value) => value.trim()).filter(Boolean);
  await validateClaimMapping(ids, citations, `storyboard claim tag ${index + 1}`);
}
const substantiveHeadings = ["Scientific argument", "Header", "Left — counterfactual concept", "Center — framework and benchmark scope", "Right — results and contributions", "Thirty-second visitor narrative", "Two-minute visitor narrative"];
const structuralLines = new Set([
  "**Identity inventory:** Exact manuscript title centered between the logos, camera-ready authors, affiliation. Show the title only once, without a subtitle, venue marker, or top color line.",
  "**Logo inventory:** XKDD above ECML-PKDD on the left; PWr above genwro.AI above Tooploox on the right. Institutional assets come from the user-provided PUMAL reference poster; conference assets were supplied in the project. Preserve all logo files and aspect ratios and keep authorship unchanged.",
  "**QR inventory:** One labelled `Code & project` QR inside the Extend contribution below the right-column results; linked to the repository, with no header or paper QR.",
  "**Evidence-view inventory:**",
  "**Footer inventory:** `uv add ce-library` and repository/documentation links; keep provenance in project files."
]);
for (const heading of substantiveHeadings) {
  const body = sectionBody(heading);
  if (!body) fail(`${heading} lacks a substantive body`);
  const assertionLines = body.split("\n").filter((line) => line.trim() !== "" && !structuralLines.has(line.trim()));
  if (assertionLines.length === 0 || assertionLines.some((line) => !/<!--\s*claims:/.test(line))) fail(`${heading} contains an unmapped scientific assertion line`);
}

// The top-level argument stores only a recipe over claim-ledger prose. Additional free text is
// forbidden, so an appended unsupported sentence cannot inherit a broad claim mapping.
if (!content.argument || Array.isArray(content.argument) || typeof content.argument !== "object") fail("Top-level poster argument must be a claim-ledger composition recipe");
const argumentKeys = Object.keys(content.argument).sort();
if (JSON.stringify(argumentKeys) !== JSON.stringify(["claimIds", "composeFrom", "separator"])) fail("Top-level poster argument contains standalone or unsupported fields");
if (content.argument.composeFrom !== "verdict" || content.argument.separator !== " ") fail("Top-level poster argument must compose claim verdicts with one space");
if (!Array.isArray(content.argument.claimIds) || content.argument.claimIds.length === 0) fail("Top-level poster argument lacks claim inputs");
for (const id of content.argument.claimIds) {
  const claim = claimsById.get(id);
  if (!claim || typeof claim.verdict !== "string" || claim.verdict.trim() === "") fail(`Top-level poster argument cannot derive verdict text for ${id}`);
  await validateCitation(claim.source.file, `top-level poster argument claim ${id}`);
}
const composedArgument = content.argument.claimIds.map((id) => claimsById.get(id).verdict).join(content.argument.separator);
if (!composedArgument.trim()) fail("Top-level poster argument composes to empty text");
const sectionIds = new Set();
const orders = new Set();
const utilitySectionIds = new Set(["reproducibility"]);
for (const section of content.sections) {
  if (sectionIds.has(section.id)) fail(`Duplicate section ownership: ${section.id}`);
  if (orders.has(section.order)) fail(`Duplicate section order: ${section.order}`);
  sectionIds.add(section.id);
  orders.add(section.order);
  if (!section.owner || !section.heading || !Array.isArray(section.copy) || section.copy.some((line) => typeof line !== "string" || line.trim() === "")) fail(`Incomplete section: ${section.id}`);
  const scientific = !utilitySectionIds.has(section.id) || section.assetRoles.some((role) => /(?:chart|tradeoff|applicability|scope|protocol|selection)/.test(role));
  await validateClaimMapping(section.claimIds, section.sourceCitations, `poster section ${section.id}`, {required: scientific});
  for (const linkId of section.linkIds) if (!linkIds.has(linkId)) fail(`${section.id} references unknown link ID: ${linkId}`);
}
for (const required of ["header", "problem", "scope", "protocol", "results", "local-tradeoff", "group-tradeoff", "applicability", "guidance-limitations", "reproducibility"]) {
  if (!sectionIds.has(required)) fail(`Missing required poster section: ${required}`);
}
for (const [linkId, link] of Object.entries(content.links)) {
  if (!identity.links[link.identityLink]) fail(`${linkId} does not resolve to identity.json`);
}
if (content.sections.length !== 10 || sectionIds.has("regression-tradeoff")) fail("Poster must contain seven top-level sections and three nested result panels without regression results");
if (JSON.stringify([...content.resultVisuals].sort()) !== JSON.stringify(["global-manuscript-figure", "group-manuscript-figure", "local-manuscript-figure"])) fail("Poster must own exactly the local, global, and group-wise result visuals");
const conceptSection = content.sections.find((section) => section.id === "problem");
if (conceptSection.copy.join(" ") !== claimsById.get("concept.counterfactual").posterWording || !conceptSection.claimIds.includes("concept.counterfactual")) fail("CE definition must resolve to its manuscript claim");
for (const [id, owner] of [["problem", "left"], ["applicability", "right"], ["protocol", "center"], ["scope", "center"], ["results", "right"], ["local-tradeoff", "right"], ["guidance-limitations", "right"], ["group-tradeoff", "right"]]) {
  if (content.sections.find((section) => section.id === id)?.owner !== owner) fail(`Incorrect column ownership for ${id}`);
}
const scopeSection = content.sections.find((section) => section.id === 'scope');
if (!scopeSection.claimIds.includes('scope.metrics') || scopeSection.order <= content.sections.find((section) => section.id === 'protocol').order) fail('Scope with the reported metric count must follow the evaluation framework');
const resultsSection = content.sections.find((section) => section.id === 'results');
if (resultsSection.heading !== 'Results' || resultsSection.order >= content.sections.find((section) => section.id === 'guidance-limitations').order) fail('Unified Results must precede contributions');
for (const id of ['result.global.overview', 'result.local.overview', 'result.group.overview']) if (!resultsSection.claimIds.includes(id)) fail(`Unified Results lacks ${id}`);
const qrOwners = content.sections.filter((section) => section.assetRoles.includes("project-qr"));
if (qrOwners.length !== 1 || qrOwners[0].id !== "guidance-limitations" || qrOwners[0].owner !== "right" || !qrOwners[0].claimIds.includes("contribution.library") || !qrOwners[0].linkIds.includes("repository")) fail("Exactly one project QR must be owned by the right-column Extend contribution");
const headerSection = content.sections.find((section) => section.id === "header");
if (headerSection.heading !== identity.title) fail("Poster title must exactly match the manuscript title");
if (headerSection.copy.length !== 0) fail("The title header must not contain a subtitle");
if (/hold\s+the\s+protocol\s+constant/i.test([headerSection.heading, ...headerSection.copy].join(" ")) && !headerSection.claimIds.includes("scope.protocol")) fail("Header protocol-constant copy must map to scope.protocol");
const guidanceSection = content.sections.find((section) => section.id === "guidance-limitations");
for (const id of ["contribution.protocol", "contribution.benchmark", "contribution.library"]) {
  if (!guidanceSection.claimIds.includes(id)) fail(`Contribution section lacks manuscript contribution claim: ${id}`);
}
for (const [sectionId, claimId] of [
  ["local-tradeoff", "result.local.overview"],
  ["applicability", "result.global.overview"],
  ["group-tradeoff", "result.group.overview"],
]) {
  if (!content.sections.find((section) => section.id === sectionId)?.claimIds.includes(claimId)) fail(`${sectionId} lacks manuscript result claim: ${claimId}`);
  if (!storyboardTags.some((tag) => tag[1].split(",").map((value) => value.trim()).includes(claimId))) fail(`Storyboard lacks manuscript result mapping: ${claimId}`);
}

// Numeric provenance scan. Source IDs, citation metadata, and ordered-list markers are structural;
// the only permitted prose numerals are exact geometry strings already owned by visual-spec.json.
const blankPreservingLength = (match) => " ".repeat(match.length);
const stripStructuralNumbers = (text) => {
  let stripped = text
    .replace(/Sources:\s*\[[\s\S]*?\]\./g, blankPreservingLength)
    .replace(/^\s*[1-4][.)]\s+(?=[^\n]*<!--\s*claims:)/gm, blankPreservingLength)
    .replace(/<!--[\s\S]*?-->/g, blankPreservingLength);
  for (const id of sourceIds) stripped = stripped.replaceAll(id, blankPreservingLength);
  return stripped;
};
const numericLiteral = /(?<![\p{L}\p{N}_])[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?(?:\s*(?:×|x)\s*[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?)?/gu;
const scanNumericLiterals = (text, owner, allowedPhrases = []) => {
  const stripped = stripStructuralNumbers(text);
  const allowedRanges = [];
  for (const phrase of allowedPhrases) {
    let start = 0;
    while ((start = stripped.indexOf(phrase, start)) !== -1) {
      allowedRanges.push([start, start + phrase.length]);
      start += phrase.length;
    }
  }
  for (const match of stripped.matchAll(numericLiteral)) {
    const start = match.index;
    const end = start + match[0].length;
    if (!allowedRanges.some(([allowedStart, allowedEnd]) => start >= allowedStart && end <= allowedEnd)) fail(`${owner} contains an untracked numeric literal: ${match[0]}`);
  }
};
scanNumericLiterals(guidelines, "design guidelines", [
  `${visual.canvas.widthPx} × ${visual.canvas.heightPx}`,
  `${visual.page.widthMm} × ${visual.page.heightMm} mm`,
  visual.macroGrid.columns.join(" "),
  "1fr 2fr 1fr",
  visual.print.pageCss
]);
scanNumericLiterals(storyboard, "storyboard");
scanNumericLiterals(composedArgument, "poster argument composed from claim verdicts");
for (const section of content.sections) scanNumericLiterals([section.heading, ...section.copy].join("\n"), `poster section ${section.id}`);

const scannedText = [guidelines, storyboard, JSON.stringify(content)].join("\n");
if (/\b(?:TBD|TODO|FIXME|TBA)\b|example\.com/i.test(scannedText)) fail("Brief contains placeholder text");
const wordCount = content.sections.flatMap((section) => [section.heading, ...section.copy]).join(" ").trim().split(/\s+/).filter(Boolean).length;
if (wordCount > 110) fail(`Poster copy exceeds the 110-word section heading+copy budget: ${wordCount}`);

console.log(`Validated brief: ${ruleBlocks.length} source-backed rules, ${content.sections.length} sections, ${content.resultVisuals.length} result visuals`);
console.log(`Section heading+copy word count: ${wordCount}/110`);
