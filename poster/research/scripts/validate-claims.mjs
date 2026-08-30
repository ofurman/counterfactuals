#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import path from "node:path";
import { buildClaims, rootDir, serialize } from "./extract-claims.mjs";

function fail(message) {
  throw new Error(message);
}

function requireMatch(input, regex, description) {
  const match = input.match(regex);
  if (!match) fail(`Missing source anchor: ${description}`);
  return match;
}

function cleanIdentityTex(value) {
  return value
    .replace(/\{\\L\}/g, "Ł")
    .replace(/\{\\l\}/g, "ł")
    .replace(/\{\\k\s+E\}/g, "Ę")
    .replace(/\{\\k\s+e\}/g, "ę")
    .replace(/\\(?:inst|orcidID)\{[^}]+\}/g, "")
    .replace(/\\\\/g, " ")
    .replace(/[{}]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function tableBlock(input, label) {
  const labelAt = input.indexOf(`\\label{${label}}`);
  if (labelAt < 0) fail(`Missing table label ${label}`);
  const start = input.lastIndexOf("\\begin{table", labelAt);
  const end = input.indexOf("\\end{table", labelAt);
  if (start < 0 || end < 0) fail(`Malformed table ${label}`);
  return input.slice(start, input.indexOf("\n", end) + 1);
}

function tableRowCells(block, dataset, method) {
  let currentDataset = null;
  for (const rawLine of block.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("%") || !line.endsWith("\\\\") || !line.includes("&")) continue;
    const cells = line.slice(0, -2).split("&").map((cell) => cell.trim());
    const datasetMatch = cells[0].match(/\\multirow(?:\[[^\]]+\])?\{[^}]+\}\{(?:\*|[^}]+)\}\{([^}]+)\}/);
    if (datasetMatch) currentDataset = cleanIdentityTex(datasetMatch[1]);
    if (currentDataset === dataset && cleanIdentityTex(cells[1]) === method) return cells;
  }
  return null;
}

function cleanResultTex(value) {
  return value
    .replace(/\\boldsymbol\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, "$1")
    .replace(/\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, "$1")
    .replace(/\\times\s*10\^\{(-?\d+)\}/g, "e$1")
    .replace(/\\pm\{([^{}]+)\}/g, "±$1")
    .replace(/\\pm/g, "±")
    .replace(/\$/g, "")
    .replace(/[{}]/g, "")
    .trim();
}

function parseResultCell(cell) {
  const clean = cleanResultTex(cell);
  if (clean === "--") return {kind: "missing", display: "—", mean: null, std: null};
  if (/infty/i.test(clean)) return {kind: "non_finite", display: clean, mean: null, std: null};
  const match = clean.match(/^(-?(?:\d+(?:\.\d+)?|\.\d+)(?:e-?\d+)?)(?:±(-?(?:\d+(?:\.\d+)?|\.\d+)(?:e-?\d+)?))?$/i);
  if (!match) fail(`Cannot independently parse result cell: ${cell}`);
  const mean = Number(match[1]);
  const std = match[2] === undefined ? null : Number(match[2]);
  return {
    kind: "finite",
    display: std === null ? match[1] : `${match[1]} ± ${match[2]}`,
    mean,
    std,
    roundedZero: mean === 0 && (std === 0 || std === null)
  };
}

const claimsPath = path.join(rootDir, "poster/research/claims/claims.generated.json");
const identityPath = path.join(rootDir, "poster/research/identity.json");
const namesPath = path.join(rootDir, "poster/research/method-names.json");

const [committedClaims, identityText, namesText, main, supplementary, pyproject, readme, ...tableTexts] = await Promise.all([
  readFile(claimsPath, "utf8"),
  readFile(identityPath, "utf8"),
  readFile(namesPath, "utf8"),
  readFile(path.join(rootDir, "manuscript/main_lncs.tex"), "utf8"),
  readFile(path.join(rootDir, "manuscript/supplementary.tex"), "utf8"),
  readFile(path.join(rootDir, "pyproject.toml"), "utf8"),
  readFile(path.join(rootDir, "README.md"), "utf8"),
  ...["results_categorical.tex", "results_numerical.tex", "results_global.tex", "results_group.tex", "results_regression.tex"]
    .map((file) => readFile(path.join(rootDir, "manuscript/tables", file), "utf8"))
]);

const regenerated = await buildClaims();
if (serialize(regenerated) !== committedClaims) fail("claims.generated.json is stale; run extract-claims.mjs --write");

const identity = JSON.parse(identityText);
const names = JSON.parse(namesText);
const claims = JSON.parse(committedClaims);

const forbidden = /\b(?:TBD|TODO|FIXME)\b|example\.com/i;
if (forbidden.test(identityText + namesText + committedClaims)) fail("Placeholder metadata found");
if (!main.includes(`\\title{${identity.title}}`)) fail("Identity title is not sourced from manuscript/main_lncs.tex");
const cameraBlock = requireMatch(main, /%% ===== CAMERA-READY VERSION =====([\s\S]*?)\\maketitle/, "camera-ready identity block")[1]
  .split("\n")
  .map((line) => line.replace(/^\s*%\s?/, ""))
  .join("\n");
const authorBody = requireMatch(cameraBlock, /\\author\{([\s\S]*?)\}\s*\n\\authorrunning/, "camera-ready authors")[1];
const sourcedAuthors = authorBody.split(/\s+\\and\s+/).map((entry) => {
  const orcid = requireMatch(entry, /\\orcidID\{([^}]+)\}/, "camera-ready author ORCID")[1];
  const name = cleanIdentityTex(entry.slice(0, entry.indexOf("\\inst")));
  return {name, orcid};
});
if (JSON.stringify(identity.authors) !== JSON.stringify(sourcedAuthors)) fail("Identity authors do not exactly match the camera-ready block");
const sourcedAffiliation = cleanIdentityTex(requireMatch(cameraBlock, /\\institute\{([\s\S]*?)\\\\\s*\n\\email/, "camera-ready affiliation")[1]);
if (identity.affiliation !== sourcedAffiliation) fail(`Identity affiliation does not match the camera-ready block: ${sourcedAffiliation}`);
const sourcedVenue = requireMatch(main, /DOUBLE-BLIND REVIEW VERSION \(([^()]+) submission\)/, "venue marker")[1].trim();
if (identity.venue !== sourcedVenue) fail(`Identity venue does not match the manuscript marker: ${sourcedVenue}`);
if (!pyproject.includes(`Repository = "${identity.links.repository}"`)) fail("Repository URL is not sourced from [project.urls]");
if (!readme.includes(`**Live Docs**: ${identity.links.documentation}`)) fail("Documentation URL is not sourced from README.md");
if (identity.qr.url !== identity.links.repository || !identity.qr.label) fail("QR must be the labelled repository/project link");
if (identity.paperUrl !== null) fail("Paper URL must remain null until a publication URL exists");
if (identity.output.widthMm !== 594 || identity.output.heightMm !== 841 || identity.output.format !== "A1 portrait") fail("Output contract is not A1 portrait 594 × 841 mm");

const methodTable = tableBlock(main, "tab:methods");
function methodNameByCitation(citation) {
  const match = requireMatch(methodTable, new RegExp(`&\\s*([^&]+?)\\s*&\\s*\\\\cite\\{${escapeRegex(citation)}\\}`), `tab:methods citation ${citation}`);
  return cleanIdentityTex(match[1]);
}
const canonicalNames = new Map([
  ["wachter2017counterfactual", methodNameByCitation("wachter2017counterfactual")],
  ["bewley2024tcrex", methodNameByCitation("bewley2024tcrex")],
  ["mothilal2020explaining", methodNameByCitation("mothilal2020explaining")],
  ["kavouras2024glance", methodNameByCitation("kavouras2024glance")],
  ["dataset.give-me-some-credit", requireMatch(supplementary, /German Credit, ([A-Za-z ]+), Law, and Lending Club/, "Give Me Some Credit canonical dataset name")[1].trim()]
]);
const sourceCorpus = [main, supplementary, ...tableTexts].join("\n");
const aliases = new Map();
const sourceKeys = new Set();
for (const method of names.methods) {
  if (!method.display || !method.aliases?.length || !method.sourceKey) fail("Method-name entry lacks display name, aliases, or source key");
  if (sourceKeys.has(method.sourceKey)) fail(`Duplicate method-name source key: ${method.sourceKey}`);
  sourceKeys.add(method.sourceKey);
  const sourcedDisplay = canonicalNames.get(method.sourceKey);
  if (!sourcedDisplay) fail(`Unknown method-name source key: ${method.sourceKey}`);
  if (method.display !== sourcedDisplay) fail(`Display name for ${method.sourceKey} is not source-derived: expected ${sourcedDisplay}`);
  for (const alias of method.aliases) {
    if (aliases.has(alias)) fail(`Duplicate source alias: ${alias}`);
    if (!new RegExp(`(?<![A-Za-z0-9_-])${escapeRegex(alias)}(?![A-Za-z0-9_-])`).test(sourceCorpus)) fail(`Alias is not present in a manuscript source: ${alias}`);
    aliases.set(alias, method.display);
  }
}
if (sourceKeys.size !== canonicalNames.size || [...canonicalNames.keys()].some((key) => !sourceKeys.has(key))) fail("Method-name registry does not cover each required source identity exactly once");
for (const required of ["WACH", "Wachter", "TCREx", "T-CREx", "TCREX", "DICE", "DiCE", "GLANCE", "GlobalGLANCE", "GMC", "Give Me Some Credit"]) {
  if (!aliases.has(required)) fail(`Missing normalized alias: ${required}`);
}

const ids = new Set();
for (const claim of claims.claims) {
  if (ids.has(claim.id)) fail(`Duplicate claim ID: ${claim.id}`);
  ids.add(claim.id);
  for (const field of ["claimKind", "posterWording", "verdict", "source", "extractionRule", "direction", "qualifier", "status"]) {
    if (claim[field] === undefined || claim[field] === null || claim[field] === "") fail(`${claim.id} lacks ${field}`);
  }
  const sourceFile = path.join(rootDir, claim.source.file);
  try { await readFile(sourceFile, "utf8"); } catch { fail(`${claim.id} has no live source file: ${claim.source.file}`); }
  if (claim.value?.kind === "non_finite" && claim.status === "publishable") fail(`${claim.id} presents a non-finite value as publishable`);
  if (claim.value?.kind === "contradictory" && !/Resolution:/i.test(claim.qualifier)) fail(`${claim.id} has an unresolved contradiction`);
  if (claim.status === "contradictory" && !/omit/i.test(claim.qualifier)) fail(`${claim.id} contradiction is not resolved by omission or qualification`);
}

const allowedScopeCounts = new Set(["scope.datasets", "scope.methods", "scope.backbones", "scope.folds", "scope.metrics"]);
const metricHeadingRow = tableBlock(tableTexts[0], 'tab:cat_metrics_mlp').split('\n').find((line) => line.includes('Method') && line.includes('Cov.'));
if (!metricHeadingRow || claims.claims.find((claim) => claim.id === 'scope.metrics')?.value.total !== metricHeadingRow.split('&').length - 2) fail('Reported metric count does not match the live classification table heading');
const allowedResultMetrics = new Set(["coverage", "validity", "sparsity", "probabilistic_plausibility", "log_density", "lof", "isolation_forest", "l2_hamming", "time_s", "mae", "l2", "l1"]);
const resultMetricOrder = new Map([
  ["tab:cat_metrics_mlp", ["coverage", "validity", "sparsity", "probabilistic_plausibility", "log_density", "lof", "isolation_forest", "l2_hamming", "time_s"]],
  ["tab:num_metrics_mlp", ["coverage", "validity", "sparsity", "probabilistic_plausibility", "log_density", "lof", "isolation_forest", "l2_hamming", "time_s"]],
  ["tab:global_metrics_mlp", ["coverage", "validity", "sparsity", "probabilistic_plausibility", "log_density", "lof", "isolation_forest", "l2_hamming", "time_s"]],
  ["tab:group_metrics_mlp", ["coverage", "validity", "sparsity", "probabilistic_plausibility", "log_density", "lof", "isolation_forest", "l2_hamming", "time_s"]],
  ["tab:regression_all_dnn", ["mae", "probabilistic_plausibility", "l2", "lof", "isolation_forest", "log_density", "l1", "time_s"]]
]);
const resultClaims = claims.claims.filter((claim) => claim.claimKind === "benchmark-result");
if (resultClaims.length === 0) fail("No selected result claims found");
for (const claim of resultClaims) {
  const anchor = claim.source.anchor.match(/^(tab:[^/]+) \/ ([^/]+) \/ ([^/]+) \/ ([a-z0-9_]+)$/);
  if (!claim.source.file.startsWith("manuscript/tables/") || !anchor) fail(`${claim.id} is not anchored to a manuscript table row and metric`);
  const [, tableLabel, dataset, method, metric] = anchor;
  if (!allowedResultMetrics.has(metric)) fail(`${claim.id} names an unknown result metric: ${metric}`);
  const sourceText = await readFile(path.join(rootDir, claim.source.file), "utf8");
  const sourceTable = tableBlock(sourceText, tableLabel);
  const cells = tableRowCells(sourceTable, dataset, method);
  if (!cells) fail(`${claim.id} row anchor does not exist in ${claim.source.file}`);
  const metricOrder = resultMetricOrder.get(tableLabel);
  const metricIndex = metricOrder?.indexOf(metric) ?? -1;
  if (metricIndex < 0) fail(`${claim.id} metric is not part of the independently known schema for ${tableLabel}`);
  const measuredValue = parseResultCell(cells[metricIndex + 2]);
  if (JSON.stringify(claim.value) !== JSON.stringify(measuredValue)) {
    fail(`${claim.id} value does not equal its independently parsed source cell`);
  }
  if (claim.extractionRule !== `Parse the ${metric} cell from the named TeX table row.`) {
    fail(`${claim.id} extraction rule does not describe its measured source cell`);
  }
}
for (const claim of claims.claims.filter((item) => item.value?.kind === "finite")) {
  if (claim.claimKind === "benchmark-result") continue;
  if (claim.claimKind !== "scope-count" || !allowedScopeCounts.has(claim.id)) fail(`${claim.id} is a finite numeric claim without benchmark-row provenance or an allowed scope-count category`);
}
for (const claim of claims.claims.filter((item) => item.claimKind === "scope-count")) {
  if (!allowedScopeCounts.has(claim.id) || claim.value?.kind !== "finite") fail(`${claim.id} is not a recognized finite scope count`);
}

const counts = claims.claims.reduce((acc, claim) => {
  acc[claim.status] = (acc[claim.status] ?? 0) + 1;
  return acc;
}, {});
console.log(`Validated ${claims.claims.length} claims: ${Object.entries(counts).map(([key, value]) => `${key}=${value}`).join(", ")}`);
