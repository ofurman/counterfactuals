#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
export const rootDir = path.resolve(scriptDir, "../../..");

const expectedMetricColumns = {
  classification: ["coverage", "validity", "sparsity", "probabilistic_plausibility", "log_density", "lof", "isolation_forest", "l2_hamming", "time_s"],
  regression: ["mae", "probabilistic_plausibility", "l2", "lof", "isolation_forest", "log_density", "l1", "time_s"]
};

const metricHeadings = new Map([
  ["Cov.$\\uparrow$", "coverage"],
  ["Valid.$\\uparrow$", "validity"],
  ["Spars.$\\uparrow$", "sparsity"],
  ["Sparse.$\\uparrow$", "sparsity"],
  ["Prob. Plaus.$\\uparrow$", "probabilistic_plausibility"],
  ["Log Dens.$\\uparrow$", "log_density"],
  ["LOF$\\downarrow$", "lof"],
  ["IsoForest$\\uparrow$", "isolation_forest"],
  ["L2-Ham.$\\downarrow$", "l2_hamming"],
  ["L2$\\downarrow$", "l2_hamming"],
  ["Time(s)$\\downarrow$", "time_s"],
  ["MAE $\\downarrow$", "mae"],
  ["PP $\\uparrow$", "probabilistic_plausibility"],
  ["L2 $\\downarrow$", "l2"],
  ["LOF $\\downarrow$", "lof"],
  ["IF $\\uparrow$", "isolation_forest"],
  ["LD $\\uparrow$", "log_density"],
  ["L1 $\\downarrow$", "l1"],
  ["Time $\\downarrow$", "time_s"]
]);

const expectedHeadingText = {
  "tab:cat_metrics_mlp": ["Cov.$\\uparrow$", "Valid.$\\uparrow$", "Spars.$\\uparrow$", "Prob. Plaus.$\\uparrow$", "Log Dens.$\\uparrow$", "LOF$\\downarrow$", "IsoForest$\\uparrow$", "L2-Ham.$\\downarrow$", "Time(s)$\\downarrow$"],
  "tab:num_metrics_mlp": ["Cov.$\\uparrow$", "Valid.$\\uparrow$", "Sparse.$\\uparrow$", "Prob. Plaus.$\\uparrow$", "Log Dens.$\\uparrow$", "LOF$\\downarrow$", "IsoForest$\\uparrow$", "L2$\\downarrow$", "Time(s)$\\downarrow$"],
  "tab:global_metrics_mlp": ["Cov.$\\uparrow$", "Valid.$\\uparrow$", "Sparse.$\\uparrow$", "Prob. Plaus.$\\uparrow$", "Log Dens.$\\uparrow$", "LOF$\\downarrow$", "IsoForest$\\uparrow$", "L2-Ham.$\\downarrow$", "Time(s)$\\downarrow$"],
  "tab:group_metrics_mlp": ["Cov.$\\uparrow$", "Valid.$\\uparrow$", "Sparse.$\\uparrow$", "Prob. Plaus.$\\uparrow$", "Log Dens.$\\uparrow$", "LOF$\\downarrow$", "IsoForest$\\uparrow$", "L2-Ham.$\\downarrow$", "Time(s)$\\downarrow$"],
  "tab:regression_all_dnn": ["MAE $\\downarrow$", "PP $\\uparrow$", "L2 $\\downarrow$", "LOF $\\downarrow$", "IF $\\uparrow$", "LD $\\uparrow$", "L1 $\\downarrow$", "Time $\\downarrow$"]
};

const directions = {
  coverage: "higher",
  validity: "higher",
  sparsity: "excluded: manuscript direction is contradictory",
  probabilistic_plausibility: "higher",
  log_density: "higher only within the same dataset and model",
  lof: "lower",
  isolation_forest: "higher",
  l2_hamming: "lower",
  mae: "lower",
  l2: "lower",
  l1: "lower",
  time_s: "lower"
};

async function text(relativePath) {
  return readFile(path.join(rootDir, relativePath), "utf8");
}

function requireMatch(input, regex, description) {
  const match = input.match(regex);
  if (!match) throw new Error(`Missing source anchor: ${description}`);
  return match;
}

function cleanTex(value) {
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

function parseMeasurement(cell) {
  const clean = cleanTex(cell);
  if (clean === "--") return {kind: "missing", display: "—", mean: null, std: null};
  if (/infty/i.test(clean)) return {kind: "non_finite", display: clean, mean: null, std: null};
  const match = clean.match(/^(-?(?:\d+(?:\.\d+)?|\.\d+)(?:e-?\d+)?)(?:±(-?(?:\d+(?:\.\d+)?|\.\d+)(?:e-?\d+)?))?$/i);
  if (!match) throw new Error(`Cannot parse measurement cell: ${cell} -> ${clean}`);
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

function tableBlock(input, label) {
  const labelToken = `\\label{${label}}`;
  const labelAt = input.indexOf(labelToken);
  if (labelAt < 0) throw new Error(`Missing table label ${label}`);
  const start = input.lastIndexOf("\\begin{table", labelAt);
  const end = input.indexOf("\\end{table", labelAt);
  if (start < 0 || end < 0) throw new Error(`Malformed table ${label}`);
  return input.slice(start, input.indexOf("\n", end) + 1);
}

function parseHeading(cell) {
  const heading = cell.replace(/\\textbf\{([^{}]+)\}/g, "$1").replace(/\s+/g, " ").trim();
  const metric = metricHeadings.get(heading);
  if (!metric) throw new Error(`Unknown or renamed metric heading: ${heading}`);
  return metric;
}

function parseRows(block, kind, label) {
  const headerLine = block.split("\n").find((rawLine) => {
    const line = rawLine.trim();
    return line.endsWith("\\\\") && line.includes("&") && /(?:^|&)\s*(?:\\textbf\{)?Method(?:\})?\s*&/.test(line);
  });
  if (!headerLine) throw new Error("Missing metric table heading row");
  const headerCells = headerLine.trim().slice(0, -2).split("&").map((cell) => cell.trim());
  const datasetHeading = headerCells[0].replace(/\\textbf\{([^{}]+)\}/g, "$1").trim();
  const methodHeading = headerCells[1].replace(/\\textbf\{([^{}]+)\}/g, "$1").trim();
  if (datasetHeading !== "" && datasetHeading !== "Dataset") throw new Error(`Unexpected dataset heading: ${datasetHeading}`);
  if (methodHeading !== "Method") throw new Error(`Unexpected method heading: ${methodHeading}`);
  const headingText = headerCells.slice(2).map((cell) => cell.replace(/\\textbf\{([^{}]+)\}/g, "$1").replace(/\s+/g, " ").trim());
  const requiredHeadingText = expectedHeadingText[label];
  if (!requiredHeadingText || headingText.length !== requiredHeadingText.length || headingText.some((heading, index) => heading !== requiredHeadingText[index])) {
    throw new Error(`Metric heading text mismatch for ${label}: ${headingText.join(" | ")}`);
  }
  const columns = headingText.map(parseHeading);
  const expected = expectedMetricColumns[kind];
  if (columns.length !== expected.length || columns.some((column, index) => column !== expected[index])) {
    throw new Error(`Metric heading schema mismatch for ${kind}: ${columns.join(", ")}`);
  }

  const rows = [];
  let dataset = null;
  for (const rawLine of block.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("%") || !line.endsWith("\\\\")) continue;
    if (!line.includes("&")) continue;
    const cells = line.slice(0, -2).split("&").map((cell) => cell.trim());
    if (rawLine === headerLine || /Dataset|Method|Cov\.|MAE/.test(cells.join(" "))) continue;
    const first = cells[0];
    const datasetMatch = first.match(/\\multirow(?:\[[^\]]+\])?\{[^}]+\}\{(?:\*|[^}]+)\}\{([^}]+)\}/);
    if (datasetMatch) dataset = cleanTex(datasetMatch[1]);
    if (!dataset) continue;
    const method = cleanTex(cells[1]);
    const metrics = {};
    columns.forEach((column, index) => {
      metrics[column] = parseMeasurement(cells[index + 2]);
    });
    rows.push({dataset, method, metrics, raw: line});
  }
  return rows;
}

function findRow(parsedTables, tableLabel, dataset, method) {
  const row = parsedTables[tableLabel].find((item) => item.dataset === dataset && item.method === method);
  if (!row) throw new Error(`Missing row ${tableLabel} / ${dataset} / ${method}`);
  return row;
}

function finiteMean(value, description) {
  if (value?.kind !== "finite" || !Number.isFinite(value.mean)) throw new Error(`Comparative claim requires a finite value: ${description}`);
  return value.mean;
}

function comparisonContext(parsedTables, spec, row, value) {
  const peers = parsedTables[spec.table].filter((item) => item.dataset === spec.dataset);
  const metric = (method, metricName = spec.metric) => {
    const peer = peers.find((item) => item.method === method);
    if (!peer) throw new Error(`Missing comparison peer ${spec.table} / ${spec.dataset} / ${method}`);
    return peer.metrics[metricName];
  };
  const finitePeers = (metricName = spec.metric) => peers
    .map((peer) => ({method: peer.method, value: peer.metrics[metricName]}))
    .filter((peer) => peer.value.kind === "finite");
  return {row, value, peers, metric, finitePeers};
}

function highestVerdict(ctx, subject, metricLabel) {
  const target = finiteMean(ctx.value, subject);
  const ranked = ctx.finitePeers().map((peer) => ({...peer, mean: finiteMean(peer.value, peer.method)})).sort((a, b) => b.mean - a.mean);
  if (target !== ranked[0]?.mean) throw new Error(`${subject} is no longer highest for ${metricLabel}`);
  const next = ranked.find((peer) => peer.mean < target);
  const comparison = next ? `; next is ${next.method} at ${next.value.display}` : "";
  return `${subject} has the highest displayed ${metricLabel} mean among ${ranked.length} methods (${ctx.value.display}${comparison}).`;
}

function roundedZeroVerdict(ctx, subject) {
  if (finiteMean(ctx.value, subject) !== 0) throw new Error(`${subject} no longer has a displayed mean of zero`);
  return `${subject}'s displayed mean rounds to zero.`;
}

function resultClaim(parsedTables, spec) {
  const row = findRow(parsedTables, spec.table, spec.dataset, spec.method);
  const value = row.metrics[spec.metric];
  if (!value) throw new Error(`Missing metric ${spec.metric} in ${spec.id}`);
  const context = comparisonContext(parsedTables, spec, row, value);
  const verdict = typeof spec.verdict === "function" ? spec.verdict(context) : spec.verdict;
  const qualifierParts = [spec.qualifier];
  if (value.roundedZero) qualifierParts.push("Reported as 0.00 ± 0.00 after rounding; not asserted to be an exact zero.");
  if (value.kind === "missing") qualifierParts.push("The source reports a missing-value marker; this is missing output, not proof of method inapplicability.");
  return {
    id: spec.id,
    claimKind: "benchmark-result",
    posterWording: spec.wording(value),
    value,
    unit: spec.unit ?? null,
    verdict,
    source: {file: spec.file, anchor: `${spec.table} / ${spec.dataset} / ${spec.method} / ${spec.metric}`},
    extractionRule: `Parse the ${spec.metric} cell from the named TeX table row.`,
    direction: directions[spec.metric],
    qualifier: qualifierParts.filter(Boolean).join(" "),
    status: spec.status ?? (value.kind === "finite" ? "publishable" : "qualified")
  };
}

export async function buildClaims() {
  const main = await text("manuscript/main_lncs.tex");
  const supplementary = await text("manuscript/supplementary.tex");
  const tableFiles = {
    categorical: "manuscript/tables/results_categorical.tex",
    numerical: "manuscript/tables/results_numerical.tex",
    global: "manuscript/tables/results_global.tex",
    group: "manuscript/tables/results_group.tex",
    regression: "manuscript/tables/results_regression.tex"
  };
  const tableText = Object.fromEntries(await Promise.all(Object.entries(tableFiles).map(async ([key, file]) => [key, await text(file)])));
  const tableDefinitions = [
    ["tab:cat_metrics_mlp", "categorical", "classification"],
    ["tab:num_metrics_mlp", "numerical", "classification"],
    ["tab:global_metrics_mlp", "global", "classification"],
    ["tab:group_metrics_mlp", "group", "classification"],
    ["tab:regression_all_dnn", "regression", "regression"]
  ];
  const parsedTables = {};
  for (const [label, fileKey, kind] of tableDefinitions) {
    parsedTables[label] = parseRows(tableBlock(tableText[fileKey], label), kind, label);
  }

  const datasetScope = requireMatch(main, /CEL includes (\d+) pre-configured datasets covering classification \((\d+)\) and regression \((\d+)\) tasks/, "dataset scope");
  const methodScope = requireMatch(main, /The library implements (\d+) counterfactual explanation methods, categorized into local, global, and group-wise approaches/, "method scope");
  const backboneScope = requireMatch(main, /using (\w+) predictive backbones per task type/, "predictive backbone scope");
  const foldScope = requireMatch(supplementary, /averaged over (\d+)-fold cross-validation/, "cross-validation scope");
  requireMatch(main, /identifying minimal changes required to alter a model’s prediction/, "counterfactual explanation definition");
  requireMatch(
    main,
    /protocol-level control across datasets, predictive backbones, preprocessing steps, constraint handling, and metric definitions/,
    "controlled protocol components"
  );
  const conclusionScope = requireMatch(
    main,
    /\\section\{Conclusions\}[\s\S]*?Our benchmark of (\d+) methods on (\d+) datasets reveals that no single method uniformly dominates; each excels on some quality dimensions at the expense of others\./,
    "trade-off conclusion"
  );
  requireMatch(
    main,
    /\\item \\textbf\{A controlled evaluation protocol\} for counterfactual explanations that standardizes datasets, preprocessing, predictive backbones, constraint handling, and metric definitions, enabling fair and reproducible comparison\./,
    "controlled-protocol contribution"
  );
  requireMatch(
    main,
    /\\item \\textbf\{A benchmark including\} 18 datasets and 14 counterfactual generation methods implemented within a unified framework, supporting evaluation across local, global, and group-wise paradigms\./,
    "benchmark-breadth contribution"
  );
  requireMatch(
    main,
    /\\item \\textbf\{An open-source programming library\} designed to facilitate future method integration, transparent reporting, and community-driven benchmarking\./,
    "open-source contribution"
  );
  requireMatch(
    main,
    /The consistency of CCHVAE and PPCEF makes them suitable for applications where validity is the primary concern\.[\s\S]*?CADEX generates the closest counterfactuals[\s\S]*?PPCEF and CCHVAE demonstrate the high log-density/,
    "local benchmark result summary"
  );
  requireMatch(
    main,
    /For regression tasks[\s\S]*?WACH consistently outperforms CEARM\.[\s\S]*?WACH produces significantly more plausible counterfactuals[\s\S]*?nearly 9[\s\S]{0,20}faster/,
    "regression benchmark result summary"
  );
  requireMatch(
    main,
    /GLOBE-CE and GLANCE achieve perfect or near-perfect validity while AReS shows only moderate success rates[\s\S]*?AReS shows the lowest log-density/,
    "global benchmark result summary"
  );
  requireMatch(
    main,
    /GLANCE consistently achieves higher validity, but T-CREx, when applicable, produces closer and more plausible counterfactuals, however with very low success rates\./,
    "group-wise benchmark result summary"
  );
  const sparsityDefinition = requireMatch(
    supplementary,
    /\\item \\textbf\{Sparsity:\} The average proportion of features modified to achieve the counterfactual\.[\s\S]*?\\text\{Sparsity\}\(x, x'\) = \\frac\{1\}\{d\} \\sum_\{i=1\}\^\{d\} \\mathbb\{I\}\(x_i \\neq x'_i\)/,
    "sparsity modified-feature definition and equation"
  );
  const sparsityProse = requireMatch(
    main,
    /DICE achieves the lowest sparsity, followed by CEGP and CADEX\./,
    "sparsity prose interpretation"
  );
  const sparsityTableDefinitions = tableDefinitions.filter(([, , kind]) => kind === "classification");
  const sparsityHeadings = sparsityTableDefinitions.map(([label]) => {
    const block = tableBlock(tableText[tableDefinitions.find(([candidate]) => candidate === label)[1]], label);
    return requireMatch(block, /(?:Spars\.|Sparse\.)\$\\uparrow\$/, `${label} sparsity heading`)[0];
  });
  if (!sparsityDefinition || !sparsityProse || sparsityHeadings.length !== sparsityTableDefinitions.length) {
    throw new Error("Could not derive the sparsity-direction contradiction from live sources");
  }

  const datasetCounts = {
    total: Number(datasetScope[1]),
    classification: Number(datasetScope[2]),
    regression: Number(datasetScope[3])
  };
  if (datasetCounts.classification + datasetCounts.regression !== datasetCounts.total) {
    throw new Error("Dataset task counts do not sum to the declared total");
  }
  const declaredMethodTotal = Number(methodScope[1]);
  const numberWords = new Map([["one", 1], ["two", 2], ["three", 3], ["four", 4], ["five", 5]]);
  const backboneCount = numberWords.get(backboneScope[1].toLowerCase());
  if (!backboneCount) throw new Error(`Unsupported predictive backbone count: ${backboneScope[1]}`);
  const methodTable = tableBlock(main, "tab:methods");
  const methodCategoryCounts = {Local: 0, Global: 0, "Group-wise": 0};
  let currentMethodCategory = null;
  for (const rawLine of methodTable.split("\n")) {
    const line = rawLine.trim();
    const category = line.match(/\\multirow\{\d+\}\{\*\}\{(Local|Global|Group-wise)\}/)?.[1];
    if (category) currentMethodCategory = category;
    if (currentMethodCategory && line.includes("&") && /\\cite\{/.test(line)) {
      methodCategoryCounts[currentMethodCategory] += 1;
    }
  }
  if (!methodCategoryCounts.Local || !methodCategoryCounts.Global || !methodCategoryCounts["Group-wise"]) {
    throw new Error("Could not derive all method category counts from tab:methods");
  }
  const countedMethodTotal = Object.values(methodCategoryCounts).reduce((sum, count) => sum + count, 0);
  if (countedMethodTotal !== declaredMethodTotal) {
    throw new Error(`Method category counts (${countedMethodTotal}) do not match declared total (${declaredMethodTotal})`);
  }
  const conclusionMethodTotal = Number(conclusionScope[1]);
  const conclusionDatasetTotal = Number(conclusionScope[2]);
  if (conclusionMethodTotal !== declaredMethodTotal || conclusionDatasetTotal !== datasetCounts.total) {
    throw new Error(
      `Conclusion scope (${conclusionMethodTotal} methods, ${conclusionDatasetTotal} datasets) does not match the benchmark declarations (${declaredMethodTotal}, ${datasetCounts.total})`
    );
  }
  const foldCount = Number(foldScope[1]);

  const claims = [
    {
      id: "concept.counterfactual",
      claimKind: "qualitative",
      posterWording: "A small input change that flips a model's prediction.",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "Counterfactual explanations identify input changes that produce a desired model prediction.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Introduction > identifying minimal changes required to alter a model’s prediction"},
      extractionRule: "Require the introductory definition of counterfactual explanations.",
      direction: "qualitative",
      qualifier: "The loan application example uses invented profiles and a consistency rule recorded separately in ce-example.json; it is not a manuscript experiment or lending recommendation.",
      status: "publishable"
    },
    {
      id: "scope.datasets",
      claimKind: "scope-count",
      posterWording: `${datasetCounts.total} datasets: ${datasetCounts.classification} classification + ${datasetCounts.regression} regression`,
      value: {kind: "finite", classification: datasetCounts.classification, regression: datasetCounts.regression, total: datasetCounts.total},
      unit: "datasets",
      verdict: "CEL spans both classification and regression benchmarks.",
      source: {file: "manuscript/main_lncs.tex", anchor: `Benchmark > Datasets: ‘CEL includes ${datasetCounts.total} pre-configured datasets…’`},
      extractionRule: "Regex-extract total and task counts from the Datasets subsection.",
      direction: "scope",
      qualifier: "Dataset counts, not the number of train/test folds.",
      status: "publishable"
    },
    {
      id: "scope.methods",
      claimKind: "scope-count",
      posterWording: `${declaredMethodTotal} methods across local, global, and group-wise explanations`,
      value: {kind: "finite", total: declaredMethodTotal, local: methodCategoryCounts.Local, global: methodCategoryCounts.Global, groupWise: methodCategoryCounts["Group-wise"]},
      unit: "methods",
      verdict: "All three counterfactual paradigms share one benchmark protocol.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Benchmark > Methods and tab:methods"},
      extractionRule: "Read the declared method total and count the method rows under each tab:methods category.",
      direction: "scope",
      qualifier: `GLANCE also appears in the global experiment configured as one group; the ${declaredMethodTotal} unique-method count follows tab:methods.`,
      status: "qualified"
    },
    {
      id: "scope.protocol",
      claimKind: "qualitative",
      posterWording: "One controlled protocol: fixed splits, preprocessing, predictors, constraints, and metrics",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "CEL reduces experimental setup as a confounder in method comparisons.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Benchmark opening paragraph"},
      extractionRule: "Require the standardized-evaluation-protocol paragraph and its enumerated controls.",
      direction: "qualitative",
      qualifier: "A design objective of the benchmark, not a statistical guarantee that every source of variation is eliminated.",
      status: "qualified"
    },
    {
      id: "scope.backbones",
      claimKind: "scope-count",
      posterWording: `${backboneCount} predictive backbones per task type`,
      value: {kind: "finite", total: backboneCount},
      unit: "backbones",
      verdict: `Each task type is evaluated with ${backboneCount} predictive backbones.`,
      source: {file: "manuscript/main_lncs.tex", anchor: "Benchmark opening paragraph"},
      extractionRule: "Regex-extract the predictive-backbone count from the benchmark scope statement.",
      direction: "scope",
      qualifier: "Classification and regression use task-appropriate backbone pairs.",
      status: "publishable"
    },
    {
      id: "scope.folds",
      claimKind: "scope-count",
      posterWording: `Results averaged over ${foldCount}-fold cross-validation`,
      value: {kind: "finite", total: foldCount},
      unit: "folds",
      verdict: `Reported table values summarize ${foldCount} folds.`,
      source: {file: "manuscript/supplementary.tex", anchor: "Full Results opening paragraph"},
      extractionRule: "Regex-extract the fold count from the Full Results introduction.",
      direction: "scope",
      qualifier: "The displayed ± values are those printed in the manuscript tables.",
      status: "publishable"
    },
    {
      id: "conclusion.tradeoffs",
      claimKind: "qualitative",
      posterWording: "No method wins every metric",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "Counterfactual quality is a trade-off among success, change size, plausibility, applicability, and runtime.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Conclusions: ‘no single method uniformly dominates; each excels on some quality dimensions at the expense of others’"},
      extractionRule: "Require the conclusion sentence and retain its multi-dimensional qualification.",
      direction: "qualitative",
      qualifier: "Do not convert this conclusion into an aggregate ranking.",
      status: "qualified"
    },
    {
      id: "contribution.protocol",
      claimKind: "qualitative",
      posterWording: "Controlled evaluation protocol",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "CEL standardizes datasets, preprocessing, predictive backbones, constraints, and metric definitions for fairer CE comparison.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Introduction > Our main contributions > controlled evaluation protocol"},
      extractionRule: "Require the first contribution item and its complete standardized-control list.",
      direction: "qualitative",
      qualifier: "This is the benchmark design contribution, not a guarantee that all external variation is removed.",
      status: "publishable"
    },
    {
      id: "contribution.benchmark",
      claimKind: "qualitative",
      posterWording: "Broad benchmark across CE paradigms",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "CEL benchmarks local, global, and group-wise counterfactual explanations inside one framework.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Introduction > Our main contributions > benchmark"},
      extractionRule: "Require the second contribution item and its dataset, method, and paradigm scope.",
      direction: "qualitative",
      qualifier: "Exact dataset and method totals remain owned by the scope-count claims.",
      status: "publishable"
    },
    {
      id: "contribution.library",
      claimKind: "qualitative",
      posterWording: "Extensible open-source benchmark workbench",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "The CEL library supports method integration, transparent reporting, and community-driven benchmarking.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Introduction > Our main contributions > open-source programming library"},
      extractionRule: "Require the third contribution item and its extension and reporting objectives.",
      direction: "qualitative",
      qualifier: "An implementation and extensibility contribution alongside the benchmark protocol.",
      status: "publishable"
    },
    {
      id: "result.local.overview",
      claimKind: "qualitative",
      posterWording: "Local CE quality is multi-dimensional",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "Local methods trade validity and proximity against plausibility and runtime; high performance on one axis does not imply dominance elsewhere.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Results > Local Methods and fig:local_methods"},
      extractionRule: "Require the local-results paragraph that contrasts validity, proximity, density, sparsity, and runtime.",
      direction: "qualitative",
      qualifier: "Summary of the manuscript figure and narrative across the representative local-method datasets and both classification backbones.",
      status: "qualified"
    },
    {
      id: "result.regression.overview",
      claimKind: "qualitative",
      posterWording: "Regression CE performance separates target error from recourse cost",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "In the manuscript regression comparison, Wachter pairs comparable target accuracy with smaller changes, stronger plausibility, and lower runtime than CEARM.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Results > Regression Methods and fig:regression_methods"},
      extractionRule: "Require the regression-results paragraph comparing accuracy, change size, plausibility, and runtime.",
      direction: "qualitative",
      qualifier: "Aggregate manuscript result across the representative regression datasets; inspect per-dataset panels for variation.",
      status: "qualified"
    },
    {
      id: "result.global.overview",
      claimKind: "qualitative",
      posterWording: "Global CE methods expose validity and applicability failure modes",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "The manuscript reports stronger validity for GLOBE-CE and GlobalGLANCE, while AReS is less reliable and frequently lacks results.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Results > Global Methods and fig:global_methods"},
      extractionRule: "Require the global-results paragraph contrasting validity, distance, density, and method success.",
      direction: "qualitative",
      qualifier: "Aggregate manuscript result across representative datasets and both classification backbones; missing output is not interpreted as zero.",
      status: "qualified"
    },
    {
      id: "result.group.overview",
      claimKind: "qualitative",
      posterWording: "Group-wise CEs trade effectiveness for minimally disruptive shifts",
      value: {kind: "qualitative"},
      unit: null,
      verdict: "GLANCE achieves higher validity, whereas T-CREx produces closer and more plausible shifts when it applies, but with very low success rates.",
      source: {file: "manuscript/main_lncs.tex", anchor: "Results > Group-wise Methods and fig:group_methods"},
      extractionRule: "Require the group-wise-results sentence that contrasts validity, proximity, plausibility, and applicability.",
      direction: "qualitative",
      qualifier: "Aggregate manuscript result across representative datasets and both classification backbones.",
      status: "qualified"
    }
  ];

  const specs = [
    {id: "local.adult.sace.pp", file: tableFiles.categorical, table: "tab:cat_metrics_mlp", dataset: "Adult Census", method: "SACE", metric: "probabilistic_plausibility", verdict: (ctx) => highestVerdict(ctx, "SACE", "probabilistic-plausibility"), qualifier: "Adult Census, MLP; comparison is within this dataset/model only.", wording: (v) => `Adult / local: SACE plausibility ${v.display}`},
    {id: "local.adult.cadex.validity", file: tableFiles.categorical, table: "tab:cat_metrics_mlp", dataset: "Adult Census", method: "CADEX", metric: "validity", verdict: (ctx) => {
      const mean = finiteMean(ctx.value, "CADEX validity");
      if (Math.abs(mean - 0.5) > 0.1) throw new Error("CADEX validity is no longer roughly one half");
      return `CADEX's displayed validity mean is ${ctx.value.display}, roughly half, in this configuration.`;
    }, qualifier: "Adult Census, MLP. Validity must be read with coverage.", wording: (v) => `Adult / local: CADEX validity ${v.display}`},
    {id: "local.blobs.ppcef.pp", file: tableFiles.numerical, table: "tab:num_metrics_mlp", dataset: "Blobs", method: "PPCEF", metric: "probabilistic_plausibility", verdict: (ctx) => highestVerdict(ctx, "PPCEF", "probabilistic-plausibility"), qualifier: "Blobs, MLP; compare only within this dataset/model.", wording: (v) => `Blobs / local: PPCEF plausibility ${v.display}`},
    {id: "local.blobs.ppcef.l2", file: tableFiles.numerical, table: "tab:num_metrics_mlp", dataset: "Blobs", method: "PPCEF", metric: "l2_hamming", verdict: (ctx) => {
      const target = finiteMean(ctx.value, "PPCEF distance");
      const ranked = ctx.finitePeers().map((peer) => ({...peer, mean: finiteMean(peer.value, peer.method)})).sort((a, b) => a.mean - b.mean);
      if (!(target > ranked[0].mean)) throw new Error("PPCEF distance is no longer larger than the row-group minimum");
      return `PPCEF's displayed distance (${ctx.value.display}) is larger than the row-group minimum, ${ranked[0].value.display} for ${ranked[0].method}.`;
    }, qualifier: "Blobs has numerical features; the source table heading is L2.", wording: (v) => `Blobs / local: PPCEF distance ${v.display}`},
    {id: "global.adult.globe.validity", file: tableFiles.global, table: "tab:global_metrics_mlp", dataset: "Adult Census", method: "GLOBE-CE", metric: "validity", verdict: (ctx) => highestVerdict(ctx, "GLOBE-CE", "validity"), qualifier: "Adult Census, MLP; validity is conditional on the experiment's result semantics.", wording: (v) => `Adult / global: GLOBE-CE validity ${v.display}`},
    {id: "global.adult.globe.distance", file: tableFiles.global, table: "tab:global_metrics_mlp", dataset: "Adult Census", method: "GLOBE-CE", metric: "l2_hamming", verdict: (ctx) => roundedZeroVerdict(ctx, "GLOBE-CE distance"), qualifier: "Adult Census, MLP.", wording: (v) => `Adult / global: GLOBE-CE distance ${v.display}`},
    {id: "global.blobs.ares.missing", file: tableFiles.global, table: "tab:global_metrics_mlp", dataset: "Blobs", method: "AReS", metric: "coverage", verdict: (ctx) => {
      if (ctx.value.kind !== "missing") throw new Error("AReS Blobs coverage is no longer missing");
      return "The table does not report a coverage result for AReS in this configuration.";
    }, qualifier: "Blobs, MLP.", wording: () => "Blobs / global: AReS result not reported"},
    {id: "group.adult.tcrex.validity", file: tableFiles.group, table: "tab:group_metrics_mlp", dataset: "Adult Census", method: "TCREx", metric: "validity", verdict: (ctx) => {
      const target = finiteMean(ctx.value, "T-CREx validity");
      const glanceValidity = finiteMean(ctx.metric("GLANCE"), "GLANCE validity");
      const targetDistance = finiteMean(ctx.metric("TCREx", "l2_hamming"), "T-CREx distance");
      const glanceDistance = finiteMean(ctx.metric("GLANCE", "l2_hamming"), "GLANCE distance");
      if (!(target < glanceValidity && targetDistance < glanceDistance)) throw new Error("T-CREx/GLANCE validity-distance trade-off no longer holds");
      return `T-CREx's smaller displayed shift (${targetDistance.toFixed(2)}) accompanies lower validity (${ctx.value.display}) than GLANCE (${ctx.metric("GLANCE").display}).`;
    }, qualifier: "Adult Census, MLP.", wording: (v) => `Adult / group-wise: T-CREx validity ${v.display}`},
    {id: "group.adult.tcrex.distance", file: tableFiles.group, table: "tab:group_metrics_mlp", dataset: "Adult Census", method: "TCREx", metric: "l2_hamming", verdict: (ctx) => roundedZeroVerdict(ctx, "T-CREx distance"), qualifier: "Adult Census, MLP.", wording: (v) => `Adult / group-wise: T-CREx distance ${v.display}`},
    {id: "group.adult.glance.validity", file: tableFiles.group, table: "tab:group_metrics_mlp", dataset: "Adult Census", method: "GLANCE", metric: "validity", verdict: (ctx) => {
      const target = finiteMean(ctx.value, "GLANCE validity");
      const peer = finiteMean(ctx.metric("TCREx"), "T-CREx validity");
      if (!(target > peer)) throw new Error("GLANCE validity is no longer higher than T-CREx validity");
      return `GLANCE's displayed validity (${ctx.value.display}) is higher than T-CREx's (${ctx.metric("TCREx").display}).`;
    }, qualifier: "Adult Census, MLP.", wording: (v) => `Adult / group-wise: GLANCE validity ${v.display}`},
    {id: "group.adult.glance.distance", file: tableFiles.group, table: "tab:group_metrics_mlp", dataset: "Adult Census", method: "GLANCE", metric: "l2_hamming", verdict: (ctx) => {
      const target = finiteMean(ctx.value, "GLANCE distance");
      const peer = finiteMean(ctx.metric("TCREx"), "T-CREx distance");
      if (!(target > peer)) throw new Error("GLANCE distance is no longer larger than T-CREx distance");
      return `GLANCE's higher-validity result changes more (${ctx.value.display}) than T-CREx (${ctx.metric("TCREx").display}).`;
    }, qualifier: "Adult Census, MLP.", wording: (v) => `Adult / group-wise: GLANCE distance ${v.display}`},
    {id: "regression.concrete.cearm.mae", file: tableFiles.regression, table: "tab:regression_all_dnn", dataset: "Concrete", method: "CEARM", metric: "mae", verdict: (ctx) => {
      const difference = finiteMean(ctx.value, "CEARM MAE") - finiteMean(ctx.metric("WACH"), "Wachter MAE");
      if (!(difference > 0)) throw new Error("CEARM MAE is no longer higher than Wachter MAE");
      return `CEARM's displayed target-error mean is ${difference.toFixed(2)} higher than Wachter's on Concrete.`;
    }, qualifier: "Concrete, MLP regressor; regression validity is MAE, so lower is better.", wording: (v) => `Concrete: CEARM target MAE ${v.display}`},
    {id: "regression.concrete.wachter.mae", file: tableFiles.regression, table: "tab:regression_all_dnn", dataset: "Concrete", method: "WACH", metric: "mae", verdict: (ctx) => {
      const difference = finiteMean(ctx.metric("CEARM"), "CEARM MAE") - finiteMean(ctx.value, "Wachter MAE");
      if (!(difference > 0)) throw new Error("Wachter MAE is no longer lower than CEARM MAE");
      return `Wachter's displayed target-error mean is ${difference.toFixed(2)} lower than CEARM's on Concrete.`;
    }, qualifier: "Concrete, MLP regressor; regression validity is MAE, so lower is better.", wording: (v) => `Concrete: Wachter target MAE ${v.display}`},
    {id: "regression.concrete.cearm.l2", file: tableFiles.regression, table: "tab:regression_all_dnn", dataset: "Concrete", method: "CEARM", metric: "l2", verdict: (ctx) => {
      const peer = finiteMean(ctx.metric("WACH"), "Wachter L2");
      if (!(peer > 0)) throw new Error("Wachter L2 must be positive for the ratio comparison");
      const ratio = finiteMean(ctx.value, "CEARM L2") / peer;
      if (!(ratio > 1)) throw new Error("CEARM L2 is no longer larger than Wachter L2");
      return `CEARM's displayed perturbation is ${ratio.toFixed(1)}× Wachter's on Concrete.`;
    }, qualifier: "Concrete, MLP regressor.", wording: (v) => `Concrete: CEARM L2 ${v.display}`},
    {id: "regression.concrete.wachter.l2", file: tableFiles.regression, table: "tab:regression_all_dnn", dataset: "Concrete", method: "WACH", metric: "l2", verdict: (ctx) => {
      const target = finiteMean(ctx.value, "Wachter L2");
      if (!(target > 0)) throw new Error("Wachter L2 must be positive for the ratio comparison");
      const ratio = finiteMean(ctx.metric("CEARM"), "CEARM L2") / target;
      if (!(ratio > 1)) throw new Error("Wachter L2 is no longer smaller than CEARM L2");
      return `Wachter's displayed perturbation is ${ratio.toFixed(1)}× smaller than CEARM's on Concrete.`;
    }, qualifier: "Concrete, MLP regressor.", wording: (v) => `Concrete: Wachter L2 ${v.display}`},
    {id: "regression.concrete.wachter.pp", file: tableFiles.regression, table: "tab:regression_all_dnn", dataset: "Concrete", method: "WACH", metric: "probabilistic_plausibility", verdict: (ctx) => {
      const target = finiteMean(ctx.value, "Wachter plausibility");
      const peer = finiteMean(ctx.metric("CEARM"), "CEARM plausibility");
      if (!(target > peer)) throw new Error("Wachter plausibility is no longer higher than CEARM plausibility");
      return `Wachter's displayed plausibility mean (${ctx.value.display}) is higher than CEARM's (${ctx.metric("CEARM").display}).`;
    }, qualifier: "Concrete, MLP regressor.", wording: (v) => `Concrete: Wachter plausibility ${v.display}`}
  ];
  claims.push(...specs.map((spec) => resultClaim(parsedTables, spec)));
  claims.push({
    id: "caveat.sparsity-direction",
    claimKind: "caveat",
    posterWording: "Sparsity direction unresolved; no ranking.",
    value: {
      kind: "contradictory",
      definition: "fraction of features modified",
      definitionPreferredDirection: "lower",
      tableHeadingDirection: "higher",
      prosePreferredDirection: "lower"
    },
    unit: null,
    verdict: "Do not make a comparative sparsity claim on the poster.",
    source: {file: "manuscript/supplementary.tex", anchor: "metric definition vs. table Sparse.↑ headings and prose interpretation"},
    extractionRule: "Compare the sparsity definition with the table direction markers and manuscript prose.",
    direction: "contradictory",
    qualifier: "Resolution: omit any poster sparsity ranking; retain the manuscript figure without adding a comparative sparsity claim.",
    status: "contradictory"
  });

  return {
    schemaVersion: 1,
    generatedFrom: ["manuscript/main_lncs.tex", "manuscript/supplementary.tex", ...Object.values(tableFiles)],
    claims
  };
}

export function serialize(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const output = serialize(await buildClaims());
  const outputPath = path.join(rootDir, "poster/research/claims/claims.generated.json");
  if (process.argv.includes("--write")) {
    await mkdir(path.dirname(outputPath), {recursive: true});
    await writeFile(outputPath, output);
    console.log(`Wrote ${path.relative(rootDir, outputPath)}`);
  } else {
    process.stdout.write(output);
  }
}
