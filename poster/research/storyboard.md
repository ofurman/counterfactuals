# CEL Scientific Benchmark Poster Storyboard

## Scientific argument

Protocol differences can confound counterfactual-method comparisons. CEL holds the listed benchmark controls constant, reducing protocol variation as a confounder; its evidence shows trade-offs rather than one universal winner. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->

## Explicit reading order

**Header → Left → Center → Right/bottom → Footer.** The header poses the claim and establishes
identity. The left column names the confound and benchmark scope. The dominant center explains what
CEL standardizes and how evidence flows through the benchmark. The right and lower band show four
source-backed result visuals, applicability, practical selection guidance, limitations, and the
reproducibility handoff. This is a visitor path, not a dump of manuscript sections or a full result
table.

## Header

- Paper title and a short hook: shared benchmark controls expose method trade-offs. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->
**Identity inventory:** Camera-ready authors, affiliation, XKDD venue marker.
**QR inventory:** One labelled `Code & project` QR linked to the repository; no paper QR.

## Left — why and how broad

1. **Why protocol matters:** CEL standardizes splits, preprocessing, predictors, constraints, and metrics to reduce experimental setup as a comparison confounder. <!-- claims: scope.protocol | sources: manuscript/main_lncs.tex#Benchmark -->
2. **Scope strip:** show the benchmark's dataset, method-paradigm, and cross-validation scope from the claim ledger. <!-- claims: scope.datasets, scope.methods, scope.folds | sources: manuscript/main_lncs.tex#Datasets, manuscript/main_lncs.tex#Methods, manuscript/supplementary.tex#app:full_results -->
3. **What CEL standardizes:** one compact checklist leads visually into the center protocol. <!-- claims: scope.protocol | sources: manuscript/main_lncs.tex#Benchmark -->

## Center — controlled protocol

1. A shared-controls flow: splits and preprocessing → predictors and constraints → explanation methods → shared metrics. <!-- claims: scope.protocol | sources: manuscript/main_lncs.tex#Benchmark -->
2. A metric-family rail frames success, change size, plausibility, applicability, and runtime as separate qualities. <!-- claims: conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Conclusions -->
3. A benchmark matrix shows classification and regression across local, global, and group-wise paradigms without listing every experiment. <!-- claims: scope.datasets, scope.methods | sources: manuscript/main_lncs.tex#Datasets, manuscript/main_lncs.tex#Methods -->
4. A central takeaway: shared controls reduce protocol variation as a confounder, while method behavior still trades off by objective and applicability. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->

## Right/bottom — evidence and decisions

**Evidence-view inventory:**

1. **Local plausibility vs change:** Blobs contrasts PPCEF's displayed plausibility and distance. <!-- claims: local.blobs.ppcef.pp, local.blobs.ppcef.l2 | sources: manuscript/tables/results_numerical.tex#tab:num_metrics_mlp -->
2. **Group-wise effectiveness vs change:** Adult contrasts T-CREx and GLANCE validity and distance. <!-- claims: group.adult.tcrex.validity, group.adult.tcrex.distance, group.adult.glance.validity, group.adult.glance.distance | sources: manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
3. **Regression target error vs change:** Concrete contrasts CEARM and Wachter target error, distance, and plausibility. <!-- claims: regression.concrete.cearm.mae, regression.concrete.wachter.mae, regression.concrete.cearm.l2, regression.concrete.wachter.l2, regression.concrete.wachter.pp | sources: manuscript/tables/results_regression.tex#tab:regression_all_dnn -->
4. **Applicability is evidence:** a compact note shows a missing AReS result without interpreting the em dash as inapplicability. <!-- claims: global.blobs.ares.missing | sources: manuscript/tables/results_global.tex#tab:global_metrics_mlp -->

Selection guidance maps priorities to evidence: choose against the metric and task that matter. <!-- claims: conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Conclusions -->
Inspect coverage or missingness before validity, because validity can be conditional on the experiment's result semantics. <!-- claims: local.adult.cadex.validity, global.adult.globe.validity, global.blobs.ares.missing | sources: manuscript/tables/results_categorical.tex#tab:cat_metrics_mlp, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
Compare plausibility or density only within a named dataset and model. <!-- claims: local.blobs.ppcef.pp | sources: manuscript/tables/results_numerical.tex#tab:num_metrics_mlp -->
Displayed zeroes can be rounded rather than exact. <!-- claims: group.adult.tcrex.distance | sources: manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
Do not use the unresolved sparsity direction. <!-- claims: caveat.sparsity-direction | sources: manuscript/supplementary.tex#app:metrics -->

**Footer inventory:** `uv add ce-library`, repository/documentation links, and a short claim-provenance note.

## Thirty-second visitor narrative

Read the hook, scope strip, center protocol, and the large takeaway. Leave knowing that CEL reduces protocol variation as a confounder and supports trade-off decisions rather than a leaderboard. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->

## Two-minute visitor narrative

Follow the left-to-right flow, inspect the metric rail, then compare the four evidence views. Read the applicability and limitations notes before scanning the selection guidance and repository handoff. Leave able to explain both what CEL controls and why a method choice depends on the task. <!-- claims: scope.protocol, conclusion.tradeoffs, global.blobs.ares.missing, caveat.sparsity-direction | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions, manuscript/tables/results_global.tex#tab:global_metrics_mlp, manuscript/supplementary.tex#app:metrics -->
