# CEL Scientific Benchmark Poster Storyboard

## Scientific argument

Counterfactual-explanation results are difficult to compare when protocols differ. CEL is framed first as a controlled CE benchmark: it fixes the evaluation context, spans three explanation paradigms, and exposes trade-offs rather than naming one universal winner. <!-- claims: scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Explicit reading order

**Header → Left → Center → Right/bottom → Footer.** The header poses the claim and establishes
identity. The left column names the comparison problem and the paper's three contributions. The
dominant center uses the manuscript architecture and local-results figure to explain what CEL
benchmarks. The right column uses the manuscript global, group-wise, and regression figures to show
paradigm-specific findings and failure modes before the reproducibility handoff. This is a visitor
path, not a dump of manuscript sections or a full result table.

## Header

- Paper identity and a direct benchmark hook: CEL enables controlled comparison of counterfactual explanations across multiple quality dimensions. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->
**Identity inventory:** Camera-ready authors, affiliation, XKDD venue marker.
**QR inventory:** One labelled `Code & project` QR linked to the repository; no paper QR.

## Left — why and how broad

1. **Motivation:** inconsistent splits, preprocessing, predictors, constraints, and metric definitions can make setup effects look like method effects. <!-- claims: scope.protocol | sources: manuscript/main_lncs.tex#Benchmark -->
2. **Scope strip:** show the benchmark's dataset, method-paradigm, backbone, and cross-validation scope from the claim ledger. <!-- claims: scope.datasets, scope.methods, scope.backbones, scope.folds | sources: manuscript/main_lncs.tex#Datasets, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Benchmark, manuscript/supplementary.tex#app:full_results -->
3. **Contribution stack:** controlled evaluation, broad CE benchmark coverage, and an extensible open-source workbench. <!-- claims: contribution.protocol, contribution.benchmark, contribution.library | sources: manuscript/main_lncs.tex#Introduction -->

## Center — controlled protocol

1. Use the manuscript architecture figure at readable scale: Data Module and Model Module feed the Explanation Engine, then the Metrics Orchestrator and counterfactual reports. <!-- claims: scope.protocol, scope.methods, contribution.library | sources: manuscript/main_lncs.tex#Introduction -->
2. The architecture must make the benchmark surface explicit: local, global, and group-wise CEs share data, models, constraints, and metrics. <!-- claims: scope.methods, contribution.protocol | sources: manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Introduction -->
3. Use a focused crop of the manuscript local-results figure rather than a redrawn chart. Its role is to show multi-dimensional behavior, not an aggregate ranking. <!-- claims: result.local.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
4. Keep the source figure label and derivative-crop note visible so the graphic remains traceable to the manuscript. <!-- claims: result.local.overview | sources: manuscript/main_lncs.tex#Results -->

## Right/bottom — evidence and decisions

**Evidence-view inventory:**

1. **Local methods:** a focused view of the manuscript local-results figure shows validity, proximity, density, sparsity, and runtime moving differently. <!-- claims: result.local.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
2. **Global methods:** the manuscript figure and result text surface stronger GLOBE-CE/GlobalGLANCE validity and AReS reliability limitations. <!-- claims: result.global.overview, global.blobs.ares.missing | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
3. **Group-wise methods:** the manuscript figure shows the effectiveness-versus-change trade-off between GLANCE and T-CREx. <!-- claims: result.group.overview, group.adult.tcrex.validity, group.adult.glance.validity | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
4. **Regression methods:** the manuscript figure separates target error from change size, plausibility, and runtime for CEARM and Wachter. <!-- claims: result.regression.overview, regression.concrete.cearm.mae, regression.concrete.wachter.mae, regression.concrete.cearm.l2, regression.concrete.wachter.l2 | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_regression.tex#tab:regression_all_dnn -->

The contribution stack frames CEL as a benchmark contribution first and a supporting library second. <!-- claims: contribution.protocol, contribution.benchmark, contribution.library | sources: manuscript/main_lncs.tex#Introduction -->
Inspect coverage or missingness before validity, because validity can be conditional on the experiment's result semantics. <!-- claims: local.adult.cadex.validity, global.adult.globe.validity, global.blobs.ares.missing | sources: manuscript/tables/results_categorical.tex#tab:cat_metrics_mlp, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
Compare plausibility or density only within a named dataset and model. <!-- claims: local.blobs.ppcef.pp | sources: manuscript/tables/results_numerical.tex#tab:num_metrics_mlp -->
Displayed zeroes can be rounded rather than exact. <!-- claims: group.adult.tcrex.distance | sources: manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
Do not use the unresolved sparsity direction. <!-- claims: caveat.sparsity-direction | sources: manuscript/supplementary.tex#app:metrics -->

**Footer inventory:** `uv add ce-library`, repository/documentation links, and a short claim-provenance note.

## Thirty-second visitor narrative

Read the benchmark hook, scope strip, manuscript architecture, and one result panel. Leave knowing that CEL standardizes CE evaluation across paradigms and that the outcome is a trade-off map rather than a leaderboard. <!-- claims: scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Two-minute visitor narrative

Follow the left-to-right flow from motivation and contributions into the manuscript architecture, then compare the local, global, group-wise, and regression figures. Read the figure captions and failure-mode qualifiers before the repository handoff. Leave able to explain what CEL controls, what it contributes, and why CE method choice depends on the task and metric. <!-- claims: scope.protocol, contribution.protocol, contribution.benchmark, contribution.library, result.local.overview, result.global.overview, result.group.overview, result.regression.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
