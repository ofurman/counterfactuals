# CEL Scientific Benchmark Poster Storyboard

## Scientific argument

Counterfactual-explanation results are difficult to compare when protocols differ. CEL is framed first as a controlled CE benchmark: it fixes the evaluation context, spans three explanation paradigms, and exposes trade-offs rather than naming one universal winner. <!-- claims: scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Explicit reading order

**Header → Left → Center → Right → Footer.** The header uses the exact manuscript title and establishes
identity. The left column explains counterfactual explanations with an illustrative before/after
example at the top and global results below. The unchanged center contains the manuscript
architecture above local results. The right column shows contributions above group-wise results.
Regression results are excluded from this poster. This is a visitor
path, not a dump of manuscript sections or a full result table.

## Header

- Paper identity and a direct benchmark hook: CEL enables controlled comparison of counterfactual explanations across multiple quality dimensions. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->
**Identity inventory:** Exact manuscript title, camera-ready authors, affiliation, XKDD venue marker. Show the title only once; use a short benchmark subtitle.
**Logo inventory:** PWr, genwro.AI, and Tooploox assets copied unchanged from the user-provided PUMAL reference poster; preserve their aspect ratios and keep authorship unchanged.
**QR inventory:** One labelled `Code & project` QR inside the Extend contribution at top right; linked to the repository, with no header or paper QR.

## Left — concept and global results

1. **Counterfactual example:** explain a prediction-changing input modification through a labelled toy loan model. Invented inputs and the approval rule live separately from experimental claims; they are not benchmark data or lending advice. <!-- claims: concept.counterfactual | sources: manuscript/main_lncs.tex#Introduction -->
2. **Scope strip:** show the benchmark's dataset, method-paradigm, backbone, and cross-validation scope from the claim ledger. <!-- claims: scope.datasets, scope.methods, scope.backbones, scope.folds | sources: manuscript/main_lncs.tex#Datasets, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Benchmark, manuscript/supplementary.tex#app:full_results -->
3. **Global results below the example:** show the manuscript's global-method figure with its validity and missing-output takeaway. <!-- claims: result.global.overview, global.blobs.ares.missing | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->

## Center — controlled protocol

1. Use the manuscript architecture figure at readable scale: Data Module and Model Module feed the Explanation Engine, then the Metrics Orchestrator and counterfactual reports. <!-- claims: scope.protocol, scope.methods, contribution.library | sources: manuscript/main_lncs.tex#Introduction -->
2. The architecture must make the benchmark surface explicit: local, global, and group-wise CEs share data, models, constraints, and metrics. <!-- claims: scope.methods, contribution.protocol | sources: manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Introduction -->
3. Use a focused crop of the manuscript local-results figure rather than a redrawn chart. Its role is to show multi-dimensional behavior, not an aggregate ranking. <!-- claims: result.local.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
4. Keep the source figure label and derivative-crop note visible so the graphic remains traceable to the manuscript. <!-- claims: result.local.overview | sources: manuscript/main_lncs.tex#Results -->

## Right — contributions and group-wise results

**Evidence-view inventory:**

1. **Local methods:** a focused view of the manuscript local-results figure shows validity, proximity, density, sparsity, and runtime moving differently. <!-- claims: result.local.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
2. **Global methods:** the manuscript figure and result text surface stronger GLOBE-CE/GlobalGLANCE validity and AReS reliability limitations. <!-- claims: result.global.overview, global.blobs.ares.missing | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
3. **Group-wise methods:** the manuscript figure shows the effectiveness-versus-change trade-off between GLANCE and T-CREx. <!-- claims: result.group.overview, group.adult.tcrex.validity, group.adult.glance.validity | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_group.tex#tab:group_metrics_mlp -->

The top-right contribution stack presents the controlled protocol, broad benchmark, and extensible library; group-wise results occupy the bottom-right panel. <!-- claims: contribution.protocol, contribution.benchmark, contribution.library, result.group.overview | sources: manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Results -->
Inspect coverage or missingness before validity, because validity can be conditional on the experiment's result semantics. <!-- claims: local.adult.cadex.validity, global.adult.globe.validity, global.blobs.ares.missing | sources: manuscript/tables/results_categorical.tex#tab:cat_metrics_mlp, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
Compare plausibility or density only within a named dataset and model. <!-- claims: local.blobs.ppcef.pp | sources: manuscript/tables/results_numerical.tex#tab:num_metrics_mlp -->
Displayed zeroes can be rounded rather than exact. <!-- claims: group.adult.tcrex.distance | sources: manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
Do not use the unresolved sparsity direction. <!-- claims: caveat.sparsity-direction | sources: manuscript/supplementary.tex#app:metrics -->

**Footer inventory:** `uv add ce-library` and repository/documentation links; keep provenance in the source notes.

## Thirty-second visitor narrative

Read the counterfactual example, benchmark scope, manuscript architecture, and one result panel. Leave knowing what a CE changes, what CEL standardizes, and why the evidence describes trade-offs rather than a leaderboard. <!-- claims: concept.counterfactual, scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Two-minute visitor narrative

Start with the illustrative CE example, follow the unchanged center architecture and local results, then read the contributions at top right. Compare global results at bottom left with group-wise results at bottom right. Read the captions and failure-mode qualifiers before the repository handoff. Leave able to explain the concept, CEL's controls, and task-dependent method trade-offs. <!-- claims: concept.counterfactual, scope.protocol, contribution.protocol, contribution.benchmark, contribution.library, result.local.overview, result.global.overview, result.group.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
