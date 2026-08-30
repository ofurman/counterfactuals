# CEL Scientific Benchmark Poster Storyboard

## Scientific argument

Counterfactual-explanation results are difficult to compare when protocols differ. CEL is framed first as a controlled CE benchmark: it fixes the evaluation context, spans three explanation paradigms, and exposes trade-offs rather than naming one universal winner. <!-- claims: scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Explicit reading order

**Header → Left → Center → Right → Footer.** The header uses the exact manuscript title and establishes
identity. The left column explains counterfactual explanations with an illustrative before/after
example. The center contains the manuscript architecture above a compact benchmark-scope grid.
The right column combines global, local, and group-wise findings under one Results heading,
followed by the contribution stack and project QR.
Regression results are excluded from this poster. This is a visitor
path, not a dump of manuscript sections or a full result table.

## Header

- Paper identity and a direct benchmark hook: CEL enables controlled comparison of counterfactual explanations across multiple quality dimensions. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->
**Identity inventory:** Exact manuscript title centered between the logos, camera-ready authors, affiliation. Show the title only once, without a subtitle, venue marker, or top color line.
**Logo inventory:** XKDD above ECML-PKDD on the left; PWr above genwro.AI above Tooploox on the right. Institutional assets come from the user-provided PUMAL reference poster; conference assets were supplied in the project. Preserve all logo files and aspect ratios and keep authorship unchanged.
**QR inventory:** One labelled `Code & project` QR inside the Extend contribution below the right-column results; linked to the repository, with no header or paper QR.

## Left — counterfactual concept

1. **Counterfactual example:** show the original and counterfactual applicant profiles side by side on the light poster background. Highlight income and debt changes while age, employment, credit history, and loan amount stay fixed. Keep the example caption and decision flip; retain invented-data provenance and model assumptions in project files rather than visible notes. <!-- claims: concept.counterfactual | sources: manuscript/main_lncs.tex#Introduction -->

## Center — framework and benchmark scope

1. Use the manuscript architecture figure at readable scale: crop its outer whitespace, enlarge it to the center-column width, and omit the repeated caption and panel background. Preserve all schema content: Data Module and Model Module feed the Explanation Engine, then the Metrics Orchestrator and counterfactual reports. <!-- claims: scope.protocol, scope.methods, contribution.library | sources: manuscript/main_lncs.tex#Introduction -->
2. The architecture must make the benchmark surface explicit: local, global, and group-wise CEs share data, models, constraints, and metrics. <!-- claims: scope.methods, contribution.protocol | sources: manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Introduction -->
3. **Scope grid below the framework:** show the benchmark's dataset, method-paradigm, backbone, and cross-validation scope from the claim ledger in two rows. Add the classification-table metric count, including runtime; proximity variants count as a single metric. <!-- claims: scope.datasets, scope.methods, scope.backbones, scope.folds, scope.metrics | sources: manuscript/main_lncs.tex#Datasets, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Benchmark, manuscript/supplementary.tex#app:full_results, manuscript/tables/results_categorical.tex#tab:cat_metrics_mlp -->

## Right — results and contributions

**Evidence-view inventory:**

1. **Local methods:** a focused view of the manuscript local-results figure shows validity, proximity, density, sparsity, and runtime moving differently. <!-- claims: result.local.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
2. **Global methods:** the manuscript figure and result text surface stronger GLOBE-CE/GlobalGLANCE validity and AReS reliability limitations. <!-- claims: result.global.overview, global.blobs.ares.missing | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
3. **Group-wise methods:** the manuscript figure shows the effectiveness-versus-change trade-off between GLANCE and T-CREx. <!-- claims: result.group.overview, group.adult.tcrex.validity, group.adult.glance.validity | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_group.tex#tab:group_metrics_mlp -->

Combine the global, local, and group-wise panels into one Results section above the contribution stack. Keep their concise takeaway headings and manuscript graphics, without repeated explanatory paragraphs or dataset/method captions. Contributions below present the controlled protocol, broad benchmark, and extensible library. <!-- claims: contribution.protocol, contribution.benchmark, contribution.library, result.local.overview, result.global.overview, result.group.overview | sources: manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Results -->
Reflow the local figure's metric panels into two rows to enlarge the original axes and labels without stretching them. Preserve every metric, axis, and method label. Keep dataset identity in accessible metadata and source mappings in project files rather than printed captions. <!-- claims: result.local.overview | sources: manuscript/main_lncs.tex#Results -->
Inspect coverage or missingness before validity, because validity can be conditional on the experiment's result semantics. <!-- claims: local.adult.cadex.validity, global.adult.globe.validity, global.blobs.ares.missing | sources: manuscript/tables/results_categorical.tex#tab:cat_metrics_mlp, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
Compare plausibility or density only within a named dataset and model. <!-- claims: local.blobs.ppcef.pp | sources: manuscript/tables/results_numerical.tex#tab:num_metrics_mlp -->
Displayed zeroes can be rounded rather than exact. <!-- claims: group.adult.tcrex.distance | sources: manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
Do not use the unresolved sparsity direction. <!-- claims: caveat.sparsity-direction | sources: manuscript/supplementary.tex#app:metrics -->

**Footer inventory:** `uv add ce-library` and repository/documentation links; keep provenance in project files.

## Thirty-second visitor narrative

Read the counterfactual example, benchmark scope, manuscript architecture, and one result panel. Leave knowing what a CE changes, what CEL standardizes, and why the evidence describes trade-offs rather than a leaderboard. <!-- claims: concept.counterfactual, scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Two-minute visitor narrative

Start with the illustrative CE example, follow the center architecture and benchmark scope, then compare all three result paradigms together on the right. Finish with the contributions and repository handoff below those findings. Read the plot axes and failure-mode headings before choosing a method. Leave able to explain the concept, CEL's controls, and task-dependent method trade-offs. <!-- claims: concept.counterfactual, scope.protocol, contribution.protocol, contribution.benchmark, contribution.library, result.local.overview, result.global.overview, result.group.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
