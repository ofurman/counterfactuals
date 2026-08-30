# CEL Scientific Benchmark Poster Storyboard

## Scientific argument

Counterfactual-explanation results are difficult to compare when protocols differ. CEL is framed first as a controlled CE benchmark: it fixes the evaluation context, spans three explanation paradigms, and exposes trade-offs rather than naming one universal winner. <!-- claims: scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Explicit reading order

**Header → Upper left → Upper right → Bottom.** The A1 portrait header uses the exact manuscript title and establishes
identity. The upper-left column explains counterfactual explanations with an illustrative before/after
example. The upper-right column contains the manuscript architecture above a compact benchmark-scope grid,
followed by the contribution stack and project QR.
The full-width bottom section combines global, local, group-wise, and regression findings in a two-by-two grid under one Results heading.
Each result category has its own transparent dashed rounded frame. This is a visitor
path, not a dump of manuscript sections or a full result table.

## Header

- Paper identity and a direct benchmark hook: CEL enables controlled comparison of counterfactual explanations across multiple quality dimensions. <!-- claims: scope.protocol, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Conclusions -->
**Identity inventory:** Exact manuscript title centered between the logos, camera-ready authors, affiliation. Show the title only once, without a subtitle, venue marker, or top color line.
**Typography:** Use a ninety-six-point Georgia title, twenty-eight-point Georgia subheadings, and Arial body and figure labels. Set body copy near eighteen points and manuscript figure labels to at least seventeen points at A1. Preserve manuscript vector artwork and lossless plot interiors in poster-only typography derivatives; do not infer or reconstruct benchmark statistics.
**Logo inventory:** XKDD above ECML-PKDD on the left; PWr above genwro.AI above Tooploox on the right. Institutional assets come from the user-provided PUMAL reference poster; conference assets were supplied in the project. Preserve all logo files and aspect ratios and keep authorship unchanged.
**QR inventory:** One labelled `Code & project` QR inside the Extend contribution below the upper-right scope tiles; linked to the repository, with no header or paper QR.

## Upper left — counterfactual concept

1. **Counterfactual example:** show income, debt payments, and unchanged employment in a local before/after profile. Extend it to global and group-wise examples using the same original applicants, shared axes, and decision boundary. Every original is declined and every arrow ends at an approved counterfactual. Show a shared income increase globally, and distinct debt-payment or income changes per group. Retain invented-data provenance and model assumptions in project files rather than visible notes; keep the light background and concise captions. <!-- claims: concept.counterfactual | sources: manuscript/main_lncs.tex#Introduction -->

## Upper right — framework and benchmark scope

**Presentation:** Omit both printed section headings and the divider between the architecture and inventory tiles. Keep section names in accessible metadata only.
**Tile styling:** Echo the schema with rounded, pale-blue module containers, dashed dark-blue outlines, and cream heading boxes with solid rounded borders. Keep inventory text and the two-by-two arrangement unchanged.
**Tile details:** Use title-case headings and longer dashes with slightly thicker outlines.
**Section separation:** Use whitespace without horizontal rules between sections, including the header and footer. Preserve table rules, plot axes, and tile outlines.

Group the dataset inventory under Classification and Regression using each dataset's task in the manuscript table, while retaining the combined dataset total. <!-- claims: scope.datasets | sources: manuscript/main_lncs.tex#Datasets -->

1. Use the manuscript architecture figure at readable scale: crop its outer whitespace, enlarge it to the upper-right column width, and omit the repeated caption and panel background. Preserve all schema content: Data Module and Model Module feed the Explanation Engine, then the Metrics Orchestrator and counterfactual reports. <!-- claims: scope.protocol, scope.methods, contribution.library | sources: manuscript/main_lncs.tex#Introduction -->
2. The architecture must make the benchmark surface explicit: local, global, and group-wise CEs share data, models, constraints, and metrics. <!-- claims: scope.methods, contribution.protocol | sources: manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Introduction -->
3. **Scope tiles below the framework:** name the full dataset and method inventories, task-specific backbones, and classification-table metrics in a compact two-by-two grid. Group methods into local, global, and group-wise lists. Remove standalone folds and paradigm-count tiles. Include runtime among metrics and count proximity variants once. All displayed names resolve to the manuscript-derived claim inventory. <!-- claims: scope.datasets, scope.methods, scope.backbones, scope.metrics | sources: manuscript/main_lncs.tex#Datasets, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Models, manuscript/supplementary.tex#app:full_results, manuscript/tables/results_categorical.tex#tab:cat_metrics_mlp -->
4. **Contributions beneath the tiles:** present the controlled protocol, broad benchmark, and extensible library in a compact row, with the repository QR inside Extend. <!-- claims: contribution.protocol, contribution.benchmark, contribution.library | sources: manuscript/main_lncs.tex#Introduction -->

## Bottom — results

**Evidence-view inventory:**
**Result frames:** Put each category inside a dashed, rounded, transparent rectangle matching the scope outlines. Keep spacing rather than standalone section divider lines.

1. **Local methods:** a focused view of the manuscript local-results figure shows validity, proximity, density, sparsity, and runtime moving differently. <!-- claims: result.local.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
2. **Global methods:** the manuscript figure and result text surface stronger GLOBE-CE/GlobalGLANCE validity and AReS reliability limitations. <!-- claims: result.global.overview, global.blobs.ares.missing | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
3. **Group-wise methods:** the manuscript figure shows the effectiveness-versus-change trade-off between GLANCE and T-CREx. <!-- claims: result.group.overview, group.adult.tcrex.validity, group.adult.glance.validity | sources: manuscript/main_lncs.tex#Results, manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
4. **Regression methods:** show the complete Concrete row of the manuscript figure, retaining target MAE, change distance, sparsity, log-density, and runtime. Keep the comparison scoped to the pictured dataset without asserting universal dominance. <!-- claims: result.regression.overview | sources: manuscript/main_lncs.tex#Results -->

Combine the global, local, group-wise, and regression panels into one right-column Results section. Keep their concise takeaway headings and manuscript graphics, without repeated explanatory paragraphs or dataset/method captions. <!-- claims: result.local.overview, result.global.overview, result.group.overview, result.regression.overview | sources: manuscript/main_lncs.tex#Results -->
Reflow the local figure's metric panels into two rows to enlarge the original axes and labels without stretching them. Preserve every metric, axis, and method label. Keep dataset identity in accessible metadata and source mappings in project files rather than printed captions. <!-- claims: result.local.overview | sources: manuscript/main_lncs.tex#Results -->
Inspect coverage or missingness before validity, because validity can be conditional on the experiment's result semantics. <!-- claims: local.adult.cadex.validity, global.adult.globe.validity, global.blobs.ares.missing | sources: manuscript/tables/results_categorical.tex#tab:cat_metrics_mlp, manuscript/tables/results_global.tex#tab:global_metrics_mlp -->
Compare plausibility or density only within a named dataset and model. <!-- claims: local.blobs.ppcef.pp | sources: manuscript/tables/results_numerical.tex#tab:num_metrics_mlp -->
Displayed zeroes can be rounded rather than exact. <!-- claims: group.adult.tcrex.distance | sources: manuscript/tables/results_group.tex#tab:group_metrics_mlp -->
Do not use the unresolved sparsity direction. <!-- claims: caveat.sparsity-direction | sources: manuscript/supplementary.tex#app:metrics -->

**Footer inventory:** No bottom reproduction strip or repository/documentation links. Retain the contribution QR and non-printing provenance. Keep a compact twelve-pixel gap between the header and main body with no header bottom padding.

## Thirty-second visitor narrative

Read the counterfactual example, benchmark scope, manuscript architecture, and one result panel. Leave knowing what a CE changes, what CEL standardizes, and why the evidence describes trade-offs rather than a leaderboard. <!-- claims: concept.counterfactual, scope.protocol, scope.methods, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Methods, manuscript/main_lncs.tex#Conclusions -->

## Two-minute visitor narrative

Start with the illustrative CE example, follow the center architecture and benchmark scope, and read the contributions beneath the tiles. Then compare all three explanation paradigms and the regression task on the right. Return to the center QR for the repository handoff. Read plot axes and failure-mode headings before choosing a method. Leave able to explain the concept, CEL's controls, and task-dependent method trade-offs. <!-- claims: concept.counterfactual, scope.protocol, contribution.protocol, contribution.benchmark, contribution.library, result.local.overview, result.global.overview, result.group.overview, result.regression.overview, conclusion.tradeoffs | sources: manuscript/main_lncs.tex#Benchmark, manuscript/main_lncs.tex#Introduction, manuscript/main_lncs.tex#Results, manuscript/main_lncs.tex#Conclusions -->
