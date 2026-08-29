# Planning Findings

This is research context for the executor. It is descriptive, not a status record.

## Manuscript story and evidence

- Core thesis: inconsistent splits, preprocessing, predictive models, constraints, and metric
  definitions confound comparison of counterfactual methods; CEL fixes the protocol.
- Scope: 18 datasets (13 classification, 5 regression), 14 methods (10 local, 2 global,
  2 group-wise), two predictive backbones per task type, and five-fold cross-validation.
- Dataset range: 178–93,888 samples and 2–64 input features.
- Scientific conclusion: no method uniformly dominates; validity, proximity, sparsity,
  plausibility, applicability, and runtime trade off.
- Strong source-backed exemplars include Adult Census for global/group-wise/local comparisons,
  Blobs for local probabilistic plausibility, and Concrete for regression. Exact values must be
  extracted from the named `manuscript/tables/*.tex` files into the claim ledger.

## Manuscript hazards

- Sparsity direction conflicts between supplementary definitions, tables, figures, and prose.
- Validity may be conditional on producing a candidate; zero coverage plus perfect validity is
  possible and must not be presented without qualification.
- Raw log-density is not comparable across datasets or dimensionalities.
- `0.00±0.00` values can be rounding, not literal zero.
- `--`, non-finite values, and method/dataset inapplicability need distinct semantics.
- The regression log-density aggregate and local runtime aggregates in prose do not reconcile
  cleanly with supplied tables. Regenerate or omit them.
- Normalize WACH/Wachter, TCREx/T-CREx/TCREX, DiCE/DICE, GLANCE/GlobalGLANCE, and GMC names.
- The source figures are too dense for direct poster use; redraw from TeX tables with provenance.

## Local reference poster

Source: `/Users/ofurman/pwr/unified_cfs/pumal-scientific-poster/bundle.html`.

- Fixed `1800 × 1273` CSS-pixel canvas and exact A0 landscape print target (`1189 × 841 mm`).
- Asymmetric `1fr 2fr 1fr` body grid with a dominant center, restrained navy/teal/orange
  palette, warm paper ground, serif title, sans body, and section top rules.
- Problem → method → evidence → reproducibility reading flow; semantic HTML and offline assets.
- Preserve print geometry, hierarchy, contrast, semantic structure, and labelled QR links.
- Adapt the center to benchmark protocol/taxonomy and evidence; avoid the dense results table,
  algorithm-objective framing, unlabelled QR codes, and direct edits to the minified bundle.

## Ranked NeurIPS sources

1. **OpenXAI: Towards a Transparent Evaluation of Model Explanations** (NeurIPS 2022) — closest
   XAI library/benchmark analogue.
   - Page: <https://neurips.cc/virtual/2022/poster/55636>
   - Poster: <https://neurips.cc/media/PosterPDFs/NeurIPS%202022/70c639df5e30bdee440e4cdf599fec2b.png>
2. **M4: A Unified XAI Benchmark for Faithfulness Evaluation** (NeurIPS 2023) — best metric
   taxonomy and matrix-to-heatmap analogue.
   - Page: <https://neurips.cc/virtual/2023/poster/73690>
   - Poster: <https://neurips.cc/media/PosterPDFs/NeurIPS%202023/73690.png>
3. **Why do tree-based models still outperform deep learning on typical tabular data?**
   (NeurIPS 2022) — strongest evidence-first tabular benchmark story.
   - Page: <https://neurips.cc/virtual/2022/poster/55627>
   - Poster: <https://neurips.cc/media/PosterPDFs/NeurIPS%202022/55627.png>
4. **ADBench: Anomaly Detection Benchmark** (NeurIPS 2022) — broad benchmark design and
   findings structure.
   - Page: <https://neurips.cc/virtual/2022/poster/55709>
   - Poster: <https://neurips.cc/media/PosterPDFs/NeurIPS%202022/9766527f2b5d3e95d4a733fcfb77bd7e.png>
5. **What I Cannot Predict, I Do Not Understand** (NeurIPS 2022) — strongest question-led
   communication hierarchy.
   - Page: <https://neurips.cc/virtual/2022/poster/55282>
   - Poster: <https://neurips.cc/media/PosterPDFs/NeurIPS%202022/55282.png>

The official controls returned PNGs despite the `/PosterPDFs/` path. Store the original PNGs as
local research inputs, record source pages and SHA-256 hashes, and do not assume PDF availability.
