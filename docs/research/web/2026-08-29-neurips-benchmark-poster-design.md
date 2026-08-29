---
date: 2026-08-29T12:30:20+02:00
researcher: Oleksii Furman
topic: "NeurIPS benchmark-poster exemplars and design guidance for a CEL scientific benchmark poster"
tags: [neurips, benchmark, poster, xai, counterfactuals, scientific-communication]
sources: [official-conference-pages, official-poster-assets, local-web-artifact, manuscript]
status: complete
last_updated: 2026-08-29
---

# Research: NeurIPS Benchmark-Poster Design for CEL

**Date**: 2026-08-29T12:30:20+02:00
**Researcher**: Oleksii Furman

## Research Question

Research and identify strong NeurIPS posters for related or significant benchmark papers, then
derive poster guidelines for the CEL counterfactual-explanations library and benchmark.

## Summary

The closest reference is OpenXAI because it combines an XAI evaluation framework, tabular data,
metrics, leaderboards, code, and a project link in one poster. M4 is the strongest analogue for
organizing an explanation benchmark around a pipeline and metric taxonomy, while the tabular deep
learning benchmark is the clearest evidence-first poster in the set. Across nine inspected
official assets, the most effective pattern is a landscape claim-led story with a benchmark
pipeline before results and only a few large evidence graphics. CEL should use one labelled QR,
an A0 print-first hierarchy, and a trade-off thesis rather than a universal-winner leaderboard.

## Detailed Findings

### Closest topical exemplars

- **OpenXAI: Towards a Transparent Evaluation of Model Explanations** is the closest topical
  match: an open XAI evaluation framework with benchmark datasets, methods, models, many metrics,
  leaderboards, and extensibility. Its three-column poster puts TL;DR/contributions first, the
  library/API in the center, and benchmark evidence plus a labelled project QR on the right
  ([official page](https://neurips.cc/virtual/2022/poster/55636),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/70c639df5e30bdee440e4cdf599fec2b.png)).
- **M4: A Unified XAI Benchmark for Faithfulness Evaluation** is a direct explanation-benchmark
  analogue. Its pipeline, metric taxonomy, heatmaps, and grouped bars show how to compress a large
  experiment matrix into a small number of visual summaries
  ([official page](https://neurips.cc/virtual/2023/poster/73690),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/73690.png)).
- **pyKT** is an exact library-plus-benchmark structural analogue with standardized preprocessing,
  datasets, models, and evaluation. Its dense tables and three QR codes make it more useful as a
  content checklist and density warning than as a visual model
  ([official page](https://neurips.cc/virtual/2022/poster/55769),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/55769.png)).

### Benchmark storytelling exemplars

- **Why do tree-based models still outperform deep learning on typical tabular data?** uses the
  clearest evidence-first structure: two top-level questions, large charts, short benchmark
  bullets, and three numbered explanations. This is a strong model for CEL's “what is controlled?”
  and “what did we learn?” split
  ([official page](https://neurips.cc/virtual/2022/poster/55627),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/55627.png)).
- **ADBench** organizes a very broad benchmark as motivation → design → findings → future work →
  contributions. Its many small plots show the cost of over-representing comprehensive results at
  poster distance
  ([official page](https://neurips.cc/virtual/2022/poster/55709),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/9766527f2b5d3e95d4a733fcfb77bd7e.png)).
- **What I Cannot Predict, I Do Not Understand** has the strongest communication hierarchy: a dark
  hero band with one provocative question, numbered research questions, and a summary box. It is
  less topically similar but valuable for title-level narrative design
  ([official page](https://neurips.cc/virtual/2022/poster/55282),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/55282.png)).

### Secondary references

- **Evaluating the Robustness of Interpretability Methods** offers a rigorous five-step sequence
  from failure example to formalization, empirical results, and further information
  ([official page](https://neurips.cc/virtual/2023/poster/72800),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/72800.png)).
- **VeriX** provides counterfactual/explanation examples but is too dense to copy as a benchmark
  hierarchy
  ([official page](https://neurips.cc/virtual/2023/poster/72338),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/72338.png)).
- A **reproducibility study of Label-Free Explainability** uses claim-by-claim green/red assessment,
  a useful fact-audit device even though its text and plots are small
  ([official page](https://neurips.cc/virtual/2023/poster/74159),
  [official poster asset](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/74159.png)).

### Cross-poster design guidance

- Use one landscape canvas with a strong title strip and two or three reading columns.
- Let the headline state the scientific problem or answer; do not reuse the paper abstract.
- Establish the benchmark's visual grammar—scope, protocol, architecture, and metric taxonomy—
  before showing results.
- Use a left-to-right narrative rather than manuscript section order.
- Replace comprehensive result walls with a few large plots, heatmaps, or ranked summaries.
- Keep prose short and move operational detail into diagrams, compact bullets, and one project QR.
- Use a white or warm-paper ground, dark text, and one or two navigation accents; reserve additional
  colors for meaningful chart encodings.
- Label every QR destination and keep it subordinate to the title.
- Perform print-distance QA for body text and chart labels; several official examples are complete
  but visibly too dense at overview scale.

## Sources Consulted

- [OpenXAI official poster page](https://neurips.cc/virtual/2022/poster/55636) — closest XAI library/benchmark analogue.
- [M4 official poster page](https://neurips.cc/virtual/2023/poster/73690) — explanation-benchmark taxonomy and result synthesis.
- [Tabular deep learning benchmark poster page](https://neurips.cc/virtual/2022/poster/55627) — evidence-first storytelling.
- [ADBench official poster page](https://neurips.cc/virtual/2022/poster/55709) — comprehensive benchmark structure.
- [pyKT official poster page](https://neurips.cc/virtual/2022/poster/55769) — library/benchmark content structure and density warning.
- [Human-centered explainability evaluation poster page](https://neurips.cc/virtual/2022/poster/55282) — question-led hierarchy.
- [Interpretability robustness poster page](https://neurips.cc/virtual/2023/poster/72800) — metric formalization story.
- [VeriX official poster page](https://neurips.cc/virtual/2023/poster/72338) — counterfactual visual examples.
- [Explainability reproducibility poster page](https://neurips.cc/virtual/2023/poster/74159) — claim-level validation structure.
- `/Users/ofurman/pwr/unified_cfs/pumal-scientific-poster/bundle.html` — A0 print implementation and local visual baseline.
- `manuscript/` — CEL scope, scientific claims, tables, figures, and identity metadata.

## Key Insights

CEL's best visual story is not “here are all benchmark results.” It is: incompatible protocols
hide real method trade-offs; CEL holds the experiment constant; the resulting evidence shows no
universal winner; the open library makes that comparison reproducible and extensible. The central
graphic should therefore explain controlled evaluation, while the results area should use a small
set of source-backed cross-paradigm contrasts. The local poster's A0 implementation is a strong
technical baseline, but its algorithm-objective center must become a benchmark-protocol center.

## Confidence Notes

- The official NeurIPS poster controls for the inspected 2022–2023 examples resolved to PNG files
  under paths named `/PosterPDFs/`; no direct PDF was confirmed for the shortlist.
- Ranking reflects topical and design relevance, not citation count or claimed paper impact.
- Fine QR captions on some posters were not legible at overview scale, so exact destinations were
  not inferred.

## Open Questions

None for planning. Poster identity, link, and output defaults are resolved in plan decision D-3 and
must be rechecked against repository-tracked metadata during Stage 1.

## Clarifications Log

No clarifications requested.
