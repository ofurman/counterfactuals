# Plan: CEL Scientific Benchmark Poster

**Date**: 2026-08-29
**Branch**: `xkdd-manuscript`
**Predecessors**: None
**Goal**: Deliver a fact-checked, print-ready A0 landscape CEL benchmark poster as a self-contained web artifact, one-page PDF, and preview image.

Executed per [PROTOCOL.md](PROTOCOL.md). Status of record: [state.json](state.json).
Runtime record: [journal.md](journal.md) · [decisions.md](decisions.md) · [backlog.md](backlog.md)

---

## Context

The manuscript presents CEL as a controlled protocol and reusable library for comparing 14
counterfactual methods on 18 datasets across local, global, group-wise, classification, and
regression settings. The poster must foreground the confounding problem, what CEL standardizes,
benchmark breadth, and the empirical conclusion that method quality is a trade-off rather than a
single ranking.

The manuscript's existing architecture figure is a useful concept, but its result images are
paper-scale walls of small multiples. Selected results must be redrawn from the TeX tables. The
manuscript also contains known ambiguities—especially sparsity direction, conditional validity,
rounded zeroes, missing-result semantics, and unsupported aggregate prose claims—so no poster
number may bypass a source-backed claim ledger.

The local reference poster establishes a strong implementation baseline: fixed `1800 × 1273`
canvas, exact A0 landscape print CSS (`1189 × 841 mm`), asymmetric `1fr 2fr 1fr` hierarchy,
semantic HTML, restrained serif/sans typography, and offline bundling. Its algorithm-centric
content and dense table should not be copied.

NeurIPS research favors a claim-led title, benchmark pipeline/taxonomy before results, a few large
evidence graphics, limited prose, restrained color, and one clearly labelled QR destination. See
[planning findings](resources/planning-findings.md) and the
[web research report](../../research/web/2026-08-29-neurips-benchmark-poster-design.md).

---

## Strategy

**Phase A — Evidence** (Stages 1–3): freeze identity and claims, download/manifest poster
references, and synthesize a source-traceable design brief plus storyboard.

**Phase B — Build** (Stages 4–5): initialize a fresh web-artifacts-builder project, implement the
A0 layout, and replace manuscript-scale figures with reproducible SVG evidence graphics.

**Phase C — Verify and deliver** (Stages 6–7): render and inspect the print artifact, audit every
claim and layout invariant, then create the final self-contained bundle and handoff package.

---

## Success Criteria

Every row declares a **Kind**. GATE blocks the owning stage; REPORT is recorded and never blocks.
`NOT MEASURED` never means `PASS`; on a GATE it blocks the stage.

| Metric | Baseline | Target | Kind | If missed | If unmeasurable |
|--------|----------|--------|------|-----------|-----------------|
| Poster claim provenance | manuscript has known contradictions and unverified aggregates | every user-visible benchmark number and comparative claim has a claim ID derived from named manuscript/table inputs; unknown or contradictory claims are omitted or explicitly qualified | GATE | block stage | REPORT `NOT MEASURED` and block |
| Web artifact build | no poster project | `pnpm run build` and TypeScript checks pass in `poster/cel-benchmark-poster` | GATE | block stage | n/a |
| Print geometry | local example is A0 landscape | rendered PDF has exactly one `1189 × 841 mm` page within ±1 pt per dimension, measured from the run's PDF with `pdfinfo` | GATE | block stage | REPORT `NOT MEASURED` and block |
| Layout containment | no CEL render | DOM audit on the rendered poster reports no clipped section, horizontal overflow, or element outside the poster safe area | GATE | block stage | REPORT `NOT MEASURED` and block |
| Self-contained delivery | no bundle | `bundle.html` loads with network disabled and contains no required remote runtime asset | GATE | block stage | REPORT `NOT MEASURED` and block |
| Poster-distance readability and visual hierarchy | existing figures are too dense | independent review of the current full-page PNG records title/section/body/chart-label legibility, reading order, contrast, and visible defects | REPORT | publish findings and continue | publish `NOT MEASURED` |
| Every GATE value is derived from a measurement of this run's own inputs | n/a | no status is a literal, default, band midpoint, copied output, or row generated to satisfy a count | GATE | block stage | REPORT `NOT MEASURED` and block |

---

## Files That May Be Changed

### Research and plan records
- `docs/research/web/2026-08-29-neurips-benchmark-poster-design.md` — external poster research.
- `docs/plans/cel-scientific-benchmark-poster/` — execution state, evidence, decisions, and stage briefs.
- `docs/plans/LESSONS.md` — durable manuscript/poster caveat.

### Poster research inputs
- `poster/research/` — claim ledger, source manifest, downloaded reference posters, guidelines, and storyboard.

### Poster artifact
- `poster/cel-benchmark-poster/` — React/TypeScript source, styles, assets, tests, scripts, bundle, and deliverables.
- `.gitignore` — only if needed to exclude reproducible third-party reference binaries or build outputs.

The manuscript is an input and is not edited by this plan.

---

## Stages

Routing table only. Status, notes, and commits live in `state.json` and nowhere else.

| # | Stage |
|---|-------|
| 1 | [Freeze poster claims and identity](stages/01-freeze-claims-and-identity.md) |
| 2 | [Collect NeurIPS poster references](stages/02-collect-neurips-references.md) |
| 3 | [Synthesize guidelines and storyboard](stages/03-synthesize-guidelines-and-storyboard.md) |
| 4 | [Scaffold the A0 web artifact](stages/04-scaffold-a0-web-artifact.md) |
| 5 | [Implement evidence graphics and poster content](stages/05-implement-evidence-poster.md) |
| 6 | [Audit facts, print layout, and readability](stages/06-audit-print-and-content.md) |
| 7 | [Bundle and package deliverables](stages/07-bundle-and-package.md) |
