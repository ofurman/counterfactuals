# Stage 5: Implement Evidence Graphics and Poster Content

**Goal**: Complete the CEL poster with a simplified protocol diagram and readable, source-derived evidence graphics that support the trade-off thesis.
**Dependencies**: Stage 4

---

## Steps

1. Implement the header, problem, and scope narrative from `poster-content.json`.
   - Where: components under `poster/cel-benchmark-poster/src/components/poster/`.
   - Present the confounding problem, `18 datasets · 14 methods · 3 paradigms · 2 backbones ·
     5 folds`, controlled variables, final identity, and a visibly labelled project QR.
   - Render values through claim/link IDs; never duplicate numbers or URLs in JSX.

2. Redraw the CEL architecture as responsive inline SVG.
   - Where: `BenchmarkPipeline` under `src/components/figures/`.
   - Preserve data → model → explanation engine → metrics → reports, but reduce nested boxes and
     foreground the controlled protocol and local/global/group-wise taxonomy.
   - Add meaningful title/description text for accessibility.

3. Implement three or four focused evidence graphics from generated claims.
   - Use source-backed Adult Census comparisons for global/group-wise/local, a Concrete regression
     comparison, and/or an applicability strip as selected by the storyboard.
   - Prefer large paired bars/dumbbells, compact trade-off plots, and explicit direction arrows.
     Never place the manuscript's full PNG walls or aggregate raw log-density across datasets.
   - Encode missing/inapplicable, conditional validity, and rounded zeroes visibly and consistently.

4. Add the practical takeaway and reproducibility footer.
   - Include a small priority-based selection matrix, limitations, `pip install ce-library`,
     repository/docs links, citations for benchmark precedents, and one labelled QR.
   - Keep claims neutral; do not highlight a universal “our method” winner.

5. Complete accessibility and asset handling.
   - Use portable/local fonts or robust system stacks, inline SVG where possible, alt text or
     descriptions for substantive graphics, and optimized local raster assets only when necessary.
   - Keep all poster-critical assets local and compatible with later HTML inlining.

6. Extend tests to audit claim use.
   - Collect every rendered `data-claim-id`; verify it exists in the regenerated claim ledger and
     that no user-visible result number bypasses the claim formatter.
   - Verify QR/link destinations against `identity.json` and source citations against content IDs.

---

## Verification

- [ ] GATE `node poster/research/scripts/validate-claims.mjs && node poster/research/scripts/validate-brief.mjs` — poster inputs still derive from the current manuscript and research manifests.
- [ ] GATE `pnpm --dir poster/cel-benchmark-poster test` — every rendered result number resolves through a live claim ID; an untracked literal, stale source row, bad link, or missing qualifier turns the gate red.
- [ ] GATE `pnpm --dir poster/cel-benchmark-poster run typecheck && pnpm --dir poster/cel-benchmark-poster run build` — completed poster compiles and builds.
- [ ] REPORT record section word counts, number of visible findings, and build asset sizes in `journal.md`.

---

## Commit

`feat(poster): add CEL benchmark story and evidence`
