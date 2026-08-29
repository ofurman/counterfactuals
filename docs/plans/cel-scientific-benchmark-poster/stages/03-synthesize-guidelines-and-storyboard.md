# Stage 3: Synthesize Guidelines and Storyboard

**Goal**: Turn the manuscript, local A0 example, and downloaded NeurIPS posters into a source-traceable visual brief and final poster storyboard.
**Dependencies**: Stages 1 and 2
**Reference**: [planning findings](../resources/planning-findings.md)

---

## Steps

1. Write `poster/research/design-guidelines.md` as a decision document, not generic advice.
   - Compare the local poster with each NeurIPS source on canvas, grid, title claim, visual order,
     typography, color, chart density, table use, reproducibility links, citations, and print QA.
   - For every preserve/adapt/avoid rule, cite at least one observed source ID and state how it
     applies to CEL. Flag rules based only on the local example.

2. Define the CEL visual system in `poster/research/visual-spec.json`.
   - Fix A0 landscape and a native `1800 × 1273` canvas, safe area, `1fr 2fr 1fr` macro-grid,
     warm-paper/navy base, restrained teal/orange accents, serif title, portable sans body, semantic
     chart colors, print-color adjustment, and minimum intended print sizes.
   - Avoid Inter, purple gradients, excessive centering, generic shadows, and uniform rounded cards
     in accordance with the web-artifacts-builder design guidance.

3. Write `poster/research/storyboard.md` around one scientific argument.
   - Header: paper identity, one-sentence hook, authors/affiliation, venue, labelled project QR.
   - Left: why comparison is confounded, scope strip, and what CEL standardizes.
   - Center: simplified CEL protocol/architecture and benchmark design.
   - Right/bottom: three or four source-backed trade-off findings, applicability, practical
     selection guidance, limitations, install command, and reproducibility links.
   - Include the 30-second and two-minute visitor narratives and explicit reading order.

4. Create `poster/research/poster-content.json`.
   - Store final copy blocks, section order, referenced claim IDs, source citations, link IDs, and
     asset roles. Keep numbers as claim references, never duplicated literals.
   - Apply a prose budget and convert manuscript paragraphs into short declarative statements.

5. Add `poster/research/scripts/validate-brief.mjs`.
   - Validate every guideline source ID, poster claim ID, link ID, and required storyboard section.
   - Fail on untracked numeric literals, placeholders, duplicate section ownership, or a guideline
     with no observed source.

---

## Verification

- [ ] GATE `node poster/research/scripts/validate-brief.mjs` — every guideline and storyboard claim resolves to the current manifests/claim ledger; missing source IDs or copied numeric literals turn the gate red.
- [ ] GATE `visual-spec.json` contains exact A0 page dimensions, native canvas dimensions, safe-area values, print-color handling, font stacks, and minimum intended print sizes.
- [ ] GATE `storyboard.md` names a complete left-to-right reading order and both visitor narratives, with no manuscript-section dump or full result table.
- [ ] REPORT record total copy word count, number of result visuals, and expected bundle asset budget in `journal.md`.

---

## Commit

`docs(poster): define CEL poster guidelines and storyboard`
