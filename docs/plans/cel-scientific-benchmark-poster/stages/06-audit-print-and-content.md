# Stage 6: Audit Facts, Print Layout, and Readability

**Goal**: Produce current PDF/PNG renders and prove factual, geometric, and offline correctness before packaging.
**Dependencies**: Stage 5

---

## Steps

1. Add a reproducible local render harness under `poster/cel-benchmark-poster/scripts/`.
   - Use a pinned headless-browser dev dependency and a temporary local server to render the current
     source at native canvas size, export a one-page A0 PDF, and capture a full poster PNG.
   - Save `deliverables/cel-benchmark-poster-a0.pdf` and
     `deliverables/cel-benchmark-poster-preview.png`; do not depend on the in-app browser's blocked
     `file://` path.

2. Add `scripts/audit-layout.mjs` against the rendered DOM.
   - Assert the poster canvas/native dimensions, safe area, absence of horizontal/document overflow,
     every named section's containment, no clipped text/figure, toolbar exclusion in print, and no
     unresolved image or font request.
   - Record the smallest computed body, caption, and chart-label sizes as measurements from the
     current render. Zero/missing values are `NOT MEASURED`, never pass.

3. Audit the PDF with Poppler.
   - Use `pdfinfo` to assert one page and A0 landscape dimensions within ±1 pt.
   - Render the PDF back to PNG with `pdftoppm` so the visual review inspects the print artifact,
     not only the screen DOM.

4. Run a cold fact audit.
   - Give a verifier the rendered poster, current claim ledger, and manuscript/table inputs. For
     each visible number/comparison, it must identify the source input and qualifier; it must also
     check title, authors, affiliation, venue, URLs, method naming, citations, and limitations.

5. Run an independent visual review on the current PDF-derived PNG.
   - Record PASS/FAIL observations for title hierarchy, left-to-right reading order, body/chart
     legibility, contrast, chart semantics, whitespace balance, alignment, clipping, and QR label.
   - Fix deterministic defects and rerender. Subjective residuals are recorded as REPORT findings;
     a factual, clipped, missing, or unreadable essential element remains a GATE failure.

---

## Verification

- [ ] GATE `pnpm --dir poster/cel-benchmark-poster run audit:claims` — every visible scientific claim is derived from the current manuscript/table inputs; a missing source or qualifier turns it red.
- [ ] GATE `pnpm --dir poster/cel-benchmark-poster run render && pnpm --dir poster/cel-benchmark-poster run audit:layout` — the current rendered DOM has no clipped section, overflow, missing asset, or safe-area breach.
- [ ] GATE `pdfinfo poster/cel-benchmark-poster/deliverables/cel-benchmark-poster-a0.pdf` reports exactly one landscape page measuring `1189 × 841 mm` within ±1 pt; the input is this run's PDF and wrong size/page count turns it red.
- [ ] GATE the independent fact review maps each current visible benchmark value to its claim ID and manuscript source; unmatched or contradicted content blocks the stage.
- [ ] REPORT publish the independent print-derived PNG review and computed minimum text/chart sizes in `journal.md`.

---

## Commit

`test(poster): verify CEL print artifact and claims`
