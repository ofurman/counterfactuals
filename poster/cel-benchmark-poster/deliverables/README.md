# CEL scientific benchmark poster deliverables

Source revision: `263b6177fc0ffd754e3b967f035324d8f74e9a63`

## Contents

- `cel-benchmark-poster-a0.pdf` — one-page A0 landscape PDF (1189 × 841 mm).
- `cel-benchmark-poster-preview.png` — 72 DPI PNG rendered from the final PDF with Poppler.
- `cel-benchmark-poster.html` — self-contained offline HTML poster; repository and documentation links remain clickable but are not required for rendering.
- `SHA256SUMS` — SHA-256 hashes for the three final artifacts above.
- `audit-layout.json` — measured native-canvas layout report.
- `cel-benchmark-poster-print-review.png` — retained Stage 6 print-review evidence.

## Print

Print the PDF at 100% / actual size on A0 landscape media. Do not use “fit to page.” The page is 1189 × 841 mm and includes the poster's configured safe area.

## Regenerate and verify

From `poster/cel-benchmark-poster`:

```sh
pnpm install
pnpm run verify:all
cd deliverables
shasum -a 256 -c SHA256SUMS
```

The full check validates manuscript-derived claims, the visual brief, TypeScript, tests, both Vite and single-file Parcel builds, offline rendering, native layout, PDF size/content, and package checksums.

Verified links: https://github.com/ofurman/counterfactuals and https://ofurman.github.io/counterfactuals/

## Known limitations

- Supplemental provenance/footer source text is near the lower comfortable viewing size, but it is not essential content.
- The lower evidence row uses repeated `BOTTOM` owner labels; its spatial order remains clear.
- Sparsity comparisons are intentionally omitted until the manuscript's direction convention is reconciled.
