# CEL scientific benchmark poster deliverables

Source revision: `6b1a51a19845d73d78cf894b099eb1a08656b099`

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

The full check validates manuscript-derived claims and graphics, the three-feature CE example, local/global/group-wise actions and declined-to-approved transitions, non-printing provenance, reference-logo hashes, column order, the visual brief, TypeScript, tests, both Vite and single-file Parcel builds, offline rendering, native layout, PDF size/content, and package checksums.

Verified links: https://github.com/ofurman/counterfactuals and https://ofurman.github.io/counterfactuals/

## Known limitations

- The loan application example uses invented profiles and an illustrative affordability rule, not benchmark data or lending advice. Its feature values, unchanged fields, and model assumptions are recorded in `poster/research/ce-example.json`; these provenance notes are not printed on the poster.
- PWr, genwro.AI, and Tooploox logos are unchanged copies from the user-supplied PUMAL reference poster; their provenance and hashes are recorded in `poster/research/brand-assets.json`.
- XKDD and ECML-PKDD logos use the supplied project assets unchanged; their original external sources are not recorded. Local file hashes are retained in the same brand manifest.
- The poster uses focused views of the manuscript result graphics; consult the paper and supplement for the full figure matrices and exact tables.
- Raster result panels preserve the manuscript's published plotting style and are intended for A0 print or zoomed digital inspection.
- Sparsity rankings are intentionally omitted until the manuscript's direction convention is reconciled; the source figure is retained without an added comparative claim.
