# CEL scientific benchmark poster deliverables

Source revision: `b12e297603b36f123ca21a0af675af54f890da16`

## Contents

- `cel-benchmark-poster-a1.pdf` — one-page A1 portrait PDF (594 × 841 mm).
- `cel-benchmark-poster-preview.png` — 72 DPI PNG rendered from the final PDF with Poppler.
- `cel-benchmark-poster.html` — self-contained offline HTML poster; the contribution QR links to the repository but is not required for rendering. No bottom reproduction strip is shown.
- `SHA256SUMS` — SHA-256 hashes for the three final artifacts above.
- `audit-layout.json` — measured native-canvas layout report.
- `cel-benchmark-poster-print-review.png` — retained Stage 6 print-review evidence.

## Print

Print the PDF at 100% / actual size on A1 portrait media. Do not use “fit to page.” The page is 594 × 841 mm with 13.5 mm safe margins. The previous A0 PDF is retained for reference only and is not the workshop print file.

## Regenerate and verify

From `poster/cel-benchmark-poster`:

```sh
pnpm install
pnpm run verify:all
cd deliverables
shasum -a 256 -c SHA256SUMS
```

The full check validates manuscript-derived claims and graphics, the three-feature CE example, regenerated plot-only Matplotlib SVGs, local/global/group-wise actions and declined-to-approved transitions, non-printing provenance, reference-logo hashes, column order, the visual brief, TypeScript, tests, both Vite and single-file Parcel builds, offline rendering, native layout, PDF size/content, QR decoding from PDF pixels at 72 and 150 DPI, and package checksums. The QR audit uses an isolated uv environment with zxing-cpp 3.1.1 and Pillow 10.4.0; it does not change the project's Python dependencies.

Verified links: https://github.com/ofurman/counterfactuals and https://ofurman.github.io/counterfactuals/

## Known limitations

- The loan application example uses invented profiles and an illustrative affordability rule, not benchmark data or lending advice. Its feature values, unchanged fields, and model assumptions are recorded in `poster/research/ce-example.json`; these provenance notes are not printed on the poster.
- PWr, genwro.AI, and Tooploox logos are unchanged copies from the user-supplied PUMAL reference poster; their provenance and hashes are recorded in `poster/research/brand-assets.json`.
- XKDD and ECML-PKDD logos use the supplied project assets unchanged; their original external sources are not recorded. Local file hashes are retained in the same brand manifest.
- The poster uses focused views of the manuscript result graphics; consult the paper and supplement for the full figure matrices and exact tables.
- The example plots are transparent, font-outlined SVGs generated with CEL plotting helpers. Original circles and counterfactual diamonds distinguish status without relying only on color; arrows have stronger contrast, and the decision-boundary legend matches the opaque teal plot stroke. Their data, generator, and style hashes are validated before packaging.
- Result SVGs retain lossless manuscript plot crops with enlarged outlined Arial labels. The architecture is an unmodified vector conversion of the original manuscript PDF, retaining all original glyphs, line breaks, boxes, and connectors. Source hashes, crop pixels, complete schema SVG content, and uniform scaling are checked. The source manuscript files are unchanged.
- Typography uses an 80pt Georgia title, matching 32pt bold Results and Contributions headings, 28pt Georgia subheadings, approximately 18pt Arial body text, and result labels of at least 17pt at A1. The original diagram retains its smaller Poppins/Canva Sans typography (approximately 13.6pt on the poster). Regression density ticks use k for thousands. The examples and framework occupy the upper two columns; expanded scope tiles align with the bottom of the examples. A two-by-two results grid has padded, row-aligned frames; Global and Local use three-plus-two metric layouts, with a numbered full-name method key for Global. The unboxed closing contribution strip has three equal-width messages with small teal ordinals, 28pt bold Georgia headings, and 18pt supporting lines. Contribution wording and supporting lines come from the manuscript-backed claim ledger.
- The caption-free repository QR is 96px (43.2mm) square, with high error correction, a four-module quiet zone, and a centered 22px official GitHub clear-space SVG in an excavated area. The unchanged asset comes from https://brand.github.com/GitHub_Logos.zip; its provenance and hash are recorded under src/assets/qr. Its destination remains the CEL repository, and an accessible link label is retained.
- Sparsity rankings are intentionally omitted until the manuscript's direction convention is reconciled; the source figure is retained without an added comparative claim.
