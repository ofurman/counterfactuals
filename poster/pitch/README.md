# CEL poster pitch

One static, editable 16:9 slide for a two-minute poster pitch. The A1 poster is unchanged.

## Deliverables

- `deliverables/cel-poster-pitch.pptx`: one slide with editable text and speaker notes.
- `deliverables/cel-poster-pitch.pdf`: presentation-ready PDF exported from that PowerPoint.
- `deliverables/cel-poster-pitch.png`: PDF-rendered preview.
- `deliverables/two-minute-script.md`: approximately 240 spoken words with pacing and pointing cues; the generated file reports the exact count.

The timing targets assume approximately 120 words per minute. Rehearse aloud and allow short pauses. No animation or slide change is required.

## Story and source choices

The slide shows four compact benchmark-scope tiles on the left and three result rows on the right: local, global, and group-wise. Each row compares validity, proximity, and log-density plausibility before inviting a poster conversation. The script introduces the comparison problem and controlled protocol. The slide uses the exact manuscript title on one line, all authors underneath without an affiliation line, the poster's Georgia/Arial typography and navy/teal palette, rounded blue/cream scope tiles, and original institutional logos.

The four tiles summarize 18 datasets (13 classification, 5 regression), 14 methods (10 local, 2 global, 2 group-wise), two backbones per task, and nine reported metrics. Counts and inventories come from the manuscript-backed scope ledger in `poster/research/claims/claims.generated.json`. The nine-metric count applies to classification tables, as stated in the script and notes; it is not the total library registry size. The scope, protocol controls, and no-single-winner conclusion describe the benchmark, not a fully populated Cartesian product of every method, task, and dataset.

The nine boxplots are the Adult Census validity, L2+Hamming, and log-density panels from `manuscript/figures/metrics_boxplot_local.png`, `metrics_boxplot_global.png`, and `metrics_boxplot_group_wise.png`, through the existing poster typography derivatives. Complete metric groups are translated into aligned columns; their original boxplot interiors, axis ticks, method names, directions, and aspect ratios are preserved. All three comparisons describe LR and MLP together, not MLP alone. No values were reconstructed from summary statistics. `assets/provenance.json` records the nine original PNG regions and hashes. The dataset label, plausibility explanation, and global method key use native slide text: 1 AReS, 2 GLOBE-CE, 3 GLANCE configured with one group (GlobalGLANCE in the manuscript). Higher log-density indicates greater distributional plausibility under the fitted density estimator, not causal feasibility.

The loan-application opening in the script is a conceptual example, not a claim about the Adult Census task. Smaller changes in CADEX and T-CREx must be read together with their lower validity. These selected panels illustrate within-paradigm trade-offs, not an aggregate or cross-paradigm ranking, and do not establish real-world causal actionability. Each plot retains its original axis range. Other reported metrics remain part of the benchmark, but are omitted from the pitch plots for readability.

## Rebuild

Use Node.js with `pptxgenjs`, `playwright`, `sharp`, and `jszip` available through `NODE_PATH`, Chrome, LibreOffice (`soffice`), and Poppler. Set `PPTX_HTML_CONVERTER` to the PPTX skill's `scripts/html2pptx.js` if its default local path differs. After HTML conversion, the build restores long-dashed borders on exactly four native rounded tile shapes in the slide XML; the validator verifies those shapes.

```sh
NODE_PATH=/Users/ofurman/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node poster/pitch/build.cjs
uv run --quiet --no-project --with zxing-cpp==3.1.1 --with pillow==10.4.0 python poster/pitch/validate.py
```

Edit `slide.html` for layout and `speech.json` for the spoken script. The build reads the existing poster's offline QR, retaining its GitHub logo and destination. Figure/QR PNGs are embedded; the presentation has no required external assets. The build checks manuscript wording, exact title, loaded assets, HTML bounds, and SVG label bounds. Validation checks slide count, native editable text, speaker notes, source pixels, 16:9 PDF dimensions, required PDF text, and QR decoding at two projection resolutions. Review the final PDF-derived PNG after every layout change.
