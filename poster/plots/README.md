# Matplotlib loan examples

Generate local, global, and group-wise examples from the poster's existing
[`ce-example.json`](../research/ce-example.json), plus a side-by-side comparison.
These are illustrative profiles, not benchmark results. The React poster imports
the three individual SVGs. Its full verification command regenerates transparent
figures before rebuilding the HTML and PDF.

From the repository root, using the existing Python environment:

```sh
uv run --no-sync python poster/plots/plot_ce_examples.py --transparent
```

The output is eight files in `poster/plots/generated/`: three individual plots
and one comparison, each as a 300-DPI PNG and a vector SVG with outlined fonts.
Axes, points, arrows, and decision-region labels have no figure titles or action
captions. A three-item legend below the group-wise plot identifies both original
group colors, counterfactual points, and the decision boundary. The other plots
have no legends. The extra legend row preserves the plot-area size and scale.
Arial labels are outlined and render consistently in
the browser and PDF without requiring the Matplotlib font to be installed.

For one example or a transparent asset:

```sh
uv run --no-sync python poster/plots/plot_ce_examples.py --paradigm group-wise --transparent
uv run --no-sync python poster/plots/plot_ce_examples.py --formats svg --output-dir /tmp/cel-examples
```

Options also include `--data PATH` and `--dpi NUMBER`. Paths default relative to
the script, so the generator does not depend on the shell's working directory.
It uses the headless Matplotlib Agg backend; no GUI or model training is needed.

From `poster/cel-benchmark-poster`, run `pnpm run verify:all` to regenerate these
assets, test them, and rebuild the integrated poster. Use `pnpm run plots:generate`
to regenerate only the transparent assets. SVG metadata records input, style,
and generator hashes; the poster audit rejects stale assets.

## Style and data

The script calls the actual observation, counterfactual, arrow, and classifier
boundary functions in [`cel/plotting/plot_utils.py`](../../cel/plotting/plot_utils.py):

- Originals use `tab10`, with opacity 0.8 and marker size 50. Group-wise originals
  use blue for A-B and green for C-D; all are declined.
- Counterfactuals use orange diamond markers, distinct from original circles;
  all are approved. The group-wise legend uses the same shapes.
- Arrows retain the helper's exact endpoints, with a wider shaft, larger
  arrowheads, and slate color at opacity 0.9 for clearer transitions.
- The helper's decision contours use one opaque teal stroke, also used in the
  legend. The light grid follows `counterfactual_visualization.py`.

Coordinates are normalized to the unit square expected by the CEL helpers, but
ticks and axes show monthly euro amounts. All panels share the same limits.
The classifier evaluates the full rule from the JSON, with employment and the
undisplayed features held fixed. No density contours are fabricated.

The utility module is loaded directly from its source file. This avoids CEL's
top-level imports of unrelated optional model dependencies, including GPyOpt.
The generator still requires Matplotlib, NumPy, PyTorch, scikit-learn, and Pillow,
which are already available in the repository environment. A standalone setup
can use `uv run --no-project --with matplotlib --with numpy --with torch --with
scikit-learn --with pillow python poster/plots/plot_ce_examples.py`.

## Checks

```sh
uv run --no-sync pytest -q poster/plots
uv run --no-sync ruff check poster/plots
uv run --no-sync ruff format --check poster/plots
```

Tests check the shared applicant population, prescribed actions, fixed features,
declined-to-approved predictions, title/caption absence, source metadata, plotted
coordinates, arrow endpoints and direction, helper provenance and style, equal
plot-area sizes, legend symbols and placement, text containment, CLI
output, and transparent exports. The current CEL boundary helper emits a PyTorch
warning about future `meshgrid` indexing defaults; this does not affect rendering.

## Manuscript typography derivatives

`manuscript_typography.py` creates four result SVGs with outlined Arial labels
and an original-manuscript architecture SVG. Run it with
`uv run --no-sync python poster/plots/manuscript_typography.py`.
It requires the existing Matplotlib, NumPy, and Pillow environment, Arial regular
and bold fonts, and Poppler's `pdftocairo` command. It does not modify manuscript files.

The architecture is an unmodified vector conversion of `manuscript/figures/teaser.pdf`,
using its existing CropBox. Original Poppins/Canva Sans glyphs, line breaks,
boxes, and connectors are preserved; no poster labels are substituted. The four
result derivatives embed lossless PNG crops of the original plot interiors,
including every box, whisker, median, and gridline. No statistics are inferred or
replotted. Each crop retains its source aspect ratio. Local and global metrics
use a three-plus-two arrangement. Global source axes are approximately 39% wider
in the SVG than in the former five-across version. Numbered ticks refer to one
shared key with all three original method names, avoiding repeated long labels.
The other result categories retain method names at each axis. Dataset labels
and all method names remain visible.

Tick strings are transcribed from the manuscript figures. Their anchors snap to
the original high-resolution gridlines. Regression log-density uses `k` for
thousands to keep its long negative tick labels readable. SVG metadata records
source and generator hashes, crop pixel bounds, embedded crop hashes, method
ordering, the global numbered method key, layout, dataset identity, tick anchors,
and minimum font sizes.

Tests compare the embedded crop pixels to the original PNGs and the complete schema
SVG, including every glyph and its viewport, to an independent source PDF conversion.
The poster audit checks source hashes, uniform image scaling, result-label containment,
and at least 17pt result labels at A1. The original diagram deliberately retains
its smaller manuscript typography (approximately 13.6pt on the poster).
The title is 80pt, the main Results heading is 32pt bold, subheadings are 28pt,
and body/result text uses Arial. The A1 portrait layout uses upper example/framework
columns and a lower two-by-two results grid with aligned frames and larger internal
padding. Scope inventories have more line and group spacing. Bottom contribution
statements are bold and top-aligned, with a separate QR column. Logo artwork is unchanged.

The poster's Parcel configuration bypasses SVG optimization only for
`manuscript-architecture*.svg`: the original nested glyph definitions make the
optimizer stall. Other assets retain their normal optimization pipeline.
