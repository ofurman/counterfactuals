# CEL Poster Design Guidelines

This is a decision record for the CEL poster. NeurIPS source IDs resolve through
`neurips/sources.json`; observations behind them are in `neurips/notes.md`. `local-reference` is
bound to the repository file
[local-reference](../../docs/plans/cel-scientific-benchmark-poster/resources/planning-findings.md)
at `docs/plans/cel-scientific-benchmark-poster/resources/planning-findings.md`,
which records the inspected PUMAL A0 bundle and its exact `1800 × 1273` canvas,
`1189 × 841 mm` A0 target, and `1fr 2fr 1fr` body grid. A local-only fact is explicitly
labelled and is not presented as a NeurIPS convention. The user-supplied workshop
requirement supersedes that reference geometry: the final poster is A1 portrait,
594 × 841 mm. Reference observations below remain historical evidence, not a
claim that the workshop uses A0.

## Canvas and grid

- **Preserve — fixed print geometry.** Use one portrait canvas with explicit print dimensions;
  the local reference demonstrates the `1800 × 1273`/A0 implementation, while the observed
  NeurIPS examples support one-page overview composition. Sources: [local-reference (local-only),
  openxai-2022, tabular-dl-2022].
- **Adapt — asymmetric macro-grid.** Use a `1fr 2.1fr` upper body so the CEL
  framework retains the widest column, with a full-width two-by-two results grid below. Adapt the navigable concept/framework/evidence progression visible in OpenXAI and
  M4. Sources: [local-reference (local-only), openxai-2022, m4-2023].
- **Avoid — manuscript columns.** Do not reproduce paper sections or equal-weight text columns;
  ADBench shows that comprehensiveness can overwhelm overview reading. Sources: [adbench-2022].

## Title claim and visual order

- **Preserve — claim-led header.** State the problem and answer before details, following the
  question-led tabular and human-centered posters. Sources: [tabular-dl-2022, human-centered-xai-2022].
- **Adapt — problem → control → evidence.** Begin with the comparison confound, make the controlled
  CEL protocol the center, then show cross-paradigm trade-offs. Sources: [openxai-2022, m4-2023,
  tabular-dl-2022].
- **Avoid — abstract as hero copy.** The title band must not carry a paragraph; the compact summary
  in the human-centered source demonstrates a stronger hierarchy. Sources: [human-centered-xai-2022].

## Typography and color

- **Preserve — restrained type roles.** Use a serif display face only for the title and a portable
  sans stack for content. The exact stacks are a local choice; OpenXAI visibly supports a clear
  serif-title/sans-body distinction. Sources: [local-reference (local-only), openxai-2022].
- **Adapt — semantic accents.** Navy structures the page; teal marks controlled protocol and
  plausibility; orange marks trade-offs or cautions. The two-sided encoding in the tabular poster
  and semantic coral/green accents in the human-centered source justify meaning-driven color.
  Sources: [tabular-dl-2022, human-centered-xai-2022].
- **Avoid — decorative competition with evidence.** Prefer a small number of purposeful regions to
  repeated decorative containers. M4's pale section strips support navigation, while the
  human-centered poster's rounded panels work because they group a few large visuals; neither
  observation establishes a universal component style. Sources: [m4-2023,
  human-centered-xai-2022].

### Web-artifact implementation constraints

The implementation must avoid Inter, purple gradients, excessive centering, generic shadows, and
uniform rounded cards. These are web-artifacts-builder constraints recorded in `visual-spec.json`;
they are not observations attributed to the NeurIPS posters.

## Charts, tables, and evidence

- **Preserve — few large evidence graphics.** Favor direct annotations and large contrasts, as in
  the tabular benchmark. Sources: [tabular-dl-2022].
- **Adapt — taxonomy before metrics.** Use one protocol diagram and one metric-family strip before
  the result comparisons, borrowing M4's pipeline-to-taxonomy sequence. Sources: [m4-2023].
- **Avoid — full result walls.** No full benchmark table and no grids of miniature plots; the small
  tables in OpenXAI and dense center of ADBench are explicit distance-reading warnings. Sources:
  [openxai-2022, adbench-2022].
- **Avoid — aggregate leaderboard.** CEL's scientific thesis is multi-dimensional trade-off, so
  charts compare paired metrics within named configurations rather than manufacture one score.
  Sources: [m4-2023, tabular-dl-2022].

## Reproducibility, citations, and print QA

- **Preserve — one labelled project QR.** Use the repository destination and spell out its purpose,
  following OpenXAI's `Project website` treatment. Sources: [openxai-2022].
- **Adapt — compact provenance.** Put claim IDs in accessible metadata and use short visible source
  notes, keeping the scientific surface uncluttered. Sources: [openxai-2022, m4-2023].
- **Avoid — unlabeled or repeated QR codes.** Use one labelled destination. M4 and the
  human-centered poster each visibly use one QR without a nearby plain-language label, while
  OpenXAI labels its destination. Sources: [m4-2023, human-centered-xai-2022, openxai-2022].
- **Preserve — real-distance QA.** Inspect the full-page render, chart labels, and smallest type;
  dense OpenXAI, M4, and ADBench regions show why source completeness is not the same as legibility.
  Sources: [openxai-2022, m4-2023, adbench-2022].
- **Adapt — exact A1 CSS.** Use `@page { size: 594mm 841mm; margin: 0; }`, print-color adjustment,
  and safe-area containment. These dimensions come from the user-supplied workshop requirement;
  the reference sources support explicit single-page sizing, not this CSS value. Sources:
  [local-reference (local-only), tabular-dl-2022].
