# Stage 1: Freeze Poster Claims and Identity

**Goal**: Create a machine-checkable claim ledger and final identity/link configuration from repository-tracked sources before any layout work begins.
**Dependencies**: None
**Reference**: [planning findings](../resources/planning-findings.md)

---

## Steps

1. Inventory the scientific story and known contradictions without editing the manuscript.
   - Where: the abstract, contribution list, `Benchmark`, `Experimental Setup`, and `Conclusions`
     sections in `manuscript/main_lncs.tex`; `Full Results` in
     `manuscript/supplementary.tex`; and `manuscript/tables/*.tex`.
   - Record the controlled-protocol thesis, scope, evaluation setup, headline conclusion, and
     caveats for sparsity, conditional validity, missing results, rounded values, and log-density.

2. Build a source-derived claim pipeline under `poster/research/claims/`.
   - Add an extraction script that reads the named TeX inputs and generates `claims.generated.json`.
   - Each claim must have a stable ID, poster wording, value/unit or qualitative verdict, source
     file, stable section/table/row anchor, extraction rule, direction/qualifier, and status.
   - Include scope claims and only selected local/global/group-wise/regression exemplars that can
     be regenerated. Do not include the unreconciled aggregate log-density/runtime prose claims.
   - Mark reported zeroes as rounded where appropriate and encode missing/inapplicable separately.

3. Freeze presentation metadata in `poster/research/identity.json`.
   - Source the title and camera-ready authors/affiliation from the commented camera-ready block in
     `manuscript/main_lncs.tex`, venue from the XKDD 2026 manuscript marker, repository URL from
     `[project.urls]` in `pyproject.toml`, and documentation URL from `README.md`.
   - Use one labelled repository/project QR. Omit a paper QR until a real publication URL exists.
   - Set the output contract to A0 landscape, `1189 × 841 mm`, plus self-contained HTML and PNG.

4. Normalize naming in `poster/research/method-names.json`.
   - Choose one displayed name for WACH/Wachter, TCREx variants, DiCE/DICE, GLANCE variants, and
     Give Me Some Credit/GMC; preserve source aliases for extraction.

5. Add `poster/research/scripts/validate-claims.mjs`.
   - Re-run extraction, fail on missing anchors, unresolved contradictions, placeholder metadata,
     non-finite values presented as finite, or a poster claim without a live manuscript source.
   - Make the validator compare regenerated output byte-for-byte with the committed generated file.

---

## Verification

- [ ] GATE `node poster/research/scripts/validate-claims.mjs` — every claim is regenerated from the current manuscript inputs; a changed/missing table row, anchor, qualifier, or identity source turns the gate red.
- [ ] GATE `rg -n "TBD|TODO|FIXME|example\.com" poster/research/claims poster/research/identity.json poster/research/method-names.json` returns no matches.
- [ ] GATE every selected result in `claims.generated.json` names a real `manuscript/tables/*.tex` input and table/row anchor; values copied only from prose turn the gate red.
- [ ] REPORT record the final count of publishable, qualified, excluded, and contradictory claims in `journal.md`.

---

## Commit

`docs(poster): freeze CEL claims and identity`
