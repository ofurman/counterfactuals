# Journal

Append-only. Newest entries at the bottom. Never rewrite an earlier entry.

One entry per invocation, in this shape:

```
## YYYY-MM-DD HH:MM -- Stage N: [Name] -- DONE
**Did**: [1-3 lines]
**Verification**: GATE lines passed. REPORT values: [metric]=[value]
**Provenance**: [per measured GATE: the input the value was read from, and the defect that would
turn it red] - [or `NOT MEASURED` for any that could not be produced from this run's own inputs]
**Problems**: [symptom -> root cause -> resolution -> inline/subagent] or "none"
**Commit**: `abc1234`
```

---

## 2026-08-29 13:12 -- Stage 1: Freeze Poster Claims and Identity -- DONE
**Did**: Added source-derived identity and method-name registries plus a TeX table extractor that generated 22 stable claims (17 publishable, 4 qualified, 1 contradiction resolved by omission). Added an independent validator for exact identity, headings, scope consistency, comparative verdicts, result cells, and byte-for-byte regeneration.
**Verification**: GATE `node poster/research/scripts/validate-claims.mjs` passed; placeholder scan returned no matches; 16 selected results resolved to named manuscript table cells. Independent mutation audit passed after proving affiliation, venue, naming, heading, peer comparison, sparsity, count-drift, prose-value, and literal-value defects turn red. REPORT claim counts: publishable=17, qualified=4, excluded=0, contradictory=1.
**Provenance**: Scope values are parsed from the current manuscript declarations and cross-checked with conclusion/table totals; result values are independently parsed from the named TeX cells and compared exactly with the generated ledger; identity and aliases resolve to the camera-ready block, repository metadata, README, and manuscript sources. Any changed declaration, row, metric heading/cell, identity field, comparison peer, or unresolved contradiction turns validation red.
**Problems**: The initial conclusion anchor mismatch was repaired by a focused worker. Two independent audits exposed false-pass paths (unchecked identity/headings/verdicts, then decorative row anchors and hard-coded count text); repairs added live-source and independent cell checks. `uv run pytest` cannot collect because `GPyOpt` is absent in the existing project environment; deferred as B-1 because it is unrelated to this Node-only poster stage.
**Commit**: `this commit` (`docs(poster): freeze CEL claims and identity`)
