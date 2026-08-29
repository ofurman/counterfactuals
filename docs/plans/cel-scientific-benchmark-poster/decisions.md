# Decisions

Append-only. **<=15 lines per entry** — detail goes in `resources/`.

---

### D-1: Use a print-first A0 landscape web artifact
**Date**: 2026-08-29 — **Stage**: planning
**Options**: A) A0 print-first artifact B) interactive dashboard C) condensed-paper poster
**Chosen**: A0 print-first React artifact with PDF and self-contained HTML outputs.
**Rationale**: It matches the local reference, NeurIPS poster conventions, and conference-distance reading while preserving a shareable web artifact.

### D-2: Make trade-offs, not a leaderboard, the empirical thesis
**Date**: 2026-08-29 — **Stage**: planning
**Options**: A) one overall winner B) selected source-backed trade-offs C) full result table
**Chosen**: Selected local/global/group-wise/regression comparisons and an applicability summary.
**Rationale**: The manuscript concludes that no method dominates; aggregate ranking would hide metric and applicability trade-offs.

### D-3: Resolve unspecified poster identity deterministically
**Date**: 2026-08-29 — **Stage**: planning
**Options**: A) retain anonymous manuscript identity B) use camera-ready metadata already present in source
**Chosen**: Use the commented camera-ready authors/affiliation, `XKDD 2026`, the repository URL in `pyproject.toml`, and A0 landscape unless a newer repository-tracked brief overrides them before Stage 1.
**Rationale**: This avoids placeholders and keeps unattended execution possible without inventing metadata.

### D-4: Omit comparative sparsity claims
**Date**: 2026-08-29 — **Stage**: 1
**Options**: A) choose one direction B) reproduce the manuscript wording C) omit the comparison
**Chosen**: Omit comparative sparsity claims; retain the contradiction as a non-poster ledger caveat.
**Rationale**: The live metric definition, table arrows, and prose do not support one unambiguous direction.
