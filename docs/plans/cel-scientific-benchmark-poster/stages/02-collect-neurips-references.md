# Stage 2: Collect NeurIPS Poster References

**Goal**: Download, verify, and document the strongest NeurIPS benchmark-poster references as reproducible local research inputs.
**Dependencies**: None
**Reference**: [web research report](../../../research/web/2026-08-29-neurips-benchmark-poster-design.md)

---

## Steps

1. Create `poster/research/neurips/sources.json` from the ranked shortlist.
   - Include OpenXAI, M4, the tabular deep-learning benchmark, ADBench, and the human-centered
     explainability evaluation poster.
   - For each entry record title, year, official event page, direct official poster asset URL,
     topical/design role, and observed file format. Preserve the distinction between page and asset.

2. Add `poster/research/neurips/download.mjs` and run it.
   - Download the five real official poster PNGs into `poster/research/neurips/assets/`.
   - Generate `manifest.generated.json` from the downloaded bytes with final URL, media type,
     dimensions, byte size, and SHA-256. Do not fabricate a PDF extension for PNG content.
   - Treat the assets as local research material: add a scoped `.gitignore` for the reproducible
     third-party binaries, while committing the source list, manifest, and download script.

3. Add `poster/research/neurips/notes.md`.
   - Record a structured visual inspection of each downloaded original: hierarchy, reading order,
     chart/table density, typography, palette, QR treatment, strongest pattern, and failure mode.
   - Cite the source ID for every observation and distinguish direct observation from inference.

4. Add `poster/research/neurips/verify.mjs`.
   - Verify that each manifest row corresponds to a real downloaded file from its URL, has a valid
     PNG signature and decoded dimensions, and matches its SHA-256.
   - Reject duplicate bytes, generated placeholder images, missing source pages, or padded rows.

---

## Verification

- [ ] GATE `node poster/research/neurips/verify.mjs` — all five real downloaded posters match current local bytes and named official source records; missing, duplicate, generated, corrupt, or substituted files turn the gate red.
- [ ] GATE every `sources.json` entry has both an `https://neurips.cc/virtual/` page and an official `https://neurips.cc/media/` asset URL, and every observed claim in `notes.md` cites one of those IDs.
- [ ] REPORT record download sizes, pixel dimensions, and any redirected/failing source in `journal.md`; source availability is reported without inventing a substitute.

---

## Commit

`docs(poster): catalog NeurIPS benchmark poster references`
