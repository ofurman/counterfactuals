# Stage 7: Bundle and Package Deliverables

**Goal**: Create the final self-contained HTML bundle and a checksummed handoff containing the verified PDF and preview.
**Dependencies**: Stage 6

---

## Steps

1. Produce the single-file web artifact with the named skill script.
   - From `poster/cel-benchmark-poster`, run
     `bash /Users/ofurman/.agents/skills/web-artifacts-builder/scripts/bundle-artifact.sh`.
   - Keep the generated `bundle.html` at the project root and rerun the render/audit harness against
     the bundle, not only the Vite source build.

2. Prove offline behavior.
   - Load `bundle.html` through the local test harness with network access disabled.
   - Fail on a required remote script, stylesheet, font, image, fetch/XHR, or unresolved local path.
     Ordinary clicked repository/docs links may be external but must not be required for rendering.

3. Finalize `poster/cel-benchmark-poster/deliverables/`.
   - Include the final A0 PDF, PDF-derived preview PNG, a copy of the self-contained HTML bundle,
     and `SHA256SUMS` generated from those exact bytes.
   - Add `README.md` with dimensions, contents, print instructions, source revision, generation
     commands, verified links, known limitations, and how to rerun the claim/layout audits.

4. Run the complete final check from a clean build state.
   - Re-run claim, brief, type, test, build, bundle-offline, layout, PDF, and checksum checks.
   - Inspect `git status` so generated caches/dependency trees are ignored and intended source,
     research manifests, bundle, and deliverables are accounted for.

5. Present the outputs.
   - Open the bundle, PDF, and preview in Codex when execution finishes and report their absolute
     paths plus any remaining REPORT findings from the journal/backlog.

---

## Verification

- [ ] GATE `pnpm --dir poster/cel-benchmark-poster run verify:all` — current claims, typed source, tests, build, layout, PDF, and package checks all pass on the run's own inputs.
- [ ] GATE `pnpm --dir poster/cel-benchmark-poster run audit:offline` — final `bundle.html` renders with network disabled and no required remote asset; a blocked or unresolved runtime request turns it red.
- [ ] GATE `cd poster/cel-benchmark-poster/deliverables && shasum -a 256 -c SHA256SUMS` — final HTML, PDF, and PNG match the manifest generated from this run's bytes.
- [ ] GATE final PDF still has exactly one A0 landscape page and the PDF-derived preview contains all required poster sections.
- [ ] REPORT record final bundle/PDF/PNG sizes and all surviving visual-review notes in `journal.md`.

---

## Commit

`feat(poster): package CEL scientific poster deliverables`
