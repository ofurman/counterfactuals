# Stage 4: Scaffold the A0 Web Artifact

**Goal**: Initialize a fresh web-artifacts-builder project and implement a buildable, print-aware A0 poster shell driven by the frozen content model.
**Dependencies**: Stage 3

---

## Steps

1. Initialize the project from the named skill, from the repository root.
   - Run `bash /Users/ofurman/.agents/skills/web-artifacts-builder/scripts/init-artifact.sh poster/cel-benchmark-poster`.
   - Preserve the generated React/TypeScript/Vite/Tailwind foundation and package manager lockfile;
     do not copy or edit the minified reference `bundle.html`.

2. Replace the Vite demo with semantic poster components.
   - Where: `App` in `poster/cel-benchmark-poster/src/App.tsx` and purpose-built components under
     `src/components/poster/`.
   - Add `PosterStage`, `PosterCanvas`, `PosterHeader`, `ScopeStrip`, `SectionBlock`, `PosterFooter`,
     and a screen-only print toolbar. Use `main`, `header`, `section`, `figure`, and `footer`.
   - Load `poster-content.json`, `claims.generated.json`, identity, names, and visual spec through a
     typed adapter; no content literals should drift into layout components.

3. Implement fixed-canvas screen scaling and exact print CSS.
   - Where: poster tokens and `@media print` in `src/index.css`.
   - Use `1800 × 1273` native canvas, A0 landscape `@page` with zero margin, deterministic top-left
     scaling, `print-color-adjust: exact`, warm paper background, safe area, and no print toolbar.
   - Screen preview may scale the whole canvas but must not reflow or exceed native scale.

4. Add the macro layout and visual tokens from `visual-spec.json`.
   - Implement the asymmetric grid and section hierarchy with thin rules and restrained accents;
     avoid generic shadcn card styling and unnecessary interactivity.

5. Add project checks.
   - Configure `typecheck`, `test`, and `build` scripts. Add tests for required semantic regions,
     typed content resolution, A0 CSS tokens, print-toolbar exclusion, and placeholder rejection.

---

## Verification

- [ ] GATE `pnpm --dir poster/cel-benchmark-poster run typecheck` — passes on the current generated claim/content inputs.
- [ ] GATE `pnpm --dir poster/cel-benchmark-poster test` — semantic regions, content references, A0 tokens, and print exclusions pass; removed/changed inputs turn the relevant test red.
- [ ] GATE `pnpm --dir poster/cel-benchmark-poster run build` — produces a build from the fresh project without importing the reference bundle.
- [ ] REPORT record the initial JS/CSS/assets build sizes in `journal.md`.

---

## Commit

`feat(poster): scaffold print-first CEL artifact`
