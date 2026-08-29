# Backlog (Deferred Issues)

Each entry must be self-contained enough for a future run to pick it up cold.

| # | Title | Origin | Severity | Why deferred | Next step | Status |
|---|-------|--------|----------|--------------|-----------|--------|
| B-1 | Python suite cannot import `GPyOpt` | Stage 1 | Low | `uv run pytest` fails during collection in six modules because the existing project environment does not provide `GPyOpt`; the poster claim pipeline is Node-only and its gates pass | Decide whether to add/restore the project dependency or make CEARM an optional import, then rerun `uv run pytest` | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`.

When an item flips to RESOLVED, **revisit its origin stage in the same commit** -- a stage may
not stay BLOCKED on a resolved item. Summarize the fix in `journal.md`. Heavy items may warrant
their own follow-up plan; link it here.
