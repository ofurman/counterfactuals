# Stage 5: Seed aggregator script

**Goal**: Write `scripts/aggregate_seed_results.py`, which walks the per-seed result roots and
produces mean ± std across seeds per (method, dataset) — the script `slurm/README.md` records
as "not yet written".
**Dependencies**: Stage 1 DONE (defines the `cf_model_train_time` column). **Independent of
Stage 4 completion** — build and test this while the sweep runs, using synthetic fixtures and
the Stage 3 smoke CSV.

---

## Why a new script

`scripts/calculate_metrics.py::load_and_aggregate_metrics` (line 45) aggregates
`models_root/<dataset>/<method>/fold_<i>/cf_metrics_<model>.csv` over **folds within one
root**. The sweep produces one `fold_0` per **seed root**
(`results/<tag>/seed_<N>/…`), so that function cannot see across seeds. Reuse its
*conventions*, not its traversal.

Two behaviours from it are worth keeping deliberately:
- Rows with `validity == 0` are excluded from aggregation, so a degenerate seed cannot drag a
  cell to 0.00.
- `number_of_instances`-weighted means for quality metrics.

And one that must **not** be copied: weighting is wrong for *timing* columns. A time is a
duration per run, not a per-instance average, so times aggregate as an unweighted mean over
seeds. Getting this backwards silently biases the deliverable.

---

## Steps

1. Create the script.
   - File: `scripts/aggregate_seed_results.py` (new)
   - Module docstring, Google-style docstrings on public functions, full type hints, `logging`
     (no `print`), ruff line-length 100 — per `CLAUDE.md`.

2. Implement discovery.
   - Signature sketch:
     ```python
     def discover_runs(results_root: Path, tag: str) -> pd.DataFrame: ...
     ```
   - Glob `results_root/<tag>/seed_*/<dataset>_split/<Method>/fold_0/cf_metrics_*.csv`.
     From each path derive `tag`, `seed` (from `seed_<N>`), `dataset` (strip the `_split`
     suffix), `method` (the class-name directory), and `disc_model` (from the filename
     suffix). Concatenate into one long DataFrame with those as columns plus every metric
     column.
   - Map class directory names to display names:
     `DiceExplainerWrapper → DiCE`, `CCHVAE → CCHVAE`, `DiCoFlex → DiCoFlex`.
   - Read the sibling `../../<stem>_<method>_run.json` when present and carry `git_commit`
     through, so the report can flag cells produced by pre-instrumentation code.

3. Implement the derived timing columns.
   - `total_train_time = disc_train_time + gen_train_time + cf_model_train_time`
     (valid because Stage 1's contract makes the three non-overlapping).
   - `method_train_time`, via an explicit, documented mapping — **not** a heuristic:
     | method | `method_train_time` |
     |---|---|
     | DiCE | `0.0` (training-free) |
     | CCHVAE | `cf_model_train_time` (the VAE) |
     | DiCoFlex | `gen_train_time` (the conditional generator) |
     Keep this as a module-level constant dict so the report can cite it and a reviewer can
     audit it in one place.
   - `inference_time = cf_search_time`, plus
     `inference_time_per_factual = cf_search_time / number_of_instances`. The per-factual
     figure is **required**, not optional: DiCoFlex at `tc=1` and the others at `tc=0` run on
     disjoint, differently-sized query sets (adult 3674 vs 6326), so absolute search times are
     not comparable across target classes while per-factual times are.
   - Emit `n_seeds` per cell so a reader can tell a 3-seed std from a 1-seed one.

4. Implement aggregation.
   - Signature sketch:
     ```python
     def aggregate_over_seeds(runs: pd.DataFrame) -> pd.DataFrame: ...
     ```
   - Group by `(tag, method, dataset)`. Drop `validity == 0` rows first, mirroring
     `calculate_metrics.py` lines 76-91, and log how many were dropped.
   - Timing columns → unweighted `mean` and `std` (`ddof=1`) over seeds.
   - Quality metrics → `number_of_instances`-weighted mean, `std` unweighted, matching
     `calculate_metrics.py`.
   - Reuse the existing `format_mean_std` presentation (`f"{mean:.2f} ± {std:.2f}"`) by
     importing it from `scripts.calculate_metrics` rather than redefining it, so a future
     format change stays in one place. If the import proves awkward (script, not package),
     duplicating it with a comment pointing at the original is acceptable — record the choice
     in Decisions.
   - With `n_seeds == 1` the std is undefined; emit `nan` and let the report render it as
     `n/a (1 seed)`. Never print `± 0.00` for a single seed — that reads as "no variance
     measured", which is a different and false claim.

5. Implement the CLI.
   - Arguments: `--results-root` (default `results`), `--tags` (repeatable, default
     `seeds seeds-tc0`), `--disc-model` (default `SimpleMLPClassifier`),
     `--out-csv`, `--out-markdown`.
   - Emit a tidy long-form CSV (one row per tag/method/dataset/metric with mean, std,
     n_seeds) for downstream reuse, plus the markdown the report embeds.
   - Missing cells: warn via `logging` and represent them explicitly as `n/a` in the
     markdown. Never impute, never silently omit a row.

6. Unit-test it.
   - File: `tests/aggregate_seed_results_test.py` (new — matches the existing `*_test.py`
     convention in `tests/`)
   - Build a `tmp_path` fixture tree with 3 fake seed roots x 2 datasets x 2 methods, then
     assert:
     - discovery finds every CSV and parses seed/dataset/method correctly
     - the mean/std of a hand-computed timing column is exact
     - a `validity == 0` row is excluded and logged
     - `total_train_time` equals the sum of its three parts
     - `method_train_time` follows the mapping for each of the three methods
     - a single-seed cell yields `nan` std, not `0.0`
     - a missing cell surfaces as `n/a`, not a dropped row

7. Retire the stale note.
   - File: `slurm/README.md`
   - Details: The "Output layout" section says aggregating the three seed roots "still needs a
     small aggregator script — **not yet written**". Replace that with the actual invocation.

---

## Verification

- [ ] `uv run ruff check scripts/aggregate_seed_results.py tests/aggregate_seed_results_test.py` — clean
- [ ] `uv run ruff format --check scripts/ tests/` — clean
- [ ] `uv run pytest tests/aggregate_seed_results_test.py -q` — all pass
- [ ] `uv run pytest tests/ -q` — no regressions
- [ ] Runs against real data: point `--results-root` at the Stage 3 smoke tag and confirm it
      produces a 1-row table with `nan` std and `n_seeds=1` rather than crashing
- [ ] `uv run python scripts/aggregate_seed_results.py --help` prints all documented flags
- [ ] `grep -c "not yet written" slurm/README.md` returns 0

---

## Commit

`feat(scripts): aggregate cf metrics across seed roots`
