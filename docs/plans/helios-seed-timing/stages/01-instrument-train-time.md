# Stage 1: Instrument method-own training time

**Goal**: Add a `cf_model_train_time` column to all three train/test pipelines so the CF
method's own training cost is measured, closing the gap where CCHVAE's VAE training is
timed by nothing.
**Dependencies**: None. **Must be DONE before Stage 2** — the sweep bakes these columns
into 60 CSVs and re-running to add a column is unaffordable.

---

## Why

`CCHVAE.__init__` → `_load_vae` *trains* the VAE (`vae_params.train: true`, 10 epochs,
batch 32) at `run_cchvae_traintest_pipeline.py:108`. That is **before** `time_start` on
line 119, so the cost appears in neither `gen_train_time` (lines 225-227, which wraps the
MAF density model) nor `cf_search_time`. Without this stage the report's train column
understates CCHVAE by the entire VAE fit while its inference column is unaffected — the
time simply vanishes.

## Column contract

`cf_model_train_time` is **additional** method-specific training not already counted by
`disc_train_time` or `gen_train_time`. Keeping it non-overlapping means
`disc_train_time + gen_train_time + cf_model_train_time` is always a valid total.

| Method | Value | Reason |
|---|---|---|
| DiCE | `0.0` | training-free; `dice_ml.Dice(...)` is construction, not fitting |
| CCHVAE | VAE fit duration | currently unmeasured — the whole point of this stage |
| DiCoFlex | `0.0` | its generator is already `gen_train_time` (`train_dicoflex_generator`, lines 281-293) |

The "method-own train time" the report needs is derived in Stage 5 from a per-method
mapping (DiCoFlex → `gen_train_time`, CCHVAE → `cf_model_train_time`, DiCE → 0.0), **not**
by overloading this column.

---

## Steps

1. Time the CCHVAE construction.
   - File: `counterfactuals/pipelines/run_cchvae_traintest_pipeline.py`
   - Details: Wrap line 108 (`exp = CCHVAE(wrapped_model, hyperparams)`) with
     `cf_model_train_start = time()` / `cf_model_train_time = time() - cf_model_train_start`.
     Log it in the style of the neighbouring `logger.info("Counterfactual search time:
     %.4f seconds", ...)`.
   - Extend `search_counterfactuals`' return tuple with `cf_model_train_time`, update its
     return type annotation and the `Returns:` docstring block, and update the unpacking at
     line 238.
   - Add `df_metrics["cf_model_train_time"] = cf_model_train_time` beside the existing
     assignments at lines 258-261.

2. Emit the column from the DiCE pipeline.
   - File: `counterfactuals/pipelines/run_dice_traintest_pipeline.py`
   - Details: Same shape as step 1 — extend `search_counterfactuals` (return tuple at lines
     222-229, annotation at line 104, its docstring, and the unpacking at line 317) and add
     `df_metrics["cf_model_train_time"] = cf_model_train_time` near line 339. The value is
     `0.0`; add a one-line comment stating DiCE is training-free so a reader does not
     mistake it for a missing measurement.

3. Emit the column from the DiCoFlex pipeline.
   - File: `counterfactuals/pipelines/run_dicoflex_traintest_pipeline.py`
   - Details: Add `df_metrics["cf_model_train_time"] = 0.0` near line 509, with a comment
     that the generator's cost is already in `gen_train_time` (lines 281-293) and must not
     be double counted.

4. Confirm the runners need no change.
   - Details: Both `slurm/run-baselines.sbatch` and `run_seed_experiments.sh` pass identical
     Hydra overrides and neither names metric columns, so nothing should need editing.
     Verify by grepping both for `cf_search_time` and `gen_train_time` — expect no hits. If
     there *are* hits, update them here rather than discovering it after submission.

5. Note the schema change where the schema is documented.
   - File: `slurm/README.md`
   - Details: The "Timing comparability" section states each run records `disc_train_time`,
     `gen_train_time` and `seed`. Add `cf_model_train_time` and state the non-overlap
     contract in one sentence.

---

## Verification

- [ ] `uv run ruff check counterfactuals/pipelines/` — clean
- [ ] `uv run ruff format --check counterfactuals/pipelines/` — clean
- [ ] `grep -l cf_model_train_time counterfactuals/pipelines/run_dice_traintest_pipeline.py
      counterfactuals/pipelines/run_cchvae_traintest_pipeline.py
      counterfactuals/pipelines/run_dicoflex_traintest_pipeline.py` — all three listed
- [ ] Cheapest real end-to-end check:
      `./run_seed_experiments.sh --methods dicoflex --datasets default --seeds 0 --tag stage1-check`
      then confirm the header of
      `results/stage1-check/seed_0/default_split/DiCoFlex/fold_0/cf_metrics_SimpleMLPClassifier.csv`
      contains `cf_model_train_time`, `disc_train_time`, `gen_train_time`, `cf_search_time`,
      `seed`
- [ ] Same for CCHVAE on `default`, asserting `cf_model_train_time > 0` — a zero means the
      timer did not wrap the VAE fit and this stage has **not** met its goal. CCHVAE is
      slow; if the cell exceeds ~2 h locally, defer the assertion to Stage 3's smoke output
      (switch that smoke task to cchvae) and record the deferral in the tracker notes rather
      than grinding
- [ ] `uv run pytest tests/ -q` — no new failures
- [ ] `rm -rf results/stage1-check` before committing, so scratch output is not committed

---

## Commit

`feat(pipelines): record cf_model_train_time for baseline methods`
