# Stage 6: Timing and variance report

**Goal**: Retrieve the sweep results and produce the single deliverable document reporting
train time, inference time, and seed-to-seed std for DiCE, CCHVAE and DiCoFlex.
**Dependencies**: Stage 5 DONE (aggregator exists). Stage 4 DONE, **or** partially complete —
see the index partial-sweep rule.

---

## Steps

1. Pull results back from group storage.
   - Command:
     ```bash
     set -a; . ./.env; set +a
     rsync -av \
       "$PLG_LOGIN@login01.helios.cyfronet.pl:/net/pr2/projects/plgrid/plggcfsgenwro/$PLG_LOGIN/counterfactuals/results/" \
       ./results/
     ```
   - Details: Resolve the real `$PLG_GROUPS_STORAGE` path by echoing it **on the login node**
     first — do not trust the literal above, it is a placeholder shape. Quote correctly so the
     variable expands remotely, not locally.
   - `results/` is gitignored; the CSVs stay local and only the report is committed.

2. Run the aggregator over both tags.
   - Command:
     ```bash
     uv run python scripts/aggregate_seed_results.py \
       --results-root results --tags seeds seeds-tc0 \
       --out-csv results/seed_aggregate.csv \
       --out-markdown results/seed_aggregate.md
     ```

3. Write the report.
   - File: `docs/benchmarks/helios-seed-timing-report.md` (new — the deliverable)
   - Structure:
     1. **What was run** — 3 methods x 5 datasets x 3 seeds on Helios
        `plgrid-gpu-gh200`, CPU-only (all pipelines force `CUDA_VISIBLE_DEVICES=-1`),
        `CPUS_PER_TASK=8` with `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` pinned to it. State the
        git commit, the array job IDs, and the date. Note that DiCoFlex appears twice
        (`tc=1`, `tc=0`).
     2. **How to read the timing columns** — reproduce the definition table below verbatim.
        This section is what makes the numbers interpretable; do not compress it away.
     3. **Table 1 — training time (s), mean ± std over 3 seeds**, rows = method x target
        class, columns = the 5 datasets. Report `method_train_time` as the headline and
        `disc_train_time` / `gen_train_time` / `total_train_time` in a following table, so the
        shared discriminator and the metrics-only density flow are visible but not conflated
        with method cost.
     4. **Table 2 — inference time**, both absolute `cf_search_time` and
        `inference_time_per_factual`, mean ± std. Flag directly under the table that
        absolute times are **not** comparable between `tc=0` and `tc=1` rows because the
        query sets differ in size; per-factual times are.
     5. **Table 3 — seed-to-seed variance of the quality metrics** for the metrics in
        `counterfactuals/pipelines/conf/metrics/default.yaml` (coverage, validity,
        actionability, sparsity, proximity_*, prob_plausibility, …), mean ± std over seeds.
        This is the "std for the re-evaluated methods" figure.
     6. **Caveats** — a short, honest list:
        - CCHVAE's VAE training was untimed before this sweep (Stage 1); numbers from earlier
          runs are not comparable on the train column.
        - `gen_train_time` means different things per method (see the definition table).
        - DiCoFlex `tc=1` vs everything else is a disjoint query set; only the `tc=0` rows
          support paired tests.
        - Whether no-`--gres` jobs on the GPU partition are billed as GPU-hours (answer
          recorded in Stage 3).
        - Any missing cells, named explicitly with their Backlog reference.
     7. **Reproduce** — the exact `submit-all.sh` and `aggregate_seed_results.py` invocations.

   - The definition table to reproduce:

     | Column | Meaning | Comparable across methods? |
     |---|---|---|
     | `disc_train_time` | SimpleMLP discriminator fit; identical config for all methods | yes |
     | `gen_train_time` | DiCE/CCHVAE: MAF density model, used only for log-density metrics. DiCoFlex: its own conditional generator | **no** — different objects |
     | `cf_model_train_time` | Method-specific training not counted above. DiCE 0 (training-free), CCHVAE the VAE, DiCoFlex 0 (already in `gen_train_time`) | yes, as a component |
     | `method_train_time` | Derived: the method's own training. DiCE 0, CCHVAE `cf_model_train_time`, DiCoFlex `gen_train_time` | **yes** — the headline train figure |
     | `total_train_time` | `disc + gen + cf_model` | yes |
     | `cf_search_time` | CF generation for the whole factual set | only within the same target class |
     | `inference_time_per_factual` | `cf_search_time / number_of_instances` | yes |

4. Link it from the benchmarks index.
   - File: `docs/benchmarks/index.md`
   - Details: Add one line pointing at the new report, matching the existing entry style.

5. Sanity-check the numbers before believing them.
   - Details: Cross-read three cells against their raw CSVs by hand. Specifically confirm
     CCHVAE's `method_train_time > 0` (if it is 0, Stage 1 did not take effect on the cluster
     and the report is wrong — treat as heavy, Backlog, and say so in the doc rather than
     publishing a zero). Confirm a std is not identically `0.00` across every cell, which
     would indicate seeds are not actually varying — cross-check the `seed` column in the raw
     CSVs holds 0/1/2.

---

## Verification

- [ ] `results/seed_aggregate.csv` and `results/seed_aggregate.md` exist and are non-empty
- [ ] `docs/benchmarks/helios-seed-timing-report.md` exists and contains all three tables plus
      the definition table and the caveats section
- [ ] Every cell is a `mean ± std` or an explicit `n/a (<reason>)` — no blanks, no imputed
      values
- [ ] Each table states `n_seeds` (or flags cells where it is below 3)
- [ ] DiCoFlex appears as two clearly labelled rows (`target_class=1`, `target_class=0`) and
      the doc states which rows are poolable with DiCE/CCHVAE
- [ ] CCHVAE `method_train_time > 0` in the report, or the discrepancy is documented as a
      Backlog item in the doc itself
- [ ] Timing std values are not all identically zero
- [ ] `uv run pytest tests/ -q` — no regressions
- [ ] `git status` shows no `results/` files staged

---

## Commit

`docs(benchmarks): add Helios seed timing and variance report`
