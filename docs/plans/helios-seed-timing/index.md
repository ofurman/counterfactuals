# Plan: Helios seed sweep + unified timing/variance report

**Date**: 2026-07-26
**Branch**: tabdce
**Predecessors**: None (builds directly on `slurm/`, added in 8a5e1a9 "seed wiring, training-time metrics, Helios slurm runner")
**Goal**: Run the 3-baseline x 5-dataset x 3-seed sweep on Helios and collapse it into one
document reporting train time, inference time, and seed-to-seed std per method.

---

## Context

### What already exists

`slurm/` is a complete, live-verified Helios runner. `slurm/cluster.env` holds values
confirmed against the cluster on 2026-07-26:

| Setting | Value | Note |
|---|---|---|
| `PLG_ACCOUNT` | `plgcountercontex-gpu-gh200` | the grant's **only** allocation |
| `PLG_PARTITION` | `plgrid-gpu-gh200` | `plgrid`, `plgrid-long`, `cpu` all rejected by `sbatch --test-only` |
| `PLG_GROUP` | `plggcfsgenwro` | group storage for env, cache, results |
| `CPUS_PER_TASK` | `8` | identical for every method **on purpose** — see Timing comparability |
| `PLG_QOS` | `normal` | `now` allows only 1 submitted job per user, useless for arrays |

The work is CPU-only: all three pipelines set `CUDA_VISIBLE_DEVICES = "-1"` at the top of
`main()`. Jobs are submitted **without `--gres`**, allocating GH200 cores without reserving
a GPU — the least wasteful option the grant permits.

`slurm/run-baselines.sbatch` and the local `run_seed_experiments.sh` issue byte-identical
Hydra overrides (`disc_model=simple_mlp`, both `train_model=true`, `experiment.seed`,
per-seed `output_folder`, per-run `hydra.run.dir`), so cluster and laptop results are
interchangeable and either can feed the aggregator.

### What is missing

Three gaps, in the order they must be closed:

1. **CCHVAE's VAE training time is measured by nothing.** `CCHVAE.__init__` calls
   `_load_vae`, which *trains* the VAE (`vae_params.train: true`, 10 epochs, batch 32 over
   ~24k rows) at `run_cchvae_traintest_pipeline.py:108` — **before** `time_start` on line
   119. So it lands in neither `gen_train_time` (that timer wraps the MAF density model,
   lines 225-227) nor `cf_search_time`. A train-time table built on today's columns
   understates CCHVAE. This must be fixed **before** submission, or 60 runs record the
   wrong thing.
2. **No seed aggregator.** `slurm/README.md` states it outright: *"Aggregating the three
   seed roots into the `mean ± std` cells that `scripts/generate_latex_tables.py` already
   parses still needs a small aggregator script — not yet written."*
   `scripts/calculate_metrics.py` aggregates over **folds inside one root**
   (`load_and_aggregate_metrics`, line 45) and cannot cross seed roots — each seed root has
   only `fold_0`.
3. **No single report doc.** Nothing turns 60 CSVs into the one document asked for.

### Timing semantics (why the report needs caveats, not just numbers)

The three columns do not mean the same thing for every method:

| Method | `disc_train_time` | `gen_train_time` | `cf_search_time` | Method's own training |
|---|---|---|---|---|
| DiCE | SimpleMLP fit | MAF density model (metrics only; DiCE doesn't use it to search) | random search over 100 CFs | none — DiCE is training-free |
| CCHVAE | SimpleMLP fit | MAF density model (metrics only) | latent search, `num_counterfactuals=100` | **VAE, currently untimed** → Stage 1 |
| DiCoFlex | SimpleMLP fit | **the DiCoFlex conditional generator** (`train_dicoflex_generator`) | flow sampling | counted in `gen_train_time` |

So "train time" is only comparable once the method's own training is isolated from the
shared discriminator and the metrics-only density model. Stage 1 adds
`cf_model_train_time` for exactly that, and the report tables the shared components
separately.

### The DiCoFlex target_class asymmetry

DiCoFlex defaults to `target_class: 1` (`dicoflex_traintest_config.yaml:48`) while DiCE and
CCHVAE default to `0`. Since all three select factuals as `y_test != target_class`,
DiCoFlex explains a **disjoint, opposite-direction** query set (saved factual counts confirm
complementarity: adult 3674 vs 6326, bank 2746 vs 7254). That blocks any paired test
involving DiCoFlex. The `seeds-tc0` array re-runs it at `target_class=0` to make the
methods poolable. Per the Decisions below, the report tables **both**, clearly labelled.

---

## Strategy

Two phases. Phase A is sequential and gated on the cluster; Phase B is local analysis and
can be developed while Phase A's jobs queue.

**Phase A — instrument, then run (Stages 1-4)**
Stage 1 patches the timing instrumentation *before* anything is synced, because the sweep
bakes its columns into 60 CSVs. Stages 2-4 sync, build the aarch64 environment, calibrate
walltimes with a single smoke task, then submit and monitor 60 array tasks.

**Phase B — aggregate and report (Stages 5-6)**
Stage 5 writes the missing seed aggregator; it depends only on the CSV *schema*, not on the
sweep finishing, so it can be built and unit-tested against synthetic fixtures and the
smoke output while Stage 4's jobs run. Stage 6 pulls results back and emits the single
document.

Submission is authorized: the run proceeds smoke → full sweep unattended. Scope is the 3
baselines only — the `cf-constraints` array is deliberately **not** submitted (achieved by
passing `--only <method>` to `submit-all.sh`, which skips the constraints block).

---

## Resources

- [`resources/commands.md`](resources/commands.md) — every command this plan runs: credential
  handling, sync, login-node setup, submission, monitoring, verification, retrieval, plus the
  result layout and array-index mapping.
- [`resources/metrics-schema.md`](resources/metrics-schema.md) — the `cf_metrics_*.csv` schema
  with per-method timing semantics, the CCHVAE gap, the `target_class` asymmetry and factual
  counts, and the aggregation conventions Stage 5 must follow.

Read `metrics-schema.md` before Stage 1 or Stage 5; read `commands.md` before Stages 2-4 and 6.

---

## Success Criteria

| Metric | Baseline | Target | Rationale |
|--------|----------|--------|-----------|
| Method-own train time measured | 2 of 3 methods (CCHVAE's VAE untimed) | 3 of 3 | A train-time column that omits CCHVAE's VAE is wrong, not merely incomplete |
| `cf_metrics_SimpleMLPClassifier.csv` files present | 0 on cluster | 60 (45 `seeds` + 15 `seeds-tc0`), each non-empty | 3 methods x 5 datasets x 3 seeds, plus the tc0 re-run |
| Array tasks `COMPLETED` with `ExitCode 0:0` | n/a | 60/60, or every shortfall logged in Backlog with its sacct state | A job ID is not evidence of success (plgrid-run rule 8) |
| Seed aggregator | does not exist | `scripts/aggregate_seed_results.py`, mean ± std across seed roots | The gap `slurm/README.md` names |
| Timing report | none | one committed doc: train + inference time per method, ± std over 3 seeds, DiCoFlex at tc=0 **and** tc=1 | The requested deliverable |
| Walltime accuracy | extrapolated from local `cf_search_time`, retraining never measured on this hardware | smoke-calibrated before the 60-task submission | Avoids burning the grant on TIMEOUT |

---

## Files That May Be Changed

### Pipeline instrumentation (Stage 1)
- `counterfactuals/pipelines/run_cchvae_traintest_pipeline.py` — time VAE construction, emit `cf_model_train_time`
- `counterfactuals/pipelines/run_dice_traintest_pipeline.py` — emit `cf_model_train_time` (0.0, training-free)
- `counterfactuals/pipelines/run_dicoflex_traintest_pipeline.py` — emit `cf_model_train_time` (= generator training)

### Cluster assets (Stages 2-4)
- `slurm/submit-all.sh` — `WALLTIME` values only, if smoke calibration contradicts them
- `slurm/cluster.env` — **do not overwrite**; it holds live-verified values (see Stage 2 step 3)

### Analysis (Stages 5-6)
- `scripts/aggregate_seed_results.py` — new
- `tests/test_aggregate_seed_results.py` — new
- `docs/benchmarks/helios-seed-timing-report.md` — new, the deliverable
- `slurm/README.md` — drop the "not yet written" note once Stage 5 lands

### Not changed
- `.env` — read-only, gitignored. Never synced, never copied to PLGrid (plgrid-run rule 1).
- `run_seed_experiments.sh` — the local twin stays as-is

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Instrument method-own train time](stages/01-instrument-train-time.md) | PENDING | Must land before Stage 2 sync | |
| 2 | [Sync code and build cluster env](stages/02-sync-and-env.md) | PENDING | | |
| 3 | [Smoke calibration](stages/03-smoke-calibration.md) | PENDING | | |
| 4 | [Submit and monitor the sweep](stages/04-submit-sweep.md) | PENDING | 60 tasks; long-running | |
| 5 | [Seed aggregator script](stages/05-seed-aggregator.md) | PENDING | Independent of Stage 4 completion | |
| 6 | [Timing and variance report](stages/06-timing-report.md) | PENDING | The deliverable | |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

---

## Execution Protocol

This plan is built for **autonomous, unattended execution**. The guiding principle is
**keep making progress**: resolve problems in place when you can, defer them when you
can't, and never halt the whole plan over a single fixable or deferrable issue.

For each stage:

1. **Read the progress tracker** above and pick the stage to work on. If a stage is
   **IN_PROGRESS**, a previous run was interrupted mid-stage — resume and finish that one
   (re-read its steps, inspect the working tree to see what's already done) before
   starting anything new. Otherwise, take the first **PENDING** stage.
2. **Read the stage file** -- follow the link in the tracker to the stage's .md file.
3. **Read resources** -- if the stage references shared resources, find them in `resources/`.
4. **Resolve ambiguity yourself** -- there is no user to ask during an autonomous run.
   Pick the most reasonable interpretation that fits the codebase and existing
   conventions, record it under **Decisions**, and proceed. Only defer to the Backlog
   if the ambiguity genuinely blocks any sensible implementation.
5. **Implement** -- execute the steps described in the stage.
6. **Validate** -- run the verification checks and the test suite. **If anything fails,
   do not stop — triage it via the self-healing loop below.**
7. **Update this index** -- mark the stage DONE in the progress tracker, add brief notes
   about what was done and any deviations. Log every problem you hit in **Fixed Issues**
   (if resolved) or **Backlog** (if deferred). Never silently drop a problem.
8. **Commit** -- create an atomic commit with the message specified in the stage.
   Include all changed files (code, config, docs, and this plan's index.md).

Repeat until every stage is DONE or terminally deferred. After the last stage, **sweep
the Backlog**: attempt any items that are now resolvable, and leave the rest for a
follow-up run.

### Stage 4 is a long wait, not a long task

The 60-task sweep may take days of wall-clock (CCHVAE requests 24 h per task, capped at 8
concurrent, and `sbatch --test-only` estimated a ~4 day start under the `normal` QOS). Do
**not** busy-poll. Stage 4 specifies a polling cadence. While waiting, work **Stage 5**,
which has no dependency on the sweep finishing. If the sweep is still running when Stage 5
is DONE, record the outstanding job IDs in the tracker notes and end the run cleanly — a
later invocation resumes at Stage 4's verification.

### Cluster-specific guardrails

These come from `plgrid-run/SKILL.md`; they override convenience:

- Never print, commit, or copy `.env` or the PLGrid password to the cluster. Use
  `PLG_LOGIN` only, over an SSH key/agent. If SSH auth fails, that is an **external
  blocker** → Backlog, do not attempt password automation.
- Run `sbatch --test-only` before any **new** job shape.
- Never overwrite an existing path or unexpected symlink during storage setup.
- Treat `$SCRATCH` as temporary; durable results live in group storage.
- Verify with `sacct`, logs, and the expected result files. A job ID is not success.

### Self-healing loop (handling problems)

When a step fails — failing test, build/lint/type error, a bug in the new code, an
unexpected runtime error:

1. **Triage** the problem as *light* or *heavy*.
   - **Light** -- self-contained and fixable in a focused effort: a failing unit test,
     a lint/type error, a missing import, a small logic bug in code you just wrote.
   - **Heavy** -- needs an architectural decision, spans many files, depends on an
     external blocker, contradicts the plan's assumptions, or has already survived a
     fix attempt.
2. **Light → delegate the fix to a subagent.** Spawn a focused subagent (Agent/Task
   tool) with: the failing command and its full output, the relevant file paths, the
   stage goal, and a crisp deliverable (e.g. "make `<test>` pass without weakening
   assertions"). Delegating keeps the main execution context clean. Re-run verification
   when it returns. Cap at **2 attempts per issue** — if still failing, treat it as heavy.
3. **Heavy → defer to the Backlog.** Add a self-contained entry (see the Backlog table).
   Do **not** keep grinding and do **not** halt the plan.
4. **Decide the stage's disposition:**
   - If the stage's core goal is met without the deferred item → mark **DONE**, note the
     backlog reference, and continue.
   - If the deferred item is essential to this stage → mark **BLOCKED**, note the backlog
     reference, and continue to the next *independent* stage. Only stop the run when every
     remaining stage depends on blocked work.
5. **Record** the outcome: resolved problems → **Fixed Issues**; deferred problems → **Backlog**.

**Partial-sweep rule.** If some array tasks fail while others succeed, do **not** block
Stage 6. Generate the report from whatever completed, mark every missing cell explicitly
(`n/a (task failed)`, never a silent gap or an imputed value), and log the failures in the
Backlog with their sacct states. A report over 57 of 60 cells that says so is useful; a
report that hides 3 holes is not.

### Guardrails

- Keep every commit in a working, buildable state.
- **Never weaken, skip, or delete a test to make it pass.** If a test is genuinely wrong,
  fix it correctly and note it in Fixed Issues.
- Never use `git commit --no-verify`.
- Don't expand a stage's scope to chase a heavy problem — that's what the Backlog is for.
- Never resubmit an array that is already queued or running. Check
  `slurm/logs/submitted.tsv` and `squeue -u $USER` first — a double submission burns the
  grant twice and interleaves writes into the same result directories.

---

## Fixed Issues

Problems encountered during execution and resolved (in place or via a fix subagent).
Leave empty until execution surfaces something.

| # | Stage | Symptom | Root Cause | Resolution | Fixed By |
|---|-------|---------|-----------|------------|----------|
| | | | | | |

---

## Backlog (Deferred Issues)

Problems deferred for later — too heavy to fix inline without derailing the plan.
Each entry must be **self-contained enough for a future run to pick it up cold**:
state the symptom, where it came from, and a concrete lead for resolving it.

| # | Title | Origin Stage | Severity | Why Deferred | Suggested Next Step | Status |
|---|-------|--------------|----------|--------------|---------------------|--------|
| | | | | | | |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`. When an item is resolved, flip its
status and summarize the fix in **Fixed Issues**. Heavy items may warrant their own
follow-up plan — link it here.

---

## Decisions

Made during planning, with the user present:

1. **Authorization: smoke, then full sweep, unattended.** The run is authorized to spend
   gpu-gh200 grant hours. It must still submit the smoke task first and reconcile walltimes
   before the 60-task submission.
2. **Scope: 3 baselines only.** DiCE, CCHVAE, DiCoFlex — 45 `seeds` tasks plus 15
   `seeds-tc0`. The `cf-constraints` array is out of scope; TabDCE is out of scope (GPU, no
   seed sbatch exists, and adding one was judged extra risk for this deliverable).
   *Implementation note:* `submit-all.sh` submits constraints only when `--only` is empty,
   so three `--only <method>` invocations give exactly the 60 wanted tasks.
3. **DiCoFlex reported at both target classes, clearly labelled.** `tc=1` continues the
   current paper numbers; `tc=0` is the poolable set. The report must state which rows are
   comparable to DiCE/CCHVAE and must not silently mix them.
4. **`cf_model_train_time` added rather than redefining existing columns.** Additive keeps
   `scripts/calculate_metrics.py`, `generate_latex_tables.py`, and any already-collected
   CSVs working; redefining `gen_train_time` would silently change the meaning of prior
   results.

To be recorded during execution:

5. *(reserved — record runtime decisions here)*
