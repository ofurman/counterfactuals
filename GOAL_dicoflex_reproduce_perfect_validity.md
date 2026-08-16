# GOAL: reproduce DiCoFlex's perfect validity by running the existing working code

Status: **OPEN.** Opened 2026-08-11.

We once had ~100% validity for DiCoFlex; the current re-implementation
(`cf_methods/local_methods/dicoflex`, on `develop` / `dictum-aligned-eval`) gives
only ~0.65 honest validity. This GOAL is to **find and RUN the previously-existing
correctly-working code** to reproduce perfect validity — **not** to write a new
implementation or patch the current one. Two parallel investigations (git
archaeology + code diff) have already located the working version and the
regression; this file records that and lays out the run/verify plan.

## 0. Which "validity 1.0" is real (read this first)

There are three things that read as "DiCoFlex validity ≈ 1.0"; only one is real:
1. **Degenerate** (`results/dictum`, commit `abbe59d`/`926b179`, StandardScaler):
   validity reads ≈1.0 only because the classifier is confidently wrong on
   off-manifold garbage (prox 1e9–1e33, LOF `inf`). NOT real. See
   `GOAL_dicoflex_divergence.md`.
2. **Metric artifact** (`results/dicoflex_clamp/mm.csv`, current worktree): the
   DICTUM-**mis-ported** strict `validity` column hits 1.000 for Bank, but the
   paper-faithful metric `pool_validity` is only 0.54–0.65. See
   `GOAL_dicoflex_validity.md` §3a — the reference validity is
   `(argmax clf(x_cf) == y_target).mean()` = our `pool_validity`, NOT `validity`.
3. **Genuine** — the original DiCoFlex (`origin/dicoflex`) reaches real ~100%
   `pool_validity`, matching the paper (`_NIPS2026__TabDCE/results.tex` DiCoFlex
   Val = 1.00 on all 5 datasets). No local result file exists for it yet; that is
   what we must reproduce by running its code.

**Success is measured by `pool_validity` (fraction of ALL generated CFs that flip),
never the strict `validity` column.**

## 1. The working version (identified)

Branch **`origin/dicoflex`** — a faithful, self-contained DiCoFlex:
- Code: `counterfactuals/cf_methods/dicoflex/{dataset,generation,dicoflex,training}.py`,
  `counterfactuals/generative_models/maf.py`, entrypoint
  `counterfactuals/examples/train_dicoflex.py`.
- Bundled data + pre-trained classifiers: `data/<ds>/{train,test}.csv`,
  `data/<ds>/model.pt`, `data/<ds>/y_{train,test}_pred.npy`. So it runs
  standalone — no external data prep.
- It reaches ~100% validity because of features the current code dropped.

## 2. Why current is ~0.65 (the regression — from the code diff)

| aspect | `origin/dicoflex` (~100%) | current `dictum-aligned-eval` (~0.65) |
|---|---|---|
| **classifier-confidence neighbour filter** (PRIMARY) | **YES** — `dataset.py:81-86`: for training-target neighbours, `posterior = clf.predict_proba(X_cf)[:,cf_class]`; `dist_matrix[:, posterior < prob_threshold] = inf`. `prob_threshold` 0.55 (tabular) – 0.98. Flow trains ONLY on confidently-target neighbours → generates deep-in-target points. | **ABSENT** — `data.py::_compute_neighbors_chunked` selects nearest target-class neighbours by raw label only, incl. boundary/misclassified points → flow learns the boundary → ~35% of samples fall on the wrong side. |
| scaler | MinMax [0,1] | `standard` (unbounded) — secondary |
| sampling temperature | `temperature=0.8` | none passed (default) — secondary |
| relabeling by classifier | YES | YES (same — NOT the cause) |

Root cause = the **dropped `prob_threshold` neighbour filter**. Confirmed no
inference knob (temperature) moves validity (`GOAL_dicoflex_validity.md` §4.2
sweep), so it is a trained-flow/training-data problem, exactly what this filter
controls.

## 3. Plan — run the existing working code (do NOT rewrite)

### Phase A — reproduce ~100% validity from `origin/dicoflex`, one dataset, local
1. `git worktree add ../cf-dicoflex-ref origin/dicoflex` (isolated, read-only run).
2. Build its env with `uv` (its own `pyproject.toml`/`requirements`). It targets
   an older API; use its pinned deps, not the current worktree's.
3. Read `examples/train_dicoflex.py` for the exact call (dataset class, disc
   model, `n_nearest`, `prob_threshold`, `n_samples`, `temperature`). Run it for
   **bank** (fast) as-is.
4. Compute `pool_validity` = fraction of generated CFs with
   `argmax clf(x_cf) == y_target`. Confirm ≈1.0. If yes, the working code and the
   `prob_threshold` mechanism are confirmed.

### Phase B — full grid on the cluster (WCSS)
5. rsync the `cf-dicoflex-ref` worktree to WCSS
   (`/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/`), build env with
   cache on lustre (mind the inode quota — CPU torch if possible), and run
   `train_dicoflex.py` for **all 5 datasets × 3 seeds** as Slurm jobs (the harness
   from the earlier sweep can be adapted; GPU only if the code supports it).
6. Pull the generated counterfactuals + the per-run validity.

### Phase C — score in the paper's aligned space
7. The reference emits CFs in **original units** (it inverse-transforms). Score
   them with our `scripts/compute_dictum_metrics` using
   `--generation-scaler minmax --scaler standard --metric-encoding ordinal` (the
   reference generates in MinMax, not minmax_qt), reporting **`pool_validity`** as
   validity, alongside prox/spars/lof/div. Compare to the paper's DiCoFlex row on
   all 5 datasets.

### Phase D — attribute the regression (confirmation, optional)
8. To PROVE the filter is the cause without writing new code: run `origin/dicoflex`
   twice — once with `prob_threshold` at its real value (0.55), once with
   `prob_threshold=0.0` (its own config knob, which disables the filter — this is
   NOT new code, just its existing parameter). Expect validity to drop toward the
   current ~0.65 when the filter is off. This isolates the mechanism using only
   existing code.

## 4. Experiments to test (existing, recorded)

- `origin/dicoflex` `examples/train_dicoflex.py` — the canonical run.
- Its bundled `data/<ds>/model.pt` classifiers — reuse so validity is measured
  against a fixed classifier (matches how the paper fixes the classifier per
  dataset).
- Cross-check against the paper table `_NIPS2026__TabDCE/results.tex` (DiCoFlex Val
  = 1.00, and the prox/spars/lof/div columns) as ground truth.

## 5. Deliverable / definition of done

- A recorded run of the existing `origin/dicoflex` code showing **`pool_validity`
  ≈ 1.0** on ≥2 datasets, with prox/lof/div in the paper's range, scored in the
  aligned metric space.
- Phase D confirming `prob_threshold` on/off flips validity ~1.0 ↔ ~0.65.
- A one-line statement of the exact reproduction command + config, so the DiCoFlex
  baseline row can be regenerated on demand.

## 6. Constraints (per the request)

- **Run existing code; create nothing new.** No new method, no patching the
  current `local_methods/dicoflex`. Use `origin/dicoflex` as-is, driving behaviour
  only through its existing parameters (`prob_threshold`, `temperature`,
  `n_samples`, dataset).
- Local for the fast single-dataset check; **cluster (WCSS)** for the full
  5×3 grid.
- Report `pool_validity`, never the strict `validity` column.

## 7. Evidence index

- Code diff + `prob_threshold` lines: `origin/dicoflex:cf_methods/dicoflex/dataset.py:81-86`.
- Metric definition: `GOAL_dicoflex_validity.md` §3a; DICTUM `metrics.py:7`,
  `advanced_metrics.py:184`; original `ofurman/DiCoFlex:metrics/metrics.py:155`.
- Temperature refuted: `GOAL_dicoflex_validity.md` §4.2.
- Paper ground truth: `_NIPS2026__TabDCE/results.tex` `tab:protocol_B`.
