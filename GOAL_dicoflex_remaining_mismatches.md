# GOAL: find the REMAINING DiCoFlex mismatches (after scaling/temp/selection/clamp)

Status: **OPEN.** Opened 2026-08-09. For a separate agent to pick up.

This continues `GOAL_dicoflex_working_version.md`. The catastrophic failure
(proximity 1e32, LOF `inf`) is fixed. What remains is a **quality gap**: our
DiCoFlex is bounded and on-manifold but its mean proximity/diversity are still
several × the paper. This file lists the *other* potential mismatches to diff
against the working reference so the gap can be closed.

The reference to diff against is **`origin/dicoflex`** (a faithful port;
`counterfactuals/cf_methods/dicoflex/*.py`, `counterfactuals/generative_models/maf.py`).
Secondary references: `origin/ofurman/dicoflex`, and the external
`github.com/ofurman/DiCoFlex` (`counterfactuals/dicoflex/`, `datasets/DCENF/*.py`).
Our code under investigation: `counterfactuals/cf_methods/local_methods/dicoflex/`
on `dictum-aligned-eval`.

---

## 1. Progress so far (bank, seed 42, small flow, 40 epochs)

| stage | DiCoFlex prox | DiCoFlex LOF | validity |
|---|---|---|---|
| standard scaler | 3.4e32 | inf | 1.0 |
| + `minmax_qt` (MinMax num + QT cat) | 1e17 | inf | 1.0 |
| + temperature 0.8 | 338 | finite | 0.95 |
| + proximity selection | 28.6 | 0.327 | 1.0 |
| + numeric clamp to box | **1.08** | **0.248** | **1.00** |
| **paper DiCoFlex** | **0.89** | **0.16** | **1.00** |

With all four fixes the bank row (prox 1.08 / LOF 0.25 / div 0.46 / val 1.0) is
within undertraining noise of the paper (0.89 / 0.16 / 0.39 / 1.0) — this is an
undertrained probe (small flow, 40 epochs, 500 queries). The remaining §2 items
are about closing the last ~20% and confirming the numbers are not an artefact of
the clamp/selection but of the flow genuinely generating near points (axis A).

Fixes already landed on this branch:
- `preprocessing/scalers.py::QuantileTransformCategoricalStep` + `factory.py`
  `minmax_qt`; wired in `conf/dictum_dicoflex_config.yaml`.
- Temperature: `models/generative/maf/maf.py::sample_and_log_proba(temp=...)`,
  `DiCoFlexParams.temperature`, config `temperature: 0.8`.
- Proximity selection: `cf_methods/.../dicoflex/method.py::_select_topk_candidates`
  now ranks by L1 distance to the factual (was `-target_probs`).
- Numeric clamp: `run_dicoflex_traintest_pipeline.py` clips numeric cols to [0,1]
  before inverse-transform (`clamp_numeric_to_box`, default on).

Metrics are read with
`compute_dictum_metrics --generation-scaler minmax_qt --scaler standard --metric-encoding ordinal`.

## 2. Remaining mismatches to investigate — ranked by expected impact

### A. Flow training targets / neighbour mining (LIKELY THE BIG ONE)
The reference trains the flow to generate the `n_nearest` **nearest target-class
points** to each factual, so it learns to produce *nearby* counterfactuals. If our
neighbour mining differs (count, distance metric, per-class handling, filtering),
the flow learns to generate farther points and no inference trick fully recovers.
- Ours: `cf_methods/local_methods/dicoflex/data.py::create_dicoflex_dataloaders`
  (n_neighbors, p_values, noise, chunking).
- Ref: `origin/dicoflex:counterfactuals/cf_methods/dicoflex/dataset.py`
  (`nearest_indices = np.argsort(dist_matrix)[:, :n_nearest]`, line ~90;
  `prob_threshold` filtering line ~81-83; `n_nearest` default 5 vs our
  `n_neighbors: 16`).
- Test: match `n_nearest`, the distance metric, and the prob_threshold filter;
  retrain; re-score.

### B. Context construction
- Ours: `method.py::_build_context` = `[X, class_one_hot, mask, p_value]`
  (mask = actionability mask, currently off; `p_value` from `inference_p_value`).
- Ref: `origin/dicoflex:generation.py` context = `[factual, class_one_hot, mask, p]`.
- Check: mask semantics (ref masks = `ones*1e-3`, effectively none; ours?),
  `p_value`/`p_values` (ref `[1e-2, 2.0]`, ours `[2.0]`), ordering/dims must match
  what the flow was trained with.

### C. Training noise level
- Ref: `noise_level` on numeric + `N(0,0.08)` on categoricals
  (`dataset.py:200,229`); ours `noise_level: 0.02` (`data.py::_apply_noise`).
- Higher training noise smooths the density and reduces the sampling tail.

### D. Epochs / early stopping
- Ref trains the flow ~10 epochs; ours 40 (and 200 in the full config). More
  epochs sharpen the density → bigger inverse-transform tail. Test 10 vs 40.

### E. Sample count & metric-side selection
- Ours draws `num_counterfactuals: 100`, keeps `cf_samples_per_factual: 100`; ref
  `n_samples=10`. Test 10.
- **Scorer still keeps 10 per factual by VALIDITY, not proximity**
  (`compute_dictum_metrics.py` `KEEP_PER_FACTUAL=10`, `order = argsort(~pool_valid)`).
  Even with proximity selection in the method, verify the scorer isn't re-selecting
  a worse 10. Consider proximity-ranked keep in the scorer, or confirm the method's
  order is preserved.

### F. Classifier (decision boundary shapes the target region)
- Ref MLP hidden `[256,256]`, ~10 epochs; ours `dictum_mlp` hidden `[32,32]`,
  100 epochs. A different boundary changes where "valid" is and thus proximity.
  Confirm this is intended (the aligned setup fixes the classifier for all methods).

### G. Scaler variant: QT vs categorical-noise
- We implemented QuantileTransformer on categoricals (external GitHub repo's
  recipe). But `origin/dicoflex:dataset.py:400` uses **plain MinMax + `N(0,0.08)`
  categorical noise, no QT**. Determine which variant produced the paper's
  DiCoFlex numbers and match it. This changes the whole categorical gen space.

### H. MAF architecture
- Ref CF flow: hidden 64 / 5 layers / 2 blocks. Ours: `small_maf` 16/2/2 (current
  test), `large_maf` 16/8/4 (config default). Match the reference arch once A–G are
  settled.

## 3. Method

1. Get `origin/dicoflex` into a worktree:
   `git worktree add ../cf-dicoflex-ref origin/dicoflex`.
2. Diff, in order A→H, our file against the reference file; note every concrete
   difference (default value, formula, ordering).
3. Change ONE axis at a time, retrain bank/seed42 (`gen_model=small_maf`,
   `gen_model.epochs=40`, `n_test_samples=500` for speed), re-score, record the
   prox/LOF/div delta in a table. Keep the ones that help.
4. Confirm on a second dataset (adult or default) before trusting a change.
5. Success = DiCoFlex prox ≈ 0.9, LOF ≈ 0.16, validity ≈ 1.0, matching the paper
   row, with the same numbers across seeds 42/43/44.

## 4. Watch-outs

- Every generation-space change (A, C, D, G, H) requires **retraining** the flow;
  temperature/selection/clamp/keep (temp, B-partial, E-scorer) are inference-only.
- The mean-based prox/div are tail-sensitive; report **median** alongside mean so a
  single far factual doesn't hide real progress (LOF is already median-based).
- Do not "fix" numbers by tightening the clamp or the selection into the factual —
  that would understate proximity artificially. The goal is that the *flow itself*
  generates near, valid, on-manifold points (axis A), with clamp only a backstop.
