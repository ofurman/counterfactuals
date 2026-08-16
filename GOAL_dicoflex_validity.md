# GOAL: fix DiCoFlex's low validity, with real testing

Status: **OPEN.** Opened 2026-08-11.

## 0. CORRECTION (2026-08-11) — "perfect validity" = COVERAGE, and the filter is dead

Two earlier claims were WRONG; measured on the real reference code (`origin/dicoflex`,
run `../cf-dicoflex-ref`, bank, its own classifier):

- **Reference pool_validity = 0.81, NOT 1.0.** (0.81 for 0→1, 0.85 for 1→0.)
- **Reference coverage (≥1 valid CF per factual) = 1.000** (mean 81 valid /100).
- So the paper's DiCoFlex **"Val 1.00" is coverage**, not the fraction-of-CFs-valid
  (`pool_validity`). Nobody's DiCoFlex — reference or ours — has pool_validity 1.0.
- **The `prob_threshold` neighbour-confidence filter is dead code in the reference
  run**: `prepare_dataset_and_models` returns `disc_model = None`
  (`train_dicoflex.py:74`), so the wrapper gets `classifier=None` and
  `dataset.py:81` (`if self.classifier is not None and prob_threshold>0`) is
  skipped. The earlier "the filter is the cause" conclusion is REFUTED.

**Consequences for this whole investigation:**
- Compare against the paper using **coverage** (≥1 valid per factual) for the Val
  column, and report pool_validity separately as a secondary number.
- Our real deficit is coverage, not pool per se: on adult our flow leaves **21% of
  factuals with ZERO valid CF** (n_scored 79/100 → coverage 0.79), whereas the
  reference reaches coverage 1.0 because ~81% of its samples are valid so every
  factual gets several.
- The mechanism is NOT the filter. The reference simply has a **much better-fit
  flow** (trains up to 10000 epochs, patience 50; ours ran 40–200) that reaches
  ~0.81 pool → coverage 1.0. Our undertrained flow reaches ~0.65 pool with a
  fraction of factuals it never flips → coverage < 1.0.
- Likely fix for OUR coverage: train the flow properly (many more epochs) and/or
  draw more samples; NOT a neighbour-confidence filter.

DiCoFlex now produces bounded, on-manifold counterfactuals (proximity/LOF/div all
sane — see `GOAL_dicoflex_working_version.md`), but **validity is too low**. The
paper's DiCoFlex row is **Val. = 1.00**; ours on the WCSS-trained checkpoints is
well below that.

## 1. The measurement (adult, seed 42, scored with minmax_qt→ordinal)

| quantity | value | meaning |
|---|---|---|
| `validity` | **0.634** | mean over factuals of (valid fraction within the kept 10) |
| `pool_validity` | **0.544** | fraction of ALL 100 raw flow samples that flip the class |
| `n_scored / n_factuals` | **79 / 100** | 21% of factuals have **zero** valid CF among 100 samples |
| prox / lof / div | 1.31 / 0.55 / 0.57 | bounded, ballpark of paper (0.85 / 0.31 / 0.40) |

**Key fact: `pool_validity = 0.54` is a property of the raw flow output, before
any selection.** So this is NOT a selection/filtering bug — the flow itself only
generates a valid (target-class) sample ~54% of the time, and for 21% of
factuals it *never* reaches the target class in 100 draws. Selection can reorder
but cannot exceed what the flow produces.

## 2. How selection currently works (so we stop re-deriving it)

- Method (`dicoflex/method.py`): draws `num_counterfactuals=100`, keeps
  `cf_samples_per_factual=100` (i.e. ALL), ordered **valid-first by proximity**
  (`_select_topk_candidates`, now distance-ranked).
- Scorer (`compute_dictum_metrics.py`): `KEEP_PER_FACTUAL=10`,
  `order = argsort(~pool_valid)` → keeps 10 **valid-first**; if fewer than 10
  valid exist, it fills with invalid ones. `validity = mean_i( kept_valid[i].mean() )`,
  and factuals with zero valid contribute 0.

Consequence: to get `validity = 1.0` the flow must yield **≥10 valid per factual**
for essentially every factual. Right now many factuals fall short, and 21% yield
zero. "Generate 100, choose best 10" is already what happens; the problem is the
100 don't contain enough valid ones.

## 3. The reference question (answer FIRST — it defines the target)

**How does the original DiCoFlex / the paper get Val. = 1.00?** Options, and where
to look:
- Does the original **keep only valid** CFs (up to 10) and report validity as
  fraction-of-kept (trivially ~1 for covered factuals), with a separate coverage
  metric? Check `ofurman/DiCoFlex` `counterfactuals/dicoflex/utils.py`
  `evaluate_counterfactuals` / `CFMetrics.validity`, and `origin/dicoflex`
  `cf_methods/dicoflex/*`.
- Does it **regenerate / rejection-sample until valid** (guaranteed validity)?
- Does the DICTUM `advanced_metrics.validity` (the formula we port) count
  per-kept validity, coverage, or "≥1 valid"? Read
  `DICTUM/src/tabdce/utils/advanced_metrics.py`.
- Is the paper's DiCoFlex Val. actually "coverage" (fraction of factuals with any
  valid CF) rather than per-CF validity?

Until this is known we may be chasing a metric-definition mismatch, not a real
generation deficit. **Do not tune knobs before settling this.**

### 3a. ANSWERED (2026-08-11): our `validity` column is a MIS-PORT

The reference validity is **fraction of all scored CFs that flip the class**, in
BOTH the paper's DICTUM code and the original DiCoFlex:
- DICTUM `DICTUM/src/tabdce/utils/metrics.py:7` — `validity(clf, x_cf, y_target)`:
  `correct = (preds == y_target).sum(); return correct / len` = fraction valid.
- DICTUM `advanced_metrics.py:184-185` — `valid_mask = (preds == y_target);
  validity_score = valid_mask.mean()`.
- original DiCoFlex `ofurman/DiCoFlex:counterfactuals/metrics/metrics.py:155` —
  `return (y_cf == self.y_target).mean()`; with a SEPARATE `coverage()` =
  `1 - isnan(X_cf).any(1).mean()`.

Our `compute_dictum_metrics.py` reports **two** columns:
- `pool_validity = pool_valid.mean()` — **this is the reference definition.**
- `validity = mean_i(kept_valid[i].mean())` over the kept-10, valid-first,
  invalid-filled, zero-valid factuals counting 0 — **this is a stricter,
  non-standard metric we invented; it is NOT what the paper reports.**

**Action:** compare the paper's DiCoFlex Val. against our **`pool_validity`**, not
`validity`. Consider renaming/retiring the `validity` column to avoid this trap
(or make it the reference definition and keep the strict one under another name).

adult/42: `pool_validity = 0.544` vs paper 1.00 — so a REAL generation deficit
remains (54% of samples valid, 21% of factuals zero-valid). The metric mismatch
explained the 0.634-vs-something confusion; it does NOT close the gap to 1.0.
That gap is now cleanly a generation problem → hypotheses 2–5.

## 4. Hypotheses (ranked), each a lever to test

1. **Metric definition** (§3) — most likely. Our `validity` averages per-kept-10
   including invalid fills; the paper may keep-only-valid or report coverage.
   Cheap to test: re-score keeping only valid, and separately report coverage.
2. **Temperature. — REFUTED (2026-08-11).** Swept 0.8/1.0/1.2/1.5 on adult/42
   (500 queries, `scripts/probe_validity.sh`). Validity is **dead flat ~0.65**
   and proximity ~1.3–1.4 across the whole range:

   | temp | pool/first-CF valid | strict val | prox | lof | div |
   |---|---|---|---|---|---|
   | 0.8 | 0.652 | 0.670 | 1.37 | 0.62 | 0.62 |
   | 1.0 | 0.650 | 0.666 | 1.36 | 0.60 | 0.66 |
   | 1.2 | 0.654 | 0.664 | 1.32 | 0.59 | 0.68 |
   | 1.5 | 0.652 | 0.658 | 1.35 | 0.59 | 0.71 |

   Temperature does not move validity, so the ~35% invalid rate is NOT a
   sampling-spread issue — the flow's conditional does not reach the target class
   for those factuals at any temperature. This is a TRAINED-FLOW problem, not an
   inference knob. Adult is class-imbalanced (76/24); the hard direction is
   toward the minority class, consistent with a stuck ~0.65.

   → The real lever is **GOAL_dicoflex_remaining_mismatches axis A** (neighbour
   mining / training targets): the reference trains the flow on nearest
   target-class neighbours so it reliably generates flipping points. Diff our
   `dicoflex/data.py::create_dicoflex_dataloaders` against
   `origin/dicoflex:cf_methods/dicoflex/dataset.py` (n_nearest, distance metric,
   `prob_threshold` filtering of training targets). That is the next test.
3. **Sample count.** Draw more than 100 (e.g. 300–1000) so more valid candidates
   exist per factual; raises coverage for the 21% zero-valid factuals *only if*
   their conditional reaches the target at all.
4. **Selection semantics.** Keep the 10 **closest valid** (drop invalid entirely,
   report coverage separately) instead of 10 valid-first-with-invalid-fill.
5. **Flow training.** 200-epoch GPU flow may be undertrained for adult;
   retraining is expensive — treat as last resort.

## 5. Real testing protocol (no more one-off greps)

Build a reusable harness `scripts/probe_validity.py` (or a shell driver) that:

1. Fixes a cheap cell: **adult, seed 42**, checkpoints in
   `results/wcss/seed_42/` (already local, CPU).
2. For each config in a grid `{temperature} × {n_samples} × {selection_rule}`:
   - regenerate CFs (pipeline, `train_model=false`, override temperature /
     `num_counterfactuals`), to a **per-config output dir** (do NOT overwrite a
     shared path — that caused earlier confusion).
   - score with `compute_dictum_metrics` (minmax_qt→ordinal), **capturing stdout
     and stderr to a file** — never pipe scoring to `/dev/null`.
   - append one row to `results/validity_probe.csv`:
     `temp, n_samples, rule, validity, pool_validity, coverage, prox, lof, div`.
3. Print the CSV as a table at the end. Read the table, not scrollback.

Also compute **coverage = n_scored / n_factuals** and a **valid-only validity**
(keep only valid in the kept set) in the scorer, gated behind a flag, so
hypothesis 1/4 can be measured without regenerating.

## 6. Success criterion

Either (a) confirm the paper's Val. is coverage / valid-only and our numbers
already match under that definition, or (b) find the (temperature, n_samples,
selection) that brings honest per-CF validity to ~1.0 without blowing proximity
past the paper's range. Validate on a second dataset (bank or default) before
trusting it. Never inflate validity by tightening selection into the factual or
by silently dropping factuals.

## 7. Do NOT

- Do not measure by piping generation/scoring to `/dev/null` and grepping — write
  results to files and read them.
- Do not overwrite `results/wcss/seed_*/…/counterfactuals_*.csv` during probing —
  use a scratch output dir per config, or the "final" WCSS CFs get clobbered.
- Do not change the shipped selection/metric until §3 is answered.
