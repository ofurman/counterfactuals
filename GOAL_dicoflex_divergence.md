# GOAL: find out why DiCoFlex fails in the DICTUM-aligned sweep

Status: **OPEN**. Opened 2026-08-07, after the first full aligned sweep
(`results/dictum`, 45 cells, tag `dictum`, commit `abbe59d`).

The aligned sweep produced usable numbers for DiCE and CCHVAE. DiCoFlex's are
not usable: proximity of order 1e33, LOF of `inf`, and one cell that failed
outright. This file records what is known so the investigation does not restart
from scratch.

There is a second, independent problem affecting **all three methods**, recorded
at the bottom. It is arguably the more serious of the two.

---

## 1. Symptoms

`scripts/compute_dictum_metrics.py` over `results/dictum`, seed 42, z-scored
ordinal metric space:

| Dataset | DiCE prox | CCHVAE prox | DiCoFlex prox |
|---|---|---|---|
| Adult | 1.065 | 0.334 | 1.5e9 |
| Bank | 1.117 | 0.498 | 3.4e32 |
| Default | 1.074 | 0.429 | 1.5e30 |
| GMC | 2.169 | 0.363 | 3.4e33 |
| Lending Club | 2.481 | 0.587 | cell failed |

DiCoFlex validity still reads ≈1.0, because the classifier confidently assigns
points that far outside the data to the target class. **Validity does not detect
this failure at all**; only the proximity magnitude does.

## 2. What actually failed

DiCoFlex's own conditional MAF (`gen_model`, trained in
`train_dicoflex_generator`, `counterfactuals/pipelines/run_dicoflex_traintest_pipeline.py:87`)
diverged to NaN in **7 of 15 cells**:

| Cell | NaN epochs | Outcome |
|---|---|---|
| lending-club / 42 | 101 (from epoch 0) | **run failed** |
| lending-club / 43 | 101 | "ok", checkpoint from before divergence |
| lending-club / 44 | 95 | "ok" |
| gmc / 43 | 98 | "ok" |
| gmc / 44 | 78 | "ok" |
| default / 43 | 83 | "ok" |
| adult / 44 | 42 | "ok" |

The cells marked "ok" are the dangerous ones: they logged success. The training
loop only checkpoints when `val_loss < best_val - eps`
(`run_dicoflex_traintest_pipeline.py:137`), so once the loss is NaN the
comparison is always False and the last good checkpoint is silently kept. Those
runs generated from a flow that had trained for a handful of epochs.

lending-club / 42 is the same bug taken to its limit: NaN from epoch 0 means no
checkpoint was ever written, so `model.load(model_path)` at
`run_dicoflex_traintest_pipeline.py:147` raised `FileNotFoundError` for a file
that was never created. The traceback names a missing file, which is misleading
— nothing tried to delete it, it was never saved.

## 3. Two distinct failure shapes

Worth separating, because they may have different causes:

- **NaN at epoch 0** (lending-club, all seeds). The first forward/backward pass
  already produces NaN. This points at the input data or the very first
  parameter draw, not at gradient accumulation.
- **NaN after healthy training** (adult / 44). Trains smoothly down to −189 over
  ~11 epochs, then diverges. This looks like a gradient explosion.

Even the healthy runs are unstable: adult / 42 oscillates
−169 → **+7441** → −99 → +8.45 → −138 between consecutive epochs and still
finishes. The flow training is fragile everywhere; the failures are where the
fragility tipped over.

## 4. Leading hypothesis: unbounded z-scored inputs

The aligned setup changed the model space from MinMax to StandardScaler
(`experiment.model_space_scaler`, added in commit `926b179`). Only the
continuous columns are scaled; one-hot columns stay 0/1. So the flow now fits a
density over a tensor mixing unbounded heavy-tailed columns with binary ones.

Measured on the training split, z-scored:

| Dataset | cont. cols | max abs z | 99.9th pct | NaN cells |
|---|---|---|---|---|
| lending-club | 8 of 31 | **77.7** | 4.95 | 3 of 3 |
| gmc | 7 of 42 | **73.7** | 8.21 | 2 of 3 |
| default | 14 of 91 | **69.8** | 9.15 | 1 of 3 |
| bank | 7 of 50 | 32.3 | 8.14 | 0 of 3 |
| adult | 4 of 66 | 13.1 | 13.11 | 1 of 3 |

The ordering of `max abs z` tracks NaN severity closely. lending-club is the
extreme case: a 99.9th percentile of 4.95 against a maximum of 77.7, i.e. a
handful of points ~78 standard deviations out. Under MinMax all of this
compresses into `[0, 1]`.

This is a hypothesis, not a conclusion. It has **not** been confirmed that the
NaN originates from those specific rows.

### 4a. Hypothesis CONFIRMED (2026-08-08)

Isolated the flow from the neighbour-search dataloader and trained the real
`large_maf` (the same 8-layer / 4-block / 16-hidden MAF, Adam, lr 1e-3, **no
gradient clipping** — exactly as `train_dicoflex_generator`) on real training
rows, sweeping only the scaler. Context and target were both drawn from the
scaled training split, so the flow sees the same value distribution the real run
feeds it; the neighbour pairing is the only thing omitted, and it is not what the
hypothesis is about.

| Dataset | scaler | max\|val on cont\| | outcome (60 steps) |
|---|---|---|---|
| adult | minmax | 1.00 | stable, loss 259 → 20 |
| adult | standard | 13.11 | stable, loss 259 → 20 |
| lending-club | minmax | 1.00 | stable, loss 120 → 11 |
| lending-club | standard | **77.75** | **non-finite gradient at step 44** |

The result is decisive: the *only* variable changed is MinMax vs StandardScaler,
and StandardScaler diverges on exactly the dataset with the heavy tail
(lending-club, which was 3/3 NaN in the real sweep) while MinMax trains it
cleanly. Adult (max\|z\| 13) survives 60 steps here, consistent with it being
only 1/3 NaN in the real run — its tail is mild enough that divergence is a
later-epoch gradient-explosion event, not an immediate one. Under MinMax every
input is in [0, 1] and the flow is stable on every dataset.

So the chain is established end to end:
StandardScaler leaves heavy-tailed continuous columns unbounded (max\|z\| up to
77.7) → the unclipped MAF's gradient explodes to non-finite → `val_loss` becomes
NaN → the `val_loss < best_val - eps` guard is never satisfied so a diverged /
never-written checkpoint is silently kept → sampling from that flow yields
counterfactuals tens-to-hundreds of sigma outside the data → proximity 1e9–1e33,
LOF `inf`, diversity huge; validity stays ≈1.0 because the classifier confidently
labels far-off points, so it never flags the failure.

Reproduce: `scratchpad/probe_nan.py` (feeds real scaled rows through the flow and
reports the first non-finite step per scaler).

## 5. What has not been checked yet

Ranked by how quickly they would settle it:

1. Feed the lending-club / 42 training batches through the flow once and find
   the first NaN — is it in the input, the log-det term, or the base log-prob?
   NaN at epoch 0 should make this immediate.
2. Check whether `create_dicoflex_dataloaders`
   (`counterfactuals/cf_methods/local_methods/dicoflex/data.py:385`) emits
   non-finite values, and whether the neighbour search degenerates when a few
   points sit ~78 sigma from everything else.
3. Re-run lending-club / 42 with `experiment.model_space_scaler=minmax`,
   everything else identical. If it trains cleanly, the space is confirmed as
   the cause.
4. Test whether gradient clipping plus a lower learning rate alone rescues it.
   `train_dicoflex_generator` has neither; `cfg.gen_model.lr` is 1e-3
   (`counterfactuals/pipelines/conf/dictum_dicoflex_config.yaml`).

## 6. Fixes worth considering, independent of the diagnosis

- **Do not silently succeed on a diverged run.** If `val_loss` is NaN for a
  whole epoch, or no checkpoint was ever written, the run should fail loudly.
  Six cells reported "ok" on a flow that had effectively stopped training.
- Clip gradients in `train_dicoflex_generator`; there is currently no
  `clip_grad_norm_` anywhere in that loop.
- Generate in MinMax and convert to the z-scored ordinal space only for
  metrics. Already supported: `--generation-scaler minmax --scaler standard
  --metric-encoding ordinal`. This sidesteps the divergence rather than fixing
  it, and is the current plan of record.

---

## 7. Related, and probably more serious: counterfactuals are not valid one-hot

Found while checking why `spars_cat` moved between the one-hot and ordinal
metric spaces. This affects **DiCE and CCHVAE too**, and is unrelated to
scaling.

Adult, seed 42, first 5000 counterfactual rows, fraction of one-hot blocks by
block sum:

| | sum = 0 | sum = 1 (valid) | sum ≥ 2 | rows with every block valid |
|---|---|---|---|---|
| DiCE | 0.016 | 0.856 | 0.129 | **0.271** |
| CCHVAE | 0.073 | 0.876 | 0.051 | **0.314** |
| factuals | 0.000 | 1.000 | 0.000 | 1.000 |

The values themselves are exactly 0 or 1 — this is not a rounding artefact. The
methods are setting several categories of one feature at once (13% of DiCE's
blocks), or none at all. Only **27% of DiCE counterfactuals and 31% of CCHVAE's
are valid data points**. The factuals are 100% valid, so this is produced by the
generators, not inherited from the data.

Consequences, all of which touch the current tables:

- The classifier is being asked to score inputs that cannot occur in its
  training distribution, so every reported validity is measured off-manifold.
- `spars_cat` differs between metric encodings purely because of this. In the
  one-hot space an invalid block counts as changed; in the ordinal space
  `argmax` maps an all-zero block to code 0, silently colliding with the genuine
  first category. That is why Adult DiCE reads 0.139 one-hot and 0.080 ordinal.
  **Neither number is trustworthy while the blocks are invalid.**
- LOF and diversity in the ordinal space inherit the same `argmax` repair.

`DiCoFlex` alone runs its output through `apply_categorical_discretization`
(`run_dicoflex_traintest_pipeline.py:380`). DiCE and CCHVAE do not. Projecting
every method's counterfactuals back onto valid one-hot blocks before scoring —
and ideally before the validity check — looks like a prerequisite for any table
being publishable, whichever scaling is chosen.

### 7a. CCHVAE proves the point: its "perfect validity" was never real (2026-08-08)

CCHVAE reports validity 1.000 on every dataset/seed in the raw scorer and in its
own native pipeline. In the snap-first `_fixed` scorer that validity collapses:
0.87 / 0.87 / 0.63 / 0.96 / 0.86 across the five datasets, with Default/seed-43
as low as **0.578**. CCHVAE is a good diagnostic precisely because it is supposed
to be perfectly valid, so the drop cannot be dismissed as a weak method.

Analysed directly on CCHVAE's counterfactuals (`scratchpad/probe_cchvae.py`,
367 400 Adult rows):

| quantity | value |
|---|---|
| rows with every one-hot block valid | **0.310** |
| one-hot blocks that are valid (sum≈1) | 0.878 |
| blocks all-zero (sum≈0) | 0.073 |
| blocks multi-hot (sum≥2) | 0.049 |
| per-block argmax gap (top1−top2), median | 1.000 |
| rows moved by snapping (cat L1 > 0.5) | **0.690** |

The mechanism is now unambiguous:

1. CCHVAE decodes each one-hot column independently, so per **block** it is right
   ~88% of the time but sets two categories at once (5%) or none (7%) otherwise.
2. With 8 blocks per Adult row the per-row validity compounds:
   `0.878**8 ≈ 0.31`, which is exactly the measured fraction of fully valid rows.
3. The classifier was trained on exact 0/1 one-hot data, yet in the raw scorer it
   is asked to judge these near-one-hot / multi-hot vectors. It confidently
   returns the target class for them → validity 1.000. **This number describes
   points that are not valid rows of the dataset.**
4. Force each counterfactual to be a real single-category point (`argmax` snap)
   and 4–40% flip class. That is the honest validity, and it is what `_fixed`
   reports.

So the answer to "why does even CCHVAE look bad now" is: it does not look bad —
it is finally being scored on counterfactuals that could actually be shown to a
user, and it turns out only ~60–96% of them survive as valid data points. The
1.000 was an artefact of scoring off-manifold vectors, shared by DiCE and by the
whole baseline column of every earlier table.

This reframes the DiCoFlex story of §1–§6 as the *smaller* of the two problems.
DiCoFlex is the only method that snapped at generation, so in the raw scorer it
was already being held to a stricter (honest) standard than the baselines it is
compared against. Any fair table has to snap every method's output — and check
validity on the snapped point — before a single number is read off.

Caveat on the snap itself: 7% of blocks are all-zero, where `argmax` picks an
essentially arbitrary category. For those blocks the snapped validity is partly a
coin flip rather than a property of CCHVAE, so the exact `_fixed` numbers should
be read as "≈, and no better than the repair rule", not as ground truth. The
conclusion — CCHVAE is not truly 100% valid at the level of real data points —
does not depend on the repair rule.

### 7b. The code-level root cause: CCHVAE's own categorical repair is wrong

The invalid one-hot blocks are not an accident of the metric scorer; they are
baked into CCHVAE's search. Every decoded candidate is passed through
`reconstruct_encoding_constraints`
(`counterfactuals/cf_methods/local_methods/c_chvae/utils.py:104`), which is meant
to force categorical blocks back to valid one-hot before the label check. With
`binary_cat=True` — the value wired in `dictum_cchvae_config.yaml:69` — its entire
body is:

```python
for pos in feature_pos:
    x_enc[:, pos] = torch.round(x_enc[:, pos])
```

`feature_pos` is `dataset.categorical_features_indices`, i.e. the flat list of
**all** one-hot columns. So it rounds each column independently to 0/1 and never
enforces "exactly one 1 per block". A decoded block therefore lands:

- all-zero when every column rounds below 0.5 (measured 7.3% of blocks),
- multi-hot when two columns round to 1 (4.9%),
- accidentally valid otherwise (87.8%).

This is the direct cause of §7 / §7a. The `binary_cat=False` branch is no better
— it assumes categoricals come in consecutive drop-if-binary *pairs* and would
corrupt any multi-category feature. Neither branch can produce valid
multi-category one-hot; the function was written for CARLA / Pawelczyk-style
binary (drop_first) encodings, and our datasets use `drop_first: false` full
one-hot (`config/datasets/adult_split.yaml`), so the assumption is simply false
here.

Why validity still reads 1.000: the label check inside the search
(`c_chvae.py`, `y_candidate = predict_proba(x_ce)`) runs on this *same* rounded
`x_ce`. CCHVAE only accepts a candidate whose rounded form flips the class, so it
is 100% valid *on its own malformed output by construction*. The native
`calculate_metrics` and the raw aligned scorer both re-check on that same
malformed vector, so both echo 1.000. Only a repair that actually collapses each
block to one category (the `_fixed` argmax snap) moves the point and exposes the
true ~0.58–0.96.

Blast radius: this fires on every dataset and both the native CCHVAE metric CSVs
and the aligned table. DiCE has the analogous independent-column problem from a
different route. `DiCoFlex` is the only method whose generation calls a real
per-block discretiser (`run_dicoflex_traintest_pipeline.py:380`), so it alone was
already reporting honest validity — and was being compared against baselines
whose validity was inflated by this bug.

Minimal fix: replace the `binary_cat=True` body with a per-block argmax over
`categorical_features_lists` (the same operation as
`apply_categorical_discretization`), so the reconstruction the label check sees is
a genuine one-hot row. Then the raw and `_fixed` scorers agree and no post-hoc
snap is needed.
