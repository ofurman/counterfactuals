# GOAL: find the code version where DiCoFlex worked, and diff it against ours

Status: **OPEN.** Opened 2026-08-09.

The paper's Table 1 (`_NIPS2026__TabDCE/results.tex`, `tab:protocol_B`) reports a
DiCoFlex baseline with **bounded, sane** numbers — e.g. Bank Prox.-Cont ≈ 0.89,
LOF ≈ 0.16, validity 1.00. So DiCoFlex *did* work at some point, in some version
of the code. Our current `dictum-aligned-eval` implementation instead produces
**numeric proximity of order 1e18–1e32 with LOF = inf** on every dataset (see
`GOAL_dicoflex_divergence.md`). The counterfactuals' numeric columns literally
contain `inf`.

This file's goal: **identify the exact code version that produced correct DiCoFlex
counterfactuals, enumerate every difference from the current implementation, and
port the missing pieces back** so the aligned sweep reproduces the paper's
DiCoFlex row.

---

## 1. Candidate "working" versions (branches / repos)

| source | path/layout | notes |
|---|---|---|
| **`origin/dicoflex`** | `counterfactuals/cf_methods/dicoflex/{dicoflex,generation,training,dataset,utils}.py` + `counterfactuals/generative_models/maf.py` | **Leading candidate.** A faithful port of the original: temperature-aware MAF, `n_samples`/`temperature` defaults matching the reference. Diff THIS first. |
| `origin/ofurman/dicoflex` | `counterfactuals/cf_methods/local/DiCoFlex/method.py` | The author's own layout; cross-check against `origin/dicoflex`. |
| `ofurman/DiCoFlex` (GitHub) | `counterfactuals/dicoflex/`, `datasets/DCENF/*.py` | The external original. Uses MinMax(numeric) + **QuantileTransformer**(one-hot cat) + `temp=0.8`, `n_samples=10`. Already partly mirrored: see `minmax_qt` scaler added this session. |
| `DICTUM/` (in-repo) | `DICTUM/src/tabdce/` | NOT DiCoFlex — this is TabDCE/DICTUM, a **conditional diffusion** model (our method). Different architecture; keep separate. |
| **ours (broken)** | `counterfactuals/cf_methods/local_methods/dicoflex/{method,data}.py` on `dictum-aligned-eval` | The re-implementation under investigation. |

## 2. Confirmed differences (working `origin/dicoflex` vs ours)

All verified by `git grep` on 2026-08-09.

| aspect | `origin/dicoflex` (working) | ours (`dictum-aligned-eval`) | file evidence |
|---|---|---|---|
| **temperature** | base noise scaled by `temp`: `torch.randn(...) * temp`; `sample_and_log_prob(..., temp=temperature)`; `temperature=0.8` | **no temperature at all** — `sample_and_log_proba(n_samples, context)` samples the base at full variance (1.0) | working: `generative_models/maf.py:26,29,75,90`; `cf_methods/dicoflex/generation.py:12,65`. ours: `models/generative/maf/maf.py:144` (no temp), `cf_methods/local_methods/dicoflex/method.py:174` |
| **samples/factual** | `n_samples=10` | `num_counterfactuals=100` | working: `generation.py:11`. ours: `conf/dictum_dicoflex_config.yaml:65` |
| **candidate selection** | none in generation; returns all 10, metric aggregates | **added `_select_topk_candidates` ranking by classifier confidence** (deepest-in-target = farthest) | ours: `method.py:271` |
| **numeric scaler** | **MinMaxScaler** | was StandardScaler; now `minmax_qt` (this session) | working: `dataset.py:400`, `utils.py:24`. ours: `conf/dictum_dicoflex_config.yaml` `model_space_scaler` |
| **categorical handling** | MinMax + Gaussian noise `N(0,0.08)` on cat cols (this branch); the GitHub repo uses a QuantileTransformer instead | one-hot only (now QT via `minmax_qt`) | working: `dataset.py:200`; GitHub: `datasets/DCENF/*.py` |
| **clamp/clip** | **none** — relies on `temp=0.8` (+ short training) to keep samples bounded | none | `git grep clamp/clip` empty on both |
| **training epochs** | ~10 (reference) | 200 (patience 100) → reaches gradient explosion | `conf/dictum_dicoflex_config.yaml` gen_model.epochs |
| **MAF architecture** | reference CF flow hidden 64 / 5 layers / 2 blocks | `large_maf` 16 / 8 / 4 | — |

## 3. The single most likely cause

**Temperature.** The working MAF scales the base Gaussian by `temp` (`_sample`:
`torch.randn(...) * temp`) and samples at `temp=0.8`. Our current MAF has **no
temperature parameter at all** — it samples at full variance. The failure is an
unbounded numeric tail (values reach `inf`), and the working version has **no
clamp**, so temperature is the only thing keeping its samples bounded. This is
the mechanism we dropped.

Corroborating experiment (this session, `minmax_qt`, Bank, 40 epochs):

| flow | numeric maxabs | internal prox |
|---|---|---|
| large_maf (16/8/4) | inf | 7e20 |
| small_maf (16/2/2) | inf | 5e18 |

Flow size does not matter — every architecture explodes because the base is
sampled at variance 1.0 with no bound. The `minmax_qt` change fixed the
**categorical** columns (maxabs 1.22, clean one-hot) but not the numeric ones.

## 4. Plan (ranked)

1. **Port temperature.** Add a `temp` argument to
   `counterfactuals/models/generative/maf/maf.py::sample_and_log_proba` that
   scales the base draw (mirror `origin/dicoflex:generative_models/maf.py:26-90`),
   and pass `temp=0.8` from `method.py::_sample_counterfactuals` (add it to
   `DiCoFlexParams` / config). This is the faithful fix.
2. **Match sample count / selection.** Set `num_counterfactuals=10` (or keep 100
   but select the *closest*), and remove/replace the confidence-based
   `_select_topk_candidates` so far-from-factual samples are not preferred.
3. **Numeric clamp as a hard backstop** (optional, not in the original): clip
   numeric columns to `[0,1]` in the MinMax gen space before inverse-transform,
   guaranteeing bounded proximity even if a temp-0.8 draw is still extreme.
4. **Match epochs / architecture** to the reference (≈10 epochs; hidden 64 / 5
   layers / 2 blocks) if 1–3 do not fully close the gap.

## 5. Verification

Re-run one fast cell (Bank, seed 42) after each step and score with
`scripts/compute_dictum_metrics --generation-scaler minmax_qt --scaler standard
--metric-encoding ordinal`. Success = numeric proximity O(1) (target ≈ 0.89),
LOF finite (≈0.16), no `inf`. Confirm on a second dataset before trusting it.

## 6. What is already done (this session)

- `minmax_qt` scaler added (`preprocessing/scalers.py::QuantileTransformCategoricalStep`,
  `preprocessing/factory.py`), wired into `dictum_dicoflex_config.yaml`. Fixes the
  categorical space; does **not** fix the numeric tail.
- Confirmed the numeric explosion is independent of flow size (§3).
- Identified `origin/dicoflex` as the working reference to diff against (§1–§2).

The critical path is step 1 (temperature); everything else is secondary.
