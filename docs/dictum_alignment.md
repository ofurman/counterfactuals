# DICTUM-aligned evaluation setup

This page documents the experimental setup added so that results from this
repository and from the [DICTUM](https://github.com/JanMaslowski/DICTUM)
codebase can be placed in the same table. It records what the two setups did
differently, which choice was adopted, and how to run and score the aligned
experiments.

The aligned setup lives alongside the existing one rather than replacing it.
The original `dice_traintest_config.yaml` / `cchvae_traintest_config.yaml` /
`dicoflex_traintest_config.yaml` runs are unchanged, so previously generated
results stay reproducible.

## Where the two setups differed

| Aspect | DICTUM | This repository (before) | Aligned setup |
|---|---|---|---|
| Continuous scaling | `StandardScaler` | `MinMaxScaler(0, 1)` | `StandardScaler` |
| Categorical encoding in model space | Ordinal codes | One-hot | One-hot (unchanged) |
| Classifier | MLP `[32, 32]`, dropout 0.2 | `SimpleMLP [64, 64]`, dropout 0.2 | MLP `[32, 32]`, dropout 0.2 |
| Classifier training | 100 epochs, batch 64, patience 15 | 5000 epochs, batch 128, patience 300 | 100 epochs, batch 64, patience 15 |
| Early stopping split | `val.csv`, else 80/20 of train | the test split | `val.csv`, else 80/20 of train |
| Classifier per method | one per (dataset, seed), reloaded | each method trained its own | one per (dataset, seed), reloaded |
| Explained instances | 2000 sampled test rows | every eligible test row | 2000 sampled test rows |
| Flip direction | both, per-instance flip | one class per run, and not the same class for every method | both, per-instance flip |
| Candidates per factual | 100 | 100 | 100 |
| Kept for metrics | 10, valid-first | 10, valid-first | 10, valid-first |
| Actionability during generation | unrestricted in the main table | DiCE restricted to actionable features | unrestricted |
| Seeds | 42, 43, 44 | 0, 1, 2 | 42, 43, 44 |

The direction mismatch was the most consequential: DiCE and CCHVAE ran at
`target_class: 0` while DiCoFlex ran at `1`, so the three methods were scored on
disjoint sets of query instances. `target_class: null` explains every selected
row towards the flip of its own predicted label, which covers both directions
and gives all methods the same query set.

## Metric formula differences

Both codebases report columns named validity, proximity, sparsity, ε-sparsity,
LOF and diversity, but four of them are computed differently. The aligned
scorer, `scripts/compute_dictum_metrics.py`, implements DICTUM's definitions:

- **ε-sparsity** counts continuous features moved by more than `0.05 × train_range`
  in original units. The in-house version thresholds the *relative* change
  `|Δx| / (|x| + 1e-8)` instead.
- **Categorical sparsity** is the fraction of one-hot *groups* changed (8 on
  Adult). The in-house version averages over one-hot *columns* (62 on Adult).
- **Diversity** uses cityblock distance on the continuous block and Hamming over
  group-collapsed categorical codes; the in-house version uses Euclidean and
  raw one-hot columns.
- **Aggregation** is per factual first, then averaged over factuals, with LOF
  taking the median within a factual's counterfactual set. The in-house version
  pools every kept counterfactual into one mean.

Proximity and LOF share their formula across the two, but not their units: both
are computed in the model space, so they are only comparable once the scaling
matches, which is what `model_space_scaler: standard` provides.

Because these definitions disagree, numbers produced by
`scripts/compute_actionability_metrics.py` and by
`scripts/compute_dictum_metrics.py` must never be mixed in one table.

## Checking the setup first

`./run_dictum_smoke.sh` runs the single cheapest cell — DiCE on lending-club at
seed 42 — and scores it, which exercises the whole path (shared classifier, both
flip directions, CSV layout, scorer) before the full sweep is worth starting.

```bash
./run_dictum_smoke.sh            # full fidelity, real numbers, ~2.5 min
./run_dictum_smoke.sh --quick    # shrunken epochs, plumbing check only, ~20 s
./run_dictum_smoke.sh --method cchvae --dataset bank --seed 43
./run_dictum_smoke.sh --keep     # leave results/<tag> in place
```

lending-club is the cheapest dataset (31 one-hot columns against 42-91 for the
others) and DiCE the cheapest method: it generates in one batched call, while
CCHVAE loops its sampler 100 times per factual and DiCoFlex trains a flow.
Measured on an M-series laptop, the full-fidelity cell takes about 150 s, of
which roughly 47 s is the MAF, 5 s the classifier, and the rest CF search.

Numbers from `--quick` are not meaningful — it trains for a handful of epochs
and exists only to answer "does this run".

## Running the aligned experiments

```bash
# preview the commands without running anything
./run_dictum_experiments.sh --dry-run

# full sweep: 3 methods x 5 datasets x 3 seeds
./run_dictum_experiments.sh

# a single cell
./run_dictum_experiments.sh --methods dice --datasets adult --seeds 42

# extra hydra overrides, applied to the classifier pretrain too
./run_dictum_experiments.sh --override gen_model.epochs=500
```

The runner first trains one classifier per (dataset, seed) via
`scripts/train_shared_disc_model.py`, then runs each method with
`disc_model.train_model=false` so every method explains that exact model. An
existing classifier checkpoint is reused rather than retrained, so an
interrupted sweep can be resumed by re-running with a narrower selection.

Results land in:

```
results/<tag>/seed_<N>/<dataset>_split/<Method>/fold_0/
    counterfactuals_<Method>_SimpleMLPClassifier.csv   # N*100 rows, factual-major
    factuals_<Method>_SimpleMLPClassifier.csv          # N rows, aligned row-for-row
    cf_metrics_<...>.csv                               # the in-pipeline metric registry
```

## Getting notified as cells finish

`scripts/notify_experiments.sh` is a background shell loop that watches the
per-cell `.status` files the runner writes and sends a notification for each
one, plus a summary when the sweep process exits. It can be started at any
time, including after the sweep is already running; cells that finished
beforehand are recorded but not re-announced.

### Email

Put the SMTP settings in a file rather than on the command line, so the
password stays out of shell history and out of `ps`:

```bash
cat > ~/.config/cf-notify.env <<'EOF'
NOTIFY_EMAIL_TO=you@gmail.com
SMTP_USER=you@gmail.com
SMTP_PASS=your-16-char-app-password
EOF
chmod 600 ~/.config/cf-notify.env

set -a; . ~/.config/cf-notify.env; set +a
./scripts/notify_experiments.sh --tag dictum --final-only > notify.log 2>&1 &
```

Defaults target Gmail (`smtp.gmail.com:465`, implicit TLS). Gmail requires an
**App Password**, which in turn requires 2-Step Verification on the account — a
normal account password is rejected. Other providers work through `SMTP_HOST`,
`SMTP_PORT` and `SMTP_SSL=0` for STARTTLS.

`--final-only` is the sensible default for email: failures still arrive
immediately, but the 45 successful cells collapse into one closing summary
instead of 45 messages.

Send one by hand to check the settings before relying on it:

```bash
set -a; . ~/.config/cf-notify.env; set +a
uv run python -m scripts.send_email_notification "test" "from the sweep host"
```

### Other backends

```bash
# phone push via ntfy.sh, no account needed
NTFY_TOPIC=some-long-unguessable-name \
  ./scripts/notify_experiments.sh --tag dictum > notify.log 2>&1 &

# Slack-style incoming webhook
NOTIFY_WEBHOOK=https://hooks.slack.com/services/... \
  ./scripts/notify_experiments.sh --tag dictum > notify.log 2>&1 &
```

Backends are independent and all optional; each enabled one receives every
notification, and with none configured it still logs to stdout.

An ntfy.sh topic is a **public channel** — anyone who knows or guesses the name
can read it. Use a long random topic name, keep sensitive detail out of run
labels, or point `NTFY_SERVER` at your own instance.

## Scoring

```bash
uv run python -m scripts.compute_dictum_metrics \
    --results-root results/dictum \
    --datasets adult bank default gmc lending-club \
    --seeds 42 43 44 \
    --output results/dictum/dictum_metrics
```

This writes `.per_seed.csv` (one row per dataset/method/seed), `.csv`
(mean/std/count per cell), `.md` and `.tex`. Cells backed by fewer than the
requested number of seeds are logged as a warning, so a partially failed sweep
cannot quietly look complete.

## Configuration knobs introduced

These are additive; every one defaults to the previous behaviour, so configs
that do not set them are unaffected.

| Key | Default | Effect |
|---|---|---|
| `experiment.model_space_scaler` | `minmax` | `standard` z-scores continuous features instead of min-max scaling them. |
| `disc_model.validation_source` | `test` | `val` early-stops on `val.csv`, falling back to a seeded 80/20 split of train when the dataset ships none. |
| `dataset.val_data_path` | unset | Path to a validation CSV. A path that does not exist is treated as absent. |
| `counterfactuals_params.target_class` | per config | `null` explains both flip directions, targeting the opposite of each factual's own label. |
| `counterfactuals_params.n_test_samples` | unset | Caps how many test rows are explained, sampled with a seed-local generator. |
| `counterfactuals_params.restrict_to_actionable` | `true` | DiCE only; `false` lifts the `features_to_vary` restriction for unconstrained benchmarks. |

## Known caveats

- CCHVAE's `clamp` option clips its latent search back into `[0, 1]`, which
  assumes a min-max model space. The aligned config disables it; validity in
  z-scored space should be checked against the min-max runs before the numbers
  are used.
- Timings from these runs are only comparable within one machine and one
  concurrency setting. Do not compare a `--jobs 2` local run against a cluster
  run.
- The aligned setup covers the unconstrained benchmark. The constraint
  experiments (`scripts/run_constraint_setup_experiments.py`) still use the
  original configs.
