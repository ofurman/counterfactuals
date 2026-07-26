# Helios experiment runner

Slurm assets for the NeurIPS revision experiments: seed-to-seed variance,
baseline timing, and the constraint sweep on a second dataset.

## Why these jobs are CPU-only, on a GPU partition

All three baseline pipelines set `CUDA_VISIBLE_DEVICES = "-1"` at the top of
`main()`:

- `counterfactuals/pipelines/run_dice_traintest_pipeline.py:345`
- `counterfactuals/pipelines/run_cchvae_traintest_pipeline.py:261`
- `counterfactuals/pipelines/run_dicoflex_traintest_pipeline.py:513`

DiCoFlex line 514 then evaluates `torch.cuda.is_available()`, which is already
`False` by that point, so its `use_gpu` flag can never take effect. The work is
pure CPU.

Grant `plgcountercontex` nevertheless holds **only** a gpu-gh200 allocation.
Verified on 2026-07-26 with `sbatch --test-only`:

| Partition | Result for account `plgcountercontex-gpu-gh200` |
|---|---|
| `plgrid` | Invalid account or account/partition combination |
| `plgrid-long` | Invalid account or account/partition combination |
| `cpu` | Invalid account or account/partition combination |
| `plgrid-gpu-gh200` | accepted |

So CPU-only work has to run on the GPU partition. The jobs are submitted
**without `--gres`**, which allocates cores on a GH200 node without reserving
a GPU — the least wasteful option available. Whether PLGrid still bills these
as GPU-hours against the 10 000 h allocation is worth confirming with
`sacct --format=JobID,AllocTRES,CPUTimeRAW` after the smoke run.

Getting a CPU allocation added to the grant would be the real fix.

TabDCE itself does use the GPU, but it is not run by these scripts.

## Order of operations

```bash
# 1. locally, from the repo root
PLG_LOGIN=plgyourlogin ./slurm/sync-code.sh

# 2. on the Helios login node
cd ~/projects/counterfactuals
./slurm/preflight.sh --write     # then edit slurm/cluster.env
./slurm/bootstrap-storage.sh
sbatch --account=... --partition=... \
       --mail-user=... --mail-type=END,FAIL slurm/setup-env.sbatch

# 3. calibrate before spending the grant
./slurm/submit-all.sh --smoke              # dry run
./slurm/submit-all.sh --smoke --submit
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode

# 4. full set
./slurm/submit-all.sh                      # dry run of every array
./slurm/submit-all.sh --submit
```

## What gets submitted

| Array | Tasks | Purpose |
|---|---|---|
| `cf-dice` | 15 | 5 datasets x 3 seeds |
| `cf-cchvae` | 15 | 5 datasets x 3 seeds |
| `cf-dicoflex` | 15 | 5 datasets x 3 seeds |
| `cf-dicoflex-tc0` | 15 | DiCoFlex re-run at `target_class=0` |
| `cf-constraints` | 5 | constraint setups 1-5 on adult |

65 tasks, capped at `MAX_CONCURRENT=8` concurrent per array. Array index maps
as `dataset_idx = id / n_seeds`, `seed_idx = id % n_seeds`, with datasets
ordered `adult bank default gmc lending-club`.

## The DiCoFlex target_class re-run

DiCoFlex runs at `target_class: 1` (`dicoflex_traintest_config.yaml:48`) while
DiCE, CCHVAE and TabDCE run at `0` — see `DEFAULT_TARGETS` in
`scripts/run_constraint_setup_experiments.py:110`. The saved factual counts
confirm the two sets are exactly complementary (adult 3674 vs 6326, bank 2746
vs 7254, and so on).

So DiCoFlex's row in Table 1 is computed on a **disjoint set of queries going
the opposite direction** from every other row. That blocks any paired
significance test involving DiCoFlex, and it is a second instance of the
asymmetry the reviewer already flags in weakness #1. The `cf-dicoflex-tc0`
array exists to fix it; DiCoFlex is the cheapest method, so this costs little.

Decide deliberately whether the paper reports the `tc0` numbers or keeps the
current ones — do not mix them silently.

## Job alerts

`PLG_MAIL` and `PLG_MAIL_TYPE` in `cluster.env` drive `--mail-user` /
`--mail-type`, applied by `submit-all.sh` to every array. Leave `PLG_MAIL`
empty to switch alerts off.

Volume is low by design. Slurm sends one message per **array**, not per array
task, unless `ARRAY_TASKS` is added to the type list — so `BEGIN,END,FAIL`
across the five arrays is roughly ten messages, not several hundred. Do not
add `ARRAY_TASKS` unless you want one mail per dataset-seed pair.

Whether the mail actually arrives is site-dependent: many HPC sites disable
outbound mail from Slurm. `preflight.sh` now reports `MailProg`/`MailDomain`
from `scontrol show config`; if `MailProg` is missing or points at
`/bin/true`, the directives are accepted and silently dropped. In that case
poll with `sacct` instead:

```bash
watch -n 300 "sacct -X --format=JobID,JobName%20,State,Elapsed,ExitCode"
```

## Walltimes are estimates

`WALLTIME` in `submit-all.sh` is extrapolated from local `cf_search_time`
values (CCHVAE on adult: 11586 s of CF search alone) plus headroom for
retraining the discriminative and generative models, which has **never been
measured on this hardware**. The generative flow trains for up to 2000 epochs
on CPU and could dominate. Run the smoke job and read `sacct` Elapsed before
trusting these numbers. Check the partition `MaxTime` from `sinfo` allows the
24 h CCHVAE request.

## Timing comparability

`CPUS_PER_TASK` is set once in `cluster.env` and applied to every method, and
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` are pinned to it inside the job. This is
deliberate: the revision adds a Time column comparing methods, and BLAS thread
heuristics that vary by node would make that column meaningless. Do not vary
`--cpus-per-task` between methods.

Each run now records `disc_train_time`, `gen_train_time` and `seed` alongside
the existing `cf_search_time` in `cf_metrics_*.csv`.

## Output layout

```
$PLG_GROUPS_STORAGE/<group>/<user>/counterfactuals/results/
├── seeds/seed_{0,1,2}/<dataset>_split/<Method>/fold_0/cf_metrics_*.csv
├── seeds-tc0/seed_{0,1,2}/...
└── constraints/<dataset>_setup<N>/...
```

This matches what `scripts/calculate_metrics.py` expects when pointed at a
single seed root, so `--num-folds 1` works per seed. Aggregating the three
seed roots into the `mean ± std` cells that
`scripts/generate_latex_tables.py` already parses still needs a small
aggregator script — **not yet written**.

## Retrieving results

```bash
rsync -av \
  plgyourlogin@login01.helios.cyfronet.pl:'$PLG_GROUPS_STORAGE/<group>/<user>/counterfactuals/results/' \
  ./results/
```

A submitted job ID is not evidence of success. Check `sacct` State is
`COMPLETED`, `ExitCode` is `0:0`, and that each expected `cf_metrics_*.csv`
exists and is non-empty.
