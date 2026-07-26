# Commands

Every command is copy-pasteable. `<placeholders>` must be resolved from live output, never
guessed.

## Credentials

`.env` is gitignored, mode 600, and holds `PLG_LOGIN`, `PLG_PASSWORD`, `PLG_GRANT`.
Per `plgrid-run/SKILL.md` rule 1, only `PLG_LOGIN` may be used and nothing from `.env` may be
printed, committed, or copied to PLGrid.

```bash
set -a; . ./.env; set +a          # exports PLG_LOGIN — never echo PLG_PASSWORD
export PLG_HOST=login01.helios.cyfronet.pl
ssh -o BatchMode=yes "$PLG_LOGIN@$PLG_HOST" true    # fails fast if no key/agent
```

## Local (repo root)

```bash
# Stage 1 spot check — cheapest cell
./run_seed_experiments.sh --methods dicoflex --datasets default --seeds 0 --tag stage1-check
rm -rf results/stage1-check

# Quality gates
uv run ruff check counterfactuals/ scripts/ tests/
uv run ruff format --check counterfactuals/ scripts/ tests/
uv run pytest tests/ -q

# Stage 2 sync
PLG_LOGIN="$PLG_LOGIN" ./slurm/sync-code.sh
```

## On the Helios login node

```bash
cd ~/projects/counterfactuals

# Stage 2 — report only. NEVER --write: it overwrites the live-verified cluster.env.
./slurm/preflight.sh
echo "$PLG_GROUPS_STORAGE"        # capture the real path for the Stage 6 rsync
./slurm/bootstrap-storage.sh

sbatch --account=plgcountercontex-gpu-gh200 --partition=plgrid-gpu-gh200 \
       --qos=now slurm/setup-env.sbatch

# Stage 3 — calibrate before spending the grant
./slurm/submit-all.sh --smoke              # dry run
./slurm/submit-all.sh --smoke --submit

# Stage 4 — exactly 60 tasks; --only skips the constraints array
./slurm/submit-all.sh --only dice          # dry run each shape first
./slurm/submit-all.sh --only cchvae
./slurm/submit-all.sh --only dicoflex
./slurm/submit-all.sh --submit --only dice        # 15
./slurm/submit-all.sh --submit --only cchvae      # 15
./slurm/submit-all.sh --submit --only dicoflex    # 15 + 15 tc0
```

Bare `./slurm/submit-all.sh --submit` also queues 5 constraint tasks — out of scope.

## Monitoring

```bash
squeue -u "$USER" -o '%i %j %t %M %L'
cat slurm/logs/submitted.tsv

sacct -X --format=JobID,JobName%20,State,Elapsed,ExitCode --starttime today
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode,AllocTRES%40

# Task logs
tail -50 slurm/logs/cf-<method>-<arrayjobid>_<taskid>.err
```

Poll at 30-minute intervals or wider. Never tighter than 5 minutes.

## Result verification (on the cluster)

```bash
RES="$PLG_GROUPS_STORAGE/plggcfsgenwro/$USER/counterfactuals/results"
find "$RES/seeds" "$RES/seeds-tc0" -name 'cf_metrics_*.csv' -size +0 | wc -l   # expect 60
head -1 "$(find "$RES/seeds" -name 'cf_metrics_*.csv' | head -1)"              # expect cf_model_train_time
find "$RES" -name '*_run.json' -exec grep -h git_commit {} \; | sort -u        # expect one SHA
```

## Retrieval and aggregation (local)

```bash
rsync -av "$PLG_LOGIN@$PLG_HOST:<PLG_GROUPS_STORAGE>/plggcfsgenwro/$PLG_LOGIN/counterfactuals/results/" \
  ./results/

uv run python scripts/aggregate_seed_results.py \
  --results-root results --tags seeds seeds-tc0 \
  --out-csv results/seed_aggregate.csv \
  --out-markdown results/seed_aggregate.md
```

## Result layout

```
results/<tag>/seed_{0,1,2}/<dataset>_split/<Method>/fold_0/cf_metrics_SimpleMLPClassifier.csv
results/<tag>/seed_{0,1,2}/<dataset>_split/fold_0/disc_model_SimpleMLPClassifier.pt
results/<tag>/seed_{0,1,2}/<stem>_<method>_run.json
```

- `<tag>`: `seeds`, `seeds-tc0`, `smoke`
- `<dataset>_split`: `adult_split`, `bank_split`, `default_split`, `gmc_split`, `lending_club_split`
  (note `lending-club` the directory vs `lending_club_split` the config stem)
- `<Method>`: `DiceExplainerWrapper`, `CCHVAE`, `DiCoFlex`
- Array index mapping: `dataset_idx = id / 3`, `seed_idx = id % 3`, datasets ordered
  `adult bank default gmc lending-club`
- Model checkpoints sit one level **above** the method directory and are therefore **shared**
  across methods at the same (dataset, seed). Harmless while `train_model=true` retrains under
  a fixed seed; it would cross-contaminate if anyone ever sets `train_model=false`.
