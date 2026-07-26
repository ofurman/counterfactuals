#!/usr/bin/env bash
# Validate or submit the full experiment set on Helios.
#
# Default is a dry run: nothing consumes allocation until you pass --submit.
#
#   ./slurm/submit-all.sh                 # sbatch --test-only for every array
#   ./slurm/submit-all.sh --smoke         # dry-run the 1-task calibration job
#   ./slurm/submit-all.sh --smoke --submit
#   ./slurm/submit-all.sh --submit        # the real thing
#   ./slurm/submit-all.sh --submit --only cchvae
#
# Run the smoke job FIRST. The walltimes below are extrapolated from local
# cf_search_time measurements plus headroom for retraining, and the retraining
# half has never been measured on this hardware.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/cluster.env"

: "${PLG_ACCOUNT:?Set PLG_ACCOUNT in slurm/cluster.env}"
: "${PLG_PARTITION:?Set PLG_PARTITION in slurm/cluster.env}"
: "${CPUS_PER_TASK:?Set CPUS_PER_TASK in slurm/cluster.env}"

if [[ "$PLG_ACCOUNT" == CHANGE-ME* || "$PLG_PARTITION" == CHANGE-ME* ]]; then
  echo "slurm/cluster.env still holds CHANGE-ME values; run preflight.sh and edit it" >&2
  exit 1
fi

SUBMIT=0
SMOKE=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1 ;;
    --smoke) SMOKE=1 ;;
    --only) ONLY="$2"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$REPO_ROOT/slurm/logs"
cd "$REPO_ROOT"

# Concurrency cap per array. Raise if the queue is empty and the grant allows.
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

# Walltime per method. CCHVAE dominates: 3.2 h of CF search on adult locally,
# before any model retraining.
declare -A WALLTIME=(
  [dice]="08:00:00"
  [cchvae]="24:00:00"
  [dicoflex]="12:00:00"
)

submit_or_test() {
  local name="$1"; shift
  if [[ "$SUBMIT" -eq 1 ]]; then
    local job_id
    job_id=$(sbatch --parsable "$@")
    printf '%s\t%s\n' "$name" "$job_id" | tee -a "$REPO_ROOT/slurm/logs/submitted.tsv"
  else
    printf '\n--- %s ---\n' "$name"
    sbatch --test-only "$@"
  fi
}

common_args=(
  --account="$PLG_ACCOUNT"
  --partition="$PLG_PARTITION"
  --cpus-per-task="$CPUS_PER_TASK"
  --qos="${PLG_QOS:-normal}"
)

# The smoke path overrides to the 'now' QOS: it starts immediately but allows
# only one submitted job per user, so it suits a single calibration task and
# nothing larger.
smoke_args=(
  --account="$PLG_ACCOUNT"
  --partition="$PLG_PARTITION"
  --cpus-per-task="$CPUS_PER_TASK"
  --qos=now
)

# Job alerts. Slurm sends one mail per array as a whole, not per array task,
# unless ARRAY_TASKS is included in the type list -- so this stays quiet.
if [[ -n "${PLG_MAIL:-}" ]]; then
  common_args+=(
    --mail-user="$PLG_MAIL"
    --mail-type="${PLG_MAIL_TYPE:-BEGIN,END,FAIL}"
  )
  echo "job alerts -> $PLG_MAIL (${PLG_MAIL_TYPE:-BEGIN,END,FAIL})"
else
  echo "job alerts disabled (PLG_MAIL empty in cluster.env)"
fi

if [[ "$SMOKE" -eq 1 ]]; then
  # Single task: DiCoFlex on default, seed 0. Cheapest method x smallest
  # dataset (860 factuals), so it calibrates training cost fastest.
  # Index = dataset_idx * n_seeds + seed_idx; default is DATASETS[2], so 2*3+0.
  submit_or_test "smoke-dicoflex-default-seed0" \
    "${smoke_args[@]}" \
    --job-name=cf-smoke \
    --time=04:00:00 \
    --array=6 \
    --export="ALL,CF_METHOD=dicoflex,CF_SEEDS=0 1 2,CF_RESULTS_TAG=smoke" \
    slurm/run-baselines.sbatch
  exit 0
fi

METHODS=(dice cchvae dicoflex)
[[ -n "$ONLY" ]] && METHODS=("$ONLY")

for method in "${METHODS[@]}"; do
  submit_or_test "seeds-$method" \
    "${common_args[@]}" \
    --job-name="cf-$method" \
    --time="${WALLTIME[$method]}" \
    --array="0-14%$MAX_CONCURRENT" \
    --export="ALL,CF_METHOD=$method,CF_SEEDS=0 1 2,CF_RESULTS_TAG=seeds" \
    slurm/run-baselines.sbatch
done

# DiCoFlex currently runs at target_class=1 while DiCE/CCHVAE/TabDCE run at 0
# (dicoflex_traintest_config.yaml:48, and DEFAULT_TARGETS in
# scripts/run_constraint_setup_experiments.py:110), so its Table 1 row is
# computed on a disjoint query set. This array re-runs it at target_class=0 so
# the methods become poolable and paired tests become possible.
if [[ -z "$ONLY" || "$ONLY" == "dicoflex" ]]; then
  submit_or_test "seeds-dicoflex-tc0" \
    "${common_args[@]}" \
    --job-name=cf-dicoflex-tc0 \
    --time="${WALLTIME[dicoflex]}" \
    --array="0-14%$MAX_CONCURRENT" \
    --export="ALL,CF_METHOD=dicoflex,CF_SEEDS=0 1 2,CF_RESULTS_TAG=seeds-tc0,CF_TARGET_CLASS=0" \
    slurm/run-baselines.sbatch
fi

# Slack task 3. run_constraint_setup_experiments.py only accepts adult,
# default and lending-club; default is already covered by outputs/sweep_2026-04-25.
if [[ -z "$ONLY" ]]; then
  submit_or_test "constraints-adult" \
    "${common_args[@]}" \
    --job-name=cf-constraints \
    --time=24:00:00 \
    --array="0-4%$MAX_CONCURRENT" \
    --export="ALL,CF_SWEEP_DATASETS=adult,CF_SWEEP_SETUPS=1 2 3 4 5,CF_RESULTS_TAG=constraints" \
    slurm/run-constraints.sbatch
fi

if [[ "$SUBMIT" -eq 0 ]]; then
  cat <<'NOTE'

Dry run only. Nothing was queued.

Before --submit:
  1. Run --smoke --submit and check the elapsed time in sacct.
  2. Compare that against the walltimes in this script and adjust.
  3. Confirm the partition MaxTime allows the CCHVAE 24 h request.
NOTE
fi
