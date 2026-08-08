#!/usr/bin/env bash
# Run the seed sweep locally: methods x datasets x seeds, no Slurm.
#
# Local equivalent of slurm/run-baselines.sbatch. Same hydra overrides, same
# output layout, so scripts/calculate_metrics.py and the seed aggregator work
# against either.
#
# Usage:
#   ./run_seed_experiments.sh --dry-run
#   ./run_seed_experiments.sh
#   ./run_seed_experiments.sh --methods dice dicoflex --seeds 0 1 2
#   ./run_seed_experiments.sh --methods dicoflex --target-class 0 --tag seeds-tc0
#
# Results:
#   results/<tag>/seed_<N>/<dataset>_split/<Method>/fold_0/cf_metrics_*.csv
#
# Each (method, dataset, seed) writes to a distinct directory, so a rerun of
# one cell never disturbs another and an interrupted sweep can be resumed by
# re-running with a narrower --methods/--datasets/--seeds.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# A non-interactive ssh session does not source the login profile, so uv may
# not be on PATH even when it is installed. Resolve it explicitly.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  echo "uv not found on PATH ($PATH)" >&2
  exit 1
fi

METHODS=(dice cchvae dicoflex)
DATASETS=(adult bank default gmc lending-club)
SEEDS=(0 1 2)
TAG=seeds
TARGET_CLASS=""
DRY_RUN=0
JOBS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) shift; METHODS=(); while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done ;;
    --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --tag) TAG="$2"; shift 2 ;;
    --target-class) TARGET_CLASS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# A case statement rather than an associative array: macOS ships bash 3.2,
# which has no `declare -A`. The sbatch files can use one because the cluster
# runs bash 4+, but this script has to work on the laptop too.
pipeline_for() {
  case "$1" in
    dice) echo "counterfactuals.pipelines.run_dice_traintest_pipeline" ;;
    cchvae) echo "counterfactuals.pipelines.run_cchvae_traintest_pipeline" ;;
    dicoflex) echo "counterfactuals.pipelines.run_dicoflex_traintest_pipeline" ;;
    *) echo "Unknown method '$1'" >&2; return 1 ;;
  esac
}

RESULTS="$REPO_ROOT/results/$TAG"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

total=$(( ${#METHODS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))
echo "methods:  ${METHODS[*]}"
echo "datasets: ${DATASETS[*]}"
echo "seeds:    ${SEEDS[*]}"
echo "runs:     $total  ->  $RESULTS"
echo

n=0
failed=()
for method in "${METHODS[@]}"; do
  PIPELINE="$(pipeline_for "$method")"
  for dataset in "${DATASETS[@]}"; do
    if [[ "$dataset" == "lending-club" ]]; then
      CFG_STEM="lending_club_split"
    else
      CFG_STEM="${dataset}_split"
    fi
    for seed in "${SEEDS[@]}"; do
      n=$(( n + 1 ))
      OUT_ROOT="$RESULTS/seed_$seed"
      label="$method/$dataset/seed$seed"
      log="$LOGS/${method}_${CFG_STEM}_seed${seed}.log"

      CMD=(
        "$UV" run python -m "$PIPELINE"
        disc_model=simple_mlp
        disc_model.train_model=true
        gen_model.train_model=true
        "experiment.seed=$seed"
        "experiment.output_folder=$OUT_ROOT"
        "dataset.config_path=config/datasets/${CFG_STEM}.yaml"
        "dataset.train_data_path=data_train_test_val/${dataset}/train.csv"
        "dataset.test_data_path=data_train_test_val/${dataset}/test.csv"
        "hydra.run.dir=$OUT_ROOT/hydra/${CFG_STEM}_${method}"
      )
      [[ -n "$TARGET_CLASS" ]] && CMD+=("++counterfactuals_params.target_class=$TARGET_CLASS")

      if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[%d/%d] %s\n      %s\n' "$n" "$total" "$label" "${CMD[*]}"
        continue
      fi

      # Wait for a free slot. bash 3.2 (macOS) has no `wait -n`, so poll.
      while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$JOBS" ]; do sleep 5; done

      printf '[%d/%d] start %-28s log: %s\n' "$n" "$total" "$label" "$log"
      rm -f "$log.status"
      (
        start=$SECONDS
        if "${CMD[@]}" > "$log" 2>&1; then
          printf 'ok %s %ds\n' "$label" "$(( SECONDS - start ))" > "$log.status"
        else
          printf 'FAILED %s %ds\n' "$label" "$(( SECONDS - start ))" > "$log.status"
        fi
      ) &
    done
  done
done

wait

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

for st in "$LOGS"/*.status; do
  [[ -f "$st" ]] || continue
  read -r state rest < "$st"
  if [[ "$state" == "FAILED" ]]; then failed+=("$rest"); fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "${#failed[@]} of $total runs failed:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "all $total runs completed"
