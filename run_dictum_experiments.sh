#!/usr/bin/env bash
# Run the DICTUM-aligned sweep locally: methods x datasets x seeds, no Slurm.
#
# Same layout and job handling as run_seed_experiments.sh, but pointed at the
# dictum_*_config.yaml configs, which z-score the model space, use DICTUM's
# [32, 32] classifier with val-based early stopping, explain 2000 test rows in
# both flip directions, and default to seeds 42/43/44.
#
# Usage:
#   ./run_dictum_experiments.sh --dry-run
#   ./run_dictum_experiments.sh
#   ./run_dictum_experiments.sh --methods dice --datasets adult --seeds 42
#
# Results:
#   results/<tag>/seed_<N>/<dataset>_split/<Method>/fold_0/{counterfactuals,factuals,cf_metrics}_*.csv
#
# Score them with:
#   uv run python -m scripts.compute_dictum_metrics --results-root results/<tag>
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
SEEDS=(42 43 44)
TAG=dictum
N_TEST_SAMPLES=2000
DRY_RUN=0
JOBS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) shift; METHODS=(); while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done ;;
    --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --tag) TAG="$2"; shift 2 ;;
    --n-test-samples) N_TEST_SAMPLES="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# A case statement rather than an associative array: macOS ships bash 3.2,
# which has no `declare -A`.
pipeline_for() {
  case "$1" in
    dice) echo "counterfactuals.pipelines.run_dice_traintest_pipeline" ;;
    cchvae) echo "counterfactuals.pipelines.run_cchvae_traintest_pipeline" ;;
    dicoflex) echo "counterfactuals.pipelines.run_dicoflex_traintest_pipeline" ;;
    *) echo "Unknown method '$1'" >&2; return 1 ;;
  esac
}

config_for() {
  case "$1" in
    dice) echo "dictum_dice_config" ;;
    cchvae) echo "dictum_cchvae_config" ;;
    dicoflex) echo "dictum_dicoflex_config" ;;
    *) echo "Unknown method '$1'" >&2; return 1 ;;
  esac
}

stem_for() {
  if [[ "$1" == "lending-club" ]]; then echo "lending_club_split"; else echo "${1}_split"; fi
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

# One classifier per (dataset, seed), trained before any method runs and then
# loaded by all of them. Without this each method trains its own into the same
# path, so the baselines would explain separately-trained models and would race
# when --jobs > 1.
echo "=== training shared classifiers ==="
for dataset in "${DATASETS[@]}"; do
  CFG_STEM="$(stem_for "$dataset")"
  for seed in "${SEEDS[@]}"; do
    OUT_ROOT="$RESULTS/seed_$seed"
    DISC_PT="$OUT_ROOT/$CFG_STEM/fold_0/disc_model_SimpleMLPClassifier.pt"
    log="$LOGS/disc_${CFG_STEM}_seed${seed}.log"

    PRE_CMD=(
      "$UV" run python -m scripts.train_shared_disc_model
      "--config-name=dictum_dice_config"
      "experiment.seed=$seed"
      "experiment.output_folder=$OUT_ROOT"
      "dataset.config_path=config/datasets/${CFG_STEM}.yaml"
      "dataset.train_data_path=data_train_test_val/${dataset}/train.csv"
      "dataset.test_data_path=data_train_test_val/${dataset}/test.csv"
      "dataset.val_data_path=data_train_test_val/${dataset}/val.csv"
      "hydra.run.dir=$OUT_ROOT/hydra/${CFG_STEM}_disc"
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf 'disc %s/seed%s\n      %s\n' "$dataset" "$seed" "${PRE_CMD[*]}"
      continue
    fi

    if [[ -f "$DISC_PT" ]]; then
      echo "  reuse $CFG_STEM/seed$seed"
      continue
    fi
    echo "  train $CFG_STEM/seed$seed  log: $log"
    if ! "${PRE_CMD[@]}" > "$log" 2>&1; then
      echo "FAILED to train classifier for $dataset/seed$seed — see $log" >&2
      exit 1
    fi
  done
done
echo

n=0
failed=()
for method in "${METHODS[@]}"; do
  PIPELINE="$(pipeline_for "$method")"
  CONFIG="$(config_for "$method")"
  for dataset in "${DATASETS[@]}"; do
    CFG_STEM="$(stem_for "$dataset")"
    for seed in "${SEEDS[@]}"; do
      n=$(( n + 1 ))
      OUT_ROOT="$RESULTS/seed_$seed"
      label="$method/$dataset/seed$seed"
      log="$LOGS/${method}_${CFG_STEM}_seed${seed}.log"

      CMD=(
        "$UV" run python -m "$PIPELINE"
        "--config-name=$CONFIG"
        "experiment.seed=$seed"
        "experiment.output_folder=$OUT_ROOT"
        "disc_model.train_model=false"
        "dataset.config_path=config/datasets/${CFG_STEM}.yaml"
        "dataset.train_data_path=data_train_test_val/${dataset}/train.csv"
        "dataset.test_data_path=data_train_test_val/${dataset}/test.csv"
        "dataset.val_data_path=data_train_test_val/${dataset}/val.csv"
        "++counterfactuals_params.n_test_samples=$N_TEST_SAMPLES"
        "hydra.run.dir=$OUT_ROOT/hydra/${CFG_STEM}_${method}"
      )

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
