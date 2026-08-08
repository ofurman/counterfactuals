#!/bin/bash -l
#SBATCH --job-name=cf-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

# One array task per (dataset, seed) for a single baseline method.
#
# NO --gres=gpu ON PURPOSE. All three baseline pipelines set
# CUDA_VISIBLE_DEVICES=-1 at the top of main(), so they are CPU-only and a
# requested GPU would sit idle while consuming the GPU grant.
#
# Driven by environment, set by submit-all.sh:
#   CF_METHOD        dice | cchvae | dicoflex          (required)
#   CF_SEEDS         space-separated seeds, default "0 1 2"
#   CF_RESULTS_TAG   subdirectory under results/, default "seeds"
#   CF_TARGET_CLASS  optional target_class override (used for the DiCoFlex
#                    target_class=0 re-run that makes it poolable with the
#                    other methods; see README)
#
# --cpus-per-task is set at submission from cluster.env and is identical for
# every method so the reported timings stay comparable.

set -euo pipefail

HERE="$SLURM_SUBMIT_DIR/slurm"
# shellcheck disable=SC1091
source "$HERE/cluster.env"

: "${CF_METHOD:?Set CF_METHOD to dice, cchvae or dicoflex}"
: "${PLG_GROUP:?Set PLG_GROUP in slurm/cluster.env}"

CF_SEEDS="${CF_SEEDS:-0 1 2}"
CF_RESULTS_TAG="${CF_RESULTS_TAG:-seeds}"

PROJECT="$HOME/projects/$PROJECT_NAME"
HEAVY="$PLG_GROUPS_STORAGE/$PLG_GROUP/$USER/$PROJECT_NAME"
RESULTS="$HEAVY/results/$CF_RESULTS_TAG"

declare -A PIPELINES=(
  [dice]="counterfactuals.pipelines.run_dice_traintest_pipeline"
  [cchvae]="counterfactuals.pipelines.run_cchvae_traintest_pipeline"
  [dicoflex]="counterfactuals.pipelines.run_dicoflex_traintest_pipeline"
)
PIPELINE="${PIPELINES[$CF_METHOD]:?Unknown CF_METHOD '$CF_METHOD'}"

DATASETS=(adult bank default gmc lending-club)
read -r -a SEEDS <<< "$CF_SEEDS"

n_seeds=${#SEEDS[@]}
dataset_idx=$(( SLURM_ARRAY_TASK_ID / n_seeds ))
seed_idx=$(( SLURM_ARRAY_TASK_ID % n_seeds ))

if (( dataset_idx >= ${#DATASETS[@]} )); then
  echo "Array index $SLURM_ARRAY_TASK_ID exceeds the task matrix" >&2
  exit 1
fi

DATASET="${DATASETS[$dataset_idx]}"
SEED="${SEEDS[$seed_idx]}"

# Only lending-club's directory name differs from its config stem.
if [[ "$DATASET" == "lending-club" ]]; then
  CFG_STEM="lending_club_split"
else
  CFG_STEM="${DATASET}_split"
fi

OUT_ROOT="$RESULTS/seed_$SEED"

export XDG_CACHE_HOME="$HEAVY/cache"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export TORCH_HOME="$HEAVY/cache/torch"

module purge
module load ML-bundle/25.10

# Thread pinning MUST come after `module load`. ML-bundle/25.10 forces
# OMP_NUM_THREADS=1 ("Due to performance issues OMP_NUM_THREADS is set to 1"),
# so setting it beforehand is silently discarded: the 2026-07-27 sweep ran
# every task single-threaded on 8 allocated CPUs, which pushed DiCoFlex to
# ~4 min/epoch and killed 7 of 15 runs on the 12h wall.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
echo "threads: OMP_NUM_THREADS=$OMP_NUM_THREADS (cpus-per-task=${SLURM_CPUS_PER_TASK:-unset})"

cd "$PROJECT"

mkdir -p "$OUT_ROOT"

# Reproducibility record, one file per array task.
META="$OUT_ROOT/${CFG_STEM}_${CF_METHOD}_run.json"
cat > "$META" <<META_EOF
{
  "job_id": "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}",
  "method": "$CF_METHOD",
  "dataset": "$DATASET",
  "config_stem": "$CFG_STEM",
  "seed": $SEED,
  "target_class_override": "${CF_TARGET_CLASS:-none}",
  "cpus_per_task": "${SLURM_CPUS_PER_TASK:-unset}",
  "omp_num_threads": "$OMP_NUM_THREADS",
  "node": "$(hostname)",
  "arch": "$(uname -m)",
  "git_commit": "$(git -C "$PROJECT" rev-parse HEAD 2>/dev/null || echo unknown)",
  "modules": "$(module -t list 2>&1 | tr '\n' ' ')"
}
META_EOF

echo "method=$CF_METHOD dataset=$DATASET seed=$SEED out=$OUT_ROOT"

CMD=(
  "$PROJECT/.venv/bin/python" -m "$PIPELINE"
  disc_model=simple_mlp
  disc_model.train_model=true
  gen_model.train_model=true
  "experiment.seed=$SEED"
  "experiment.output_folder=$OUT_ROOT"
  "dataset.config_path=config/datasets/${CFG_STEM}.yaml"
  "dataset.train_data_path=data_train_test_val/${DATASET}/train.csv"
  "dataset.test_data_path=data_train_test_val/${DATASET}/test.csv"
  # Hydra defaults to outputs/<date>/<HH-MM-SS>, which two array tasks
  # starting in the same second would share, interleaving their logs and
  # overwriting each other's .hydra/config.yaml. Results are unaffected
  # (they go to OUT_ROOT) but the per-run provenance would be corrupted.
  "hydra.run.dir=$OUT_ROOT/hydra/${CFG_STEM}_${CF_METHOD}"
)

if [[ -n "${CF_TARGET_CLASS:-}" ]]; then
  CMD+=("++counterfactuals_params.target_class=$CF_TARGET_CLASS")
fi

srun "${CMD[@]}"

echo "done method=$CF_METHOD dataset=$DATASET seed=$SEED"
