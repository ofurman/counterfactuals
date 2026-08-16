#!/bin/bash -l
#SBATCH --job-name=dictum-dicoflex-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

# DICTUM-aligned DiCoFlex: FLOW TRAINING ONLY, on GPU.
#
# One array task per dataset (array 0-4), single seed via CF_SEED (default 42).
# Uses --config-name=dictum_dicoflex_config (minmax_qt generation space) with
# the reference training budget of 10000 epochs / patience 50 — the
# undertrained flow (40-200 epochs) is the identified cause of the coverage
# gap (GOAL_dicoflex_validity.md §0).
#
# experiment.train_only=true stops the pipeline after the flow checkpoint is
# saved. Generation (sampling, temp 0.8, M=100 candidates) runs later on the
# experiment machine with gen_model.train_model=false, so CF-generation
# timings stay comparable with the other baselines measured there.
#
# Environment:
#   CF_SEED          seed, default 42
#   CF_RESULTS_TAG   subdirectory under $HEAVY/results/, default dictum-longtrain

set -euo pipefail

HERE="$SLURM_SUBMIT_DIR/slurm"
# shellcheck disable=SC1091
source "$HERE/cluster.env"

: "${PLG_GROUP:?Set PLG_GROUP in slurm/cluster.env}"

CF_SEED="${CF_SEED:-42}"
CF_RESULTS_TAG="${CF_RESULTS_TAG:-dictum-longtrain}"

PROJECT="$HOME/projects/$PROJECT_NAME"
HEAVY="$PLG_GROUPS_STORAGE/$PLG_GROUP/$USER/$PROJECT_NAME"
OUT_ROOT="$HEAVY/results/$CF_RESULTS_TAG/seed_$CF_SEED"

DATASETS=(adult bank default gmc lending-club)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]:?Array index out of range}"
if [[ "$DATASET" == "lending-club" ]]; then
  CFG_STEM="lending_club_split"
else
  CFG_STEM="${DATASET}_split"
fi

export XDG_CACHE_HOME="$HEAVY/cache"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export TORCH_HOME="$HEAVY/cache/torch"

module purge
module load ML-bundle/25.10

# Thread pinning MUST come after `module load` (ML-bundle forces OMP_NUM_THREADS=1).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

cd "$PROJECT"
PY="$PROJECT/.venv/bin/python"
mkdir -p "$OUT_ROOT"

COMMON_DATA=(
  "experiment.seed=$CF_SEED"
  "experiment.output_folder=$OUT_ROOT"
  "dataset.config_path=config/datasets/${CFG_STEM}.yaml"
  "dataset.train_data_path=data_train_test_val/${DATASET}/train.csv"
  "dataset.test_data_path=data_train_test_val/${DATASET}/test.csv"
  "dataset.val_data_path=data_train_test_val/${DATASET}/val.csv"
)

# Shared classifier, one per (dataset, seed), reused by every method scored in
# this results root. Same recipe as run_dictum_experiments.sh.
DISC_PT="$OUT_ROOT/$CFG_STEM/fold_0/disc_model_SimpleMLPClassifier.pt"
if [[ ! -f "$DISC_PT" ]]; then
  echo "training shared classifier for $DATASET seed $CF_SEED"
  srun "$PY" -m scripts.train_shared_disc_model \
    "--config-name=dictum_dice_config" \
    "${COMMON_DATA[@]}" \
    "hydra.run.dir=$OUT_ROOT/hydra/${CFG_STEM}_disc"
fi

META="$OUT_ROOT/${CFG_STEM}_dicoflex_run.json"
cat > "$META" <<META_EOF
{
  "job_id": "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}",
  "method": "dicoflex",
  "dataset": "$DATASET",
  "seed": $CF_SEED,
  "gen_model_epochs": 10000,
  "gen_model_patience": 50,
  "cpus_per_task": "${SLURM_CPUS_PER_TASK:-unset}",
  "node": "$(hostname)",
  "arch": "$(uname -m)"
}
META_EOF

echo "dicoflex TRAIN dataset=$DATASET seed=$CF_SEED out=$OUT_ROOT"
"$PY" -c "import torch; print('cuda available:', torch.cuda.is_available())"

srun "$PY" -m counterfactuals.pipelines.run_dicoflex_traintest_pipeline \
  "--config-name=dictum_dicoflex_config" \
  "${COMMON_DATA[@]}" \
  "disc_model.train_model=false" \
  "experiment.use_gpu=true" \
  "++experiment.train_only=true" \
  "gen_model.epochs=10000" \
  "gen_model.patience=50" \
  "hydra.run.dir=$OUT_ROOT/hydra/${CFG_STEM}_dicoflex"

echo "done TRAIN dataset=$DATASET seed=$CF_SEED"
