#!/bin/bash -l
#SBATCH --job-name=dictum-dicoflex-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -p lem-gpu
#SBATCH -A hpc-maciej.zieba-1766404231
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

# WCSS (LEM) variant of run-dictum-dicoflex.sbatch: flow training only, on GPU.
# Same contract: array 0-4 selects the dataset, CF_SEED / CF_RESULTS_TAG /
# CF_EXTRA_OVERRIDES via environment. Uses the project uv venv directly
# (x86_64 + H100, no module system needed) and keeps caches on Lustre because
# the home quota is nearly full.

set -euo pipefail

CF_SEED="${CF_SEED:-42}"
CF_RESULTS_TAG="${CF_RESULTS_TAG:-dictum-longtrain}"

read -r -a EXTRA_OVERRIDES <<< "${CF_EXTRA_OVERRIDES:-}"

PROJECT="/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/ce-dicoflex"
OUT_ROOT="$PROJECT/results/$CF_RESULTS_TAG/seed_$CF_SEED"

export UV_CACHE_DIR="/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/.uv-cache"
export XDG_CACHE_HOME="/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/.cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

DATASETS=(adult bank default gmc lending-club)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]:?Array index out of range}"
if [[ "$DATASET" == "lending-club" ]]; then
  CFG_STEM="lending_club_split"
else
  CFG_STEM="${DATASET}_split"
fi

cd "$PROJECT"
PY="$PROJECT/.venv/bin/python"
mkdir -p "$OUT_ROOT" slurm/logs

COMMON_DATA=(
  "experiment.seed=$CF_SEED"
  "experiment.output_folder=$OUT_ROOT"
  "dataset.config_path=config/datasets/${CFG_STEM}.yaml"
  "dataset.train_data_path=data_train_test_val/${DATASET}/train.csv"
  "dataset.test_data_path=data_train_test_val/${DATASET}/test.csv"
  "dataset.val_data_path=data_train_test_val/${DATASET}/val.csv"
)

echo "dicoflex wcss dataset=$DATASET seed=$CF_SEED out=$OUT_ROOT"
"$PY" -c "import torch; print('cuda available:', torch.cuda.is_available())"

srun "$PY" -m counterfactuals.pipelines.run_dicoflex_traintest_pipeline \
  "--config-name=dictum_dicoflex_config" \
  "${COMMON_DATA[@]}" \
  "gen_model.epochs=10000" \
  "gen_model.patience=50" \
  "experiment.use_gpu=true" \
  "++experiment.train_only=true" \
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
  "hydra.run.dir=$OUT_ROOT/hydra/${CFG_STEM}_dicoflex"

echo "done dataset=$DATASET seed=$CF_SEED"
