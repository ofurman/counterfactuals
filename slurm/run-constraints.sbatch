#!/bin/bash -l
#SBATCH --job-name=cf-constraints
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

# Constraint-strictness sweep on an additional dataset (Slack task 3).
# One array task per (dataset, setup); the existing runner loops methods.
#
#   CF_SWEEP_DATASETS  space-separated dataset dirs, default "adult"
#   CF_SWEEP_SETUPS    space-separated setup numbers, default "1 2 3 4 5"
#   CF_RESULTS_TAG     subdirectory under results/, default "constraints"
#
# CPU-only, same reasoning as run-baselines.sbatch.

set -euo pipefail

HERE="$SLURM_SUBMIT_DIR/slurm"
# shellcheck disable=SC1091
source "$HERE/cluster.env"

: "${PLG_GROUP:?Set PLG_GROUP in slurm/cluster.env}"

CF_SWEEP_DATASETS="${CF_SWEEP_DATASETS:-adult}"
CF_SWEEP_SETUPS="${CF_SWEEP_SETUPS:-1 2 3 4 5}"
CF_RESULTS_TAG="${CF_RESULTS_TAG:-constraints}"

PROJECT="$HOME/projects/$PROJECT_NAME"
HEAVY="$PLG_GROUPS_STORAGE/$PLG_GROUP/$USER/$PROJECT_NAME"
RESULTS="$HEAVY/results/$CF_RESULTS_TAG"

read -r -a SWEEP_DATASETS <<< "$CF_SWEEP_DATASETS"
read -r -a SWEEP_SETUPS <<< "$CF_SWEEP_SETUPS"

n_setups=${#SWEEP_SETUPS[@]}
dataset_idx=$(( SLURM_ARRAY_TASK_ID / n_setups ))
setup_idx=$(( SLURM_ARRAY_TASK_ID % n_setups ))

if (( dataset_idx >= ${#SWEEP_DATASETS[@]} )); then
  echo "Array index $SLURM_ARRAY_TASK_ID exceeds the task matrix" >&2
  exit 1
fi

DATASET="${SWEEP_DATASETS[$dataset_idx]}"
SETUP="${SWEEP_SETUPS[$setup_idx]}"

export XDG_CACHE_HOME="$HEAVY/cache"
export TORCH_HOME="$HEAVY/cache/torch"
module purge
module load ML-bundle/25.10

# After `module load`: ML-bundle/25.10 forces OMP_NUM_THREADS=1, so exporting
# it earlier is silently overwritten. See run-baselines.sbatch.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
echo "threads: OMP_NUM_THREADS=$OMP_NUM_THREADS (cpus-per-task=${SLURM_CPUS_PER_TASK:-unset})"

cd "$PROJECT"

mkdir -p "$RESULTS"

echo "constraint sweep dataset=$DATASET setup=$SETUP out=$RESULTS"

srun "$PROJECT/.venv/bin/python" scripts/run_constraint_setup_experiments.py \
  --datasets "$DATASET" \
  --setups "$SETUP" \
  --results-root "$RESULTS"

echo "done dataset=$DATASET setup=$SETUP"
