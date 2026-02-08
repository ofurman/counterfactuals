#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_metrics.sh [METHOD] [OUTPUT_DIR] [MODELS_ROOT] [DISC_MODELS_CSV] [METRICS_CONF_PATH]
#
# Examples:
#   ./run_metrics.sh DiceExplainerWrapper metrics_tables_dice models "" counterfactuals/pipelines/conf/metrics/default.yaml
#   ./run_metrics.sh CADEX metrics_tables models "MLPClassifier,MultinomialLogisticRegression" counterfactuals/pipelines/conf/metrics/default.yaml

METHOD="${1:-DiceExplainerWrapper}"
OUTPUT_DIR="${2:-metrics_tables_dice}"
MODELS_ROOT="${3:-models}"
DISC_MODELS_CSV="${4:-}"
METRICS_CONF_PATH="${5:-counterfactuals/pipelines/conf/metrics/default.yaml}"

# --- Dataset tiers ---
SMALL_DATASETS=(
  "moons"
  "blobs"
  "heloc"
  "audit"
  "digits"
  "wine"
)

FULL_DATASETS=(
  "lending_club"
  "adult_census"
  "give_me_some_credit"
  "bank_marketing"
  "heloc"
  "wine"
  "law"
  "german_credit"
  "blobs"
  "moons"
  "credit_default"
  "audit"
  "digits"
)

FULL_METHODS=(
  "DiceExplainerWrapper"
  "CADEX"
  "GlobalGLANCE"
  "GroupGLANCE"
  "GLOBE_CE"
  "DiCE"
  "PPCEF"
  "AReS"
  "TCREx"
  "CCHVAE"
  "CaseBasedSACE"
)

DISC_MODELS_DEFAULT=(
  "MLPClassifier"
  "MultinomialLogisticRegression"
)

is_in_array() {
  local needle="$1"
  shift
  local element
  for element in "$@"; do
    [[ "$element" == "$needle" ]] && return 0
  done
  return 1
}

DISC_MODELS=()
if [[ -n "$DISC_MODELS_CSV" ]]; then
  IFS=',' read -r -a DISC_MODELS <<< "$DISC_MODELS_CSV"
else
  DISC_MODELS=("${DISC_MODELS_DEFAULT[@]}")
fi

# Pick dataset list based on method membership
if is_in_array "$METHOD" "${FULL_METHODS[@]}"; then
  DATASETS=("${FULL_DATASETS[@]}")
  DATASET_TIER="FULL"
else
  DATASETS=("${SMALL_DATASETS[@]}")
  DATASET_TIER="SMALL"
fi

echo "METHOD=$METHOD"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MODELS_ROOT=$MODELS_ROOT"
echo "METRICS_CONF_PATH=$METRICS_CONF_PATH"
echo "DATASET_TIER=$DATASET_TIER"
echo "DATASETS_COUNT=${#DATASETS[@]}"
echo "DISC_MODELS_COUNT=${#DISC_MODELS[@]}"
echo

for dataset in "${DATASETS[@]}"; do
  for model in "${DISC_MODELS[@]}"; do
    echo "Calculating metrics for dataset=${dataset} method=${METHOD} model=${model}"
    uv run python scripts/calculate_metrics.py \
      --dataset "${dataset}" \
      --method "${METHOD}" \
      --model-name "${model}" \
      --output-dir "${OUTPUT_DIR}" \
      --models-root "${MODELS_ROOT}" \
      --metrics-conf-path "${METRICS_CONF_PATH}"
  done
done
