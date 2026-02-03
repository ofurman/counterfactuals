#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-DiceExplainerWrapper}"
OUTPUT_DIR="${2:-metrics_tables_dice}"
MODELS_ROOT="${3:-models}"
METRICS_CONF_PATH="${5:-counterfactuals/pipelines/conf/metrics/default.yaml}"

DATASETS=(
  "moons"
  # "heloc"
  "audit"
)

# DATASETS=(
#   "moons"
#   "law"
#   "heloc"
#   "german_credit"
#   "audit"
#   "lending_club"
#   "adult_census"
#   "credit_default"
#   "give_me_some_credit"
#   "bank_marketing"
# )

DISC_MODELS=(
  "MLPClassifier"
  "MultinomialLogisticRegression"
)

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
