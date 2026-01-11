#!/bin/bash
# Run DiCoFlex pairwise experiments for multiple datasets and discriminative models
#
# Usage: ./run_dicoflex_pairwise_experiments.sh
#
# This script runs the DiCoFlex pairwise pipeline for:
# - Datasets: lending_club, adult_census, credit_default, give_me_some_credit, bank_marketing
# - Discriminative models: mlp, mlr

set -e

DATASETS=(
    "lending_club"
    "adult_census"
    "credit_default"
    "give_me_some_credit"
    "bank_marketing"
)

DISC_MODELS=(
    "mlp"
    "mlr"
)

echo "Starting DiCoFlex pairwise experiments..."
echo "Datasets: ${DATASETS[*]}"
echo "Discriminative models: ${DISC_MODELS[*]}"
echo ""

for dataset in "${DATASETS[@]}"; do
    for disc_model in "${DISC_MODELS[@]}"; do
        echo "========================================"
        echo "Running: dataset=$dataset, disc_model=$disc_model"
        echo "========================================"

        uv run python -m counterfactuals.pipelines.run_dicoflex_pairwise_pipeline \
            disc_model="$disc_model" \
            dataset.config_path="config/datasets/${dataset}.yaml" \
            experiment.output_folder="models/dicoflex_pairwise/${dataset}/${disc_model}"

        echo "Completed: dataset=$dataset, disc_model=$disc_model"
        echo ""
    done
done

echo "All experiments completed!"
