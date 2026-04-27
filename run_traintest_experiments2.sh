#!/bin/bash
# Run CCHVAE + DiCE traintest pipelines sequentially across datasets.
#
# Usage: ./run_traintest_experiments.sh
#
# Splits live under data_train_test_val/<dir>/{train,test}.csv
#
# Actionability is taken from the shared config/datasets/*_split.yaml files.
# These methods cannot enforce directional constraints (e.g. "age may only
# grow"), so features like `age` are marked non-actionable there and will be
# held fixed — this is the intended safe default. DiCoFlex handles the
# directional case separately in run_dicoflex_traintest_experiments.sh.

set -e

DATA_ROOT="data_train_test_val"

DATASETS=(
    "gmc"
    "lending-club"
)

PIPELINES=(
    "counterfactuals.pipelines.run_cchvae_traintest_pipeline"
    "counterfactuals.pipelines.run_dice_traintest_pipeline"
)

# Map splits dir → dataset config name (differs only for lending-club → lending_club_split)
config_name_for() {
    case "$1" in
        lending-club) echo "lending_club_split" ;;
        *) echo "${1}_split" ;;
    esac
}

echo "Starting traintest experiments..."
echo "Datasets:  ${DATASETS[*]}"
echo "Pipelines: ${PIPELINES[*]}"
echo ""

for dataset in "${DATASETS[@]}"; do
    dataset_cfg="$(config_name_for "$dataset")"
    train_path="${DATA_ROOT}/${dataset}/train.csv"
    test_path="${DATA_ROOT}/${dataset}/test.csv"

    for pipeline in "${PIPELINES[@]}"; do
        echo "========================================"
        echo "Running: pipeline=$pipeline, dataset=$dataset"
        echo "========================================"

        uv run python -m "$pipeline" \
            disc_model=simple_mlp \
            disc_model.train_model=true \
            gen_model.train_model=true \
            dataset.config_path="config/datasets/${dataset_cfg}.yaml" \
            dataset.train_data_path="$train_path" \
            dataset.test_data_path="$test_path"

        echo "Completed: pipeline=$pipeline, dataset=$dataset"
        echo ""
    done
done

echo "All experiments completed!"
