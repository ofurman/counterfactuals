#!/bin/bash
# Run DiCoFlex traintest pipeline sequentially across datasets.
#
# Usage: ./run_dicoflex_traintest_experiments.sh
#
# Splits live under data_train_test_val/<dir>/{train,test}.csv

set -e

DATA_ROOT="data_train_test_val"

DATASETS=(
    "bank"
    "default"
)
    # "adult"
    # "gmc"
    # "lending-club"

PIPELINES=(
    "counterfactuals.pipelines.run_dicoflex_traintest_pipeline"
)

# Map splits dir → dataset config name (differs only for lending-club → lending_club_split)
config_name_for() {
    case "$1" in
        lending-club) echo "lending_club_split" ;;
        *) echo "${1}_split" ;;
    esac
}

# Monotonic overrides are DiCoFlex-only directional constraints (e.g., "age may
# only grow"). Other methods cannot enforce direction, so shared dataset yamls
# keep such features non-actionable; we re-enable them here with a direction.
monotonic_overrides_for() {
    case "$1" in
        adult)   echo '{age: INCREASE}' ;;
        bank)    echo '{age: INCREASE}' ;;
        default) echo '{AGE: INCREASE}' ;;
        gmc)     echo '{age: INCREASE}' ;;
        *)       echo '{}' ;;
    esac
}

echo "Starting DiCoFlex traintest experiments..."
echo "Datasets:  ${DATASETS[*]}"
echo "Pipelines: ${PIPELINES[*]}"
echo ""

for dataset in "${DATASETS[@]}"; do
    dataset_cfg="$(config_name_for "$dataset")"
    train_path="${DATA_ROOT}/${dataset}/train.csv"
    test_path="${DATA_ROOT}/${dataset}/test.csv"
    monotonic_overrides="$(monotonic_overrides_for "$dataset")"

    for pipeline in "${PIPELINES[@]}"; do
        echo "========================================"
        echo "Running: pipeline=$pipeline, dataset=$dataset"
        echo "Monotonic overrides: $monotonic_overrides"
        echo "========================================"

        uv run python -m "$pipeline" \
            disc_model=simple_mlp \
            disc_model.train_model=true \
            gen_model.train_model=true \
            dataset.config_path="config/datasets/${dataset_cfg}.yaml" \
            dataset.train_data_path="$train_path" \
            dataset.test_data_path="$test_path" \
            "++counterfactuals_params.monotonic_overrides=${monotonic_overrides}"

        echo "Completed: pipeline=$pipeline, dataset=$dataset"
        echo ""
    done
done

echo "All experiments completed!"
