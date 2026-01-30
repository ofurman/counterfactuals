#!/bin/bash

set -e

# Dataset configurations for multirun
# Excludes mnist.yaml (missing data)
DATASETS="dataset.config_path=\
config/datasets/adult.yaml,\
config/datasets/adult_census.yaml,\
config/datasets/audit.yaml,\
config/datasets/bank_marketing.yaml,\
config/datasets/blobs.yaml,\
config/datasets/compas.yaml,\
config/datasets/credit_default.yaml,\
config/datasets/digits.yaml,\
config/datasets/german_credit.yaml,\
config/datasets/give_me_some_credit.yaml,\
config/datasets/heloc.yaml,\
config/datasets/law.yaml,\
config/datasets/lending_club.yaml,\
config/datasets/moons.yaml,\
config/datasets/wine.yaml\
"

echo "Running GLOBE-CE Pipeline on all datasets..."
uv run python -m counterfactuals.pipelines.run_globe_ce_pipeline --multirun $DATASETS

echo "Running ARES Pipeline on all datasets..."
uv run python -m counterfactuals.pipelines.run_ares_pipeline --multirun $DATASETS

echo "Benchmark run complete!"
