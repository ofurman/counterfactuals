#!/bin/bash

set -e

# --- DATASET GROUPS ---

# 1. BINARY ROBUST (Standard configs, Batch=32 safe)
# ARES + GLOBE-CE
DATASETS_BINARY_ROBUST="dataset.config_path=\
config/datasets/adult.yaml,\
config/datasets/adult_census.yaml,\
config/datasets/compas.yaml,\
config/datasets/credit_default.yaml,\
config/datasets/give_me_some_credit.yaml,\
config/datasets/heloc.yaml,\
config/datasets/law.yaml,\
config/datasets/lending_club.yaml\
"

# 2. BINARY FRAGILE (Require Batch=8 for stability)
# ARES + GLOBE-CE
DATASETS_BINARY_FRAGILE="dataset.config_path=\
config/datasets/german_credit.yaml,\
config/datasets/bank_marketing.yaml\
"

# 3. SPECIAL (Require Target=0 + Batch=8)
# ARES + GLOBE-CE (audit, blobs, moons)
DATASETS_SPECIAL="dataset.config_path=\
config/datasets/audit.yaml,\
config/datasets/blobs.yaml,\
config/datasets/moons.yaml\
"

# 4. MULTI-CLASS (GLOBE-CE ONLY)
# Require Batch=8 for stability (Wine, Digits)
DATASETS_MULTICLASS="dataset.config_path=\
config/datasets/digits.yaml,\
config/datasets/wine.yaml\
"

echo "Creating timestamped results directory..."
# Create timestamped results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/benchmark_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Saving results to: $OUTPUT_DIR"


# --- EXECUTION FUNCTIONS ---

run_globe_ce() {
    GROUP=$1
    DATASETS=$2
    BATCH=$3
    EXTRA=$4
    echo "Running GLOBE-CE on $GROUP (Batch=$BATCH)..."
    uv run python -m counterfactuals.pipelines.run_globe_ce_pipeline --multirun $DATASETS \
        disc_model.train_model=true disc_model.epochs=10 disc_model.batch_size=$BATCH \
        gen_model.train_model=true gen_model.epochs=10 gen_model.batch_size=$BATCH \
        counterfactuals_params.batch_size=$BATCH \
        experiment.output_folder="$OUTPUT_DIR/globe_ce" \
        $EXTRA
}

run_ares() {
    GROUP=$1
    DATASETS=$2
    BATCH=$3
    EXTRA=$4
    echo "Running ARES on $GROUP (Batch=$BATCH)..."
    uv run python -m counterfactuals.pipelines.run_ares_pipeline --multirun $DATASETS \
        disc_model.train_model=true disc_model.epochs=10 disc_model.batch_size=$BATCH \
        gen_model.train_model=true gen_model.epochs=10 gen_model.batch_size=$BATCH \
        counterfactuals_params.batch_size=$BATCH \
        experiment.output_folder="$OUTPUT_DIR/ares" \
        $EXTRA
}

# --- RUNNING BENCHMARKS ---

# 1. Binary Robust
run_globe_ce "BINARY_ROBUST" "$DATASETS_BINARY_ROBUST" 32 ""
run_ares "BINARY_ROBUST" "$DATASETS_BINARY_ROBUST" 32 ""

# 2. Binary Fragile
run_globe_ce "BINARY_FRAGILE" "$DATASETS_BINARY_FRAGILE" 8 ""
run_ares "BINARY_FRAGILE" "$DATASETS_BINARY_FRAGILE" 8 ""

# 3. Special (Target 0)
run_globe_ce "SPECIAL" "$DATASETS_SPECIAL" 8 "counterfactuals_params.target_class=0"
run_ares "SPECIAL" "$DATASETS_SPECIAL" 8 "counterfactuals_params.target_class=0"

# 4. Multi-class (GLOBE-CE Only)
run_globe_ce "MULTICLASS" "$DATASETS_MULTICLASS" 8 ""

echo "Benchmark run complete! Results saved in $OUTPUT_DIR"
