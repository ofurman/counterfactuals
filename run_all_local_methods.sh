#!/bin/bash

set -e

export PYTHONUNBUFFERED=1

# NODE extra experiments — subset of datasets and local methods
DATASETS="dataset.config_path=config/datasets/moons.yaml,config/datasets/heloc.yaml,config/datasets/law.yaml,config/datasets/adult.yaml"
# DATASETS="dataset.config_path=config/datasets/law.yaml"

# PPCEF
# uv run counterfactuals/pipelines/run_ppcef_pipeline.py --multirun $DATASETS disc_model=node

# DiCE
# uv run counterfactuals/pipelines/run_dice_pipeline.py --multirun $DATASETS disc_model=node

# CCHVAE
uv run counterfactuals/pipelines/run_cchvae_pipeline.py --multirun $DATASETS disc_model=node

# CADEX
# uv run counterfactuals/pipelines/run_cadex_pipeline.py --multirun $DATASETS disc_model=node

# # CEFlow
# uv run counterfactuals/pipelines/run_ceflow_pipeline.py --multirun $DATASETS disc_model=node

# # WACH Ours
# uv run counterfactuals/pipelines/run_wach_ours_pipeline.py --multirun $DATASETS disc_model=node
