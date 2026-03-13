#!/bin/bash

set -e

export PYTHONUNBUFFERED=1

# Dataset configurations for multirun
# DATASETS="dataset.config_path=config/datasets/adult_census.yaml,config/datasets/audit.yaml,config/datasets/bank_marketing.yaml,config/datasets/blobs.yaml,config/datasets/credit_default.yaml,config/datasets/digits.yaml,config/datasets/german_credit.yaml,config/datasets/give_me_some_credit.yaml,config/datasets/heloc.yaml,config/datasets/law.yaml,config/datasets/lending_club.yaml,config/datasets/moons.yaml,config/datasets/wine.yaml"

# PPCEF
# uv run counterfactuals/pipelines/run_ppcef_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# CCHVAE
# uv run counterfactuals/pipelines/run_cchvae_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# DiCE
# uv run counterfactuals/pipelines/run_dice_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# CADEX
# uv run counterfactuals/pipelines/run_cadex_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# CaseBased SACE
# uv run counterfactuals/pipelines/run_casebased_sace_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# CEFlow
# uv run counterfactuals/pipelines/run_ceflow_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# PPCEF
# uv run counterfactuals/pipelines/run_ppcef_pipeline.py --multirun $DATASETS disc_model=mlr


# DATASETS="dataset.config_path=config/datasets/blobs.yaml,config/datasets/digits.yaml,config/datasets/wine.yaml"
DATASETS="dataset.config_path=config/datasets/audit.yaml,config/datasets/moons.yaml,config/datasets/heloc.yaml,config/datasets/blobs.yaml,config/datasets/digits.yaml,config/datasets/wine.yaml"

# CEGP
# uv run counterfactuals/pipelines/run_cegp_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# CEM
# uv run counterfactuals/pipelines/run_cem_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# Artelt
DATASETS="dataset.config_path=config/datasets/digits.yaml"
uv run counterfactuals/pipelines/run_artelt_pipeline.py --multirun $DATASETS disc_model=mlp

# WACH Ours
# uv run counterfactuals/pipelines/run_wach_ours_pipeline.py --multirun $DATASETS disc_model=mlp,mlr
