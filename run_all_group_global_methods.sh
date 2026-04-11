#!/bin/bash

set -e

export PYTHONUNBUFFERED=1

# NODE extra experiments — subset of datasets and group/global methods
DATASETS="dataset.config_path=config/datasets/adult_census.yaml,config/datasets/heloc.yaml,config/datasets/german_credit.yaml,config/datasets/moons.yaml,config/datasets/digits.yaml"

# Group GLANCE (s=3)
DATASETS_SMALL="dataset.config_path=config/datasets/german_credit.yaml,config/datasets/moons.yaml"
uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS_SMALL disc_model=node counterfactuals_params.cf_method._target_=counterfactuals.cf_methods.glance.GroupGLANCE counterfactuals_params.cf_method.s=3

# Group GLANCE (s=3, k=1) for smaller-cluster datasets
DATASETS_CLUSTERS="dataset.config_path=config/datasets/digits.yaml"
uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS_CLUSTERS disc_model=node counterfactuals_params.cf_method._target_=counterfactuals.cf_methods.glance.GroupGLANCE counterfactuals_params.cf_method.s=3 counterfactuals_params.cf_method.k=1

# Group GLANCE (s=10) for larger datasets
DATASETS_LARGE="dataset.config_path=config/datasets/adult_census.yaml,config/datasets/heloc.yaml"
uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS_LARGE disc_model=node counterfactuals_params.cf_method._target_=counterfactuals.cf_methods.glance.GroupGLANCE counterfactuals_params.cf_method.s=10

# Global GLANCE
uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS disc_model=node

# AReS
uv run counterfactuals/pipelines/run_ares_pipeline.py --multirun $DATASETS disc_model=node

# GLOBE-CE
uv run counterfactuals/pipelines/run_globe_ce_pipeline.py --multirun $DATASETS disc_model=node

# T-CREX
uv run counterfactuals/pipelines/run_tcrex_pipeline.py --multirun $DATASETS disc_model=node
