#!/bin/bash

DATASETS="dataset.config_path=config/datasets/adult_census.yaml,config/datasets/audit.yaml,config/datasets/bank_marketing.yaml,config/datasets/blobs.yaml,config/datasets/credit_default.yaml,config/datasets/digits.yaml,config/datasets/german_credit.yaml,config/datasets/give_me_some_credit.yaml,config/datasets/heloc.yaml,config/datasets/law.yaml,config/datasets/lending_club.yaml,config/datasets/moons.yaml,config/datasets/wine.yaml"

# 3 groups for GLANCE
# DATASETS="dataset.config_path=config/datasets/law.yaml,config/datasets/blobs.yaml,config/datasets/german_credit.yaml"
# uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS disc_model=mlp,mlr counterfactuals_params.cf_method._target_=counterfactuals.cf_methods.glance.GroupGLANCE counterfactuals_params.cf_method.s=3

# 3 groups for GLANCE with smaller clusters
# DATASETS="dataset.config_path=config/datasets/digits.yaml,config/datasets/wine.yaml"
# uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS disc_model=mlp,mlr counterfactuals_params.cf_method._target_=counterfactuals.cf_methods.glance.GroupGLANCE counterfactuals_params.cf_method.s=3 counterfactuals_params.cf_method.k=1

# 10 groups for GLANCE
# DATASETS="dataset.config_path=config/datasets/adult_census.yaml,config/datasets/bank_marketing.yaml,config/datasets/give_me_some_credit.yaml,config/datasets/heloc.yaml,config/datasets/lending_club.yaml,config/datasets/credit_default.yaml"
# uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS disc_model=mlp,mlr counterfactuals_params.cf_method._target_=counterfactuals.cf_methods.glance.GroupGLANCE counterfactuals_params.cf_method.s=10

# Rerun for global GLANCE
# DATASETS="dataset.config_path=config/datasets/german_credit.yaml,config/datasets/wine.yaml"
# uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS disc_model=mlp,mlr counterfactuals_params.cf_method.k=1

# GLANCE
# uv run counterfactuals/pipelines/run_glance_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# AReS
# DATASETS="dataset.config_path=config/datasets/heloc.yaml,config/datasets/lending_club.yaml,config/datasets/credit_default.yaml"
# uv run counterfactuals/pipelines/run_ares_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# GLOBE-CE
# uv run counterfactuals/pipelines/run_globe_ce_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# T-CREX
# uv run counterfactuals/pipelines/run_tcrex_pipeline.py --multirun $DATASETS disc_model=mlp,mlr

# AReS
# DATASETS="dataset.config_path=config/datasets/adult_census.yaml,config/datasets/bank_marketing.yaml,config/datasets/blobs.yaml,config/datasets/credit_default.yaml,config/datasets/digits.yaml,config/datasets/german_credit.yaml,config/datasets/give_me_some_credit.yaml,config/datasets/heloc.yaml,config/datasets/law.yaml,config/datasets/lending_club.yaml,config/datasets/moons.yaml,config/datasets/wine.yaml"
# DATASETS="dataset.config_path=config/datasets/credit_default.yaml,config/datasets/digits.yaml,"
DATASETS="dataset.config_path=config/datasets/german_credit.yaml,config/datasets/give_me_some_credit.yaml,config/datasets/heloc.yaml,config/datasets/law.yaml,config/datasets/lending_club.yaml,config/datasets/moons.yaml,config/datasets/wine.yaml"
uv run counterfactuals/pipelines/run_ares_pipeline.py --multirun $DATASETS disc_model=mlp,mlr
