#!/bin/bash

DATASETS=("BlobsDataset" "MoonsDataset" "LawDataset" "HelocDataset" "WineDataset" "DigitsDataset")


python counterfactuals/pipelines/run_wach_ours_pipeline.py -m\
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    counterfactuals_params.alpha=1 \
    disc_model=mlp,mlr &

python counterfactuals/pipelines/run_wach_ours_pipeline.py -m\
    dataset._target_=counterfactuals.datasets.LawDataset \
    counterfactuals_params.alpha=1 \
    disc_model=mlp,mlr &

python counterfactuals/pipelines/run_wach_ours_pipeline.py -m\
    dataset._target_=counterfactuals.datasets.HelocDataset \
    counterfactuals_params.alpha=1 \
    disc_model=mlp,mlr &

python counterfactuals/pipelines/run_wach_ours_pipeline.py -m\
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    counterfactuals_params.alpha=1 \
    disc_model=mlp,mlr &

python counterfactuals/pipelines/run_wach_ours_pipeline.py -m\
    dataset._target_=counterfactuals.datasets.WineDataset \
    counterfactuals_params.alpha=1 \
    gen_model=medium_maf \
    disc_model=mlp,mlr &

python counterfactuals/pipelines/run_wach_ours_pipeline.py -m\
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    counterfactuals_params.alpha=1 \
    gen_model=digits_maf \
    disc_model=mlp_large,mlr &

wait
