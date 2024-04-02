python alternative_methods/cem_exp.py --multirun \
    neptune.enable=true \
    dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \
    disc_model=mlr
