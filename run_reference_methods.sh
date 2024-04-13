python alternative_methods/artelt_exp_cv.py --multirun \
    neptune.enable=true \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset,\
counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \
    disc_model=mlr,mlp
