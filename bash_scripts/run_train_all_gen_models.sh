python3 train_gen_model.py --multirun \
    dataset._target_=counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset,\
counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \
    gen_model=kde,real_nvp,nice

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\