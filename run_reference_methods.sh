python alternative_methods/wach_exp_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
counterfactuals.datasets.HelocDataset,\
counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \
    disc_model=mlr

python alternative_methods/wach_exp_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset\
    disc_model=mlr \
    gen_model=large_maf