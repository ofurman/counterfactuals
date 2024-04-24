python alternative_methods/wach_exp_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
counterfactuals.datasets.HelocDataset\
    disc_model=node


python alternative_methods/wach_exp_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,\
counterfactuals.datasets.DigitsDataset\
    disc_model=node

python alternative_methods/wach_exp_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset\
    disc_model=node \
    gen_model=large_maf

# node
# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,\
