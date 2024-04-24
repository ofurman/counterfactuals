python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
counterfactuals.datasets.HelocDataset\
    disc_model=mlp

python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,\
counterfactuals.datasets.DigitsDataset \
    disc_model=mlp

python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset\
    disc_model=mlp \
    gen_model=large_maf

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \