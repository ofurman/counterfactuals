python run_train_and_experiment_cv.py --multirun \
    neptune.enable=true \
            dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset,\
counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \
    disc_model=mlp,mlr \
    gen_model=medium_maf
