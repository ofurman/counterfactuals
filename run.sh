python run_train_and_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
counterfactuals.datasets.HelocDataset \
    counterfactuals.origin_class=0,1 \
    disc_model=lr,mlp

python run_train_and_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,\
counterfactuals.datasets.DigitsDataset \
    disc_model=mlr,mlp \
    counterfactuals.origin_class=0,1,2 \
    counterfactuals.disc_loss._target_=counterfactuals.losses.MulticlassDiscLoss

python run_train_and_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset\
    disc_model=lr,mlp \
    gen_model=large_maf \
    counterfactuals.origin_class=0,1

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \