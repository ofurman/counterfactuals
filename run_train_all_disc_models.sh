python3 train_disc_model.py --multirun \
    neptune.enable=false \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset \
    disc_model=lr,mlp
