python3 train_disc_model.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,counterfactuals.datasets.HelocDataset \
    disc_model.model=LR,MLP
