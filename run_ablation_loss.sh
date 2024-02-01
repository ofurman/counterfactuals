python run_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset \
    disc_model.model=LR \
    counterfactuals.disc_loss._target_=counterfactuals.losses.BinaryDiscLoss,torch.nn.BCEWithLogitsLoss
