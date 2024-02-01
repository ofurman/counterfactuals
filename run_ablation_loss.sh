python generative_model_exp.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset \
    disc_model.model=LR \
    counterfactuals.disc_loss._target_=counterfactuals.losses.BinaryDiscLoss,torch.nn.BCEWithLogitsLoss
