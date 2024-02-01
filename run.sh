python generative_model_exp.py --multirun \
    neptune.enable=false \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset \
    disc_model.model=LR,MLP \
    gen_model.model=FLOW,KDE
