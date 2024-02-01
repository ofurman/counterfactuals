python3 train_gen_model.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset \
    gen_model.model=FLOW,KDE \
    disc_model.model=null
