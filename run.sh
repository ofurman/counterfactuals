python run_experiment.py --multirun \
    neptune.enable=false \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AuditDataset \
    disc_model=lr,mlp \
    gen_model=medium_maf,kde
