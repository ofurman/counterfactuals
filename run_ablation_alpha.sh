python run_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset \
    disc_model.model=LR,MLP \
    counterfactuals.alpha=1,2,5,10,100,1000
