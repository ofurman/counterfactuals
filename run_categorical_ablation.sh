python3 train_disc_model.py --multirun \
    neptune.enable=true \
    dataset._target_=counterfactuals.datasets.AdultDataset \
    disc_model=lr,mlp

python3 train_gen_model.py --multirun \
    neptune.enable=true \
    dataset._target_=counterfactuals.datasets.AdultDataset \
    gen_model=medium_maf

python run_experiment.py --multirun \
    neptune.enable=true \
    dataset._target_=counterfactuals.datasets.AdultDataset \
    disc_model=lr,mlp \
    gen_model=medium_maf
