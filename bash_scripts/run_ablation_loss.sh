python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.AuditDataset \
    disc_model=lr \
    gen_model=large_maf \
    counterfactuals.disc_loss._target_=torch.nn.BCEWithLogitsLoss
