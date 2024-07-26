# python run_train_and_experiment.py --multirun \
#     dataset._target_=counterfactuals.datasets.ToyRegressionDataset,counterfactuals.datasets.YachtDataset,counterfactuals.datasets.ConcreteDataset,\
# counterfactuals.datasets.DiabetesDataset,counterfactuals.datasets.Scm20dDataset \
#     disc_model=linear_regression,nn_regression \
#     gen_model=large_maf


python alternative_methods/cearm.py --multirun \
    dataset._target_=counterfactuals.datasets.Scm20dDataset \
    disc_model=linear_regression,nn_regression \
    gen_model=large_maf

# python run_train_and_experiment_cv.py --multirun \
#     dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,\
# counterfactuals.datasets.DigitsDataset \
#     disc_model=mlp

# python run_train_and_experiment_cv.py --multirun \
#     dataset._target_=counterfactuals.datasets.AuditDataset\
#     disc_model=mlp \
#     gen_model=large_maf

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \