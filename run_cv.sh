python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset \
    counterfactuals.origin_class=0,1 \
    disc_model=mlp \
    counterfactuals.delta._target_=counterfactuals.cf_methods.regional_ppcef.GLOBAL_CE,counterfactuals.cf_methods.regional_ppcef.PPCEF_2
    counterfactuals.cf_methods.regional_ppcef.GCE,\


python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset \
    disc_model=mlr,mlp \
    counterfactuals.origin_class=0,1,2 \
    counterfactuals.disc_loss._target_=counterfactuals.losses.MulticlassDiscLoss \
    counterfactuals.delta._target_=counterfactuals.cf_methods.regional_ppcef.GCE,\
counterfactuals.cf_methods.regional_ppcef.GLOBAL_CE,counterfactuals.cf_methods.regional_ppcef.PPCEF_2

# python run_train_and_experiment.py --multirun \
#     dataset._target_=counterfactuals.datasets.AuditDataset\
#     disc_model=lr,mlp \
#     gen_model=large_maf \
#     counterfactuals.origin_class=0,1

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \
# counterfactuals.cf_methods.regional_ppcef.GCE,counterfactuals.cf_methods.regional_ppcef.ARES,\
# counterfactuals.cf_methods.regional_ppcef.GLOBAL_CE,counterfactuals.cf_methods.regional_ppcef.PPCEF_2


python run_train_and_experiment_cv.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    counterfactuals.origin_class=1 \
    disc_model=mlp \
    counterfactuals.delta._target_=counterfactuals.cf_methods.regional_ppcef.GLOBAL_CE