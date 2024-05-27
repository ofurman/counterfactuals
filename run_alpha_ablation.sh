python run_train_and_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset \
    counterfactuals.origin_class=0,1 \
    disc_model=lr,mlp \
    counterfactuals.delta._target_=counterfactuals.cf_methods.regional_ppcef.GCE \
    counterfactuals.alpha=1,10,100,1000 \
    counterfactuals.alpha_s=1,10,100,1000 \
    counterfactuals.alpha_k=1,10,100,1000

python run_train_and_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset \
    disc_model=mlr,mlp \
    counterfactuals.origin_class=0,1,2 \
    counterfactuals.disc_loss._target_=counterfactuals.losses.MulticlassDiscLoss \
    counterfactuals.delta._target_=counterfactuals.cf_methods.regional_ppcef.GCE \
    counterfactuals.alpha=1,10,100,1000 \
    counterfactuals.alpha_s=1,10,100,1000 \
    counterfactuals.alpha_k=1,10,100,1000

# python run_train_and_experiment.py --multirun \
#     dataset._target_=counterfactuals.datasets.AuditDataset\
#     disc_model=lr,mlp \
#     gen_model=large_maf \
#     counterfactuals.origin_class=0,1

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \