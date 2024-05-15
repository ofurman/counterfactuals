python run_train_and_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset \
    counterfactuals.origin_class=0,1 \
    disc_model=lr,mlp \
    counterfactuals.K=1,2,4,6,8,10,20,30,40,50,60,70,80,90,100

python run_train_and_experiment.py --multirun \
    dataset._target_=counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset \
    disc_model=mlr,mlp \
    counterfactuals.origin_class=0,1,2 \
    counterfactuals.disc_loss._target_=counterfactuals.losses.MulticlassDiscLoss \
    counterfactuals.K=1,2,4,6,8,10,20,30,40,50,60,70,80,90,100

# python run_train_and_experiment.py --multirun \
#     dataset._target_=counterfactuals.datasets.AuditDataset\
#     disc_model=lr,mlp \
#     gen_model=large_maf \
#     counterfactuals.origin_class=0,1

# counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.LawDataset,\
# counterfactuals.datasets.HelocDataset,\
# counterfactuals.datasets.BlobsDataset,counterfactuals.datasets.WineDataset,counterfactuals.datasets.DigitsDataset \