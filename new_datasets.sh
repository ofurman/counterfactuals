# python counterfactuals/pipelines/run_ppcef_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.LendingClubDataset,counterfactuals.datasets.BankMarketingDataset,counterfactuals.datasets.GiveMeSomeCreditDataset,counterfactuals.datasets.AdultCensusDataset \
#     disc_model=mlp,mlr

python counterfactuals/pipelines/run_model_training.py -m \
    dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
    disc_model=mlp,mlr,node

# python counterfactuals/pipelines/run_cchvae_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.AdultCensusDataset \
#     disc_model=mlp,mlr

#python counterfactuals/pipelines/run_cchvae_pipeline.py -m \
#    dataset._target_=counterfactuals.datasets.LendingClubDataset,counterfactuals.datasets.BankMarketingDataset \
#    disc_model=mlp,mlr


# python counterfactuals/pipelines/run_ppcef_pipeline.py -m \
#      dataset._target_=counterfactuals.datasets.LawDataset,counterfactuals.datasets.GermanCreditDataset,counterfactuals.datasets.AdultDataset \
#      disc_model=mlp,mlr

# python counterfactuals/pipelines/run_cchvae_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.LawDataset,counterfactuals.datasets.GermanCreditDataset,counterfactuals.datasets.AdultDataset \
#     disc_model=mlp,mlr