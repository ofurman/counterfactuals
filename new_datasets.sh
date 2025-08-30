# python counterfactuals/pipelines/run_ppcef_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.LendingClubDataset,counterfactuals.datasets.BankMarketingDataset,counterfactuals.datasets.GiveMeSomeCreditDataset \
#     disc_model=mlp,mlr

python counterfactuals/pipelines/run_cchvae_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LendingClubDataset,counterfactuals.datasets.BankMarketingDataset \
    disc_model=mlp,mlr
