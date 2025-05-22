########################
#### Moons Dataset #####
########################
python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    counterfactuals_params.tau=0.9 \
    counterfactuals_params.rho=0.02 \
    disc_model=mlp

python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    counterfactuals_params.tau=0.9 \
    counterfactuals_params.rho=0.02 \
    disc_model=mlr

########################
#### Law Dataset #####
########################
python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    counterfactuals_params.tau=0.85 \
    counterfactuals_params.rho=0.05 \
    disc_model=mlp

python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    counterfactuals_params.tau=0.85 \
    counterfactuals_params.rho=0.05 \
    disc_model=mlr

########################
#### Blobs Dataset #####
########################
python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    counterfactuals_params.tau=0.9 \
    counterfactuals_params.rho=0.02 \
    disc_model=mlp

python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    counterfactuals_params.tau=0.9 \
    counterfactuals_params.rho=0.02 \
    disc_model=mlr

########################
#### Wine Dataset #####
########################
python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    counterfactuals_params.tau=0.8 \
    counterfactuals_params.rho=0.05 \
    counterfactuals_params.surrogate_tree_params.max_leaf_nodes=12 \
    disc_model=mlp \
    gen_model=medium_maf

python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    counterfactuals_params.tau=0.8 \
    counterfactuals_params.rho=0.05 \
    counterfactuals_params.surrogate_tree_params.max_leaf_nodes=12 \
    disc_model=mlr \
    gen_model=medium_maf

########################
#### Heloc Dataset #####
########################
python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    counterfactuals_params.tau=0.8 \
    counterfactuals_params.rho=0.01 \
    counterfactuals_params.surrogate_tree_params.max_leaf_nodes=16 \
    disc_model=mlp

python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    counterfactuals_params.tau=0.8 \
    counterfactuals_params.rho=0.01 \
    counterfactuals_params.surrogate_tree_params.max_leaf_nodes=16 \
    disc_model=mlr

#########################
#### Digits Dataset #####
#########################
python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    counterfactuals_params.tau=0.85 \
    counterfactuals_params.rho=0.03 \
    counterfactuals_params.surrogate_tree_params.max_leaf_nodes=20 \
    disc_model=mlp_large \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    counterfactuals_params.tau=0.85 \
    counterfactuals_params.rho=0.03 \
    counterfactuals_params.surrogate_tree_params.max_leaf_nodes=20 \
    disc_model=mlr \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

########################
#### Credit Default #####
########################
# python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
#     counterfactuals_params.tau=0.8 \
#     counterfactuals_params.rho=0.03 \
#     counterfactuals_params.surrogate_tree_params.max_leaf_nodes=16 \
#     disc_model=mlp

# python3 counterfactuals/pipelines/run_tcrex_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
#     counterfactuals_params.tau=0.8 \
#     counterfactuals_params.rho=0.03 \
#     counterfactuals_params.surrogate_tree_params.max_leaf_nodes=16 \
#     disc_model=mlr