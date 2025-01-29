########################
#### Moons Dataset #####
########################
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlp

python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlr

########################
#### Law Dataset #####
########################
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlp

python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlr


########################
#### Blobs Dataset #####
########################
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlp

python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlr

########################
#### Wine Dataset #####
########################
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlp \
    gen_model=medium_maf

python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlr \
    gen_model=medium_maf

########################
#### Heloc Dataset #####
########################
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlp \
    counterfactuals_params.alpha_plaus=100 \
    counterfactuals_params.alpha_class=10000 \
    counterfactuals_params.alpha_s=1000


python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlr \
    counterfactuals_params.alpha_plaus=100 \
    counterfactuals_params.alpha_class=10000 \
    counterfactuals_params.alpha_s=1000


#########################
#### Digits Dataset #####
#########################
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlp_large \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
    counterfactuals_params.cf_method.K=1 \
    counterfactuals_params.alpha_k=0 \
    counterfactuals_params.alpha_d=0 \
    disc_model=mlr \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

########################
#### Credit Default #####
########################
# python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
# counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
#     counterfactuals_params.cf_method.K=1 \
#     counterfactuals_params.alpha_k=0 \
#     counterfactuals_params.alpha_d=0 \
#     disc_model=mlp

# python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
# counterfactuals_params.cf_method.cf_method_type="GLOBAL_CE" \
#     counterfactuals_params.cf_method.K=1 \
#     counterfactuals_params.alpha_k=0 \
#     counterfactuals_params.alpha_d=0 \
#     disc_model=mlr