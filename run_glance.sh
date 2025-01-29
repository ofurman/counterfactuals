########################
#### Moons Dataset #####
########################
python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=mlp \
    counterfactuals_params.cf_method.s=2

python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=mlr \
    counterfactuals_params.cf_method.s=2

########################
#### Law Dataset #######
########################
python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    disc_model=mlp \
    counterfactuals_params.cf_method.s=2

python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    disc_model=mlr \
    counterfactuals_params.cf_method.s=2


########################
#### Blobs Dataset #####
########################
python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    disc_model=mlp \
    counterfactuals_params.cf_method.s=2

python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    disc_model=mlr \
    counterfactuals_params.cf_method.s=2

########################
#### Wine Dataset #####
########################
python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    disc_model=mlp \
    gen_model=medium_maf \
    counterfactuals_params.cf_method.s=2 \
    counterfactuals_params.cf_method.k=5

python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    disc_model=mlr \
    gen_model=medium_maf \
    counterfactuals_params.cf_method.s=2 \
    counterfactuals_params.cf_method.k=5

########################
#### Heloc Dataset #####
########################
python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    disc_model=mlp \
    counterfactuals_params.cf_method.s=10


python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    disc_model=mlr \
    counterfactuals_params.cf_method.s=10


#########################
#### Digits Dataset #####
#########################
python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    disc_model=mlp_large \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003 \
    counterfactuals_params.cf_method.s=4

python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    disc_model=mlr \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003 \
    counterfactuals_params.cf_method.s=4

########################
#### Credit Default #####
########################
# python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
#     disc_model=mlp

# python3 counterfactuals/pipelines/run_glance_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
#     disc_model=mlr