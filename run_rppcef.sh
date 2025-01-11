########################
#### Moons Dataset #####
########################
python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=mlp

python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=mlr

########################
#### Law Dataset #####
########################
python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    disc_model=mlp

python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    disc_model=mlr


########################
#### Blobs Dataset #####
########################
python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    disc_model=mlp

python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    disc_model=mlr

########################
#### Wine Dataset #####
########################
python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    disc_model=mlp \
    gen_model=medium_maf

python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    disc_model=mlr \
    gen_model=medium_maf

########################
#### Heloc Dataset #####
########################
python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    disc_model=mlp \
    cf_method.K=10 \
    cf_method.alpha_plaus=100 \
    cf_method.alpha_class=10000 \
    cf_method.alpha_s=1000 \
    cf_method.alpha_k=100 \
    cf_method.alpha_d=10


python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    disc_model=mlr \
    counterfactuals_params.cf_method.K=10 \
    counterfactuals_params.alpha_plaus=100 \
    counterfactuals_params.alpha_class=10000 \
    counterfactuals_params.alpha_s=1000 \
    counterfactuals_params.alpha_k=100 \
    counterfactuals_params.alpha_d=10


#########################
#### Digits Dataset #####
#########################
python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    disc_model=mlp_large \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    disc_model=mlr \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

########################
#### Credit Default #####
########################
python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
    disc_model=mlp

python3 counterfactuals/pipelines/run_rppcef_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
    disc_model=mlr