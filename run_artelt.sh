########################
#### Moons Dataset #####
########################
python3 counterfactuals/pipelines/run_artelt_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=mlr

########################
#### Law Dataset #####
########################
python3 counterfactuals/pipelines/run_artelt_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    disc_model=mlr


########################
#### Blobs Dataset #####
########################
python3 counterfactuals/pipelines/run_artelt_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    disc_model=mlr

########################
#### Wine Dataset #####
########################
python3 counterfactuals/pipelines/run_artelt_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    disc_model=mlr \
    gen_model=medium_maf

########################
#### Heloc Dataset #####
########################
python3 counterfactuals/pipelines/run_artelt_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    disc_model=mlr


#########################
#### Digits Dataset #####
#########################
python3 counterfactuals/pipelines/run_artelt_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    disc_model=mlr \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

########################
#### Credit Default #####
########################
# python3 counterfactuals/pipelines/run_artelt_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
#     disc_model=mlr