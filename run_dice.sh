########################
#### Moons Dataset #####
########################
python3 counterfactuals/pipelines/run_dice_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=tabnet 
 

########################
#### Law Dataset #######
########################
python3 counterfactuals/pipelines/run_dice_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.LawDataset \
    disc_model=tabnet 
 


########################
#### Blobs Dataset #####
########################
python3 counterfactuals/pipelines/run_dice_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.BlobsDataset \
    disc_model=tabnet 
 

########################
#### Wine Dataset #####
########################
python3 counterfactuals/pipelines/run_dice_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.WineDataset \
    disc_model=tabnet \
    gen_model=medium_maf

########################
#### Heloc Dataset #####
########################
python3 counterfactuals/pipelines/run_dice_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.HelocDataset \
    disc_model=tabnet 

#########################
#### Digits Dataset #####
#########################
python3 counterfactuals/pipelines/run_dice_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.DigitsDataset \
    disc_model=mlp_large \
    gen_model=digits_maf \
    gen_model.noise_lvl=0.003

########################
#### Credit Default #####
########################
# python3 counterfactuals/pipelines/run_dice_pipeline.py -m \
#     dataset._target_=counterfactuals.datasets.CreditDefaultDataset \
#     disc_model=mlp

# python3 counterfactuals/pipelines/run_dice_pipeline.py -m \