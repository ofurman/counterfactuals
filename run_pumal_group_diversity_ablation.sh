
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=mlp \
    counterfactuals_params.alpha_d=0,0.1,10,100,1000
