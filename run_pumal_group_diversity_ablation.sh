
python3 counterfactuals/pipelines/run_pumal_pipeline.py -m \
    dataset._target_=counterfactuals.datasets.MoonsDataset \
    disc_model=mlp \
    counterfactuals_params.cf_method.K=10 \
    counterfactuals_params.alpha_plaus=10000 \
    counterfactuals_params.alpha_class=100000 \
    counterfactuals_params.alpha_s=10000 \
    counterfactuals_params.alpha_k=1000 \
    counterfactuals_params.alpha_d=0,0.1,10,100,1000
