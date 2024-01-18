python generative_model_exp.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.CompasDataset,counterfactuals.datasets.LawDataset \
    experiment.relabel_with_disc_model=true,false \
    disc_model._target_=sklearn.neural_network.MLPClassifier,sklearn.linear_model.LogisticRegression \
    gen_model=small_flow,medium_flow,large_flow \
    gen_model.batch_size=64,128 \
    gen_model.epochs=500 \
    counterfactuals.epochs=200 \
    counterfactuals.lr=0.005 \
    counterfactuals.alpha=10 \
    counterfactuals.beta=0.01
