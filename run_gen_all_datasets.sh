python generative_model_exp.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.CompasDataset,counterfactuals.datasets.LawDataset,counterfactuals.datasets.HelocDataset,counterfactuals.datasets.AdultDataset,counterfactuals.datasets.GermanCreditDataset,counterfactuals.datasets.MnistDataset \
    experiment.relabel_with_disc_model=true \
    disc_model._target_=sklearn.linear_model.LogisticRegression \
    gen_model=small_flow \
    gen_model.batch_size=128 \
    gen_model.epochs=200 \
    counterfactuals.epochs=200 \
    counterfactuals.lr=0.005 \
    counterfactuals.alpha=10 \
    counterfactuals.beta=0.01
