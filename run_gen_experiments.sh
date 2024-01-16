python generative_model_exp.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.CompasDataset,counterfactuals.datasets.LawDataset \
    experiment.relabel_with_disc_model=true,false \
    disc_model._target_=sklearn.neural_network.MLPClassifier,sklearn.linear_model.LogisticRegression \
    gen_model.hidden_features=4 \
    gen_model.num_blocks_per_layer=2 \
    gen_model.num_layers=2 \
    gen_model.batch_size=64,128 \
    gen_model.epochs=50,100,200 \
    counterfactuals.epochs=100,200,500,1000 \
    counterfactuals.lr=0.001,0.005,0.01 \
    counterfactuals.alpha=1,5,10,20 \
    counterfactuals.beta=0.001,0.01,0.1

python generative_model_exp.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.CompasDataset,counterfactuals.datasets.LawDataset \
    experiment.relabel_with_disc_model=true,false \
    disc_model._target_=sklearn.neural_network.MLPClassifier,sklearn.linear_model.LogisticRegression \
    gen_model.hidden_features=4 \
    gen_model.num_blocks_per_layer=2 \
    gen_model.num_layers=5 \
    gen_model.batch_size=64,128 \
    gen_model.epochs=50,100,200 \
    counterfactuals.epochs=100,200,500,1000 \
    counterfactuals.lr=0.001,0.005,0.01 \
    counterfactuals.alpha=1,5,10,20 \
    counterfactuals.beta=0.001,0.01,0.1

python generative_model_exp.py --multirun \
    dataset._target_=counterfactuals.datasets.MoonsDataset,counterfactuals.datasets.CompasDataset,counterfactuals.datasets.LawDataset \
    experiment.relabel_with_disc_model=true,false \
    disc_model._target_=sklearn.neural_network.MLPClassifier,sklearn.linear_model.LogisticRegression \
    gen_model.hidden_features=16 \
    gen_model.num_blocks_per_layer=4 \
    gen_model.num_layers=8 \
    gen_model.batch_size=64,128 \
    gen_model.epochs=50,100,200 \
    counterfactuals.epochs=100,200,500,1000 \
    counterfactuals.lr=0.001,0.005,0.01 \
    counterfactuals.alpha=1,5,10,20 \
    counterfactuals.beta=0.001,0.01,0.1