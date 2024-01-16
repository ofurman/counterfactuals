import os
import hydra
import mlflow
import numpy as np
import torch
import neptune
import pandas as pd
from joblib import dump, load
from hydra.utils import instantiate
from nflows.flows import MaskedAutoregressiveFlow
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.datasets import (CompasDataset,  # AdultDataset,
                                      HelocDataset, LawDataset, MoonsDataset)
from counterfactuals.metrics.metrics import (categorical_distance,
                                             continuous_distance,
                                             distance_l2_jaccard,
                                             distance_mad_hamming,
                                             perc_valid_actionable_cf,
                                             perc_valid_cf, plausibility)
from counterfactuals.optimizers.approach_three import ApproachThree
from counterfactuals.utils import add_prefix_to_dict, process_classification_report


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
    ) 

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Log parameters using Hydra config
    run["parameters"] = add_prefix_to_dict(cfg.experiment, "experiment")
    run["parameters"] = add_prefix_to_dict(cfg.dataset, "dataset")
    run["parameters"] = add_prefix_to_dict(cfg.disc_model, "disc_model")
    run["parameters"] = add_prefix_to_dict(cfg.gen_model, "gen_model")

    dataset = instantiate(cfg.dataset)
    train_dataloader=dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True)
    test_dataloader=dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)

    disc_model = instantiate(cfg.disc_model)
    disc_model.fit(dataset.X_train, dataset.y_train.reshape(-1))
    report = classification_report(dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True)

    mlflow.log_metrics(process_classification_report(report, prefix="disc_test"))
    run["metrics"] = process_classification_report(report, prefix="disc_test")

    disc_model_path = os.path.join(output_folder, "disc_model.joblib")
    dump(disc_model, disc_model_path)
    run["disc_model"].upload(disc_model_path)
    
    pred_train_dataloader = DataLoader(
        dataset=TensorDataset(
            torch.from_numpy(dataset.X_train),
            torch.from_numpy(disc_model.predict(dataset.X_train))
        ),
        shuffle=True,
        batch_size=cfg.gen_model.batch_size
    )

    pred_test_dataloader = DataLoader(
        dataset=TensorDataset(
            torch.from_numpy(dataset.X_test),
            torch.from_numpy(disc_model.predict(dataset.X_test))
        ),
        shuffle=True,
        batch_size=cfg.gen_model.batch_size
    )

    X_test_pred_path = os.path.join(output_folder, "X_test_pred.csv")
    pd.DataFrame(disc_model.predict(dataset.X_test)).to_csv(X_test_pred_path, index=False)
    run["X_test_pred"].upload(X_test_pred_path)

    if cfg.experiment.relabel_with_disc_model:
        train_dataloader, test_dataloader = pred_train_dataloader, pred_test_dataloader

    if cfg.gen_model.checkpoint_path:
        flow = torch.load(cfg.gen_model.checkpoint_path)
        cf = ApproachThree(model=flow)
        gen_model_path = cfg.gen_model.checkpoint_path
    else:
        flow = MaskedAutoregressiveFlow(
            features=dataset.X_train.shape[1],
            hidden_features=cfg.gen_model.hidden_features,
            num_layers=cfg.gen_model.num_layers,
            num_blocks_per_layer=cfg.gen_model.num_blocks_per_layer,
            context_features=1
        )
        cf = ApproachThree(model=flow, neptune_run=run)
        cf.train_model(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=cfg.gen_model.epochs,
        )
        gen_model_path = os.path.join(output_folder, "gen_model.pt")
        torch.save(cf.model, gen_model_path)
    run["gen_model"].upload(gen_model_path)

    report = cf.test_model(test_loader=test_dataloader)
    run["metrics"] = process_classification_report(report, prefix="gen_test")

    Xs_cfs, Xs, ys_orig = cf.search_batch(
        dataloader=test_dataloader,
        epochs=cfg.counterfactuals.epochs,
        lr=cfg.counterfactuals.lr,
        alpha=cfg.counterfactuals.alpha,
        beta=cfg.counterfactuals.beta
    )
    pd.DataFrame(Xs_cfs).to_csv("data/counterfactuals.csv", index=False)
    run["counterfactuals"].upload("data/counterfactuals.csv")

    ys_cfs_gen_pred = cf.predict_model(Xs_cfs)
    ys_orig_gen_pred = cf.predict_model(Xs)

    ys_cfs_disc_pred = disc_model.predict(Xs_cfs)
    ys_orig_disc_pred = disc_model.predict(Xs)

    ys_orig = ys_orig.flatten()

    metrics = {
        "valid_cf_gen_orig": perc_valid_cf(ys_orig, y_cf=ys_cfs_gen_pred),
        "valid_cf_disc_orig": perc_valid_cf(ys_orig, y_cf=ys_cfs_disc_pred),
        "valid_cf_gen": perc_valid_cf(ys_orig_gen_pred, y_cf=ys_cfs_gen_pred),
        "valid_cf_disc": perc_valid_cf(ys_orig_disc_pred, y_cf=ys_cfs_disc_pred),
        # "perc_valid_actionable_cf": perc_valid_actionable_cf(X=dataset.X_test[:100], X_cf=Xs_cfs, y=ys_orig_pred, y_cf=ys_cfs_pred,
        #                                                      actionable_features=dataset.actionable_features),
        # "continuous_distance": continuous_distance(X=Xs, X_cf=Xs_cfs, 
        #                                            continuous_features=dataset.numerical_features, metric='mad', X_all=dataset.X_test),
        # "categorical_distance": categorical_distance(X=Xs, X_cf=Xs_cfs,
        #                                              categorical_features=dataset.categorical_features, metric='jaccard', agg='mean'),
        # "distance_l2_jaccard": distance_l2_jaccard(X=Xs, X_cf=Xs_cfs,
        #                                            continuous_features=dataset.numerical_features, categorical_features=dataset.categorical_features),
        # "distance_mad_hamming": distance_mad_hamming(X=Xs, X_cf=Xs_cfs,
        #                                             continuous_features=dataset.numerical_features, categorical_features=dataset.categorical_features, X_all=Xs, agg='mean'),
        # "plausibility": plausibility(
        #     Xs, Xs_cfs, ys_orig,
        #     continuous_features_all=dataset.numerical_features,
        #     categorical_features_all=dataset.categorical_features,
        #     X_train=dataset.X_train,
        #     ratio_cont=None
        # ),
        # "log_density": np.mean(cf.predict_model(test_dataloader))
    }
    run["metrics/counterfactuals"] = metrics

if __name__ == "__main__":
    main()

