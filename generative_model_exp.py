import hydra
import mlflow
import numpy as np
import torch
import pandas as pd
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
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():

        # Log parameters using Hydra config
        mlflow.log_params(add_prefix_to_dict(cfg.dataset, "dataset"))
        mlflow.log_params(add_prefix_to_dict(cfg.disc_model, "disc_model"))
        mlflow.log_params(add_prefix_to_dict(cfg.gen_model, "gen_model"))

        dataset = instantiate(cfg.dataset)
        train_dataloader=dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True)
        test_dataloader=dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)
        

        if cfg.gen_model.checkpoint_path:
            flow = torch.load(cfg.gen_model.checkpoint_path)
            cf = ApproachThree(model=flow, mlflow=mlflow)
            model_path = cfg.gen_model.checkpoint_path
        else:
            flow = MaskedAutoregressiveFlow(features=dataset.X_train.shape[1], hidden_features=4, context_features=1)
            cf = ApproachThree(model=flow, mlflow=mlflow)
            cf.train_model(
                train_loader=train_dataloader,
                test_loader=test_dataloader,
                epochs=cfg.gen_model.epochs,
            )
            model_path = f"models/model_{mlflow.active_run().info.run_name}.pt"
            torch.save(cf.model, model_path)
        mlflow.log_artifact(model_path)

        report = cf.test_model(test_loader=test_dataloader)
        mlflow.log_metrics(process_classification_report(report, prefix="gen_test"))

        Xs_cfs, Xs, ys_orig = cf.search_batch(
            dataloader=test_dataloader,
            epochs=cfg.counterfactuals.epochs,
            lr=cfg.counterfactuals.lr,
            alpha=cfg.counterfactuals.alpha,
            beta=cfg.counterfactuals.beta
        )
        print(Xs_cfs)
        pd.DataFrame(Xs_cfs).to_csv("data/counterfactuals.csv", index=False)
        mlflow.log_artifact("data/counterfactuals.csv")

        # Xs_cfs = cf.generate_counterfactuals(
        #     Xs=dataset.X_test[:100],
        #     ys=dataset.y_test[:100],
        #     epochs=cfg.counterfactuals.epochs,
        #     lr=cfg.counterfactuals.lr,
        #     alpha=cfg.counterfactuals.alpha,
        #     beta=cfg.counterfactuals.beta
        # )
        # Xs_cfs = torch.concat(Xs_cfs).detach()

        ys_cfs_pred = cf.predict_model(Xs_cfs)
        ys_orig_pred = cf.predict_model(Xs)
        ys_orig = ys_orig.flatten()

        metrics = {
            "valid_cf": perc_valid_cf(ys_orig_pred, y_cf=ys_cfs_pred),
            "valid_cf_for_orig_data": perc_valid_cf(ys_orig, y_cf=ys_cfs_pred),
            # "perc_valid_actionable_cf": perc_valid_actionable_cf(X=dataset.X_test[:100], X_cf=Xs_cfs, y=ys_orig_pred, y_cf=ys_cfs_pred,
            #                                                      actionable_features=dataset.actionable_features),
            "continuous_distance": continuous_distance(X=Xs, X_cf=Xs_cfs, 
                                                       continuous_features=dataset.numerical_features, metric='mad', X_all=dataset.X_test),
            "categorical_distance": categorical_distance(X=Xs, X_cf=Xs_cfs,
                                                         categorical_features=dataset.categorical_features, metric='jaccard', agg='mean'),
            "distance_l2_jaccard": distance_l2_jaccard(X=Xs, X_cf=Xs_cfs,
                                                       continuous_features=dataset.numerical_features, categorical_features=dataset.categorical_features),
            "distance_mad_hamming": distance_mad_hamming(X=Xs, X_cf=Xs_cfs,
                                                        continuous_features=dataset.numerical_features, categorical_features=dataset.categorical_features, X_all=Xs, agg='mean'),
            "plausibility": plausibility(
                Xs, Xs_cfs, ys_orig,
                continuous_features_all=dataset.numerical_features,
                categorical_features_all=dataset.categorical_features,
                X_train=dataset.X_train,
                ratio_cont=None
            ),
            "log_density": np.mean(cf.predict_model(test_dataloader))
        }

        mlflow.log_metrics(add_prefix_to_dict(metrics, "counterfactuals"))

if __name__ == "__main__":
    main()

