import hydra
import mlflow
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


from nflows.flows import MaskedAutoregressiveFlow
from counterfactuals.datasets import (
    CompasDataset,
    # AdultDataset,
    HelocDataset,
    LawDataset,
    MoonsDataset
)
from counterfactuals.metrics.metrics import (categorical_distance,
                                             continuous_distance,
                                             distance_l2_jaccard,
                                             distance_mad_hamming,
                                             perc_valid_actionable_cf,
                                             perc_valid_cf, plausibility)
from counterfactuals.optimizers.approach_gen_disc import ApproachGenDisc
from counterfactuals.utils import (add_prefix_to_dict,
                                   process_classification_report)


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

        
        disc_model = instantiate(cfg.disc_model)
        disc_model.fit(dataset.X_train, dataset.y_train)
        report = classification_report(dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True)
        mlflow.log_metrics(process_classification_report(report, prefix="disc_test"))

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

        
        if cfg.gen_model.checkpoint_path:
            flow = torch.load(cfg.gen_model.checkpoint_path)
            cf = ApproachGenDisc(model=flow, disc_model=disc_model, mlflow=mlflow)
            model_path = cfg.gen_model.checkpoint_path
        else:
            flow = MaskedAutoregressiveFlow(features=dataset.X_train.shape[1], hidden_features=4, context_features=1)
            cf = ApproachGenDisc(model=flow, disc_model=disc_model, mlflow=mlflow)
            cf.train_model(
                train_loader=pred_train_dataloader,
                test_loader=pred_test_dataloader,
                epochs=cfg.gen_model.epochs,
            )
            model_path = f"models/model_{mlflow.active_run().info.run_name}.pt"
            torch.save(cf.model, model_path)
        mlflow.log_artifact(model_path)

        # report = cf.test_model(test_loader=pred_test_dataloader)
        # mlflow.log_metrics(process_classification_report(report, prefix="gen_test"))

        Xs_cfs, Xs, ys_orig = cf.search_batch(
            dataloader=pred_test_dataloader,
            epochs=cfg.counterfactuals.epochs,
            lr=cfg.counterfactuals.lr,
            alpha=cfg.counterfactuals.alpha,
            beta=cfg.counterfactuals.beta
        )

        # Xs_cfs = cf.generate_counterfactuals(
        #     Xs=dataset.X_test,
        #     ys=dataset.y_test,
        #     epochs=cfg.counterfactuals.epochs,
        #     lr=cfg.counterfactuals.lr,
        #     alpha=cfg.counterfactuals.alpha,
        #     beta=cfg.counterfactuals.beta
        # )
        # Xs_cfs = torch.concat(Xs_cfs).detach()
        pd.DataFrame(Xs_cfs).to_csv("data/counterfactuals.csv", index=False)
        mlflow.log_artifact("data/counterfactuals.csv")

        ys_cfs_pred = disc_model.predict(Xs_cfs)
        ys_orig_pred = disc_model.predict(Xs)
        ys_orig = ys_orig.flatten()

        metrics = {
            "valid_cf": perc_valid_cf(ys_orig_pred, y_cf=ys_cfs_pred),
            "valid_cf_for_orig_data": perc_valid_cf(ys_orig, y_cf=ys_cfs_pred),
            # "perc_valid_actionable_cf": perc_valid_actionable_cf(X=dataset.X_test[:100], X_cf=Xs_cfs, y=ys_orig_pred, y_cf=ys_cfs_pred,
            #                                                      actionable_features=dataset.actionable_features),
            "disc_prob_delta": np.mean(np.abs(disc_model.predict_proba(Xs)[:, 1] - disc_model.predict_proba(Xs_cfs)[:, 1])),
            "continuous_distance": continuous_distance(X=Xs, X_cf=Xs_cfs, 
                                                       continuous_features=dataset.numerical_features, metric='mad', X_all=dataset.X_test),
            "categorical_distance": categorical_distance(X=Xs, X_cf=Xs_cfs,
                                                         categorical_features=dataset.categorical_features, metric='jaccard', agg='mean'),
            "distance_l2_jaccard": distance_l2_jaccard(X=Xs, X_cf=Xs_cfs,
                                                       continuous_features=dataset.numerical_features, categorical_features=dataset.categorical_features),
            "distance_mad_hamming": distance_mad_hamming(X=Xs, X_cf=Xs_cfs,
                                                        continuous_features=dataset.numerical_features, categorical_features=dataset.categorical_features, X_all=dataset.X_train, agg='mean'),
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

