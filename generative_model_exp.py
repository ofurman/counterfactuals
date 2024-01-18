import hydra
import logging
import os
import numpy as np
import torch
import neptune
from uuid import uuid4
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
from counterfactuals.metrics.metrics import (
    categorical_distance,
    continuous_distance,
    distance_l2_jaccard,
    distance_mad_hamming,
    perc_valid_actionable_cf,
    perc_valid_cf,
    plausibility,
    kde_density,
    sparsity,
)
from counterfactuals.optimizers.approach_three import ApproachThree
from counterfactuals.utils import process_classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model"] = cfg.gen_model
    run["parameters/counterfactuals"] = cfg.counterfactuals

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    train_dataloader=dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=0)
    test_dataloader=dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)
    
    logger.info("Training discriminator model")
    disc_model = instantiate(cfg.disc_model)
    disc_model.fit(dataset.X_train, dataset.y_train.reshape(-1))

    logger.info("Evaluating discriminator model")
    report = classification_report(dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True)
    run["metrics"] = process_classification_report(report, prefix="disc_test")

    disc_model_path = os.path.join(output_folder, f"disc_model_{uuid4()}.joblib")
    dump(disc_model, disc_model_path)
    run["disc_model"].upload(disc_model_path)

    X_test_pred_path = os.path.join(output_folder, "X_test_pred.csv")
    pd.DataFrame(disc_model.predict(dataset.X_test)).to_csv(X_test_pred_path, index=False)
    run["X_test_pred"].upload(X_test_pred_path)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)
        train_dataloader=dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=1e-5)
        test_dataloader=dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)
    else:
        train_dataloader=dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=1e-5)
        test_dataloader=dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)

    logger.info("Training generator model")
    if cfg.gen_model.checkpoint_path:
        flow = torch.load(cfg.gen_model.checkpoint_path)
        cf = ApproachThree(model=flow)
        gen_model_path = cfg.gen_model.checkpoint_path
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{uuid4()}.pt")
        flow = MaskedAutoregressiveFlow(
            features=dataset.X_train.shape[1],
            hidden_features=cfg.gen_model.hidden_features,
            num_layers=cfg.gen_model.num_layers,
            num_blocks_per_layer=cfg.gen_model.num_blocks_per_layer,
            context_features=1
        )
        cf = ApproachThree(model=flow, neptune_run=run, checkpoint_path=gen_model_path)
        cf.train_model(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
        )
    run["gen_model"].upload(gen_model_path)

    logger.info("Evaluating generator model")
    report = cf.test_model(test_loader=test_dataloader)
    run["metrics"] = process_classification_report(report, prefix="gen_test")

    logger.info("Handling counterfactual generation")
    Xs_cfs, Xs, ys_orig = cf.search_batch(
        dataloader=test_dataloader,
        epochs=cfg.counterfactuals.epochs,
        lr=cfg.counterfactuals.lr,
        alpha=cfg.counterfactuals.alpha,
        beta=cfg.counterfactuals.beta
    )
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    logger.info("Calculating metrics")
    log_p_zeros, log_p_ones, ys_cfs_gen_pred = cf.predict_model(Xs_cfs)
    _, _, ys_orig_gen_pred = cf.predict_model(Xs)

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
        "dissimilarity_proximity_categorical_hamming": categorical_distance(X=Xs, X_cf=Xs_cfs, categorical_features=dataset.categorical_features, metric='hamming', agg='mean'),
        "dissimilarity_proximity_categorical_jaccard": categorical_distance(X=Xs, X_cf=Xs_cfs, categorical_features=dataset.categorical_features, metric='jaccard', agg='mean'),
        
        "dissimilarity_proximity_continuous_manhatan": continuous_distance(X=Xs, X_cf=Xs_cfs, continuous_features=dataset.numerical_features, metric='cityblock', X_all=dataset.X_test),
        "dissimilarity_proximity_continuous_euclidean": continuous_distance(X=Xs, X_cf=Xs_cfs, continuous_features=dataset.numerical_features, metric='euclidean', X_all=dataset.X_test),
        "dissimilarity_proximity_continuous_mad": continuous_distance(X=Xs, X_cf=Xs_cfs, continuous_features=dataset.numerical_features, metric='mad', X_all=dataset.X_test),

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

        "flow_log_density": np.mean(np.concatenate([log_p_zeros[ys_orig == 0], log_p_ones[ys_orig == 1]])),
        "kde_log_density": kde_density(dataset.X_train, dataset.y_train, Xs, Xs_cfs, ys_orig),

        "sparsity": sparsity(Xs, Xs_cfs),
    }
    run["metrics/cf"] = metrics

    logger.info("Finalizing and stopping run")
    run.stop()

if __name__ == "__main__":
    main()

