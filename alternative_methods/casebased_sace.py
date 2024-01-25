import logging
import os
from time import time
from uuid import uuid4

import hydra
import neptune
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from hydra.utils import instantiate
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from tqdm import tqdm

from counterfactuals.discriminative_models import LogisticRegression, MultilayerPerceptron
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.optimizers.approach_gen_disc_loss import ApproachGenDiscLoss
from counterfactuals.sace.blackbox import BlackBox
from counterfactuals.sace.casebased_sace import CaseBasedSACE
from counterfactuals.utils import process_classification_report

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf/other_methods", config_name="config_cbce", version_base="1.2")
def main(cfg: DictConfig):
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Log parameters using Hydra config
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model"] = cfg.gen_model
    run["parameters/reference_method"] = "CBCE"

    available_disc_models = ["LR", "MLP"]
    if cfg.disc_model not in available_disc_models:
        raise ValueError(f"Disc model not supported. Please choose one of {available_disc_models}")

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train.reshape(-1), dataset.y_test.reshape(-1)
    train_dataloader = dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=1e-5)
    test_dataloader = dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)

    logger.info("Training discriminator model")
    disc_models = {
        "LR": LogisticRegression(X_train.shape[1], 1),
        "MLP": MultilayerPerceptron(layer_sizes=[X_train.shape[1], 128, 1]),
    }
    disc_model = disc_models[cfg.disc_model]
    disc_model.fit(train_dataloader)

    logger.info("Evaluating discriminator model")
    report = classification_report(dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True)
    run["metrics"] = process_classification_report(report, prefix="disc_test")

    disc_model_path = os.path.join(output_folder, f"disc_model_{uuid4()}.joblib")
    torch.save(disc_model, disc_model_path)
    run["disc_model"].upload(disc_model_path)

    X_test_pred_path = os.path.join(output_folder, "X_test_pred.csv")
    pd.DataFrame(disc_model.predict(dataset.X_test)).to_csv(X_test_pred_path, index=False)
    run["X_test_pred"].upload(X_test_pred_path)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)
        X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train.reshape(-1), dataset.y_test.reshape(-1)
        train_dataloader = dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=1e-5)
        test_dataloader = dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)
    else:
        train_dataloader = dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=1e-5)
        test_dataloader = dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)

    logger.info("Training generator model")
    if cfg.gen_model.checkpoint_path:
        flow = torch.load(cfg.gen_model.checkpoint_path)
        cf = ApproachGenDiscLoss(
            gen_model=flow,
            disc_model=disc_model,
            disc_model_criterion=torch.nn.BCELoss(),
            neptune_run=neptune
        )
        gen_model_path = cfg.gen_model.checkpoint_path
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{uuid4()}.pt")
        flow = MaskedAutoregressiveFlow(
            features=dataset.X_train.shape[1],
            hidden_features=cfg.gen_model.hidden_features,
            num_layers=cfg.gen_model.num_layers,
            num_blocks_per_layer=cfg.gen_model.num_blocks_per_layer,
            context_features=1,
        )
        cf_class = ApproachGenDiscLoss(
            gen_model=flow,
            disc_model=disc_model,
            disc_model_criterion=torch.nn.BCELoss(),
            neptune_run=run,
            checkpoint_path=gen_model_path
        )
        cf_class.train_model(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
        )
    run["gen_model"].upload(gen_model_path)

    logger.info("Evaluating generator model")
    report = cf_class.test_model(test_loader=test_dataloader)
    run["metrics"] = process_classification_report(report, prefix="gen_test")

    # Start CBCE Method
    variable_features = dataset.numerical_features + dataset.categorical_features
    metric = ('euclidean', 'jaccard')
        #         ('cosine', 'jaccard'),
        #         # ('euclidean', 'hamming')
    cf = CaseBasedSACE(
        variable_features=variable_features,
        weights=None,
        metric=metric,
        feature_names=None,
        continuous_features=dataset.numerical_features,
        categorical_features_lists=dataset.categorical_features_lists,
        normalize=False,
        random_samples=None,
        diff_features=5,
        tolerance=0.001,
    )
    bb = BlackBox(disc_model)
    cf.fit(bb, X_train)
    cf_time = time()

    Xs_cfs = []
    model_returned = []
    for x in tqdm(X_test):
        x_cf = cf.get_counterfactuals(x, k=1)
        Xs_cfs.append(x_cf)
        model_returned.append(True)

    run["metrics/avg_time_one_cf"] = time() - cf_time / len(X_test)

    Xs_cfs = np.array(Xs_cfs, dtype=np.float32).squeeze()
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    metrics = evaluate_cf(
        cf_class=cf_class,
        disc_model=disc_model,
        X=X_test,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print(metrics)
    run["metrics/cf"] = metrics

    run.stop()


if __name__ == "__main__":
    main()
