import hydra
import os
import torch
import logging
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import neptune
from uuid import uuid4
from hydra.utils import instantiate
from tqdm import tqdm

from omegaconf import DictConfig
from sklearn.metrics import classification_report
from alibi.explainers import Counterfactual

from counterfactuals.optimizers.approach_gen_disc_loss import ApproachGenDiscLoss
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from counterfactuals.discriminative_models import LogisticRegression, MultilayerPerceptron
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.utils import process_classification_report

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf/other_methods", config_name="config_wach", version_base="1.2")
def main(cfg: DictConfig):
    tf.compat.v1.disable_eager_execution()

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
    run["parameters/reference_method"] = "Artelt"

    available_disc_models = ["LR", "MLP"]
    if cfg.disc_model not in available_disc_models:
        raise ValueError(f"Disc model not supported. Please choose one of {available_disc_models}")
    use_decision_tree = cfg.disc_model == "DecisionTreeClassifier"

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


    target_proba = 1.0
    tol = 0.01 # want counterfactuals with p(class)>0.99
    target_class = 'other' # any class other than origin will do
    max_iter = 1000
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (X_train.min(),X_train.max())

    Xs_cfs = []
    model_returned = []
    start_time = time()
    for X, y in tqdm(zip(X_test, y_test), total=len(X_test)):
        target_class = np.abs(y - 1).flatten().astype(int)[0]
        X = X.reshape((1,) + X.shape)
        shape = (1,) + X_train.shape[1:]
        cf = Counterfactual(disc_model.predict_proba, shape=shape, target_proba=target_proba, tol=tol,
                            target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                            max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                            feature_range=feature_range)
        explanation = cf.explain(X)
        if explanation.cf is None:
            model_returned.append(False)
        else:
            Xs_cfs.append(explanation.cf['X'])
            model_returned.append(True)
    run["metrics/avg_time_one_cf"] = (time() - start_time) / X_test.shape[0]

    Xs_cfs = np.array(Xs_cfs).squeeze()
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
        y_train=y_train.reshape(-1),
        X_test=X_test,
        y_test=y_test.reshape(-1),
    )
    run["metrics/cf"] = metrics

    run.stop()

if __name__ == "__main__":
    main()

