import os

import hydra
import logging
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from alibi.explainers import Counterfactual
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from counterfactuals.generative_models.base import BaseGenModel
from counterfactuals.metrics.metrics import evaluate_cf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)


def generate_cf(dataset, disc_model):
    X_train, X_test, _, y_test = (
        dataset.X_train,
        dataset.X_test,
        dataset.y_train.reshape(-1),
        dataset.y_test.reshape(-1),
    )

    logger.info("Handling counterfactual generation")
    target_proba = 1.0
    tol = 0.49  # want counterfactuals with p(class)>0.99
    target_class = "other"  # any class other than origin will do
    max_iter = 1000
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (X_train.min(), X_train.max())

    Xs_cfs = []
    model_returned = []
    start_time = time()
    shape = (1,) + X_train.shape[1:]
    print(f"Shape: {shape}")

    def predict_proba(x):
        return disc_model.predict_proba(x).numpy()

    cf = Counterfactual(
        predict_proba,
        shape=shape,
        target_proba=target_proba,
        tol=tol,
        target_class=target_class,
        max_iter=max_iter,
        lam_init=lam_init,
        max_lam_steps=max_lam_steps,
        learning_rate_init=learning_rate_init,
        feature_range=feature_range,
    )
    for X, y in tqdm(zip(X_test, y_test), total=len(X_test)):
        # target_class = np.abs(y - 1).flatten().astype(int)[0]
        try:
            X = X.reshape((1,) + X.shape)
            explanation = cf.explain(X).cf
        except Exception as e:
            explanation = None
            print(e)
        if explanation is None:
            explanation = np.empty_like(X)
            explanation[:] = np.nan
            Xs_cfs.append(explanation)
            model_returned.append(False)
        else:
            Xs_cfs.append(explanation["X"])
            model_returned.append(True)

    cf_search_time = time() - start_time
    # run["metrics/avg_time_one_cf"] = (cf_search_time) / X_test.shape[0]
    # run["metrics/eval_time"] = np.mean(cf_search_time)

    Xs_cfs = np.array(Xs_cfs).squeeze()

    return model_returned, Xs_cfs, cf_search_time


@hydra.main(
    config_path="../conf/other_methods", config_name="config_wach", version_base="1.2"
)
def main(cfg: DictConfig):
    tf.compat.v1.disable_eager_execution()
    logger.info("Initializing Neptune run")
    # run = neptune.init_run(
    #     mode="async" if cfg.neptune.enable else "offline",
    #     project=cfg.neptune.project,
    #     api_token=cfg.neptune.api_token,
    #     tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    # )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    save_folder = os.path.join(output_folder, cfg.reference_method)
    os.makedirs(save_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    # run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    # run["parameters/disc_model/model_name"] = disc_model_name
    # run["parameters/disc_model"] = cfg.disc_model
    # run["parameters/gen_model/model_name"] = gen_model_name
    # run["parameters/gen_model"] = cfg.gen_model
    # # run["parameters/counterfactuals"] = cfg.counterfactuals
    # run["parameters/experiment"] = cfg.experiment
    # run["parameters/dataset"] = dataset_name
    # run["parameters/reference_method"] = "Artelt"
    # # run["parameters/pca_dim"] = cfg.pca_dim
    # run.wait()

    log_df = pd.DataFrame()

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    for fold_n, (_, _, _, _) in enumerate(dataset.get_cv_splits(n_splits=5)):
        tf.keras.backend.clear_session()
        disc_model_path = os.path.join(
            output_folder, f"disc_model_{fold_n}_{disc_model_name}.pt"
        )
        if cfg.experiment.relabel_with_disc_model:
            gen_model_path = os.path.join(
                output_folder,
                f"gen_model_{fold_n}_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
            )
        else:
            gen_model_path = os.path.join(
                output_folder, f"gen_model_{fold_n}_{gen_model_name}.pt"
            )

        logger.info("Loading discriminator model")
        num_classes = len(np.unique(dataset.y_train))
        disc_model = instantiate(
            cfg.disc_model.model,
            input_size=dataset.X_train.shape[1],
            target_size=2 if num_classes == 2 else num_classes,
        )
        disc_model.load(disc_model_path)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        logger.info("Loading generator model")
        gen_model: BaseGenModel = instantiate(
            cfg.gen_model.model, features=dataset.X_train.shape[1], context_features=1
        )
        gen_model.load(gen_model_path)

        model_returned, Xs_cfs, cf_search_time = generate_cf(dataset, disc_model)

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{disc_model_name}_{fold_n}.csv"
        )
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        # run["counterfactuals"].upload(counterfactuals_path)

        # Xs_cfs = pca.inverse_transform(Xs_cfs)
        train_dataloader_for_log_prob = dataset.train_dataloader(
            batch_size=cfg.counterfactuals.batch_size, shuffle=False
        )
        delta = torch.median(gen_model.predict_log_prob(train_dataloader_for_log_prob))
        # run["parameters/delta"] = delta
        print(delta)
        metrics = evaluate_cf(
            disc_model=disc_model,
            gen_model=gen_model,
            X_cf=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=dataset.X_test,
            y_test=dataset.y_test.reshape(-1),
            delta=delta,
        )
        # run["metrics/cf"] = metrics

        metrics["time"] = cf_search_time

        log_df = pd.concat([log_df, pd.DataFrame(metrics, index=[fold_n])])

        log_df.to_csv(
            os.path.join(save_folder, f"metrics_{disc_model_name}_cv.csv"), index=False
        )
        break

    # run.stop()


if __name__ == "__main__":
    main()
