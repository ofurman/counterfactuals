import os

import hydra
import logging
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import GPyOpt
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from counterfactuals.generative_models.base import BaseGenModel
from counterfactuals.metrics.metrics import evaluate_regression_cf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)


def generate_cf(dataset, disc_model):
    _, X_test, _, _ = (
        dataset.X_train,
        dataset.X_test,
        dataset.y_train,
        dataset.y_test,
    )

    logger.info("Handling counterfactual generation")

    # Define the EP potential function
    def ep_potential(y, y_query, w=0.2):
        term = (y_query - y) / w
        z = np.maximum(term, 0)
        aep_p = z**2 * np.exp(-(z**2))
        # z = - np.minimum(term, 0)
        # aep_m = z ** 2 * np.exp(-(z**2))
        return aep_p
        # return (y - y_query)**2 * np.exp(-((y - y_query)**2) / w)

    # Objective function for Bayesian Optimization
    def objective_function(X_new, index):
        X_new = X_new.reshape(1, -1)
        y_query = disc_model.predict(
            torch.from_numpy(X_test[index].reshape(1, -1))
        ).numpy()  # Current prediction we want to change
        y_pred = disc_model.predict(X_new).numpy()
        potential = ep_potential(y_query, y_pred)
        return potential

    start_time = time()

    Xs_cfs = []
    model_returned = []

    for i in tqdm(range(len(X_test))):
        domain = [
            {"name": f"var_{j}", "type": "continuous", "domain": (0, 1)}
            for j in range(X_test.shape[1])
        ]
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=lambda X_new: objective_function(X_new, i),
            domain=domain,
            model_type="GP",
            acquisition_type="EI",
            normalize_Y=False,
            evaluator_type="thompson_sampling",
            maximize=True,
        )
        optimizer.run_optimization(eps=1e-3, max_iter=5)
        Xs_cfs.append(optimizer.x_opt.reshape(1, -1))
        model_returned.append(True)

    cf_search_time = time() - start_time
    # run["metrics/avg_time_one_cf"] = (cf_search_time) / X_test.shape[0]
    # run["metrics/eval_time"] = np.mean(cf_search_time)

    Xs_cfs = np.array(Xs_cfs, dtype=np.float32).squeeze()

    return model_returned, Xs_cfs, cf_search_time


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
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
    save_folder = os.path.join(output_folder, "CEARM")
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
    disc_model_path = os.path.join(output_folder, f"disc_model_{disc_model_name}.pt")
    if cfg.experiment.relabel_with_disc_model:
        gen_model_path = os.path.join(
            output_folder,
            f"gen_model_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
        )
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{gen_model_name}.pt")

    logger.info("Loading discriminator model")
    num_targets = dataset.y_train.shape[1] if len(dataset.y_train.shape) > 1 else 1
    disc_model = instantiate(
        cfg.disc_model.model,
        input_size=dataset.X_train.shape[1],
        target_size=num_targets,
    )
    disc_model.load(disc_model_path)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train).numpy()
        dataset.y_test = disc_model.predict(dataset.X_test).numpy()

    logger.info("Loading generator model")
    gen_model: BaseGenModel = instantiate(
        cfg.gen_model.model,
        features=dataset.X_train.shape[1],
        context_features=num_targets,
    )
    gen_model.load(gen_model_path)

    model_returned, Xs_cfs, cf_search_time = generate_cf(dataset, disc_model)

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    # run["counterfactuals"].upload(counterfactuals_path)

    # Xs_cfs = pca.inverse_transform(Xs_cfs)
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals.batch_size, shuffle=False
    )
    # delta = torch.median(gen_model.predict_log_prob(train_dataloader_for_log_prob))
    delta = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob), 0.25
    )
    # run["parameters/delta"] = delta
    print(delta)
    metrics = evaluate_regression_cf(
        disc_model=disc_model,
        gen_model=gen_model,
        X_cf=Xs_cfs,
        target_delta=0.2,
        model_returned=model_returned,
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        delta=delta,
    )
    print(metrics)
    # run["metrics/cf"] = metrics

    metrics["time"] = cf_search_time

    log_df = pd.DataFrame(metrics, index=[0])

    log_df.to_csv(
        os.path.join(output_folder, f"metrics_cearm_{disc_model_name}.csv"), index=False
    )

    # log_df.to_csv(
    #     os.path.join(save_folder, f"metrics_{disc_model_name}_cv.csv"), index=False
    # )

    # run.stop()


if __name__ == "__main__":
    main()
