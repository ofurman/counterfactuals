import logging
import os
import json
import hydra
import neptune
from neptune.utils import stringify_unsupported
import numpy as np
import pandas as pd
from time import time
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.cf_methods.ppcef import PPCEF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    disc_model_path = os.path.join(output_folder, f"disc_model_{disc_model_name}.pt")
    if cfg.experiment.relabel_with_disc_model:
        gen_model_path = os.path.join(output_folder, f"gen_model_{gen_model_name}_relabeled_by_{disc_model_name}.pt")
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{gen_model_name}.pt")
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    run["parameters/disc_model/model_name"] = disc_model_name
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model/model_name"] = gen_model_name
    run["parameters/gen_model"] = cfg.gen_model
    run["parameters/counterfactuals"] = cfg.counterfactuals
    run.wait()

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    logger.info("Loading discriminator model")
    disc_model = instantiate(cfg.disc_model.model, input_size=dataset.X_train.shape[1], target_size=1)
    disc_model.load(disc_model_path)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    logger.info("Loading generator model")
    gen_model = instantiate(cfg.gen_model.model, features=dataset.X_train.shape[1], context_features=1)
    gen_model.load(gen_model_path)
    cf = PPCEF(
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=instantiate(cfg.counterfactuals.disc_loss),
        neptune_run=run,
    )

    logger.info("Handling counterfactual generation")
    train_dataloader_for_log_prob = dataset.train_dataloader(batch_size=cfg.counterfactuals.batch_size, shuffle=False)
    delta = torch.median(gen_model.predict_log_prob(train_dataloader_for_log_prob))
    run["parameters/delta"] = delta
    print(delta)

    test_dataloader = dataset.test_dataloader(batch_size=cfg.counterfactuals.batch_size, shuffle=False)
    time_start = time()
    Xs_cfs, Xs, ys_orig = cf.search_batch(
        dataloader=test_dataloader,
        epochs=cfg.counterfactuals.epochs,
        lr=cfg.counterfactuals.lr,
        patience=cfg.counterfactuals.patience,
        alpha=cfg.counterfactuals.alpha,
        beta=cfg.counterfactuals.beta,
        delta=delta
    )
    run["metrics/eval_time"] = np.mean(time() - time_start)
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)

    logger.info("Calculating metrics")
    ys_orig = ys_orig.flatten()

    metrics = evaluate_cf(
        gen_model=gen_model,
        disc_model=disc_model,
        X=Xs,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        delta=delta.numpy()
    )
    run["metrics/cf"] = stringify_unsupported(metrics)
    logger.info("Finalizing and stopping run")
    run.stop()


if __name__ == "__main__":
    main()
