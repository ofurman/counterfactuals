import logging
import os
import json
import hydra
import neptune
from neptune.utils import stringify_unsupported
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.optimizers.approach_gen_disc_loss import ApproachGenDiscLoss
from counterfactuals.utils import load_model

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

    models_folder = cfg.experiment.models_folder
    os.makedirs(models_folder, exist_ok=True)
    logger.info("Creatied output folder %s", models_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    run["parameters/disc_model/model"] = cfg.disc_model.model
    # run["parameters/gen_model"] = cfg.gen_model
    run["parameters/counterfactuals"] = cfg.counterfactuals
    run.wait()

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    logger.info("Loading discriminator model")
    print(cfg.disc_model.model)
    if cfg.disc_model.model in ["LR", "MLP", "DTC"]:
        if cfg.disc_model.model == "DTC":
            disc_model_path = os.path.join(models_folder, f"disc_model_{cfg.disc_model.model}_{run['parameters/dataset'].fetch()}.joblib")
        else:
            disc_model_path = os.path.join(models_folder, f"disc_model_{cfg.disc_model.model}_{run['parameters/dataset'].fetch()}.pt")
        disc_model = load_model(disc_model_path)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)
            gen_model_path = os.path.join(models_folder, f"gen_model_relabeled_{cfg.disc_model.model}_{run['parameters/dataset'].fetch()}.pt")
        else:
            gen_model_path = os.path.join(models_folder, f"gen_model_orig_{run['parameters/dataset'].fetch()}.pt")
        
    elif cfg.disc_model.model == "FLOW":
        disc_model=None
        disc_model_path = os.path.join(models_folder, f"gen_model_orig_{run['parameters/dataset'].fetch()}.pt")
        flow = load_model(disc_model_path)
        disc_model_flow = ApproachGenDiscLoss(
            gen_model=flow,
            disc_model=None,
            disc_model_criterion=torch.nn.BCELoss(),
            neptune_run=run,
            checkpoint_path=disc_model_path
        )
        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model_flow.predict(dataset.X_train)
            dataset.y_test = disc_model_flow.predict(dataset.X_test)

        gen_model_path = os.path.join(models_folder, f"gen_model_orig_{run['parameters/dataset'].fetch()}.pt")

    logger.info("Loading generator model")
    flow = load_model(gen_model_path)
    cf = ApproachGenDiscLoss(
        gen_model=flow,
        disc_model=disc_model,
        disc_model_criterion=instantiate(cfg.counterfactuals.disc_loss),
        neptune_run=run,
        checkpoint_path=gen_model_path
    )

    logger.info("Handling counterfactual generation")
    train_dataloader_for_log_prob = dataset.train_dataloader(batch_size=cfg.counterfactuals.batch_size, shuffle=False)
    delta = cf.calculate_median_log_prob(train_dataloader_for_log_prob)
    run["parameters/delta"] = delta
    print(delta)

    test_dataloader = dataset.test_dataloader(batch_size=cfg.counterfactuals.batch_size, shuffle=False)
    Xs_cfs, Xs, ys_orig = cf.search_batch(
        dataloader=test_dataloader,
        epochs=cfg.counterfactuals.epochs,
        lr=cfg.counterfactuals.lr,
        alpha=cfg.counterfactuals.alpha,
        beta=cfg.counterfactuals.beta,
        delta=delta
    )
    counterfactuals_path = os.path.join(models_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)

    logger.info("Calculating metrics")
    ys_orig = ys_orig.flatten()

    metrics = evaluate_cf(
        cf_class=cf,
        disc_model=disc_model if disc_model else disc_model_flow,
        X=Xs,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        delta=delta
    )
    run["metrics/cf"] = stringify_unsupported(metrics)
    results_path = os.path.join("results/", f"results_{cfg.disc_model.model}_{run['parameters/dataset'].fetch()}.json")
    # write results dict to json file using json
    with open(results_path, "w") as f:
        json.dump({k: str(v) for k, v in metrics.items()}, f)


    logger.info("Finalizing and stopping run")
    run.stop()


if __name__ == "__main__":
    main()
