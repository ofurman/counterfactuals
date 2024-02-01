import logging
import os
from uuid import uuid4
import json

import hydra
import neptune
import numpy as np
import pandas as pd
import torch
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from nflows.flows import MaskedAutoregressiveFlow, SimpleRealNVP
from counterfactuals.generative_models.cnf import ContinuousNormalizingFlowRegressor
from omegaconf import DictConfig
from sklearn.metrics import classification_report

from counterfactuals.discriminative_models import LogisticRegression, MultilayerPerceptron

from counterfactuals.optimizers.ppcef import PPCEF
from counterfactuals.generative_models.kde import KDE
from counterfactuals.utils import process_classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config_train_gen_model", version_base="1.2")
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
    os.makedirs("results/model_train/", exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    run["parameters/disc_model"] = stringify_unsupported(cfg.disc_model)
    run["parameters/gen_model"] = stringify_unsupported(cfg.gen_model)
    run.wait()

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    logger.info("Loading discriminator model")
    if cfg.disc_model.model in ["LR", "MLP"]:
        disc_model_path = os.path.join(output_folder, f"disc_model_{cfg.disc_model.model}_{run['parameters/dataset'].fetch()}.pt")
        disc_model = torch.load(disc_model_path)
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)
        train_dataloader = dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=cfg.gen_model.noise_lvl)
        test_dataloader = dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)
        gen_model_path = os.path.join(output_folder, f"gen_model_{cfg.gen_model.model}_relabeled_{cfg.disc_model.model}_{run['parameters/dataset'].fetch()}.pt")
    else:
        disc_model = None
        gen_model_path = os.path.join(output_folder, f"gen_model_{cfg.gen_model.model}_orig_{run['parameters/dataset'].fetch()}.pt")

    logger.info("Training generator model")
    train_dataloader = dataset.train_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=True, noise_lvl=cfg.gen_model.noise_lvl)
    test_dataloader = dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)
    gen_model_path = os.path.join(output_folder, f"gen_model_{cfg.gen_model.model}_orig_{run['parameters/dataset'].fetch()}.pt")
    if cfg.gen_model.model == "FLOW":
        gen_model = MaskedAutoregressiveFlow(
            features=dataset.X_train.shape[1],
            hidden_features=cfg.gen_model.hidden_features,
            num_layers=cfg.gen_model.num_layers,
            num_blocks_per_layer=cfg.gen_model.num_blocks_per_layer,
            context_features=1,
        )
        cf = PPCEF(
            gen_model=gen_model,
            disc_model=disc_model,
            disc_model_criterion=torch.nn.BCELoss(),
            neptune_run=run,
            checkpoint_path=gen_model_path
        )
        cf.train_model(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
            lr=cfg.gen_model.lr,
        )
    elif cfg.gen_model.model == "KDE":
        gen_model = KDE(bandwidth=0.1)
        gen_model.fit(train_dataloader)
        torch.save(gen_model, gen_model_path)
        cf = PPCEF(
            gen_model=gen_model,
            disc_model=disc_model,
            disc_model_criterion=torch.nn.BCELoss(),
            neptune_run=run,
            checkpoint_path=gen_model_path
        )
    run["gen_model"].upload(gen_model_path)

    logger.info("Evaluating generator model")
    report = cf.test_model(test_loader=test_dataloader)
    print(report)
    run["metrics"] = process_classification_report(report, prefix="gen_test_orig")
    results_path = os.path.join("results/model_train/", f"results_flow_orig_{run['parameters/dataset'].fetch()}.json")
    with open(results_path, "w") as f:
        json.dump({k: str(v) for k, v in process_classification_report(report).items()}, f)
    run.stop()


if __name__ == "__main__":
    main()
