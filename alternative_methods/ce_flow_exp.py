import hydra
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import timeit
import numpy as np
import neptune
from hydra.utils import instantiate

from omegaconf import DictConfig

from counterfactuals.generative_models.base import BaseGenModel
from counterfactuals.cf_methods.ce_flow.src.counterfactual_explanation.flow_ce.flow_method import (
    FindCounterfactualSample,
    CounterfactualSimpleBn,
    CounterfactualAdult,
)

from counterfactuals.cf_methods.ce_flow.src.counterfactual_explanation.utils.mlcatalog import (
    negative_prediction_index,
    prediction_instances,
    positive_prediction_index,
    find_latent_mean_two_classes,
)

from counterfactuals.cf_methods.ce_flow.src.counterfactual_explanation.utils.mlcatalog import (
    save_pytorch_model_to_model_path,
)

from counterfactuals.cf_methods.ce_flow.src.counterfactual_explanation.flow_ssl.realnvp.realnvp import (
    RealNVPTabular,
)
from counterfactuals.cf_methods.ce_flow.src.counterfactual_explanation.flow_ssl import (
    FlowLoss,
)
from counterfactuals.cf_methods.ce_flow.src.counterfactual_explanation.flow_ssl.distributions import (
    SSLGaussMixture,
)

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def train_flow(input_features, means, train_loader):
    LR_INIT = 1e-2
    EPOCHS = 10
    BATCH_SIZE = 128
    PRINT_FREQ = 2
    MEAN_VALUE = 0.5
    prior = SSLGaussMixture(means=means)
    # deq = DequantizationOriginal()
    flow = RealNVPTabular(
        num_coupling_layers=3, in_dim=input_features, num_layers=5, hidden_dim=8
    )
    loss_fn = FlowLoss(prior)
    # optimizer = torch.optim.Adam(flow.parameters(), lr=LR_INIT, weight_decay=1e-3)
    optimizer = torch.optim.SGD(flow.parameters(), lr=LR_INIT, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # sldj = torch.zeros(batch_size)

    sldj_deq = torch.zeros(
        1,
    )

    cur_lr = scheduler.optimizer.param_groups[0]["lr"]

    print("Learning rate ", cur_lr)

    for t in range(EPOCHS):
        for local_batch, local_labels in train_loader:
            local_z = flow(local_batch)
            z = flow(local_batch)
            sldj = flow.logdet()
            flow_loss = loss_fn(local_z, sldj, local_labels)
            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        scheduler.step()

        if t % PRINT_FREQ == 0:
            print(
                "iter %s:" % t, "loss = %.3f" % flow_loss, "learning rate: %s" % cur_lr
            )

    return flow


@hydra.main(
    config_path="../conf/other_methods",
    config_name="config_ce_flow",
    version_base="1.2",
)
def main(cfg: DictConfig):
    DATA_NAME = cfg.data_name
    weight = cfg.weight

    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    run["parameters/disc_model/model_name"] = disc_model_name
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model/model_name"] = gen_model_name
    run["parameters/gen_model"] = cfg.gen_model
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = dataset_name
    run["parameters/reference_method"] = "WACH"
    run.wait()

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
    num_features = len(np.unique(dataset.y_train))
    disc_model = instantiate(
        cfg.disc_model.model,
        input_size=dataset.X_train.shape[1],
        target_size=1 if num_features == 2 else num_features,
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

    X_train, X_test, y_train, y_test = (
        dataset.X_train,
        dataset.X_test,
        dataset.y_train.reshape(-1),
        dataset.y_test.reshape(-1),
    )
    ######################################################
    ### Start CeFlow Method
    ######################################################
    means = torch.from_numpy(dataset.X_train.mean(axis=0)).unsqueeze(0).float()

    train_loader = dataset.train_dataloader(batch_size=128, shuffle=True)

    predictive_model = disc_model
    flow_model = train_flow(X_train.shape[1], means, train_loader)
    features = X_train

    features = torch.tensor(features)
    target = y_train

    predictions = predictive_model(features)
    negative_index = negative_prediction_index(predictions)
    negative_instance_features = prediction_instances(features, negative_index)

    positive_index = positive_prediction_index(predictions)
    positive_instance_features = prediction_instances(features, positive_index)

    factual_sample = negative_instance_features[0:2, :]

    mean_z0, mean_z1 = find_latent_mean_two_classes(
        flow_model, negative_instance_features, positive_instance_features
    )

    if DATA_NAME == "simple_bn":
        counterfactual_instance = CounterfactualSimpleBn(
            predictive_model, flow_model, mean_z0, mean_z1, weight
        )
    elif DATA_NAME == "adult":
        deq = DequantizationOriginal()
        counterfactual_instance = CounterfactualAdult(
            predictive_model, flow_model, mean_z0, mean_z1, weight, deq
        )

    # Run algorithm
    start = timeit.default_timer()
    cf_sample = []
    for single_factual in factual_sample:
        counterfactual = (
            counterfactual_instance.find_counterfactual_via_optimizer(
                single_factual.reshape(1, -1)
            )
            .cpu()
            .detach()
            .numpy()
        )
        cf_sample.append(counterfactual)
    stop = timeit.default_timer()
    run_time = stop - start


if __name__ == "__main__":
    main()
