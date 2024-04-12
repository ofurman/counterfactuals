import hydra
import os
import torch
import logging
import timeit
from time import time
import numpy as np
import pandas as pd
import neptune
from hydra.utils import instantiate
from tqdm import tqdm

from omegaconf import DictConfig
from alibi.explainers import Counterfactual

from counterfactuals.generative_models.base import BaseGenModel
from counterfactuals.metrics.metrics import evaluate_cf

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf/other_methods", config_name="ce_flow", version_base="1.2"
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
    disc_model = instantiate(
        cfg.disc_model.model,
        input_size=dataset.X_train.shape[1],
        target_size=len(np.unique(dataset.y_train)),
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

    # data_frame = encoder_normalize_data_catalog.data_frame
    # target = encoder_normalize_data_catalog.target
    # features = data_frame.drop(
    #     columns=[target], axis=1).values.astype(np.float32)
    # features = torch.Tensor(features)
    # features = features.cuda()

    # predictions = model_prediction(predictive_model, features)
    # negative_index = negative_prediction_index(predictions)
    # negative_instance_features = prediction_instances(
    #     features, negative_index)

    predictive_model = disc_model
    flow_model = gen_model
    features = X_train
    target = y_train

    predictions = model_prediction(predictive_model, features)
    negative_index = negative_prediction_index(predictions)
    negative_instance_features = prediction_instances(
        features, negative_index)

    positive_index = positive_prediction_index(predictions)
    positive_instance_features = prediction_instances(
        features, positive_index)

    factual_sample = negative_instance_features[0:2, :]

    mean_z0, mean_z1 = find_latent_mean_two_classes(
        flow_model, negative_instance_features, positive_instance_features)

    result_path = ''
    if DATA_NAME == 'simple_bn':
        counterfactual_instance = CounterfactualSimpleBn(
            predictive_model, flow_model, mean_z0, mean_z1, weight)
        result_path = configuration_for_proj['result_simple_bn']
    elif DATA_NAME == 'adult':
        counterfactual_instance = CounterfactualAdult(
            predictive_model, flow_model, mean_z0, mean_z1, weight, deq)
        result_path = configuration_for_proj['result_adult']

    # Run algorithm
    start = timeit.default_timer()
    cf_sample = []
    for single_factual in factual_sample:
        counterfactual = counterfactual_instance.find_counterfactual_via_optimizer(
            single_factual.reshape(1, -1)).cpu().detach().numpy()
        cf_sample.append(counterfactual)
    stop = timeit.default_timer()
    run_time = stop - start


def model_prediction(predictive_model, features):
    return predictive_model(features)


def negative_prediction_index(prediction):
    return torch.lt(prediction, 0.5).reshape(-1)


def positive_prediction_index(prediction):
    return torch.gt(prediction, 0.5).reshape(-1)


def prediction_instances(instances, indexes):
    return instances[indexes]


def find_latent_mean_two_classes(flow, x0, x1):
    z0 = flow(x0) 
    z1 = flow(x1)
    mean_z0 = torch.mean(z0)
    mean_z1 = torch.mean(z1)
    return mean_z0, mean_z1