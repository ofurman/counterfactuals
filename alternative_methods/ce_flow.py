import hydra
import os
import torch
import torch.nn as nn
import torch.optim as optim
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

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf/other_methods", config_name="config_ce_flow", version_base="1.2"
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

    if DATA_NAME == 'simple_bn':
        counterfactual_instance = CounterfactualSimpleBn(
            predictive_model, flow_model, mean_z0, mean_z1, weight)
    elif DATA_NAME == 'adult':
        deq = DequantizationOriginal()
        counterfactual_instance = CounterfactualAdult(
            predictive_model, flow_model, mean_z0, mean_z1, weight, deq)

    # Run algorithm
    start = timeit.default_timer()
    cf_sample = []
    for single_factual in factual_sample:
        counterfactual = counterfactual_instance.find_counterfactual_via_optimizer(
            single_factual.reshape(1, -1)).cpu().detach().numpy()
        cf_sample.append(counterfactual)
    stop = timeit.default_timer()
    run_time = stop - start


class FindCounterfactualSample(ABC):
    @abstractmethod
    def initialize_latent_representation(self):
        pass

    @abstractmethod
    def distance_loss(self):
        pass

    @abstractmethod
    def prediction_loss(self):
        pass

    @abstractmethod
    def fair_loss(self):
        pass


class CounterfactualSimpleBn(FindCounterfactualSample):
    def __init__(self, predictive_model, flow_model, z_mean0, z_mean1, weight):
        # self.original_instance = original_instance
        self.flow_model = flow_model
        self.predictive_model = predictive_model
        self.distance_loss_func = torch.nn.MSELoss()
        # self.distance_loss_func = torch.nn.L1Loss()
        self.predictive_loss_func = torch.nn.BCELoss()
        self.lr = 1e-1
        self.n_epochs = 1000
        self.z_mean0 = z_mean0
        self.z_mean1 = z_mean1
        self.weight = weight

    @property
    def _flow_model(self):
        return self.flow_model

    @property
    def _predictive_model(self):
        return self.predictive_model

    def initialize_latent_representation(self):
        pass

    def distance_loss(self, factual, counterfactual):
        return self.distance_loss_func(factual, counterfactual)

    def prediction_loss(self, representation_counterfactual):
        counterfactual = self._original_space_value_from_latent_representation(
            representation_counterfactual)
        yhat = self._predictive_model(counterfactual).reshape(-1)
        yexpected = torch.ones(
            yhat.shape, dtype=torch.float).reshape(-1).cuda()
        self.predictive_loss_func(yhat, yexpected)
        return self.predictive_loss_func(yhat, yexpected)

    def fair_loss(self):
        return 0

    def combine_loss(self, factual, counterfactual):
        return self.weight * self.prediction_loss(counterfactual) + (1 - self.weight) * self.distance_loss(factual,
                                                                                                           counterfactual)

    def make_perturbation(self, z_value, delta_value):
        return z_value + delta_value

    def _get_latent_representation_from_flow(self, input_value):
        return get_latent_representation_from_flow(self.flow_model, input_value)

    def _original_space_value_from_latent_representation(self, z_value):
        return original_space_value_from_latent_representation(self.flow_model, z_value)

    # def find_counterfactual_via_iterations(self, factual):
    #     z_value = self._get_latent_representation_from_flow(factual)
    #     index_ = 0
    #     for _ in tqdm(range(self.n_epochs)):
    #         index_ += 1
    #         delta_value = torch.rand(z_value.shape[1]).cuda()
    #         z_hat = self.make_perturbation(z_value, delta_value)
    #         x_hat = self._original_space_value_from_latent_representation(
    #             z_hat)
    #         prediction = self._predictive_model(x_hat)
    #         if torch.gt(prediction[0], 0.5):
    #             return x_hat[0]
    #     return x_hat[0]

    def find_counterfactual_via_optimizer(self, factual):
        z_value = self._get_latent_representation_from_flow(factual)
        # delta_value = nn.Parameter(torch.rand(z_value.shape[1]).cuda())
        delta_value = nn.Parameter(torch.zeros(z_value.shape[1]).cuda())

        representation_factual = self._get_latent_representation_from_flow(
            factual)
        z_hat = self.make_perturbation(z_value, delta_value)
        x_hat = self._original_space_value_from_latent_representation(
            z_hat)
        optimizer = optim.Adam([delta_value], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
        candidates = []
        for epoch in (range(self.n_epochs)):
            epoch += 1
            z_hat = self.make_perturbation(z_value, delta_value)
            x_hat = self._original_space_value_from_latent_representation(
                z_hat)
            total_loss = self.combine_loss(representation_factual, z_hat)
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()
            if epoch % 10 == 0:
                scheduler.step()
                cur_lr = scheduler.optimizer.param_groups[0]['lr']
                print("\n Epoch {}, Loss {:.4f}, Learning rate {:.4f}, \n Prediction {}".format(
                    epoch, total_loss, cur_lr, prediction[0]))

            prediction = self._predictive_model(x_hat)
            if torch.gt(prediction[0], 0.5):
                candidates.append(x_hat[0].detach())

        try:
            candidates = torch.stack(candidates)
        except:
            return x_hat[0]
        candidate_distances = torch.abs(factual - candidates).mean(axis=1)
        return candidates[torch.argmax(candidate_distances)]

    # def find_counterfactual_via_gradient_descent(self, factual):
    #     print(factual)
    #     z_factual = self._get_latent_representation_from_flow(
    #         factual)
    #     delta_value = nn.Parameter(torch.rand(z_factual.shape).cuda())
    #     # optimizer = optim.SGD([delta_value], lr=self.lr, momentum=0.9)
    #     optimizer = optim.Adam([delta_value], lr=self.lr)
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=500, gamma=0.1)

    #     candidates = []
    #     for epoch in tqdm(range(self.n_epochs)):
    #         epoch += 1
    #         z_hat = self.make_perturbation(z_factual, delta_value)
    #         x_hat = self._original_space_value_from_latent_representation(
    #             z_hat)
    #         total_loss = self.combine_loss(z_factual, z_hat)
    #         optimizer.zero_grad()
    #         total_loss.backward(retain_graph=True)
    #         optimizer.step()
    #         scheduler.step()
    #         cur_lr = scheduler.optimizer.param_groups[0]['lr']
    #         if epoch % 10 == 0:
    #             print("\n Epoch {}, Loss {:.4f}, Learning rate {:.4f}, Prediction {}".format(
    #                 epoch, total_loss, cur_lr, prediction[0]))
    #             # print("Perturbation ", z_hat)

    #         prediction = self._predictive_model(x_hat)
    #         if torch.gt(prediction[0], 0.5):
    #             candidates.append(x_hat)

    #     return x_hat

    # def find_counterfactual_by_scaled_vector(self, factual):
    #     z_factual = self._get_latent_representation_from_flow(factual)
    #     scaled = 1
    #     delta_value = scaled*torch.abs(self.z_mean0 - self.z_mean1)
    #     z_hat = self.make_perturbation(z_factual, delta_value)
    #     x_hat = self._original_space_value_from_latent_representation(
    #         z_hat)
    #     return x_hat


class CounterfactualAdult(CounterfactualSimpleBn):
    def __init__(self, predictive_model, flow_model, z_mean0, z_mean1, weight, deq):
        super().__init__(predictive_model, flow_model, z_mean0, z_mean1, weight)
        self.deq = deq

    def _original_space_value_from_latent_representation(self, z_value):
        # return get_latent_representation_from_flow_mixed_type(self.flow, self.deq, z_value, 3)
        return original_space_value_from_latent_representation_mixed_type(self.flow_model, self.deq, z_value, 3)

    def _get_latent_representation_from_flow(self, input_value):
        return get_latent_representation_from_flow_mixed_type(self.flow_model, self.deq, input_value, 3)


class DequantizationOriginal(nn.Module):
    def __init__(self, alpha=1e-5, quants=256):
        """
        Args:
            alpha: small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants: Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        # self.quants = quants
        self.quants = torch.Tensor(np.array([8, 5, 6])).cuda()

    def forward(self, z, ldj=None, reverse=False):
        if not reverse:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            # ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z[:, 0] = torch.floor(z[:, 0]).clamp(min=0, max=self.quants[0] - 1).to(torch.int32)
            z[:, 1] = torch.floor(z[:, 1]).clamp(min=0, max=self.quants[1] - 1).to(torch.int32)
            z[:, 2] = torch.floor(z[:, 2]).clamp(min=0, max=self.quants[2] - 1).to(torch.int32)
        return z, ldj

    def sigmoid(self, z, ldj=None, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            # ldj += (-z - 2 * F.softplus(-z)).sum()
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            # ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            # ldj += (-torch.log(z) - torch.log(1 - z)).sum()
            z = torch.log(z) - torch.log(1 - z)
        return z, ldj

    def dequant(self, z, ldj=None):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        # ldj -= np.log(8) * np.prod(z.shape[1:])
        return z, ldj



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


def get_latent_representation_from_flow(flow, input_value):
    return flow(input_value)


def original_space_value_from_latent_representation(flow, z_value):
    return flow.inverse(z_value)

def get_latent_representation_from_flow_mixed_type(flow, deq, input_value, index_):
    discrete_value = input_value[:,:index_]
    continuous_transformation, _ = deq.forward(discrete_value, ldj=None, reverse=False)
    continuous_value = input_value[:, index_:]
    continuous_representation = torch.hstack([continuous_transformation, continuous_value])

    # z_value = flow(continuous_representation)
    # continuous_value_ = flow.inverse(z_value)
    # discrete_value_ = z_value[:,:index_]
    # continuous_value_ = z_value[:, index_:]
    # discrete_value_, _ = deq.forward(discrete_value_, ldj=None, reverse=True)
    # input_value = torch.hstack([discrete_value_, continuous_value_])


    return flow(continuous_representation)

def original_space_value_from_latent_representation_mixed_type(flow, deq, z_value, index_):
    continuous_value = flow.inverse(z_value)
    discrete_value = continuous_value[:,:index_]
    continuous_value = continuous_value[:, index_:]
    discrete_value, _ = deq.forward(discrete_value, ldj=None, reverse=True)
    input_value = torch.hstack([discrete_value, continuous_value])

    return input_value


if __name__ == "__main__":
    main()
