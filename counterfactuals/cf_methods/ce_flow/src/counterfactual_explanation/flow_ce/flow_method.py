from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from counterfactuals.cf_methods.ce_flow.src.counterfactual_explanation.utils.mlcatalog import (
    get_latent_representation_from_flow,
    get_latent_representation_from_flow_mixed_type,
    original_space_value_from_latent_representation,
    original_space_value_from_latent_representation_mixed_type)


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
