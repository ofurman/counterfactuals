"""
# Approach 2

$$agmin\quad d(x, x') - \lambda (log p(x'|y') - log(p(x'|y) + p(x'|y')))$$
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial

from counterfactuals.optimizers.base import AbstractCounterfactualModel


class ApproachThree(AbstractCounterfactualModel):
    def search_step(self, x_param, x_origin, context_origin, context_target, **kwargs):
        """
        Performs a single training step on a batch of data.

        Args:
            data (dict): A dictionary containing input data and target data.

        Returns:
            float: The loss for the current training step.
        """
        alpha = kwargs.get("alpha", None)
        beta = kwargs.get("beta", None)
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if beta is None:
            raise ValueError("Parameter 'beta' should be in kwargs")

        dist = torch.linalg.norm(x_origin-x_param, axis=1)

        p_x_param_c_orig = self.model.log_prob(x_param, context=context_origin)
        p_x_param_c_target = self.model.log_prob(x_param, context=context_target)
        p_x_orig_c_orig = self.model.log_prob(x_origin, context=context_origin.flatten()[0].repeat((x_origin.shape[0], 1)))

        p_x_param_c_orig_with_beta = p_x_param_c_orig + beta
        max_inner = torch.nn.functional.relu(p_x_orig_c_orig-p_x_param_c_target)
        max_outer = torch.nn.functional.relu(p_x_param_c_orig_with_beta - p_x_param_c_target)
        loss = dist + alpha * (max_outer + max_inner)
        return loss, dist, max_inner, max_outer
    
    def generate_counterfactuals(self, Xs, ys, epochs, lr, alpha, beta):
        Xs = Xs[:, np.newaxis, :]
        ys = ys.reshape(-1, 1)
        ys_hat = np.abs(1-ys).reshape(-1, 1)
        x_cfs = []
        for X, y, y_hat in tqdm(zip(Xs, ys, ys_hat)):
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            y_hat = torch.Tensor(y_hat)
            x_cf = self.search(X, y, y_hat, num_epochs=epochs, lr=lr, alpha=alpha, beta=beta, verbose=False)
            x_cfs.append(x_cf)

        # x_cfs = np.array([x.detach().numpy() for x in x_cfs]).squeeze()
        return x_cfs