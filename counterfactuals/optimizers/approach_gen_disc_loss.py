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

from counterfactuals.optimizers.base import BaseCounterfactualModel


class ApproachGenDiscLoss(BaseCounterfactualModel):
    def __init__(self, gen_model, disc_model, disc_model_criterion, device=None, neptune_run=None, checkpoint_path=None):
        self.disc_model_criterion = disc_model_criterion
        super().__init__(gen_model, disc_model, device, neptune_run, checkpoint_path)

    def search_step(self, x_param, x_origin, context_origin, context_target, step, **search_step_kwargs):
        alpha = search_step_kwargs.get("alpha", None)
        beta = search_step_kwargs.get("beta", None)
        delta = search_step_kwargs.get("delta", None)
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if beta is None:
            raise ValueError("Parameter 'beta' should be in kwargs")
        if delta is None:
            raise ValueError("Parameter 'delta' should be in kwargs")

        dist = torch.linalg.norm(x_origin-x_param, axis=1)

        if self.disc_model and (step < 2000):
            outputs = self.disc_model.forward(x_param)
            outputs = outputs.reshape(-1) if outputs.shape[0] == 1 else outputs
            context_target = context_target.reshape(-1) if context_target.shape[0] == 1 else context_target
            # loss_d = self.disc_model_criterion(outputs, context_target)
            loss_d = torch.nn.functional.relu((-2 * context_target + 1) * (outputs - torch.Tensor([0.5])))
        else:
            loss_d = torch.zeros(1)

        p_x_param_c_orig = self.gen_model.log_prob(x_param, context=context_origin)
        p_x_param_c_target = self.gen_model.log_prob(x_param, context=context_target)
        p_x_orig_c_orig = delta # self.gen_model.log_prob(x_origin, context=context_origin.flatten()[0].repeat((x_origin.shape[0], 1)))

        p_x_param_c_orig_with_beta = p_x_param_c_orig + beta
        max_inner = torch.nn.functional.relu(p_x_orig_c_orig-p_x_param_c_target)
        max_outer = torch.nn.functional.relu(p_x_param_c_orig_with_beta - p_x_param_c_target)
        loss = dist + alpha * (max_outer + 100*max_inner + loss_d)
        return loss, dist, max_inner, max_outer, loss_d

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