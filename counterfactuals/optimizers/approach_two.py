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

from counterfactuals.optimizers.base import BaseCounterfactualModel


class ApproachTwo(BaseCounterfactualModel):
    def __init__(self, model, device=None):
        self.model = model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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

        self.model.eval()
        dist = torch.linalg.norm(x_origin - x_param, axis=1)

        p_x_param_c_orig = self.model.log_prob(x_param, context=context_origin).exp()
        p_x_param_c_target = self.model.log_prob(x_param, context=context_target).exp()
        p_x_orig_c_orig = self.model.log_prob(x_origin, context=context_origin).exp()
        
        max_inner = torch.max(torch.cat((p_x_param_c_orig + beta, p_x_orig_c_orig)))
        max_outer = torch.max(torch.cat((max_inner - p_x_param_c_target, torch.Tensor([0.0]))))
        loss = dist + alpha * max_outer
        return loss
