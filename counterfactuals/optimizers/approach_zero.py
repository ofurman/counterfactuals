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

from counterfactuals.optimizers.base import AbstractCounterfactualModel


class MainCounterfactuals(AbstractCounterfactualModel):
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
        if alpha is None:
            raise ValueError("Paramtere 'alpha' should be in kwargs")

        dist = torch.linalg.norm(x_origin - x_param, axis=1)

        p_orig = self.model.log_prob(x_param, context=context_origin)
        p_hat = self.model.log_prob(x_param, context=context_target)
        loss = dist - alpha * (p_hat - torch.logsumexp(torch.concat([p_orig, p_hat]), dim=0))
        return loss

