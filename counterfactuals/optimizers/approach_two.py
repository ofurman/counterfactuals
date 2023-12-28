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


class ApproachTwo(AbstractCounterfactualModel):
    def __init__(self, model, device=None):
        self.model = model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def configure_search_optimizer(self):
        return optim.Adam(self.parameters(), lr=1e-3)

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

    def train_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        for i in tqdm(range(epochs), desc="Epochs: ", leave=False):
            train_losses = []
            test_losses = []
            for x, y in train_loader:
                y = y.reshape(-1, 1)
                optimizer.zero_grad()
                loss = -self.model.log_prob(inputs=x, context=y).mean()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            for x, y in test_loader:
                with torch.no_grad():
                    y = y.reshape(-1, 1)
                    loss = -self.model.log_prob(inputs=x, context=y).mean()
                    test_losses.append(loss.item())
            if i % 10 == 0:
                print(f"Epoch {i}, Train: {np.mean(train_losses)}, test: {np.mean(test_losses)}")
