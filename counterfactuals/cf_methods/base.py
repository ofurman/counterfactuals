from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class BaseCounterfactualModel(ABC):
    def __init__(self, gen_model, disc_model=None, device=None, neptune_run=None):
        """
        Initializes the trainer with the provided model.

        Args:
            model (nn.Module): The PyTorch model to be trained.
        """
        self.gen_model = gen_model
        self.disc_model = disc_model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_model.to(self.device)
        if self.disc_model:
            self.disc_model.to(self.device)
        self.neptune_run = neptune_run

    @abstractmethod
    def search_step(self, x_param, x_origin, context_origin, context_target):
        """
        Performs a single training step on a batch of data.

        Args:
            data (dict): A dictionary containing input data and target data.

        Returns:
            float: The loss for the current training step.
        """
        pass
    
    def search_batch(
        self,
        dataloader: DataLoader,
        epochs: int = 1000,
        lr: float = 0.0005,
        patience: int = 100,
        verbose: bool = False,
        **search_step_kwargs,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

        if self.disc_model:
            self.disc_model.eval()
            for param in self.disc_model.parameters():
                param.requires_grad = False

        counterfactuals = []
        original = []
        original_class = []
        min_loss = np.inf
        no_improve = 0
        for xs_origin, contexts_origin in tqdm(dataloader):
            xs_origin = xs_origin.to(self.device)
            contexts_origin = contexts_origin.to(self.device)

            contexts_origin = contexts_origin.reshape(-1, 1)
            contexts_target = torch.abs(1-contexts_origin)

            xs_origin = torch.as_tensor(xs_origin)
            xs = xs_origin.clone()
            xs_origin.requires_grad = False
            xs.requires_grad = True

            optimizer = optim.Adam([xs], lr=lr)

            for epoch in range(epochs):
                optimizer.zero_grad()
                loss_components = self.search_step(xs, xs_origin, contexts_origin, contexts_target, **search_step_kwargs)
                mean_loss = loss_components["loss"].mean()
                mean_loss.backward()
                optimizer.step()

                if self.neptune_run:
                    for loss_name, loss in loss_components.items():
                        self.neptune_run[f"cf_search/{loss_name}"].append(loss.mean().detach().cpu().numpy())
                if mean_loss.item() < min_loss:
                    min_loss = mean_loss.item()
                else:
                    no_improve += 1
                if no_improve > patience:
                    break

            counterfactuals.append(xs.detach().cpu().numpy())
            original.append(xs_origin.detach().cpu().numpy())
            original_class.append(contexts_origin.detach().cpu().numpy())
        return np.concatenate(counterfactuals, axis=0), np.concatenate(original, axis=0), np.concatenate(original_class, axis=0)