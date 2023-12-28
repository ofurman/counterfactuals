from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class AbstractCounterfactualModel(ABC):
    def __init__(self, model, device=None):
        """
        Initializes the trainer with the provided model.

        Args:
            model (nn.Module): The PyTorch model to be trained.
        """
        self.model = model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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

    @abstractmethod
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
        pass

    def search(
        self,
        x_origin: torch.Tensor,
        context_origin: torch.Tensor,
        context_target: torch.Tensor,
        num_epochs: int = 10,
        lr: float = 0.01,
        verbose: bool = True,
        **search_step_kwargs,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        self.model.eval()
        
        x_origin = torch.as_tensor(x_origin)
        x = x_origin.clone()
        x_origin.requires_grad = False
        x.requires_grad = True

        optimizer = optim.Adam([x], lr=lr)

        loss_hist = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.search_step(x, x_origin, context_origin, context_target, **search_step_kwargs)
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            if verbose and (epoch % 10 == 0):
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss_hist[-1]:.4f}")
        
        if verbose:
            print("Search finished!")
        return x
