import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

class AbstractCounterfactualModel(ABC, nn.Module):
    def __init__(self, model, device=None):
        """
        Initializes the trainer with the provided model.

        Args:
            model (nn.Module): The PyTorch model to be trained.
        """
        super().__init__()
        self.model = model
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = self.configure_optimizer()

    @abstractmethod
    def train_step(self, x_param, x_origin, context_origin, context_target):
        """
        Performs a single training step on a batch of data.

        Args:
            data (dict): A dictionary containing input data and target data.

        Returns:
            float: The loss for the current training step.
        """
        pass

    def configure_optimizer(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer for updating model parameters.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train(self, x_origin: torch.Tensor, context_origin: torch.Tensor, context_target: torch.Tensor, num_epochs: int =10):
        """
        Trains the model for a specified number of epochs.
        """
        self.model.eval()
        optimizer = self.configure_optimizer()

        x_origin = torch.as_tensor(x_origin)
        x = x_origin.copy()
        x_origin.requires_grad = False
        x.requires_grad = True
        
        loss_hist = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.train_step(x, x_origin, context_origin, context_target)
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss_hist[-1]:.4f}")
        print("Training finished!")
