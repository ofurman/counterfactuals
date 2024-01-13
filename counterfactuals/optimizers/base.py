from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import classification_report

from counterfactuals.utils import plot_distributions

class AbstractCounterfactualModel(ABC):
    def __init__(self, model, with_context: bool = False, device=None):
        """
        Initializes the trainer with the provided model.

        Args:
            model (nn.Module): The PyTorch model to be trained.
        """
        self.model = model
        self.with_context = with_context
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
        verbose: bool = False,
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
            loss, _, _, _ = self.search_step(x, x_origin, context_origin, context_target, **search_step_kwargs)
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            if verbose and (epoch % 10 == 0):
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss_hist[-1]:.4f}")
        
        if verbose:
            print("Search finished!")
        return x
    

    def _model_log_prob(self, inputs, context):
        context = context if self.with_context else None
        return self.model.log_prob(inputs=inputs, context=context)
        

    def train_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True, #TODO: add support of this parameter.
    ):
        """
        Trains the model for a specified number of epochs.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        for i in tqdm(range(epochs), desc="Epochs: "):
            train_losses = []
            test_losses = []
            for x, y in train_loader:
                y = y.reshape(-1, 1)
                optimizer.zero_grad()
                loss = -self._model_log_prob(inputs=x, context=y).mean()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            for x, y in test_loader:
                with torch.no_grad():
                    y = y.reshape(-1, 1)
                    loss = -self._model_log_prob(inputs=x, context=y).mean()
                    test_losses.append(loss.item())
            if i % 10 == 0:
                print(f"Epoch {i}, Train: {np.mean(train_losses)}, test: {np.mean(test_losses)}")

    def test_model(
        self,
        test_loader: DataLoader,
    ):
        """
        Test the model within defined metrics.
        """

        self.model.eval()

        y_pred = []
        y_true = []

        with torch.no_grad():
            for x, y in test_loader:
                y_zero = torch.zeros((x.shape[0], 1))
                y_one = torch.ones((x.shape[0], 1))
                log_p_zero = self._model_log_prob(inputs=x, context=y_zero)
                log_p_one = self._model_log_prob(inputs=x, context=y_one)
                result = log_p_one > log_p_zero
                y_pred.append(result)
                y_true.append(y)
        
        y_pred = torch.concat(y_pred)
        y_true = torch.concat(y_true)
        print(classification_report(y_true=y_true, y_pred=y_pred))


    def predict_model_point(self, x: np.ndarray):
        y_zero = torch.zeros((x.shape[0], 1))
        y_one = torch.ones((x.shape[0], 1))
        if self.with_context:
            log_p_zero = self.model.log_prob(inputs=x, context=y_zero)
            log_p_one = self.model.log_prob(inputs=x, context=y_one)
            result = log_p_one > log_p_zero
            return log_p_zero, log_p_one, result
        else:
            log_p = self.model.log_prob(inputs=x, context=None)
            return None, None, log_p


    def predict_model(self, test_data: DataLoader | np.ndarray, batch_size: int = 64):
        """
        Predict class using generative model.
        """

        if not isinstance(test_data, (np.ndarray, DataLoader, torch.Tensor)):
            raise TypeError("Data should be numpy array or torch dataloader!")
        if isinstance(test_data, (np.ndarray, torch.Tensor)):
            test_data = torch.Tensor(test_data)
            dataloader = DataLoader(
                dataset=TensorDataset(test_data),
                shuffle=False,
                batch_size=batch_size
            )
        else:
            dataloader = test_data

        results = []
        with torch.no_grad():
            for x in dataloader:
                x = x[0]
                _, _, result = self.predict_model_point(x)
                results.append(result)
            results = torch.concat(results)
        return results.numpy()

        

