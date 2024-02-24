from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from counterfactuals.utils import plot_distributions


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
            contexts_target = torch.abs(1 - contexts_origin)

            xs_origin = torch.as_tensor(xs_origin)
            xs = xs_origin.clone()
            xs_origin.requires_grad = False
            xs.requires_grad = True

            optimizer = optim.Adam([xs], lr=lr)

            for epoch in range(epochs):
                optimizer.zero_grad()
                loss_components = self.search_step(xs, xs_origin, contexts_origin, contexts_target,
                                                   **search_step_kwargs)
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
        return np.concatenate(counterfactuals, axis=0), np.concatenate(original, axis=0), np.concatenate(original_class,
                                                                                                         axis=0)

    def predict_gen_log_prob(self, x: np.ndarray):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        with torch.no_grad():
            y_zero = torch.zeros((x.shape[0], 1), dtype=x.dtype).to(self.device)
            y_one = torch.ones((x.shape[0], 1), dtype=x.dtype).to(self.device)
            log_p_zero = self.gen_model.predict_log_probs(x, y_zero)
            log_p_one = self.gen_model.predict_log_probs(x, y_one)
        result = torch.vstack([log_p_zero, log_p_one])
        return result

    def test_model(self, test_loader: DataLoader):
        """
        Test the model within defined metrics.
        """
        self.gen_model.eval()
        ys_pred = []
        ys_true = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                log_p = self.predict_gen_log_prob(x)
                y_pred = torch.argmax(log_p, axis=0)
                ys_pred.append(y_pred)
                ys_true.append(y)

        ys_pred = torch.concat(ys_pred).cpu()
        ys_true = torch.concat(ys_true).cpu()
        return classification_report(y_true=ys_true, y_pred=ys_pred, output_dict=True)

    def calculate_median_log_prob(self, train_dataloader: DataLoader) -> float:
        """
        Test the model within defined metrics.
        """
        log_probs = self.gen_model.predict_log_prob(train_dataloader)
        return np.median(log_probs)

    def predict(self, test_data: Union[DataLoader, torch.Tensor, np.ndarray], batch_size: int = 64) -> np.ndarray:
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
                x = x[0].to(self.device)
                log_p = self.gen_model.predict_log_probs(x)
                y_pred = torch.argmax(log_p, axis=0)
                results.append(y_pred)
            results = torch.concat(results)
        return results.cpu().numpy().astype(np.float32)
