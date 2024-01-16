from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import classification_report
from sklearn.base import RegressorMixin, ClassifierMixin

from counterfactuals.utils import plot_distributions

class AbstractCounterfactualModel(ABC):
    def __init__(self, model, disc_model=None, device=None, neptune_run=None, checkpoint_path=None):
        """
        Initializes the trainer with the provided model.

        Args:
            model (nn.Module): The PyTorch model to be trained.
        """
        if isinstance(model, str):
            model = torch.load(model)
        else:
            self.model = model
        self.classifier = disc_model
        self.with_context = False if self.classifier else True
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.neptune_run = neptune_run
        self.checkpoint_path = checkpoint_path

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
    
    def search_batch(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        lr: float = 0.01,
        verbose: bool = False,
        **search_step_kwargs,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        self.model.eval()
        counterfactuals = []
        original = []
        original_class = []

        loss_hist = []
        for xs_origin, contexts_origin in tqdm(dataloader):
            contexts_origin = contexts_origin.reshape(-1, 1)
            contexts_target = torch.abs(1-contexts_origin)

            xs_origin = torch.as_tensor(xs_origin)
            xs = xs_origin.clone()
            xs_origin.requires_grad = False
            xs.requires_grad = True

            optimizer = optim.Adam([xs], lr=lr)

            for epoch in range(epochs):
                optimizer.zero_grad()
                loss, _, _, _ = self.search_step(xs, xs_origin, contexts_origin, contexts_target, **search_step_kwargs)
                mean_loss = loss.mean()
                mean_loss.backward()
                optimizer.step()

                loss_hist.append(loss.detach().cpu().numpy())
            counterfactuals.append(xs.detach().cpu().numpy())
            original.append(xs_origin.detach().cpu().numpy())
            original_class.append(contexts_origin.detach().cpu().numpy())
        return np.concatenate(counterfactuals, axis=0), np.concatenate(original, axis=0), np.concatenate(original_class, axis=0)
    

    def _model_log_prob(self, inputs, context):
        context = context if self.with_context else None
        return self.model.log_prob(inputs=inputs, context=context)
        

    def train_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        train_losses = []
        i=0
        min_loss = np.inf
        for i in (pbar := tqdm(range(epochs), desc=f"Epochs: {i}, Loss: {np.mean(train_losses)}")):
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
                    if loss.item() < min_loss:
                        min_loss = loss.item()
                        torch.save(self.model, self.checkpoint_path)
                    test_losses.append(loss.item())
            if self.neptune_run:
                self.neptune_run["gen_train_nll"].append(np.mean(train_losses))
                self.neptune_run["gen_test_nll"].append(np.mean(test_losses))
            pbar.set_description(f"Epoch {i}, Train: {np.mean(train_losses):.4f}, test: {np.mean(test_losses):.4f}")
        self.model = torch.load(self.checkpoint_path)

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
        return classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)


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
        log_p_zeros = []
        log_p_ones = []
        with torch.no_grad():
            for x in dataloader:
                x = x[0]
                log_p_zero, log_p_one, result = self.predict_model_point(x)
                results.append(result)
                log_p_zeros.append(log_p_zero)
                log_p_ones.append(log_p_one)
            results = torch.concat(results)
            log_p_zeros = torch.concat(log_p_zeros)
            log_p_ones = torch.concat(log_p_ones)
        return log_p_zeros.numpy(), log_p_ones.numpy(), results.numpy()

        

