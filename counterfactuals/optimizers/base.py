from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from typing import Union

import numpy as np
from sklearn.metrics import classification_report
from sklearn.base import RegressorMixin, ClassifierMixin

from counterfactuals.utils import plot_distributions

class BaseCounterfactualModel(ABC):
    def __init__(self, gen_model, disc_model=None, device=None, neptune_run=None, checkpoint_path=None):
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
        raise NotImplementedError("Use search batch method.")
        self.gen_model.eval()
        self.disc_model.eval()
        
        x_origin = torch.as_tensor(x_origin)
        x = x_origin.clone()
        x_origin.requires_grad = False
        x.requires_grad = True

        optimizer = optim.Adam([x], lr=lr)

        loss_hist = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss, dist, max_inner, max_outer = self.search_step(x, x_origin, context_origin, context_target, **search_step_kwargs)
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
        lr: float = 0.001,
        patience: int = 20,
        eps: float = 1e-3,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        
        optimizer = optim.Adam(self.gen_model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=2, total_iters=400)
        train_losses = []
        i=0
        min_loss = np.inf
        epochs_no_improve = 0

        for i in (pbar := tqdm(range(epochs))):
            train_losses = []
            test_losses = []
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y = y.reshape(-1, 1)
                optimizer.zero_grad()
                loss = -self.gen_model.log_prob(inputs=x, context=y).mean()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    y = y.reshape(-1, 1)
                    loss = -self.gen_model.log_prob(inputs=x, context=y).mean()
                    if np.abs(loss.item() - min_loss) > eps:
                        min_loss = loss.item()
                        torch.save(self.gen_model, self.checkpoint_path)
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    test_losses.append(loss.item())
            if self.neptune_run:
                self.neptune_run["gen_train_nll"].append(np.mean(train_losses))
                self.neptune_run["gen_test_nll"].append(np.mean(test_losses))
            pbar.set_description(f"Epoch {i}, Train: {np.mean(train_losses):.4f}, test: {np.mean(test_losses):.4f}")
            if epochs_no_improve > patience:
                print("Early stopping!")
                break
        self.model = torch.load(self.checkpoint_path)

    def predict_gen_log_prob(self, x: np.ndarray):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        with torch.no_grad():
            y_zero = torch.zeros((x.shape[0], 1), dtype=x.dtype).to(self.device)
            y_one = torch.ones((x.shape[0], 1), dtype=x.dtype).to(self.device)
            log_p_zero = self.gen_model.log_prob(inputs=x, context=y_zero)
            log_p_one = self.gen_model.log_prob(inputs=x, context=y_one)
        result = torch.vstack([log_p_zero, log_p_one])
        return result

    def test_model(
        self,
        test_loader: DataLoader,
    ):
        """
        Test the model within defined metrics.
        """
        self.model.eval()
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

    def calculate_median_log_prob(
            self,
            test_loader: DataLoader,
    ):
        """
        Test the model within defined metrics.
        """
        self.gen_model.eval()
        log_probs = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device).reshape(-1, 1)
                log_p = self.gen_model.log_prob(x, y)
                log_probs.append(log_p)

        log_probs = torch.concat(log_probs)
        return np.median(log_probs)
    

    def predict(self, test_data: Union[DataLoader, np.ndarray], batch_size: int = 64):
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
                log_p = self.predict_gen_log_prob(x)
                y_pred = torch.argmax(log_p, axis=0)
                results.append(y_pred)
            results = torch.concat(results)
        return results.cpu().numpy().astype(np.float32)

        

