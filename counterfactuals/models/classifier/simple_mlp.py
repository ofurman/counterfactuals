"""Simple MLP classifier with dropout, class-weighted CE, and early stopping."""

import copy
import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from counterfactuals.models.classifier_mixin import ClassifierPytorchMixin
from counterfactuals.models.pytorch_base import PytorchBase

logger = logging.getLogger(__name__)


class SimpleMLPClassifier(PytorchBase, ClassifierPytorchMixin):
    """Feed-forward classifier with ReLU + dropout blocks.

    Training uses Adam, class-weighted cross-entropy, and best-validation-loss
    early stopping kept in memory.
    """

    def __init__(
        self,
        num_inputs: int,
        num_targets: int,
        hidden_layers: List[int] = [64, 64],
        dropout: float = 0.2,
    ):
        super().__init__(num_inputs, num_targets)
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout

        output_dim = max(2, num_targets)
        self._output_dim = output_dim

        layers: List[nn.Module] = []
        in_dim = num_inputs
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _compute_class_weights(self, loader: DataLoader, device: torch.device) -> torch.Tensor:
        counts = torch.zeros(self._output_dim, dtype=torch.float)
        for _, labels in loader:
            labels = labels.view(-1).long().cpu()
            counts += torch.bincount(labels, minlength=self._output_dim).float()
        if (counts > 0).sum() < 2:
            return torch.ones(self._output_dim, device=device)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum()
        return weights.to(device)

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        lr: float = 0.001,
        patience: int = 15,
        checkpoint_path: str = "best_model.pth",
        **kwargs,
    ) -> None:
        device = next(self.parameters()).device
        class_weights = self._compute_class_weights(train_loader, device)
        logger.info("SimpleMLP class weights: %s", class_weights.detach().cpu().numpy())

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state: Optional[dict] = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for examples, labels in train_loader:
                examples = examples.to(device)
                labels = labels.view(-1).long().to(device)
                optimizer.zero_grad()
                logits = self.forward(examples)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            if test_loader is None:
                if (epoch + 1) % 10 == 0:
                    logger.info("Epoch %d/%d | train_loss=%.4f", epoch + 1, epochs, train_loss)
                continue

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for examples, labels in test_loader:
                    examples = examples.to(device)
                    labels = labels.view(-1).long().to(device)
                    logits = self.forward(examples)
                    val_loss += criterion(logits, labels).item()
            val_loss /= max(len(test_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | best=%.4f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                    best_val_loss,
                )

            if epochs_no_improve >= patience:
                logger.info("SimpleMLP early stopping at epoch %d", epoch + 1)
                break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.save(checkpoint_path)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        device = next(self.parameters()).device
        X_test = X_test.to(device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(X_test)
            probs = self.final_activation(logits)
        return probs.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
