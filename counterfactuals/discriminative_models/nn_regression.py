from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from counterfactuals.discriminative_models.pytorch_base import PytorchBase
from counterfactuals.discriminative_models.regression_mixin import RegressionPytorchMixin


class MLPRegressor(PytorchBase, RegressionPytorchMixin):
    def __init__(
        self,
        num_inputs: int,
        num_targets: int,
        hidden_layer_sizes: List[int],
        dropout: float = 0.2,
    ):
        super(MLPRegressor, self).__init__(num_inputs, num_targets)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout
        
        layer_sizes = [num_inputs] + hidden_layer_sizes + [num_targets]
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.relu(self.dropout(layer(x)))
            else:
                x = layer(x)
        return x

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        lr: float = 0.001,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
        **kwargs
    ) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        patience_counter = 0
        min_loss = float("inf")

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.train()
            losses = []
            test_losses = []
            for examples, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(examples)
                loss = criterion(outputs, labels.view(outputs.shape).float())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            if test_loader:
                self.eval()
                with torch.no_grad():
                    for examples, labels in test_loader:
                        outputs = self.forward(examples)
                        test_loss = criterion(
                            outputs, labels.view(outputs.shape).float()
                        )
                        test_losses.append(test_loss.item())
                avg_test_loss = np.mean(test_losses)
                if avg_test_loss < min_loss:
                    min_loss = avg_test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter == patience:
                    break

            pbar.set_description(
                f"Epoch {epoch}, Train Loss: {np.mean(losses):.4f}, Test Loss: {avg_test_loss:.4f}, Patience: {patience_counter}"
            )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).type(torch.float32)
        with torch.no_grad():
            preds = self.forward(X_test)
            return preds.cpu().numpy()

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
