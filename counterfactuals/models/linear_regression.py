from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from counterfactuals.models.pytorch_base import PytorchBase
from counterfactuals.models.regression_mixin import RegressionPytorchMixin


class LinearRegression(PytorchBase, RegressionPytorchMixin):
    def __init__(self, num_inputs: int, num_targets: int):
        super(LinearRegression, self).__init__(num_inputs, num_targets)
        self.linear = torch.nn.Linear(num_inputs, num_targets)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        lr: float = 0.003,
        patience: int = 100,
        **kwargs,
    ) -> None:
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        patience_counter = 0
        for epoch in (pbar := tqdm(range(epochs))):
            losses = []
            test_losses = []
            min_loss = float("inf")
            for i, (examples, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(examples)
                labels = labels.reshape(-1, 1)
                loss = criterion(outputs, labels.view(outputs.shape).float())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            if test_loader:
                with torch.no_grad():
                    for i, (examples, labels) in enumerate(test_loader):
                        outputs = self.forward(examples)
                        labels = labels.reshape(-1, 1)
                        test_loss = criterion(
                            outputs, labels.view(outputs.shape).float()
                        )
                        test_losses.append(test_loss.item())
                if np.mean(test_losses) < min_loss:
                    min_loss = np.mean(test_losses)
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter == patience:
                    break

            pbar.set_description(
                f"Epoch {epoch}, Train Loss: {np.mean(losses):.4f}, Test Loss: {np.mean(test_losses):.4f}, Patience: {patience_counter}"
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
