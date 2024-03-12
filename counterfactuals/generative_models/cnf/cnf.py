import uuid
from typing import Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .flow import ContinuousNormalizingFlow


class ContinuousNormalizingFlowRegressor(BaseEstimator, RegressorMixin, nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int = 32,
        hidden_dims: Iterable[int] = (64, 64),
        num_blocks: int = 3,
        layer_type: str = "concatsquash",
        nonlinearity: str = "tanh",
        device: str = None,
    ):
        """
        Initialization of Continuous Normalizing Flow model.
        """
        nn.Module.__init__(self)

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.layer_type = layer_type
        self.nonlinearity = nonlinearity

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh(),
        ).to(self.device)

        self.flow_model = ContinuousNormalizingFlow(
            input_dim=output_dim,
            hidden_dims=hidden_dims,
            context_dim=embedding_dim,
            num_blocks=num_blocks,
            conditional=True,  # It must be true as we are using Conditional CNF model.
            layer_type=layer_type,
            nonlinearity=nonlinearity,
            device=self.device,
        )

    def log_prob(self, X: torch.Tensor, y: torch.Tensor):
        """Calculate the log probability of the model (batch). Internal method used for training."""
        x = self.feature_extractor(X)
        x = self.flow_model.log_prob(y, x)
        # x += np.log(np.abs(np.prod(self.target_scaler.scale_)))  # Target scaling correction. log(abs(det(jacobian)))
        return x

    @torch.no_grad()
    def _log_prob(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 128
    ) -> np.ndarray:
        """Calculate the log probability of the model."""

        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        y: torch.Tensor = torch.as_tensor(data=y, dtype=torch.float, device=self.device)

        dataset_loader: DataLoader = DataLoader(
            dataset=TensorDataset(X, y), shuffle=False, batch_size=batch_size
        )

        logpxs: List[torch.Tensor] = [
            self._log_prob(X=x_batch, y=y_batch) for x_batch, y_batch in dataset_loader
        ]
        logpx: torch.Tensor = torch.cat(logpxs, dim=0)
        logpx: torch.Tensor = logpx.detach().cpu()
        logpx: np.ndarray = logpx.numpy()
        return logpx

    @torch.no_grad()
    def _sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        x = self.feature_extractor(X)
        x = self.flow_model.sample(x, num_samples=num_samples)
        return x

    @torch.no_grad()
    def sample(
        self, X: np.ndarray, num_samples: int = 10, batch_size: int = 128
    ) -> np.ndarray:
        """Sample from the model."""
        X: np.ndarray = self.feature_scaler.transform(X)

        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        dataset_loader: DataLoader = DataLoader(
            dataset=TensorDataset(X), shuffle=False, batch_size=batch_size
        )

        all_samples: List[torch.Tensor] = []

        for x in dataset_loader:
            sample: torch.Tensor = self._sample(x[0], num_samples)
            all_samples.append(sample)

        samples: torch.Tensor = torch.cat(all_samples, dim=0)
        samples: torch.Tensor = samples.detach().cpu()
        samples: np.ndarray = samples.numpy()

        # Inverse target transformation
        samples_size = samples.shape

        samples: np.ndarray = samples.reshape(
            (samples_size[0] * samples_size[1], samples_size[2])
        )
        samples: np.ndarray = self.target_scaler.inverse_transform(samples)
        samples: np.ndarray = samples.reshape(
            (samples_size[0], samples_size[1], samples_size[2])
        )

        samples: np.ndarray = samples.squeeze()
        return samples

    def fit(
        self,
        dataset_loader_train,
        X_val: Union[np.ndarray, None] = None,
        y_val: Union[np.ndarray, None] = None,
        n_epochs: int = 100,
        batch_size: int = 128,
        max_patience: int = 50,
        verbose: bool = False,
    ):
        """Fit Continuous Normalizing Flow model.

        Method supports the best epoch model selection and early stopping (max_patience param)
        if validation dataset is available.
        """
        self.optimizer_ = optim.Adam(self.parameters())

        patience: int = 0
        mid: str = str(
            uuid.uuid4()
        )  # To be able to run multiple experiments in parallel.
        loss_best: float = np.inf

        with tqdm(range(n_epochs)) as pbar:
            for _ in pbar:
                self.train()
                for x_batch, y_batch in dataset_loader_train:
                    self.optimizer_.zero_grad()

                    logpx = self._log_prob(x_batch, y_batch)
                    loss = -logpx.mean()

                    loss.backward()
                    self.optimizer_.step()

                self.eval()
                if X_val is not None and y_val is not None:
                    loss_val: float = self.nll(X_val, y_val)
                    pbar.set_description(f"Validation loss: {round(loss_val, 4)}.")

                    # Save model if better
                    if loss_val < loss_best:
                        loss_best = loss_val
                        self._save_temp(mid)
                        patience = 0

                    else:
                        patience += 1

                    if patience > max_patience:
                        break

        if X_val is not None and y_val is not None:
            return self._load_temp(mid)
        return self

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        method: str = "mean",
        num_samples: int = 1000,
        batch_size: int = 128,
        **kwargs,
    ) -> np.ndarray:
        samples: np.ndarray = self.sample(
            X=X, num_samples=num_samples, batch_size=batch_size
        )

        if method == "mean":
            y_pred: np.ndarray = samples.mean(axis=1)
        else:
            raise ValueError(f"Method {method} not supported.")

        y_pred: np.ndarray = np.array(y_pred)
        return y_pred

    @torch.no_grad()
    def nll(self, X: np.ndarray, y: np.ndarray) -> float:
        return -self.log_prob(X, y).mean()

    def _save_temp(self, mid: str):
        torch.save(self, f"/tmp/model_{mid}.pt")

    def _load_temp(self, mid: str):
        return torch.load(f"/tmp/model_{mid}.pt")
