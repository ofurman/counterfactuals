from abc import ABC

import numpy as np
import torch


class RegressionPytorchMixin(ABC):
    """Mixin providing regression interface for PyTorch models.

    Subclasses must define:
        - forward(x: torch.Tensor) -> torch.Tensor
    """

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions on test data.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,) or (n_samples, n_outputs).
        """
        X_test = self._to_tensor(X_test)
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_test)
            return preds.cpu().numpy()

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Not applicable for regression models.

        Raises:
            NotImplementedError: This method is not applicable for regression.
        """
        raise NotImplementedError("predict_proba is not applicable for regression models")
