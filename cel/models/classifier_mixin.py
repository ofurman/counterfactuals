from abc import ABC

import numpy as np
import torch


class ClassifierPytorchMixin(ABC):
    """Mixin providing classification interface for PyTorch models.

    Subclasses must define:
        - forward(x: torch.Tensor) -> torch.Tensor  (raw logits)
        - final_activation: torch.nn.Module          (e.g. Sigmoid, Softmax)
        - num_targets: int
    """

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make class predictions on test data.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        X_test = self._to_tensor(X_test)
        with torch.no_grad():
            probs = self.predict_proba(X_test)
            if isinstance(probs, np.ndarray):
                probs = torch.from_numpy(probs)
            predicted = torch.argmax(probs, dim=1)
            return predicted.squeeze().cpu().numpy()

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return probabilities for each class on test data.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes). Each row sums to 1.0.
        """
        X_test = self._to_tensor(X_test)
        with torch.no_grad():
            logits = self.forward(X_test)
            probs = self.final_activation(logits)
            if self.num_targets == 1:
                probs = torch.hstack([1 - probs, probs])
            return probs.cpu().numpy()
