import numpy as np
import tensorflow as tf
from alibi.explainers import CEM
from torch.utils.data import DataLoader
from tqdm import tqdm

from cel.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from cel.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)
from cel.cf_methods.tf_compat import ensure_tf_session
from cel.models.pytorch_base import PytorchBase


class CEM_CF(BaseCounterfactualMethod, LocalCounterfactualMixin):
    def __init__(
        self,
        disc_model: PytorchBase,
        mode: str = "PN",
        kappa: float = 0.2,
        beta: float = 0.1,
        c_init: float = 10.0,
        c_steps: int = 5,
        max_iterations: int = 200,
        learning_rate_init: float = 1e-2,
        device: str | None = None,
        **kwargs,  # ignore other arguments
    ) -> None:
        # Initialize base/mixin (moves model to device if applicable)
        super().__init__(disc_model=disc_model, device=device)

        tf.compat.v1.disable_eager_execution()
        ensure_tf_session()
        predict_proba = lambda x: disc_model.predict_proba(x)  # noqa: E731
        num_features = disc_model.num_inputs
        shape = (1, num_features)

        # Set gradient clipping
        clip = (-1000.0, 1000.0)

        # Get feature ranges from model
        feature_range = (0, 1)  # Default range, should be adjusted based on data

        self.cf = CEM(
            predict_proba,
            mode=mode,
            shape=shape,
            kappa=kappa,
            beta=beta,
            feature_range=feature_range,
            max_iterations=max_iterations,
            c_init=c_init,
            c_steps=c_steps,
            learning_rate_init=learning_rate_init,
            clip=clip,
        )

    def fit(self, X_train: np.ndarray) -> None:
        """Fit the CEM model on training data"""
        self.cf.fit(X_train, no_info_type="median")

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        **kwargs,
    ) -> ExplanationResult:
        if X_train is not None and not hasattr(self.cf, "X_train"):
            self.fit(X_train)

        try:
            X_in = X.copy()
            X_proc = X.reshape((1,) + X.shape)
            explanation = self.cf.explain(X_proc, verbose=False)
            if explanation is None or getattr(explanation, "PN", None) is None:
                raise ValueError("No counterfactual found")
            x_cfs = np.array(explanation.PN)
        except Exception as e:
            print(e)
            x_cfs = np.full_like(X.reshape(1, -1), np.nan).squeeze()

        return ExplanationResult(
            x_cfs=np.array(x_cfs),
            y_cf_targets=np.array(y_target),
            x_origs=np.array(X_in),
            y_origs=np.array(y_origin),
            logs=None,
        )

    def explain_dataloader(
        self, dataloader: DataLoader, target_class: int, *args, **kwargs
    ) -> ExplanationResult:
        Xs, ys = dataloader.dataset.tensors

        # Fit on all training data if not already fitted
        if not hasattr(self.cf, "X_train"):
            self.fit(Xs.numpy())

        # Create target labels array
        ys_target = np.full(ys.shape, target_class)

        Xs_cfs = []
        model_returned = []

        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            try:
                X = X.reshape((1,) + X.shape)
                explanation = self.cf.explain(X, verbose=False)
                if explanation.PN is None:
                    raise ValueError("No counterfactual found")
                Xs_cfs.append(explanation.PN)
                model_returned.append(True)
            except Exception as e:
                print(e)
                explanation = np.empty_like(X.reshape(1, -1))
                explanation[:] = np.nan
                Xs_cfs.append(explanation)
                model_returned.append(False)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs = np.array(Xs)
        ys = np.array(ys)
        ys_target = np.array(ys_target)

        return ExplanationResult(
            x_cfs=Xs_cfs,
            y_cf_targets=ys_target,
            x_origs=Xs,
            y_origs=ys,
            logs={"model_returned": model_returned},
        )
