import numpy as np
import tensorflow as tf
from alibi.explainers import CEM
from torch.utils.data import DataLoader
from tqdm import tqdm

from counterfactuals.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from counterfactuals.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)
from counterfactuals.cf_methods.tf_compat import ensure_tf_session
from counterfactuals.models.pytorch_base import PytorchBase


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
        no_info_type: str = "median",
        feature_range: tuple[float, float] = (0.0, 1.0),
        clip: tuple[float, float] = (-1000.0, 1000.0),
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

        self.no_info_type = no_info_type

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
        self.cf.fit(X_train, no_info_type=self.no_info_type)

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
            x_cfs = X.reshape(1, -1)

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
            X_train = kwargs.get("X_train")
            fit_data = np.asarray(X_train) if X_train is not None else Xs.numpy()
            self.fit(fit_data)

        # Create target labels array
        ys_target = np.full(ys.shape, target_class)

        Xs_cfs = []
        model_returned = []

        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            x_input = X.numpy().reshape(1, -1)
            try:
                explanation = self.cf.explain(x_input, verbose=False)
                if explanation.PN is None:
                    raise ValueError("No counterfactual found")
                Xs_cfs.append(np.asarray(explanation.PN).reshape(-1))
                model_returned.append(True)
            except Exception as e:
                print(e)
                Xs_cfs.append(x_input.reshape(-1))
                model_returned.append(False)

        Xs_cfs = np.stack(Xs_cfs, axis=0)
        Xs = Xs.numpy()
        ys = ys.numpy()
        ys_target = np.array(ys_target)

        return ExplanationResult(
            x_cfs=Xs_cfs,
            y_cf_targets=ys_target,
            x_origs=Xs,
            y_origs=ys,
            logs={"model_returned": model_returned},
        )
