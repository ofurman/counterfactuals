import numpy as np
import tensorflow as tf
from alibi.explainers import CounterFactualProto
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


class CEGP(BaseCounterfactualMethod, LocalCounterfactualMixin):
    def __init__(
        self,
        disc_model: PytorchBase,
        beta: float = 0.01,
        c_init: float = 1.0,
        c_steps: int = 5,
        max_iterations: int = 500,
        feature_range: tuple[float, float] = (0.0, 1.0),
        d_type: str = "abdm",
        disc_perc: list[int] | None = None,
        device: str | None = None,
        **kwargs,  # ignore other arguments
    ) -> None:
        """Initialize CEGP counterfactual method.

        Args:
            disc_model: Discriminative model to use for counterfactual generation
            beta: Trade-off parameter for distance computation
            c_init: Initial value of c for the attack loss term
            c_steps: Number of steps to adjust c
            max_iterations: Maximum number of iterations to run optimization
        """
        # Initialize base/mixin (moves model to device if applicable)
        super().__init__(disc_model=disc_model, device=device)

        tf.compat.v1.disable_eager_execution()
        session = ensure_tf_session()
        predict_proba = lambda x: disc_model.predict_proba(x)  # noqa: E731
        num_features = disc_model.num_inputs
        shape = (1, num_features)

        self.d_type = d_type
        self.disc_perc = [25, 50, 75] if disc_perc is None else disc_perc

        self.cf = CounterFactualProto(
            predict_proba,
            shape,
            beta=beta,
            max_iterations=max_iterations,
            feature_range=feature_range,
            c_init=c_init,
            c_steps=c_steps,
            sess=session,
        )

        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> None:
        """Fit the CEGP model on training data.

        Args:
            X_train: Training data to fit the model on
        """
        self.cf.fit(
            X_train.astype(np.float32),
            d_type=self.d_type,
            disc_perc=self.disc_perc,
        )
        self.is_fitted = True

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactual explanations for given samples.

        Args:
            X: Samples to explain
            y_origin: Original labels
            y_target: Target labels
            X_train: Training data (used for fitting if not already fitted)
            y_train: Training labels
        """
        if X_train is not None and not self.is_fitted:
            self.fit(X_train)

        try:
            X_in = X.copy()
            X_proc = X.reshape((1,) + X.shape)
            explanation = self.cf.explain(X_proc).cf
            if explanation is None:
                raise ValueError("No counterfactual found")
            x_cfs = np.array(explanation.get("X"))
        except Exception as e:
            print(f"Error in CEGP explanation: {e}")
            x_cfs = X.reshape(1, -1)

        # Wrap results in ExplanationResult
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
        """Generate counterfactual explanations for all samples in dataloader.

        Args:
            dataloader: DataLoader containing samples to explain
            target_class: Target class for counterfactuals
        """
        Xs, ys = dataloader.dataset.tensors

        # Fit on first batch if not already fitted
        if not self.is_fitted:
            X_train = kwargs.get("X_train")
            fit_data = np.asarray(X_train) if X_train is not None else Xs.numpy()
            self.fit(fit_data)

        # create ys_target numpy array same shape as ys but with target class
        ys_target = np.full(ys.shape, target_class)
        Xs_cfs = []
        model_returned = []

        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            x_input = X.numpy().reshape(1, -1)
            try:
                explanation = self.cf.explain(x_input).cf
                if explanation is None:
                    raise ValueError("No counterfactual found")
                Xs_cfs.append(np.asarray(explanation["X"]).reshape(-1))
                model_returned.append(True)
            except Exception as e:
                print(f"Error in CEGP explanation: {e}")
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
