import numpy as np
from tqdm import tqdm
import tensorflow as tf
from torch.utils.data import DataLoader
from alibi.explainers import CEM

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.discriminative_models.base import BaseDiscModel


class CEM_CF(BaseCounterfactual):
    def __init__(
        self,
        disc_model: BaseDiscModel,
        mode: str = "PN",
        kappa: float = 0.2,
        beta: float = 0.1,
        c_init: float = 10.0,
        c_steps: int = 5,
        max_iterations: int = 200,
        learning_rate_init: float = 1e-2,
        **kwargs,  # ignore other arguments
    ) -> None:
        tf.compat.v1.disable_eager_execution()
        predict_proba = lambda x: disc_model.predict_proba(x).numpy()  # noqa: E731
        num_features = disc_model.input_size
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

    def fit(self, X_train: np.ndarray):
        """Fit the CEM model on training data"""
        self.cf.fit(X_train, no_info_type="median")

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        if not hasattr(self.cf, "X_train"):
            self.fit(X_train)

        try:
            X = X.reshape((1,) + X.shape)
            explanation = self.cf.explain(X, verbose=False)
            if explanation.PN is None:
                raise ValueError("No counterfactual found")
        except Exception as e:
            explanation = None
            print(e)
        return explanation.PN if explanation else None, X, y_origin, y_target

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
        return Xs_cfs, Xs, ys, ys_target, model_returned
