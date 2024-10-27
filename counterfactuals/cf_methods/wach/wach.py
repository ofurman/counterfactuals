import numpy as np
from tqdm import tqdm
import tensorflow as tf
from torch.utils.data import DataLoader
from alibi.explainers import Counterfactual

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.discriminative_models.base import BaseDiscModel


class WACH(BaseCounterfactual):
    def __init__(
        self,
        disc_model: BaseDiscModel,
        target_class: int = "other",  # any class other than origin will do
        **kwargs,  # ignore other arguments
    ) -> None:
        tf.compat.v1.disable_eager_execution()
        target_proba = 1.0
        tol = 0.51  # want counterfactuals with p(class)>0.99
        self.target_class = target_class
        max_iter = 1000
        lam_init = 1e-1
        max_lam_steps = 10
        learning_rate_init = 0.1
        predict_proba = lambda x: disc_model.predict_proba(x).numpy()  # noqa: E731
        num_features = disc_model.input_size

        # TODO: Change in future to allow for different feature ranges
        feature_range = (0, 1)

        self.cf = Counterfactual(
            predict_proba,
            shape=(1, num_features),
            target_proba=target_proba,
            tol=tol,
            target_class=target_class,
            max_iter=max_iter,
            lam_init=lam_init,
            max_lam_steps=max_lam_steps,
            learning_rate_init=learning_rate_init,
            feature_range=feature_range,
        )

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        try:
            X = X.reshape((1,) + X.shape)
            explanation = self.cf.explain(X).cf
        except Exception as e:
            explanation = None
            print(e)
        return explanation, X, y_origin, y_target
        # return ExplanationResult(
        #     x_cfs=explanation, y_cf_targets=y_target, x_origs=X, y_origs=y_origin
        # )

    def explain_dataloader(
        self, dataloader: DataLoader, target_class: int, *args, **kwargs
    ) -> ExplanationResult:
        Xs, ys = dataloader.dataset.tensors
        # create ys_target numpy array same shape as ys but with target class
        ys_target = np.full(ys.shape, target_class)
        Xs_cfs = []
        model_returned = []
        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            try:
                X = X.reshape((1,) + X.shape)
                explanation = self.cf.explain(X).cf["X"]
                model_returned.append(True)
            except Exception as e:
                explanation = [None, None]
                print(e)
                model_returned.append(False)
            Xs_cfs.append(explanation)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs = np.array(Xs)
        ys = np.array(ys)
        ys_target = np.array(ys_target)
        return Xs_cfs, Xs, ys, ys_target, model_returned
        # return ExplanationResult(x_cfs=Xs_cfs, y_cf_targets=ys, x_origs=Xs, y_origs=ys)
