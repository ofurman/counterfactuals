import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.discriminative_models.base import BaseDiscModel
from counterfactuals.cf_methods.sace.blackbox import BlackBox
from counterfactuals.cf_methods.sace.casebased_sace import (
    CaseBasedSACE as OrigCaseBasedSACE,
)


class CaseBasedSACE(BaseCounterfactual):
    def __init__(
        self,
        disc_model: BaseDiscModel,
        variable_features=None,
        weights=None,
        metric=("euclidean", "jaccard"),
        feature_names=None,
        continuous_features=None,
        categorical_features_lists=None,
        normalize=False,
        random_samples=None,
        diff_features=10,
        tolerance=0.001,
        **kwargs,
    ) -> None:
        self.disc_model = disc_model
        self.variable_features = variable_features
        self.weights = weights
        self.metric = metric
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features_lists = categorical_features_lists
        self.normalize = normalize
        self.random_samples = random_samples
        self.diff_features = diff_features
        self.tolerance = tolerance

        self.cf = None
        self.bb = None

    def fit(self, X_train):
        self.bb = BlackBox(self.disc_model)
        self.cf = OrigCaseBasedSACE(
            variable_features=self.variable_features,
            weights=self.weights,
            metric=self.metric,
            feature_names=self.feature_names,
            continuous_features=self.continuous_features,
            categorical_features_lists=self.categorical_features_lists,
            normalize=self.normalize,
            random_samples=self.random_samples,
            diff_features=self.diff_features,
            tolerance=self.tolerance,
        )
        self.cf.fit(self.bb, X_train)

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        if self.cf is None:
            self.fit(X_train)

        try:
            x_cf = self.cf.get_counterfactuals(X, k=1)
            model_returned = True
        except Exception as e:
            x_cf = np.full_like(X, np.nan)
            model_returned = False
            print(e)

        return x_cf, X, y_origin, y_target, model_returned

    def explain_dataloader(
        self,
        dataloader: DataLoader,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *args,
        **kwargs,
    ):
        if self.cf is None:
            self.fit(X_train)

        Xs, ys = dataloader.dataset.tensors
        Xs = Xs.numpy()
        ys = ys.numpy()

        Xs_cfs = []
        model_returned = []

        for X in tqdm(Xs, total=len(Xs)):
            try:
                x_cf = self.cf.get_counterfactuals(X, k=1)
                model_returned.append(True)
            except Exception as e:
                x_cf = np.full_like(X, np.nan)
                model_returned.append(False)
                print(e)
            Xs_cfs.append(x_cf)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        ys_target = np.abs(1 - ys)  # Assuming binary classification

        return Xs_cfs, Xs, ys, ys_target, model_returned
