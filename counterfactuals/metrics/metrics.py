import logging
from typing import List, Optional, Union

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from counterfactuals.metrics.distances import (
    categorical_distance,
    continuous_distance,
    distance_combined,
)

logger = logging.getLogger(__name__)


class CFMetrics:
    """
    Class for computing counterfactual metrics.
    Args:
        X_cf (np.ndarray | torch.Tensor): Counterfactual instances.
        y_target (np.ndarray | torch.Tensor): Target labels for counterfactual instances.
        X_train (np.ndarray | torch.Tensor): Training instances.
        y_train (np.ndarray | torch.Tensor): Training labels.
        X_test (np.ndarray | torch.Tensor): Test instances.
        y_test (np.ndarray | torch.Tensor): Test labels.
        gen_model (torch.nn.Module): Generator model.
        disc_model (torch.nn.Module): Discriminator model.
        continuous_features (list[int]): List of indices of continuous features.
        categorical_features (list[int]): List of indices of categorical features.
        ratio_cont (Optional[float], optional): Ratio of continuous features to be perturbed. Defaults to None.
        prob_plausibility_threshold (Optional[float], optional): Log Likelihood Threshold for prob. plausibility. Defaults to None.
    """

    def __init__(
        self,
        X_cf: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_test: Union[np.ndarray, torch.Tensor],
        y_test: Union[np.ndarray, torch.Tensor],
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        continuous_features: List[int],
        categorical_features: List[int],
        ratio_cont: Optional[float] = None,
        prob_plausibility_threshold: Optional[float] = None,
    ) -> None:
        # precheck input assumptions
        assert X_cf.shape[1] == X_train.shape[1] == X_test.shape[1], (
            "All input data should have the same number of features"
        )
        assert X_train.shape[0] == y_train.shape[0], (
            "X_train and y_train should have the same number of samples"
        )
        assert X_test.shape[0] == y_test.shape[0], (
            "X_test and y_test should have the same number of samples"
        )
        assert X_cf.shape[0] == y_test.shape[0], (
            "X_cf and y_test should have the same number of samples"
        )
        assert len(continuous_features) + len(categorical_features) == X_cf.shape[1], (
            "The sum of continuous and categorical features should equal the number of features in X_cf"
        )
        assert ratio_cont is None or 0 <= ratio_cont <= 1, (
            "ratio_cont should be between 0 and 1"
        )

        # convert everything to torch tensors if not already
        self.X_cf = self._convert_to_numpy(X_cf)
        self.y_target = self._convert_to_numpy(np.squeeze(y_target))
        self.X_train = self._convert_to_numpy(X_train)
        self.y_train = self._convert_to_numpy(y_train)
        self.X_test = self._convert_to_numpy(X_test)
        self.y_test = self._convert_to_numpy(y_test)

        # write class properties
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.prob_plausibility_threshold = (
            prob_plausibility_threshold.item()
            if isinstance(prob_plausibility_threshold, torch.Tensor)
            else prob_plausibility_threshold
        )

        # set models to evaluation mode
        self.gen_model = self.gen_model.eval()
        self.disc_model = self.disc_model.eval()

        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.ratio_cont = ratio_cont

        # filter only valid counterfactuals and test instances
        self.y_cf_pred = self._convert_to_numpy(self.disc_model.predict(self.X_cf))
        self.X_cf_valid = self.X_cf[self.y_cf_pred == self.y_target]
        self.X_test_valid = self.X_test[self.y_cf_pred == self.y_target]

    def _convert_to_numpy(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert input data to numpy array.

        Args:
            X (np.ndarray | torch.Tensor): Input data.

        Returns:
            np.ndarray: Converted array.

        Raises:
            ValueError: If X is neither a numpy array nor a torch tensor.
        """
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, torch.Tensor):
            return X.detach().numpy()
        else:
            raise ValueError("X should be either a numpy array or a torch tensor")

    def coverage(self) -> float:
        """
        Compute the coverage metric.

        Returns:
            float: Coverage metric value.
        """
        # check how many vectors of dim 0 contain NaN in X_cf
        return 1 - np.isnan(self.X_cf).any(axis=1).mean()

    def validity(self) -> float:
        """
        Compute the validity metric.

        Returns:
            float: Validity metric value.
        """
        y_cf = self.disc_model.predict(self.X_cf).numpy()
        return (y_cf != self.y_test.squeeze()).mean()

    def actionability(self) -> float:
        """
        Compute the actionability metric.

        Returns:
            float: Actionability metric value.
        """
        return np.all(self.X_test == self.X_cf, axis=1).mean()

    def sparsity(self) -> float:
        """
        Compute the sparsity metric.

        Returns:
            float: Sparsity metric value.
        """
        return (self.X_test != self.X_cf).mean()

    def prob_plausibility(self, cf: bool = True) -> float:
        """
        Compute the probability plausibility metric.
        This metric is computed as the average number of counterfactuals that are more plausible than the threshold.
        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.

        Returns:
            float: Avg number of counterfactuals that are more plausible than the threshold.
        """
        X = self.X_cf if cf else self.X_test
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(self.y_target).float()
        gen_log_probs = self.gen_model(X, y).detach().numpy()
        return (gen_log_probs > self.prob_plausibility_threshold).mean()

    def log_density(self, cf: bool = True) -> float:
        """
        Compute the log density metric.
        This metric is computed as the average log density of the counterfactuals.
        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.

        Returns:
            float: Average log density of the counterfactuals.
        """
        X = self.X_cf if cf else self.X_test
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(self.y_target).float()
        gen_log_probs = self.gen_model(X, y).detach().numpy()
        return gen_log_probs.mean()

    def lof_scores(self, cf: bool = True, n_neighbors: int = 20) -> float:
        """
        Compute the Local Outlier Factor (LOF) metric.
        This metric is computed as the average LOF score of the counterfactuals.
        LOF(k) ~ 1 means Similar density as neighbors,

        LOF(k) < 1 means Higher density than neighbors (Inlier),

        LOF(k) > 1 means Lower density than neighbors (Outlier)
        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.
            n_neighbors (int, optional): Number of neighbors to consider. Defaults to 20.

        Returns:
            float: Average LOF score of the counterfactuals.
        """
        X = self.X_cf if cf else self.X_test
        X_train = self.X_train

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(X_train)

        # ORIG: It is the opposite as bigger is better, i.e. large values correspond to inliers.
        # NEG: smaller is better, i.e. small values correspond to inliers.
        lof_scores = -lof.score_samples(X)
        return lof_scores.mean()

    def isolation_forest_scores(
        self, cf: bool = True, n_estimators: int = 100
    ) -> float:
        """
        Compute the Isolation Forest metric.
        This metric is computed as the average Isolation Forest score of the counterfactuals.
        The score is between -0.5 and 0.5, where smaller values mean more anomalous.
        https://stackoverflow.com/questions/45223921/what-is-the-range-of-scikit-learns-isolationforest-decision-function-scores#51882974

        The anomaly score of the input samples. The lower, the more abnormal.
        Negative scores represent outliers, positive scores represent inliers.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest.decision_function

        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.

        Returns:
            float: Average Isolation Forest score of the counterfactuals.
        """
        X = self.X_cf if cf else self.X_test
        X_train = self.X_train

        clf = IsolationForest(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train)
        lof_scores = clf.decision_function(X)
        return lof_scores.mean()

        # isolation_forest_scores = isolation_forest_metric(X_train, X, self.X_test, n_estimators) TODO: fix this
        # return isolation_forest_scores.mean()

    def feature_distance(
        self,
        continuous_metric: Optional[str] = "euclidean",
        categorical_metric: Optional[str] = "jaccard",
        X_train: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the distance metric.
        Args:
            cf (bool, optional): Whether to compute the metric for counterfactuals of for X_test. Defaults to True.
            metric (str, optional): Distance metric to use. Defaults to "euclidean".

        Returns:
            float: Distance metric value.
        """
        X = self.X_cf_valid
        X_test = self.X_test_valid

        if not any([continuous_metric, categorical_metric]):
            raise ValueError(
                "At least one of continuous_metric or categorical_metric should be provided"
            )
        elif categorical_metric is None:
            return continuous_distance(
                X_test=X_test,
                X_cf=X,
                continuous_features=self.continuous_features,
                metric=continuous_metric,
                X_all=X_train,
                agg="mean",
            )
        elif continuous_metric is None:
            return categorical_distance(
                X_test=X_test,
                X_cf=X,
                categorical_features=self.categorical_features,
                metric=categorical_metric,
                agg="mean",
            )
        else:
            return distance_combined(
                X_test=X_test,
                X_cf=X,
                X_all=X_train,
                continuous_metric=continuous_metric,
                categorical_metric=categorical_metric,
                continuous_features=self.continuous_features,
                categorical_features=self.categorical_features,
                ratio_cont=self.ratio_cont,
            )

    def target_distance(self, metric: str = "euclidean") -> float:
        """
        Compute the distance metric between targets (used for regression setup).

        Returns:
            float: Distance metric value between targets.
        """
        return continuous_distance(
            X_test=self.y_target,
            X_cf=self.y_cf_pred,
            continuous_features=[0],
            metric=metric,
            X_all=None,
            agg="mean",
        )

    def calc_all_metrics(self) -> dict:
        """
        Calculate all metrics.

        Returns:
            dict: Dictionary of metric names and values.
        """
        metrics = {
            "coverage": self.coverage(),
            "validity": self.validity(),
            "actionability": self.actionability(),
            "sparsity": self.sparsity(),
            # "target_distance": self.target_distance(),
            "proximity_categorical_hamming": self.feature_distance(
                categorical_metric="hamming"
            ),
            "proximity_categorical_jaccard": self.feature_distance(
                categorical_metric="jaccard"
            ),
            "proximity_continuous_manhattan": self.feature_distance(
                continuous_metric="cityblock"
            ),
            "proximity_continuous_euclidean": self.feature_distance(
                continuous_metric="euclidean"
            ),
            "proximity_continuous_mad": self.feature_distance(
                continuous_metric="mad", X_train=self.X_train
            ),
            "proximity_l2_jaccard": self.feature_distance(
                continuous_metric="euclidean", categorical_metric="jaccard"
            ),
            "proximity_mad_hamming": self.feature_distance(
                continuous_metric="mad",
                categorical_metric="hamming",
                X_train=self.X_train,
            ),
            "prob_plausibility": self.prob_plausibility(cf=True),
            "log_density_cf": self.log_density(cf=True),
            "log_density_test": self.log_density(cf=False),
            "lof_scores_cf": self.lof_scores(cf=True),
            "lof_scores_test": self.lof_scores(cf=False),
            "isolation_forest_scores_cf": self.isolation_forest_scores(cf=True),
            "isolation_forest_scores_test": self.isolation_forest_scores(cf=False),
        }
        return metrics


def evaluate_cf(
    disc_model: torch.nn.Module,
    gen_model: torch.nn.Module,
    X_cf: np.ndarray,
    model_returned: np.ndarray,
    continuous_features: list,
    categorical_features: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: np.ndarray,
    y_target: np.ndarray = None,
):
    y_target = torch.abs(1 - torch.from_numpy(y_test)) if y_target is None else y_target
    y_target = y_target.numpy() if isinstance(y_target, torch.Tensor) else y_target

    metrics_cf = CFMetrics(
        disc_model=disc_model,
        gen_model=gen_model,
        X_cf=X_cf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_target=y_target,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        ratio_cont=None,
        prob_plausibility_threshold=median_log_prob,
    )
    metrics = metrics_cf.calc_all_metrics()
    return metrics


def evaluate_cf_for_rppcef(
    disc_model: torch.nn.Module,
    gen_model: torch.nn.Module,
    X_cf: np.ndarray,
    model_returned: np.ndarray,
    continuous_features: list,
    categorical_features: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_target: np.ndarray,
    median_log_prob: np.ndarray,
    S_matrix: np.ndarray = None,
    X_test_target: np.ndarray = None,
):
    metrics = evaluate_cf(
        disc_model=disc_model,
        gen_model=gen_model,
        X_cf=X_cf,
        model_returned=model_returned,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_target=y_target,
        median_log_prob=median_log_prob,
    )

    if S_matrix is not None:
        cf_belongs_to_group = (
            np.sum(np.any(S_matrix == 1.0, axis=1)) / S_matrix.shape[0]
        )
        metrics.update(
            {
                "cf_belongs_to_group": cf_belongs_to_group,
                "K_vectors": (S_matrix.sum(axis=0) != 0).sum(),
            }
        )
    return metrics


# def median_abs_deviation(data, axis=None):
#     """
#     Calculate the Median Absolute Deviation (MAD) of a dataset along a specified axis.

#     Args:
#     data (list or numpy array): The input data for which the MAD is to be computed.
#     axis (int, optional): The axis along which the median should be computed.
#                           The default is None, which computes the MAD of the flattened array.

#     Returns:
#     numpy array or float: The MAD of the data along the given axis.
#     """
#     median = np.median(data, axis=axis)
#     if axis is None:
#         deviations = np.abs(data - median)
#     else:
#         deviations = np.abs(data - np.expand_dims(median, axis=axis))
#     mad = np.median(deviations, axis=axis)
#     return mad


# class DummyScaler:
#     def __init__(self):
#         pass

#     def fit(self, X):
#         pass

#     def fit_transform(self, X):
#         return X

#     def transform(self, X):
#         return X

#     def inverse_transform(self, X):
#         return X


# def valid_cf(y, y_cf):
#     return y_cf != y


# def number_valid_cf(y, y_cf):
#     val = torch.sum(valid_cf(y, y_cf))
#     return val


# def perc_valid_cf(y, y_cf):
#     n_val = number_valid_cf(y, y_cf=y_cf)
#     res = n_val / len(y)
#     return res


# def actionable_cf(X, X_cf, actionable_features: list):
#     # TODO: rewrite
#     assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
#     actionable = np.all((X == X_cf)[:, actionable_features], axis=1)
#     return actionable


# def number_actionable_cf(X, X_cf, actionable_features: list):
#     assert X.shape == X_cf.shape
#     number_actionable = np.sum(actionable_cf(X, X_cf, actionable_features), axis=1)
#     return number_actionable


# def perc_actionable_cf(X, X_cf, actionable_features: list):
#     assert X.shape == X_cf.shape
#     n_val = number_actionable_cf(X, X_cf, actionable_features)
#     res = n_val / len(X_cf)
#     return res


# def valid_actionable_cf(X, X_cf, y, y_cf, actionable_features):
#     valid = valid_cf(y, y_cf)
#     actionable = actionable_cf(X, X_cf, actionable_features)

#     assert valid.shape == actionable.shape
#     return np.logical_and(valid, actionable)


# def number_valid_actionable_cf(X, X_cf, y, y_cf, actionable_features):
#     return np.sum(valid_actionable_cf(X, X_cf, y, y_cf, actionable_features))


# def perc_valid_actionable_cf(X, X_cf, y, y_cf, actionable_features):
#     n_val = number_valid_actionable_cf(X, X_cf, y, y_cf, actionable_features)
#     return n_val / len(y)


# def number_violations_per_cf(X, X_cf, actionable_features: list):
#     assert X.shape == X_cf.shape
#     res = np.sum((X == X_cf)[:, actionable_features], axis=1)
#     return res


# def avg_number_violations_per_cf(X, X_cf, actionable_features):
#     return np.mean(number_violations_per_cf(X, X_cf, actionable_features))


# def avg_number_violations(X, X_cf, actionable_features):
#     val = np.sum(number_violations_per_cf(X, X_cf, actionable_features))
#     number_cf, number_features = X_cf.shape
#     return val / (number_cf * number_features)


# def sparsity(X, X_cf, actionable_features=None):
#     number_cf, number_features = X_cf.shape
#     val = X != X_cf
#     if actionable_features is not None:
#         val = val[:, actionable_features]
#     val = np.sum(val.numpy())
#     return val / (number_cf * number_features)


# def mad_cityblock(u, v, mad):
#     u = _validate_vector(u)
#     v = _validate_vector(v)
#     l1_diff = abs(u - v)
#     l1_diff_mad = l1_diff / mad
#     return l1_diff_mad.sum()


# def continuous_distance(
#     X, X_cf, continuous_features, metric="euclidean", X_all=None, agg="mean", _diag=True
# ):
#     assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
#     if metric == "mad":
#         mad = median_abs_deviation(X_all[:, continuous_features], axis=0)
#         mad = np.array([v if v != 0 else 1.0 for v in mad])

#         def _mad_cityblock(u, v):
#             return mad_cityblock(u, v, mad)

#         metric = _mad_cityblock

#     dist = cdist(X[:, continuous_features], X_cf[:, continuous_features], metric=metric)

#     dist = np.diag(dist) if _diag else dist
#     agg_funcs = {"mean": np.mean, "max": np.max, "min": np.min, "no": lambda x: x}
#     assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
#     return agg_funcs[agg](dist)


# def categorical_distance(
#     X, X_cf, categorical_features, metric="jaccard", agg=None, _diag=True
# ):
#     assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
#     dist = cdist(
#         X[:, categorical_features], X_cf[:, categorical_features], metric=metric
#     )
#     dist = np.diag(dist) if _diag else dist
#     agg_funcs = {"mean": np.mean, "max": np.max, "min": np.min, "no": lambda x: x}
#     assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
#     return agg_funcs[agg](dist)


# def distance_l2_jaccard(
#     X, X_cf, continuous_features, categorical_features, ratio_cont=None
# ):
#     number_features = X_cf.shape[1]
#     dist_cont = continuous_distance(
#         X, X_cf, continuous_features, metric="euclidean", X_all=None, agg="mean"
#     )
#     dist_cate = categorical_distance(
#         X, X_cf, categorical_features, metric="jaccard", agg="mean"
#     )
#     if ratio_cont is None:
#         ratio_continuous = len(continuous_features) / number_features
#         ratio_categorical = len(categorical_features) / number_features
#     else:
#         ratio_continuous = ratio_cont
#         ratio_categorical = 1.0 - ratio_continuous
#     dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
#     return dist


# def distance_mad_hamming(
#     X,
#     X_cf,
#     continuous_features,
#     categorical_features,
#     X_all,
#     ratio_cont=None,
#     agg=None,
#     diag=True,
# ):
#     number_features = X_cf.shape[1]
#     dist_cont = continuous_distance(
#         X, X_cf, continuous_features, metric="mad", X_all=X_all, agg=agg, _diag=diag
#     )
#     dist_cate = categorical_distance(
#         X, X_cf, categorical_features, metric="hamming", agg=agg, _diag=diag
#     )
#     if ratio_cont is None:
#         ratio_continuous = len(continuous_features) / number_features
#         ratio_categorical = len(categorical_features) / number_features
#     else:
#         ratio_continuous = ratio_cont
#         ratio_categorical = 1.0 - ratio_cont
#     dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
#     return dist


# def number_changes_per_cf(X, X_cf, continuous_features, agg="mean"):
#     assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
#     result = np.sum(X[:, continuous_features] == X_cf[:, continuous_features], axis=1)
#     agg_funcs = {
#         "mean": np.mean,
#         "max": np.max,
#         "min": np.min,
#         "sum": np.sum,
#         "no": lambda x: x,
#     }
#     assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
#     return agg_funcs[agg](result)


# def avg_number_changes(X, X_cf, continuous_features):
#     number_cf, number_features = X_cf.shape[1]
#     val = number_changes_per_cf(X, X_cf, continuous_features, agg="sum")
#     return val / (number_cf * number_features)


# def plausibility(
#     X_test,
#     X_cf,
#     y_test,
#     continuous_features_all,
#     categorical_features_all,
#     X_train,
#     ratio_cont=None,
# ):
#     dist_neighb = distance_mad_hamming(
#         X_test,
#         X_test,
#         continuous_features_all,
#         categorical_features_all,
#         X_train,
#         ratio_cont=ratio_cont,
#         agg="no",
#         diag=False,
#     )
#     dist_neighb[y_test == 0, y_test == 0] = np.inf
#     dist_neighb[y_test == 1, y_test == 1] = np.inf
#     idx_neighb = np.argmin(dist_neighb, axis=0)
#     dist_neighb = distance_mad_hamming(
#         X_test[idx_neighb],
#         X_cf,
#         continuous_features_all,
#         categorical_features_all,
#         X_train,
#         ratio_cont=ratio_cont,
#         agg="mean",
#         diag=True,
#     )
#     return dist_neighb


# def delta_proba(x, cf_list, classifier, agg=None):
#     y_val = classifier.predict_proba(x)
#     y_cf = classifier.predict_proba(cf_list)
#     deltas = np.abs(y_cf - y_val)

#     if agg is None or agg == "mean":
#         return np.mean(deltas)

#     if agg == "max":
#         return np.max(deltas)

#     if agg == "min":
#         return np.min(deltas)


# def calc_gen_model_density(gen_log_probs_cf, gen_log_probs_xs, ys):
#     log_density_cfs = []
#     log_density_xs = []
#     for y in np.unique(ys):
#         log_density_cfs.append(gen_log_probs_cf[y.astype(int), ys != y])
#         log_density_xs.append(gen_log_probs_xs[y.astype(int), ys == y])
#     return np.mean(np.hstack(log_density_cfs)), np.mean(np.hstack(log_density_xs))


# def local_outlier_factor(X_train, X, X_cf, n_neighbors=20):
#     """
#     Calculate the Local Outlier Factor (LOF) for each sample in X and X_cf.
#     LOF(k) ~ 1 means Similar density as neighbors,

#     LOF(k) < 1 means Higher density than neighbors (Inlier),

#     LOF(k) > 1 means Lower density than neighbors (Outlier)
#     """
#     lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
#     lof.fit(X_train)

#     # ORIG: It is the opposite as bigger is better, i.e. large values correspond to inliers.
#     # NEG: smaller is better, i.e. small values correspond to inliers.
#     lof_scores_x = -lof.score_samples(X)
#     lof_scores_x_cf = -lof.score_samples(X_cf)
#     return lof_scores_x, lof_scores_x_cf


# def isolation_forest_metric(X_train, X, X_cf, n_estimators=100):
#     """
#     Calculate the Isolation Forest score for each sample in X and X_cf.
#     The score is between -0.5 and 0.5, where smaller values mean more anomalous.
#     https://stackoverflow.com/questions/45223921/what-is-the-range-of-scikit-learns-isolationforest-decision-function-scores#51882974

#     The anomaly score of the input samples. The lower, the more abnormal.
#     Negative scores represent outliers, positive scores represent inliers.
#     See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest.decision_function
#     """
#     clf = IsolationForest(n_estimators=n_estimators, random_state=42)
#     clf.fit(X_train)
#     lof_scores_x = clf.decision_function(X)
#     lof_scores_x_cf = clf.decision_function(X_cf)
#     return lof_scores_x, lof_scores_x_cf


# def evaluate_cf(
#     disc_model: torch.nn.Module,
#     gen_model: torch.nn.Module,
#     X_cf: np.ndarray,
#     model_returned: np.ndarray,
#     continuous_features: list,
#     categorical_features: list,
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     delta: np.ndarray,
# ):
#     assert isinstance(X_cf, np.ndarray)
#     assert X_cf.dtype == np.float32
#     X_cf = np.atleast_2d(X_cf)

#     X_cf = X_cf[model_returned]

#     lof_scores_xs, lof_scores_cfs = local_outlier_factor(X_train, X_test, X_cf)
#     isolation_forest_scores_xs, isolation_forest_scores_cfs = isolation_forest_metric(
#         X_train, X_test, X_cf
#     )

#     X_test = X_test[model_returned]
#     y_test = y_test[model_returned]

#     ys_pred = disc_model.predict(X_test)
#     y_target = torch.abs(1 - ys_pred)

#     if X_cf.shape[0] == 0:
#         return {
#             "model_returned_smth": 0.0,
#         }

#     X_train = torch.from_numpy(X_train)
#     X_test = torch.from_numpy(X_test)
#     X_cf = torch.from_numpy(X_cf)

#     y_train = torch.from_numpy(y_train.reshape(-1))
#     y_test = torch.from_numpy(y_test.reshape(-1))

#     # Define variables for metrics
#     model_returned_smth = np.sum(model_returned) / len(model_returned)
#     ys_cfs_disc_pred = disc_model.predict(X_cf)
#     valid_cfs = ys_cfs_disc_pred != ys_pred
#     valid_cf_disc_metric = valid_cfs.float().mean().item()

#     X_cf = X_cf[valid_cfs]
#     X_test = X_test[valid_cfs]
#     y_test = y_test[valid_cfs]

#     ys_pred = disc_model.predict(X_test)
#     y_target = torch.abs(1 - ys_pred)

#     gen_log_probs_xs = gen_model(X_test, y_test.type(torch.float32))
#     gen_log_probs_cf = gen_model(X_cf, y_target.type(torch.float32))
#     flow_prob_condition_acc = torch.sum(delta < gen_log_probs_cf) / len(
#         gen_log_probs_cf
#     )

#     hamming_distance_metric = categorical_distance(
#         X=X_test,
#         X_cf=X_cf,
#         categorical_features=categorical_features,
#         metric="hamming",
#         agg="mean",
#     )
#     jaccard_distance_metric = categorical_distance(
#         X=X_test,
#         X_cf=X_cf,
#         categorical_features=categorical_features,
#         metric="jaccard",
#         agg="mean",
#     )
#     manhattan_distance_metric = continuous_distance(
#         X=X_test,
#         X_cf=X_cf,
#         continuous_features=continuous_features,
#         metric="cityblock",
#         X_all=X_test,
#     )
#     euclidean_distance_metric = continuous_distance(
#         X=X_test,
#         X_cf=X_cf,
#         continuous_features=continuous_features,
#         metric="euclidean",
#         X_all=X_test,
#     )
#     mad_distance_metric = continuous_distance(
#         X=X_test,
#         X_cf=X_cf,
#         continuous_features=continuous_features,
#         metric="mad",
#         X_all=X_test,
#     )
#     l2_jaccard_distance_metric = distance_l2_jaccard(
#         X=X_test,
#         X_cf=X_cf,
#         continuous_features=continuous_features,
#         categorical_features=categorical_features,
#     )
#     mad_hamming_distance_metric = distance_mad_hamming(
#         X=X_test,
#         X_cf=X_cf,
#         continuous_features=continuous_features,
#         categorical_features=categorical_features,
#         X_all=X_test,
#         agg="mean",
#     )
#     sparsity_metric = sparsity(X_test, X_cf)

#     # Create a dictionary of metrics
#     metrics = {
#         "model_returned_smth": model_returned_smth,
#         "valid_cf_disc": valid_cf_disc_metric,
#         "dissimilarity_proximity_categorical_hamming": hamming_distance_metric,
#         "dissimilarity_proximity_categorical_jaccard": jaccard_distance_metric,
#         "dissimilarity_proximity_continuous_manhatan": manhattan_distance_metric,
#         "dissimilarity_proximity_continuous_euclidean": euclidean_distance_metric,
#         "dissimilarity_proximity_continuous_mad": mad_distance_metric,
#         "distance_l2_jaccard": l2_jaccard_distance_metric,
#         "distance_mad_hamming": mad_hamming_distance_metric,
#         "sparsity": sparsity_metric,
#     }

#     metrics.update(
#         {
#             "flow_log_density_cfs": gen_log_probs_cf.mean().item(),
#             "flow_log_density_xs": gen_log_probs_xs.mean().item(),
#             "flow_prob_condition_acc": flow_prob_condition_acc.item(),
#             "lof_scores_xs": lof_scores_xs.mean(),
#             "lof_scores_cfs": lof_scores_cfs.mean(),
#             "isolation_forest_scores_xs": isolation_forest_scores_xs.mean(),
#             "isolation_forest_scores_cfs": isolation_forest_scores_cfs.mean(),
#         }
#     )
#     return metrics
