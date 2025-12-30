import logging
from typing import List, Optional, Union

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from counterfactuals.metrics.distances import (
    categorical_distance,
    continuous_distance,
    distance_combined,
)
from counterfactuals.metrics.orchestrator import MetricsOrchestrator

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
        if X_cf.shape[1] != X_train.shape[1] or X_cf.shape[1] != X_test.shape[1]:
            raise ValueError("All input data should have the same number of features")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train should have the same number of samples"
            )
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("X_test and y_test should have the same number of samples")
        if X_cf.shape[0] != y_test.shape[0]:
            raise ValueError("X_cf and y_test should have the same number of samples")
        if len(continuous_features) + len(categorical_features) != X_cf.shape[1]:
            raise ValueError(
                "The sum of continuous and categorical features should equal the number of features in X_cf"
            )
        if ratio_cont is not None and not 0 <= ratio_cont <= 1:
            raise ValueError("ratio_cont should be between 0 and 1")

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
        y_cf = self._convert_to_numpy(self.disc_model.predict(self.X_cf))
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
    cf_group_ids: np.ndarray | None = None,
    metrics_conf_path: str = "counterfactuals/pipelines/conf/metrics/default.yaml",
):
    y_target = torch.abs(1 - torch.from_numpy(y_test)) if y_target is None else y_target
    y_target = y_target.numpy() if isinstance(y_target, torch.Tensor) else y_target

    metrics_cf = MetricsOrchestrator(
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
        metrics_conf_path=metrics_conf_path,
        cf_group_ids=cf_group_ids,
    )
    metrics = metrics_cf.calculate_all_metrics()
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


def evaluate_cf_for_pumal(
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
    S_matrix: np.ndarray | None = None,
    D_matrix: np.ndarray | None = None,
    X_test_target: np.ndarray | None = None,
    metrics_conf_path: str = "counterfactuals/pipelines/conf/metrics/group_metrics.yaml",
):
    cf_group_ids = None
    if S_matrix is not None:
        cf_group_ids = np.argmax(S_matrix, axis=1)

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
        cf_group_ids=cf_group_ids,
        metrics_conf_path=metrics_conf_path,
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

    if D_matrix is not None and D_matrix.shape[0] > 1:
        metrics.update(
            {
                "pairwise_cosine_sim_mean": pdist(D_matrix, "euclidean").mean(),
                "pairwise_cosine_sim_std": pdist(D_matrix, "euclidean").std(),
                "pairwise_cosine_sim_min": pdist(D_matrix, "euclidean").min(),
                "pairwise_cosine_sim_max": pdist(D_matrix, "euclidean").max(),
                "pairwise_euclidean_dist_mean": pdist(D_matrix, "cosine").mean(),
                "pairwise_euclidean_dist_std": pdist(D_matrix, "cosine").std(),
                "pairwise_euclidean_dist_min": pdist(D_matrix, "cosine").min(),
                "pairwise_euclidean_dist_max": pdist(D_matrix, "cosine").max(),
                "pairwise_wasserstein_dist_mean": pdist(
                    D_matrix, wasserstein_distance
                ).mean(),
                "pairwise_wasserstein_dist_std": pdist(
                    D_matrix, wasserstein_distance
                ).std(),
                "pairwise_wasserstein_dist_min": pdist(
                    D_matrix, wasserstein_distance
                ).min(),
                "pairwise_wasserstein_dist_max": pdist(
                    D_matrix, wasserstein_distance
                ).max(),
                "distance_to_centroid_mean": np.linalg.norm(
                    D_matrix - np.mean(D_matrix, axis=0), axis=1
                ).mean(),
                "distance_to_centroid_std": np.linalg.norm(
                    D_matrix - np.mean(D_matrix, axis=0), axis=1
                ).std(),
                "distance_to_centroid_min": np.linalg.norm(
                    D_matrix - np.mean(D_matrix, axis=0), axis=1
                ).min(),
                "distance_to_centroid_max": np.linalg.norm(
                    D_matrix - np.mean(D_matrix, axis=0), axis=1
                ).max(),
                "variance_of_distances_to_centroid": np.power(
                    np.linalg.norm(D_matrix - np.mean(D_matrix, axis=0), axis=1), 2
                ).mean(),
            }
        )

    return metrics


def evaluate_cf_for_glance(
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
    cf_group_ids: np.ndarray | None = None,
    metrics_conf_path: str = "counterfactuals/pipelines/conf/metrics/group_metrics.yaml",
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
        cf_group_ids=cf_group_ids,
        metrics_conf_path=metrics_conf_path,
    )

    if cf_group_ids is not None:
        unique_groups, counts = np.unique(cf_group_ids, return_counts=True)
        metrics.update(
            {
                "number_of_groups_direct": float(unique_groups.shape[0]),
                "group_size_mean": float(counts.mean()),
                "group_size_std": float(counts.std()),
                "group_size_min": float(counts.min()),
                "group_size_max": float(counts.max()),
            }
        )

    return metrics
