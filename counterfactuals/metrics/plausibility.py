import logging
from typing import Any

import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from counterfactuals.metrics.base import Metric
from counterfactuals.metrics.utils import register_metric

logger = logging.getLogger(__name__)


@register_metric("prob_plausibility")
class ProbabilisticPlausibility(Metric):
    """Probabilistic plausibility metric based on generative model log-likelihood."""

    name = "prob_plausibility"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {
            "X_cf",
            "y_target",
            "gen_model",
            "prob_plausibility_threshold",
        }

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the probability plausibility metric.

        This metric is computed as the average number of counterfactuals that
        are more plausible than the threshold.

        Returns:
            float: Average number of counterfactuals that are more plausible than the threshold.
        """
        X_cf = torch.from_numpy(inputs["X_cf"]).float()
        y_target = torch.from_numpy(inputs["y_target"]).float()
        gen_model = inputs["gen_model"].eval()

        with torch.no_grad():
            gen_log_probs = gen_model(X_cf, y_target).detach().numpy()

        threshold = inputs["prob_plausibility_threshold"]
        return (gen_log_probs > threshold).mean()


@register_metric("log_density_cf")
class LogDensityCF(Metric):
    """Log density metric for counterfactuals."""

    name = "log_density_cf"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_cf", "y_target", "gen_model"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the log density metric for counterfactuals.

        This metric is computed as the average log density of the counterfactuals.

        Returns:
            float: Average log density of the counterfactuals.
        """
        X_cf = torch.from_numpy(inputs["X_cf"]).float()
        y_target = torch.from_numpy(inputs["y_target"]).float()
        gen_model = inputs["gen_model"].eval()

        with torch.no_grad():
            gen_log_probs = gen_model(X_cf, y_target).detach().numpy()

        return gen_log_probs.mean()


@register_metric("log_density_test")
class LogDensityTest(Metric):
    """Log density metric for test instances."""

    name = "log_density_test"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_test", "y_test", "gen_model"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the log density metric for test instances.

        This metric is computed as the average log density of the test instances.

        Returns:
            float: Average log density of the test instances.
        """
        X_test = torch.from_numpy(inputs["X_test"]).float()
        y_test = torch.from_numpy(inputs["y_test"]).float()
        gen_model = inputs["gen_model"].eval()

        with torch.no_grad():
            gen_log_probs = gen_model(X_test, y_test).detach().numpy()

        return gen_log_probs.mean()


@register_metric("lof_scores_cf")
class LOFScoresCF(Metric):
    """Local Outlier Factor (LOF) scores for counterfactuals."""

    name = "lof_scores_cf"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_cf", "X_train"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the Local Outlier Factor (LOF) metric for counterfactuals.

        LOF(k) ~ 1 means Similar density as neighbors,
        LOF(k) < 1 means Higher density than neighbors (Inlier),
        LOF(k) > 1 means Lower density than neighbors (Outlier)

        Args:
            n_neighbors: Number of neighbors to consider. Defaults to 20.

        Returns:
            float: Average LOF score of the counterfactuals.
        """
        X_cf = inputs["X_cf"]
        X_train = inputs["X_train"]
        n_neighbors = inputs.get("n_neighbors", 20)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(X_train)

        # ORIG: It is the opposite as bigger is better, i.e. large values correspond to inliers.
        # NEG: smaller is better, i.e. small values correspond to inliers.
        lof_scores = -lof.score_samples(X_cf)
        return lof_scores.mean()


@register_metric("lof_scores_test")
class LOFScoresTest(Metric):
    """Local Outlier Factor (LOF) scores for test instances."""

    name = "lof_scores_test"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_test", "X_train"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the Local Outlier Factor (LOF) metric for test instances.

        LOF(k) ~ 1 means Similar density as neighbors,
        LOF(k) < 1 means Higher density than neighbors (Inlier),
        LOF(k) > 1 means Lower density than neighbors (Outlier)

        Args:
            n_neighbors: Number of neighbors to consider. Defaults to 20.

        Returns:
            float: Average LOF score of the test instances.
        """
        X_test = inputs["X_test"]
        X_train = inputs["X_train"]
        n_neighbors = inputs.get("n_neighbors", 20)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(X_train)

        # ORIG: It is the opposite as bigger is better, i.e. large values correspond to inliers.
        # NEG: smaller is better, i.e. small values correspond to inliers.
        lof_scores = -lof.score_samples(X_test)
        return lof_scores.mean()


@register_metric("isolation_forest_scores_cf")
class IsolationForestScoresCF(Metric):
    """Isolation Forest anomaly scores for counterfactuals."""

    name = "isolation_forest_scores_cf"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_cf", "X_train"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the Isolation Forest metric for counterfactuals.

        The score is between -0.5 and 0.5, where smaller values mean more anomalous.
        The anomaly score of the input samples. The lower, the more abnormal.
        Negative scores represent outliers, positive scores represent inliers.

        Args:
            n_estimators: Number of trees in the forest. Defaults to 100.

        Returns:
            float: Average Isolation Forest score of the counterfactuals.
        """
        X_cf = inputs["X_cf"]
        X_train = inputs["X_train"]
        n_estimators = inputs.get("n_estimators", 100)

        clf = IsolationForest(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train)
        scores = clf.decision_function(X_cf)
        return scores.mean()


@register_metric("isolation_forest_scores_test")
class IsolationForestScoresTest(Metric):
    """Isolation Forest anomaly scores for test instances."""

    name = "isolation_forest_scores_test"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_test", "X_train"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the Isolation Forest metric for test instances.

        The score is between -0.5 and 0.5, where smaller values mean more anomalous.
        The anomaly score of the input samples. The lower, the more abnormal.
        Negative scores represent outliers, positive scores represent inliers.

        Args:
            n_estimators: Number of trees in the forest. Defaults to 100.

        Returns:
            float: Average Isolation Forest score of the test instances.
        """
        X_test = inputs["X_test"]
        X_train = inputs["X_train"]
        n_estimators = inputs.get("n_estimators", 100)

        clf = IsolationForest(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train)
        scores = clf.decision_function(X_test)
        return scores.mean()
