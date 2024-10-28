from typing import Optional, List, Union
import numpy as np
import torch

from counterfactuals.metrics.metrics import CFMetrics


class RegressionCFMetrics(CFMetrics):
    """
    Class for computing counterfactual metrics specific to regression problems.
    Extends the base CFMetrics class with regression-specific metrics.
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
        target_tolerance: float = 0.05,
    ):
        """
        Initialize RegressionCFMetrics with additional regression-specific parameters.

        Args:
            target_tolerance: Acceptable relative difference between predicted and target values
            (all other parameters are inherited from CFMetrics)
        """
        # precheck input assumptions
        assert (
            X_cf.shape[1] == X_train.shape[1] == X_test.shape[1]
        ), "All input data should have the same number of features"
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train should have the same number of samples"
        assert (
            X_test.shape[0] == y_test.shape[0]
        ), "X_test and y_test should have the same number of samples"
        assert (
            X_cf.shape[0] == y_test.shape[0]
        ), "X_cf and y_test should have the same number of samples"
        assert (
            len(continuous_features) + len(categorical_features) == X_cf.shape[1]
        ), "The sum of continuous and categorical features should equal the number of features in X_cf"
        assert (
            ratio_cont is None or 0 <= ratio_cont <= 1
        ), "ratio_cont should be between 0 and 1"

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
        self.X_cf_valid = self.X_cf
        self.X_test_valid = self.X_test
        self.target_tolerance = target_tolerance

    def validity(self) -> float:
        """
        Compute the validity metric for regression.
        A counterfactual is considered valid if its predicted value is within
        the target_tolerance of the target value.

        Returns:
            float: Proportion of valid counterfactuals
        """
        y_cf_pred = self.disc_model.predict(self.X_cf).numpy().reshape(-1)
        relative_diff = np.abs(y_cf_pred - self.y_test)
        return np.mean(relative_diff)

    def target_achievement(self) -> float:
        """
        Compute how close the counterfactuals get to their target values on average.

        Returns:
            float: Mean relative difference between predicted and target values
        """
        y_cf_pred = self.disc_model.predict(self.X_cf).numpy().reshape(-1)
        relative_diff = np.abs(y_cf_pred - self.y_target)
        return np.mean(relative_diff)

    def calc_all_metrics(self) -> dict:
        """
        Calculate all metrics, including regression-specific ones.

        Returns:
            dict: Dictionary containing all metric values
        """
        # Get base metrics
        metrics = {
            "coverage": self.coverage(),
            "validity": self.validity(),
            "target_achievement": self.target_achievement(),
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


def evaluate_cf_regression(
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
    median_log_prob: float,
    target_tolerance: float = 0.1,
) -> dict:
    """
    Evaluate counterfactuals for regression problems.

    Args:
        target_tolerance: Acceptable relative difference between predicted and target values
        (all other parameters are the same as in the base evaluate_cf function)

    Returns:
        dict: Dictionary containing all metric values
    """
    metrics_cf = RegressionCFMetrics(
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
        target_tolerance=target_tolerance,
    )

    metrics = metrics_cf.calc_all_metrics()
    return metrics
