"""Orchestrator for computing counterfactual metrics."""

import logging
from typing import Any, Optional, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from counterfactuals.metrics.utils import _METRIC_REGISTRY
from counterfactuals.metrics.validation import convert_to_numpy, validate_metric_inputs

logger = logging.getLogger(__name__)


class MetricsOrchestrator:
    """
    Orchestrator for computing counterfactual metrics.

    Args:
        X_cf: Counterfactual instances.
        y_target: Target labels for counterfactual instances.
        X_train: Training instances.
        y_train: Training labels.
        X_test: Test instances.
        y_test: Test labels.
        gen_model: Generator model.
        disc_model: Discriminator model.
        continuous_features: List of indices of continuous features.
        categorical_features: List of indices of categorical features.
        ratio_cont: Ratio of continuous features to be perturbed. Defaults to None.
        prob_plausibility_threshold: Log Likelihood Threshold for prob. plausibility.
    """

    def __init__(
        self,
        X_cf: Union[np.ndarray, torch.Tensor],
        y_target: Union[np.ndarray, torch.Tensor],
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_test: Union[np.ndarray, torch.Tensor],
        y_test: Union[np.ndarray, torch.Tensor],
        disc_model: torch.nn.Module,
        continuous_features: list[int],
        categorical_features: list[int],
        ratio_cont: Optional[float] = None,
        gen_model: Optional[torch.nn.Module] = None,
        prob_plausibility_threshold: Optional[float] = None,
        metrics_conf_path: str = "counterfactuals/pipelines/conf/metrics/default.yaml",
    ) -> None:
        """Initialize the metrics orchestrator with data and models."""
        self.metrics_to_compute = OmegaConf.load(metrics_conf_path).metrics_to_compute
        # Convert everything to numpy arrays if not already
        self.X_cf = convert_to_numpy(X_cf)
        self.y_target = convert_to_numpy(np.squeeze(y_target))
        self.X_train = convert_to_numpy(X_train)
        self.y_train = convert_to_numpy(y_train)
        self.X_test = convert_to_numpy(X_test)
        self.y_test = convert_to_numpy(y_test)

        # Validate all inputs once at initialization
        validate_metric_inputs(
            X_cf=self.X_cf,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            y_target=self.y_target,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            ratio_cont=ratio_cont,
        )

        # Store models and set to evaluation mode
        self.disc_model = disc_model.eval()
        self.gen_model = gen_model

        # Store threshold
        self.prob_plausibility_threshold = (
            prob_plausibility_threshold.item()
            if isinstance(prob_plausibility_threshold, torch.Tensor)
            else prob_plausibility_threshold
        )

        # Store feature information
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.ratio_cont = ratio_cont

        # Compute predictions for counterfactuals
        self.y_cf_pred = convert_to_numpy(self.disc_model.predict(self.X_cf))

        # Filter only valid counterfactuals and test instances
        self.X_cf_valid = self.X_cf[self.y_cf_pred == self.y_target]
        self.X_test_valid = self.X_test[self.y_cf_pred == self.y_target]

        logger.info(
            f"Initialized MetricsOrchestrator with {len(self.X_cf)} counterfactuals, "
            f"{len(self.X_cf_valid)} valid"
        )

    def _prepare_inputs(self) -> dict[str, Any]:
        """
        Prepare all inputs that metrics might need.

        Returns:
            Dictionary of all available inputs for metrics.
        """
        return {
            "X_cf": self.X_cf,
            "X_test": self.X_test,
            "X_train": self.X_train,
            "X_cf_valid": self.X_cf_valid,
            "X_test_valid": self.X_test_valid,
            "y_target": self.y_target,
            "y_test": self.y_test,
            "y_train": self.y_train,
            "y_cf_pred": self.y_cf_pred,
            "gen_model": self.gen_model,
            "disc_model": self.disc_model,
            "continuous_features": self.continuous_features,
            "categorical_features": self.categorical_features,
            "ratio_cont": self.ratio_cont,
            "prob_plausibility_threshold": self.prob_plausibility_threshold,
        }

    def calculate_all_metrics(self) -> dict[str, float]:
        """
        Calculate all registered metrics.

        Returns:
            Dictionary of metric names and values.
        """
        # Prepare all available inputs
        available_inputs = self._prepare_inputs()

        results: dict[str, float] = {}

        for metric_name in self.metrics_to_compute:
            # Get metric class from registry
            metric_cls = _METRIC_REGISTRY.get(metric_name)
            if metric_cls is None:
                logger.warning(
                    f"Metric '{metric_name}' not found in registry. "
                    f"Available: {list(_METRIC_REGISTRY.keys())}"
                )
                continue

            # Instantiate metric
            metric = metric_cls()

            # Get required inputs for this metric
            required_inputs = metric.required_inputs()

            # Check if all required inputs are available
            missing = [inp for inp in required_inputs if inp not in available_inputs]
            if missing:
                logger.warning(
                    f"Skipping metric '{metric_name}': missing inputs {missing}"
                )
                continue

            # Prepare kwargs for this metric
            kwargs = {k: available_inputs[k] for k in required_inputs}

            try:
                value = metric(**kwargs)
                results[metric_name] = value.item()
            except Exception as e:
                logger.error(f"Error computing metric '{metric_name}': {e}")
                continue

        return results
