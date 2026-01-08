import logging
from typing import Any

import numpy as np

from counterfactuals.metrics.base import Metric
from counterfactuals.metrics.utils import register_metric

logger = logging.getLogger(__name__)


@register_metric("coverage")
class Coverage(Metric):
    """
    Coverage metric - proportion of instances for which counterfactuals were generated.

    Computed as 1 - proportion of instances with NaN values.
    """

    name = "coverage"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_cf"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the coverage metric.

        Returns:
            float: Coverage metric value (proportion without NaNs).
        """
        X_cf = inputs["X_cf"]
        # Check how many vectors of dim 0 contain NaN in X_cf
        return 1 - np.isnan(X_cf).any(axis=1).mean()


@register_metric("validity")
class Validity(Metric):
    """
    Validity metric - proportion of counterfactuals that changed the prediction.

    Computed as the proportion of counterfactuals where y_cf != y_test.
    """

    name = "validity"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_cf", "y_test", "disc_model"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the validity metric.

        Returns:
            float: Validity metric value (proportion with changed predictions).
        """
        disc_model = inputs["disc_model"]
        y_test = inputs["y_test"]
        X_cf = inputs["X_cf"]
        y_cf_pred = disc_model.predict(X_cf)
        if not isinstance(y_cf_pred, np.ndarray):
            y_cf_pred = y_cf_pred.numpy()
        return (y_cf_pred != y_test.squeeze()).mean()


@register_metric("actionability")
class Actionability(Metric):
    """
    Actionability metric - proportion of counterfactuals that are identical to originals.

    This metric identifies cases where no changes were made (which is generally undesirable).
    """

    name = "actionability"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_test", "X_cf"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the actionability metric.

        Returns:
            float: Actionability metric value (proportion identical to originals).
        """
        X_test = inputs["X_test"]
        X_cf = inputs["X_cf"]
        return np.all(X_test == X_cf, axis=1).mean()


@register_metric("sparsity")
class Sparsity(Metric):
    """
    Sparsity metric - average proportion of features that changed.

    Lower sparsity means fewer features were changed, which is generally desirable.
    """

    name = "sparsity"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_test", "X_cf"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the sparsity metric.

        Returns:
            float: Sparsity metric value (average proportion of changed features).
        """
        X_test = inputs["X_test"]
        X_cf = inputs["X_cf"]
        return (X_test != X_cf).mean()


@register_metric("number_of_instances")
class NumberOfInstances(Metric):
    """
    Number of instances metric - total count of counterfactual instances.
    """

    name = "number_of_instances"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"X_cf"}

    def __call__(self, **inputs: Any) -> float:
        """
        Compute the number of instances metric.

        Returns:
            float: Total number of instances in the counterfactual set.
        """
        X_cf = inputs["X_cf"]
        return float(X_cf.shape[0])
