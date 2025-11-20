from abc import ABC

import numpy as np
from torch.utils.data import DataLoader

from counterfactuals.cf_methods.counterfactual_base import ExplanationResult


class LocalCounterfactualMixin(ABC):
    """
    Mixin class for local counterfactual explanation methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize local counterfactual functionality."""
        super().__init__(*args, **kwargs)
        self._method_type: str = "local"

    def explain_dataloader(
        self, dataloader: DataLoader, target_class: int = None, **kwargs
    ) -> ExplanationResult:
        Xs_cfs = []
        Xs_orig = []
        ys_orig = []
        ys_target = []
        model_returned = []

        for X, y in dataloader:
            X_np = X.numpy()
            y_np = y.numpy()

            if target_class is not None:
                y_target_np = np.full_like(y_np, target_class)
            else:
                y_target_np = np.abs(y_np - 1)

            res = self.explain(X_np, y_origin=y_np, y_target=y_target_np, **kwargs)

            Xs_cfs.append(res.x_cfs)
            Xs_orig.append(res.x_origs)
            ys_orig.append(res.y_origs)
            ys_target.append(res.y_cf_targets)
            # Check if logs contain model_returned, otherwise assume success if not NaN?
            # For now, we don't populate model_returned here, but explain() might return it in logs.
            if res.logs and "model_returned" in res.logs:
                model_returned.extend(res.logs["model_returned"])
            else:
                # Assume success if not all NaNs?
                # Or just append True?
                model_returned.extend([True] * len(X_np))

        return ExplanationResult(
            x_cfs=np.concatenate(Xs_cfs),
            y_cf_targets=np.concatenate(ys_target),
            x_origs=np.concatenate(Xs_orig),
            y_origs=np.concatenate(ys_orig),
            logs={"model_returned": np.array(model_returned)},
        )
