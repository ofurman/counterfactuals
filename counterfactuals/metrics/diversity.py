"""Diversity-related metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from counterfactuals.metrics.base import Metric
from counterfactuals.metrics.utils import register_metric


@register_metric("pairwise_distance_diversity")
class PairwiseDistanceDiversity(Metric):
    """Average pairwise Euclidean distance between counterfactuals per factual."""

    name = "pairwise_distance_diversity"

    def required_inputs(self) -> set[str]:
        return {"X_cf", "cf_group_ids"}

    def __call__(self, **inputs: Any) -> float:
        X_cf: np.ndarray = inputs["X_cf"]
        cf_group_ids = inputs["cf_group_ids"]
        if cf_group_ids is None:
            raise ValueError(
                "cf_group_ids are required to compute pairwise distance diversity."
            )

        cf_group_ids = np.asarray(cf_group_ids)
        if cf_group_ids.shape[0] != X_cf.shape[0]:
            raise ValueError("cf_group_ids must align with X_cf rows.")

        diversities: list[float] = []
        for group_id in np.unique(cf_group_ids):
            group_points = X_cf[cf_group_ids == group_id]
            group_points = group_points[~np.isnan(group_points).any(axis=1)]
            if group_points.shape[0] < 2:
                continue
            diffs = group_points[:, None, :] - group_points[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            upper = dists[np.triu_indices(group_points.shape[0], k=1)]
            if upper.size > 0:
                diversities.append(float(upper.mean()))

        if not diversities:
            return 0.0
        return float(np.mean(diversities))
