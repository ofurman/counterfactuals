"""Metrics aligned with CETGFN DiCoFlex evaluation.

Each metric here mirrors the formula used in
``rgfn/trainer/metrics/counterfactual_metrics.py`` from the CETGFN project
so that results produced by CEL pipelines are directly comparable.

All metrics (except validity) operate on **valid** counterfactuals only,
i.e. instances where the discriminator prediction on X_cf matches y_target.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import LocalOutlierFactor

from cel.metrics.base import Metric
from cel.metrics.utils import register_metric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proximity – L1 on continuous features only (valid CFs)
# ---------------------------------------------------------------------------


@register_metric("proximity_l1_continuous")
class ProximityContinuousL1(Metric):
    """Mean element-wise L1 distance over continuous features, valid CFs only.

    CETGFN equivalent: ``dicoflex_proximity_l1_num``

    Formula::

        mean(|X_test[:, cont] - X_cf[:, cont]|)

    computed only on counterfactuals where ``y_cf_pred == y_target``.
    """

    name = "proximity_l1_continuous"

    def required_inputs(self) -> set[str]:
        return {"X_cf_valid", "X_test_valid", "continuous_features"}

    def __call__(self, **inputs: Any) -> float:
        X_test = inputs["X_test_valid"]
        X_cf = inputs["X_cf_valid"]
        cont = inputs["continuous_features"]

        if X_test.size == 0 or len(cont) == 0:
            return 0.0

        return float(np.abs(X_test[:, cont] - X_cf[:, cont]).mean())


# ---------------------------------------------------------------------------
# Sparsity – categorical features only (valid CFs)
# ---------------------------------------------------------------------------


@register_metric("sparsity_categorical")
class SparsityCategorical(Metric):
    """Proportion of categorical features changed, valid CFs only.

    CETGFN equivalent: ``dicoflex_sparsity_cat``

    Formula::

        mean(X_test[:, cat] != X_cf[:, cat])
    """

    name = "sparsity_categorical"

    def required_inputs(self) -> set[str]:
        return {"X_cf_valid", "X_test_valid", "categorical_features"}

    def __call__(self, **inputs: Any) -> float:
        X_test = inputs["X_test_valid"]
        X_cf = inputs["X_cf_valid"]
        cat = inputs["categorical_features"]

        if X_test.size == 0 or len(cat) == 0:
            return 0.0

        return float((X_test[:, cat] != X_cf[:, cat]).astype(float).mean())


# ---------------------------------------------------------------------------
# Epsilon-Sparsity – continuous features (valid CFs)
# ---------------------------------------------------------------------------


@register_metric("eps_sparsity")
class EpsilonSparsity(Metric):
    """Proportion of continuous features whose relative change exceeds a threshold.

    CETGFN equivalent: ``dicoflex_eps_sparsity``

    Formula::

        mean( |x - x'| / (|x| + eps) > thr )

    with ``eps = 1e-8`` and ``thr = 0.05`` (5 %).
    """

    name = "eps_sparsity"

    def required_inputs(self) -> set[str]:
        return {"X_cf_valid", "X_test_valid", "continuous_features"}

    def __call__(self, **inputs: Any) -> float:
        X_test = inputs["X_test_valid"]
        X_cf = inputs["X_cf_valid"]
        cont = inputs["continuous_features"]
        thr = 0.05
        eps = 1e-8

        if X_test.size == 0 or len(cont) == 0:
            return 0.0

        diff = np.abs(X_test[:, cont] - X_cf[:, cont]) / (np.abs(X_test[:, cont]) + eps)
        return float((diff > thr).mean())


# ---------------------------------------------------------------------------
# LOF – median of log-transformed scores (valid CFs)
# ---------------------------------------------------------------------------


@register_metric("lof_score_median_log")
class LOFScoreMedianLog(Metric):
    """Local Outlier Factor with median + log aggregation, valid CFs only.

    CETGFN equivalent: ``dicoflex_lof_score``

    Formula::

        median( log( -lof.score_samples(X_cf_valid) + 1e-8 ) )

    where LOF is fit on X_train.  Lower values indicate more plausible CFs.
    """

    name = "lof_score_median_log"

    def required_inputs(self) -> set[str]:
        return {"X_cf_valid", "X_train"}

    def __call__(self, **inputs: Any) -> float:
        X_cf = inputs["X_cf_valid"]
        X_train = inputs["X_train"]

        if X_cf.size == 0:
            return 0.0

        lof = LocalOutlierFactor(novelty=True)
        lof.fit(X_train)
        return float(np.median(np.log(-lof.score_samples(X_cf) + 1e-8)))


# ---------------------------------------------------------------------------
# Pairwise diversity – mixed Euclidean + Hamming (valid CFs)
# ---------------------------------------------------------------------------


@register_metric("pairwise_diversity_mixed")
class PairwiseDiversityMixed(Metric):
    """Pairwise diversity using Euclidean (continuous) + Hamming (categorical).

    CETGFN equivalent: ``dicoflex_pairwise_distance``

    Groups counterfactuals by their original instance (X_test row).
    For each group with K >= 2 members, computes::

        mean( (d_euclidean_cont + d_hamming_cat) / n_features )

    over all unique pairs, then averages across groups.
    """

    name = "pairwise_diversity_mixed"

    def required_inputs(self) -> set[str]:
        return {
            "X_cf_valid",
            "X_test_valid",
            "continuous_features",
            "categorical_features",
        }

    def __call__(self, **inputs: Any) -> float:
        X_cf = inputs["X_cf_valid"]
        X_test = inputs["X_test_valid"]
        num_idx = inputs["continuous_features"]
        cat_idx = inputs["categorical_features"]

        if X_cf.size == 0:
            return 0.0

        n_features = len(num_idx) + len(cat_idx)
        if n_features == 0:
            return 0.0

        # Group CFs by their original instance
        groups: dict[tuple, list[np.ndarray]] = {}
        for orig_row, cf_row in zip(X_test, X_cf):
            key = tuple(orig_row.tolist())
            if key not in groups:
                groups[key] = []
            groups[key].append(cf_row.astype(np.float32))

        group_diversities: list[float] = []

        for cf_group in groups.values():
            K = len(cf_group)
            if K < 2:
                continue

            X_cf_group = np.vstack(cf_group)
            num_pairs = K * (K - 1) // 2

            d_cont = (
                pdist(X_cf_group[:, num_idx], metric="euclidean")
                if len(num_idx) > 0
                else np.zeros(num_pairs)
            )
            d_cat = (
                pdist(X_cf_group[:, cat_idx], metric="hamming") * len(cat_idx)
                if len(cat_idx) > 0
                else np.zeros(num_pairs)
            )

            group_diversities.append(float(np.mean((d_cont + d_cat) / n_features)))

        return float(np.mean(group_diversities)) if group_diversities else 0.0
