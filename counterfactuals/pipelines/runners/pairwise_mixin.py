"""Mixin for pairwise counterfactual methods that generate multiple CFs per instance."""

from typing import Any

import numpy as np
import torch

from counterfactuals.pipelines.base_runner import SearchResult


class PairwiseMixin:
    """Mixin for pairwise CF methods that generate multiple CFs per instance.

    This mixin extends :class:`PipelineRunner` to handle counterfactual methods that
    generate multiple counterfactuals per instance. It calculates the standard metrics
    using the first counterfactual per instance (stored in ``X_cf``), and additionally
    computes the pairwise mean distance across all generated counterfactuals.

    Convention:
        - Pairwise runners store ``Xs_cfs_all`` (3D array) in ``SearchResult.extras``.
        - ``X_cf`` is set to the first counterfactual per instance (``Xs_cfs_first``).
    """

    @staticmethod
    def _build_pairwise_arrays(
        cfs_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack per-run CFs into 3D array, extract first CF per instance.

        Args:
            cfs_list: list of CF arrays, each of shape ``(n_samples, n_features)``.
                Each element represents one run/draw of the CF method.

        Returns:
            tuple of:
                - ``Xs_cfs_first``: First CF per instance, shape ``(n_samples, n_features)``.
                - ``Xs_cfs_all``: All CFs stacked, shape ``(n_samples, n_runs, n_features)``.
        """
        Xs_cfs_all = np.stack(cfs_list, axis=1)
        Xs_cfs_first = Xs_cfs_all[:, 0, :]
        return Xs_cfs_first, Xs_cfs_all

    def calculate_metrics(
        self,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        dataset: Any,
        result: SearchResult,
        log_prob_threshold: float,
    ) -> dict[str, Any]:
        """Compute evaluation metrics including pairwise mean distance.

        Args:
            gen_model: Wrapped generative model (with dequantization).
            disc_model: Trained discriminative model.
            dataset: The current fold's dataset.
            result: Output of :meth:`search_counterfactuals` containing ``X_cf`` and
                ``extras["Xs_cfs_all"]``.
            log_prob_threshold: Plausibility threshold from :meth:`compute_log_prob_threshold`.

        Returns:
            Dictionary of metric name → value, including ``pairwise_mean_distance``.
        """
        # Call base calculate_metrics using result.X_cf (first CF per instance)
        metrics = super().calculate_metrics(
            gen_model, disc_model, dataset, result, log_prob_threshold
        )

        # Extract Xs_cfs_all from extras and compute pairwise distance
        Xs_cfs_all = result.extras["Xs_cfs_all"]

        from counterfactuals.pipelines.metrics_utils import compute_pairwise_mean_distance

        metrics["pairwise_mean_distance"] = compute_pairwise_mean_distance(Xs_cfs_all)

        return metrics
