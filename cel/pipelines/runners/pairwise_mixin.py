"""Mixin for pairwise counterfactual methods that generate multiple CFs per instance."""

from typing import Any

import numpy as np
import torch

from cel.pipelines.base_runner import SearchResult


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
        """Compute evaluation metrics over all CFs (flattened) with group IDs.

        Flattens the 3-D ``extras["Xs_cfs_all"]`` array so that every generated
        counterfactual is evaluated (not just the first per instance).  Group IDs
        are passed through so that diversity metrics can compute pairwise distances
        within each original-instance group.

        Args:
            gen_model: Wrapped generative model (with dequantization).
            disc_model: Trained discriminative model.
            dataset: The current fold's dataset.
            result: Output of :meth:`search_counterfactuals` containing ``X_cf`` and
                ``extras["Xs_cfs_all"]``.
            log_prob_threshold: Plausibility threshold from :meth:`compute_log_prob_threshold`.

        Returns:
            Dictionary of metric name → value.
        """
        Xs_cfs_all = result.extras["Xs_cfs_all"]  # (n_instances, n_runs, n_features)
        n_instances, n_runs, n_features = Xs_cfs_all.shape

        # Flatten: (n_instances * n_runs, n_features)
        X_cf_flat = Xs_cfs_all.reshape(-1, n_features)
        X_test_flat = np.repeat(result.X_test, n_runs, axis=0)
        y_orig_flat = np.repeat(result.y_orig, n_runs, axis=0)
        y_target_flat = np.repeat(result.y_target, n_runs, axis=0)
        cf_group_ids = np.repeat(np.arange(n_instances), n_runs)
        model_returned_flat = np.ones(X_cf_flat.shape[0], dtype=bool)

        flat_result = SearchResult(
            X_cf=X_cf_flat,
            X_test=X_test_flat,
            y_orig=y_orig_flat,
            y_target=y_target_flat,
            model_returned=model_returned_flat,
            cf_search_time=result.cf_search_time,
            extras={"cf_group_ids": cf_group_ids},
        )

        metrics = super().calculate_metrics(
            gen_model, disc_model, dataset, flat_result, log_prob_threshold
        )

        from cel.pipelines.metrics_utils import compute_pairwise_mean_distance

        metrics["pairwise_mean_distance"] = compute_pairwise_mean_distance(Xs_cfs_all)

        return metrics
