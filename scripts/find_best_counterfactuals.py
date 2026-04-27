"""Pick the single best counterfactual per (dataset, method) by a sum of three
minimization metrics: Prox.-Cont, Spars.-Cat, and epsilon-Sparsity.

For each CF we compute those three metrics against its own factual, sum them,
and keep the argmin across all valid CFs. "Valid" uses the same criterion as
`scripts/compute_actionability_metrics.py` (disc_model predicts the target
class AND scaled continuous features fall in [-0.5, 1.5]).

Output: a markdown-style text file with one section per dataset listing the
chosen factual + CF in raw feature space, plus the per-metric breakdown.

Usage:
    uv run python -m scripts.find_best_counterfactuals \
        --datasets bank default adult gmc lending-club \
        --output best_counterfactuals.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.compute_actionability_metrics import (
    METHODS,
    _diversity_mixed,
    build_bundle,
)

logger = logging.getLogger(__name__)

TOP_K = 10


def _load_cf_model_space(bundle, cf_path: Path, cf_in_raw_space: bool) -> np.ndarray:
    cf_arr = pd.read_csv(cf_path).to_numpy(dtype=np.float32)
    if not cf_in_raw_space:
        return cf_arr
    minmax = bundle.dataset.preprocessing_pipeline.get_step("minmax")
    cont_idx = minmax._continuous_indices
    cf_model = cf_arr.copy()
    if cont_idx:
        cf_model[:, cont_idx] = minmax.scaler.transform(cf_arr[:, cont_idx])
    return cf_model.astype(np.float32)


def _per_cf_metrics(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    num_idx: list[int],
    cat_groups: list[list[int]],
    num_ranges: np.ndarray,
    epsilon: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised per-row Prox.-Cont, Spars.-Cat, eps-Sparsity."""
    n = x_orig.shape[0]

    if num_idx:
        diff_num = np.abs(x_orig[:, num_idx] - x_cf[:, num_idx])
        prox_cont = diff_num.mean(axis=1)
        thresholds = epsilon * num_ranges.reshape(1, -1)
        eps_spars = (diff_num > thresholds).sum(axis=1) / len(num_idx)
    else:
        prox_cont = np.zeros(n)
        eps_spars = np.zeros(n)

    if cat_groups:
        changes = np.zeros(n, dtype=float)
        for group in cat_groups:
            changes += np.any(x_orig[:, group] != x_cf[:, group], axis=1).astype(float)
        spars_cat = changes / len(cat_groups)
    else:
        spars_cat = np.zeros(n)

    return prox_cont, spars_cat, eps_spars


def _raw_space_row(bundle, model_space_row: np.ndarray) -> np.ndarray:
    """Inverse-transform a single model-space row into raw numeric space."""
    arr = model_space_row.reshape(1, -1).astype(np.float32)
    return bundle.dataset.inverse_transform(arr)[0]


def find_best(
    dataset_key: str,
    bundle,
    models_root: Path,
    cf_per_instance: int,
) -> list[dict]:
    results: list[dict] = []
    X_test = np.asarray(bundle.dataset.X_test)

    for pretty, method_dir, csv_suffix, raw_space, target_class in METHODS:
        cf_path = (
            models_root
            / bundle.dataset_dir_name
            / method_dir
            / "fold_0"
            / (f"counterfactuals_{csv_suffix}.csv")
        )
        if not cf_path.exists():
            logger.warning("%s/%s: missing %s", dataset_key, pretty, cf_path)
            continue

        cf_model = _load_cf_model_space(bundle, cf_path, raw_space)

        with torch.no_grad():
            test_preds = (
                torch.argmax(bundle.disc_model(torch.from_numpy(X_test).float()), 1).cpu().numpy()
            )
        factual_pool = X_test[test_preds != target_class]

        n_total = (cf_model.shape[0] // cf_per_instance) * cf_per_instance
        cf_model = cf_model[:n_total]
        n_factuals = n_total // cf_per_instance
        if n_factuals > len(factual_pool):
            n_factuals = len(factual_pool)
            n_total = n_factuals * cf_per_instance
            cf_model = cf_model[:n_total]

        factuals = factual_pool[:n_factuals]
        x_orig_expanded = np.repeat(factuals, cf_per_instance, axis=0)

        # Validity filter (same as compute_actionability_metrics.py).
        with torch.no_grad():
            preds = (
                torch.argmax(bundle.disc_model(torch.from_numpy(cf_model).float()), 1).cpu().numpy()
            )
        pred_mask = preds == target_class
        if bundle.num_idx:
            cont = cf_model[:, bundle.num_idx]
            in_range = np.all((cont >= -0.5) & (cont <= 1.5), axis=1)
        else:
            in_range = np.ones(n_total, dtype=bool)
        valid_mask = pred_mask & in_range

        if valid_mask.sum() == 0:
            logger.warning("%s/%s: no valid CFs", dataset_key, pretty)
            continue

        prox, spars, eps = _per_cf_metrics(
            x_orig_expanded,
            cf_model,
            bundle.num_idx,
            bundle.cat_groups,
            bundle.num_ranges,
        )
        score = prox + spars + eps
        score[~valid_mask] = np.inf

        finite_count = int(np.isfinite(score).sum())
        k = min(TOP_K, finite_count)
        if k == 0:
            logger.warning("%s/%s: no valid CFs for top-k", dataset_key, pretty)
            continue
        top_idx = np.argpartition(score, k - 1)[:k]
        top_cfs = cf_model[top_idx]
        diversity = _diversity_mixed(
            top_cfs,
            np.zeros(k, dtype=int),
            bundle.num_idx,
            bundle.cat_groups,
        )

        results.append(
            {
                "method": pretty,
                "dataset": dataset_key,
                "prox_cont": float(prox[top_idx].mean()),
                "spars_cat": float(spars[top_idx].mean()),
                "epsilon_spars": float(eps[top_idx].mean()),
                "diversity": float(diversity),
                "score": float(score[top_idx].mean()),
            }
        )

    return results


def _render_markdown(all_results: dict[str, list[dict]]) -> str:
    lines: list[str] = [
        f"# Mean metrics over top-{TOP_K} CFs per (dataset, method)",
        "",
        "Ranking: CFs ordered by `prox_cont + spars_cat + epsilon_spars` (invalid CFs excluded), top-K selected, then metric means reported.",
        "",
    ]
    for dataset_key, rows in all_results.items():
        lines.append(f"## {dataset_key}")
        lines.append("")
        lines.append("| Method | Prox.-Cont ↓ | Spars.-Cat ↓ | ε-Spars. ↓ | Diversity ↑ | Sum ↓ |")
        lines.append("|---|---|---|---|---|---|")
        for res in rows:
            lines.append(
                f"| {res['method']} "
                f"| {res['prox_cont']:.3f} | {res['spars_cat']:.3f} "
                f"| {res['epsilon_spars']:.3f} | {res['diversity']:.3f} "
                f"| {res['score']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["bank", "default"])
    parser.add_argument("--configs-root", default="config/datasets")
    parser.add_argument("--data-root", default="data_train_test_val")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--cf-per-instance", type=int, default=100)
    parser.add_argument("--output", default="best_counterfactuals.md")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()

    configs_root = Path(args.configs_root)
    data_root = Path(args.data_root)
    models_root = Path(args.models_root)

    all_results: dict[str, list[dict]] = {}
    for dataset_key in args.datasets:
        logger.info("=== %s ===", dataset_key)
        bundle = build_bundle(dataset_key, configs_root, data_root, models_root)
        all_results[dataset_key] = find_best(dataset_key, bundle, models_root, args.cf_per_instance)

    md = _render_markdown(all_results)
    Path(args.output).write_text(md + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.output)
    print(md)


if __name__ == "__main__":
    main()
