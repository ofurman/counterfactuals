"""Recompute validity / proximity / sparsity / LOF / diversity metrics from saved CF CSVs.

The pipelines under `counterfactuals/pipelines/run_*_traintest_pipeline.py` persist
per-method counterfactuals in the `models/<dataset>/<Method>/fold_0/` folders. This
script reads those CSVs, re-aligns them against the test split (so we can compute
the actionability-constrained metrics the same way across methods), and renders a
LaTeX table grouped by dataset.

Usage:
    uv run python -m scripts.compute_actionability_metrics \
        --datasets bank default \
        --configs-root config/datasets \
        --data-root data_train_test_val \
        --models-root models \
        --cf-per-instance 100 \
        --output scripts/actionability_table.tex

Methods are hard-coded to match `run_traintest_experiments.sh` +
`run_dicoflex_traintest_experiments.sh`:
    DICE, CCHVAE (CSVs in model/encoded space) and DiCoFlex (CSV in raw,
    MinMax-inverted space — we re-apply `dataset.transform` to align spaces).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from sklearn.neighbors import LocalOutlierFactor

from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.datasets.traintest_file_dataset import TrainTestFileDataset
from counterfactuals.models.classifier.simple_mlp import SimpleMLPClassifier
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric definitions — adapted from the user-supplied reference implementation
# so they work with MethodDataset (no `spec.num_idx` etc.).
# ---------------------------------------------------------------------------


def _proximity_continuous_l1(
    x_orig_model: torch.Tensor, x_cf_model: torch.Tensor, n_num: int
) -> float:
    if n_num == 0 or len(x_orig_model) == 0:
        return 0.0
    x_o = x_orig_model[:, :n_num]
    x_c = x_cf_model[:, :n_num]
    return float((x_o - x_c).abs().mean().item())


def _sparsity_categorical(
    x_orig_model: np.ndarray,
    x_cf_model: np.ndarray,
    cat_groups: list[list[int]],
) -> float:
    if len(x_orig_model) == 0 or not cat_groups:
        return 0.0
    D_cat = len(cat_groups)
    # A categorical "feature" has changed if any of its one-hot cols differs.
    per_row = np.zeros(len(x_orig_model), dtype=float)
    for group in cat_groups:
        diff = np.any(x_orig_model[:, group] != x_cf_model[:, group], axis=1)
        per_row += diff.astype(float)
    return float(np.mean(per_row / D_cat))


def _epsilon_sparsity(
    x_orig_model: np.ndarray,
    x_cf_model: np.ndarray,
    num_idx: list[int],
    ranges: np.ndarray,
    epsilon: float = 0.05,
) -> float:
    if len(x_orig_model) == 0 or not num_idx:
        return 0.0
    D_num = len(num_idx)
    x_o = x_orig_model[:, num_idx].astype(float)
    x_c = x_cf_model[:, num_idx].astype(float)
    abs_diff = np.abs(x_o - x_c)
    thresholds = epsilon * ranges.reshape(1, -1)
    significant = (abs_diff > thresholds).astype(float)
    return float(np.mean(np.sum(significant, axis=1) / D_num))


# ---------------------------------------------------------------------------
# Reference metric definitions.
#
# Verbatim ports of ``cel/metrics/dicoflex_metrics.py`` (commit b9715ef on
# ``origin/ofurman/CFN_baselines``), selected by
# ``cel/pipelines/conf/metrics/dicoflex.yaml``. That module is the authoritative
# source of the paper's six Table 1 column names (``proximity_l1_continuous``,
# ``sparsity_categorical``, ``eps_sparsity``, ``lof_score_median_log``,
# ``pairwise_diversity_mixed``), so these — not the ``_legacy`` variants above —
# define the published metrics.
#
# Three substantive differences from the legacy implementations:
#   * eps-sparsity thresholds the RELATIVE change |dx| / (|x| + 1e-8), not the
#     absolute change against 0.05 * feature_range.
#   * categorical sparsity averages over the ONE-HOT COLUMNS (62 on Adult), not
#     over the one-hot groups (8 on Adult).
#   * diversity uses Euclidean distance on the continuous block (legacy used
#     cityblock) and Hamming over the raw one-hot columns (legacy first collapsed
#     each group to an argmax code).
# The reference metrics also aggregate as a single pooled mean/median over all
# valid counterfactuals, rather than averaging per-factual values.
# ---------------------------------------------------------------------------


def _ref_proximity_continuous_l1(x_orig: np.ndarray, x_cf: np.ndarray, num_idx: list[int]) -> float:
    if x_orig.size == 0 or not num_idx:
        return 0.0
    return float(np.abs(x_orig[:, num_idx] - x_cf[:, num_idx]).mean())


def _ref_sparsity_categorical(x_orig: np.ndarray, x_cf: np.ndarray, cat_idx: list[int]) -> float:
    if x_orig.size == 0 or not cat_idx:
        return 0.0
    return float((x_orig[:, cat_idx] != x_cf[:, cat_idx]).astype(float).mean())


def _ref_epsilon_sparsity(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    num_idx: list[int],
    threshold: float = 0.05,
    epsilon: float = 1e-8,
) -> float:
    if x_orig.size == 0 or not num_idx:
        return 0.0
    rel = np.abs(x_orig[:, num_idx] - x_cf[:, num_idx]) / (np.abs(x_orig[:, num_idx]) + epsilon)
    return float((rel > threshold).mean())


def _ref_lof_score_median_log(lof: LocalOutlierFactor, x_cf: np.ndarray) -> float:
    if x_cf.size == 0:
        return 0.0
    return float(np.median(np.log(-lof.score_samples(x_cf) + 1e-8)))


def _ref_pairwise_diversity_mixed(
    x_cf: np.ndarray,
    x_orig: np.ndarray,
    num_idx: list[int],
    cat_idx: list[int],
) -> float:
    if x_cf.size == 0:
        return 0.0
    n_features = len(num_idx) + len(cat_idx)
    if n_features == 0:
        return 0.0

    groups: dict[tuple, list[np.ndarray]] = {}
    for orig_row, cf_row in zip(x_orig, x_cf):
        groups.setdefault(tuple(orig_row.tolist()), []).append(cf_row.astype(np.float32))

    group_diversities: list[float] = []
    for cf_group in groups.values():
        K = len(cf_group)
        if K < 2:
            continue
        X_cf_group = np.vstack(cf_group)
        num_pairs = K * (K - 1) // 2
        d_cont = (
            pdist(X_cf_group[:, num_idx], metric="euclidean") if num_idx else np.zeros(num_pairs)
        )
        d_cat = (
            pdist(X_cf_group[:, cat_idx], metric="hamming") * len(cat_idx)
            if cat_idx
            else np.zeros(num_pairs)
        )
        group_diversities.append(float(np.mean((d_cont + d_cat) / n_features)))

    return float(np.mean(group_diversities)) if group_diversities else 0.0


def _diversity_mixed(
    x_cf_model: np.ndarray,
    cf_group_ids: np.ndarray,
    num_idx: list[int],
    cat_groups: list[list[int]],
) -> float:
    if cf_group_ids is None or len(x_cf_model) == 0:
        return 0.0
    D_num = len(num_idx)
    D_cat = len(cat_groups)
    D_total = D_num + D_cat
    if D_total == 0:
        return 0.0

    X_num = x_cf_model[:, num_idx] if num_idx else None
    # Encode one-hot groups to a single integer per group.
    if cat_groups:
        cat_encoded = np.zeros((len(x_cf_model), len(cat_groups)), dtype=int)
        for j, group in enumerate(cat_groups):
            cat_encoded[:, j] = np.argmax(x_cf_model[:, group], axis=1)
    else:
        cat_encoded = None

    group_diversities: list[float] = []
    for gid in np.unique(cf_group_ids):
        idx = np.where(cf_group_ids == gid)[0]
        if len(idx) < 2:
            continue
        d_num = pdist(X_num[idx], metric="cityblock") if X_num is not None else 0.0
        d_cat = (
            pdist(cat_encoded[idx], metric="hamming") * D_cat if cat_encoded is not None else 0.0
        )
        mixed = d_num + d_cat
        if np.size(mixed) > 0:
            group_diversities.append(float(np.mean(mixed) / D_total))

    return float(np.mean(group_diversities)) if group_diversities else 0.0


def _lof_log_median(lof: LocalOutlierFactor, x_cf_model: np.ndarray) -> float:
    if len(x_cf_model) == 0:
        return 0.0
    scores = -lof.score_samples(x_cf_model) + 1e-8
    return float(np.median(np.log(scores)))


# ---------------------------------------------------------------------------
# Dataset + model loading
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    dataset: MethodDataset
    disc_model: SimpleMLPClassifier
    lof: LocalOutlierFactor
    num_idx: list[int]
    cat_groups: list[list[int]]
    num_ranges: np.ndarray
    dataset_dir_name: str  # e.g. "bank_split"


def _build_preprocessing() -> PreprocessingPipeline:
    return PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )


def _load_disc_model(models_root: Path, dataset_dir: str, input_dim: int) -> SimpleMLPClassifier:
    pt_path = models_root / dataset_dir / "fold_0" / "disc_model_SimpleMLPClassifier.pt"
    model = SimpleMLPClassifier(num_inputs=input_dim, num_targets=2)
    model.load(str(pt_path))
    model.eval()
    return model


def build_bundle(
    dataset_key: str,
    configs_root: Path,
    data_root: Path,
    models_root: Path,
) -> DatasetBundle:
    """Build MethodDataset + classifier + LOF for one dataset.

    `dataset_key` is the short name used in the shell scripts (e.g. "bank",
    "default"). Config name follows `config_name_for` in the shell script:
    `lending-club` → `lending_club_split`, everything else → `<name>_split`.
    """
    data_dir = dataset_key  # matches data_train_test_val/<dataset_key>/*
    cfg_name = "lending_club_split" if dataset_key == "lending-club" else f"{dataset_key}_split"
    config_path = configs_root / f"{cfg_name}.yaml"
    train_path = data_root / data_dir / "train.csv"
    test_path = data_root / data_dir / "test.csv"

    file_ds = TrainTestFileDataset(
        config_path=str(config_path),
        train_data_path=str(train_path),
        test_data_path=str(test_path),
    )
    dataset = MethodDataset(file_ds, _build_preprocessing())

    num_idx = list(dataset.numerical_features_indices)
    cat_groups = [list(g) for g in dataset.categorical_features_lists]

    X_train_np = np.asarray(dataset.X_train)
    if num_idx:
        train_num = X_train_np[:, num_idx].astype(float)
        mins = train_num.min(axis=0)
        maxs = train_num.max(axis=0)
        num_ranges = np.clip(maxs - mins, 1e-6, None)
    else:
        num_ranges = np.array([])

    disc_model = _load_disc_model(models_root, cfg_name, input_dim=X_train_np.shape[1])

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_train_np)

    return DatasetBundle(
        dataset=dataset,
        disc_model=disc_model,
        lof=lof,
        num_idx=num_idx,
        cat_groups=cat_groups,
        num_ranges=num_ranges,
        dataset_dir_name=cfg_name,
    )


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------


KEEP_PER_FACTUAL = 10

# "reference" = the formulas from cel/metrics/dicoflex_metrics.py that define the
# paper's Table 1 columns. "legacy" = the earlier reimplementation kept for
# back-comparison against previously generated tables.
FORMULA_MODE = "reference"


METHODS: list[tuple[str, str, str, bool, int]] = [
    # (pretty_name, method_dir, csv_suffix, cf_saved_in_raw_space, target_class)
    # target_class mirrors pipelines/conf/*.yaml: DICE/CCHVAE=0, DiCoFlex=1.
    ("DICE", "DiceExplainerWrapper", "DiceExplainerWrapper_SimpleMLPClassifier", False, 0),
    ("CCHVAE", "CCHVAE", "CCHVAE_SimpleMLPClassifier", False, 0),
    ("DiCoFlex", "DiCoFlex", "DiCoFlex_SimpleMLPClassifier", True, 1),
]


def evaluate_method(
    bundle: DatasetBundle,
    method_dir: str,
    csv_suffix: str,
    cf_in_raw_space: bool,
    target_class: int,
    models_root: Path,
    cf_per_instance: int,
) -> dict | None:
    fold_dir = models_root / bundle.dataset_dir_name / method_dir / "fold_0"
    cf_path = fold_dir / f"counterfactuals_{csv_suffix}.csv"
    if not cf_path.exists():
        logger.warning("Missing CF file: %s", cf_path)
        return None

    cf_arr = pd.read_csv(cf_path).to_numpy(dtype=np.float32)

    if cf_in_raw_space:
        # DiCoFlex stored inverse-transformed (MinMax undone) but one-hot kept.
        # We bypass MethodDataset.transform because TorchDataTypeStep requires
        # X_test to be present; we only need the MinMax step here.
        minmax = bundle.dataset.preprocessing_pipeline.get_step("minmax")
        cont_idx = minmax._continuous_indices
        cf_model = cf_arr.copy()
        if cont_idx:
            cf_model[:, cont_idx] = minmax.scaler.transform(cf_arr[:, cont_idx])
        cf_model = cf_model.astype(np.float32)
    else:
        cf_model = cf_arr

    # Match the pipeline: when relabel_with_disc_model=True, factuals are
    # filtered by disc_model predictions on X_test, not by ground-truth labels.
    X_test = np.asarray(bundle.dataset.X_test)
    with torch.no_grad():
        test_preds = (
            torch.argmax(bundle.disc_model(torch.from_numpy(X_test).float()), dim=1).cpu().numpy()
        )
    factual_pool = X_test[test_preds != target_class]

    n_total = cf_model.shape[0]
    if n_total % cf_per_instance != 0:
        logger.warning(
            "%s: CF count %d not divisible by cf_per_instance %d — truncating",
            method_dir,
            n_total,
            cf_per_instance,
        )
        n_total -= n_total % cf_per_instance
        cf_model = cf_model[:n_total]
    n_factuals = n_total // cf_per_instance

    if n_factuals > len(factual_pool):
        logger.warning(
            "%s: pipeline saved %d factuals but current split only has %d — "
            "truncating CFs to match.",
            method_dir,
            n_factuals,
            len(factual_pool),
        )
        n_factuals = len(factual_pool)
        n_total = n_factuals * cf_per_instance
        cf_model = cf_model[:n_total]

    factuals = factual_pool[:n_factuals]
    y_target = np.full(n_total, target_class, dtype=np.int64)

    # Validity mask per CF. Drop CFs whose continuous features land far outside
    # the trained MinMax range — some DiCoFlex rows degenerate into wild values
    # that the MLP still labels as the target class but aren't real CFs.
    with torch.no_grad():
        logits = bundle.disc_model(torch.from_numpy(cf_model).float())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    pred_mask = preds == y_target
    if bundle.num_idx:
        cont = cf_model[:, bundle.num_idx]
        in_range = np.all((cont >= -0.5) & (cont <= 1.5), axis=1)
    else:
        in_range = np.ones(n_total, dtype=bool)
    per_cf_valid = pred_mask & in_range  # shape (n_total,)

    # Reshape into (n_factuals, cf_per_instance, ...) blocks and subsample the
    # first `keep_per_factual` valid CFs per group, padding with invalid ones
    # when fewer than `keep_per_factual` valid CFs exist. This mirrors the
    # "take 10 valid out of 100, fall back to invalid" protocol.
    cf_blocks = cf_model.reshape(n_factuals, cf_per_instance, -1)
    valid_blocks = per_cf_valid.reshape(n_factuals, cf_per_instance)

    keep = min(KEEP_PER_FACTUAL, cf_per_instance)
    # Sort each row so valid (True) comes first; stable sort preserves order.
    order = np.argsort(~valid_blocks, axis=1, kind="stable")[:, :keep]
    row_idx = np.arange(n_factuals)[:, None]
    selected_cf = cf_blocks[row_idx, order]  # (n_factuals, keep, D)
    selected_valid = valid_blocks[row_idx, order]  # (n_factuals, keep)

    # Validity = mean fraction of selected slots that are truly valid.
    per_factual_validity = selected_valid.mean(axis=1)
    validity = float(per_factual_validity.mean())

    if FORMULA_MODE == "reference":
        # Reference aggregation: one pooled mean/median over all VALID CFs.
        orig_rep = np.repeat(factuals[:, None, :], keep, axis=1)
        flat_cf = selected_cf.reshape(-1, selected_cf.shape[-1])
        flat_orig = orig_rep.reshape(-1, orig_rep.shape[-1])
        flat_valid = selected_valid.reshape(-1)
        x_cf_valid = flat_cf[flat_valid]
        x_orig_valid = flat_orig[flat_valid]
        cat_idx = [c for group in bundle.cat_groups for c in group]

        return {
            "validity": validity,
            "prox_cont": _ref_proximity_continuous_l1(x_orig_valid, x_cf_valid, bundle.num_idx),
            "spars_cat": _ref_sparsity_categorical(x_orig_valid, x_cf_valid, cat_idx),
            "epsilon_spars": _ref_epsilon_sparsity(x_orig_valid, x_cf_valid, bundle.num_idx),
            "lof": _ref_lof_score_median_log(bundle.lof, x_cf_valid),
            "diversity": _ref_pairwise_diversity_mixed(
                x_cf_valid, x_orig_valid, bundle.num_idx, cat_idx
            ),
            "valid_count": int(per_cf_valid.sum()),
            "n_factuals": n_factuals,
            "keep_per_factual": keep,
        }

    # Legacy: per-factual metrics, then mean across factuals.
    prox_vals: list[float] = []
    spars_cat_vals: list[float] = []
    eps_spars_vals: list[float] = []
    lof_vals: list[float] = []
    diversity_vals: list[float] = []

    n_num = len(bundle.num_idx)

    for i in range(n_factuals):
        cfs_i = selected_cf[i]  # (keep, D)
        orig_i = np.tile(factuals[i], (keep, 1))
        orig_t = torch.from_numpy(orig_i).float()
        cfs_t = torch.from_numpy(cfs_i).float()

        prox_vals.append(_proximity_continuous_l1(orig_t, cfs_t, n_num))
        spars_cat_vals.append(_sparsity_categorical(orig_i, cfs_i, bundle.cat_groups))
        eps_spars_vals.append(_epsilon_sparsity(orig_i, cfs_i, bundle.num_idx, bundle.num_ranges))
        # LOF per-factual: median over the `keep` CFs, same transform as before.
        if len(cfs_i) > 0:
            scores = -bundle.lof.score_samples(cfs_i) + 1e-8
            lof_vals.append(float(np.median(np.log(scores))))
        diversity_vals.append(
            _diversity_mixed(
                cfs_i,
                np.zeros(keep, dtype=int),  # single group
                bundle.num_idx,
                bundle.cat_groups,
            )
        )

    return {
        "validity": validity,
        "prox_cont": float(np.mean(prox_vals)) if prox_vals else 0.0,
        "spars_cat": float(np.mean(spars_cat_vals)) if spars_cat_vals else 0.0,
        "epsilon_spars": float(np.mean(eps_spars_vals)) if eps_spars_vals else 0.0,
        "lof": float(np.mean(lof_vals)) if lof_vals else 0.0,
        "diversity": float(np.mean(diversity_vals)) if diversity_vals else 0.0,
        "valid_count": int(per_cf_valid.sum()),
        "n_factuals": n_factuals,
        "keep_per_factual": keep,
    }


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


METRIC_COLUMNS = [
    ("validity", "Validity $\\uparrow$", "max"),
    ("prox_cont", "Prox.-Cont $\\downarrow$", "min"),
    ("spars_cat", "Spars.-Cat $\\downarrow$", "min"),
    ("epsilon_spars", "$\\epsilon$-Spars. $\\downarrow$", "min"),
    ("lof", "LOF $\\downarrow$", "min"),
    ("diversity", "Diversity $\\uparrow$", "max"),
]

DATASET_DISPLAY = {
    "adult": "Adult Income (Neural Network)",
    "bank": "Bank (Neural Network)",
    "default": "Default of Credit Card (Neural Network)",
    "gmc": "Give Me Some Credit (Neural Network)",
    "lending-club": "Lending Club (Neural Network)",
}


def _format_cell(value: float, best: float, second: float, direction: str) -> str:
    txt = f"{value:.2f}"
    if np.isclose(value, best, atol=5e-3):
        return f"\\textbf{{{txt}}}"
    if np.isclose(value, second, atol=5e-3):
        return f"\\underline{{{txt}}}"
    return txt


def render_latex(results: dict[str, dict[str, dict[str, float]]]) -> str:
    col_spec = "l" + "c" * len(METRIC_COLUMNS)
    header = " & ".join(
        ["\\textbf{Method}"] + [f"\\textbf{{{hdr}}}" for _, hdr, _ in METRIC_COLUMNS]
    )
    lines = [
        "\\begin{table}[H]",
        "\\label{tab:protocol_B}",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]

    for dataset_key, method_rows in results.items():
        display = DATASET_DISPLAY.get(dataset_key, dataset_key)
        lines.append(
            f"\\multicolumn{{{len(METRIC_COLUMNS) + 1}}}{{c}}{{\\textit{{{display}}}}} \\\\"
        )
        lines.append("\\midrule")

        # Compute best / second-best per column.
        col_best: dict[str, tuple[float, float]] = {}
        for metric_key, _, direction in METRIC_COLUMNS:
            values = sorted(
                {row[metric_key] for row in method_rows.values() if row is not None},
                reverse=(direction == "max"),
            )
            best = values[0] if values else float("nan")
            second = values[1] if len(values) > 1 else float("nan")
            col_best[metric_key] = (best, second)

        for pretty, *_ in METHODS:
            row = method_rows.get(pretty)
            if row is None:
                cells = ["--"] * len(METRIC_COLUMNS)
            else:
                cells = []
                for metric_key, _, direction in METRIC_COLUMNS:
                    best, second = col_best[metric_key]
                    cells.append(_format_cell(row[metric_key], best, second, direction))
            lines.append(f"{pretty} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["bank", "default"])
    parser.add_argument("--configs-root", default="config/datasets")
    parser.add_argument("--data-root", default="data_train_test_val")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--cf-per-instance", type=int, default=100)
    parser.add_argument(
        "--keep-per-factual",
        type=int,
        default=KEEP_PER_FACTUAL,
        help=(
            "How many CFs to keep per factual for metric computation. Valid CFs "
            "are preferred; if fewer are available, invalid CFs fill the rest."
        ),
    )
    parser.add_argument(
        "--formula",
        choices=("reference", "legacy"),
        default=FORMULA_MODE,
        help=(
            "'reference' uses the formulas from cel/metrics/dicoflex_metrics.py, "
            "which define the paper's Table 1 columns. 'legacy' uses the earlier "
            "reimplementation (absolute eps-sparsity, group-wise categorical "
            "sparsity, cityblock diversity, per-factual aggregation)."
        ),
    )
    parser.add_argument("--output", default="scripts/actionability_table.tex")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()

    global KEEP_PER_FACTUAL, FORMULA_MODE
    KEEP_PER_FACTUAL = args.keep_per_factual
    FORMULA_MODE = args.formula
    logger.info("Metric formula mode: %s", FORMULA_MODE)

    configs_root = Path(args.configs_root)
    data_root = Path(args.data_root)
    models_root = Path(args.models_root)

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_key in args.datasets:
        logger.info("=== Dataset: %s ===", dataset_key)
        bundle = build_bundle(dataset_key, configs_root, data_root, models_root)

        per_method: dict[str, dict[str, float]] = {}
        for pretty, method_dir, csv_suffix, raw_space, target_cls in METHODS:
            logger.info("  Method: %s", pretty)
            res = evaluate_method(
                bundle,
                method_dir,
                csv_suffix,
                raw_space,
                target_cls,
                models_root,
                args.cf_per_instance,
            )
            if res is not None:
                logger.info(
                    "    %s",
                    {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
                )
            per_method[pretty] = res
        all_results[dataset_key] = per_method

    latex = render_latex(all_results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n", encoding="utf-8")
    logger.info("LaTeX table written to %s", out_path)
    print(latex)


if __name__ == "__main__":
    main()
