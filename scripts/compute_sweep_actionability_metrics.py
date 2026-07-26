"""Recompute actionability metrics across the constraint-setup sweep layout.

The sweep produced by `scripts/run_constraint_setup_experiments.py` writes
results under::

    <sweep_root>/<dataset>_setup<N>/<dataset>_setup<N>_<method>/
        fold_0/disc_model_SimpleMLPClassifier.pt
        <MethodDir>/fold_0/counterfactuals_<suffix>.csv

This script walks that layout, recomputes the same protocol-B metrics as
`scripts/compute_actionability_metrics.py` (validity / proximity / sparsity /
LOF / diversity, with the keep-10-out-of-100 protocol and ``[-0.5, 1.5]``
in-range filter for continuous features), and writes one combined LaTeX table
per setup into ``<sweep_root>/<dataset>_setup<N>/metrics.tex``.

Usage:
    uv run python scripts/compute_sweep_actionability_metrics.py \
        --sweep-root outputs/sweep_2026-04-25 \
        --datasets adult default
"""

from __future__ import annotations

import argparse
import logging
import sys as _sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.datasets.traintest_file_dataset import TrainTestFileDataset
from counterfactuals.models.classifier.simple_mlp import SimpleMLPClassifier

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_constraint_setup_experiments import SETUPS  # noqa: E402

from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)

KEEP_PER_FACTUAL = 10


# (pretty_name, method_subdir_suffix, MethodDir, csv_suffix, cf_in_raw_space, target_class)
METHODS: list[tuple[str, str, str, str, bool, int]] = [
    ("DICE", "dice", "DiceExplainerWrapper", "DiceExplainerWrapper_SimpleMLPClassifier", False, 0),
    ("CCHVAE", "cchvae", "CCHVAE", "CCHVAE_SimpleMLPClassifier", False, 0),
    ("DiCoFlex", "dicoflex", "DiCoFlex", "DiCoFlex_SimpleMLPClassifier", True, 1),
]

METRIC_COLUMNS = [
    ("validity_m", "Val.$_M$ $\\uparrow$", "max", "float"),
    ("validity", "Val.$_K$ $\\uparrow$", "max", "float"),
    ("prox_cont", "Prox.-Cont $\\downarrow$", "min", "float"),
    ("spars_cat", "Spars.-Cat $\\downarrow$", "min", "float"),
    ("epsilon_spars", "$\\epsilon$-Spars. $\\downarrow$", "min", "float"),
    ("lof", "LOF $\\downarrow$", "min", "float"),
    ("diversity", "Diversity $\\uparrow$", "max", "float"),
    ("actionability", "Act. $\\downarrow$", "min", "float"),
    ("violation", "Viol. $\\downarrow$", "min", "float"),
    ("missing", "Missing $\\downarrow$", "min", "int"),
]

DATASET_DISPLAY = {
    "adult": "Adult Income (Neural Network)",
    "default": "Default of Credit Card (Neural Network)",
    "bank": "Bank (Neural Network)",
    "gmc": "Give Me Some Credit (Neural Network)",
    "lending-club": "Lending Club (Neural Network)",
}


# ---------------------------------------------------------------------------
# Metrics — same as scripts/compute_actionability_metrics.py
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


def _build_constraint_indices(
    bundle: DatasetBundle,
    immutable: tuple[str, ...],
    monotonic_increase: tuple[str, ...],
) -> tuple[list[int], list[int]]:
    """Map constraint feature names to encoded column indices in MinMax space.

    Returns (immutable_cols, monotonic_increase_cols). For categorical immutable
    features, every one-hot column of the group is included.
    """
    features = bundle.dataset.file_dataset.features
    groups = bundle.dataset.file_dataset.one_hot_feature_groups

    def _cols(name: str) -> list[int]:
        if name in groups:
            return [features.index(c) for c in groups[name]]
        if name in features:
            return [features.index(name)]
        return []

    imm_cols: list[int] = []
    for name in immutable:
        imm_cols.extend(_cols(name))
    mono_cols: list[int] = []
    for name in monotonic_increase:
        mono_cols.extend(_cols(name))
    return imm_cols, mono_cols


def _violation_per_row(
    cf_minmax: np.ndarray,
    factual_minmax: np.ndarray,
    immutable_cols: list[int],
    monotonic_inc_cols: list[int],
    atol: float = 1e-6,
) -> np.ndarray:
    """Boolean per-row mask: True if CF violates immutable or monotonic constraints."""
    n = cf_minmax.shape[0]
    bad = np.zeros(n, dtype=bool)
    if immutable_cols:
        diff = np.abs(cf_minmax[:, immutable_cols] - factual_minmax[:, immutable_cols])
        bad |= np.any(diff > atol, axis=1)
    if monotonic_inc_cols:
        decreased = cf_minmax[:, monotonic_inc_cols] < factual_minmax[:, monotonic_inc_cols] - atol
        bad |= np.any(decreased, axis=1)
    return bad


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


# ---------------------------------------------------------------------------
# Bundle / loading
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    dataset: MethodDataset
    disc_model: SimpleMLPClassifier
    lof: LocalOutlierFactor
    num_idx: list[int]
    cat_groups: list[list[int]]
    num_ranges: np.ndarray
    scale: str
    cont_idx: list[int]
    minmax_scaler: object  # sklearn MinMaxScaler from MinMaxScalingStep
    standard_scaler: StandardScaler | None


def _build_preprocessing() -> PreprocessingPipeline:
    return PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )


def _config_name(dataset_key: str) -> str:
    return "lending_club_split" if dataset_key == "lending-club" else f"{dataset_key}_split"


def _to_metric_space(
    X_minmax: np.ndarray,
    cont_idx: list[int],
    minmax_scaler,
    standard_scaler: StandardScaler | None,
    scale: str,
) -> np.ndarray:
    """Re-express a MinMax-encoded matrix in the metric scale (continuous cols only)."""
    if scale == "minmax" or standard_scaler is None or not cont_idx:
        return X_minmax
    out = X_minmax.astype(np.float32, copy=True)
    raw = minmax_scaler.inverse_transform(X_minmax[:, cont_idx])
    out[:, cont_idx] = standard_scaler.transform(raw).astype(np.float32)
    return out


def build_bundle(
    dataset_key: str,
    configs_root: Path,
    data_root: Path,
    disc_model_path: Path,
    scale: str = "minmax",
) -> DatasetBundle:
    cfg_name = _config_name(dataset_key)
    config_path = configs_root / f"{cfg_name}.yaml"
    train_path = data_root / dataset_key / "train.csv"
    test_path = data_root / dataset_key / "test.csv"

    file_ds = TrainTestFileDataset(
        config_path=str(config_path),
        train_data_path=str(train_path),
        test_data_path=str(test_path),
    )
    dataset = MethodDataset(file_ds, _build_preprocessing())

    num_idx = list(dataset.numerical_features_indices)
    cat_groups = [list(g) for g in dataset.categorical_features_lists]

    X_train_minmax = np.asarray(dataset.X_train)

    minmax_step = dataset.preprocessing_pipeline.get_step("minmax")
    cont_idx = list(minmax_step._continuous_indices)
    minmax_scaler = minmax_step.scaler

    standard_scaler: StandardScaler | None = None
    if scale == "standard" and cont_idx:
        train_raw_cont = minmax_scaler.inverse_transform(X_train_minmax[:, cont_idx])
        standard_scaler = StandardScaler().fit(train_raw_cont)

    X_train_metric = _to_metric_space(
        X_train_minmax, cont_idx, minmax_scaler, standard_scaler, scale
    )

    if num_idx:
        train_num = X_train_metric[:, num_idx].astype(float)
        mins = train_num.min(axis=0)
        maxs = train_num.max(axis=0)
        num_ranges = np.clip(maxs - mins, 1e-6, None)
    else:
        num_ranges = np.array([])

    model = SimpleMLPClassifier(num_inputs=X_train_minmax.shape[1], num_targets=2)
    model.load(str(disc_model_path))
    model.eval()

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_train_metric)

    return DatasetBundle(
        dataset=dataset,
        disc_model=model,
        lof=lof,
        num_idx=num_idx,
        cat_groups=cat_groups,
        num_ranges=num_ranges,
        scale=scale,
        cont_idx=cont_idx,
        minmax_scaler=minmax_scaler,
        standard_scaler=standard_scaler,
    )


# ---------------------------------------------------------------------------
# Per-method evaluation against an explicit fold dir
# ---------------------------------------------------------------------------


def _load_cf_minmax(cf_csv: Path, cf_in_raw_space: bool, bundle: DatasetBundle) -> np.ndarray:
    cf_arr = pd.read_csv(cf_csv).to_numpy(dtype=np.float32)
    if cf_in_raw_space:
        cf_minmax = cf_arr.copy()
        if bundle.cont_idx:
            cf_minmax[:, bundle.cont_idx] = bundle.minmax_scaler.transform(
                cf_arr[:, bundle.cont_idx]
            )
        return cf_minmax.astype(np.float32)
    return cf_arr


def _factual_pool_for_target(bundle: DatasetBundle, target_class: int) -> np.ndarray:
    X_test_minmax = np.asarray(bundle.dataset.X_test)
    with torch.no_grad():
        test_preds = (
            torch.argmax(bundle.disc_model(torch.from_numpy(X_test_minmax).float()), dim=1)
            .cpu()
            .numpy()
        )
    return X_test_minmax[test_preds != target_class]


def _align_cf_to_pool(
    cf_minmax: np.ndarray,
    factual_pool: np.ndarray,
    cf_per_instance: int,
    label: str,
) -> tuple[np.ndarray, int]:
    """Truncate CF block so n_factuals <= len(factual_pool) and divisible by M."""
    n_total = cf_minmax.shape[0]
    if n_total % cf_per_instance != 0:
        logger.warning(
            "%s: CF count %d not divisible by cf_per_instance %d — truncating",
            label,
            n_total,
            cf_per_instance,
        )
        n_total -= n_total % cf_per_instance
        cf_minmax = cf_minmax[:n_total]
    n_factuals = n_total // cf_per_instance
    if n_factuals > len(factual_pool):
        logger.warning(
            "%s: pipeline saved %d factuals but split has only %d — truncating",
            label,
            n_factuals,
            len(factual_pool),
        )
        n_factuals = len(factual_pool)
        cf_minmax = cf_minmax[: n_factuals * cf_per_instance]
    return cf_minmax, n_factuals


def evaluate_fold(
    bundle: DatasetBundle,
    cf_csv: Path,
    cf_in_raw_space: bool,
    target_class: int,
    cf_per_instance: int,
    cf_csv_flipped: Path | None = None,
    immutable_cols: list[int] | None = None,
    monotonic_inc_cols: list[int] | None = None,
) -> dict | None:
    """Compute metrics for one method, optionally merging both target directions.

    When ``cf_csv_flipped`` is provided, factuals/CFs from both directions are
    concatenated (original first, flipped second) so metrics cover the full
    test set per the protocol's bidirectional requirement.
    """
    if not cf_csv.exists():
        logger.warning("Missing CF file: %s", cf_csv)
        return None

    cf_minmax = _load_cf_minmax(cf_csv, cf_in_raw_space, bundle)
    factual_pool = _factual_pool_for_target(bundle, target_class)
    cf_minmax, n_fact = _align_cf_to_pool(cf_minmax, factual_pool, cf_per_instance, cf_csv.name)
    factuals_minmax = factual_pool[:n_fact]
    y_target = np.full(cf_minmax.shape[0], target_class, dtype=np.int64)

    if cf_csv_flipped is not None:
        if not cf_csv_flipped.exists():
            logger.warning(
                "Flipped CF file missing — falling back to single-direction eval: %s",
                cf_csv_flipped,
            )
        else:
            flipped_target = 1 - target_class
            cf_flip = _load_cf_minmax(cf_csv_flipped, cf_in_raw_space, bundle)
            pool_flip = _factual_pool_for_target(bundle, flipped_target)
            cf_flip, n_flip = _align_cf_to_pool(
                cf_flip, pool_flip, cf_per_instance, cf_csv_flipped.name
            )
            factuals_minmax = np.concatenate([factuals_minmax, pool_flip[:n_flip]], axis=0)
            cf_minmax = np.concatenate([cf_minmax, cf_flip], axis=0)
            y_target = np.concatenate(
                [y_target, np.full(cf_flip.shape[0], flipped_target, dtype=np.int64)]
            )

    n_total = cf_minmax.shape[0]
    n_factuals = n_total // cf_per_instance

    with torch.no_grad():
        logits = bundle.disc_model(torch.from_numpy(cf_minmax).float())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    pred_mask = preds == y_target
    if bundle.num_idx:
        cont = cf_minmax[:, bundle.num_idx]
        in_range = np.all((cont >= -0.5) & (cont <= 1.5), axis=1)
    else:
        in_range = np.ones(n_total, dtype=bool)
    per_cf_valid = pred_mask & in_range
    validity_m = float(per_cf_valid.mean()) if n_total else 0.0

    # Metric space (continuous cols only re-scaled).
    cf_metric = _to_metric_space(
        cf_minmax, bundle.cont_idx, bundle.minmax_scaler, bundle.standard_scaler, bundle.scale
    )
    factuals = _to_metric_space(
        factuals_minmax,
        bundle.cont_idx,
        bundle.minmax_scaler,
        bundle.standard_scaler,
        bundle.scale,
    )

    cf_blocks = cf_metric.reshape(n_factuals, cf_per_instance, -1)
    valid_blocks = per_cf_valid.reshape(n_factuals, cf_per_instance)

    keep = min(KEEP_PER_FACTUAL, cf_per_instance)
    order = np.argsort(~valid_blocks, axis=1, kind="stable")[:, :keep]
    row_idx = np.arange(n_factuals)[:, None]
    selected_cf = cf_blocks[row_idx, order]
    selected_valid = valid_blocks[row_idx, order]

    per_factual_validity = selected_valid.mean(axis=1)
    validity = float(per_factual_validity.mean())

    valid_per_factual = selected_valid.sum(axis=1)
    missing_slots = int((keep - valid_per_factual).clip(min=0).sum())
    short = int((valid_per_factual < keep).sum())
    if short:
        logger.warning(
            "%s: %d/%d factuals have fewer than %d valid CFs (min=%d, mean=%.2f); "
            "%d slot(s) filled with invalid CFs",
            cf_csv.name,
            short,
            n_factuals,
            keep,
            int(valid_per_factual.min()),
            float(valid_per_factual.mean()),
            missing_slots,
        )

    prox_vals: list[float] = []
    spars_cat_vals: list[float] = []
    eps_spars_vals: list[float] = []
    lof_vals: list[float] = []
    diversity_vals: list[float] = []
    actionability_vals: list[float] = []

    n_num = len(bundle.num_idx)
    factuals_minmax_aligned = factuals_minmax  # for actionability + violation
    selected_cf_minmax = cf_minmax.reshape(n_factuals, cf_per_instance, -1)[row_idx, order]

    imm_cols = immutable_cols or []
    mono_cols = monotonic_inc_cols or []
    if imm_cols or mono_cols:
        flat = selected_cf_minmax.reshape(n_factuals * keep, -1)
        flat_factuals = np.repeat(factuals_minmax_aligned, keep, axis=0)
        violation_mask = _violation_per_row(flat, flat_factuals, imm_cols, mono_cols)
        violation_rate = float(violation_mask.mean())
    else:
        violation_rate = 0.0

    skipped_no_valid = 0
    skipped_diversity = 0
    for i in range(n_factuals):
        valid_mask_i = selected_valid[i].astype(bool)
        n_valid_i = int(valid_mask_i.sum())
        if n_valid_i == 0:
            skipped_no_valid += 1
            continue

        cfs_i = selected_cf[i][valid_mask_i]
        orig_i = np.tile(factuals[i], (n_valid_i, 1))
        orig_t = torch.from_numpy(orig_i).float()
        cfs_t = torch.from_numpy(cfs_i).float()

        prox_vals.append(_proximity_continuous_l1(orig_t, cfs_t, n_num))
        spars_cat_vals.append(_sparsity_categorical(orig_i, cfs_i, bundle.cat_groups))
        eps_spars_vals.append(_epsilon_sparsity(orig_i, cfs_i, bundle.num_idx, bundle.num_ranges))
        scores = -bundle.lof.score_samples(cfs_i) + 1e-8
        lof_vals.append(float(np.median(np.log(scores))))

        cfs_i_minmax = selected_cf_minmax[i][valid_mask_i]
        orig_i_minmax = np.tile(factuals_minmax_aligned[i], (n_valid_i, 1))
        actionability_vals.append(float(np.all(cfs_i_minmax == orig_i_minmax, axis=1).mean()))

        if n_valid_i >= 2:
            diversity_vals.append(
                _diversity_mixed(
                    cfs_i,
                    np.zeros(n_valid_i, dtype=int),
                    bundle.num_idx,
                    bundle.cat_groups,
                )
            )
        else:
            skipped_diversity += 1

    if skipped_no_valid:
        logger.warning(
            "%s: %d/%d factuals had 0 valid CFs — excluded from prox/spars/LOF/diversity averages",
            cf_csv.name,
            skipped_no_valid,
            n_factuals,
        )
    if skipped_diversity:
        logger.warning(
            "%s: %d factuals had only 1 valid CF — excluded from diversity (needs >=2)",
            cf_csv.name,
            skipped_diversity,
        )

    return {
        "validity_m": validity_m,
        "validity": validity,
        "prox_cont": float(np.mean(prox_vals)) if prox_vals else 0.0,
        "spars_cat": float(np.mean(spars_cat_vals)) if spars_cat_vals else 0.0,
        "epsilon_spars": float(np.mean(eps_spars_vals)) if eps_spars_vals else 0.0,
        "lof": float(np.mean(lof_vals)) if lof_vals else 0.0,
        "diversity": float(np.mean(diversity_vals)) if diversity_vals else 0.0,
        "actionability": float(np.mean(actionability_vals)) if actionability_vals else 0.0,
        "violation": violation_rate,
        "missing": missing_slots,
    }


# ---------------------------------------------------------------------------
# Sweep walker
# ---------------------------------------------------------------------------


def discover_setups(sweep_root: Path, dataset_key: str) -> list[tuple[int, Path]]:
    """Return sorted list of (setup_index, setup_dir) for a dataset."""
    out: list[tuple[int, Path]] = []
    for child in sweep_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        prefix = f"{dataset_key}_setup"
        if not name.startswith(prefix):
            continue
        try:
            idx = int(name[len(prefix) :])
        except ValueError:
            continue
        out.append((idx, child))
    out.sort(key=lambda t: t[0])
    return out


def find_method_dir(setup_dir: Path, dataset_key: str, setup_idx: int, suffix: str) -> Path | None:
    candidate = setup_dir / f"{dataset_key}_setup{setup_idx}_{suffix}"
    return candidate if candidate.is_dir() else None


def evaluate_setup(
    dataset_key: str,
    setup_idx: int,
    setup_dir: Path,
    configs_root: Path,
    data_root: Path,
    cf_per_instance: int,
    scale: str = "minmax",
    flipped_setup_dir: Path | None = None,
) -> dict[str, dict[str, float] | None]:
    """Evaluate all three methods for one setup dir. Each method uses its own
    saved disc_model to judge its own CFs (mirrors how the pipeline saved them).

    If ``flipped_setup_dir`` is provided, CFs from the opposite-target sweep are
    merged so metrics cover both 0->1 and 1->0 over the full test set.
    """
    results: dict[str, dict[str, float] | None] = {}
    bundle_cache: dict[Path, DatasetBundle] = {}

    for pretty, suffix, method_dir, csv_suffix, raw_space, target_cls in METHODS:
        method_root = find_method_dir(setup_dir, dataset_key, setup_idx, suffix)
        if method_root is None:
            logger.warning("Missing method dir for %s setup %d %s", dataset_key, setup_idx, pretty)
            results[pretty] = None
            continue

        disc_pt = method_root / "fold_0" / "disc_model_SimpleMLPClassifier.pt"
        if not disc_pt.exists():
            logger.warning("Missing disc model: %s", disc_pt)
            results[pretty] = None
            continue

        if disc_pt not in bundle_cache:
            bundle_cache[disc_pt] = build_bundle(
                dataset_key, configs_root, data_root, disc_pt, scale=scale
            )
        bundle = bundle_cache[disc_pt]

        setup_def = next((s for s in SETUPS.get(dataset_key, []) if s.index == setup_idx), None)
        if setup_def is None:
            imm_cols, mono_cols = [], []
        else:
            imm_cols, mono_cols = _build_constraint_indices(
                bundle, setup_def.immutable, setup_def.monotonic_increase
            )

        cf_csv = method_root / method_dir / "fold_0" / f"counterfactuals_{csv_suffix}.csv"
        cf_csv_flipped: Path | None = None
        if flipped_setup_dir is not None:
            flipped_method_root = find_method_dir(flipped_setup_dir, dataset_key, setup_idx, suffix)
            if flipped_method_root is None:
                logger.warning(
                    "Missing flipped method dir for %s setup %d %s", dataset_key, setup_idx, pretty
                )
            else:
                cf_csv_flipped = (
                    flipped_method_root
                    / method_dir
                    / "fold_0"
                    / f"counterfactuals_{csv_suffix}.csv"
                )
        results[pretty] = evaluate_fold(
            bundle,
            cf_csv,
            raw_space,
            target_cls,
            cf_per_instance,
            cf_csv_flipped=cf_csv_flipped,
            immutable_cols=imm_cols,
            monotonic_inc_cols=mono_cols,
        )

    return results


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def _format_cell(value: float, best: float, second: float, fmt: str = "float") -> str:
    if fmt == "int":
        txt = f"{int(round(value))}"
        atol = 0.5
    else:
        txt = f"{value:.2f}"
        atol = 5e-3
    if np.isclose(value, best, atol=atol):
        return f"\\textbf{{{txt}}}"
    if np.isclose(value, second, atol=atol):
        return f"\\underline{{{txt}}}"
    return txt


def render_setup_table(
    dataset_key: str,
    setup_idx: int,
    method_rows: dict[str, dict[str, float] | None],
) -> str:
    col_spec = "l" + "c" * len(METRIC_COLUMNS)
    header = " & ".join(
        ["\\textbf{Method}"] + [f"\\textbf{{{hdr}}}" for _, hdr, _, _ in METRIC_COLUMNS]
    )
    display = DATASET_DISPLAY.get(dataset_key, dataset_key)
    title = f"{display} — Setup {setup_idx}"

    lines = [
        "\\begin{table}[H]",
        f"\\label{{tab:{dataset_key}_setup{setup_idx}}}",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        f"\\multicolumn{{{len(METRIC_COLUMNS) + 1}}}{{c}}{{\\textit{{{title}}}}} \\\\",
        "\\midrule",
    ]

    col_best: dict[str, tuple[float, float]] = {}
    for metric_key, _, direction, _ in METRIC_COLUMNS:
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
            for metric_key, _, _, fmt in METRIC_COLUMNS:
                best, second = col_best[metric_key]
                cells.append(_format_cell(row[metric_key], best, second, fmt))
        lines.append(f"{pretty} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def render_dataset_table(
    dataset_key: str,
    per_setup: dict[int, dict[str, dict[str, float] | None]],
    meta: dict | None = None,
) -> str:
    """One table per dataset, with a row group per setup."""
    col_spec = "l" + "c" * len(METRIC_COLUMNS)
    header = " & ".join(
        ["\\textbf{Method}"] + [f"\\textbf{{{hdr}}}" for _, hdr, _, _ in METRIC_COLUMNS]
    )
    display = DATASET_DISPLAY.get(dataset_key, dataset_key)
    n_cols = len(METRIC_COLUMNS) + 1

    lines = [
        "\\begin{table}[H]",
        f"\\label{{tab:{dataset_key}_sweep}}",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{{display}}}}} \\\\",
    ]
    if meta:
        meta_line = (
            f"$N$={meta['n_test']} $\\cdot$ $M$={meta['m']} $\\cdot$ $K$={meta['k']} "
            f"$\\cdot$ scale={meta['scale']} $\\cdot$ directions={meta['directions']} "
            f"$\\cdot$ disc=per-method $h$ ({meta['disc']}) "
            f"$\\cdot$ CFs normalized to {meta['metric_space']} space"
        )
        lines.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\footnotesize {meta_line}}} \\\\")
    lines.append("\\midrule")

    for setup_idx in sorted(per_setup):
        method_rows = per_setup[setup_idx]
        lines.append(
            f"\\multicolumn{{{len(METRIC_COLUMNS) + 1}}}{{l}}{{\\textbf{{Setup {setup_idx}}}}} \\\\"
        )

        col_best: dict[str, tuple[float, float]] = {}
        for metric_key, _, direction, _ in METRIC_COLUMNS:
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
                for metric_key, _, _, fmt in METRIC_COLUMNS:
                    best, second = col_best[metric_key]
                    cells.append(_format_cell(row[metric_key], best, second, fmt))
            lines.append(f"{pretty} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["adult", "default"])
    p.add_argument("--configs-root", type=Path, default=Path("config/datasets"))
    p.add_argument("--data-root", type=Path, default=Path("data_train_test_val"))
    p.add_argument("--cf-per-instance", type=int, default=100)
    p.add_argument(
        "--keep-per-factual",
        type=int,
        default=KEEP_PER_FACTUAL,
        help="How many CFs to keep per factual; valid CFs first, fall back to invalid.",
    )
    p.add_argument(
        "--scale",
        choices=["minmax", "standard"],
        default="minmax",
        help=(
            "Feature scale for proximity / sparsity / LOF / diversity. "
            "'standard' inverse-MinMaxes continuous columns then fits StandardScaler "
            "on training data. Validity is always evaluated in MinMax space."
        ),
    )
    p.add_argument(
        "--flipped-sweep-root",
        type=Path,
        default=None,
        help=(
            "Optional second sweep root containing the opposite-target runs. "
            "When provided, factuals/CFs from both directions are concatenated "
            "before metric computation so results cover the full test set."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()

    global KEEP_PER_FACTUAL
    KEEP_PER_FACTUAL = args.keep_per_factual

    for dataset_key in args.datasets:
        logger.info("=== Dataset: %s ===", dataset_key)
        setups = discover_setups(args.sweep_root, dataset_key)
        if not setups:
            logger.warning("No setups found for %s under %s", dataset_key, args.sweep_root)
            continue

        flipped_setups: dict[int, Path] = {}
        if args.flipped_sweep_root is not None:
            flipped_setups = dict(discover_setups(args.flipped_sweep_root, dataset_key))

        per_setup: dict[int, dict[str, dict[str, float] | None]] = {}
        for setup_idx, setup_dir in setups:
            logger.info("  Setup %d (%s)", setup_idx, setup_dir.name)
            rows = evaluate_setup(
                dataset_key,
                setup_idx,
                setup_dir,
                args.configs_root,
                args.data_root,
                args.cf_per_instance,
                scale=args.scale,
                flipped_setup_dir=flipped_setups.get(setup_idx),
            )
            per_setup[setup_idx] = rows
            for pretty, res in rows.items():
                if res is None:
                    logger.info("    %s: missing", pretty)
                else:
                    logger.info(
                        "    %s: %s",
                        pretty,
                        {k: round(v, 4) for k, v in res.items()},
                    )

            setup_tex = render_setup_table(dataset_key, setup_idx, rows)
            (setup_dir / "metrics.tex").write_text(setup_tex + "\n", encoding="utf-8")
            logger.info("    wrote %s", setup_dir / "metrics.tex")

        # One combined table per dataset.
        cfg_name = _config_name(dataset_key)
        probe = TrainTestFileDataset(
            config_path=str(args.configs_root / f"{cfg_name}.yaml"),
            train_data_path=str(args.data_root / dataset_key / "train.csv"),
            test_data_path=str(args.data_root / dataset_key / "test.csv"),
        )
        n_test = int(np.asarray(probe.X_test).shape[0])
        meta = {
            "n_test": n_test,
            "m": args.cf_per_instance,
            "k": args.keep_per_factual,
            "scale": args.scale,
            "directions": "0$\\to$1 \\& 1$\\to$0"
            if args.flipped_sweep_root is not None
            else "single",
            "metric_space": "standard" if args.scale == "standard" else "MinMax",
            "disc": "SimpleMLPClassifier",
        }
        combined_tex = render_dataset_table(dataset_key, per_setup, meta=meta)
        out_path = args.sweep_root / f"{dataset_key}_metrics.tex"
        out_path.write_text(combined_tex + "\n", encoding="utf-8")
        logger.info("wrote combined table %s", out_path)


if __name__ == "__main__":
    main()
