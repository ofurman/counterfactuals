"""Score counterfactuals with DICTUM's metric definitions.

This is the evaluation half of the DICTUM-aligned experimental setup. It reads
the counterfactual/factual CSVs written by the `dictum_*_config.yaml` runs and
scores them with the formulas from DICTUM's
`src/tabdce/utils/advanced_metrics.py`, so numbers from the two codebases can be
put in the same table.

Where these formulas differ from `scripts/compute_actionability_metrics.py`
(which reproduces the older in-house column definitions):

    * epsilon-sparsity thresholds the change against `0.05 * train_range` in
      ORIGINAL units, not against the relative change `|dx| / (|x| + 1e-8)`.
    * categorical sparsity counts one-hot GROUPS changed (8 on Adult), not
      one-hot columns (62 on Adult).
    * diversity uses cityblock distance on the continuous block and Hamming over
      group-collapsed categorical codes.
    * every metric aggregates per factual first and then averages over factuals,
      with LOF taking the MEDIAN within a factual's counterfactual set.
    * factuals whose counterfactual set contains no valid counterfactual are
      skipped rather than contributing a zero.

Usage:
    uv run python -m scripts.compute_dictum_metrics \
        --results-root results/dictum \
        --datasets adult bank default gmc lending-club \
        --seeds 42 43 44 \
        --output results/dictum/dictum_metrics
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from sklearn.neighbors import LocalOutlierFactor

from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.datasets.traintest_file_dataset import TrainTestFileDataset
from counterfactuals.models.classifier.simple_mlp import SimpleMLPClassifier
from counterfactuals.models.space_adapter import SpaceAdapterClassifier
from counterfactuals.pipelines.utils import apply_categorical_discretization
from counterfactuals.preprocessing import build_model_space_pipeline

logger = logging.getLogger(__name__)

EPSILON = 0.05
LOF_N_NEIGHBORS = 20
KEEP_PER_FACTUAL = 10

# (pretty name, method directory, CSV suffix, counterfactuals saved in original units)
METHODS: list[tuple[str, str, str, bool]] = [
    ("DiCE", "DiceExplainerWrapper", "DiceExplainerWrapper_SimpleMLPClassifier", False),
    ("CCHVAE", "CCHVAE", "CCHVAE_SimpleMLPClassifier", False),
    ("DiCoFlex", "DiCoFlex", "DiCoFlex_SimpleMLPClassifier", True),
]

METRIC_KEYS = [
    "validity",
    "pool_validity",
    "prox_cont",
    "spars_cat",
    "epsilon_spars",
    "lof",
    "div",
    "dir_div",
]


# ---------------------------------------------------------------------------
# Metric definitions — ports of DICTUM/src/tabdce/utils/advanced_metrics.py
# ---------------------------------------------------------------------------


def proximity_continuous(x_orig: np.ndarray, x_cf: np.ndarray, num_idx: list[int]) -> float:
    """Mean absolute continuous change, in model (scaled) space."""
    if not num_idx or len(x_orig) == 0:
        return 0.0
    return float(np.abs(x_orig[:, num_idx] - x_cf[:, num_idx]).mean())


def sparsity_categorical(
    x_orig: np.ndarray, x_cf: np.ndarray, cat_groups: list[list[int]]
) -> float:
    """Fraction of categorical features (one-hot groups) that changed."""
    if not cat_groups or len(x_orig) == 0:
        return 0.0
    changed = np.zeros(len(x_orig), dtype=float)
    for group in cat_groups:
        changed += np.any(x_orig[:, group] != x_cf[:, group], axis=1).astype(float)
    return float(np.mean(changed / len(cat_groups)))


def epsilon_sparsity(
    x_orig_original_units: np.ndarray,
    x_cf_original_units: np.ndarray,
    num_idx: list[int],
    ranges: np.ndarray,
) -> float:
    """Fraction of continuous features moved by more than `EPSILON * range`.

    The threshold is applied in original units, so a feature's range — not its
    scaled magnitude — decides what counts as a meaningful change.
    """
    if not num_idx or len(x_orig_original_units) == 0:
        return 0.0
    diff = np.abs(x_orig_original_units[:, num_idx] - x_cf_original_units[:, num_idx])
    significant = diff > (EPSILON * ranges.reshape(1, -1))
    return float(np.mean(significant.sum(axis=1) / len(num_idx)))


def lof_log_median(lof: LocalOutlierFactor, x_cf: np.ndarray) -> float:
    """Median log outlier score over a factual's counterfactual set."""
    if len(x_cf) == 0:
        return 0.0
    return float(np.median(np.log(-lof.score_samples(x_cf) + 1e-8)))


def categorical_codes(X: np.ndarray, cat_groups: list[list[int]]) -> np.ndarray:
    """Reduce each categorical feature to one integer code per row.

    Handles both representations: a multi-column group is a one-hot block and
    collapses by argmax, while a single-column group already holds the code.

    Args:
        X: Rows in model space.
        cat_groups: Column indices belonging to each categorical feature.

    Returns:
        Array of shape (len(X), len(cat_groups)) of integer codes.
    """
    codes = np.zeros((len(X), len(cat_groups)), dtype=int)
    for j, group in enumerate(cat_groups):
        codes[:, j] = X[:, group[0]].astype(int) if len(group) == 1 else np.argmax(X[:, group], 1)
    return codes


def diversity_mixed(x_cf: np.ndarray, num_idx: list[int], cat_groups: list[list[int]]) -> float:
    """Mean pairwise mixed distance within one factual's counterfactual set.

    Continuous features contribute cityblock distance; categorical features are
    reduced to one code each and contribute Hamming distance scaled back to a
    count of differing features. Sets smaller than two pairs score zero.
    """
    n_features = len(num_idx) + len(cat_groups)
    if n_features == 0 or len(x_cf) < 2:
        return 0.0

    n_pairs = len(x_cf) * (len(x_cf) - 1) // 2
    d_cont = pdist(x_cf[:, num_idx], metric="cityblock") if num_idx else np.zeros(n_pairs)

    if cat_groups:
        d_cat = pdist(categorical_codes(x_cf, cat_groups), metric="hamming") * len(cat_groups)
    else:
        d_cat = np.zeros(n_pairs)

    return float(np.mean((d_cont + d_cat) / n_features))


def directional_diversity_cosine(x_orig: np.ndarray, x_cf: np.ndarray) -> float:
    """Angular diversity of the change vectors ``v_i = c_i - x``: 1 - mean cos.

    Vectors live in the classifier's input space (standardized numericals +
    one-hot categoricals), so the score only reflects *where* each CF moves,
    not how far. 0 = all CFs push in the same direction, 1 = orthogonal, up to
    2 for opposing directions. CFs identical to the query carry no direction
    and are dropped; sets with fewer than two directed CFs score 0.
    """
    if len(x_cf) == 0:
        return 0.0
    vectors = x_cf - x_orig
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 1e-12]
    norms = norms[norms > 1e-12]
    k = len(vectors)
    if k < 2:
        return 0.0
    unit = vectors / norms[:, None]
    cos = unit @ unit.T
    mean_cos = (cos.sum() - np.trace(cos)) / (k * (k - 1))
    return 1.0 - float(mean_cos)


def to_ordinal_space(X: np.ndarray, num_idx: list[int], cat_groups: list[list[int]]) -> np.ndarray:
    """Rewrite a one-hot model matrix into DICTUM's ordinal model space.

    The result is `[numerics | one integer code per categorical feature]`, which
    is the representation DICTUM's metrics — LOF in particular — operate on.

    Args:
        X: Rows in the one-hot model space.
        num_idx: Column indices of the continuous features.
        cat_groups: One-hot column indices per categorical feature.

    Returns:
        Array of shape (len(X), len(num_idx) + len(cat_groups)).
    """
    numeric_block = X[:, num_idx] if num_idx else np.zeros((len(X), 0), dtype=X.dtype)
    if not cat_groups:
        return numeric_block.astype(np.float32)
    codes = categorical_codes(X, cat_groups).astype(np.float32)
    return np.hstack([numeric_block.astype(np.float32), codes])


# ---------------------------------------------------------------------------
# Dataset + model loading
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    """Everything needed to score one dataset at one seed.

    Two model spaces are tracked because they need not be the same one. The run
    generated its counterfactuals in `gen_dataset`'s space, which is also the
    space its classifier was trained in, while the metrics are reported in
    `dataset`'s space. Keeping them apart lets a numerically awkward metric
    space be measured without forcing the generators to search in it.
    """

    dataset: MethodDataset
    gen_dataset: MethodDataset
    same_space: bool
    disc_model: SimpleMLPClassifier | SpaceAdapterClassifier
    lof: LocalOutlierFactor
    num_idx: list[int]
    cat_groups: list[list[int]]
    num_ranges: np.ndarray
    dataset_dir_name: str
    # Applied to metric-space rows before any metric reads them. Identity for
    # one-hot metrics, or the collapse to ordinal codes for DICTUM parity.
    metric_encoding: str = "onehot"
    metric_num_idx: list[int] = field(default_factory=list)
    metric_cat_groups: list[list[int]] = field(default_factory=list)

    def to_metric_space(self, X: np.ndarray) -> np.ndarray:
        """Project one-hot model-space rows into the space metrics are read in."""
        if self.metric_encoding == "onehot":
            return X
        return to_ordinal_space(X, self.num_idx, self.cat_groups)


def config_stem(dataset_key: str) -> str:
    """Map a short dataset name to its config file stem."""
    return "lending_club_split" if dataset_key == "lending-club" else f"{dataset_key}_split"


def build_bundle(
    dataset_key: str,
    configs_root: Path,
    data_root: Path,
    seed_root: Path,
    scaler: str,
    hidden_layers: list[int],
    gen_scaler: Optional[str] = None,
    metric_encoding: str = "onehot",
    disc_scaler: Optional[str] = None,
) -> DatasetBundle:
    """Rebuild the dataset and classifier exactly as the run saw them.

    Args:
        scaler: Model space the metrics are reported in.
        gen_scaler: Model space the run generated in. Defaults to `scaler`,
            i.e. generated and measured in one space.
        metric_encoding: "onehot" keeps the repository's representation for the
            metrics; "ordinal" collapses each one-hot block to a single code,
            matching the space DICTUM computes its metrics in.
        disc_scaler: Model space the classifier was trained in. Defaults to
            `gen_scaler`. When it differs, the classifier is wrapped so that
            generation-space rows are converted into its space before every
            prediction, mirroring SpaceAdapterClassifier in the run itself.
    """
    cfg_stem = config_stem(dataset_key)
    val_path = data_root / dataset_key / "val.csv"
    gen_scaler = gen_scaler or scaler
    disc_scaler = disc_scaler or gen_scaler
    same_space = gen_scaler == scaler

    def _load() -> TrainTestFileDataset:
        return TrainTestFileDataset(
            config_path=str(configs_root / f"{cfg_stem}.yaml"),
            train_data_path=str(data_root / dataset_key / "train.csv"),
            test_data_path=str(data_root / dataset_key / "test.csv"),
            val_data_path=str(val_path),
        )

    file_ds = _load()
    dataset = MethodDataset(file_ds, build_model_space_pipeline(scaler))
    # A second MethodDataset over its own file dataset: MethodDataset writes the
    # feature indices back onto the file dataset it wraps, so the two cannot
    # share one.
    gen_dataset = (
        dataset if same_space else MethodDataset(_load(), build_model_space_pipeline(gen_scaler))
    )

    num_idx = list(dataset.numerical_features_indices)
    cat_groups = [list(g) for g in dataset.categorical_features_lists]

    X_train = np.asarray(dataset.X_train)
    # Ranges live in original units, which is where epsilon-sparsity is judged.
    X_train_original = dataset.inverse_transform(X_train.copy())
    if num_idx:
        train_num = X_train_original[:, num_idx].astype(float)
        num_ranges = np.clip(train_num.max(axis=0) - train_num.min(axis=0), 1e-6, None)
    else:
        num_ranges = np.array([])

    disc_path = seed_root / cfg_stem / "fold_0" / "disc_model_SimpleMLPClassifier.pt"
    disc_model = SimpleMLPClassifier(
        num_inputs=X_train.shape[1], num_targets=2, hidden_layers=hidden_layers
    )
    disc_model.load(str(disc_path))
    disc_model.eval()
    if disc_scaler != gen_scaler:
        # The classifier reads a third space; the adapter converts each
        # generation-space row through original units into it.
        disc_dataset = (
            dataset
            if disc_scaler == scaler
            else MethodDataset(_load(), build_model_space_pipeline(disc_scaler))
        )
        disc_model = SpaceAdapterClassifier(
            base_model=disc_model,
            caller_dataset=gen_dataset,
            model_dataset=disc_dataset,
        )

    # In the ordinal metric space the columns are re-laid out as
    # [numerics | one code per categorical], so the metric indices differ from
    # the one-hot ones the model space uses.
    if metric_encoding == "ordinal":
        metric_num_idx = list(range(len(num_idx)))
        metric_cat_groups = [[len(num_idx) + j] for j in range(len(cat_groups))]
        lof_train = to_ordinal_space(X_train, num_idx, cat_groups)
    else:
        metric_num_idx = num_idx
        metric_cat_groups = cat_groups
        lof_train = X_train

    lof = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, novelty=True)
    lof.fit(lof_train)

    return DatasetBundle(
        dataset=dataset,
        gen_dataset=gen_dataset,
        same_space=same_space,
        disc_model=disc_model,
        lof=lof,
        num_idx=num_idx,
        cat_groups=cat_groups,
        num_ranges=num_ranges,
        dataset_dir_name=cfg_stem,
        metric_encoding=metric_encoding,
        metric_num_idx=metric_num_idx,
        metric_cat_groups=metric_cat_groups,
    )


def _predict(disc_model: SimpleMLPClassifier | SpaceAdapterClassifier, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = disc_model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        return torch.argmax(logits, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------


def evaluate_method(
    bundle: DatasetBundle,
    method_dir: str,
    csv_suffix: str,
    cf_in_original_units: bool,
    seed_root: Path,
    cf_per_instance: int,
    keep_per_factual: int,
    discretize: bool = False,
) -> Optional[dict[str, float]]:
    """Score one method's saved counterfactuals for one dataset and seed."""
    fold_dir = seed_root / bundle.dataset_dir_name / method_dir / "fold_0"
    cf_path = fold_dir / f"counterfactuals_{csv_suffix}.csv"
    factuals_path = fold_dir / f"factuals_{csv_suffix}.csv"
    if not cf_path.exists() or not factuals_path.exists():
        logger.warning("Missing CF or factuals CSV under %s", fold_dir)
        return None

    cf_raw = pd.read_csv(cf_path).to_numpy(dtype=np.float32)
    factuals = pd.read_csv(factuals_path).to_numpy(dtype=np.float32)

    # A generator whose training diverged can emit inf/NaN rows. Those are not
    # counterfactuals, so they are forced invalid below; they are zeroed first
    # only so the scaler and the classifier have finite input to work on.
    finite_rows = np.all(np.isfinite(cf_raw), axis=1)
    n_nonfinite = int((~finite_rows).sum())
    if n_nonfinite:
        logger.warning(
            "%s: %d of %d counterfactual rows are non-finite; counting them invalid",
            method_dir,
            n_nonfinite,
            len(cf_raw),
        )
        cf_raw = cf_raw.copy()
        cf_raw[~finite_rows] = 0.0

    # Original units are the pivot between the two spaces. The classifier only
    # ever sees generation-space values, because that is what it was trained on;
    # the quality metrics only ever see metric-space values.
    if cf_in_original_units:
        cf_original = cf_raw
        cf_gen = bundle.gen_dataset.transform(cf_raw.copy()).astype(np.float32)
    else:
        cf_gen = cf_raw
        cf_original = bundle.gen_dataset.inverse_transform(cf_raw.copy())

    # DiCE and CCHVAE treat each one-hot column as an independent binary
    # variable, so they emit blocks with several categories set at once, or
    # none. Snapping happens in the generation space and before the classifier
    # runs, so validity is judged on the counterfactual actually being reported.
    n_discretized = 0
    if discretize and bundle.cat_groups:
        cf_snapped = apply_categorical_discretization(bundle.cat_groups, cf_gen.copy())
        n_discretized = int(np.any(cf_snapped != cf_gen, axis=1).sum())
        if n_discretized:
            logger.info(
                "%s: snapped %d of %d counterfactual rows onto valid one-hot blocks",
                method_dir,
                n_discretized,
                len(cf_gen),
            )
        cf_gen = cf_snapped.astype(np.float32)
        cf_original = bundle.gen_dataset.inverse_transform(cf_gen.copy())

    cf_onehot = (
        cf_gen if bundle.same_space else bundle.dataset.transform(cf_original.copy())
    ).astype(np.float32)
    cf_model = bundle.to_metric_space(cf_onehot).astype(np.float32)

    n_factuals = len(factuals)
    expected = n_factuals * cf_per_instance
    if cf_model.shape[0] != expected:
        logger.warning(
            "%s: expected %d CF rows for %d factuals but found %d — skipping",
            method_dir,
            expected,
            n_factuals,
            cf_model.shape[0],
        )
        return None

    # Factuals are saved in the generation space, which is what the classifier
    # reads; the metrics need them alongside the counterfactuals in the metric
    # space, so they are converted through original units the same way.
    factuals_original = bundle.gen_dataset.inverse_transform(factuals.copy())
    factuals_onehot = (
        factuals if bundle.same_space else bundle.dataset.transform(factuals_original.copy())
    ).astype(np.float32)
    factuals_model = bundle.to_metric_space(factuals_onehot).astype(np.float32)

    # The target is the flip of the classifier's own call on each factual, which
    # is what the aligned runs generate towards in both directions.
    factual_preds = _predict(bundle.disc_model, factuals)
    y_target = np.abs(1 - factual_preds)

    cf_preds = _predict(bundle.disc_model, cf_gen)
    pool_valid = cf_preds.reshape(n_factuals, cf_per_instance) == y_target[:, None]
    pool_valid &= finite_rows.reshape(n_factuals, cf_per_instance)

    cf_blocks = cf_model.reshape(n_factuals, cf_per_instance, -1)
    cf_blocks_original = cf_original.reshape(n_factuals, cf_per_instance, -1)
    cf_blocks_onehot = cf_onehot.reshape(n_factuals, cf_per_instance, -1)

    keep = min(keep_per_factual, cf_per_instance)
    # Valid counterfactuals first, original order preserved within each class.
    order = np.argsort(~pool_valid, axis=1, kind="stable")[:, :keep]
    rows = np.arange(n_factuals)[:, None]
    kept_cf = cf_blocks[rows, order]
    kept_cf_original = cf_blocks_original[rows, order]
    kept_cf_onehot = cf_blocks_onehot[rows, order]
    kept_valid = pool_valid[rows, order]

    prox_vals: list[float] = []
    spars_cat_vals: list[float] = []
    eps_spars_vals: list[float] = []
    lof_vals: list[float] = []
    div_vals: list[float] = []
    dir_div_vals: list[float] = []
    kept_validity: list[float] = []

    for i in range(n_factuals):
        kept_validity.append(float(kept_valid[i].mean()))
        valid_mask = kept_valid[i]
        if not valid_mask.any():
            # No valid counterfactual for this factual, so it contributes to
            # validity but to none of the quality metrics.
            continue

        cfs = kept_cf[i][valid_mask]
        cfs_original = kept_cf_original[i][valid_mask]
        origs = np.tile(factuals_model[i], (len(cfs), 1))
        origs_original = np.tile(factuals_original[i], (len(cfs), 1))

        prox_vals.append(proximity_continuous(origs, cfs, bundle.metric_num_idx))
        spars_cat_vals.append(sparsity_categorical(origs, cfs, bundle.metric_cat_groups))
        eps_spars_vals.append(
            epsilon_sparsity(origs_original, cfs_original, bundle.num_idx, bundle.num_ranges)
        )
        lof_vals.append(lof_log_median(bundle.lof, cfs))
        div_vals.append(diversity_mixed(cfs, bundle.metric_num_idx, bundle.metric_cat_groups))
        dir_div_vals.append(
            directional_diversity_cosine(factuals_onehot[i], kept_cf_onehot[i][valid_mask])
        )

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "validity": _mean(kept_validity),
        "pool_validity": float(pool_valid.mean()),
        "prox_cont": _mean(prox_vals),
        "spars_cat": _mean(spars_cat_vals),
        "epsilon_spars": _mean(eps_spars_vals),
        "lof": _mean(lof_vals),
        "div": _mean(div_vals),
        "dir_div": _mean(dir_div_vals),
        "n_factuals": float(n_factuals),
        "n_scored_factuals": float(len(prox_vals)),
        "n_nonfinite_cfs": float(n_nonfinite),
        "n_discretized_cfs": float(n_discretized),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

METRIC_COLUMNS = [
    ("validity", "Validity $\\uparrow$", "max"),
    ("prox_cont", "Prox.-Cont $\\downarrow$", "min"),
    ("spars_cat", "Spars.-Cat $\\downarrow$", "min"),
    ("epsilon_spars", "$\\epsilon$-Spars. $\\downarrow$", "min"),
    ("lof", "LOF $\\downarrow$", "min"),
    ("div", "Diversity $\\uparrow$", "max"),
    ("dir_div", "Dir.-Div $\\uparrow$", "max"),
]

DATASET_DISPLAY = {
    "adult": "Adult Income",
    "bank": "Bank",
    "default": "Default of Credit Card",
    "gmc": "Give Me Some Credit",
    "lending-club": "Lending Club",
}


def render_raw_markdown(
    per_seed: pd.DataFrame, datasets: list[str], methods: list[tuple], seed: int
) -> str:
    """Render one seed's values as markdown, without any averaging.

    A method with no row for this seed prints as `--` rather than being dropped,
    so a cell that failed to run stays visible in the table.
    """
    headers = ["Method"] + [key for key, _, _ in METRIC_COLUMNS]
    rows_for_seed = per_seed[per_seed["seed"] == seed]
    blocks: list[str] = []
    for dataset_key in datasets:
        rows = rows_for_seed[rows_for_seed["dataset"] == dataset_key]
        lines = [
            f"### {DATASET_DISPLAY.get(dataset_key, dataset_key)}",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        for pretty, *_ in methods:
            row = rows[rows["method"] == pretty]
            if row.empty:
                cells = ["--"] * len(METRIC_COLUMNS)
            else:
                cells = [f"{row.iloc[0][key]:.3f}" for key, _, _ in METRIC_COLUMNS]
            lines.append("| " + " | ".join([pretty, *cells]) + " |")
        blocks.append("\n".join(lines))
    return f"# Raw results, seed {seed}\n\n" + "\n\n".join(blocks) + "\n"


def render_raw_latex(
    per_seed: pd.DataFrame, datasets: list[str], methods: list[tuple], seed: int
) -> str:
    """Render one seed's values as a LaTeX table, without any averaging."""
    rows_for_seed = per_seed[per_seed["seed"] == seed]
    lines = [
        "\\begin{table}[H]",
        f"\\caption{{DICTUM-aligned results, seed {seed} (single run, no averaging).}}",
        f"\\label{{tab:dictum_aligned_seed{seed}}}",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{l{'c' * len(METRIC_COLUMNS)}}}",
        "\\toprule",
        " & ".join(["\\textbf{Method}"] + [f"\\textbf{{{h}}}" for _, h, _ in METRIC_COLUMNS])
        + " \\\\",
        "\\midrule",
    ]
    for dataset_key in datasets:
        rows = rows_for_seed[rows_for_seed["dataset"] == dataset_key]
        display = DATASET_DISPLAY.get(dataset_key, dataset_key)
        lines.append(
            f"\\multicolumn{{{len(METRIC_COLUMNS) + 1}}}{{c}}{{\\textit{{{display}}}}} \\\\"
        )
        lines.append("\\midrule")
        for pretty, *_ in methods:
            row = rows[rows["method"] == pretty]
            if row.empty:
                cells = ["--"] * len(METRIC_COLUMNS)
            else:
                cells = [f"{row.iloc[0][key]:.2f}" for key, _, _ in METRIC_COLUMNS]
            lines.append(f"{pretty} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def render_markdown(table: pd.DataFrame, datasets: list[str], methods: list[tuple]) -> str:
    """Render the aggregated table as pipe-separated markdown, one block per dataset."""
    headers = ["Method"] + [key for key, _, _ in METRIC_COLUMNS] + ["seeds"]
    blocks: list[str] = []
    for dataset_key in datasets:
        rows = table[table["dataset"] == dataset_key]
        if rows.empty:
            continue
        lines = [
            f"### {DATASET_DISPLAY.get(dataset_key, dataset_key)}",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        for pretty, *_ in methods:
            row = rows[rows["method"] == pretty]
            if row.empty:
                lines.append(
                    "| " + " | ".join([pretty, *(["--"] * len(METRIC_COLUMNS)), "0"]) + " |"
                )
                continue
            row = row.iloc[0]
            cells = [
                f"{row[f'{key}_mean']:.3f} ± {row[f'{key}_std']:.3f}"
                for key, _, _ in METRIC_COLUMNS
            ]
            lines.append(
                "| " + " | ".join([pretty, *cells, str(int(row[f"{METRIC_KEYS[0]}_count"]))]) + " |"
            )
        blocks.append("\n".join(lines))
    return "# Combined results, mean ± std across seeds\n\n" + "\n\n".join(blocks) + "\n"


def render_latex(table: pd.DataFrame, datasets: list[str], methods: list[tuple]) -> str:
    """Render the aggregated table as a LaTeX table with mean ± std cells."""
    lines = [
        "\\begin{table}[H]",
        "\\label{tab:dictum_aligned}",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{l{'c' * len(METRIC_COLUMNS)}}}",
        "\\toprule",
        " & ".join(["\\textbf{Method}"] + [f"\\textbf{{{h}}}" for _, h, _ in METRIC_COLUMNS])
        + " \\\\",
        "\\midrule",
    ]

    for dataset_key in datasets:
        rows = table[table["dataset"] == dataset_key]
        if rows.empty:
            continue
        display = DATASET_DISPLAY.get(dataset_key, dataset_key)
        lines.append(
            f"\\multicolumn{{{len(METRIC_COLUMNS) + 1}}}{{c}}{{\\textit{{{display}}}}} \\\\"
        )
        lines.append("\\midrule")
        for pretty, *_ in methods:
            row = rows[rows["method"] == pretty]
            if row.empty:
                lines.append(f"{pretty} & " + " & ".join(["--"] * len(METRIC_COLUMNS)) + " \\\\")
                continue
            row = row.iloc[0]
            cells = [
                f"{row[f'{key}_mean']:.2f} $\\pm$ {{{row[f'{key}_std']:.2f}}}"
                for key, _, _ in METRIC_COLUMNS
            ]
            lines.append(f"{pretty} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/dictum")
    parser.add_argument(
        "--datasets", nargs="+", default=["adult", "bank", "default", "gmc", "lending-club"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--configs-root", default="config/datasets")
    parser.add_argument("--data-root", default="data_train_test_val")
    parser.add_argument(
        "--scaler",
        choices=("standard", "minmax", "minmax_qt"),
        default="standard",
        help="Model space the metrics are reported in.",
    )
    parser.add_argument(
        "--generation-scaler",
        choices=("standard", "minmax", "minmax_qt"),
        default=None,
        help=(
            "Model space the run generated in, which is also the space its "
            "classifier was trained in. Defaults to --scaler. Set it when a run "
            "generated in one space but should be measured in another, e.g. "
            "generating in minmax for numerical stability while reporting "
            "DICTUM-comparable z-scored metrics."
        ),
    )
    parser.add_argument(
        "--disc-scaler",
        choices=("standard", "minmax", "minmax_qt"),
        default=None,
        help=(
            "Model space the classifier under explanation was trained in. "
            "Defaults to --generation-scaler. Set it for runs that used "
            "experiment.disc_model_space_scaler, e.g. the shared DICTUM "
            "classifier (standard) explained by a minmax_qt DiCoFlex run."
        ),
    )
    parser.add_argument(
        "--metric-encoding",
        choices=("onehot", "ordinal"),
        default="onehot",
        help=(
            "Categorical representation the metrics are computed in. 'ordinal' "
            "collapses each one-hot block to a single integer code, which is the "
            "space DICTUM measures in and changes LOF materially; 'onehot' keeps "
            "this repository's representation."
        ),
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[32, 32],
        help="Classifier hidden layer sizes; must match the trained checkpoint.",
    )
    parser.add_argument("--cf-per-instance", type=int, default=100)
    parser.add_argument("--keep-per-factual", type=int, default=KEEP_PER_FACTUAL)
    parser.add_argument(
        "--discretize-categoricals",
        action="store_true",
        help=(
            "Snap each one-hot block to a single category before scoring, and "
            "before the validity check. DiCE and CCHVAE vary one-hot columns "
            "independently and so emit invalid blocks; without this their "
            "metrics and validity describe points that are not valid rows."
        ),
    )
    parser.add_argument(
        "--raw-seed",
        type=int,
        default=None,
        help=(
            "Also write an un-averaged table for this single seed, as "
            "<output>.seed<N>.{md,tex}. Defaults to the first --seeds entry."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[pretty for pretty, *_ in METHODS],
        help="Restrict scoring to these methods; defaults to all of them.",
    )
    parser.add_argument(
        "--output",
        default="results/dictum/dictum_metrics",
        help="Output path prefix; .csv, .md and .tex are written alongside each other.",
    )
    return parser.parse_args()


def main() -> None:
    """Score every (dataset, method, seed) cell and write per-seed and aggregate tables."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()

    results_root = Path(args.results_root)
    configs_root = Path(args.configs_root)
    data_root = Path(args.data_root)
    selected_methods = (
        [m for m in METHODS if m[0] in set(args.methods)] if args.methods else METHODS
    )

    records: list[dict] = []
    for dataset_key in args.datasets:
        for seed in args.seeds:
            seed_root = results_root / f"seed_{seed}"
            if not seed_root.exists():
                logger.warning("Missing seed root %s", seed_root)
                continue
            logger.info("=== %s / seed %d ===", dataset_key, seed)
            try:
                bundle = build_bundle(
                    dataset_key,
                    configs_root,
                    data_root,
                    seed_root,
                    args.scaler,
                    args.hidden_layers,
                    args.generation_scaler,
                    args.metric_encoding,
                    args.disc_scaler,
                )
            except FileNotFoundError as exc:
                logger.warning("Cannot build bundle for %s seed %d: %s", dataset_key, seed, exc)
                continue

            for pretty, method_dir, csv_suffix, raw_space in selected_methods:
                res = evaluate_method(
                    bundle,
                    method_dir,
                    csv_suffix,
                    raw_space,
                    seed_root,
                    args.cf_per_instance,
                    args.keep_per_factual,
                    args.discretize_categoricals,
                )
                if res is None:
                    continue
                logger.info(
                    "  %-10s validity=%.3f prox=%.3f lof=%.3f div=%.3f",
                    pretty,
                    res["validity"],
                    res["prox_cont"],
                    res["lof"],
                    res["div"],
                )
                records.append({"dataset": dataset_key, "method": pretty, "seed": seed, **res})

    if not records:
        logger.error("No results found under %s", results_root)
        return

    per_seed = pd.DataFrame(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output.with_suffix(".per_seed.csv"), index=False)

    agg = (
        per_seed.groupby(["dataset", "method"])[METRIC_KEYS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        col[0] if not col[1] else f"{col[0]}_{col[1]}" for col in agg.columns.to_flat_index()
    ]
    agg = agg.fillna({f"{key}_std": 0.0 for key in METRIC_KEYS})
    agg.to_csv(output.with_suffix(".csv"), index=False)

    thin = agg[agg[f"{METRIC_KEYS[0]}_count"] < len(args.seeds)]
    if not thin.empty:
        logger.warning(
            "Cells with fewer than %d seeds: %s",
            len(args.seeds),
            json.dumps(thin[["dataset", "method", f"{METRIC_KEYS[0]}_count"]].to_dict("records")),
        )

    output.with_suffix(".tex").write_text(render_latex(agg, args.datasets, selected_methods))
    output.with_suffix(".md").write_text(render_markdown(agg, args.datasets, selected_methods))
    logger.info("Wrote %s.{csv,per_seed.csv,md,tex}", output)

    raw_seed = args.raw_seed if args.raw_seed is not None else args.seeds[0]
    if raw_seed in set(per_seed["seed"]):
        raw_md = output.with_suffix(f".seed{raw_seed}.md")
        raw_tex = output.with_suffix(f".seed{raw_seed}.tex")
        raw_md.write_text(render_raw_markdown(per_seed, args.datasets, selected_methods, raw_seed))
        raw_tex.write_text(render_raw_latex(per_seed, args.datasets, selected_methods, raw_seed))
        logger.info("Wrote %s and %s", raw_md, raw_tex)
    else:
        logger.warning("No results for raw seed %s; skipping the single-seed table", raw_seed)


if __name__ == "__main__":
    main()
