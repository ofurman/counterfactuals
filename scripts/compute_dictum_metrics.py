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
from dataclasses import dataclass
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

METRIC_KEYS = ["validity", "pool_validity", "prox_cont", "spars_cat", "epsilon_spars", "lof", "div"]


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


def diversity_mixed(x_cf: np.ndarray, num_idx: list[int], cat_groups: list[list[int]]) -> float:
    """Mean pairwise mixed distance within one factual's counterfactual set.

    Continuous features contribute cityblock distance; categorical features are
    collapsed to one code per one-hot group and contribute Hamming distance
    scaled back to a count of differing groups. Sets smaller than two pairs
    score zero.
    """
    n_features = len(num_idx) + len(cat_groups)
    if n_features == 0 or len(x_cf) < 2:
        return 0.0

    n_pairs = len(x_cf) * (len(x_cf) - 1) // 2
    d_cont = pdist(x_cf[:, num_idx], metric="cityblock") if num_idx else np.zeros(n_pairs)

    if cat_groups:
        codes = np.zeros((len(x_cf), len(cat_groups)), dtype=int)
        for j, group in enumerate(cat_groups):
            codes[:, j] = np.argmax(x_cf[:, group], axis=1)
        d_cat = pdist(codes, metric="hamming") * len(cat_groups)
    else:
        d_cat = np.zeros(n_pairs)

    return float(np.mean((d_cont + d_cat) / n_features))


# ---------------------------------------------------------------------------
# Dataset + model loading
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    """Everything needed to score one dataset at one seed."""

    dataset: MethodDataset
    disc_model: SimpleMLPClassifier
    lof: LocalOutlierFactor
    num_idx: list[int]
    cat_groups: list[list[int]]
    num_ranges: np.ndarray
    dataset_dir_name: str


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
) -> DatasetBundle:
    """Rebuild the dataset and classifier exactly as the run saw them."""
    cfg_stem = config_stem(dataset_key)
    val_path = data_root / dataset_key / "val.csv"

    file_ds = TrainTestFileDataset(
        config_path=str(configs_root / f"{cfg_stem}.yaml"),
        train_data_path=str(data_root / dataset_key / "train.csv"),
        test_data_path=str(data_root / dataset_key / "test.csv"),
        val_data_path=str(val_path),
    )
    dataset = MethodDataset(file_ds, build_model_space_pipeline(scaler))

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

    lof = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, novelty=True)
    lof.fit(X_train)

    return DatasetBundle(
        dataset=dataset,
        disc_model=disc_model,
        lof=lof,
        num_idx=num_idx,
        cat_groups=cat_groups,
        num_ranges=num_ranges,
        dataset_dir_name=cfg_stem,
    )


def _predict(disc_model: SimpleMLPClassifier, X: np.ndarray) -> np.ndarray:
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

    cf_model = (
        bundle.dataset.transform(cf_raw.copy()).astype(np.float32)
        if cf_in_original_units
        else cf_raw
    )

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

    # The target is the flip of the classifier's own call on each factual, which
    # is what the aligned runs generate towards in both directions.
    factual_preds = _predict(bundle.disc_model, factuals)
    y_target = np.abs(1 - factual_preds)

    cf_preds = _predict(bundle.disc_model, cf_model)
    pool_valid = cf_preds.reshape(n_factuals, cf_per_instance) == y_target[:, None]
    pool_valid &= finite_rows.reshape(n_factuals, cf_per_instance)

    cf_blocks = cf_model.reshape(n_factuals, cf_per_instance, -1)
    cf_blocks_original = bundle.dataset.inverse_transform(cf_model.copy()).reshape(
        n_factuals, cf_per_instance, -1
    )
    factuals_original = bundle.dataset.inverse_transform(factuals.copy())

    keep = min(keep_per_factual, cf_per_instance)
    # Valid counterfactuals first, original order preserved within each class.
    order = np.argsort(~pool_valid, axis=1, kind="stable")[:, :keep]
    rows = np.arange(n_factuals)[:, None]
    kept_cf = cf_blocks[rows, order]
    kept_cf_original = cf_blocks_original[rows, order]
    kept_valid = pool_valid[rows, order]

    prox_vals: list[float] = []
    spars_cat_vals: list[float] = []
    eps_spars_vals: list[float] = []
    lof_vals: list[float] = []
    div_vals: list[float] = []
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
        origs = np.tile(factuals[i], (len(cfs), 1))
        origs_original = np.tile(factuals_original[i], (len(cfs), 1))

        prox_vals.append(proximity_continuous(origs, cfs, bundle.num_idx))
        spars_cat_vals.append(sparsity_categorical(origs, cfs, bundle.cat_groups))
        eps_spars_vals.append(
            epsilon_sparsity(origs_original, cfs_original, bundle.num_idx, bundle.num_ranges)
        )
        lof_vals.append(lof_log_median(bundle.lof, cfs))
        div_vals.append(diversity_mixed(cfs, bundle.num_idx, bundle.cat_groups))

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
        "n_factuals": float(n_factuals),
        "n_scored_factuals": float(len(prox_vals)),
        "n_nonfinite_cfs": float(n_nonfinite),
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
    parser.add_argument("--scaler", choices=("standard", "minmax"), default="standard")
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
