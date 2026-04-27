"""Per-criterion selection table for models/<dataset>_split runs.

Protocol (matches the comment in the user's table):
  - For each factual we have 100 candidate CFs.
  - For each of three target metrics (proximity-cont, sparsity-cat, eps-sparsity)
    we pick the 10 *valid* CFs that best optimise that metric, then report
    the mean of that metric over the 10 selected CFs and the mixed-feature
    pairwise diversity across them.
  - Continuous columns are inverse-MinMaxed and re-scaled with StandardScaler
    fitted on the training continuous columns. Validity uses MinMax space
    (the disc_model lives there).

Usage:
    uv run python scripts/compute_models_perselect_table.py \
        --models-root models \
        --datasets adult bank default gmc lending-club
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_sweep_actionability_metrics import (  # noqa: E402
    DATASET_DISPLAY,
    DatasetBundle,
    _to_metric_space,
    build_bundle,
)

logger = logging.getLogger(__name__)

KEEP = 10
CF_PER_INSTANCE = 100

METHODS: list[tuple[str, str, str, bool, int]] = [
    ("DICE", "DiceExplainerWrapper", "DiceExplainerWrapper_SimpleMLPClassifier", False, 0),
    ("CCHVAE", "CCHVAE", "CCHVAE_SimpleMLPClassifier", False, 0),
    ("DiCoFlex", "DiCoFlex", "DiCoFlex_SimpleMLPClassifier", True, 1),
]


def _spars_cat_per_cf(orig: np.ndarray, cf: np.ndarray, cat_groups: list[list[int]]) -> np.ndarray:
    if not cat_groups:
        return np.zeros(len(cf), dtype=float)
    D_cat = len(cat_groups)
    per = np.zeros(len(cf), dtype=float)
    for g in cat_groups:
        per += np.any(orig[:, g] != cf[:, g], axis=1).astype(float)
    return per / D_cat


def _eps_spars_per_cf(
    orig: np.ndarray,
    cf: np.ndarray,
    num_idx: list[int],
    ranges: np.ndarray,
    eps: float = 0.05,
) -> np.ndarray:
    if not num_idx:
        return np.zeros(len(cf), dtype=float)
    D_num = len(num_idx)
    diff = np.abs(orig[:, num_idx].astype(float) - cf[:, num_idx].astype(float))
    sig = (diff > eps * ranges.reshape(1, -1)).astype(float)
    return sig.sum(axis=1) / D_num


def _prox_cont_per_cf(orig: np.ndarray, cf: np.ndarray, num_idx: list[int]) -> np.ndarray:
    if not num_idx:
        return np.zeros(len(cf), dtype=float)
    return np.abs(orig[:, num_idx] - cf[:, num_idx]).mean(axis=1)


def _diversity_mixed(cfs: np.ndarray, num_idx: list[int], cat_groups: list[list[int]]) -> float:
    if len(cfs) < 2:
        return 0.0
    D_total = len(num_idx) + len(cat_groups)
    if D_total == 0:
        return 0.0
    d_num = pdist(cfs[:, num_idx], metric="cityblock") if num_idx else 0.0
    if cat_groups:
        cat_enc = np.zeros((len(cfs), len(cat_groups)), dtype=int)
        for j, g in enumerate(cat_groups):
            cat_enc[:, j] = np.argmax(cfs[:, g], axis=1)
        d_cat = pdist(cat_enc, metric="hamming") * len(cat_groups)
    else:
        d_cat = 0.0
    mixed = d_num + d_cat
    if np.size(mixed) == 0:
        return 0.0
    return float(np.mean(mixed) / D_total)


def evaluate_method(
    bundle: DatasetBundle,
    cf_csv: Path,
    cf_in_raw_space: bool,
    target_class: int,
) -> dict[str, float] | None:
    if not cf_csv.exists():
        return None

    cf_arr = pd.read_csv(cf_csv).to_numpy(dtype=np.float32)

    if cf_in_raw_space:
        cf_minmax = cf_arr.copy()
        if bundle.cont_idx:
            cf_minmax[:, bundle.cont_idx] = bundle.minmax_scaler.transform(
                cf_arr[:, bundle.cont_idx]
            ).astype(np.float32)
    else:
        cf_minmax = cf_arr

    X_test_minmax = np.asarray(bundle.dataset.X_test)
    with torch.no_grad():
        test_preds = (
            torch.argmax(bundle.disc_model(torch.from_numpy(X_test_minmax).float()), dim=1)
            .cpu()
            .numpy()
        )
    factual_pool_minmax = X_test_minmax[test_preds != target_class]

    n_total = (cf_minmax.shape[0] // CF_PER_INSTANCE) * CF_PER_INSTANCE
    cf_minmax = cf_minmax[:n_total]
    n_factuals = n_total // CF_PER_INSTANCE
    if n_factuals > len(factual_pool_minmax):
        n_factuals = len(factual_pool_minmax)
        cf_minmax = cf_minmax[: n_factuals * CF_PER_INSTANCE]

    # Validity in MinMax space.
    with torch.no_grad():
        preds = (
            torch.argmax(bundle.disc_model(torch.from_numpy(cf_minmax).float()), dim=1)
            .cpu()
            .numpy()
        )
    pred_ok = preds == target_class
    if bundle.num_idx:
        cont = cf_minmax[:, bundle.num_idx]
        in_range = np.all((cont >= -0.5) & (cont <= 1.5), axis=1)
    else:
        in_range = np.ones(len(cf_minmax), dtype=bool)
    valid = pred_ok & in_range

    cf_metric = _to_metric_space(
        cf_minmax, bundle.cont_idx, bundle.minmax_scaler, bundle.standard_scaler, bundle.scale
    )
    factual_metric = _to_metric_space(
        factual_pool_minmax,
        bundle.cont_idx,
        bundle.minmax_scaler,
        bundle.standard_scaler,
        bundle.scale,
    )[:n_factuals]

    cf_blocks = cf_metric.reshape(n_factuals, CF_PER_INSTANCE, -1)
    valid_blocks = valid.reshape(n_factuals, CF_PER_INSTANCE)

    prox_acc, div_prox = [], []
    sc_acc, div_sc = [], []
    es_acc, div_es = [], []

    for i in range(n_factuals):
        cfs = cf_blocks[i]
        ok = valid_blocks[i]
        if not ok.any():
            continue
        valid_cfs = cfs[ok]
        orig = np.tile(factual_metric[i], (len(valid_cfs), 1))

        prox = _prox_cont_per_cf(orig, valid_cfs, bundle.num_idx)
        sc = _spars_cat_per_cf(orig, valid_cfs, bundle.cat_groups)
        es = _eps_spars_per_cf(orig, valid_cfs, bundle.num_idx, bundle.num_ranges)

        keep = min(KEEP, len(valid_cfs))

        for vals, acc, div_acc in (
            (prox, prox_acc, div_prox),
            (sc, sc_acc, div_sc),
            (es, es_acc, div_es),
        ):
            order = np.argsort(vals, kind="stable")[:keep]
            sel = valid_cfs[order]
            acc.append(float(np.mean(vals[order])))
            div_acc.append(_diversity_mixed(sel, bundle.num_idx, bundle.cat_groups))

    if not prox_acc:
        return None
    return {
        "prox_cont": float(np.mean(prox_acc)),
        "div_prox": float(np.mean(div_prox)),
        "spars_cat": float(np.mean(sc_acc)),
        "div_sc": float(np.mean(div_sc)),
        "eps_spars": float(np.mean(es_acc)),
        "div_es": float(np.mean(div_es)),
        "n_factuals_with_valid": len(prox_acc),
        "n_factuals_total": n_factuals,
    }


def render_table(rows: dict[str, dict[str, dict[str, float] | None]]) -> str:
    """rows[dataset_key][method] -> metrics dict or None.

    GRACE values are kept from the user-provided template (hard-coded below).
    """
    grace = {
        "adult": (0.17, 0.24, 0.14, 0.18, 0.20, 0.26),
        "bank": (0.41, 0.31, 0.038, 0.21, 0.28, 0.34),
        "default": (0.30, 0.28, 0.10, 0.31, 0.14, 0.32),
        "gmc": (0.43, 0.38, 0.029, 0.39, 0.15, 0.43),
        "lending-club": (0.37, 0.36, 0.16, 0.41, 0.37, 0.40),
    }

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\label{tab:protocol_B}")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l cc|cc|cc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Method} & \\textbf{Prox.-Cont $\\uparrow$} & \\textbf{diversity $\\uparrow$} "
        "& \\textbf{Spars.-Cat $\\downarrow$} & \\textbf{diversity $\\uparrow$} "
        "& \\textbf{$\\epsilon$-Spars. $\\downarrow$} & \\textbf{ diversity $\\uparrow$}\\\\"
    )

    def fmt(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}"

    for ds in ["adult", "bank", "default", "gmc", "lending-club"]:
        display = DATASET_DISPLAY.get(ds, ds)
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{c}}{{\\textit{{{display}}}}} \\\\")
        lines.append("\\midrule")
        for pretty, *_ in METHODS:
            r = rows.get(ds, {}).get(pretty)
            if r is None:
                cells = ["--"] * 6
            else:
                cells = [
                    fmt(r["prox_cont"]),
                    fmt(r["div_prox"]),
                    fmt(r["spars_cat"]),
                    fmt(r["div_sc"]),
                    fmt(r["eps_spars"]),
                    fmt(r["div_es"]),
                ]
            lines.append(f"{pretty} & " + " & ".join(cells) + "\\\\")
        g = grace[ds]
        lines.append("GRACE & " + " & ".join(fmt(v, 3 if v < 0.05 else 2) for v in g) + "\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-root", type=Path, default=Path("models"))
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "bank", "default", "gmc", "lending-club"],
    )
    p.add_argument("--configs-root", type=Path, default=Path("config/datasets"))
    p.add_argument("--data-root", type=Path, default=Path("data_train_test_val"))
    p.add_argument("--output", type=Path, default=Path("models_perselect_table.tex"))
    args = p.parse_args()

    rows: dict[str, dict[str, dict[str, float] | None]] = {}
    for ds in args.datasets:
        logger.info("=== %s ===", ds)
        ds_dir = args.models_root / f"{ds.replace('-', '_')}_split"
        disc_pt = ds_dir / "fold_0" / "disc_model_SimpleMLPClassifier.pt"
        if not disc_pt.exists():
            logger.warning("Missing disc model %s", disc_pt)
            rows[ds] = {p: None for p, *_ in METHODS}
            continue

        bundle = build_bundle(ds, args.configs_root, args.data_root, disc_pt, scale="standard")

        # Each method has its own copy of the disc_model in the sweep layout, but
        # in this layout it's shared. Use the shared one.
        rows[ds] = {}
        for pretty, mdir, csv_suffix, raw_space, target_cls in METHODS:
            cf_csv = ds_dir / mdir / "fold_0" / f"counterfactuals_{csv_suffix}.csv"
            res = evaluate_method(bundle, cf_csv, raw_space, target_cls)
            rows[ds][pretty] = res
            if res is None:
                logger.info("  %s: missing/no-valid", pretty)
            else:
                logger.info(
                    "  %s: %s (valid factuals %d/%d)",
                    pretty,
                    {
                        k: round(v, 4)
                        for k, v in res.items()
                        if k not in ("n_factuals_with_valid", "n_factuals_total")
                    },
                    res["n_factuals_with_valid"],
                    res["n_factuals_total"],
                )

    table = render_table(rows)
    args.output.write_text(table + "\n", encoding="utf-8")
    logger.info("wrote %s", args.output)
    print()
    print(table)


if __name__ == "__main__":
    main()
