"""Generate scaling analysis plots and tables: CF search time vs. dataset dimensionality.

Reads cf_metrics CSVs from models/ directory, aggregates timing data across folds,
and produces:
  1) Scaling curves: time vs. continuous features, time vs. total features (side by side)
  2) A dataset composition chart showing continuous vs. categorical feature split
  3) A heatmap with annotated feature composition
  4) LaTeX tables summarising mean +/- std search times
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"


@dataclass
class DatasetInfo:
    """Feature composition of a dataset."""

    n_continuous: int
    n_categorical: int
    n_total_encoded: int
    n_samples: int


DATASET_INFO: dict[str, DatasetInfo] = {
    "moons": DatasetInfo(n_continuous=2, n_categorical=0, n_total_encoded=2, n_samples=1000),
    "blobs": DatasetInfo(n_continuous=2, n_categorical=0, n_total_encoded=2, n_samples=1500),
    "law": DatasetInfo(n_continuous=3, n_categorical=2, n_total_encoded=13, n_samples=22407),
    "wine": DatasetInfo(n_continuous=13, n_categorical=0, n_total_encoded=13, n_samples=178),
    "heloc": DatasetInfo(n_continuous=23, n_categorical=0, n_total_encoded=23, n_samples=10458),
    "audit": DatasetInfo(n_continuous=23, n_categorical=0, n_total_encoded=23, n_samples=776),
    "lending_club": DatasetInfo(
        n_continuous=8, n_categorical=4, n_total_encoded=32, n_samples=30000
    ),
    "give_me_some_credit": DatasetInfo(
        n_continuous=7, n_categorical=3, n_total_encoded=44, n_samples=30000
    ),
    "bank_marketing": DatasetInfo(
        n_continuous=7, n_categorical=9, n_total_encoded=50, n_samples=40004
    ),
    "german_credit": DatasetInfo(
        n_continuous=7, n_categorical=11, n_total_encoded=57, n_samples=1000
    ),
    "digits": DatasetInfo(n_continuous=64, n_categorical=0, n_total_encoded=64, n_samples=1797),
    "credit_default": DatasetInfo(
        n_continuous=14, n_categorical=9, n_total_encoded=91, n_samples=30000
    ),
    "adult_census": DatasetInfo(
        n_continuous=4, n_categorical=8, n_total_encoded=191, n_samples=30000
    ),
}

# Convenience lookups derived from DATASET_INFO
DATASET_DIM: dict[str, int] = {k: v.n_total_encoded for k, v in DATASET_INFO.items()}
DATASET_CONTINUOUS: dict[str, int] = {k: v.n_continuous for k, v in DATASET_INFO.items()}
DATASET_CATEGORICAL: dict[str, int] = {k: v.n_categorical for k, v in DATASET_INFO.items()}
DATASET_SAMPLES: dict[str, int] = {k: v.n_samples for k, v in DATASET_INFO.items()}

METHOD_DISPLAY: dict[str, str] = {
    "WACH_OURS": "WACH",
    "DiceExplainerWrapper": "DiCE",
    "CaseBasedSACE": "SACE",
    "GLOBE_CE": "GLOBE-CE",
    "CEM_CF": "CEM",
    "GlobalGLANCE": "G-GLANCE",
    "GroupGLANCE": "Gr-GLANCE",
    "TCREx": "T-CREx",
    "CeFlow": "CEFlow",
}

# Categorize methods
LOCAL_METHODS = [
    "PPCEF",
    "DiceExplainerWrapper",
    "CCHVAE",
    "CADEX",
    "CeFlow",
    "WACH_OURS",
    "Artelt",
    "CaseBasedSACE",
    "CEGP",
    "CEM_CF",
]
GLOBAL_METHODS = ["AReS", "GLOBE_CE"]
GROUP_METHODS = ["GlobalGLANCE", "GroupGLANCE", "TCREx"]


@dataclass
class TimingRecord:
    dataset: str
    method: str
    model_type: str
    fold: str
    cf_search_time: float
    number_of_instances: float


def collect_timing_data() -> list[TimingRecord]:
    """Walk models/ directory and extract cf_search_time from all metrics files."""
    records: list[TimingRecord] = []

    for metrics_file in MODELS_DIR.rglob("cf_metrics_*.csv"):
        parts = metrics_file.relative_to(MODELS_DIR).parts
        if len(parts) < 4:
            continue
        dataset, method, fold, filename = parts[0], parts[1], parts[2], parts[3]

        # Skip dataset-level fold directories
        if method.startswith("fold_"):
            continue

        model_type = filename.replace("cf_metrics_", "").replace(".csv", "")

        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    records.append(
                        TimingRecord(
                            dataset=dataset,
                            method=method,
                            model_type=model_type,
                            fold=fold,
                            cf_search_time=float(row["cf_search_time"]),
                            number_of_instances=float(row["number_of_instances"]),
                        )
                    )
                except (KeyError, ValueError) as e:
                    logger.warning("Skipping %s: %s", metrics_file, e)

    logger.info("Collected %d timing records", len(records))
    return records


def build_dataframe(records: list[TimingRecord]) -> pd.DataFrame:
    """Convert records to a DataFrame with dataset dimensionality info."""
    df = pd.DataFrame([r.__dict__ for r in records])
    df["n_features"] = df["dataset"].map(DATASET_DIM)
    df["n_continuous"] = df["dataset"].map(DATASET_CONTINUOUS)
    df["n_categorical"] = df["dataset"].map(DATASET_CATEGORICAL)
    df["n_samples"] = df["dataset"].map(DATASET_SAMPLES)
    df["cat_ratio"] = df["n_categorical"] / (df["n_continuous"] + df["n_categorical"])
    df["method_display"] = df["method"].map(lambda m: METHOD_DISPLAY.get(m, m))
    return df


def plot_scaling_curves(df: pd.DataFrame, model_type: str = "MLPClassifier") -> None:
    """Two-row panel: top = time vs continuous features, bottom = time vs total (encoded)."""
    sub = df[df["model_type"] == model_type].copy()

    agg = (
        sub.groupby(["method", "method_display", "dataset", "n_features", "n_continuous"])[
            "cf_search_time"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    categories = [
        ("Local Methods", LOCAL_METHODS),
        ("Global Methods", GLOBAL_METHODS),
        ("Group Methods", GROUP_METHODS),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharey=False)

    for col, (title, method_list) in enumerate(categories):
        for row, (x_col, x_label) in enumerate(
            [
                ("n_continuous", "Continuous (Numerical) Features"),
                ("n_features", "Total Features (after one-hot encoding)"),
            ]
        ):
            ax = axes[row, col]
            for method in sorted(method_list):
                method_data = agg[agg["method"] == method].sort_values(x_col)
                if method_data.empty:
                    continue
                display_name = METHOD_DISPLAY.get(method, method)
                ax.plot(
                    method_data[x_col],
                    method_data["mean"],
                    marker="o",
                    label=display_name,
                    linewidth=1.5,
                    markersize=5,
                )
                ax.fill_between(
                    method_data[x_col],
                    (method_data["mean"] - method_data["std"]).clip(lower=1e-6),
                    method_data["mean"] + method_data["std"],
                    alpha=0.1,
                )

            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel("CF Search Time (s)", fontsize=11)
            if row == 0:
                ax.set_title(title, fontsize=13)
            ax.set_yscale("log")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(labelsize=10)

    fig.suptitle(
        f"CF Search Time Scaling — Continuous vs. Total Features ({model_type})",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"scaling_curves_{model_type}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.info("Saved scaling curves to %s", out_path)
    plt.close(fig)


def plot_dataset_composition(model_type: str = "MLPClassifier") -> None:
    """Stacked bar chart: continuous vs. one-hot-encoded categorical features per dataset."""
    dataset_order = sorted(DATASET_INFO.keys(), key=lambda d: DATASET_INFO[d].n_total_encoded)

    cont = [DATASET_INFO[d].n_continuous for d in dataset_order]
    cat_encoded = [
        DATASET_INFO[d].n_total_encoded - DATASET_INFO[d].n_continuous for d in dataset_order
    ]
    cat_raw = [DATASET_INFO[d].n_categorical for d in dataset_order]
    samples = [DATASET_INFO[d].n_samples for d in dataset_order]
    labels = [d.replace("_", "\n") for d in dataset_order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    x = np.arange(len(dataset_order))
    width = 0.6

    bars_cont = ax1.bar(x, cont, width, label="Continuous features", color="#2196F3", zorder=3)
    bars_cat = ax1.bar(
        x,
        cat_encoded,
        width,
        bottom=cont,
        label="Categorical (one-hot encoded)",
        color="#FF9800",
        zorder=3,
    )

    # Annotate bars
    for i, (c, ce, cr) in enumerate(zip(cont, cat_encoded, cat_raw)):
        total = c + ce
        ax1.text(i, total + 1, str(total), ha="center", va="bottom", fontsize=8, fontweight="bold")
        if cr > 0:
            ax1.text(
                i,
                c + ce / 2,
                f"{cr} cat\n→{ce} cols",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    ax1.set_ylabel("Number of Features", fontsize=12)
    ax1.set_title("Dataset Feature Composition: Continuous vs. Categorical", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y", zorder=0)

    # Bottom: categorical ratio
    cat_ratio = [ce / (c + ce) if (c + ce) > 0 else 0 for c, ce in zip(cont, cat_encoded)]
    colors = plt.cm.RdYlGn_r(np.array(cat_ratio))  # red = high cat ratio
    ax2.bar(x, cat_ratio, width, color=colors, zorder=3)
    ax2.set_ylabel("Categorical\nRatio", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.2, axis="y", zorder=0)

    for i, (r, s) in enumerate(zip(cat_ratio, samples)):
        ax2.text(i, r + 0.02, f"n={s:,}", ha="center", va="bottom", fontsize=7, rotation=45)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "dataset_feature_composition.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.info("Saved dataset composition chart to %s", out_path)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, model_type: str = "MLPClassifier") -> None:
    """Heatmap: methods x datasets (ordered by dim), cell = mean search time."""
    sub = df[df["model_type"] == model_type].copy()

    agg = (
        sub.groupby(["method_display", "dataset", "n_features"])["cf_search_time"]
        .mean()
        .reset_index()
    )

    # Sort datasets by number of continuous features (primary), total (secondary)
    dataset_order = sorted(
        DATASET_DIM.keys(), key=lambda d: (DATASET_CONTINUOUS[d], DATASET_DIM[d])
    )
    dataset_labels = [
        f"{d}\ncont={DATASET_CONTINUOUS[d]}  cat={DATASET_CATEGORICAL[d]}  total={DATASET_DIM[d]}"
        for d in dataset_order
    ]

    pivot = agg.pivot_table(index="method_display", columns="dataset", values="cf_search_time")
    pivot = pivot.reindex(columns=dataset_order)

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(
        np.log10(pivot.values + 1e-6),
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells with actual values
    for i in range(len(pivot.index)):
        for j in range(len(dataset_order)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                text = f"{val:.1f}" if val >= 1 else f"{val:.2f}"
                color = "white" if np.log10(val + 1e-6) > 1.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log₁₀(CF Search Time in s)", fontsize=11)

    ax.set_title(
        f"CF Search Time Heatmap — Methods × Datasets ({model_type})\n"
        "Sorted by continuous features; labels show cont / cat / total feature counts",
        fontsize=13,
    )
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"scaling_heatmap_{model_type}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.info("Saved heatmap to %s", out_path)
    plt.close(fig)


def generate_markdown_table(df: pd.DataFrame, model_type: str = "MLPClassifier") -> None:
    """Generate a markdown table: mean +/- std search time, methods as rows, datasets as cols."""
    sub = df[df["model_type"] == model_type].copy()

    agg = (
        sub.groupby(["method_display", "dataset", "n_features"])["cf_search_time"]
        .agg(["mean", "std"])
        .reset_index()
    )

    dataset_order = sorted(
        DATASET_DIM.keys(), key=lambda d: (DATASET_CONTINUOUS[d], DATASET_DIM[d])
    )
    methods_order = sorted(agg["method_display"].unique())

    def _fmt_val(m: float, s: float) -> str:
        if pd.isna(s) or s == 0:
            return f"{m:.2f}"
        return f"{m:.2f} +/- {s:.2f}"

    # Build compact column headers: dataset name + feature breakdown
    col_headers: list[str] = []
    for d in dataset_order:
        info = DATASET_INFO[d]
        name = d.replace("_", " ")
        col_headers.append(
            f"{name} (c={info.n_continuous}, cat={info.n_categorical}, "
            f"t={info.n_total_encoded}, n={info.n_samples:,})"
        )

    lines: list[str] = []
    lines.append(f"# CF search time (seconds) vs. dataset dimensionality ({model_type})")
    lines.append("")
    lines.append(
        "Columns: c = continuous features, cat = categorical features (raw), "
        "t = total features (after one-hot), n = samples"
    )
    lines.append("")

    # Header + separator (must be rows 1-2 for valid markdown table)
    lines.append("| Method | " + " | ".join(col_headers) + " |")
    lines.append("| :--- | " + " | ".join(["---:" for _ in dataset_order]) + " |")

    # Data rows
    for method in methods_order:
        row_vals: list[str] = [method]
        for dataset in dataset_order:
            cell = agg[(agg["method_display"] == method) & (agg["dataset"] == dataset)]
            if cell.empty:
                row_vals.append("--")
            else:
                row_vals.append(_fmt_val(cell["mean"].values[0], cell["std"].values[0]))
        lines.append("| " + " | ".join(row_vals) + " |")

    TABLES_DIR.mkdir(exist_ok=True)
    out_path = TABLES_DIR / f"scaling_analysis_{model_type}.md"
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved markdown table to %s", out_path)


def generate_summary_table(df: pd.DataFrame, model_type: str = "MLPClassifier") -> None:
    """Print a summary markdown table to stdout for quick inspection."""
    sub = df[df["model_type"] == model_type].copy()

    agg = (
        sub.groupby(["method_display", "dataset", "n_features", "n_continuous"])["cf_search_time"]
        .agg(["mean", "std"])
        .reset_index()
    )

    dataset_order = sorted(
        DATASET_DIM.keys(), key=lambda d: (DATASET_CONTINUOUS[d], DATASET_DIM[d])
    )
    methods_order = sorted(agg["method_display"].unique())

    header = (
        "| Method | "
        + " | ".join(
            [
                f"{d} (cont={DATASET_CONTINUOUS[d]}, cat={DATASET_CATEGORICAL[d]}, "
                f"tot={DATASET_DIM[d]})"
                for d in dataset_order
            ]
        )
        + " |"
    )
    sep = "|---" * (len(dataset_order) + 1) + "|"
    print(header)
    print(sep)

    for method in methods_order:
        row = f"| {method} "
        for dataset in dataset_order:
            cell = agg[(agg["method_display"] == method) & (agg["dataset"] == dataset)]
            if cell.empty:
                row += "| -- "
            else:
                m, s = cell["mean"].values[0], cell["std"].values[0]
                if pd.isna(s):
                    row += f"| {m:.2f} "
                else:
                    row += f"| {m:.2f}±{s:.2f} "
        row += "|"
        print(row)


def main() -> None:
    records = collect_timing_data()
    if not records:
        logger.error("No timing records found in %s", MODELS_DIR)
        return

    df = build_dataframe(records)

    # Dataset composition chart (model-independent)
    plot_dataset_composition()

    for model_type in ["MLPClassifier", "MultinomialLogisticRegression"]:
        model_data = df[df["model_type"] == model_type]
        if model_data.empty:
            logger.warning("No data for model_type=%s, skipping", model_type)
            continue

        logger.info("=== %s ===", model_type)
        plot_scaling_curves(df, model_type)
        plot_heatmap(df, model_type)
        generate_markdown_table(df, model_type)
        print(f"\n--- Summary table ({model_type}) ---")
        generate_summary_table(df, model_type)


if __name__ == "__main__":
    main()
