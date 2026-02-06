"""Script to calculate counterfactual metrics and save a markdown table."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a markdown table string.

    Args:
        df: DataFrame to convert.

    Returns:
        Markdown table as a string.
    """
    headers = list(df.columns)
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    return markdown


def format_mean_std(mean: float, std: float) -> str:
    """Format mean and standard deviation as a string.

    Args:
        mean: Mean value.
        std: Standard deviation value.

    Returns:
        Formatted string in the form "mean ± std".
    """
    return f"{mean:.2f} ± {std:.2f}"


def load_and_aggregate_metrics(
    dataset: str,
    method: str,
    model_name: str,
    num_folds: int = 5,
    models_root: Path = Path("models"),
) -> pd.DataFrame:
    """Load metrics from all folds and calculate mean/std.

    Args:
        dataset: Name of the dataset.
        method: Name of the method.
        model_name: Name of the model.
        num_folds: Number of folds to aggregate.
        models_root: Root directory for model outputs.

    Returns:
        DataFrame with aggregated metrics (mean ± std format).
    """
    root = models_root / dataset / method
    metrics: list[pd.DataFrame] = []

    for i in range(num_folds):
        path = root / f"fold_{i}" / f"cf_metrics_{model_name}.csv"
        if not path.exists():
            logging.warning("Metrics file missing: %s", path)
            continue
        df = pd.read_csv(path)
        metrics.append(df)

    if not metrics:
        logging.warning(
            "No metrics found for dataset=%s method=%s model=%s",
            dataset,
            method,
            model_name,
        )
        return pd.DataFrame()

    merged_df = pd.concat(metrics, axis=0, ignore_index=True)
    mean_ = merged_df.mean(axis=0)
    if "number_of_instances" in merged_df.columns:
        weights = merged_df["number_of_instances"]
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_means: dict[str, float] = {}
            for col in merged_df.columns:
                # if col == "cf_search_time":
                #     weighted_means[col] = merged_df[col].mean()
                if col == "number_of_instances":
                    weighted_means[col] = float(total_weight)
                else:
                    weighted_means[col] = float(
                        (merged_df[col] * weights).sum() / total_weight
                    )
            mean_ = pd.Series(weighted_means)
    std_ = merged_df.std(axis=0)

    formatted = pd.DataFrame(
        {col: [format_mean_std(mean_[col], std_[col])] for col in merged_df.columns}
    ).T

    return formatted.T


def calculate_metrics_table(
    dataset: str,
    method: str,
    model_name: str,
    used_metrics: list[str],
    num_folds: int = 5,
    models_root: Path = Path("models"),
) -> str:
    """Calculate metrics and return as markdown table.

    Args:
        dataset: Name of the dataset.
        method: Name of the method.
        model_name: Name of the model.
        used_metrics: List of metric names to include in the table.
        num_folds: Number of folds to aggregate.
        models_root: Root directory for model outputs.

    Returns:
        Markdown table string with the metrics.
    """
    df = load_and_aggregate_metrics(dataset, method, model_name, num_folds, models_root)

    if df.empty:
        raise ValueError("No data available for the requested configuration.")

    # Select only the requested metrics that exist in the dataframe
    missing_metrics = [metric for metric in used_metrics if metric not in df.columns]
    if missing_metrics:
        logging.warning("Missing metrics in data: %s", ", ".join(missing_metrics))

    available_metrics = [m for m in used_metrics if m in df.columns]
    if not available_metrics:
        raise ValueError("No requested metrics found in data.")

    df_filtered = df[available_metrics]
    return dataframe_to_markdown(df_filtered)


def _build_table_name(
    dataset: str, method: str, model_name: str, table_name: str | None
) -> str:
    """Build a default markdown filename from the inputs."""
    if table_name:
        return table_name if table_name.endswith(".md") else f"{table_name}.md"
    return f"{dataset}_{method}_{model_name}_metrics.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate aggregated counterfactual metrics and save a markdown table."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--method", required=True, help="Method name.")
    parser.add_argument(
        "--model-name", required=True, help="Discriminative model name."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the markdown table will be saved.",
    )
    parser.add_argument(
        "--table-name",
        help="Optional output filename (without path). Defaults to dataset/method/model.",
    )
    parser.add_argument(
        "--models-root",
        default="models",
        help="Root directory containing model outputs (default: models).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds to aggregate (default: 5).",
    )
    parser.add_argument(
        "--metrics-conf-path",
        default="counterfactuals/pipelines/conf/metrics/default.yaml",
        help="Path to metrics config (default: counterfactuals/pipelines/conf/metrics/default.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Calculate metrics for a single configuration and save a markdown table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()

    try:
        metrics_conf = OmegaConf.load(args.metrics_conf_path)
        used_metrics = list(metrics_conf.metrics_to_compute)
    except Exception as exc:  # noqa: BLE001 - report config issues clearly
        logging.error("Failed to load metrics config: %s", exc)
        raise SystemExit(1) from exc

    try:
        table = calculate_metrics_table(
            dataset=args.dataset,
            method=args.method,
            model_name=args.model_name,
            used_metrics=used_metrics,
            num_folds=args.num_folds,
            models_root=Path(args.models_root),
        )
    except ValueError as exc:
        logging.error("Failed to calculate metrics: %s", exc)
        raise SystemExit(1) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / _build_table_name(
        args.dataset,
        args.method,
        args.model_name,
        args.table_name,
    )
    output_file.write_text(table + "\n", encoding="utf-8")
    logging.info("Saved metrics table to %s", output_file)


if __name__ == "__main__":
    main()
