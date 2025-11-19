"""Script to calculate and display counterfactual metrics across multiple configurations.

This script loads metric files from different dataset/method/model combinations,
calculates mean and standard deviation across folds, and outputs formatted
markdown tables both to console and to a file.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


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
    models_root: str = "../models",
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
    root = Path(models_root) / dataset / method
    metrics = []

    for i in range(num_folds):
        path = root / f"fold_{i}" / f"cf_metrics_{model_name}.csv"
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        df = pd.read_csv(path)
        metrics.append(df)

    if not metrics:
        print(f"No metrics found for {dataset}/{method}/{model_name}")
        return pd.DataFrame()

    merged_df = pd.concat(metrics, axis=0, ignore_index=True)
    mean_ = merged_df.mean(axis=0)
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
    models_root: str = "../models",
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
        return "No data available"

    # Select only the requested metrics that exist in the dataframe
    available_metrics = [m for m in used_metrics if m in df.columns]
    if not available_metrics:
        return "No requested metrics found in data"

    df_filtered = df[available_metrics]
    return dataframe_to_markdown(df_filtered)


def main() -> None:
    """Main function to calculate and print metrics for configured combinations."""
    # Configuration: List of (dataset, method, model_name) tuples
    configurations = [
        ("lending_club", "DiceExplainerWrapper", "MLPClassifier"),
        ("lending_club", "DiceExplainerWrapper", "MultinomialLogisticRegression"),
        ("give_me_some_credit", "DiceExplainerWrapper", "MLPClassifier"),
        (
            "give_me_some_credit",
            "DiceExplainerWrapper",
            "MultinomialLogisticRegression",
        ),
        ("bank_marketing", "DiceExplainerWrapper", "MLPClassifier"),
        ("bank_marketing", "DiceExplainerWrapper", "MultinomialLogisticRegression"),
        ("credit_default", "DiceExplainerWrapper", "MLPClassifier"),
        ("credit_default", "DiceExplainerWrapper", "MultinomialLogisticRegression"),
        ("adult_census", "DiceExplainerWrapper", "MLPClassifier"),
        ("adult_census", "DiceExplainerWrapper", "MultinomialLogisticRegression"),
    ]

    # Metrics to include in the output tables
    used_metrics = [
        "validity",
        "prob_plausibility",
        "lof_scores_cf",
        "isolation_forest_scores_cf",
        "log_density_cf",
        "proximity_continuous_manhattan",
        "proximity_continuous_euclidean",
        "cf_search_time",
    ]

    # Number of folds
    num_folds = 5

    # Root directory for models (relative to this script)
    models_root = "../models"

    # Output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).parent / f"metrics_summary_{timestamp}.md"

    # Collect all output
    output_lines = []
    output_lines.append("# Counterfactual Metrics Summary\n")
    output_lines.append(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    # Calculate and print metrics for each configuration
    for dataset, method, model_name in configurations:
        section_header = f"\n## {dataset} - {method} - {model_name}\n"
        print(section_header)
        output_lines.append(section_header + "\n")

        table = calculate_metrics_table(
            dataset=dataset,
            method=method,
            model_name=model_name,
            used_metrics=used_metrics,
            num_folds=num_folds,
            models_root=models_root,
        )
        print(table)
        print()
        output_lines.append(table + "\n\n")

    # Write to file
    output_content = "".join(output_lines)
    output_file.write_text(output_content, encoding="utf-8")
    print(f"\n✓ Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()
