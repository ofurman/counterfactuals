#!/usr/bin/env python3
"""
Script to generate LaTeX tables with metrics from counterfactual explanation methods.
Based on the structure similar to other.tex and distances.tex files.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def format_mean_std(mean_val: float, std_val: float) -> str:
    """Format mean ± std values for LaTeX table."""
    return f"{mean_val:.2f}$\\pm${std_val:.2f}"


def load_metrics_for_dataset_method_model(
    dataset: str, method: str, model: str, models_root: Path
) -> Optional[pd.DataFrame]:
    """Load and aggregate metrics across all folds for a specific dataset/method/model combination."""
    metrics_list = []

    for fold in range(5):
        metrics_file = (
            models_root / dataset / method / f"fold_{fold}" / f"cf_metrics_{model}.csv"
        )
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            metrics_list.append(df)
        else:
            print(f"Warning: Missing file {metrics_file}")

    if not metrics_list:
        return None

    # Concatenate all folds and calculate mean/std
    merged_df = pd.concat(metrics_list, axis=0, ignore_index=True)
    return merged_df


def calculate_aggregated_metrics(merged_df: pd.DataFrame) -> Dict[str, str]:
    """Calculate mean ± std for all metrics."""
    if merged_df is None or merged_df.empty:
        return {}

    mean_vals = merged_df.mean(axis=0)
    std_vals = merged_df.std(axis=0)

    return {
        col: format_mean_std(mean_vals[col], std_vals[col]) for col in merged_df.columns
    }


def get_dataset_display_name(dataset_code: str) -> str:
    """Convert dataset code to display name used in tables."""
    mapping = {
        "LendingClubDataset": "Lending Club",
        "GiveMeSomeCreditDataset": "GMSC",
        "BankMarketingDataset": "Bank Marketing",
        "CreditDefaultDataset": "Credit Default",
        "AdultCensusDataset": "Adult Census",
    }
    return mapping.get(dataset_code, dataset_code)


def get_method_display_name(method_code: str) -> str:
    """Convert method code to display name used in tables."""
    mapping = {"PPCEF": "PPCEF", "CCHVAE": "C-CHVAE", "DiceExplainerWrapper": "DiCE"}
    return mapping.get(method_code, method_code)


def generate_main_metrics_table(
    datasets: List[str], methods: List[str], model: str, models_root: Path
) -> str:
    """Generate the main metrics table (like other.tex)."""

    # Define the metrics to include in main table
    main_metrics_mapping = {
        "coverage": "Cov. $\\uparrow$",
        "validity": "Val. $\\uparrow$",
        "prob_plausibility": "PP $\\uparrow$",
        "lof_scores_cf": "LOF $\\downarrow$",
        "isolation_forest_scores_cf": "IF $\\downarrow$",
        "log_density_cf": "LD $\\uparrow$",
    }

    latex_lines = [
        "",
        "\\begin{table}[th]",
        "    \\centering",
        f"    \\caption{{Performance Analysis Across Different Datasets - Main Metrics. Results demonstrate the effectiveness of PPCEF in achieving high coverage, validity, and probabilistic plausibility compared to baseline methods across various financial and demographic datasets. \\textbf{{{model.replace('Multilayer', 'ML').replace('Multinomial', 'MN')}}}}}",
        f"    \\label{{tab:ours_mix_{model.lower()}_main}}",
        "    \\begin{center}",
        "    \\begin{sc}",
        "    \\begin{tiny}",
        "    \\begin{tabular}{l|l|" + "r" * len(main_metrics_mapping) + "}",
        "    \\toprule",
        "    Dataset & Method & " + " & ".join(main_metrics_mapping.values()) + " \\\\",
        "    \\midrule",
    ]

    for i, dataset in enumerate(datasets):
        display_dataset = get_dataset_display_name(dataset)

        # Process first method
        first_row = f"    \\multirow{{{len(methods)}}}{{*}}{{{display_dataset}}} & {get_method_display_name(methods[0])} & "

        merged_df = load_metrics_for_dataset_method_model(
            dataset, methods[0], model, models_root
        )
        if merged_df is not None:
            metrics = calculate_aggregated_metrics(merged_df)
            values = []
            for metric_key in main_metrics_mapping.keys():
                if metric_key in metrics:
                    values.append(metrics[metric_key])
                else:
                    values.append("-- $\\pm$ --")
            first_row += " & ".join(values) + " \\\\"
        else:
            first_row += (
                " & ".join(["-- $\\pm$ --"] * len(main_metrics_mapping)) + " \\\\"
            )

        latex_lines.append(first_row)

        # Process remaining methods
        for method in methods[1:]:
            method_row = f"     & {get_method_display_name(method)} & "

            merged_df = load_metrics_for_dataset_method_model(
                dataset, method, model, models_root
            )
            if merged_df is not None:
                metrics = calculate_aggregated_metrics(merged_df)
                values = []
                for metric_key in main_metrics_mapping.keys():
                    if metric_key in metrics:
                        values.append(metrics[metric_key])
                    else:
                        values.append("-- $\\pm$ --")
                method_row += " & ".join(values) + " \\\\"
            else:
                method_row += (
                    " & ".join(["-- $\\pm$ --"] * len(main_metrics_mapping)) + " \\\\"
                )

            latex_lines.append(method_row)

        # Add midrule between datasets (except after last)
        if i < len(datasets) - 1:
            latex_lines.append("    \\midrule")

    latex_lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            "    \\end{tiny}",
            "    \\end{sc}",
            "    \\end{center}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex_lines)


def generate_distances_table(
    datasets: List[str], methods: List[str], model: str, models_root: Path
) -> str:
    """Generate the distances/proximity table (like distances.tex)."""

    # Define the metrics to include in distances table
    distance_metrics_mapping = {
        "proximity_categorical_hamming": "Hamming $\\downarrow$",
        "proximity_continuous_manhattan": "L1 $\\downarrow$",
        "proximity_continuous_euclidean": "L2 $\\downarrow$",
        "cf_search_time": "Time $\\downarrow$",
    }

    latex_lines = [
        "",
        "\\begin{table}[th]",
        "    \\centering",
        f"    \\caption{{Performance Analysis Across Different Datasets for Proximity and Efficiency Metrics. (\\textbf{{{model.replace('Multilayer', 'ML').replace('Multinomial', 'MN')}}})}}",
        f"    \\label{{tab:ours_mix_{model.lower()}_add}}",
        "    \\begin{center}",
        "    \\begin{sc}",
        "    \\begin{tiny}",
        "    \\begin{tabular}{l|l|" + "r" * len(distance_metrics_mapping) + "}",
        "    \\toprule",
        "    Dataset & Method & "
        + " & ".join(distance_metrics_mapping.values())
        + " \\\\",
        "    \\midrule",
    ]

    for i, dataset in enumerate(datasets):
        display_dataset = get_dataset_display_name(dataset)

        # Process first method
        first_row = f"    \\multirow{{{len(methods)}}}{{*}}{{{display_dataset}}} & {get_method_display_name(methods[0])} & "

        merged_df = load_metrics_for_dataset_method_model(
            dataset, methods[0], model, models_root
        )
        if merged_df is not None:
            metrics = calculate_aggregated_metrics(merged_df)
            values = []
            for metric_key in distance_metrics_mapping.keys():
                if metric_key in metrics:
                    values.append(metrics[metric_key])
                else:
                    values.append("-- $\\pm$ --")
            first_row += " & ".join(values) + " \\\\"
        else:
            first_row += (
                " & ".join(["-- $\\pm$ --"] * len(distance_metrics_mapping)) + " \\\\"
            )

        latex_lines.append(first_row)

        # Process remaining methods
        for method in methods[1:]:
            method_row = f"     & {get_method_display_name(method)} & "

            merged_df = load_metrics_for_dataset_method_model(
                dataset, method, model, models_root
            )
            if merged_df is not None:
                metrics = calculate_aggregated_metrics(merged_df)
                values = []
                for metric_key in distance_metrics_mapping.keys():
                    if metric_key in metrics:
                        values.append(metrics[metric_key])
                    else:
                        values.append("-- $\\pm$ --")
                method_row += " & ".join(values) + " \\\\"
            else:
                method_row += (
                    " & ".join(["-- $\\pm$ --"] * len(distance_metrics_mapping))
                    + " \\\\"
                )

            latex_lines.append(method_row)

        # Add midrule between datasets (except after last)
        if i < len(datasets) - 1:
            latex_lines.append("    \\midrule")

    latex_lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            "    \\end{tiny}",
            "    \\end{sc}",
            "    \\end{center}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from counterfactual metrics"
    )
    parser.add_argument(
        "--model",
        choices=["MultilayerPerceptron", "MultinomialLogisticRegression", "NODE"],
        default="MultilayerPerceptron",
        help="Model type to generate tables for",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models"),
        help="Root directory containing model results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for LaTeX files",
    )

    args = parser.parse_args()

    # Define datasets and methods based on the provided tables
    datasets = [
        "LendingClubDataset",
        "GiveMeSomeCreditDataset",
        "BankMarketingDataset",
        "CreditDefaultDataset",
        "AdultCensusDataset",
    ]

    methods = ["PPCEF", "CCHVAE", "DiceExplainerWrapper"]

    # Generate main metrics table
    main_table = generate_main_metrics_table(
        datasets, methods, args.model, args.models_root
    )

    # Generate distances table
    distances_table = generate_distances_table(
        datasets, methods, args.model, args.models_root
    )

    # Write output files
    model_suffix = (
        args.model.lower().replace("multilayer", "ml").replace("multinomial", "mn")
    )

    main_output_file = args.output_dir / f"main_metrics_{model_suffix}.tex"
    distances_output_file = args.output_dir / f"distances_{model_suffix}.tex"

    with open(main_output_file, "w") as f:
        f.write(main_table)

    with open(distances_output_file, "w") as f:
        f.write(distances_table)

    print("Generated tables:")
    print(f"  Main metrics: {main_output_file}")
    print(f"  Distances: {distances_output_file}")


if __name__ == "__main__":
    main()
