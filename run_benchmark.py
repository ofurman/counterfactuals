"""Run benchmark across multiple CF methods and collect results.

This script runs multiple counterfactual pipelines with configurable parameters
and aggregates metrics into markdown tables.

Usage:
    # Run all methods with default (fast) settings
    uv run python run_benchmark.py

    # Run specific methods only
    uv run python run_benchmark.py --methods PPCEF DiCE WACH

    # Custom epochs for testing
    uv run python run_benchmark.py --disc-epochs 10 --gen-epochs 10 --cf-epochs 100

    # Full benchmark
    uv run python run_benchmark.py --disc-epochs 5000 --gen-epochs 2000 --cf-epochs 20000

    # Specific dataset
    uv run python run_benchmark.py --dataset config/datasets/moons.yaml
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Available pipelines (16 working methods)
AVAILABLE_METHODS = {
    # Local methods
    "PPCEF": "counterfactuals.pipelines.run_ppcef_pipeline",
    "DiCE": "counterfactuals.pipelines.run_dice_pipeline",
    "WACH": "counterfactuals.pipelines.run_wach_pipeline",
    "CCHVAE": "counterfactuals.pipelines.run_cchvae_pipeline",
    "DiCoFlex": "counterfactuals.pipelines.run_dicoflex_pipeline",
    "Artelt": "counterfactuals.pipelines.run_artelt_pipeline",
    "CET": "counterfactuals.pipelines.run_cet_pipeline",
    "CEM": "counterfactuals.pipelines.run_cem_pipeline",
    "CEGP": "counterfactuals.pipelines.run_cegp_pipeline",
    "CaseBasedSACE": "counterfactuals.pipelines.run_casebased_sace_pipeline",
    "WACHOURS": "counterfactuals.pipelines.run_wach_ours_pipeline",
    # Global methods
    "AReS": "counterfactuals.pipelines.run_ares_pipeline",
    "GLOBECE": "counterfactuals.pipelines.run_globe_ce_pipeline",
    # Group methods
    "GLANCE": "counterfactuals.pipelines.run_glance_pipeline",
    "RPPCEF": "counterfactuals.pipelines.run_rppcef_pipeline",
    "PPCE FR": "counterfactuals.pipelines.run_ppcefr_pipeline",
}


def run_pipeline(
    method_name: str,
    module_path: str,
    disc_epochs: int,
    gen_epochs: int,
    cf_epochs: int,
) -> tuple[bool, str]:
    """
    Run a single pipeline with specified parameters.

    Args:
        method_name: Name of the method
        module_path: Python module path to pipeline
        disc_epochs: Number of discriminator training epochs
        gen_epochs: Number of generator training epochs
        cf_epochs: Number of counterfactual search epochs

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Use ++key=value to force override in Hydra (adds or replaces)
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        module_path.replace(".py", ""),
        f"++disc_model.epochs={disc_epochs}",
        f"++gen_model.epochs={gen_epochs}",
        f"++counterfactuals_params.epochs={cf_epochs}",
    ]

    print(f"\n{'='*60}")
    print(f"Running {method_name}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per method
        )

        if result.returncode == 0:
            return True, "Success"
        else:
            error_lines = result.stderr.split("\n")[-10:]  # Last 10 lines
            error_msg = "\n".join(error_lines)
            return False, f"Failed with return code {result.returncode}: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Timeout (>1 hour)"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def collect_metrics(
    methods: List[str], output_folder: Path = Path("models")
) -> pd.DataFrame:
    """
    Collect metrics from all methods into a single DataFrame.

    Args:
        methods: List of method names to collect metrics for
        output_folder: Root folder containing results

    Returns:
        DataFrame with aggregated metrics across all methods
    """
    all_metrics = []

    if not output_folder.exists():
        print(f"Warning: Output folder {output_folder} does not exist")
        return pd.DataFrame()

    # Iterate over all dataset folders in output_folder
    for dataset_folder in output_folder.iterdir():
        if not dataset_folder.is_dir():
            continue

        dataset_name = dataset_folder.name

        # Get all method folders in dataset directory (case-insensitive match)
        available_folders = {f.name.lower(): f for f in dataset_folder.iterdir() if f.is_dir() and not f.name.startswith("fold_")}

        for method in methods:
            # Try to find matching folder (case-insensitive)
            method_lower = method.lower().replace(" ", "")

            method_folder = None
            for folder_name_lower, folder_path in available_folders.items():
                if method_lower in folder_name_lower or folder_name_lower in method_lower:
                    method_folder = folder_path
                    break

            if not method_folder:
                continue

            # Find all metric CSV files (across folds)
            metric_files = list(method_folder.glob("fold_*/cf_metrics_*.csv"))

            if not metric_files:
                continue

            print(f"Found results for {method} on {dataset_name} at {method_folder}")

            # Aggregate across folds
            fold_metrics = []
            for metric_file in metric_files:
                try:
                    df = pd.read_csv(metric_file)
                    fold_metrics.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {metric_file}: {e}")

            if fold_metrics:
                # Calculate mean across folds
                combined = pd.concat(fold_metrics, axis=0, ignore_index=True)
                mean_metrics = combined.mean(axis=0).to_dict()
                std_metrics = combined.std(axis=0).to_dict()

                # Format as "mean ± std"
                formatted_metrics = {
                    "method": method,
                    "dataset": dataset_name,
                    **{
                        col: f"{mean_metrics[col]:.3f} ± {std_metrics[col]:.3f}"
                        for col in combined.columns
                    },
                }
                all_metrics.append(formatted_metrics)

    return pd.DataFrame(all_metrics)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    if df.empty:
        return "No metrics available\n"

    headers = list(df.columns)
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return markdown


def main():
    parser = argparse.ArgumentParser(
        description="Run counterfactual method benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(AVAILABLE_METHODS.keys()) + ["all"],
        default=["all"],
        help="Methods to run (default: all)",
    )

    parser.add_argument(
        "--disc-epochs",
        type=int,
        default=10,
        help="Discriminator training epochs (default: 10 for quick test)",
    )

    parser.add_argument(
        "--gen-epochs",
        type=int,
        default=10,
        help="Generator training epochs (default: 10 for quick test)",
    )

    parser.add_argument(
        "--cf-epochs",
        type=int,
        default=10,
        help="Counterfactual search epochs (default: 10 for quick test)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: benchmark_results_<timestamp>.md)",
    )

    args = parser.parse_args()

    # Resolve methods to run
    if "all" in args.methods:
        methods_to_run = list(AVAILABLE_METHODS.keys())
    else:
        methods_to_run = args.methods

    print(f"\n{'='*60}")
    print(f"COUNTERFACTUAL METHODS BENCHMARK")
    print(f"{'='*60}")
    print(f"Methods: {', '.join(methods_to_run)}")
    print(f"Epochs: disc={args.disc_epochs}, gen={args.gen_epochs}, cf={args.cf_epochs}")
    print(f"{'='*60}\n")

    # Run all pipelines
    results = {}
    for method in methods_to_run:
        module_path = AVAILABLE_METHODS[method]
        success, message = run_pipeline(
            method,
            module_path,
            args.disc_epochs,
            args.gen_epochs,
            args.cf_epochs,
        )
        results[method] = {"success": success, "message": message}

        status = "✓" if success else "✗"
        print(f"\n{method}: {status} - {message}")

    # Collect metrics
    print(f"\n{'='*60}")
    print("Collecting metrics...")
    print(f"{'='*60}\n")

    metrics_df = collect_metrics(methods_to_run)

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"benchmark_results_{timestamp}.md"

    report_lines = [
        "# Counterfactual Methods Benchmark Results\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Config:** disc_epochs={args.disc_epochs}, gen_epochs={args.gen_epochs}, "
        f"cf_epochs={args.cf_epochs}\n\n",
        "## Execution Summary\n\n",
        "| Method | Status | Message |\n",
        "|--------|--------|---------|",
    ]

    for method, result in results.items():
        status = "✓" if result["success"] else "✗"
        # Truncate long error messages
        msg = result["message"]
        if len(msg) > 100:
            msg = msg[:97] + "..."
        report_lines.append(f"| {method} | {status} | {msg} |\n")

    successful_methods = [m for m, r in results.items() if r["success"]]
    report_lines.append(
        f"\n**Summary:** {len(successful_methods)}/{len(methods_to_run)} methods completed successfully\n\n"
    )

    if not metrics_df.empty:
        report_lines.append("## Metrics\n\n")
        report_lines.append(dataframe_to_markdown(metrics_df))
    else:
        report_lines.append("## Metrics\n\nNo metrics collected.\n")

    # Write report
    Path(output_file).write_text("".join(report_lines))
    print(f"\n{'='*60}")
    print(f"✓ Benchmark complete!")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*60}\n")

    # Print summary to console
    print("".join(report_lines))


if __name__ == "__main__":
    main()
