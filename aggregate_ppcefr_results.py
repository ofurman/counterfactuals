#!/usr/bin/env python3
"""
Script to aggregate PPCEFR, WACH, and CEARM results across folds and generate LaTeX tables
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Define datasets and models
DATASETS = ['toy_regression', 'concrete', 'diabetes', 'yacht', 'scm20d']
DATASET_DISPLAY_NAMES = {
    'toy_regression': 'Synthetic',
    'concrete': 'Concrete',
    'diabetes': 'Diabetes',
    'yacht': 'Yacht',
    'scm20d': 'SCM20D'
}
MODELS = ['LinearRegression', 'MLPRegressor']
METHODS = ['CEARM', 'WACH_REGR', 'PPCEFR']
METHOD_DISPLAY_NAMES = {
    'CEARM': 'CEARM',
    'WACH_REGR': 'WACH',
    'PPCEFR': '\\our{}'
}
BASE_PATH = Path('/Users/oleksiifurman/Developer/counterfactuals/models')

# Metric column mapping: CSV column -> LaTeX display
METRICS = {
    'target_achievement': 'MAE',
    'prob_plausibility': 'PP',
    'lof_scores_cf': 'LOF',
    'isolation_forest_scores_cf': 'IF',
    'log_density_cf': 'LD',
    'proximity_continuous_manhattan': 'L1',
    'proximity_continuous_euclidean': 'L2',
    'cf_search_time': 'Time'
}

def load_fold_data(dataset, method, model, fold):
    """Load metrics for a specific fold"""
    csv_path = BASE_PATH / dataset / method / f'fold_{fold}' / f'cf_metrics_{model}.csv'

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        # The second row contains the actual data (first row is header)
        if len(df) > 0:
            return df.iloc[0]
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    return None

def aggregate_results(dataset, method, model):
    """Aggregate results across all folds for a dataset, method, and model"""
    fold_data = []

    for fold in range(5):
        data = load_fold_data(dataset, method, model, fold)
        if data is not None:
            fold_data.append(data)

    if not fold_data:
        return None

    # Calculate mean and std for each metric
    results = {}
    results['n_folds'] = len(fold_data)

    for csv_col, latex_col in METRICS.items():
        if csv_col in fold_data[0]:
            values = [float(d[csv_col]) for d in fold_data if pd.notna(d[csv_col])]
            if values:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0
                results[latex_col] = (mean, std)

    return results

def format_value(mean, std, metric, show_std=True):
    """Format value as mean ± std with appropriate precision"""
    # Use 2 decimal places for consistency with user's example table
    decimals = 2

    # Always show std in format: 0.31$\pm$0.02
    if show_std:
        return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"
    else:
        return f"{mean:.{decimals}f}"

def find_best_values(all_method_results, metric):
    """Find the best value(s) for a metric across all methods"""
    # Determine if higher or lower is better
    lower_is_better = metric in ['MAE', 'LOF', 'L1', 'L2', 'Time']

    values = []
    for method in METHODS:
        if method in all_method_results and all_method_results[method] and metric in all_method_results[method]:
            mean, std = all_method_results[method][metric]
            values.append((mean, method))

    if not values:
        return set()

    if lower_is_better:
        best_value = min(values, key=lambda x: x[0])[0]
        # Consider values within 0.01 (or 1% for large values) as tied
        threshold = max(0.01, abs(best_value) * 0.01) if best_value != 0 else 0.01
        best_methods = {method for value, method in values if abs(value - best_value) <= threshold}
    else:
        best_value = max(values, key=lambda x: x[0])[0]
        threshold = max(0.01, abs(best_value) * 0.01) if best_value != 0 else 0.01
        best_methods = {method for value, method in values if abs(value - best_value) <= threshold}

    return best_methods

def generate_latex_table(model_name):
    """Generate LaTeX table for a specific model comparing all methods"""
    if model_name == 'LinearRegression':
        model_display = 'Linear Regression (LR)'
        label = 'lr'
    else:
        model_display = 'Deep Neural Network Regression (DNN)'
        label = 'dnn'

    # Collect all results: {dataset: {method: results}}
    all_results = {}
    for dataset in DATASETS:
        dataset_results = {}
        for method in METHODS:
            results = aggregate_results(dataset, method, model_name)
            if results:
                dataset_results[method] = results
        if dataset_results:
            all_results[dataset] = dataset_results

    if not all_results:
        print(f"No results found for {model_name}")
        return

    # Generate LaTeX table
    print(f"\n\\begin{{table*}}[th]")
    print("\\centering")
    print(f"\\caption{{Comparative Results of \\our{{}} for {model_display}. "
          f"\\our{{}} method performance is contrasted with Wachter et al. and CEARM. "
          f"The results demonstrate \\our{{}} consistently valid and probabilistically plausible results and its ability to produce counterfactuals even in complex scenarios like multi-target regression.}}")
    print(f"\\label{{tab:regression_ours_vs_all_{label}}}")
    print("\\begin{center}")
    print("\\begin{sc}")
    print("\\begin{tiny}")
    print("\\begin{tabular}{llrrrrrrrr}")
    print("\\toprule")
    print("Dataset & Method & MAE $\\downarrow$ & PP $\\uparrow$ & LOF $\\downarrow$ & IF $\\uparrow$ & LD $\\uparrow$ & L1 $\\downarrow$ & L2 $\\downarrow$ & Time $\\downarrow$\\\\")
    print("\\midrule")

    for dataset_idx, dataset in enumerate(DATASETS):
        if dataset not in all_results:
            continue

        display_name = DATASET_DISPLAY_NAMES[dataset]
        dataset_results = all_results[dataset]

        # Count how many methods have results for this dataset
        method_count = len(dataset_results)

        # Track if we've added the dataset name yet
        first_method_for_dataset = True

        for method_idx, method in enumerate(METHODS):
            if method not in dataset_results:
                continue

            results = dataset_results[method]
            method_display = METHOD_DISPLAY_NAMES[method]

            # Format each metric
            row_parts = []

            # First column: dataset name (only for first method with results)
            if first_method_for_dataset:
                row_parts.append(f"\\multirow[t]{{{method_count}}}{{*}}{{{display_name}}}")
                first_method_for_dataset = False
            else:
                row_parts.append("")

            # Second column: method name
            row_parts.append(method_display)

            # Get best values for this dataset across all methods
            metric_list = ['MAE', 'PP', 'LOF', 'IF', 'LD', 'L1', 'L2', 'Time']
            for metric in metric_list:
                if metric in results:
                    mean, std = results[metric]
                    value_str = format_value(mean, std, metric, show_std=True)

                    # Check if this is the best value
                    best_methods = find_best_values(dataset_results, metric)
                    if method in best_methods:
                        value_str = f"\\textbf{{{value_str}}}"

                    row_parts.append(value_str)
                else:
                    row_parts.append('-')

            print(' & '.join(row_parts) + ' \\\\')

        # Add midrule after each dataset except the last
        if dataset_idx < len([d for d in DATASETS if d in all_results]) - 1:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{tiny}")
    print("\\end{sc}")
    print("\\end{center}")
    print("\\end{table*}")

def main():
    print("="*80)
    print("PPCEFR, WACH, and CEARM Results Aggregation")
    print("="*80)

    # Generate table for Linear Regression
    print("\n" + "="*80)
    print("LINEAR REGRESSION TABLE")
    print("="*80)
    generate_latex_table('LinearRegression')

    # Generate table for MLP Regressor
    print("\n" + "="*80)
    print("MLP REGRESSOR TABLE")
    print("="*80)
    generate_latex_table('MLPRegressor')

    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == '__main__':
    main()
