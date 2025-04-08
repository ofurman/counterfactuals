#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from glob import glob
import re

# Define the datasets and models
datasets = ["BlobsDataset", "MoonsDataset", "LawDataset", "HelocDataset", "WineDataset", "DigitsDataset"]
models = {
    "MultilayerPerceptron": "MLP", 
    "MultinomialLogisticRegression": "MLR"
}

# Define a function to get dataset display name
def get_display_name(dataset_name):
    display_names = {
        "BlobsDataset": "Blobs",
        "MoonsDataset": "Moons",
        "LawDataset": "Law",
        "HelocDataset": "HELOC",
        "WineDataset": "Wine",
        "DigitsDataset": "Digits"
    }
    return display_names.get(dataset_name, dataset_name)

# Initialize lists to store results for each model
results = {model_key: [] for model_key in models.keys()}

# Process each dataset
for dataset in datasets:
    method = "WACH_OURS"  # Only interested in WACH_OURS
    base_path = f"./models/{dataset}/{method}"
    
    if not os.path.exists(base_path):
        print(f"Skipping {method} for {dataset} - path not found: {base_path}")
        continue
        
    # Find all fold directories
    fold_dirs = glob(os.path.join(base_path, "fold_*"))
    if not fold_dirs:
        print(f"No fold directories found for {method} on {dataset}")
        continue
        
    # Process each fold
    for fold_dir in fold_dirs:
        fold_num = int(re.search(r"fold_(\d+)", fold_dir).group(1))
        
        # Get the metrics file
        metrics_file = os.path.join(fold_dir, "cf_metrics.csv")
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found: {metrics_file}")
            continue
            
        # Read the metrics
        try:
            metrics_df = pd.read_csv(metrics_file)
            
            # For each model, check if it has results
            for model_key, model_display in models.items():
                # Look for model-specific counterfactuals file to confirm this model was used
                cf_file = os.path.join(fold_dir, f"counterfactuals_no_plaus_{method}_{model_key}.csv")
                if not os.path.exists(cf_file):
                    continue
                    
                # Extract the relevant metrics, with safeguards for missing columns
                try:
                    result = {
                        "Dataset": get_display_name(dataset),
                        "Fold": fold_num,
                        "Valid.↑": metrics_df["validity"].values[0] if "validity" in metrics_df.columns else np.nan,
                        "L2↓": metrics_df["proximity_continuous_euclidean"].values[0] if "proximity_continuous_euclidean" in metrics_df.columns else np.nan,
                        "Prob.Plaus.↑": metrics_df["prob_plausibility"].values[0] if "prob_plausibility" in metrics_df.columns else np.nan,
                        "LogDens.↑": metrics_df["log_density_cf"].values[0] if "log_density_cf" in metrics_df.columns else np.nan,
                        "IsoForest↑": metrics_df["isolation_forest_scores_cf"].values[0] if "isolation_forest_scores_cf" in metrics_df.columns else np.nan,
                        "LOF↓": metrics_df["lof_scores_cf"].values[0] if "lof_scores_cf" in metrics_df.columns else np.nan,
                    }
                    
                    results[model_key].append(result)
                except Exception as e:
                    print(f"Error extracting metrics for {dataset}/{method}/{model_key} fold {fold_num}: {str(e)}")
        except Exception as e:
            print(f"Error processing {metrics_file}: {str(e)}")

# Function to format mean and std in a clean way
def format_mean_std(mean, std):
    # If mean is very large or very small, format it with scientific notation
    if abs(mean) > 100 or abs(mean) < 0.01:
        return f"{mean:.2e} ± {std:.2e}"
    
    # For normal sized numbers
    return f"{mean:.2f} ± {std:.2f}"

# Process results for each model
for model_key, model_display in models.items():
    # Convert to DataFrame
    model_results_df = pd.DataFrame(results[model_key])
    
    if model_results_df.empty:
        print(f"No results found for {model_display} model")
        continue
    
    print(f"\nProcessing results for {model_display} model")
    
    # Define the metrics columns
    metric_cols = ["Valid.↑", "L2↓", "Prob.Plaus.↑", "LogDens.↑", "IsoForest↑", "LOF↓"]
    
    # Calculate mean and std for each dataset across folds
    mean_results = model_results_df.groupby("Dataset")[metric_cols].mean().reset_index()
    std_results = model_results_df.groupby("Dataset")[metric_cols].std().reset_index()
    
    # Fill NaN values in std with 0.0 (for cases with only one fold)
    std_results = std_results.fillna(0.0)
    
    # Sort datasets in specified order - ensuring HELOC and Law come first
    dataset_order = {"HELOC": 0, "Law": 1, "Blobs": 2, "Moons": 3, "Wine": 4, "Digits": 5}
    mean_results["Dataset_Order"] = mean_results["Dataset"].map(
        lambda x: dataset_order.get(x, 10)  # Any unknown datasets will come last
    )
    mean_results = mean_results.sort_values("Dataset_Order").drop(columns=["Dataset_Order"])
    
    # Create a combined table with mean ± std
    combined_results = mean_results.copy()
    for col in metric_cols:
        for i, row in combined_results.iterrows():
            dataset = row["Dataset"]
            mean_val = row[col]
            
            # Find the corresponding std value
            std_mask = std_results["Dataset"] == dataset
            if not std_mask.any():
                std_val = 0.0
            else:
                std_val = std_results.loc[std_mask, col].values[0]
            
            # Format based on the magnitude
            if abs(mean_val) > 100 or (abs(mean_val) < 0.01 and mean_val != 0):
                combined_results.at[i, col] = f"{mean_val:.2e} ± {std_val:.2e}"
            else:
                combined_results.at[i, col] = f"{mean_val:.2f} ± {std_val:.2f}"
    
    # Save results to CSV
    mean_results.to_csv(f"results_{model_display}_mean.csv", index=False)
    std_results.to_csv(f"results_{model_display}_std.csv", index=False)
    combined_results.to_csv(f"results_{model_display}_combined.csv", index=False)
    
    print(f"Results for {model_display} saved to CSV files")
    
    # Create a tab-delimited format
    with open(f"results_{model_display}_table.txt", "w") as f:
        # Write header
        f.write("Dataset\t" + "\t".join(metric_cols) + "\n")
        
        # Write data rows
        for _, row in combined_results.iterrows():
            line = f"{row['Dataset']}"
            for col in metric_cols:
                line += f"\t{row[col]}"
            f.write(line + "\n")
    
    print(f"Tab-delimited results for {model_display} saved to results_{model_display}_table.txt")
    
    # Print the table
    print(f"\n{model_display} Results (mean ± std):")
    print("Dataset\t" + "\t".join(metric_cols))
    for _, row in combined_results.iterrows():
        line = f"{row['Dataset']}"
        for col in metric_cols:
            line += f"\t{row[col]}"
        print(line) 