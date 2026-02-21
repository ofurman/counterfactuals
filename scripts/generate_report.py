
import sys
import pandas as pd
from pathlib import Path
import yaml
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from counterfactuals.datasets.file_dataset import FileDataset

# Dataset name -> config file name
DATASET_MAP = {
    "Adult Census": "adult_census.yaml",
    "Audit": "audit.yaml",
    "BankMarketing": "bank_marketing.yaml",
    "Blobs": "blobs.yaml",
    "Credit Default": "credit_default.yaml",
    "Digits": "digits.yaml",
    "GermanCredit": "german_credit.yaml",
    "Give Me Some Credit": "give_me_some_credit.yaml",
    "Heloc": "heloc.yaml",
    "Law": "law.yaml",
    "Lending Club": "lending_club.yaml",
    "Moons": "moons.yaml",
    "Wine": "wine.yaml"
}

print(f"{'Dataset':<20} | {'Rows':<10} | {'Cat':<5} | {'Num':<5} | {'Label Dist':<40}")
print("-" * 100)

for name, config_file in DATASET_MAP.items():
    config_path = project_root / "config" / "datasets" / config_file
    try:
        # Read YAML for original feature counts
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
            
        raw_cat = len(raw_cfg.get("categorical_features", []) or [])
        raw_num = len(raw_cfg.get("continuous_features", []) or [])

        # Load dataset for rows and distribution
        dataset = FileDataset(config_path)
        
        # Rows
        n_rows = dataset.X.shape[0]
        
        # Label distribution
        y = pd.Series(dataset.y)
        counts = y.value_counts(normalize=True).sort_index()
        
        dist_str_parts = []
        for cls_val, freq in counts.items():
            dist_str_parts.append(f"{cls_val}: {freq:.1%}")
        dist_str = ", ".join(dist_str_parts)
            
        print(f"{name:<20} | {n_rows:<10} | {raw_cat:<5} | {raw_num:<5} | {dist_str:<40}")
        
    except Exception as e:
        # print(f"{name:<20} | Error: {e}")
        # Retrying with error message if simple load fails
        print(f"{name:<20} | Error: {str(e).splitlines()[0]}")
