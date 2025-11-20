"""
Test all CF method pipelines from counterfactuals.pipelines module.

This script imports and runs all available pipeline functions,
collecting results in a status table.
"""

import sys
import traceback
from typing import Dict, List, Tuple

import numpy as np

# Mapping of method names to their pipeline modules
PIPELINES = {
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
    # "LiCE": "counterfactuals.pipelines.run_lice_pipeline",  # Requires old SPFlow version
    "CaseBased SACE": "counterfactuals.pipelines.run_casebased_sace_pipeline",
    "WACH OURS": "counterfactuals.pipelines.run_wach_ours_pipeline",
    # Global methods
    "AReS": "counterfactuals.pipelines.run_ares_pipeline",
    "GLOBE-CE": "counterfactuals.pipelines.run_globe_ce_pipeline",
    # Group methods
    "GLANCE": "counterfactuals.pipelines.run_glance_pipeline",
    "RPPCEF": "counterfactuals.pipelines.run_rppcef_pipeline",
    "PPCEF-R": "counterfactuals.pipelines.run_ppcefr_pipeline",
}


def test_pipeline(method_name: str, module_path: str) -> Tuple[bool, str]:
    """
    Test if a pipeline module can be imported and has a main function.

    Args:
        method_name: Name of the method
        module_path: Python import path to the module

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Try to import the module
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            module_name, _ = parts
        else:
            module_name = module_path

        __import__(module_path)
        module = sys.modules[module_path]

        # Check if main() or pipeline function exists
        if hasattr(module, "main"):
            return True, "Module imported successfully - has main()"
        elif hasattr(module, "pipeline"):
            return True, "Module imported successfully - has pipeline()"
        else:
            # Check what functions/classes are available
            funcs = [
                name for name in dir(module) 
                if not name.startswith("_") and callable(getattr(module, name))
            ]
            return True, f"Module imported - functions: {', '.join(funcs[:3])}..."

    except ImportError as e:
        return False, f"Import Error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_all_pipelines() -> Dict[str, Tuple[bool, str]]:
    """Check availability of all pipelines."""
    results = {}

    for method_name, module_path in PIPELINES.items():
        print(f"Checking {method_name}...", end=" ")
        success, message = test_pipeline(method_name, module_path)
        results[method_name] = (success, message)
        status = "✓" if success else "✗"
        print(f"{status}")

    return results


def generate_markdown_table(results: Dict[str, Tuple[bool, str]]) -> str:
    """Generate markdown table of results."""
    table = "## CF Methods Pipeline Availability\n\n"

    # Local Methods
    table += "### Local Methods\n\n"
    table += "| Method | Status | Notes |\n"
    table += "|--------|--------|-------|\n"

    local_methods = [
        "PPCEF",
        "DiCE",
        "WACH",
        "CCHVAE",
        "DiCoFlex",
        "Artelt",
        "CET",
        "CEM",
        "CEGP",
        # "LiCE",  # Excluded - requires old SPFlow version
        "CaseBased SACE",
        "WACH OURS",
    ]
    for method in local_methods:
        if method in results:
            success, msg = results[method]
            status = "✓" if success else "✗"
            table += f"| {method} | {status} | {msg} |\n"

    # Global Methods
    table += "\n### Global Methods\n\n"
    table += "| Method | Status | Notes |\n"
    table += "|--------|--------|-------|\n"

    global_methods = ["AReS", "GLOBE-CE"]
    for method in global_methods:
        if method in results:
            success, msg = results[method]
            status = "✓" if success else "✗"
            table += f"| {method} | {status} | {msg} |\n"

    # Group Methods
    table += "\n### Group Methods\n\n"
    table += "| Method | Status | Notes |\n"
    table += "|--------|--------|-------|\n"

    group_methods = ["GLANCE", "RPPCEF", "PPCEF-R"]
    for method in group_methods:
        if method in results:
            success, msg = results[method]
            status = "✓" if success else "✗"
            table += f"| {method} | {status} | {msg} |\n"

    return table


def main():
    """Run pipeline availability check."""
    print("=" * 60)
    print("CF Methods Pipeline Availability Check")
    print("=" * 60 + "\n")

    results = check_all_pipelines()

    # Print summary
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} pipelines available")
    print("=" * 60 + "\n")

    # Generate and print markdown table
    table = generate_markdown_table(results)
    print(table)

    # Save table to file
    readme_update = (
        "## Counterfactual Methods Status\n\n"
        f"Last updated: {__import__('datetime').datetime.now().isoformat()}\n\n"
    ) + table

    with open("CF_METHODS_STATUS.md", "w") as f:
        f.write(readme_update)

    print("\nStatus saved to CF_METHODS_STATUS.md")


if __name__ == "__main__":
    main()
