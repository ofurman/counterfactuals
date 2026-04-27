"""Run DiCE / DiCoFlex / CCHVAE across constraint setups for adult, default, lending-club.

Constraint setups encode two kinds of restrictions:
    * immutable: feature held fixed (actionable=false in the dataset yaml)
    * monotonic-increase ("++" in the source table): may only grow

DiCE and CCHVAE cannot enforce a direction, so monotonic features collapse to
immutable for those methods (the safe default already used elsewhere in the
repo). DiCoFlex re-enables them through ``counterfactuals_params.monotonic_overrides``.

Usage:
    uv run python scripts/run_constraint_setup_experiments.py
    uv run python scripts/run_constraint_setup_experiments.py --datasets adult
    uv run python scripts/run_constraint_setup_experiments.py --setups 1 2 --methods dicoflex
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger("constraint_setups")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data_train_test_val"
BASE_CONFIG_DIR = REPO_ROOT / "config" / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "models" / "constraint_setups"


@dataclass(frozen=True)
class Setup:
    """One constraint configuration for a dataset."""

    index: int
    immutable: tuple[str, ...]
    monotonic_increase: tuple[str, ...] = ()


# Setup tables transcribed from the constraint matrix.
# Plain feature -> immutable. Feature with ++ -> monotonic increase.
SETUPS: dict[str, list[Setup]] = {
    "adult": [
        Setup(1, ()),
        Setup(2, ("sex", "race")),
        Setup(3, ("sex", "race", "native_country"), ("age",)),
        Setup(
            4,
            ("sex", "race", "native_country", "marital_status"),
            ("age", "education"),
        ),
    ],
    "default": [
        Setup(1, ()),
        Setup(2, ("SEX", "MARRIAGE")),
        Setup(3, ("SEX", "MARRIAGE"), ("AGE", "EDUCATION")),
        Setup(
            4,
            ("SEX", "MARRIAGE"),
            ("AGE", "EDUCATION", "PAY_0", "PAY_2"),
        ),
        Setup(
            5,
            # BILL_AMT_* shown without ++ in the source table — treated as immutable.
            (
                "SEX",
                "MARRIAGE",
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
            ),
            ("AGE", "EDUCATION", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
        ),
    ],
    "lending-club": [
        Setup(1, ()),
        Setup(2, ("grade", "int_rate")),
        Setup(3, ("grade", "int_rate", "installment", "fico_range_low")),
        Setup(
            4,
            ("grade", "int_rate", "installment", "fico_range_low", "fico_range_high"),
            ("emp_length",),
        ),
    ],
}

PIPELINES: dict[str, str] = {
    "dice": "counterfactuals.pipelines.run_dice_traintest_pipeline",
    "dicoflex": "counterfactuals.pipelines.run_dicoflex_traintest_pipeline",
    "cchvae": "counterfactuals.pipelines.run_cchvae_traintest_pipeline",
}


def base_config_path(dataset: str) -> Path:
    """Map dataset folder name to its shared split yaml."""
    stem = "lending_club_split" if dataset == "lending-club" else f"{dataset}_split"
    return BASE_CONFIG_DIR / f"{stem}.yaml"


def build_setup_yaml(dataset: str, setup: Setup, out_dir: Path, method: str) -> Path:
    """Write a per-setup dataset yaml derived from the shared base config.

    For DiCE / CCHVAE, monotonic features are folded into immutable. For DiCoFlex
    they stay non-actionable here too — the runtime override re-enables them
    with a direction.
    """
    base = yaml.safe_load(base_config_path(dataset).read_text())
    cfg = deepcopy(base)

    feature_config = cfg.setdefault("feature_config", {})
    frozen = set(setup.immutable) | set(setup.monotonic_increase)

    for name in cfg["features"]:
        params = feature_config.get(name, {"actionable": True})
        params["actionable"] = name not in frozen
        feature_config[name] = params

    out_path = out_dir / f"{dataset}_setup{setup.index}_{method}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def monotonic_override_str(features: Iterable[str]) -> str:
    """Format the Hydra dict-literal expected by ``monotonic_overrides``."""
    items = ", ".join(f"{f}: INCREASE" for f in features)
    return "{" + items + "}"


def run_one(method: str, dataset: str, setup: Setup, out_dir: Path, results_root: Path) -> None:
    """Invoke the appropriate pipeline for a single (method, dataset, setup) cell."""
    cfg_path = build_setup_yaml(dataset, setup, out_dir, method)
    train = DATA_ROOT / dataset / "train.csv"
    test = DATA_ROOT / dataset / "test.csv"

    # Per-setup output folder: pipelines append <dataset>/<cf_method>/ underneath.
    setup_output = results_root / f"{dataset}_setup{setup.index}"
    setup_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        PIPELINES[method],
        "disc_model=simple_mlp",
        "disc_model.train_model=true",
        "gen_model.train_model=true",
        f"dataset.config_path={cfg_path.relative_to(REPO_ROOT)}",
        f"dataset.train_data_path={train.relative_to(REPO_ROOT)}",
        f"dataset.test_data_path={test.relative_to(REPO_ROOT)}",
        f"experiment.output_folder={setup_output}",
    ]
    if method == "dicoflex":
        cmd.append(
            f"++counterfactuals_params.monotonic_overrides={monotonic_override_str(setup.monotonic_increase)}"
        )

    logger.info(
        "RUN method=%s dataset=%s setup=%d immutable=%s monotonic=%s",
        method,
        dataset,
        setup.index,
        setup.immutable,
        setup.monotonic_increase,
    )
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    p.add_argument(
        "--setups",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to these setup indices (e.g. 1 2). Skips ones not defined for a dataset.",
    )
    p.add_argument("--methods", nargs="+", choices=list(PIPELINES), default=list(PIPELINES))
    p.add_argument(
        "--keep-configs",
        action="store_true",
        help="Keep generated per-setup yamls instead of cleaning the temp dir.",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root output directory. Each (dataset, setup) gets its own subfolder "
            f"so runs do not overwrite each other. Default: {DEFAULT_RESULTS_ROOT}"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(tempfile.mkdtemp(prefix="cf_setups_", dir=REPO_ROOT))
    args.results_root.mkdir(parents=True, exist_ok=True)
    logger.info("Generated configs will be written under %s", out_dir)
    logger.info("Experiment results will be written under %s", args.results_root)

    failures: list[tuple[str, str, int, str]] = []
    try:
        for dataset in args.datasets:
            for setup in SETUPS[dataset]:
                if args.setups is not None and setup.index not in args.setups:
                    continue
                for method in args.methods:
                    try:
                        run_one(method, dataset, setup, out_dir, args.results_root)
                    except subprocess.CalledProcessError as exc:
                        logger.error(
                            "FAILED method=%s dataset=%s setup=%d rc=%d",
                            method,
                            dataset,
                            setup.index,
                            exc.returncode,
                        )
                        failures.append((method, dataset, setup.index, str(exc)))
    finally:
        if not args.keep_configs:
            shutil.rmtree(out_dir, ignore_errors=True)

    if failures:
        logger.error("Completed with %d failure(s):", len(failures))
        for f in failures:
            logger.error("  %s", f)
        return 1
    logger.info("All experiments completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
