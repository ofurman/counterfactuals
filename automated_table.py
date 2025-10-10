#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tables_generator.py
-------------------
Builds two LaTeX tables for each model:
  1) Main metrics:   Cov., Val., PP, LOF, IF, LD
  2) Prox/Efficiency: Hamming, L1, L2, Time

You provide lists of methods, datasets, and models. For each (method, dataset, model)
the adapter `fetch_metrics` must return (mean, std) pairs or None (to show '--').

Usage (as a library):
    from tables_generator import generate_two_tables
    methods  = ["PPCEF", "C-CHVAE", "DiCE"]
    datasets = ["Lending Club", "GMSC", "Bank Marketing", "Credit Default", "Adult Census"]
    model    = "MLP"
    main_tex, prox_tex = generate_two_tables(methods, datasets, model)

Usage (CLI):
    python tables_generator.py \
      --methods PPCEF "C-CHVAE" DiCE \
      --datasets "Lending Club" GMSC "Bank Marketing" "Credit Default" "Adult Census" \
      --models MLP XGBoost \
      --outdir tables

Then include in LaTeX:
    \input{tables/main_metrics_MLP.tex}
    \input{tables/proximity_efficiency_MLP.tex}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

Number = float
MeanStd = Optional[Tuple[Optional[Number], Optional[Number]]]

# ============================================================
# 1) CONFIGURABLE ADAPTER
# ============================================================


def fetch_metrics(
    method: str, dataset: str, model: str
) -> Dict[str, Dict[str, MeanStd]]:
    """
    Adapter returning metrics for a single (method, dataset, model) triple.

    Replace the dummy body below with your real integration,
    e.g. calling `calc_metrics(dataset=..., method=..., model=...)` and mapping
    its output to the expected keys.

    Return structure (keys are EXACT column labels used in the tables):
    {
      "main": {
        "Cov.": (mean, std) or None,
        "Val.": (mean, std) or None,
        "PP":   (mean, std) or None,
        "LOF":  (mean, std) or None,
        "IF":   (mean, std) or None,
        "LD":   (mean, std) or None,
      },
      "prox_eff": {
        "Hamming": (mean, std) or None,
        "L1":      (mean, std) or None,
        "L2":      (mean, std) or None,
        "Time":    (mean, std) or None,
      }
    }

    Notes:
    - Return None for any metric you want displayed as '--'.
    - Use float('nan') or float('inf') to signal missing; those are also rendered as '--'.
    """
    # ----------------- DUMMY DEMO (DELETE/WIRE YOUR calc_metrics) -----------------
    # Deterministic fake numbers so you see formatting and layout.
    import random

    rnd = random.Random(hash((method, dataset, model)) & 0xFFFFFFFF)

    def ms(mu, sigma) -> Tuple[float, float]:
        return (round(float(mu), 2), round(abs(float(sigma)), 2))

    main = {
        "Cov.": ms(1.0, 0.0),
        "Val.": ms(1.0 if method == "PPCEF" else 0.65, 0.04),
        "PP": ms(0.75 + 0.1 * (method == "PPCEF"), 0.08),
        "LOF": ms(5.0 + rnd.random(), 0.3 + 0.1 * rnd.random()),
        "IF": ms((-0.05 if method == "PPCEF" else 0.06), 0.01),
        "LD": ms(-35.0 + 5.0 * rnd.random(), 10.0 * rnd.random()),
    }
    prox_eff = {
        "Hamming": ms(0.8 - 0.05 * (method == "DiCE"), 0.01),
        "L1": ms(0.85 - 0.08 * (method == "C-CHVAE"), 0.01),
        "L2": ms(0.83 - 0.1 * (method == "DiCE"), 0.01),
        "Time": ms(250.0 + 50.0 * rnd.random(), 20.0 + 10.0 * rnd.random()),
    }

    # Example: hide data for "Credit Default" to show '--'
    if dataset == "Credit Default":
        main = {k: None for k in main}
        prox_eff = {k: None for k in prox_eff}

    return {"main": main, "prox_eff": prox_eff}


# Example wiring (pseudo-code) if you have a calc_metrics() you can import:
#
# from my_metrics_lib import calc_metrics
# def fetch_metrics(method: str, dataset: str, model: str) -> Dict[str, Dict[str, MeanStd]]:
#     res = calc_metrics(dataset=dataset, method=method, model=model)
#     # Map your keys to the required ones:
#     def pick(name):  # expects res[name] = {"mean": ..., "std": ...} or (mean, std)
#         v = res.get(name)
#         if v is None: return None
#         if isinstance(v, tuple) and len(v) == 2:
#             return (float(v[0]), float(v[1]))
#         mu = v.get("mean"); sd = v.get("std")
#         return None if (mu is None or sd is None) else (float(mu), float(sd))
#     return {
#         "main": {
#             "Cov.": pick("Cov."),
#             "Val.": pick("Val."),
#             "PP":   pick("PP"),
#             "LOF":  pick("LOF"),
#             "IF":   pick("IF"),
#             "LD":   pick("LD"),
#         },
#         "prox_eff": {
#             "Hamming": pick("Hamming"),
#             "L1":      pick("L1"),
#             "L2":      pick("L2"),
#             "Time":    pick("Time"),
#         }
#     }


# ============================================================
# 2) LATEX GENERATION CORE
# ============================================================


@dataclass
class TableSpec:
    cols: List[str]  # ordered metric columns
    arrows: Dict[str, str]  # e.g., {"Cov.": r"$\uparrow$", ...}
    caption: str
    label: str
    header_left: List[str]  # usually ["Dataset", "Method"]


def format_mean_std(value: MeanStd, digits: int = 2) -> str:
    """Format (mean, std) as 'm±s' with given precision; return '--' if missing/NaN/Inf."""
    if value is None:
        return r"--"
    mean, std = value
    if mean is None or std is None:
        return r"--"
    for x in (mean, std):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return r"--"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(float(mean))}$\\pm${fmt.format(float(std))}"


def make_tabular_header(spec: TableSpec) -> str:
    right_cols = "r" * len(spec.cols)
    colspec = f"l|l|{right_cols}"
    header_metrics = " & ".join(
        f"{name} {spec.arrows.get(name, '')}".strip() for name in spec.cols
    )
    header = (
        "\\begin{tabular}{" + colspec + "}\n"
        "\\toprule\n" + " & ".join(spec.header_left + [header_metrics]) + " \\\\\n"
        "\\midrule\n"
    )
    return header


def latex_multirow(dataset: str, n_methods: int) -> str:
    return f"\\multirow{{{n_methods}}}{{*}}{{{dataset}}}"


def make_rows(
    datasets: List[str],
    methods: List[str],
    model: str,
    spec: TableSpec,
    metrics_key: str,
    getter: Callable[[str, str, str], Dict[str, Dict[str, MeanStd]]],
    digits_map: Optional[Dict[str, int]] = None,
) -> str:
    lines: List[str] = []
    for d_idx, dataset in enumerate(datasets):
        for m_idx, method in enumerate(methods):
            data = getter(method, dataset, model)
            block = data.get(metrics_key, {})
            formatted = []
            for col in spec.cols:
                digits = digits_map.get(col, 2) if digits_map else 2
                formatted.append(format_mean_std(block.get(col), digits))
            if m_idx == 0:
                left = f"{latex_multirow(dataset, len(methods))} & {method}"
            else:
                left = f" & {method}"
            line = left + " & " + " & ".join(formatted) + r" \\"
            lines.append(line)
        if d_idx < len(datasets) - 1:
            lines.append("\\midrule")
    return "\n".join(lines) + "\n"


def wrap_table(tabular: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[th]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{center}\n"
        "\\begin{sc}\n"
        "\\begin{scriptsize}\n" + tabular + "\\end{scriptsize}\n"
        "\\end{sc}\n"
        "\\end{center}\n"
        "\\end{table}\n"
    )


def build_table(
    datasets: List[str],
    methods: List[str],
    model: str,
    spec: TableSpec,
    metrics_key: str,
    getter: Callable[[str, str, str], Dict[str, Dict[str, MeanStd]]],
    digits_map: Optional[Dict[str, int]] = None,
) -> str:
    header = make_tabular_header(spec)
    body = make_rows(datasets, methods, model, spec, metrics_key, getter, digits_map)
    tabular = header + body + "\\bottomrule\n\\end{tabular}\n"
    return wrap_table(tabular, spec.caption, spec.label)


# ============================================================
# 3) SPECS MATCHING YOUR EXAMPLE TABLES
# ============================================================


def make_specs(model: str):
    main_cols = ["Cov.", "Val.", "PP", "LOF", "IF", "LD"]
    main_arrows = {
        "Cov.": r"$\uparrow$",
        "Val.": r"$\uparrow$",
        "PP": r"$\uparrow$",
        "LOF": r"$\downarrow$",
        "IF": r"$\downarrow$",
        "LD": r"$\uparrow$",
    }
    main_caption = (
        "Performance Analysis Across Different Datasets - Main Metrics. "
        "Results demonstrate the effectiveness of PPCEF in achieving high coverage, validity, "
        "and probabilistic plausibility compared to baseline methods across various financial and "
        f"demographic datasets. \\textbf{{{model}}}"
    )
    main_label = "tab:main_metrics_comparison"

    prox_cols = ["Hamming", "L1", "L2", "Time"]
    prox_arrows = {
        "Hamming": r"$\downarrow$",
        "L1": r"$\downarrow$",
        "L2": r"$\downarrow$",
        "Time": r"$\downarrow$",
    }
    prox_caption = f"Performance Analysis Across Different Datasets for Proximity and Efficiency Metrics. (\\textbf{{{model}}})"
    prox_label = "tab:proximity_efficiency_comparison"

    main_spec = TableSpec(
        cols=main_cols,
        arrows=main_arrows,
        caption=main_caption,
        label=main_label,
        header_left=["Dataset", "Method"],
    )
    prox_spec = TableSpec(
        cols=prox_cols,
        arrows=prox_arrows,
        caption=prox_caption,
        label=prox_label,
        header_left=["Dataset", "Method"],
    )

    # Per-metric precision (edit if you need different digits)
    main_digits = {"Cov.": 2, "Val.": 2, "PP": 2, "LOF": 2, "IF": 2, "LD": 2}
    prox_digits = {"Hamming": 2, "L1": 2, "L2": 2, "Time": 2}
    return main_spec, prox_spec, main_digits, prox_digits


# ============================================================
# 4) PUBLIC API
# ============================================================


def generate_two_tables(
    methods: List[str],
    datasets: List[str],
    model: str,
    getter: Callable[[str, str, str], Dict[str, Dict[str, MeanStd]]] = fetch_metrics,
    out_main: Optional[Path] = None,
    out_prox: Optional[Path] = None,
) -> Tuple[str, str]:
    """
    Build two LaTeX tables (main metrics and proximity/efficiency) for a given
    model across datasets×methods, using a metrics getter adapter.

    Args:
        methods: Ordered list of method names (e.g., ["PPCEF", "C-CHVAE", "DiCE"]).
        datasets: Ordered list of dataset names.
        model: Model label to print in captions (e.g., "MLP").
        getter: Function (method, dataset, model) -> metrics dict (see `fetch_metrics` docstring).
        out_main: Optional path to write the main-metrics table (.tex).
        out_prox: Optional path to write the proximity/efficiency table (.tex).

    Returns:
        (main_table_tex, proximity_efficiency_table_tex) as strings.
    """
    main_spec, prox_spec, main_digits, prox_digits = make_specs(model)

    main_tex = build_table(
        datasets=datasets,
        methods=methods,
        model=model,
        spec=main_spec,
        metrics_key="main",
        getter=getter,
        digits_map=main_digits,
    )
    prox_tex = build_table(
        datasets=datasets,
        methods=methods,
        model=model,
        spec=prox_spec,
        metrics_key="prox_eff",
        getter=getter,
        digits_map=prox_digits,
    )

    if out_main is not None:
        Path(out_main).write_text(main_tex, encoding="utf-8")
    if out_prox is not None:
        Path(out_prox).write_text(prox_tex, encoding="utf-8")

    return main_tex, prox_tex


# ============================================================
# 5) OPTIONAL CLI
# ============================================================


def _cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Generate LaTeX tables for metrics across datasets/methods per model."
    )
    p.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="List of method names (order preserved).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names (order preserved).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of models; one pair of tables per model.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("tables"),
        help="Output directory for .tex files.",
    )
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        out_main = args.outdir / f"main_metrics_{model}.tex"
        out_prox = args.outdir / f"proximity_efficiency_{model}.tex"
        generate_two_tables(
            methods=args.methods,
            datasets=args.datasets,
            model=model,
            out_main=out_main,
            out_prox=out_prox,
        )
        print(f"Wrote: {out_main}")
        print(f"Wrote: {out_prox}")


if __name__ == "__main__":
    _cli()
