"""Render the poster's loan examples with the actual CEL Matplotlib helpers.

Run from the repository root with:
    uv run --no-sync python poster/plots/plot_ce_examples.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "poster/research/ce-example.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated"
PARADIGMS = ("local", "global", "group-wise")
LABEL_SIZE = 14.5
PLOT_HEIGHT = 2.65
LEGEND_HEIGHT = 0.33
LEGEND_LABELS = ["Original", "Counterfactual", "Decision boundary"]


def load_plot_utils():
    """Bypass CEL's eager model imports without copying its plotting functions."""
    path = ROOT / "cel/plotting/plot_utils.py"
    spec = importlib.util.spec_from_file_location("cel_example_plot_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CEL plotting helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLOT_UTILS = load_plot_utils()


@dataclass(frozen=True)
class Transition:
    applicant: str
    original: dict
    counterfactual: dict
    group: int


def predict(profile, model):
    return (
        profile["monthlyIncome"] - profile["monthlyDebt"] >= model["minimumIncomeAfterDebt"]
        and profile["creditHistoryYears"] >= model["minimumCreditHistoryYears"]
        and profile["loanAmount"] / (12 * profile["monthlyIncome"])
        <= model["maximumLoanToAnnualIncomeRatio"]
    )


def build_transitions(example, paradigm):
    if paradigm not in PARADIGMS:
        raise ValueError(f"Unknown paradigm: {paradigm}")
    if paradigm == "local":
        return [Transition("A", dict(example["original"]), dict(example["counterfactual"]), 0)]
    transitions = []
    for applicant in example["applicants"]:
        identifier = applicant["id"]
        original = {**example["original"], **{k: v for k, v in applicant.items() if k != "id"}}
        matches = [
            (index, group)
            for index, group in enumerate(example["groups"])
            if identifier in group["applicants"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Applicant {identifier} must belong to exactly one group")
        index, group = matches[0]
        change = example["globalChange"] if paradigm == "global" else group["change"]
        counterfactual = {
            **original,
            **{key: original[key] + delta for key, delta in change.items()},
        }
        transitions.append(
            Transition(identifier, original, counterfactual, 0 if paradigm == "global" else index)
        )
    return transitions


def validate_example(example):
    if example["kind"] != "illustrative":
        raise ValueError("The example must stay separate from benchmark measurements")
    applicants = example["applicants"]
    if [applicant["id"] for applicant in applicants] != ["A", "B", "C", "D"]:
        raise ValueError("Expected the shared A-D population")
    if any(
        applicants[0][key] != example["original"][key] for key in ("monthlyIncome", "monthlyDebt")
    ):
        raise ValueError("The local original must match applicant A in the shared population")
    for paradigm in PARADIGMS:
        for transition in build_transitions(example, paradigm):
            if predict(transition.original, example["model"]) or not predict(
                transition.counterfactual, example["model"]
            ):
                raise ValueError(f"{paradigm}/{transition.applicant} must go Declined -> Approved")
            for key in example["original"]:
                if key not in ("monthlyIncome", "monthlyDebt"):
                    if transition.original[key] != transition.counterfactual[key]:
                        raise ValueError(f"The example must not change {key}")
            for profile in (transition.original, transition.counterfactual):
                if np.any(normalize([profile], example) < 0) or np.any(
                    normalize([profile], example) > 1
                ):
                    raise ValueError("Example points must fit the shared plot bounds")


def normalize(profiles, example):
    """CEL's boundary and arrow helpers operate on the unit square."""
    bounds = example["plot"]
    points = np.array([[p["monthlyIncome"], p["monthlyDebt"]] for p in profiles], dtype=float)
    return (points - [bounds["minimumIncome"], 0]) / [
        bounds["maximumIncome"] - bounds["minimumIncome"],
        bounds["maximumDebt"],
    ]


class ExampleClassifier:
    """Project the declared illustrative rule onto income/debt, not a fitted model."""

    def __init__(self, example):
        self.example = example

    def predict(self, points):
        bounds = self.example["plot"]
        model = self.example["model"]
        fixed = self.example["original"]
        income = bounds["minimumIncome"] + points[:, 0] * (
            bounds["maximumIncome"] - bounds["minimumIncome"]
        )
        debt = points[:, 1] * bounds["maximumDebt"]
        approved = (
            (income - debt >= model["minimumIncomeAfterDebt"])
            & (fixed["loanAmount"] / (12 * income) <= model["maximumLoanToAnnualIncomeRatio"])
            & (fixed["creditHistoryYears"] >= model["minimumCreditHistoryYears"])
        )
        return approved.int()


def draw_example(ax, example, paradigm):
    transitions = build_transitions(example, paradigm)
    original = normalize([t.original for t in transitions], example)
    counterfactual = normalize([t.counterfactual for t in transitions], example)
    PLOT_UTILS.plot_classifier_decision_region(ax, ExampleClassifier(example))
    ax.collections[-1].set_gid(f"{paradigm}-boundary")

    # Keep CEL's tab10 originals, orange counterfactuals, and translucent black arrows.
    # Skip tab10's orange index for groups so it remains reserved for counterfactuals.
    color_indices = np.array([0 if t.group == 0 else 2 for t in transitions])
    PLOT_UTILS.plot_observations(ax, original, color_indices)
    observations = ax.collections[-1]
    observations.set_clim(0, 9)
    observations.set_zorder(4)
    observations.set_gid(f"{paradigm}-originals-declined")
    PLOT_UTILS.plot_counterfactuals(ax, counterfactual)
    ax.collections[-1].set_zorder(4)
    ax.collections[-1].set_gid(f"{paradigm}-counterfactuals-approved")
    first_arrow = len(ax.patches)
    PLOT_UTILS.plot_arrows(ax, original, counterfactual)
    for arrow, transition in zip(ax.patches[first_arrow:], transitions):
        # Preserve the helper's direction, black/alpha style, and exact endpoint.
        arrow.set_data(width=0.002, head_width=0.020, head_length=0.025)
        arrow.set_zorder(3)
        arrow.set_gid(f"{paradigm}-arrow-{transition.applicant}")

    for transition, point in zip(transitions, original):
        ax.annotate(
            transition.applicant,
            point,
            xytext=(-7, 7),
            textcoords="offset points",
            ha="right",
            fontsize=LABEL_SIZE,
            fontweight="bold",
        )
    ax.text(0.035, 0.88, "Declined", transform=ax.transAxes, fontsize=LABEL_SIZE, fontweight="bold")
    ax.text(
        0.97,
        0.08,
        "Approved",
        transform=ax.transAxes,
        ha="right",
        fontsize=LABEL_SIZE,
        fontweight="bold",
    )
    bounds = example["plot"]
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Monthly income (€)", ylabel="Debt payments (€)")
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    ax.tick_params(labelsize=LABEL_SIZE)
    income_span = bounds["maximumIncome"] - bounds["minimumIncome"]
    ax.set_xticks([(value - bounds["minimumIncome"]) / income_span for value in (2000, 3000, 4000)])
    ax.set_yticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{bounds['minimumIncome'] + x * income_span:,.0f}")
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * bounds['maximumDebt']:,.0f}"))
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return ax


def create_figure(example, paradigms):
    """Keep axes sizes fixed; reserve a legend row only when group-wise is present."""
    extra = LEGEND_HEIGHT if "group-wise" in paradigms else 0
    height = PLOT_HEIGHT + extra
    fig, axes = plt.subplots(
        1, len(paradigms), figsize=(6.4 * len(paradigms), height), squeeze=False
    )
    fig.subplots_adjust(
        left=0.18 / len(paradigms),
        right=0.98,
        wspace=0.35,
        bottom=(0.26 * PLOT_HEIGHT + extra) / height,
        top=(0.94 * PLOT_HEIGHT + extra) / height,
    )
    for ax, paradigm in zip(axes[0], paradigms):
        draw_example(ax, example, paradigm)
        if paradigm == "group-wise":
            add_group_legend(fig, ax, (paradigms.index(paradigm) + 0.5) / len(paradigms))
    return fig


def add_group_legend(fig, ax, center):
    """Both original group colors share the same declined status."""
    originals = tuple(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=matplotlib.colormaps["tab10"](index),
            alpha=0.8,
            markersize=np.sqrt(50),
        )
        for index in (0, 2)
    )
    counterfactual = Line2D(
        [], [], marker="o", linestyle="none", color="orange", alpha=0.8, markersize=np.sqrt(50)
    )
    boundary = next(
        collection for collection in ax.collections if collection.get_gid() == "group-wise-boundary"
    )
    colors = boundary.get_edgecolors()
    line = Line2D([], [], color=colors[len(colors) // 2], linewidth=boundary.get_linewidths()[0])
    legend = fig.legend(
        handles=[originals, counterfactual, line],
        labels=LEGEND_LABELS,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower center",
        bbox_to_anchor=(center, 0.01),
        borderaxespad=0,
        borderpad=0,
        ncol=3,
        frameon=False,
        fontsize=LABEL_SIZE,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=0.8,
    )
    legend.set_gid("group-wise-legend")


def asset_metadata(example, data_path, selected, transparent):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "dataSha256": digest(data_path),
        "styleSha256": digest(ROOT / "cel/plotting/plot_utils.py"),
        "generatorSha256": digest(Path(__file__)),
        "transparent": transparent,
        "minimumFontPt": LABEL_SIZE,
        "fontFamily": "Arial",
        "legendLabels": LEGEND_LABELS if "group-wise" in selected else [],
        "transitions": {
            paradigm: [
                {"id": t.applicant, "original": t.original, "counterfactual": t.counterfactual}
                for t in build_transitions(example, paradigm)
            ]
            for paradigm in selected
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paradigm", choices=(*PARADIGMS, "all"), default="all")
    parser.add_argument("--formats", choices=("png", "svg"), nargs="+", default=["png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--transparent", action="store_true", help="Export without a white background"
    )
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    example = json.loads(args.data.read_text())
    validate_example(example)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paradigms = PARADIGMS if args.paradigm == "all" else (args.paradigm,)
    exports = [(paradigm, (paradigm,)) for paradigm in paradigms]
    if args.paradigm == "all":
        exports.append(("comparison", PARADIGMS))
    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.size": LABEL_SIZE,
            "svg.fonttype": "path",
            "svg.hashsalt": "cel-ce-examples",
        }
    ):
        for name, selected in exports:
            fig = create_figure(example, selected)
            try:
                for extension in dict.fromkeys(args.formats):
                    output = args.output_dir / f"ce-example-{name}.{extension}"
                    fig.savefig(
                        output,
                        dpi=args.dpi,
                        transparent=args.transparent,
                        metadata={
                            "Creator": "CEL poster example generator",
                            "Date": None,
                            "Description": json.dumps(
                                asset_metadata(example, args.data, selected, args.transparent)
                            ),
                        }
                        if extension == "svg"
                        else None,
                    )
                    if extension == "svg":
                        output.write_text(
                            "\n".join(line.rstrip() for line in output.read_text().splitlines())
                            + "\n"
                        )
                    print(output)
            finally:
                plt.close(fig)


if __name__ == "__main__":
    main()
