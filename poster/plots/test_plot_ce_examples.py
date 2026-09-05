import copy
import hashlib
import inspect
import json
import xml.etree.ElementTree as ET

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import plot_ce_examples as plots
import pytest
import torch
from matplotlib.collections import PathCollection
from matplotlib.text import Text


@pytest.fixture
def example():
    return json.loads(plots.DEFAULT_DATA.read_text())


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_reuses_library_helpers():
    for name in (
        "plot_observations",
        "plot_counterfactuals",
        "plot_arrows",
        "plot_classifier_decision_region",
    ):
        assert inspect.getsourcefile(getattr(plots.PLOT_UTILS, name)) == str(
            plots.ROOT / "cel/plotting/plot_utils.py"
        )


@pytest.mark.parametrize("paradigm,count", [("local", 1), ("global", 4), ("group-wise", 4)])
def test_every_transition_is_declined_to_approved(example, paradigm, count):
    plots.validate_example(example)
    transitions = plots.build_transitions(example, paradigm)
    assert len(transitions) == count
    classifier = plots.ExampleClassifier(example)
    for transition in transitions:
        assert not plots.predict(transition.original, example["model"])
        assert plots.predict(transition.counterfactual, example["model"])
        points = plots.normalize([transition.original, transition.counterfactual], example)
        assert classifier.predict(torch.tensor(points)).tolist() == [0, 1]
        for key in ("age", "employment", "creditHistoryYears", "loanAmount"):
            assert transition.original[key] == transition.counterfactual[key]


def test_shared_actions_and_population(example):
    global_transitions = plots.build_transitions(example, "global")
    grouped = plots.build_transitions(example, "group-wise")
    assert [t.original for t in global_transitions] == [t.original for t in grouped]

    def delta(transition):
        return tuple(
            transition.counterfactual[key] - transition.original[key]
            for key in ("monthlyIncome", "monthlyDebt")
        )

    assert [delta(t) for t in global_transitions] == [(1300, 0)] * 4
    assert [delta(t) for t in grouped] == [(0, -1000), (0, -1000), (1300, 0), (1300, 0)]


@pytest.mark.parametrize("paradigm", plots.PARADIGMS)
def test_artists_match_points_directions_and_library_style(example, paradigm):
    fig = plots.create_figure(example, (paradigm,))
    fig.canvas.draw()
    ax = fig.axes[0]
    transitions = plots.build_transitions(example, paradigm)
    original = plots.normalize([t.original for t in transitions], example)
    target = plots.normalize([t.counterfactual for t in transitions], example)
    points = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
    assert len(points) == 2
    np.testing.assert_allclose(points[0].get_offsets(), original)
    np.testing.assert_allclose(points[1].get_offsets(), target)
    np.testing.assert_allclose(points[1].get_facecolors(), [colors.to_rgba("orange", alpha=0.8)])
    assert points[0].get_cmap().name == "tab10"
    diamond = plots.MarkerStyle("D")
    np.testing.assert_allclose(
        points[1].get_paths()[0].vertices,
        diamond.get_path().transformed(diamond.get_transform()).vertices,
    )
    assert points[0].get_paths()[0].vertices.shape != points[1].get_paths()[0].vertices.shape
    assert len(ax.patches) == len(transitions)
    for arrow, start, end in zip(ax.patches, original, target):
        np.testing.assert_allclose(
            arrow.get_facecolor(), colors.to_rgba(plots.ARROW_COLOR, alpha=plots.ARROW_ALPHA)
        )
        assert arrow._width == 0.0035
        assert arrow._head_width == 0.035
        assert arrow._head_length == 0.040
        direction = end - start
        projection = (arrow.get_xy() - start) @ direction / (direction @ direction)
        assert projection.min() == pytest.approx(0)
        assert projection.max() == pytest.approx(1)
        assert any(np.allclose(vertex, end) for vertex in arrow.get_xy())
    assert ax.get_xlim() == (0, 1)
    assert ax.get_ylim() == (0, 1)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["2,000", "3,000", "4,000"]


@pytest.mark.parametrize("selected", [(p,) for p in plots.PARADIGMS] + [plots.PARADIGMS])
def test_equal_axes_and_no_clipped_text(example, selected):
    fig = plots.create_figure(example, selected)
    fig.canvas.draw()
    assert not fig.texts
    assert len(fig.legends) == int("global" in selected)
    assert all(not ax.get_title() for ax in fig.axes)
    renderer = fig.canvas.get_renderer()
    sizes = [ax.get_window_extent(renderer).size for ax in fig.axes]
    for size in sizes[1:]:
        np.testing.assert_allclose(size, sizes[0])
    for label in fig.findobj(Text):
        if not label.get_visible() or not label.get_text():
            continue
        bounds = label.get_window_extent(renderer)
        assert bounds.x0 >= -1, label.get_text()
        assert bounds.y0 >= -1, label.get_text()
        assert bounds.x1 <= fig.bbox.x1 + 1, label.get_text()
        assert bounds.y1 <= fig.bbox.y1 + 1, label.get_text()


def test_global_legend_matches_marks_and_sits_below_the_plot(example):
    local = plots.create_figure(example, ("local",))
    global_plot = plots.create_figure(example, ("global",))
    local.canvas.draw()
    global_plot.canvas.draw()
    np.testing.assert_allclose(local.axes[0].bbox.size, global_plot.axes[0].bbox.size)
    legend = global_plot.legends[0]
    assert [label.get_text() for label in legend.get_texts()] == plots.LEGEND_LABELS
    assert legend.get_gid() == "global-legend"
    renderer = global_plot.canvas.get_renderer()
    assert (
        legend.get_window_extent(renderer).y1
        < global_plot.axes[0].xaxis.label.get_window_extent(renderer).y0
    )
    lines = legend.findobj(plots.Line2D)
    assert len(lines) == 3
    assert [line.get_marker() for line in lines[:2]] == ["o", "D"]
    np.testing.assert_allclose(
        colors.to_rgba(lines[0].get_color(), lines[0].get_alpha()),
        colors.to_rgba(plots.matplotlib.colormaps["tab10"](0), 0.8),
    )
    np.testing.assert_allclose(
        colors.to_rgba(lines[1].get_color(), lines[1].get_alpha()), colors.to_rgba("orange", 0.8)
    )
    boundary = global_plot.axes[0].collections[0]
    edge_colors = boundary.get_edgecolors()
    np.testing.assert_allclose(lines[2].get_color(), edge_colors[len(edge_colors) // 2])
    assert lines[2].get_linewidth() == boundary.get_linewidths()[0]
    assert boundary.get_alpha() == 1
    assert boundary.get_linewidths()[0] == plots.BOUNDARY_WIDTH
    np.testing.assert_allclose(edge_colors, [colors.to_rgba(plots.BOUNDARY_COLOR)])


def test_invalid_examples_are_rejected(example):
    broken = copy.deepcopy(example)
    broken["globalChange"]["monthlyIncome"] = 0
    with pytest.raises(ValueError, match="Declined -> Approved"):
        plots.validate_example(broken)
    broken = copy.deepcopy(example)
    broken["groups"][1]["applicants"].append("A")
    with pytest.raises(ValueError, match="exactly one group"):
        plots.validate_example(broken)


def test_cli_exports_all_plots_and_transparent_background(tmp_path):
    plots.main(["--output-dir", str(tmp_path), "--dpi", "72", "--transparent"])
    assert {path.name for path in tmp_path.iterdir()} == {
        f"ce-example-{name}.{extension}"
        for name in (*plots.PARADIGMS, "comparison")
        for extension in ("png", "svg")
    }
    for path in tmp_path.glob("*.png"):
        image = plt.imread(path)
        assert image.shape[2] == 4
        assert image[0, 0, 3] == 0
    svg_path = tmp_path / "ce-example-comparison.svg"
    root = ET.parse(svg_path).getroot()
    description = root.find(".//{http://purl.org/dc/elements/1.1/}description").text
    metadata = json.loads(description)
    assert metadata["dataSha256"] == hashlib.sha256(plots.DEFAULT_DATA.read_bytes()).hexdigest()
    assert list(metadata["transitions"]) == list(plots.PARADIGMS)
    assert metadata["transparent"] is True
    assert not root.findall(".//{http://www.w3.org/2000/svg}image")
    assert not root.findall(".//{http://www.w3.org/2000/svg}text")


def test_cli_single_paradigm_and_invalid_dpi(tmp_path):
    plots.main(["--output-dir", str(tmp_path), "--paradigm", "local", "--formats", "svg"])
    assert [path.name for path in tmp_path.iterdir()] == ["ce-example-local.svg"]
    with pytest.raises(SystemExit):
        plots.main(["--dpi", "0"])
