"""Poster-only typography derivatives; never reconstruct benchmark statistics.

Boxplot interiors are lossless crops of the manuscript PNGs. Only surrounding
labels are replaced. The schema is an unmodified vector conversion of the PDF,
including its original font glyphs, line breaks, artwork, and connectors.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.textpath import TextPath
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).parent / "generated"
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)
FONT = FontProperties(family="Arial")
BOLD = FontProperties(family="Arial", weight="bold")
LABEL_SIZE = 13
WIDTH = 550

# Coordinates refer to the manuscript image at the indicated calibration width.
# Tick anchors are the existing gridline positions, not estimated statistics.
RESULTS = {
    "global": {
        "source": "metrics_boxplot_global.png",
        "calibration": 1000,
        "methods": ["AReS", "GLOBE-CE", "GlobalGLANCE"],
        "height": 274,
        "bounds": [
            [71, 33, 198, 150],
            [269, 33, 396, 150],
            [468, 33, 595, 150],
            [666, 33, 793, 150],
            [864, 33, 991, 150],
        ],
        "ticks": [
            [(38, "1.00"), (69, "0.75"), (100, "0.50"), (130, "0.25")],
            [(54, "0.75"), (84, "0.50"), (144, "0.00")],
            [(54, "0.75"), (84, "0.50"), (144, "0.00")],
            [(61, "−200"), (119, "−250")],
            [(52, "262"), (90, "260"), (130, "258")],
        ],
    },
    "group": {
        "source": "metrics_boxplot_group_wise.png",
        "calibration": 1000,
        "methods": ["GLANCE", "T-CREx"],
        "height": 158,
        "bounds": [
            [71, 33, 204, 177],
            [268, 33, 401, 177],
            [464, 33, 597, 177],
            [661, 33, 794, 177],
            [858, 33, 991, 177],
        ],
        "ticks": [
            [(39, "1.0"), (72, "0.8"), (105, "0.6"), (137, "0.4"), (170, "0.2")],
            [(54, "0.4"), (82, "0.3"), (111, "0.2"), (141, "0.1"), (170, "0.0")],
            [(54, "0.4"), (82, "0.3"), (111, "0.2"), (141, "0.1"), (170, "0.0")],
            [(34, "−230"), (75, "−240"), (119, "−250"), (160, "−260")],
            [(43, "262"), (97, "260"), (151, "258")],
        ],
    },
    "local": {
        "source": "metrics_boxplot_local.png",
        "calibration": 1400,
        "methods": ["CADEX", "CCHVAE", "DiCE", "SACE"],
        "height": 270,
        "bounds": [
            [55, 23, 283, 125],
            [333, 23, 561, 125],
            [610, 23, 838, 125],
            [888, 23, 1116, 125],
            [1165, 23, 1394, 125],
        ],
        "ticks": [
            [(28, "1.0"), (63, "0.8"), (98, "0.6")],
            [(32, "0.06"), (61, "0.04"), (91, "0.02"), (120, "0.00")],
            [(39, "0.06"), (71, "0.04"), (101, "0.02")],
            [(45, "−200"), (76, "−400"), (107, "−600")],
            [(47, "400"), (87, "200")],
        ],
    },
    "regression": {
        "source": "regression_metrics_boxplot.png",
        "calibration": 1100,
        "methods": ["CEARM", "WACH"],
        "height": 146,
        "bounds": [
            [87, 36, 218, 166],
            [305, 36, 436, 166],
            [523, 36, 654, 166],
            [741, 36, 872, 166],
            [959, 36, 1090, 166],
        ],
        "ticks": [
            [(41, "0.125"), (76, "0.100"), (111, "0.075"), (146, "0.050")],
            [(63, "1.0"), (115, "0.5")],
            [(43, "1.00"), (89, "0.99"), (136, "0.98")],
            [(43, "0"), (82, "−50k"), (120, "−100k"), (159, "−150k")],
            [(41, "150"), (81, "100"), (122, "50"), (162, "0")],
        ],
    },
}
METRICS = ["Validity (↑)", "L2+Hamming (↓)", "Sparsity (↓)", "Log. dens. (↑)", "Time(s) (↓)"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def element(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {k: str(v) for k, v in attrs.items()})


def label(parent, text, x, y, size=LABEL_SIZE, anchor="middle", bold=False, rotate=0):
    """Outline Arial glyphs for stable, offline browser and print typography."""
    prop = BOLD if bold else FONT
    path = TextPath((0, 0), text, size=size, prop=prop)
    box = path.get_extents()
    shift = {"middle": -box.width / 2 - box.x0, "end": -box.x1, "start": -box.x0}[anchor]
    group = element(parent, "g", transform=f"translate({x} {y}) rotate({rotate})")
    group.set("data-label", text)
    group.set("data-font-size", str(size))
    group.set("data-font-family", "Arial")
    # Matplotlib's path uses an upward y-axis; SVG uses downward y.
    data = []
    for points, code in path.iter_segments(curves=True, simplify=False):
        if code == 79:
            data.append("Z")
            continue
        command = {1: "M", 2: "L", 3: "Q", 4: "C"}[code]
        data.append(command + " ".join(f"{value:.5f}" for value in points))
    element(
        group,
        "path",
        d=" ".join(data),
        transform=f"translate({shift} 0) scale(1 -1)",
        fill="#13253c",
    )
    return group


def metadata(svg, source, **extra):
    value = {
        "source": str(source.relative_to(ROOT)),
        "sourceSha256": digest(source),
        "generatorSha256": digest(Path(__file__)),
        "fontFamily": "Arial",
        **extra,
    }
    element(svg, "metadata").text = json.dumps(value)


def source_crop(image, bounds, calibration):
    scale = image.width / calibration
    pixels = tuple(round(value * scale) for value in bounds)
    return image.crop(pixels), pixels


def grid_anchor(image, bounds, calibration, approximate_y):
    """Snap transcribed tick anchors to the original high-resolution gridlines."""
    scale = image.width / calibration
    left, right = round(bounds[0] * scale) + 3, round(bounds[2] * scale) - 3
    top, bottom = round((approximate_y - 2.5) * scale), round((approximate_y + 2.5) * scale) + 1
    pixels = np.asarray(image.crop((left, top, right, bottom))).astype(int)
    gray = (
        (pixels.max(axis=2) - pixels.min(axis=2) < 8)
        & (pixels[:, :, 0] >= 180)
        & (pixels[:, :, 0] < 245)
    )
    counts = gray.sum(axis=1)
    candidates = np.flatnonzero(counts >= counts.max() * 0.9)
    # Filled boxes can cover most of a gridline; require a visible tenth.
    if counts.max() < (right - left) * 0.10:
        raise ValueError(f"No source gridline near {approximate_y} in {bounds}")
    return (top + float(np.mean(candidates))) / scale


def result_figure(kind):
    spec = RESULTS[kind]
    source = ROOT / "manuscript/figures" / spec["source"]
    image = Image.open(source).convert("RGB")
    svg = ET.Element(
        f"{{{SVG}}}svg",
        width=str(WIDTH + 16),
        height=str(spec["height"]),
        viewBox=f"0 0 {WIDTH + 16} {spec['height']}",
    )
    crops = []
    for index, bounds in enumerate(spec["bounds"]):
        if kind in ("local", "global"):
            row = index // 3
            left = (index % 3 + (0.5 if row else 0)) * WIDTH / 3
            cell_width = WIDTH / 3
            top = row * (126 if kind == "global" else 130)
        else:
            left, cell_width, top = index * WIDTH / 5, WIDTH / 5, 0
        # Uniform scaling preserves every source box/whisker/gridline proportion.
        crop, pixels = source_crop(image, bounds, spec["calibration"])
        left += 16
        plot_x, plot_y = left + 39, top + 24
        plot_width = cell_width - 44
        if kind == "global":
            # Wider source axes fit the shared row height with numbered ticks;
            # the full method names appear once in the key below both rows.
            plot_width = 92
            plot_x += (cell_width - 44 - plot_width) / 2
        plot_height = plot_width * crop.height / crop.width
        if index == 0:
            label(
                svg,
                "Concrete" if kind == "regression" else "Adult Census",
                10,
                plot_y + plot_height / 2,
                rotate=-90,
            )
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        payload = buffer.getvalue()
        group = element(svg, "g", id=f"metric-{index}")
        element(
            group,
            "image",
            x=plot_x,
            y=plot_y,
            width=plot_width,
            height=plot_height,
            href="data:image/png;base64," + base64.b64encode(payload).decode(),
        )
        titles = ["MAE (↓)", "L2 (↓)", *METRICS[2:]] if kind == "regression" else METRICS
        label(group, titles[index], left + cell_width / 2, top + 15)
        anchored_ticks = []
        for source_y, text in spec["ticks"][index]:
            source_y = grid_anchor(image, bounds, spec["calibration"], source_y)
            y = plot_y + (source_y - bounds[1]) / (bounds[3] - bounds[1]) * plot_height
            label(group, text, plot_x - 4, y + 4, anchor="end")
            anchored_ticks.append([source_y, text])
        count = len(spec["methods"])
        for method_index, method in enumerate(spec["methods"]):
            x = plot_x + plot_width * (method_index + 0.5) / count
            if kind == "global":
                label(group, str(method_index + 1), x, plot_y + plot_height + 15)
            else:
                label(group, method, x, plot_y + plot_height + 10, anchor="end", rotate=-55)
        crops.append(
            {
                "metric": titles[index],
                "pixelBounds": pixels,
                "pngSha256": hashlib.sha256(payload).hexdigest(),
                "display": [plot_x, plot_y, plot_width, plot_height],
                "ticks": anchored_ticks,
            }
        )
    method_key = []
    if kind == "global":
        key = element(svg, "g", id="method-key")
        for index, method in enumerate(spec["methods"]):
            x = 60 + index * 166
            label(key, str(index + 1), x, 268, bold=True)
            label(key, method, x + 13, 268, anchor="start")
            method_key.append({"tick": str(index + 1), "method": method})
    metadata(
        svg,
        source,
        minimumFontSize=LABEL_SIZE,
        dataset="Concrete" if kind == "regression" else "Adult Census",
        methods=spec["methods"],
        layout="three-plus-two" if kind in ("global", "local") else "five-across",
        methodKey=method_key,
        crops=crops,
        transformation="Lossless source plot crops; replaced labels only; k denotes 1000.",
    )
    return svg


def architecture_figure():
    """Convert the original manuscript diagram without replacing any artwork or text."""
    source = ROOT / "manuscript/figures/teaser.pdf"
    with tempfile.TemporaryDirectory(prefix="cel-schema-") as temporary:
        target = Path(temporary) / "schema.svg"
        subprocess.run(["pdftocairo", "-svg", str(source), str(target)], check=True)
        svg = ET.parse(target).getroot()
    source_svg_sha256 = hashlib.sha256(ET.tostring(svg, encoding="utf-8")).hexdigest()
    metadata(
        svg,
        source,
        fontFamily="Poppins / Canva Sans (manuscript originals)",
        # All source PDF glyphs have this size, measured with pdfplumber.
        minimumFontSize=9.9975,
        presentation="original-manuscript",
        sourceSvgSha256=source_svg_sha256,
        transformation="Unmodified pdftocairo vector conversion of the source PDF CropBox.",
    )
    return svg


def main():
    findfont(FONT, fallback_to_default=False)
    OUTPUT.mkdir(exist_ok=True)
    for name, svg in [(kind, result_figure(kind)) for kind in RESULTS] + [
        ("architecture", architecture_figure())
    ]:
        target = OUTPUT / f"manuscript-{name}.svg"
        ET.ElementTree(svg).write(target, encoding="utf-8", xml_declaration=True)
        print(target)


if __name__ == "__main__":
    main()
