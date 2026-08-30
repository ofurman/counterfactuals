import base64
import hashlib
import io
import json
import subprocess
import xml.etree.ElementTree as ET

import manuscript_typography as figures
import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize("kind", figures.RESULTS)
def test_result_crops_are_lossless_and_uniformly_scaled(kind):
    svg = figures.result_figure(kind)
    meta = json.loads(svg.find(f"{{{figures.SVG}}}metadata").text)
    original = Image.open(figures.ROOT / meta["source"]).convert("RGB")
    assert meta["sourceSha256"] == figures.digest(figures.ROOT / meta["source"])
    assert meta["methods"] == figures.RESULTS[kind]["methods"]
    images = svg.findall(f".//{{{figures.SVG}}}image")
    assert len(images) == len(meta["crops"]) == 5
    for image, crop in zip(images, meta["crops"], strict=True):
        payload = base64.b64decode(image.get("href").split(",")[1])
        assert hashlib.sha256(payload).hexdigest() == crop["pngSha256"]
        raster = Image.open(io.BytesIO(payload))
        np.testing.assert_array_equal(
            np.array(raster), np.array(original.crop(crop["pixelBounds"]))
        )
        assert float(image.get("width")) / float(image.get("height")) == pytest.approx(
            raster.width / raster.height
        )
    assert meta["dataset"] == ("Concrete" if kind == "regression" else "Adult Census")
    labels = svg.findall(".//*[@data-label]")
    assert min(float(label.get("data-font-size")) for label in labels) == figures.LABEL_SIZE
    assert {label.get("data-font-family") for label in labels} == {"Arial"}
    for method in meta["methods"]:
        assert sum(label.get("data-label") == method for label in labels) == (
            1 if kind == "global" else 5
        )
    assert len(svg.findall(f".//{{{figures.SVG}}}text")) == 0


@pytest.mark.parametrize("kind", ["local", "global"])
def test_two_row_metric_layout_and_shared_global_method_key(kind):
    svg = figures.result_figure(kind)
    meta = json.loads(svg.find(f"{{{figures.SVG}}}metadata").text)
    assert meta["layout"] == "three-plus-two"
    top = meta["crops"][:3]
    bottom = meta["crops"][3:]
    assert len({crop["display"][1] for crop in top}) == 1
    assert len({crop["display"][1] for crop in bottom}) == 1
    assert bottom[0]["display"][1] > top[0]["display"][1] + top[0]["display"][3]
    if kind == "global":
        assert meta["methodKey"] == [
            {"tick": str(index + 1), "method": method}
            for index, method in enumerate(figures.RESULTS[kind]["methods"])
        ]
        assert top[0]["display"][2] > (figures.WIDTH / 5 - 44) * 1.35
        for metric in svg.findall(".//*[@id]"):
            if metric.get("id", "").startswith("metric-"):
                ticks = [node.get("data-label") for node in metric.findall(".//*[@data-label]")]
                assert ticks[-3:] == ["1", "2", "3"]


def test_schema_preserves_all_original_vectors_and_font_glyphs(tmp_path):
    target = tmp_path / "source.svg"
    subprocess.run(
        ["pdftocairo", "-svg", str(figures.ROOT / "manuscript/figures/teaser.pdf"), str(target)],
        check=True,
    )
    original = ET.parse(target).getroot()
    actual = figures.architecture_figure()
    meta_node = actual.find(f"{{{figures.SVG}}}metadata")
    meta = json.loads(meta_node.text)
    assert meta["presentation"] == "original-manuscript"
    assert meta["sourceSha256"] == figures.digest(figures.ROOT / meta["source"])
    actual.remove(meta_node)
    assert actual.find(".//*[@id='poster-typography']") is None
    assert any(node.get("id", "").startswith("glyph-") for node in actual.iter())
    assert ET.tostring(actual) == ET.tostring(original)
    assert hashlib.sha256(ET.tostring(actual)).hexdigest() == meta["sourceSvgSha256"]


def test_metric_ticks_match_source_inventory():
    # The regression density shorthand changes typography only, not tick values.
    ticks = figures.RESULTS["regression"]["ticks"]
    assert [text for _, text in ticks[3]] == ["0", "−50k", "−100k", "−150k"]
    assert [text for _, text in ticks[1]] == ["1.0", "0.5"]
    for spec in figures.RESULTS.values():
        for bounds, entries in zip(spec["bounds"], spec["ticks"], strict=True):
            assert all(bounds[1] <= y <= bounds[3] for y, _ in entries)
            assert [y for y, _ in entries] == sorted(y for y, _ in entries)


def test_tick_anchor_rejects_a_missing_source_gridline():
    blank = Image.new("RGB", (100, 100), "white")
    with pytest.raises(ValueError, match="No source gridline"):
        figures.grid_anchor(blank, [0, 0, 100, 100], 100, 50)
