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
    assert min(float(label.get("data-font-size")) for label in labels) == 12.5
    assert {label.get("data-font-family") for label in labels} == {"Arial"}
    for method in meta["methods"]:
        assert sum(label.get("data-label") == method for label in labels) == 5
    assert len(svg.findall(f".//{{{figures.SVG}}}text")) == 0


def test_schema_preserves_original_nontext_vector_artwork(tmp_path):
    target = tmp_path / "source.svg"
    subprocess.run(
        ["pdftocairo", "-svg", str(figures.ROOT / "manuscript/figures/teaser.pdf"), str(target)],
        check=True,
    )
    original = ET.parse(target).getroot()
    for parent in original.iter():
        for child in list(parent):
            if child.get(f"{{{figures.XLINK}}}href", "").startswith("#glyph-") or child.get(
                "id", ""
            ).startswith("glyph-"):
                parent.remove(child)
    actual = figures.architecture_figure()
    actual.remove(actual.find(f"{{{figures.SVG}}}metadata"))
    labels = actual.find(".//*[@id='poster-typography']")
    assert labels is not None
    for text in (
        "Datasets",
        "Preprocessing",
        "Classifiers:",
        "LR, MLP",
        "GLANCE, TCREx",
        "L1, L2,",
        "MAD",
        "Reports & Visualisations",
    ):
        assert any(label.get("data-label") == text for label in labels)
    actual.remove(labels)
    assert [ET.tostring(child) for child in actual] == [ET.tostring(child) for child in original]


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
