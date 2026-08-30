"""Check the one-slide deck, notes, figure provenance, PDF, and QR pixels."""

import base64
import hashlib
import io
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import zxingcpp
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "deliverables"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "s": "http://www.w3.org/2000/svg",
}


def main():
    identity = json.loads((ROOT / "poster/research/identity.json").read_text())
    speech = json.loads((HERE / "speech.json").read_text())
    words = sum(len(section["text"].split()) for section in speech["sections"])
    assert 220 <= words <= 260, f"Script too short or long: {words}"
    with ZipFile(OUTPUT / "cel-poster-pitch.pptx") as archive:
        slides = [
            name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ]
        assert len(slides) == 1, slides
        slide = ET.fromstring(archive.read(slides[0]))
        tiles = [
            shape
            for shape in slide.findall(".//p:sp", NS)
            if shape.find('p:spPr/a:solidFill/a:srgbClr[@val="E6F4FC"]', NS) is not None
        ]
        assert len(tiles) == 4
        for tile in tiles:
            assert tile.find('p:spPr/a:prstGeom[@prst="roundRect"]', NS) is not None
            assert tile.find('p:spPr/a:ln/a:prstDash[@val="lgDash"]', NS) is not None
        text = " ".join(node.text or "" for node in slide.findall(".//a:t", NS))
        assert identity["title"] in text
        for phrase in [
            "18",
            "14",
            "2",
            "9",
            "No single winner",
            "Validity and failures",
            "Reach versus change",
            "1 AReS · 2 GLOBE-CE",
            "3 GLANCE (one group)",
            "Bring your method to the benchmark.",
        ]:
            assert phrase in text, phrase
        assert "Wrocław University of Science and Technology" not in text
        for author in identity["authors"]:
            assert author["name"] in text
        notes = ET.fromstring(archive.read("ppt/notesSlides/notesSlide1.xml"))
        notes_text = " ".join(node.text or "" for node in notes.findall(".//a:t", NS))
        for section in speech["sections"]:
            assert section["text"] in notes_text
        assert len(slide.findall(".//p:sp", NS)) >= 15, "Slide text must be editable"
        for transform in slide.findall(".//a:xfrm", NS):
            offset, size = transform.find("a:off", NS), transform.find("a:ext", NS)
            if offset is None or size is None:
                continue
            x, y, w, h = (
                int(node.attrib[key]) / 12700
                for node, key in [(offset, "x"), (offset, "y"), (size, "cx"), (size, "cy")]
            )
            assert min(x, y) >= 0 and x + w <= 960.1 and y + h <= 540.1

    audit = json.loads((OUTPUT / "layout-audit.json").read_text())
    claims = {
        claim["id"]: claim
        for claim in json.loads(
            (ROOT / "poster/research/claims/claims.generated.json").read_text()
        )["claims"]
    }
    assert [tile["id"] for tile in audit["tiles"]] == [
        "datasets",
        "methods",
        "backbones",
        "metrics",
    ]
    for tile in audit["tiles"]:
        assert int(tile["number"]) == claims[f"scope.{tile['id']}"]["value"]["total"]
    assert [tile["label"] for tile in audit["tiles"]] == [
        "Datasets",
        "Methods",
        "Backbones / Task",
        "Counterfactual Explanations Metrics",
    ]
    assert abs(audit["header"]["titleLines"] - 1) < 0.02
    assert audit["header"]["authorsTop"] > audit["header"]["titleBottom"]
    assert audit["header"]["authorsLeft"] == audit["header"]["titleLeft"]
    assert not audit["header"]["hasAffiliation"]
    assert [result["paradigm"] for result in audit["results"]] == ["local", "global", "group"]
    assert all(result["left"] >= 640 for result in audit["results"])
    assert all(
        second["top"] >= first["bottom"]
        for first, second in zip(audit["results"], audit["results"][1:])
    )
    assert all(result["plotBottom"] <= result["bottom"] for result in audit["results"])
    assert not audit["overflow"] and not audit["broken"]
    provenance = json.loads((HERE / "assets/provenance.json").read_text())
    assert [item["paradigm"] for item in provenance["figures"]] == ["local", "global", "group"]
    for item in provenance["figures"]:
        figure = ET.parse(HERE / f"assets/{item['paradigm']}-results.svg").getroot()
        source_bytes = (ROOT / item["source"]).read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == item["sha256"]
        source_svg = ET.fromstring(source_bytes)
        groups = figure.findall("s:g", NS)
        assert [group.attrib["id"] for group in groups] == ["metric-0", "metric-1", "metric-3"]
        assert [crop["metric"] for crop in item["retainedMetrics"]] == [
            "Validity (↑)",
            "L2+Hamming (↓)",
            "Log. dens. (↑)",
        ]
        with Image.open(ROOT / item["manuscriptSource"]) as source:
            for group, crop in zip(groups, item["retainedMetrics"]):
                original_group = source_svg.find(f"s:g[@id='{group.attrib['id']}']", NS)
                assert [ET.tostring(child) for child in group] == [
                    ET.tostring(child) for child in original_group
                ], "Plot axes, labels and image geometry must be unchanged"
                image = group.find("s:image", NS)
                payload = base64.b64decode(image.attrib["href"].split(",", 1)[1])
                assert hashlib.sha256(payload).hexdigest() == crop["pngSha256"]
                with Image.open(io.BytesIO(payload)) as actual:
                    expected = source.convert("RGB").crop(crop["pixelBounds"])
                    assert actual.size == expected.size
                    assert ImageChops.difference(actual.convert("RGB"), expected).getbbox() is None

    pdf = OUTPUT / "cel-poster-pitch.pdf"
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    assert re.search(r"Pages:\s+1\b", info)
    width, height = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
    assert abs(width - 960) < 0.1 and abs(height - 540) < 0.1
    text = " ".join(subprocess.check_output(["pdftotext", str(pdf), "-"], text=True).split())
    layout_text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    assert identity["title"] in [line.strip() for line in layout_text.splitlines()]
    for phrase in [
        identity["title"],
        "No single winner",
        "Validity and failures",
        "Reach versus change",
        "Plausibility: log-density",
        "1 AReS · 2 GLOBE-CE",
        "3 GLANCE (one group)",
        "18",
        "14",
        "Backbones / Task",
        "Counterfactual Explanations Metrics",
        "Local",
        "Global",
        "Group-wise",
    ]:
        assert phrase in text, phrase
    assert "Wrocław University of Science and Technology" not in text
    for author in identity["authors"]:
        assert author["name"] in text
    with Image.open(OUTPUT / "cel-poster-pitch.png") as preview:
        for width in [1920, 1280]:
            image = preview.convert("RGB").resize((width, round(width * 9 / 16)))
            matches = zxingcpp.read_barcodes(image)
            assert len(matches) == 1 and matches[0].text == identity["links"]["repository"]
    print(
        f"PASS: one editable 16:9 slide; single-line title and authors below; four scope tiles; nine exact plot crops including plausibility across three paradigms; {words}-word notes; PDF text; QR at 1920/1280px."
    )


if __name__ == "__main__":
    main()
