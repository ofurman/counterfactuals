"""Decode the logo-bearing QR from actual PDF pixels at preview and print resolution."""

import json
import math
import subprocess
import tempfile
from pathlib import Path

import zxingcpp
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
DELIVERABLES = PROJECT / "deliverables"


def main():
    layout = json.loads((DELIVERABLES / "audit-layout.json").read_text())
    expected = json.loads((PROJECT.parent / "research/identity.json").read_text())["qr"]["url"]
    qr = layout["qrBranding"]["symbol"]
    canvas = layout["canvas"]
    scratch = ROOT / "tmp/pdfs"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qr-decode-", dir=scratch) as temporary:
        for dpi in (72, 150):
            scale = layout["typography"]["pointsPerPixel"] * dpi / 72
            x = math.floor((qr["left"] - canvas["left"]) * scale) - 8
            y = math.floor((qr["top"] - canvas["top"]) * scale) - 8
            width = math.ceil(qr["width"] * scale) + 16
            height = math.ceil(qr["height"] * scale) + 16
            prefix = Path(temporary) / f"qr-{dpi}"
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    "1",
                    "-singlefile",
                    "-png",
                    "-r",
                    str(dpi),
                    "-x",
                    str(x),
                    "-y",
                    str(y),
                    "-W",
                    str(width),
                    "-H",
                    str(height),
                    str(DELIVERABLES / "cel-benchmark-poster-a1.pdf"),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
            )
            with Image.open(prefix.with_suffix(".png")) as image:
                matches = zxingcpp.read_barcodes(image.convert("RGB"))
            if len(matches) != 1 or not matches[0].valid or matches[0].text != expected:
                raise AssertionError(f"QR did not decode to the repository at {dpi} DPI: {matches}")
            if matches[0].ec_level != "H":
                raise AssertionError(f"QR must use high error correction: {matches[0].ec_level}")
            print(f"QR decode passed: {dpi} DPI, error correction H, destination={expected}")


if __name__ == "__main__":
    main()
