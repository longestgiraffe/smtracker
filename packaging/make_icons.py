"""Make the application's icon files from the logo's SVG sources in `packaging/logo/`.

    py packaging/make_icons.py

Writes into `src/smtracker/resources/`:
- `smtracker.png`, the app icon at 512 px, and `smtracker-16.png` ... `smtracker-48.png`, the same frames the
  `.ico` holds, so Qt's window icon has a drawn frame for every small size instead of a downscale;
- `smtracker.ico`, the Windows icon at 16-256 px. The 16 and 24 px frames come from
  `smtracker-app-icon-small.svg`, the same icon with the mark larger and its lines thicker: at those sizes the
  original's track is thinner than a pixel and reads as a grey smear;
- `smtracker-logo.png` and `smtracker-logo-dark.png`, the wordmark for light and dark backgrounds (the first
  window's header).

Run it again whenever the logo changes. It needs PySide6 (QtSvg) and Pillow, both in the dev environment.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "packaging" / "logo"
OUT = ROOT / "src" / "smtracker" / "resources"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
SMALL = (16, 24)
#: The small frames also written as PNGs for Qt (`appicon.SMALL_SIZES` reads the same list).
PNG_FRAMES = (16, 24, 32, 48)
#: The wordmark's height in pixels: twice what the header draws, for high-DPI screens.
WORDMARK_HEIGHT = 160


def render(svg: Path, width: int, height: int):
    """The SVG drawn at ``width`` x ``height`` as a Pillow RGBA image."""
    from PIL import Image
    from PySide6.QtCore import QBuffer, QRectF, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    _app = QGuiApplication.instance() or QGuiApplication([])
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise SystemExit(f"cannot read {svg}")
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA")


def wordmark(svg: Path):
    from PySide6.QtSvg import QSvgRenderer

    size = QSvgRenderer(str(svg)).defaultSize()
    width = round(WORDMARK_HEIGHT * size.width() / size.height())
    return render(svg, width, WORDMARK_HEIGHT)


def main() -> None:
    icon, small = LOGO / "smtracker-app-icon.svg", LOGO / "smtracker-app-icon-small.svg"
    render(icon, 512, 512).save(OUT / "smtracker.png", optimize=True)
    frames = [render(small if s in SMALL else icon, s, s) for s in ICO_SIZES]
    frames[-1].save(OUT / "smtracker.ico", format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=frames[:-1])
    for size, frame in zip(ICO_SIZES, frames):
        if size in PNG_FRAMES:
            frame.save(OUT / f"smtracker-{size}.png", optimize=True)
    wordmark(LOGO / "smtracker-logo.svg").save(OUT / "smtracker-logo.png", optimize=True)
    wordmark(LOGO / "smtracker-logo-dark.svg").save(OUT / "smtracker-logo-dark.png", optimize=True)
    names = ["smtracker.png", "smtracker.ico", "smtracker-logo.png", "smtracker-logo-dark.png"]
    for name in names + [f"smtracker-{s}.png" for s in PNG_FRAMES]:
        print(f"{name:26s} {(OUT / name).stat().st_size:8d} bytes")


if __name__ == "__main__":
    main()
