"""The application icon and the wordmark, in one place, applied wherever a Qt application is made.

Why a module for one picture. Nine sites construct a ``QApplication`` (`app.py` three
times, `browser`, `launcher`, `prep_gui`, `tracking_gui`, `trackviewer`) and an icon set
at eight of them is an icon missing from one window. Which window would depend on how
the user got there, which is the least debuggable kind of inconsistency.

The artwork (13 Nov 2026) is the SMTracker logo: a cell outline holding a track and a
localisation. Its sources are the SVGs in `packaging/logo/`, and `packaging/make_icons.py`
writes every file here from them: the 512 px icon, its 16-48 px frames (`SMALL_SIZES`, drawn
with a larger mark and thicker lines at 16 and 24 px, where the original's track is thinner
than a pixel), the multi-size `.ico` for the Windows executable, and the wordmark for light
and dark backgrounds (`logo_path`).

Failing to find a file is not an error. A missing icon is cosmetic; refusing to open a
window over it would turn a cosmetic problem into an outage, and the frozen build is
exactly where a data file goes missing. `apply` returns whether it worked, so a caller
that cares can say so, and `icon_path` and `logo_path` return ``None`` rather than raising.
"""
from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["ICON_STEM", "SMALL_SIZES", "APP_USER_MODEL_ID", "icon_path", "logo_path", "logo_pixmap", "qicon", "apply", "apply_to_window"]

#: Both forms ship: `.ico` carries every size in one file so the Windows shell picks
#: rather than downscales, and `.png` is what Qt and the macOS/Linux paths read.
ICON_STEM = "smtracker"
_RESOURCES = Path(__file__).resolve().parent / "resources"


def icon_path(prefer: str = ".png") -> Path | None:
    """The icon file, or ``None`` if it did not ship.

    ``prefer`` is tried first, then the other form, so a caller that wants the multi-size
    `.ico` for a Windows build gets it and everything else gets the PNG.
    """
    order = [prefer] + [s for s in (".png", ".ico") if s != prefer]
    for suffix in order:
        candidate = _RESOURCES / f"{ICON_STEM}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def logo_path(dark: bool = False) -> Path | None:
    """The wordmark for a light background (or, ``dark``, for a dark one), or ``None`` if it did not ship."""
    candidate = _RESOURCES / f"{ICON_STEM}-logo{'-dark' if dark else ''}.png"
    return candidate if candidate.is_file() else None


def logo_pixmap(dark: bool, height: int, ratio: float = 1.0):
    """The wordmark as a ``QPixmap`` ``height`` logical pixels tall, sharp at device pixel ``ratio``; ``None``
    where it did not ship or does not load."""
    path = logo_path(dark)
    if path is None:
        return None
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QPixmap

    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return None
    ratio = max(1.0, float(ratio or 1.0))
    scaled = pixmap.scaledToHeight(round(height * ratio), Qt.SmoothTransformation)
    scaled.setDevicePixelRatio(ratio)
    return scaled


#: The small frames shipped beside the 512 px icon (`packaging/make_icons.py` writes them).
SMALL_SIZES = (16, 24, 32, 48)


def qicon():
    """A ``QIcon`` holding the 512 px icon and every small frame, or ``None`` where none loads.
    Qt then draws a title bar or taskbar icon from the frame drawn for that size, not a downscale."""
    path = icon_path()
    if path is None:
        return None
    from qtpy.QtGui import QIcon

    icon = QIcon(str(path))
    for size in SMALL_SIZES:
        frame = _RESOURCES / f"{ICON_STEM}-{size}.png"
        if frame.is_file():
            icon.addFile(str(frame))
    return None if icon.isNull() else icon


#: The taskbar identity (Windows): without one, a run from source groups under python.exe and shows Python's icon.
APP_USER_MODEL_ID = "SMTracker.SMTracker.3"


def _windows_app_id() -> None:
    """Give this process its own taskbar identity, so Windows shows the window's icon on the taskbar (a frozen
    build has its own executable icon either way). Declared argtypes: ctypes passes an undeclared argument as a
    32-bit int (CLAUDE.md's trap). Cosmetic: any failure is ignored."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        call = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        call.argtypes = [wintypes.LPCWSTR]
        call.restype = ctypes.c_long
        call(APP_USER_MODEL_ID)
    except Exception:                                  # noqa: BLE001 - cosmetic only
        pass


def apply(app=None) -> bool:
    """Set the window icon for the whole application. Returns whether it worked.

    Safe to call repeatedly and safe to call before any window exists, Qt applies the
    application icon to windows created afterwards.

    IT ALSO GIVES QT'S NATIVE TOOLTIP ITS COLOURS (`ui.theme.native_tooltips`), because
    this is the one call every door makes once its application exists, and
    `test_appicon` finds any door that does not. Left to Windows, the native tooltip read
    as a black box with no visible text on one machine and fine on the next. Done first
    and on its own, so a missing icon cannot cost it.
    """
    try:
        from .ui.theme import native_palette, native_tooltips

        # THE PALETTE FIRST, then the tooltip's own colours on top of it. Both are here
        # for the same reason: nothing set them, so the DESKTOP did, and two machines
        # running one build disagreed. The classic windows (mesh editor, trainer, the old
        # segmentation window) use plain QLabel / QComboBox / QDoubleSpinBox, so a
        # dark-mode desktop gave them white text inside light furniture -- which is how
        # the editor's "width px" number became invisible.
        native_palette()
        native_tooltips()
    except Exception:                                  # noqa: BLE001 - cosmetic only
        pass
    if icon_path() is None:
        return False
    _windows_app_id()
    try:
        from qtpy.QtWidgets import QApplication

        if app is None:
            app = QApplication.instance()
        if app is None:
            return False
        icon = qicon()
        if icon is None:
            return False
        app.setWindowIcon(icon)
        return True
    except Exception:                                  # noqa: BLE001 - cosmetic only
        # A missing Qt, a broken image plugin, a headless run: none of these is a reason
        # to stop a window opening. See the module head.
        return False


def apply_to_window(window) -> bool:
    """Set the icon on ONE window, for the case where no application-wide icon is
    wanted. Returns whether it worked, on the same terms as `apply`."""
    if icon_path() is None or window is None:
        return False
    try:
        icon = qicon()
        if icon is None:
            return False
        window.setWindowIcon(icon)
        return True
    except Exception:                                  # noqa: BLE001 - cosmetic only
        return False
