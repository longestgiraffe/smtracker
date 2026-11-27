"""The design system: colour, type and glyphs — what a widget consumes.

Nothing here draws a control. `tokens` is the table shadcn's CSS custom
properties become, `hcl` is the colour science under the series palette and the
day ramps, `typography` builds fonts from the Tailwind scale, and `icons` holds
lucide's path data with the one call that paints it.

**Only `tokens` and `hcl` are imported eagerly, and that is load-bearing.**
Those two are the only Qt-free modules in `smtracker.ui`, which is what lets
`smtracker.ui` be imported on a machine with no display — a colour can be
resolved and a palette checked headless. `typography` and `icons` both import
Qt, so they are left for the caller to import by name::

    from ..theme import typography as ty      # the importer pays for Qt
    from ..theme import tokens                # free

Adding either to the star below would quietly end that, and the failure would
not show up here: it would show up as the analysis spine refusing to import on a
build with no Qt.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from . import hcl, tokens

__all__ = ["hcl", "tokens", "scheme", "current", "saved_scheme", "save_scheme",
           "native_tooltips", "native_palette", "remembered", "remember",
           "settings_path", "DEFAULT_SCHEME", "UI_SETTINGS_ENV"]

#: LIGHT IS THE STANDARD, DARK IS A TOGGLE -- ruled by the experimenter. The window
#: used to open in whatever the desktop reported (`scheme`), so one machine showed it
#: dark and the next light, and a screenshot depended on who took it.
DEFAULT_SCHEME = "light"

#: Environment override for the settings file, so a test never writes the real
#: user's choice -- the same pattern as `presets.PRESET_ENV`.
UI_SETTINGS_ENV = "SMTRACKER_UI_SETTINGS"


def settings_path() -> Path:
    """Where the UI's own preferences live: per-user, never in the install folder,
    beside `presets` and the segmentation models (`APPDATA/SMTracker` on Windows)."""
    override = os.environ.get(UI_SETTINGS_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "SMTracker" / "ui.json"
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "smtracker" / "ui.json"


def _read_settings() -> dict:
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # SAID, NOT SWALLOWED -- but not fatal either: a colour preference that
        # cannot be read must not stop the analysis window from opening.
        warnings.warn(f"ignoring unreadable UI settings {path}: {exc}", stacklevel=3)
        return {}
    return data if isinstance(data, dict) else {}


def saved_scheme() -> str:
    """The scheme the user last chose with the toggle, or `DEFAULT_SCHEME`."""
    value = _read_settings().get("color_scheme")
    return value if value in ("light", "dark") else DEFAULT_SCHEME


def save_scheme(color_scheme: str) -> Path:
    """Remember `"light"` or `"dark"` for every later window. Other keys in the file
    are kept, so a second preference added later is not wiped by this one."""
    if color_scheme not in ("light", "dark"):
        raise ValueError(f"color_scheme must be 'light' or 'dark', got {color_scheme!r}")
    path = settings_path()
    data = _read_settings()
    data["color_scheme"] = color_scheme
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def remembered(key: str):
    """A preference saved with `remember`, or None. Same file as the colour scheme."""
    return _read_settings().get(key)


def remember(key: str, value) -> Path:
    """Save one JSON-able preference for every later window, keeping the other keys --
    e.g. the Fiji folder a user picked or downloaded, which `fiji.find_fiji` would not
    find again on its own when it sits outside the usual places."""
    path = settings_path()
    data = _read_settings()
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def scheme() -> str:
    """``"dark"`` or ``"light"``, from the desktop rather than from a flag.

    The window this decides for opens in whatever theme the machine is in, and
    the first launcher hard-coded ``color: #666`` for its descriptions — a grey
    picked against a light background that reads as mud on the dark one the
    experimenter actually runs. Asking is the only way one build serves both.

    ASKED IN TWO WAYS, because the good one is not always answered. Qt 6.5's
    ``colorScheme()`` is the platform's own answer and is what a user's
    per-application override reaches; older Qt, and platforms that never report
    one, return ``Unknown``, and there the window's own palette still says which
    way round it is. A dark theme is the one whose window is darker than its
    text, which is true whatever the hue.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QGuiApplication, QPalette

    app = QGuiApplication.instance()
    if app is None:
        return "light"
    hints = app.styleHints()
    reported = getattr(hints, "colorScheme", None)
    if reported is not None:
        try:
            if reported() == Qt.ColorScheme.Dark:
                return "dark"
            if reported() == Qt.ColorScheme.Light:
                return "light"
        except (AttributeError, TypeError):        # no ColorScheme on this Qt
            pass
    palette = app.palette()
    window = palette.color(QPalette.Window)
    text = palette.color(QPalette.WindowText)
    return "dark" if window.lightness() < text.lightness() else "light"


def current(color_scheme: str | None = None) -> tokens.Tokens:
    """The token table for this run. ``None`` means the SAVED choice — light unless
    the toggle picked dark (see :func:`saved_scheme`). The desktop's own setting
    (:func:`scheme`) is no longer consulted: it made the same build open in two
    different themes on two machines.

    Named for what it answers rather than for what it returns, so a call site
    reads ``theme.current()``. It cannot be called ``theme``: that is this
    package, and a function of the same name in the package above it would make
    ``smtracker.ui.theme`` resolve to the function rather than to the module.
    """
    return tokens.Tokens(color_scheme=color_scheme or saved_scheme())


#: The one app-wide style-sheet rule this package owns, marked so it is replaced rather
#: than appended to on a theme switch, and so any other rule is left alone.
_TOOLTIP_RULE_MARK = "/* smtracker:native-tooltip */"


def native_palette(t: tokens.Tokens | None = None) -> None:
    """Give Qt's OWN unstyled widgets the app's colours. A no-op before a QApplication.

    NOTHING SET THE APPLICATION PALETTE, SO WINDOWS DID -- and on a machine whose desktop
    is in dark mode while this app paints light, every widget that styles itself takes
    the desktop's: MEASURED here, `Base #2d2d2d` with `Text #ffffff` under tokens of
    `background #FFFFFF` / `foreground #09090B`. White text on a light window is
    invisible, which is how the mesh editor's "width px" number disappeared (reported
    18 Oct 2026: "the number for cell width in px is not visible, maybe again a font
    bug?" -- the right instinct at the wrong layer; the font was fine).

    THE NEW UI DOES NOT NEED THIS and is unaffected: every widget under `ui/` sets its
    own colours (that is what the thin-Qt ratchet is about). It is the CLASSIC windows --
    the mesh editor, the trainer, the old segmentation window -- that use plain
    `QLabel`, `QComboBox`, `QDoubleSpinBox` and so inherit whatever the desktop says.
    They are currently painting light furniture around dark-mode text, so matching the
    theme is strictly better than leaving them to the desktop.

    Only the roles an unstyled widget reads for TEXT and for its FIELD are set, and the
    selection colours so a selected number stays legible. Nothing else is touched:
    a palette is a blunt instrument and the styled widgets must keep deciding for
    themselves.
    """
    from qtpy.QtGui import QColor, QPalette
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    t = t or current()
    palette = QPalette(app.palette())
    pairs = (
        (QPalette.Window, t.background), (QPalette.WindowText, t.foreground),
        (QPalette.Base, t.background), (QPalette.Text, t.foreground),
        (QPalette.AlternateBase, t.muted), (QPalette.Button, t.background),
        (QPalette.ButtonText, t.foreground), (QPalette.PlaceholderText, t.muted_foreground),
        (QPalette.Highlight, t.primary), (QPalette.HighlightedText, t.primary_foreground),
    )
    for role, value in pairs:
        colour = QColor(value)
        if not colour.isValid():                      # pragma: no cover - defensive
            continue
        for group in (QPalette.Active, QPalette.Inactive):
            palette.setColor(group, role, colour)
    # DISABLED TEXT MUST STILL BE READABLE AS DISABLED, not invisible: the muted
    # foreground is the token that already means "present but secondary".
    muted = QColor(t.muted_foreground)
    if muted.isValid():
        for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
            palette.setColor(QPalette.Disabled, role, muted)
    app.setPalette(palette)


def native_tooltips(t: tokens.Tokens | None = None) -> None:
    """Give Qt's OWN tooltip -- every ``setToolTip`` and ``QToolTip.showText`` -- the
    styled tooltip's colours, explicitly. A no-op before a ``QApplication`` exists.

    NOTHING SET THEM, SO WINDOWS DID, and two machines running the same build disagreed:
    the Spatial Map's "8.8 independent tracks" read white on black on one and as a black
    box with no visible text on one user's. The panel and the text came from different
    sources there (the platform style drew a dark panel while the text took the desktop's
    dark tooltip text). Set here from the same tokens the styled tooltip paints with --
    panel ``foreground``, text ``background`` -- so both kinds of tooltip look alike and
    neither depends on the desktop. The palette covers a plain style; the style-sheet
    rule wins over a platform style that draws its own panel.
    """
    from qtpy.QtGui import QColor, QPalette
    from qtpy.QtWidgets import QApplication, QToolTip

    app = QApplication.instance()
    if app is None:
        return
    t = t or current()
    panel, text = QColor(t.foreground), QColor(t.background)
    palette = QPalette(QToolTip.palette())
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role, colour in ((QPalette.ToolTipBase, panel), (QPalette.ToolTipText, text),
                             (QPalette.Window, panel), (QPalette.WindowText, text),
                             (QPalette.Text, text)):
            palette.setColor(group, role, colour)
    QToolTip.setPalette(palette)
    rule = (f"{_TOOLTIP_RULE_MARK} QToolTip {{ color: {t.background}; "
            f"background-color: {t.foreground}; border: 1px solid {t.foreground}; "
            f"padding: 4px 8px; }}")
    kept = [line for line in (app.styleSheet() or "").splitlines()
            if _TOOLTIP_RULE_MARK not in line]
    sheet = "\n".join([*kept, rule]).strip()
    if sheet != (app.styleSheet() or "").strip():
        app.setStyleSheet(sheet)
    # AND ON THE WIDGET IT IS FOR, each time one shows (22 Nov 2026, the Jump angles page's Against chance: "just shows
    # a black box in dark mode"). Qt styles a tooltip through the WIDGET it is for, and a widget's style sheet beats the
    # application's: a page's selector-less `background:...` reached every tooltip under it, so the panel took the
    # page's colour (dark in the dark theme, the text as dark: a black box with nothing in it) or, under a container's
    # `background:transparent;`, nothing at all (drawn black). Measured offscreen: neither the tooltip label's own style
    # sheet nor a rule on the window changes it; the same QToolTip rule in the style sheet of the widget the tooltip is
    # for does (the panel #FAFAFA in the dark theme, #09090B in the light). So that widget gets it, marked, as its
    # tooltip is asked for (`_TipWatch`: a hover's ToolTip event, and a tooltip a widget shows itself).
    watch = getattr(app, "_smtracker_tip_watch", None)
    if watch is None:
        watch = _tip_watch_class()(app)
        app.installEventFilter(watch)
        app._smtracker_tip_watch = watch
    watch.rule = rule


def _with_tip_rule(widget, rule: str) -> bool:
    """``widget``'s style sheet with ``rule`` (the marked QToolTip rule, `native_tooltips`) in place of any older one;
    True when it changed."""
    sheet = widget.styleSheet() or ""
    if rule in sheet:
        return False
    kept = [line for line in sheet.splitlines() if _TOOLTIP_RULE_MARK not in line]
    widget.setStyleSheet("\n".join([*kept, rule]).strip())
    return True


def _tip_watch_class():
    from qtpy.QtCore import QEvent, QObject
    from qtpy.QtWidgets import QWidget

    class _TipWatch(QObject):
        """Puts the tooltip rule on the widget a native tooltip is for, before it shows (`native_tooltips`)."""

        rule = ""

        def eventFilter(self, obj, event):                        # noqa: N802 - Qt's name
            kind = event.type()
            if not self.rule or kind not in (QEvent.ToolTip, QEvent.Show):
                return False
            try:
                if kind == QEvent.ToolTip:
                    if isinstance(obj, QWidget) and obj.toolTip():
                        _with_tip_rule(obj, self.rule)
                elif obj.metaObject().className() == "QTipLabel":
                    # a tooltip a widget shows itself (`QToolTip.showText`, a chart's hover): its widget, then the
                    # label styled again from it
                    owner = obj.property("_q_stylesheet_parent")
                    if isinstance(owner, QWidget) and _with_tip_rule(owner, self.rule):
                        obj.style().unpolish(obj)
                        obj.style().polish(obj)
                        obj.update()
            except RuntimeError:                                  # pragma: no cover - a widget being torn down
                pass
            return False

    return _TipWatch
