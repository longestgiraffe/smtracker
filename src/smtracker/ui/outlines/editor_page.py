"""The mesh editor as a page of the outlines window, in the workspace's style.

EVERY HANDLER IS THE CLASSIC EDITOR'S. `segmentation.editor_gui._core_class()` holds the
behaviour -- every shortcut (S select, D draw, X split, M merge, B bend, N straighten,
Del, Ctrl+Z / Ctrl+Y / Ctrl+S, 1/2/3, C, R, F, [ ], + -, the arrows, Esc), every click,
drag, wheel and save -- with no construction, and this page only builds the controls the
handlers drive, in the UI designer's widgets. So the editor behaves identically in both windows, and
the 150-odd tests that pin the classic one pin this one's behaviour too.

Where the UI designer's widgets speak a different API from the Qt ones the handlers were written
against (a segmented control instead of three checkable buttons, a painted slider
instead of a `QSlider`), a few lines of adapter translate; nothing is reimplemented.

THE HELPERS: the open tool says what to do next under the canvas, and the keys card
lists every shortcut as keys, folded or not as the user left it.

ONE CONDITION'S FIELDS. The list shows the fields of the condition picked in the sidebar,
by image name only (the sidebar already names the folder). The core still holds EVERY
field (`self._fields`), so Save all and the unsaved-edits check reach every condition;
the list maps its rows onto that (`self._visible`).

CHANNELS ARE THE LAB'S. The brightfield folder and up to three fluorescence folders are
named in the Channels dialog (`channels_dialog`), remembered per user, and each
fluorescence folder gets a view and an overlay button; keys 1, 2, 3 ... follow the
buttons. Asked for, so a lab whose folders are `BF` / `gfp` / `rfp` is not stuck with
phase/ and sum/.

SMALL SCREENS. Reported: "for smaller screen it's too big and the actual micro image is
too small", with the tint names clipped to "re ma gr cy ye". The page could not be
narrower than 1656 px (window) because the display row spelled out five tint names and
three wide sliders, so at 125 % scaling on a 1920 px screen (1536 px to Qt) the window
was squeezed BELOW its minimum and Qt clipped whatever it could. The tint is now five
colour dots (`_TintDots`) and the sliders are slimmer.

FOCUS MODE (asked for: "a full screen editor mode where the image is bigger and the
menu is disappearing/appearing when hovering"). The Focus button or F11: the window goes
full screen without its sidebar and header, the keys card and the tool rows leave, and the
image takes the room. A SLIM BAR STAYS (asked for after the first version hid every tool:
"should have some interface tools at least, and a button to leave"): the three tools, the
edits, undo and redo, the views, zoom and field stepping, Save, the field on screen by
name, and Exit focus -- icons, each with its key in the tooltip, and kept in step with the
keys (`_set_tool`, `_set_view`). Tools on the bar pins the full rows over the image.
THE FIELDS COLUMN STAYS ON THE LEFT, the image beside it (asked for: "in focus mode, left
should be list of images etc, the image itself ... a bit to the right"); it used to leave
and come back over the image at the left edge. Fields on the bar hides or shows it. Every
shortcut works throughout. F11 or Exit focus leaves.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QEvent, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QCursor, QPainter, QPen
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QListWidget,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ...segmentation.editor_gui import SHORTCUTS, _core_class
from ...segmentation.editor_project import (
    channel_views,
    field_condition,
    field_label,
    fields_of,
    normalise_channels,
    scan_fields,
)
from .. import theme
from ..theme import typography as ty
from ..widgets.controls import Button, FloatField, Segmented, Slider
from ..widgets.primitives import Card, Checkbox, Icon, discard, label

__all__ = ["EditorPage"]

EditorCore = _core_class()

TOOLS = (("select", "Select", "mouse-pointer", "S"), ("draw", "Draw", "pencil", "D"),
         ("split", "Split", "scissors", "X"))
MODES = ("imagej", "septa", "outline")


class _Pill:
    """One option of a `Segmented`, answering the `setChecked` a checkable button would."""

    def __init__(self, segmented: Segmented, index: int):
        self._seg, self._index = segmented, index

    def setChecked(self, on: bool) -> None:  # noqa: N802 - the QAbstractButton name
        if on and self._seg.current() != self._index:
            self._seg.blockSignals(True)
            self._seg.select(self._index)
            self._seg.blockSignals(False)


class _Choices:
    """A `Segmented` answering the three `QComboBox` calls the handlers make."""

    def __init__(self, segmented: Segmented, items):
        self._seg, self._items = segmented, list(items)

    def count(self) -> int:
        return len(self._items)

    def itemText(self, i: int) -> str:  # noqa: N802
        return self._items[i]

    def setCurrentText(self, text: str) -> None:  # noqa: N802
        self._seg.select(self._items.index(text))      # emits: the handler runs


class _EditorSlider(Slider):
    """The UI designer's slider with the `QSlider.setValue` the handlers call."""

    def setValue(self, value) -> None:  # noqa: N802
        self.set_value(int(value))


class _TextButton(Button):
    """The UI designer's button with the `setText` the alignment handler calls."""

    def setText(self, text: str) -> None:  # noqa: N802
        self.set_text(text)

    def set_glyph(self, name: str) -> None:
        """Swap the icon (Focus / Exit focus)."""
        for child in self.findChildren(Icon):
            child.name = name
            child.update()


class _TintDots(QWidget):
    """The overlay tint as colour dots, the chosen one ringed. Five WORDS were 340 px of
    the editor's widest row; five dots are 128. Answers `select` / `current` / `changed`
    like the `Segmented` it replaces, so `_Choices` drives it unchanged."""

    changed = Signal(int)
    DOT, GAP, PAD = 16, 9, 5

    def __init__(self, t, colours: dict, current: int = 0, parent=None):
        super().__init__(parent)
        self.t = t
        self._names = list(colours)
        self._colours = [QColor.fromRgbF(*colours[name]) for name in self._names]
        self._current = current
        n = len(self._names)
        self.setFixedSize(2 * self.PAD + n * self.DOT + (n - 1) * self.GAP, 28)
        self.setCursor(Qt.PointingHandCursor)

    def current(self) -> int:
        return self._current

    def select(self, index: int) -> None:
        if index == self._current or not 0 <= index < len(self._names):
            return
        self._current = index
        self.update()
        self.changed.emit(index)

    def dot_rect(self, index: int) -> QRect:
        top = (self.height() - self.DOT) // 2
        return QRect(self.PAD + index * (self.DOT + self.GAP), top, self.DOT, self.DOT)

    def index_at(self, pos) -> int | None:
        for i in range(len(self._names)):
            if self.dot_rect(i).adjusted(-4, -6, 4, 6).contains(pos):
                return i
        return None

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802 - Qt's name
        if e.button() == Qt.LeftButton:
            index = self.index_at(e.position().toPoint())
            if index is not None:
                self.select(index)

    def event(self, e):
        if e.type() == QEvent.ToolTip:
            index = self.index_at(e.pos())
            if index is None:
                QToolTip.hideText()
            else:
                QToolTip.showText(e.globalPos(), f"overlay tint: {self._names[index]}", self)
            return True
        return super().event(e)

    def paintEvent(self, _e) -> None:  # noqa: N802 - Qt's name
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        for i, colour in enumerate(self._colours):
            rect = self.dot_rect(i)
            p.setPen(QPen(QColor(self.t.border), 1))
            p.setBrush(colour)
            p.drawEllipse(rect)
            if i == self._current:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(self.t.foreground), 2))
                p.drawEllipse(rect.adjusted(-3, -3, 3, 3))
        p.end()


class _Shell(QFrame):
    """A floating panel over the image in focus mode: the card's fill and border, so
    the controls stay legible on top of a phase image."""

    def __init__(self, t, parent):
        super().__init__(parent)
        self.setObjectName("focusShell")
        self.setStyleSheet(f"QFrame#focusShell {{ background:{t.card};"
                           f" border:1px solid {t.border};"
                           f" border-radius:{t.radius_md:g}px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(0)
        self.hide()


class _Status:
    """The status line: `showMessage` and `setToolTip`, as the status bar answered."""

    def __init__(self, line):
        self._line = line

    def showMessage(self, text: str) -> None:  # noqa: N802
        self._line.setText(text)

    def setToolTip(self, text: str) -> None:  # noqa: N802
        self._line.setToolTip(text)


def _list_css(t) -> str:
    return (f"QListWidget {{ background:{t.card}; color:{t.foreground}; border:none;"
            f" outline:0; }}"
            f"QListWidget::item {{ padding:4px 6px; border-radius:{t.radius_md:g}px; }}"
            f"QListWidget::item:selected {{ background:{t.accent}; color:{t.foreground}; }}"
            f"QListWidget::item:hover {{ background:{t.muted}; }}")


class EditorPage(EditorCore, QWidget):
    """The editor, built in the UI designer's widgets on the classic editor's handlers."""

    #: Outlines were written (Save field or Save all), so a window can recount them.
    saved = Signal()
    #: Whether any field holds an unsaved edit, emitted when that changes (the window's
    #: Next button waits for none).
    edits_changed = Signal(bool)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt's name
        """No: the canvas takes whatever height it is given. In its scroll area
        (`OutlinesWindow.editor_scroll`, 30 Oct 2026) a page whose height follows its width
        is laid out at its PREFERRED height for the width, 1126 px, so the canvas scrolled on
        a 1536 x 816 laptop that holds the page; answering no fills the scroll area's view
        down to the page's minimum and scrolls only below it."""
        return False

    def __init__(self, t, root=None, parent=None):
        QWidget.__init__(self, parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._init_state(None)
        #: The condition whose fields the list shows, and their indexes in `_fields`.
        self._condition_shown: str | None = None
        self._visible: list = []
        self._set_channels(normalise_channels(theme.remembered("editor_channels")))
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        outer.addLayout(self._build_left(t), 0)

        right = QVBoxLayout()
        right.setSpacing(8)
        self._right = right
        #: The tool rows as ONE widget, so focus mode can lift them over the image.
        self._controls_panel = QWidget()
        self._controls_panel.setStyleSheet("background:transparent;")
        rows = QVBoxLayout(self._controls_panel)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(8)
        rows.addLayout(self._build_tools(t))
        rows.addLayout(self._build_view(t))
        rows.addLayout(self._build_adjust(t))
        right.addWidget(self._controls_panel)
        canvas_card = Card(t, shadowed=False)
        self._canvas_card = canvas_card
        cl = QVBoxLayout(canvas_card)
        cl.setContentsMargins(6, 6, 6, 6)
        self._figure = Figure(figsize=(7.6, 7.6), layout="constrained")
        self._figure.set_facecolor(t.card)
        self._canvas = FigureCanvasQTAgg(self._figure)
        # CLICKING THE PICTURE TAKES THE KEYBOARD, as the classic window's does: without a
        # focus policy the image never had it, so after typing in Width the shortcuts stayed
        # off until some list was clicked (the UI designer, 26 Nov 2026).
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 320, not 360 (30 Oct 2026): the editor then fits a 1536 x 816 laptop's maximised window without scrolling;
        # the canvas is Expanding and takes every pixel the window has above it
        self._canvas.setMinimumHeight(320)
        self._axes = self._figure.add_subplot(111)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_move)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("draw_event", self._on_draw)
        cl.addWidget(self._canvas)
        right.addWidget(canvas_card, 1)

        status_row = QHBoxLayout()
        self._status_line = label("", ty.ui("xs"), t.muted_foreground)
        self._status_line.setWordWrap(True)
        status_row.addWidget(self._status_line, 1)
        self._hint = label("", ty.ui("xs", "medium"), t.foreground)
        status_row.addWidget(self._hint)
        right.addLayout(status_row)
        self._status = _Status(self._status_line)
        self._keys = self._keys_card(t)
        right.addWidget(self._keys)
        outer.addLayout(right, 1)
        self._build_focus(t)

        self._start()
        if root is not None:
            self._scan(root)

    # -- the left column ------------------------------------------------------------ #
    def _build_left(self, t):
        col = QVBoxLayout()
        col.setSpacing(10)
        fields = Card(t, shadowed=False)
        fl = QVBoxLayout(fields)
        fl.setContentsMargins(10, 10, 10, 10)
        fl.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label("Fields", ty.ui("sm", "semibold"), t.foreground))
        head.addStretch(1)
        self.fields_count = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        head.addWidget(self.fields_count)
        fl.addLayout(head)
        self._field_list = QListWidget()
        self._field_list.setFont(ty.ui("sm", tabular=True))
        self._field_list.setStyleSheet(_list_css(t))
        self._field_list.currentRowChanged.connect(self._open_row)
        fl.addWidget(self._field_list)
        col.addWidget(fields, 3)

        cells = Card(t, shadowed=False)
        cel = QVBoxLayout(cells)
        cel.setContentsMargins(10, 10, 10, 10)
        cel.setSpacing(6)
        cel.addWidget(label("Cells", ty.ui("sm", "semibold"), t.foreground))
        self._cell_list = QListWidget()
        self._cell_list.setFont(ty.ui("sm", tabular=True))
        self._cell_list.setStyleSheet(_list_css(t))
        self._cell_list.currentRowChanged.connect(self._select_from_list)
        self._controls["cell_list"] = self._cell_list
        cel.addWidget(self._cell_list, 1)
        self._stats = label("–", ty.ui("xs", tabular=True), t.foreground)
        self._stats.setTextFormat(Qt.RichText)
        self._controls["cell_stats"] = self._stats
        cel.addWidget(self._stats)
        col.addWidget(cells, 2)

        align = Card(t, shadowed=False)
        al = QVBoxLayout(align)
        al.setContentsMargins(10, 10, 10, 10)
        al.setSpacing(6)
        self._align_note = label("", ty.ui("xs"), t.muted_foreground)
        self._align_note.setWordWrap(True)
        self._align_button = _TextButton(t, "Shift to fluorescence", size="sm")
        self._align_button.setEnabled(False)
        self._align_button.setToolTip(
            "Move every outline by the measured offset between the outlines and the "
            "fluorescence channel. One undo step. Available only when the offset is "
            "larger than the tolerance.")
        self._align_button.clicked.connect(self._apply_offset)
        al.addWidget(self._align_button, 0, Qt.AlignLeft)
        al.addWidget(self._align_note)
        col.addWidget(align)

        saves = QHBoxLayout()
        self._controls["save_field"] = save = Button(t, "Save field", variant="default",
                                                     glyph="save")
        save.setToolTip("Ctrl+S. Writes cell_meshes/<field>.mat and its mask in cell_masks/. "
                        "The automatic outlines are kept once in cell_meshes_auto/.")
        save.clicked.connect(self._save)
        saves.addWidget(save)
        self._controls["save_all"] = save_all = Button(t, "Save all")
        save_all.setToolTip("Save every field edited in this session.")
        save_all.clicked.connect(self._save_all)
        saves.addWidget(save_all)
        col.addLayout(saves)
        holder = QWidget()
        holder.setLayout(col)
        holder.setFixedWidth(270)
        wrap = QVBoxLayout()
        wrap.addWidget(holder)
        self._left_holder, self._left_wrap = holder, wrap
        return wrap

    # -- the rows above the canvas ------------------------------------------------- #
    def _build_tools(self, t):
        row = QHBoxLayout()
        row.setSpacing(8)
        self._tool_seg = Segmented(t, [name for _k, name, _g, _s in TOOLS],
                                   glyphs=[g for _k, _n, g, _s in TOOLS],
                                   tips=[f"{n} ({s})" for _k, n, _g, s in TOOLS], size="sm")
        self._tool_seg.changed.connect(lambda i: self._set_tool(TOOLS[i][0]))
        self._tool_buttons = {k: _Pill(self._tool_seg, i) for i, (k, *_r) in enumerate(TOOLS)}
        for i, (k, *_r) in enumerate(TOOLS):
            self._controls[f"tool_{k}"] = self._tool_buttons[k]
        row.addWidget(self._tool_seg)
        for key, text, glyph, slot, keys in (
                ("merge", "Merge", "merge", self._merge, "M"),
                ("delete", "Delete", "trash", self._delete, "Del"),
                ("add_bend", "Add bend", "spline", self._add_bend, "B"),
                ("remove_bend", "Straighten", "minus", self._remove_bend, "N")):
            b = Button(t, text, size="sm", glyph=glyph)
            b.setToolTip(f"{text} ({keys})")
            b.clicked.connect(slot)
            row.addWidget(b)
            self._controls[key] = b
        row.addStretch(1)
        for key, text, glyph, slot, keys in (("undo", "Undo", "undo", self._undo, "Ctrl+Z"),
                                             ("redo", "Redo", "redo", self._redo, "Ctrl+Y")):
            b = Button(t, text, size="sm", glyph=glyph)
            b.setToolTip(f"{text} ({keys})")
            b.clicked.connect(slot)
            row.addWidget(b)
            self._controls[key] = b
        return row

    def _build_view(self, t):
        rows = QVBoxLayout()
        rows.setSpacing(8)
        # The view buttons on a row of their own: three per fluorescence channel would
        # not fit beside the stretch, width and zoom controls.
        self._views_row = QHBoxLayout()
        self._views_row.setSpacing(8)
        self._view_seg = None
        self._rebuild_views()
        channels = Button(t, "Channels…", size="sm", glyph="layers")
        channels.setToolTip("Name the brightfield folder and up to three fluorescence "
                            "folders (sum, gfp, rfp ...). They are remembered for the next session.")
        channels.clicked.connect(self.open_channels)
        self._controls["channels"] = channels
        self._views_row.addWidget(channels)
        self._views_row.addStretch(1)
        # On THIS row, not the tools row: that one is the editor's widest already.
        self.focus_button = _TextButton(t, "Focus", size="sm", glyph="expand")
        self.focus_button.setToolTip(
            "Full-screen editing (F11). The image fills the screen; the tools appear "
            "at the top edge and the fields at the left edge.")
        self.focus_button.clicked.connect(lambda: self.set_focus(not self._focus))
        self._controls["focus"] = self.focus_button
        self._views_row.addWidget(self.focus_button)
        rows.addLayout(self._views_row)
        row = QHBoxLayout()
        row.setSpacing(8)
        rows.addLayout(row)
        row.addWidget(label("Stretch", ty.ui("sm", "medium"), self.t.muted_foreground))
        mode_seg = Segmented(t, list(MODES), current=MODES.index(self._mode), size="sm",
                             tips=["what ImageJ shows", "septa stand out",
                                   "outlines stand out"])
        mode_seg.changed.connect(lambda i: self._set_mode(MODES[i]))
        self._mode_box = _Choices(mode_seg, MODES)
        self._controls["stretch_mode"] = self._mode_box
        row.addWidget(mode_seg)
        row.addWidget(label("Width", ty.ui("sm", "medium"), self.t.muted_foreground))
        self._width_box = FloatField(t, 1.0, 60.0, float(self._width), step=0.5, decimals=1,
                                     suffix=" px", width=120)
        self._width_box.setToolTip("The width of a newly drawn cell and of the selected cell.")
        self._width_box.changed.connect(self._set_width)
        # ENTER or leaving the box hands the keyboard back to the picture (the classic window's
        # `editingFinished` -> `_release_focus`); the page's key filter declines every key while
        # a spin box has the focus.
        self._width_box.finished.connect(self._release_focus)
        self._controls["width"] = self._width_box
        row.addWidget(self._width_box)
        row.addStretch(1)
        for key, glyph, slot, tip in (
                ("zoom_out", "zoom-out", lambda: self._zoom(1.4), "Zoom out (-)"),
                ("zoom_in", "zoom-in", lambda: self._zoom(1 / 1.4), "Zoom in (+)"),
                ("zoom_fit", "maximize", self._fit, "Fit to the window (F)"),
                ("prev_field", "chevron-left", lambda: self._step_field(-1),
                 "Previous field (Left arrow, or the wheel)"),
                ("next_field", "chevron-right", lambda: self._step_field(+1),
                 "Next field (Right arrow, or the wheel)")):
            b = Button(t, "", size="sm", glyph=glyph)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
            self._controls[key] = b
        return rows

    # -- channels ------------------------------------------------------------------ #
    def _set_channels(self, channels: dict) -> None:
        self._channels = channels
        self._bright_folder = channels["bright"]
        self._fluor_folders = tuple(channels["fluor"])
        self._view_order = tuple(key for key, _label, _k in channel_views(channels))

    def _rebuild_views(self) -> None:
        """The view buttons for the channel set: the brightfield, then each fluorescence
        folder and its overlay. Keys 1, 2, 3 ... follow them."""
        views = channel_views(self._channels)
        keys = [key for key, _label, _k in views]
        if self._view not in keys:
            self._view = "phase"
        seg = Segmented(self.t, [lab for _k, lab, _i in views],
                        tips=[f"{lab} ({n})" for n, (_k, lab, _i) in enumerate(views, start=1)],
                        current=keys.index(self._view), size="sm")

        def picked(i):
            view = keys[i]
            before = self._view
            self._set_view(view)
            if self._view != view and before in self._view_buttons:   # refused: no image
                self._view_buttons[before].setChecked(True)

        seg.changed.connect(picked)
        for key in [k for k in self._controls if k.startswith("view_")]:
            del self._controls[key]
        self._view_buttons = {k: _Pill(seg, i) for i, k in enumerate(keys)}
        for k in keys:
            self._controls[f"view_{k}"] = self._view_buttons[k]
        old, self._view_seg = self._view_seg, seg
        if old is not None:
            self._views_row.replaceWidget(old, seg)
            discard(old)
        else:
            self._views_row.insertWidget(0, seg)
        self._rebuild_bar_views()

    def open_channels(self) -> None:
        from .channels_dialog import ChannelsDialog

        window = self.window()
        content = window.centralWidget() if hasattr(window, "centralWidget") else None
        dialog = ChannelsDialog(self.t, self._channels, content=content, parent=window)
        dialog.saved.connect(self.apply_channels)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def apply_channels(self, raw: dict) -> bool:
        """Use and remember a channel set. A new BRIGHTFIELD folder re-reads the fields
        (they are found in it), so it waits while edits are unsaved; new fluorescence
        folders only reload the images of the field on screen."""
        channels = normalise_channels(raw)
        rescan = channels["bright"].lower() != self._bright_folder.lower()
        if rescan and self.unsaved():
            self._say("Save the edited fields first: a new brightfield folder re-reads "
                      "every field.")
            return False
        try:
            theme.remember("editor_channels", channels)
        except OSError:
            import warnings

            warnings.warn("could not remember the image channels", stacklevel=2)
        self._set_channels(channels)
        self._rebuild_views()
        if rescan and self._root is not None:
            self._scan(self._root)
        elif self._index >= 0:
            self._open_index(self._index)
        names = ", ".join(channels["fluor"]) or "no fluorescence"
        self._say(f"Channels: {channels['bright']}/ with {names}")
        return True

    def _build_adjust(self, t):
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label("Tint", ty.ui("sm", "medium"), t.muted_foreground))
        colours = list(self._overlay_colours())
        tint = _TintDots(t, self._overlay_colours(), current=colours.index(self._colour))
        tint.changed.connect(lambda i: self._set_colour(colours[i]))
        self._colour_box = _Choices(tint, colours)
        self._controls["overlay_colour"] = self._colour_box
        row.addWidget(tint)
        self._value_labels = {}
        for key, text, low, high, start, suffix in (
                ("overlay_opacity", "Alpha", 0, 100, 50, " %"),
                ("brightness", "Bright", 20, 300, 100, " %"),
                ("contrast", "Contrast", 20, 400, 100, " %")):
            row.addWidget(label(text, ty.ui("sm", "medium"), t.muted_foreground))
            bar = _EditorSlider(t, low, high, start)
            bar.setMinimumWidth(64)
            readout = label(f"{start}{suffix}", ty.ui("xs", tabular=True), t.muted_foreground)
            readout.setMinimumWidth(34)
            bar.changed.connect(lambda value, lbl=readout, sfx=suffix: (
                lbl.setText(f"{value}{sfx}"), self._redraw_image()))
            row.addWidget(bar, 1)
            row.addWidget(readout)
            self._controls[key] = bar
            self._value_labels[key] = (readout, suffix)
        self._opacity = self._controls["overlay_opacity"]
        self._bright = self._controls["brightness"]
        self._contrast = self._controls["contrast"]
        self._controls["reset_display"] = reset = Button(t, "Reset", size="sm")
        reset.setToolTip("Brightness and contrast back to 100 % (R)")
        reset.clicked.connect(self._reset_display)
        row.addWidget(reset)
        self._controls["open_folder"] = None        # the window's header opens folders
        return row

    # -- the keys card: every shortcut, as keys ----------------------------------------- #
    def _keys_card(self, t):
        card = Card(t, shadowed=False)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label("Keys", ty.ui("sm", "semibold"), t.foreground))
        head.addStretch(1)
        shown = theme.remembered("outlines_keys")
        self.keys_toggle = Checkbox(t, checked=True if shown is None else bool(shown))
        head.addWidget(self.keys_toggle)
        head.addWidget(label("Show", ty.ui("sm"), t.muted_foreground))
        lay.addLayout(head)
        self.keys_body = QWidget()
        self.keys_body.setStyleSheet("background:transparent;")
        grid = QGridLayout(self.keys_body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        per_column = (len(SHORTCUTS) + 3) // 4
        for i, (keys, what, _control) in enumerate(SHORTCUTS):
            if keys == "1 / 2 / 3":
                keys, what = "1 2 3", "the view buttons, in order"
            cell = QHBoxLayout()
            cell.setSpacing(6)
            chip = label(keys, ty.mono("xs"), t.foreground)
            chip.setStyleSheet(f"background:{t.muted}; color:{t.foreground};"
                               f" border:1px solid {t.border}; border-radius:4px;"
                               f" padding:1px 5px;")
            cell.addWidget(chip)
            cell.addWidget(label(what, ty.ui("xs"), t.muted_foreground))
            cell.addStretch(1)
            grid.addLayout(cell, i % per_column, i // per_column)
        lay.addWidget(self.keys_body)
        self.keys_body.setVisible(self.keys_toggle.checked)
        self.keys_toggle.toggled.connect(self._keys_toggled)
        return card

    def _keys_toggled(self, on: bool) -> None:
        self.keys_body.setVisible(bool(on))
        try:
            theme.remember("outlines_keys", bool(on))
        except OSError:
            import warnings

            warnings.warn("could not remember the keys card's fold", stacklevel=2)

    # -- focus mode ------------------------------------------------------------------ #
    #: Reveal ticks of grace before a panel hides again, so it does not flicker away
    #: while the pointer crosses its border.
    HIDE_AFTER = 4

    def _build_focus(self, t) -> None:
        self._focus = False
        self._top_shell = _Shell(t, self._canvas_card)
        self._idle = {"top": 0}
        #: The bar's Tools pins the rows over the image; its Fields shows the fields column.
        self._pinned = {"top": False, "left": True}
        self._reveal = QTimer(self)
        self._reveal.setInterval(120)
        self._reveal.timeout.connect(self._reveal_tick)
        self._bar = self._build_bar(t)
        # In the column FIRST, hidden after: a widget hidden or shown while detached is a
        # window of its own (`primitives.discard`'s note).
        self._right.insertWidget(0, self._bar)
        self._bar.hide()

    def _build_bar(self, t) -> QWidget:
        """The slim row focus mode keeps: icons with their keys in the tooltips."""
        bar = QWidget()
        bar.setStyleSheet("background:transparent;")
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._bar_tools = Segmented(t, ["" for _ in TOOLS], glyphs=[g for _k, _n, g, _s in TOOLS],
                                    tips=[f"{n} ({s})" for _k, n, _g, s in TOOLS], size="sm",
                                    current=[k for k, *_r in TOOLS].index(self._tool)
                                    if self._tool in [k for k, *_r in TOOLS] else 0)
        self._bar_tools.changed.connect(lambda i: self._set_tool(TOOLS[i][0]))
        row.addWidget(self._bar_tools)
        self._bar_buttons = {}

        def icon_button(key, glyph, tip, slot):
            b = Button(t, "", size="sm", glyph=glyph)
            # SQUARE: an icon button's own width is a text button's padding around nothing
            # (about 55 px), and thirteen of them squeezed the view names at 1536 px.
            b.setFixedWidth(34)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            self._bar_buttons[key] = b
            row.addWidget(b)

        for key, glyph, tip, slot in (
                ("merge", "merge", "Merge with the nearest neighbour (M)", self._merge),
                ("delete", "trash", "Delete the selected cell (Del)", self._delete),
                ("add_bend", "spline", "Add a bend (B)", self._add_bend),
                ("remove_bend", "minus", "Straighten (N)", self._remove_bend),
                ("undo", "undo", "Undo (Ctrl+Z)", self._undo),
                ("redo", "redo", "Redo (Ctrl+Y)", self._redo)):
            icon_button(key, glyph, tip, slot)
        self._bar_views_row = row
        self._bar_views = None
        self._rebuild_bar_views()
        row.addStretch(1)
        self._bar_field = label("", ty.ui("sm", "medium"), t.foreground)
        self._bar_field.setToolTip("The field on screen. Use the arrows or the wheel to step "
                                   "through the condition's fields.")
        row.addWidget(self._bar_field)
        row.addStretch(1)
        for key, glyph, tip, slot in (
                ("zoom_out", "zoom-out", "Zoom out (-)", lambda: self._zoom(1.4)),
                ("zoom_in", "zoom-in", "Zoom in (+)", lambda: self._zoom(1 / 1.4)),
                ("zoom_fit", "maximize", "Fit to the window (F)", self._fit),
                ("prev_field", "chevron-left", "Previous field (Left arrow)",
                 lambda: self._step_field(-1)),
                ("next_field", "chevron-right", "Next field (Right arrow)",
                 lambda: self._step_field(+1))):
            icon_button(key, glyph, tip, slot)
        self.bar_tools_toggle = _TextButton(t, "Tools", size="sm", glyph="sliders-horizontal")
        self.bar_tools_toggle.setToolTip("Show the tool rows over the image: stretch, width, "
                                         "tint, brightness and channels.")
        self.bar_tools_toggle.clicked.connect(lambda: self.pin("top", not self._pinned["top"]))
        row.addWidget(self.bar_tools_toggle)
        self.bar_fields_toggle = _TextButton(t, "Fields", size="sm", glyph="list")
        self.bar_fields_toggle.setToolTip("Show or hide the fields and cells column on the "
                                          "left of the image.")
        self.bar_fields_toggle.clicked.connect(
            lambda: self.pin("left", not self._pinned["left"]))
        row.addWidget(self.bar_fields_toggle)
        icon_button("save_field", "save", "Save this field (Ctrl+S)", self._save)
        self.exit_focus_button = Button(t, "Exit focus", size="sm", variant="default",
                                        glyph="shrink")
        self.exit_focus_button.setToolTip("Back to the full editor (F11)")
        self.exit_focus_button.clicked.connect(lambda: self.set_focus(False))
        row.addWidget(self.exit_focus_button)
        return bar

    def _rebuild_bar_views(self) -> None:
        """The view pills on the bar, rebuilt with the channel set like the main row's."""
        if not hasattr(self, "_bar_views_row"):
            return
        views = channel_views(self._channels)
        keys = [key for key, _label, _k in views]
        seg = Segmented(self.t, [lab for _k, lab, _i in views],
                        tips=[f"{lab} ({n})" for n, (_k, lab, _i) in enumerate(views, start=1)],
                        current=keys.index(self._view) if self._view in keys else 0,
                        size="sm")

        def picked(i):
            self._set_view(keys[i])

        seg.changed.connect(picked)
        old, self._bar_views = self._bar_views, seg
        self._bar_view_keys = keys
        if old is not None:
            self._bar_views_row.replaceWidget(old, seg)
            discard(old)
        else:
            self._bar_views_row.addWidget(seg)

    def _set_tool(self, name: str) -> None:
        EditorCore._set_tool(self, name)
        bar = getattr(self, "_bar_tools", None)
        if bar is not None:
            keys = [k for k, *_r in TOOLS]
            bar.blockSignals(True)
            bar.select(keys.index(name))
            bar.blockSignals(False)

    def _set_view(self, view: str) -> None:
        EditorCore._set_view(self, view)
        bar = getattr(self, "_bar_views", None)
        if bar is not None and self._view in self._bar_view_keys:
            # A refused view (no such image) leaves `_view` alone: the bar follows it back.
            bar.blockSignals(True)
            bar.select(self._bar_view_keys.index(self._view))
            bar.blockSignals(False)

    def _refresh(self, *, full: bool = False, fit: bool = False, lists: bool = True) -> None:
        EditorCore._refresh(self, full=full, fit=fit, lists=lists)
        if lists and getattr(self, "_bar_field", None) is not None:
            self._bar_field.setText(self._bar_field_text())
        document = getattr(self, "_document", None)
        dirty = bool(getattr(document, "modified", False)) or any(
            d.modified for d in getattr(self, "_documents", {}).values())
        if dirty != getattr(self, "_dirty", False):
            self._dirty = dirty
            self.edits_changed.emit(dirty)

    def _bar_field_text(self) -> str:
        if not 0 <= self._index < len(self._fields):
            return ""
        field = self._fields[self._index]
        n = len(self._document.cells)
        mark = "  · unsaved" if self._document.modified else ""
        return (f"{field_condition(field, self._root)} / {Path(field['phase']).stem}"
                f"  · {n} cell{'s' if n != 1 else ''}{mark}")

    def pin(self, which: str, on: bool) -> None:
        """"top": pin every tool row over the image, or not. "left": show the fields column
        beside the image, or hide it (in focus mode; outside it the column always shows)."""
        self._pinned[which] = bool(on)
        button = self.bar_tools_toggle if which == "top" else self.bar_fields_toggle
        button.set_text(("Hide tools" if which == "top" else "Hide fields") if on
                        else ("Tools" if which == "top" else "Fields"))
        if which == "left":
            self._left_holder.setVisible(on or not self._focus)
            return
        if on:
            self._idle["top"] = 0
            self._place_shells()
            self._top_shell.show()
            self._top_shell.raise_()
        else:
            self._top_shell.hide()

    @property
    def focused(self) -> bool:
        return self._focus

    def set_focus(self, on: bool) -> None:
        """Enter or leave focus mode (see the module docstring)."""
        on = bool(on)
        if on == self._focus:
            return
        self._focus = on
        window = self.window()
        if hasattr(window, "set_focus_mode"):
            window.set_focus_mode(on)
        top = self._top_shell.layout()
        if on:
            self._right.removeWidget(self._controls_panel)
            top.addWidget(self._controls_panel)
            self._keys.hide()
            self._bar.show()
            self._bar_field.setText(self._bar_field_text())
            self.focus_button.set_text("Exit focus")
            self.focus_button.set_glyph("shrink")
            self.pin("left", True)               # the fields stay, on the left of the image
            self._reveal.start()
            self._say("Focus mode: the bar holds the tools. Tools shows the full rows, Fields "
                      "shows or hides the fields column, and Exit focus or F11 leaves.")
        else:
            self._reveal.stop()
            self.pin("top", False)
            top.removeWidget(self._controls_panel)
            self._right.insertWidget(self._right.indexOf(self._bar) + 1, self._controls_panel)
            self._bar.hide()
            self._keys.show()
            self.focus_button.set_text("Focus")
            self.focus_button.set_glyph("expand")
            self.pin("left", True)
            self._say("Focus mode off.")
        # Added to its layout FIRST, shown after: a widget shown while detached opens as a
        # window of its own (`primitives.discard`'s note).
        self._controls_panel.show()

    def _place_shells(self) -> None:
        card = self._canvas_card
        hint = self._controls_panel.sizeHint().height() + 18
        self._top_shell.setGeometry(8, 8, max(200, card.width() - 16), hint)

    def _reveal_tick(self) -> None:
        if self._focus and self.isVisible():
            self.reveal_at(self.mapFromGlobal(QCursor.pos()),
                           QApplication.mouseButtons() != Qt.NoButton)

    def reveal_at(self, pos, held: bool = False) -> None:
        """Keep the PINNED tool rows over the image for a pointer at ``pos``.

        The rows come only by their button: the bar sits along the top, so a top-edge
        hover would pull them over the image every time the bar was used. The fields
        column no longer comes at the left edge: it is a column beside the image."""
        if self._pinned["top"]:
            self._show(self._top_shell, "top", True)

    def _show(self, shell, key: str, wanted: bool) -> None:
        if wanted:
            self._idle[key] = 0
            if not shell.isVisible():
                self._place_shells()
                shell.show()
                shell.raise_()
        elif shell.isVisible():
            self._idle[key] += 1
            if self._idle[key] >= self.HIDE_AFTER:
                shell.hide()

    def resizeEvent(self, e) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(e)
        if self._focus:
            self._place_shells()

    # -- behaviour the page adds ----------------------------------------------------- #
    def eventFilter(self, obj, event):  # noqa: N802 - Qt's name
        # ONLY WHILE THIS PAGE IS ON SCREEN: the filter sits on the application, and a
        # letter typed on the Segment page must not switch the editor's tool.
        if not self.isVisible():
            return False
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_F11:
            self.set_focus(not self._focus)
            return True
        return EditorCore.eventFilter(self, obj, event)

    @property
    def bright_folder(self) -> str:
        """The folder outlines are drawn on (the Channels dialog's), for a scan run elsewhere."""
        return self._bright_folder

    def _scan(self, root) -> None:
        """Every field under ``root`` into the core; the list then shows one condition's."""
        self.set_fields(root, scan_fields(root, bright=self._bright_folder) if root else [])

    def set_fields(self, root, fields: list) -> None:
        """``scan_fields``' result for ``root``, read elsewhere (the window reads it on a worker
        thread: a large tree froze the window while it was read, 15 Oct 2026)."""
        self._root = Path(root) if root else None
        self._fields = list(fields)
        self._documents.clear()
        self._index = -1
        self._show_condition(self._condition_shown)
        if not self._fields:
            self._say(f"No {self._bright_folder}/ folder with images under {root}.")

    def _show_condition(self, condition) -> None:
        self._stash()
        if condition is None and self._fields:
            condition = field_condition(self._fields[0], self._root)
        self._condition_shown = condition
        self._visible = fields_of(self._fields, condition, self._root)
        self._field_list.blockSignals(True)
        self._field_list.clear()
        for i in self._visible:
            self._field_list.addItem(field_label(self._fields[i]))
        self._field_list.blockSignals(False)
        self._count()
        if self._visible:
            self._field_list.setCurrentRow(0)
            if self._index != self._visible[0]:
                self._open_row(0)
        elif condition is not None and self._fields:
            self._say(f"{condition}: no {self._bright_folder}/ images to outline.")

    def _count(self) -> None:
        n = len(self._visible)
        done = sum(1 for i in self._visible if self._fields[i].get("has_mat"))
        self.fields_count.setText(f"{done}/{n} outlined" if n else "")

    def _relabel(self) -> None:
        """After a save: the marks and the count follow the files."""
        for row, i in enumerate(self._visible):
            item = self._field_list.item(row)
            if item is not None:
                item.setText(field_label(self._fields[i]))
        self._count()

    def _open_row(self, row: int) -> None:
        if 0 <= row < len(self._visible):
            self._open_index(self._visible[row])

    def _step_field(self, delta: int) -> None:
        pos = self._visible.index(self._index) if self._index in self._visible else -1
        target = pos + delta
        if 0 <= target < len(self._visible):
            self._field_list.setCurrentRow(target)

    def _save(self) -> None:
        EditorCore._save(self)
        self._relabel()
        self.saved.emit()

    def _save_all(self) -> None:
        EditorCore._save_all(self)
        self._relabel()
        self.saved.emit()

    def unsaved(self) -> list:
        """Names of the fields with edits not written yet."""
        self._stash()
        return [f["name"] for f in self._fields
                if self._documents.get(str(f["mat"])) and self._documents[str(f["mat"])].modified]

    def open_condition(self, condition: str) -> None:
        """Show ``condition``'s fields (the sidebar's picker), opening its first; picking
        the condition already shown keeps the field on screen."""
        if condition == self._condition_shown and self._visible:
            return
        self._show_condition(condition)

    def shutdown(self) -> None:
        """The filter is on the application; leaving it would route every later key to a
        page that no longer exists."""
        self._reveal.stop()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
