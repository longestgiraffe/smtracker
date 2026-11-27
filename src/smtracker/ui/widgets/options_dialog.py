"""The options screen every stage opens from its header: a few standards, set once.

Asked for (13 Oct 2026): "an option screen for each tool (imageprep/track, segmentation,
analysis). should be not too much, just maybe some important things. maybe like a standard
setting editor for tracking".

THIN QT: which options exist, their words, ranges and defaults, validation and what
changed are `smtracker.options`; this lays them out, reads the controls back, and writes
the record (`theme.remember(OPTIONS_KEY, ...)`). The window that opened it applies the
saved standards (`saved`), moving only the values still at the old standard.

An in-window `OverlayDialog` (the UI designer's), never a `QDialog`: a modal OS dialog would block a
headless test forever.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ... import options as opts
from .. import theme
from ..theme import typography as ty
from .controls import Button, FloatField, NumberField, Segmented
from .dialogs import CloseButton, OverlayDialog
from .primitives import Checkbox, FitScroll, label, scrollbar_css
from .settings import CUT_INK

__all__ = ["OptionsDialog", "open_options"]

#: The body scrolls past this height, so the buttons stay on screen on a laptop.
BODY_MAX_H = 520
#: The card's width. FIXED, body included: a word-wrapped help line made the body wider
#: than its scroll area, which cut the controls at the right edge (seen in the first
#: render), and a fixed width is what the wrapping can follow.
CARD_W = 720
#: The room a control has beside its name: the body (the card less its margins and the
#: scrollbar) less a name column. A choice whose pills need more is a drop-down list.
CONTROL_ROOM = CARD_W - 48 - 18 - 200
#: A choice of more than this many is a drop-down list whatever its words' width: a list
#: that grows (the segmentation models, the user's own among them) must never widen the card.
MAX_PILLS = 5


class _FolderField(QFrame):
    """A folder: typed, or picked with Choose... (the system's folder dialog, opened only on
    the click). Empty means none."""

    def __init__(self, t, value: str, parent=None):
        super().__init__(parent)
        self.setObjectName("folderfield")
        self.setStyleSheet(
            f"QFrame#folderfield {{ background:{t.card}; border:1px solid {t.input};"
            f" border-radius:{t.radius_md:g}px; }}")
        self.setFixedHeight(t.control_height)
        self.edit = QLineEdit(value or "")
        self.edit.setFrame(False)
        self.edit.setFont(ty.ui("sm"))
        self.edit.setPlaceholderText("in place, beside the data")
        self.edit.setStyleSheet(f"QLineEdit {{ background:transparent; color:{t.foreground};"
                                f" border:none; padding:0px; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 4, 0)
        lay.setSpacing(6)
        lay.addWidget(self.edit, 1)
        choose = Button(t, "Choose…", size="sm", variant="ghost")
        choose.clicked.connect(self._choose)
        lay.addWidget(choose)
        self.setMinimumWidth(320)

    def _choose(self) -> None:
        got = QFileDialog.getExistingDirectory(self, "Choose a folder", self.edit.text())
        if got:
            self.edit.setText(got)

    def value(self) -> str:
        return self.edit.text().strip()

    def setValue(self, value) -> None:
        self.edit.setText(str(value or ""))


class _ChoiceBox(QComboBox):
    """A LONG choice as a drop-down list, answering the two calls a `Segmented` answers here
    (`current`, `select`). Found 3 Nov 2026 (tester B3): the outlines screen's model list
    (5 built-in, 11 saved here) as pills made the card about 5,700 px wide, with Save,
    Cancel and the close button off the screen."""

    def __init__(self, t, words, current: int, parent=None):
        super().__init__(parent)
        self.setFont(ty.ui("sm"))
        self.setStyleSheet(
            f"QComboBox {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:3px 8px; min-height:{t.control_height_sm - 8}px; }}"
            f"QComboBox QAbstractItemView {{ background:{t.card}; color:{t.foreground};"
            f" selection-background-color:{t.accent}; selection-color:{t.foreground}; }}"
            f"QComboBox::drop-down {{ border:none; width:22px; }}")
        self.addItems(list(words))
        # its width from a few dozen characters, never the longest name: a long saved
        # model's name is elided in the box and read whole in the open list
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(22)
        self.setMaximumWidth(CONTROL_ROOM)
        self.setCurrentIndex(max(0, int(current)))

    def current(self) -> int:
        return self.currentIndex()

    def select(self, index: int) -> None:
        self.setCurrentIndex(int(index))


def _step(decimals: int) -> float:
    return 1.0 if decimals <= 0 else 0.5 if decimals == 1 else 10.0 ** (1 - decimals)


class OptionsDialog(OverlayDialog):
    """One stage's options. ``choices`` gives a choice option its list when the window
    owns it (the segmentation models, the user's own included)."""

    #: The new record, written already; the window applies it.
    saved = Signal(dict)

    def __init__(self, t, tool: str, stored=None, *, choices: dict | None = None,
                 content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.tool = tool
        self._stored = dict(stored) if isinstance(stored, dict) else {}
        self._choices = dict(choices or {})
        self._card.setFixedWidth(CARD_W)

        head = QHBoxLayout()
        head.addWidget(label(opts.TOOL_TITLES[tool], ty.ui("lg", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        lead = label("The standards a new session starts from. Saving updates only the pages that still "
                     "use the old standard; a value you set on a page stays.", ty.ui("sm"), t.muted_foreground)
        lead.setWordWrap(True)
        self.card_lay.addWidget(lead)

        body = QWidget()
        body.setFixedWidth(CARD_W - 48 - 18)               # card margins, the scrollbar
        body.setStyleSheet("background:transparent;")
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 8, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        self.controls: dict = {}
        current = opts.values(tool, self._stored, choices=self._choices)
        row, group = 0, None
        for option in opts.options_for(tool):
            if option.group != group:
                group = option.group
                if row:
                    grid.setRowMinimumHeight(row, 10)
                    row += 1
                grid.addWidget(label(group, ty.ui("sm", "semibold"), t.foreground),
                               row, 0, 1, 2)
                row += 1
            name = option.label + (f"  ({option.unit})" if option.unit else "")
            grid.addWidget(label(name, ty.ui("sm", "medium"), t.foreground), row, 0)
            control = self._control(option, current[option.key])
            self.controls[option.key] = control
            grid.addWidget(control, row, 1, alignment=Qt.AlignRight)
            row += 1
            words = label(option.help, ty.ui("xs"), t.muted_foreground)
            words.setWordWrap(True)
            grid.addWidget(words, row, 0, 1, 2)
            row += 1
        # `FitScroll` (30 Oct 2026): the list up to `BODY_MAX_H` where the window has the room, a few rows where it
        # has not. Its FIXED height of 520 held the card at 687 px, so on a 1280 x 720 laptop (641 px) Save and Cancel
        # fell off the window's foot (`scripts/gui_sweep.py`, OVERFLOW).
        scroll = FitScroll()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }"
                             + scrollbar_css(t))
        scroll.viewport().setStyleSheet("background:transparent;")
        # THE CARD GROWS TO ITS WIDEST ROW: a choice keeps its pills' own width (`_control`), so five of them beside
        # their name can need more than `CARD_W` (25 Oct 2026: Uncertainty's, with Chunks, 678 px in a 654 px body)
        needed = grid.minimumSize().width()
        if needed > body.width():
            body.setFixedWidth(needed)
            self._card.setFixedWidth(needed + 48 + 18)
        body.adjustSize()
        scroll.setMaximumHeight(min(BODY_MAX_H, body.sizeHint().height() + 4))
        self.card_lay.addWidget(scroll)

        self.error = label("", ty.ui("sm"), CUT_INK)
        self.error.setWordWrap(True)
        self.error.setVisible(False)
        self.card_lay.addWidget(self.error)
        foot = QHBoxLayout()
        reset = Button(t, "Reset to defaults", size="sm", variant="ghost", glyph="undo")
        reset.clicked.connect(self.reset)
        foot.addWidget(reset)
        foot.addStretch(1)
        cancel = Button(t, "Cancel", size="sm")
        cancel.clicked.connect(self._dismiss)
        foot.addWidget(cancel)
        self.save_button = Button(t, "Save", size="sm", variant="default")
        self.save_button.clicked.connect(self.save)
        foot.addWidget(self.save_button)
        self.card_lay.addLayout(foot)

    # -- the controls ------------------------------------------------------------------ #
    def _choice_list(self, option) -> tuple:
        return tuple(self._choices.get(option.key, option.choices))

    def _control(self, option, value):
        t = self.t
        if option.kind == "switch":
            return Checkbox(t, checked=bool(value))
        if option.kind == "folder":
            return _FolderField(t, value)
        if option.kind == "choice":
            choices = self._choice_list(option)
            keys = [v for v, _w in choices]
            words = [w for _v, w in choices]
            index = keys.index(value) if value in keys else 0
            seg = Segmented(t, words, current=index, size="sm")
            if len(words) > MAX_PILLS or seg.sizeHint().width() > CONTROL_ROOM:
                seg.deleteLater()
                return _ChoiceBox(t, words, index)
            # ITS OWN WIDTH, never less (found by `scripts/gui_sweep.py`, 25 Oct 2026: with Chunks, Uncertainty's five
            # pills were squeezed in the grid's right column and "Replicates" read "Replica")
            seg.setMinimumWidth(seg.sizeHint().width())
            return seg
        low = option.low if option.low is not None else -1e9
        high = option.high if option.high is not None else 1e9
        if option.integer:
            return NumberField(t, int(low), int(high), int(value), width=132)
        return FloatField(t, float(low), float(high), float(value),
                          step=_step(option.decimals), decimals=option.decimals, width=132)

    def value(self, key: str):
        option = opts.option(key, self.tool)
        control = self.controls[key]
        if option.kind == "switch":
            return bool(control.checked)
        if option.kind == "choice":
            return self._choice_list(option)[control.current()][0]
        return control.value()                 # a number, or a folder's text

    def set_value(self, key: str, value) -> None:
        option = opts.option(key, self.tool)
        control = self.controls[key]
        if option.kind == "switch":
            control.set_checked(bool(value), animate=False)
        elif option.kind == "choice":
            keys = [v for v, _w in self._choice_list(option)]
            if value in keys:
                control.select(keys.index(value))
        elif option.kind == "folder":
            control.setValue(value)
        else:
            control.setValue(int(value) if option.integer else float(value))

    # -- the buttons ------------------------------------------------------------------- #
    def reset(self) -> None:
        """Every control back to its default. Nothing is written until Save."""
        for option in opts.options_for(self.tool):
            self.set_value(option.key, option.default)
        self.error.setVisible(False)

    def save(self) -> bool:
        """Validate every value (one refusal refuses the save, named), write the record,
        emit it and close. Returns whether it saved."""
        changes = {key: self.value(key) for key in self.controls}
        try:
            record = opts.update(self._stored, self.tool, changes, choices=self._choices)
        except opts.OptionError as exc:
            self.error.setText(f"Not saved: {exc}")
            self.error.setVisible(True)
            return False
        stored = theme.remembered(opts.OPTIONS_KEY)
        merged = dict(stored) if isinstance(stored, dict) else {}
        for option in opts.options_for(self.tool):
            merged.pop(option.key, None)
        merged.update({k: v for k, v in record.items()
                       if k in {o.key for o in opts.options_for(self.tool)}})
        theme.remember(opts.OPTIONS_KEY, merged)
        self.saved.emit(merged)
        self._dismiss()
        return True


def open_options(window, t, tool: str, *, choices: dict | None = None, on_saved=None, content=None):
    """Open ``tool``'s options over ``window``'s content (``content``, else its ``centralWidget()``: the analysis
    start screen is a plain widget); ``on_saved(old, new)`` gets the stored records before and after a save.
    Returns the dialog."""
    old = theme.remembered(opts.OPTIONS_KEY)
    old = dict(old) if isinstance(old, dict) else {}
    dialog = OptionsDialog(t, tool, old, choices=choices,
                           content=content if content is not None else window.centralWidget(),
                           parent=window)
    if on_saved is not None:
        dialog.saved.connect(lambda new: on_saved(old, new))
    dialog.closed.connect(dialog.deleteLater)
    dialog.open()
    return dialog
