"""The Composer section: charts added from their Export dialogs, arranged into one figure.

Asked for: "composer can be a new tab. u run the functions and in composer u can put it
together and exoprt". Any chart's Export dialog has **Add to composer**; this page lists what
was added, lets each panel be moved, given its width or removed, and draws the whole
figure at its printed size -- lettered per the journal, one Source Data sheet per panel.

THIN QT. The panels, their order, widths and letters are `smtracker.composer.Composition`;
the figure is `figure_export.compose`; the preview is drawn on a worker (the Export dialog's
own `_PreviewJob`), only the latest request drawn. The composition's style (journal, width,
font, files, rows, letters) is remembered per user (``ui.json``, ``figure_composer``);
the panels are not -- see `smtracker.composer`.

THE GRID IS SET HERE (asked for: "the width setting in the composer is a bit useless, instead
we need to define the grid (4x2, etc)" and "a way to rescale the single images"): Rows (Auto or
fixed) make the grid; each panel row says its WIDTH, a fraction of the figure's full width (1/4,
1/3, 1/2, 2/3, 3/4, Full; the Columns control is gone since 3 Nov 2026, asked: "why not 1/3, 1/2,
2/3, 1", so the Export dialog and this page name one width alike), the rows it spans and its Size
in its place. The figure's full width sits in the Size row (a journal fills its own).

EDIT (30 Oct 2026, asked: "in COMPOSER; edit button for single images - for example if title u
wanna edit, etc basically it gets u to the same page u would have on sole export screen for this
image to change stuff"): each panel row's pencil opens the chart's own Export dialog in its panel
mode (`ExportDialog(edit_panel=letter)`) on `Composition.edit_spec`; Apply to panel writes the
words back with `Composition.set_words` (that panel only) and the figure is drawn again; Cancel
changes nothing. A panel whose kind has no Export dialog has the button off, its tooltip saying why
(`Composition.edit_refusal`).

THE COMPOSITION FILE (30 Oct 2026, asked: "if a composer file is saved with export and u could open it
with same settings if u just wanna change small things?"): Export writes ``<name>.composer`` beside
the figure (`figure_export.write_figure`'s ``composition``); the Panels card's **Open…** and **Save…**
read and write one alone (`composer.read_composition` / `Composition.write`). Opening builds the whole
composition first and only then replaces the page's (panels, order, grid, letters, journal, size, font,
files); a refused file leaves the page as it was and says why in the hint. Opening a file does not
change the remembered Composer style (`ui.json`): that follows what you set here.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from qtpy.QtCore import Qt, QThreadPool, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import figure_export as fx
from ...composer import COMPOSITION_SUFFIX, ComposerError, Composition, read_composition, width_tip
from .. import theme
from ..theme import typography as ty
from ..widgets.controls import Button, FloatField, Segmented
from ..widgets.primitives import Card, Checkbox, discard, label
from .export_dialog import REMEMBER_FOLDER, ExportDialog, _PreviewJob, _Signals, show_redrawing
from .toolbar import SelectButton, ToolButton, rem

__all__ = ["ComposerSection"]

REMEMBER_KEY = "figure_composer"
#: The preview is DRAWN this wide and shown at whatever width the page leaves it, never
#: taller than `PREVIEW_MAX_H`.
PREVIEW_W, PREVIEW_MAX_H = 760, 720
EMPTY_TEXT = ("No panels yet.\n\nOpen any chart's Export (the download button on a chart) "
              "and press Add to composer. Each chart is kept as it is set there.")


def _remembered() -> tuple:
    """``(spec, rows, letters)`` as last used, or Nature's full width with as many rows as the
    panels need (a remembered ``columns`` from before 3 Nov 2026 is ignored)."""
    data = theme.remembered(REMEMBER_KEY) or {}
    spec = fx.FigureSpec.from_dict(data.get("spec")) if data.get("spec") else \
        fx.FigureSpec(width="double")
    rows = data.get("rows") if data.get("rows") in fx.COMPOSE_ROWS else None
    letters = data.get("letters") if data.get("letters") in fx.LETTER_CASES else "journal"
    return spec, rows, letters


ROW_LABELS = ["Auto"] + [str(n) for n in fx.COMPOSE_ROWS[1:]]


class _PanelRow(QWidget):
    """One panel: its letter, what it is, move / remove; under it its width, the rows it spans
    and its size in its place."""

    #: uid, a key of `figure_export.PANEL_WIDTHS`
    width_changed = Signal(int, str)
    row_span_changed = Signal(int, int)
    scale_changed = Signal(int, int)
    #: uid, "height:<uid>" / "width:<uid>" / "" (no snap)
    match_changed = Signal(int, str)
    moved = Signal(int, int)
    removed = Signal(int)
    #: uid: open this panel's own Export dialog to edit its words
    edit_requested = Signal(int)

    def __init__(self, t, row: dict, max_rows: int, match_options=(), width_tips=None,
                 parent=None):
        super().__init__(parent)
        self.uid = row["uid"]
        self.setStyleSheet("background:transparent;")
        both = QVBoxLayout(self)
        both.setContentsMargins(0, 4, 0, 6)
        both.setSpacing(4)
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        both.addLayout(lay)
        letter = label(row["letter"], ty.ui("base", "semibold"), t.foreground)
        letter.setFixedWidth(rem(1.5))
        lay.addWidget(letter, 0, Qt.AlignVCenter)
        words = QVBoxLayout()
        words.setSpacing(0)
        title = row["title"]
        self.title = label(title if len(title) <= 46 else title[:45] + "…",
                           ty.ui("sm", "medium"), t.foreground)
        self.title.setToolTip(title)
        words.addWidget(self.title)
        words.addWidget(label(row["detail"], ty.ui("xs"), t.muted_foreground))
        lay.addLayout(words, 1)
        # Under the title: its width (a line of its own), then the rows it spans and its size.
        wide = QHBoxLayout()
        wide.setContentsMargins(rem(1.5) + 10, 0, 0, 0)
        wide.setSpacing(6)
        wide.addWidget(label("width", ty.ui("xs"), t.muted_foreground))
        self.width_choice = Segmented(t, [fx.PANEL_WIDTH_LABELS[w] for w in fx.PANEL_WIDTHS],
                               current=fx.PANEL_WIDTHS.index(row["width"]), size="sm")
        self.width_choice.setToolTip(width_tips or "Its width, a fraction of the figure's full width")
        self.width_choice.changed.connect(lambda i: self.width_changed.emit(self.uid, fx.PANEL_WIDTHS[i]))
        wide.addWidget(self.width_choice)
        wide.addStretch(1)
        both.addLayout(wide)
        layout = QHBoxLayout()
        layout.setContentsMargins(rem(1.5) + 10, 0, 0, 0)
        layout.setSpacing(6)
        self.row_span = None
        if max_rows > 1:
            self.row_span = Segmented(t, [str(n) for n in range(1, max_rows + 1)],
                                      current=min(row["row_span"], max_rows) - 1, size="sm")
            self.row_span.setToolTip("How many rows this panel spans")
            self.row_span.changed.connect(
                lambda i: self.row_span_changed.emit(self.uid, i + 1))
            layout.addWidget(self.row_span)
            layout.addWidget(label("rows", ty.ui("xs"), t.muted_foreground))
            layout.addSpacing(6)
        lo, hi = fx.PANEL_SCALE
        self.scale = FloatField(t, lo, hi, row["scale"], step=5, decimals=0, suffix=" %",
                                width=112)
        self.scale.setToolTip("Its size in its cell. The panel is drawn again at that size and centred, "
                              "with text at its own point size, so a PDF or SVG loses nothing.")
        self.scale.changed.connect(lambda v: self.scale_changed.emit(self.uid, round(v)))
        layout.addWidget(label("size", ty.ui("xs"), t.muted_foreground))
        layout.addWidget(self.scale)
        layout.addStretch(1)
        both.addLayout(layout)
        # SNAP (asked for: "snap A to height of B, or width of C"): its plot or image area
        # made exactly as tall or wide as another panel's and lined up with it.
        self.match = None
        if len(match_options) > 1:
            snap = QHBoxLayout()
            snap.setContentsMargins(rem(1.5) + 10, 0, 0, 0)
            snap.setSpacing(6)
            snap.addWidget(label("snap to", ty.ui("xs"), t.muted_foreground))
            self.match = SelectButton(t, match_options, current=row["match"])
            self.match.setFixedWidth(rem(9))
            self.match.setToolTip("Make this panel's plot or image as tall (or wide) "
                                  "as another panel's and line it up: top edges for a "
                                  "height, left edges for a width. It replaces the size.")
            self.match.changed.connect(lambda key: self.match_changed.emit(self.uid, key))
            snap.addWidget(self.match)
            snap.addStretch(1)
            both.addLayout(snap)
            if row["match"]:
                self.scale.setEnabled(False)
                self.scale.setToolTip("Set by the snap below")
        # EDIT: the chart's own Export dialog, on this panel's words (`ComposerSection.edit_panel`)
        self.edit = ToolButton(t, "edit", "pencil", size=rem(1.75), rest=t.card)
        why = row.get("edit_why", "")
        self.edit.setToolTip(why or "Edit this panel: its title, axis titles and ranges, names and what "
                                    "it shows, in its chart's own Export dialog")
        self.edit.setEnabled(not why)
        lay.addWidget(self.edit, 0, Qt.AlignVCenter)
        self.edit.clicked.connect(lambda _k: self.edit_requested.emit(self.uid))
        self.up = ToolButton(t, "up", "chevron-up", size=rem(1.75), rest=t.card)
        self.down = ToolButton(t, "down", "chevron-down", size=rem(1.75), rest=t.card)
        self.remove = ToolButton(t, "remove", "trash", size=rem(1.75), rest=t.card)
        for button, tip in ((self.up, "Earlier"), (self.down, "Later"),
                            (self.remove, "Remove from the figure")):
            button.setToolTip(tip)
            lay.addWidget(button, 0, Qt.AlignVCenter)
        self.up.setEnabled(not row["first"])
        self.down.setEnabled(not row["last"])
        self.up.clicked.connect(lambda _k: self.moved.emit(self.uid, -1))
        self.down.clicked.connect(lambda _k: self.moved.emit(self.uid, 1))
        self.remove.clicked.connect(lambda _k: self.removed.emit(self.uid))


class ComposerSection(QWidget):
    """The Composer: settings and the panel list left, the figure right, Export below."""

    #: The panel count changed (the window may say so in its status line).
    panels_changed = Signal(int)
    exported = Signal(list)
    #: "Export every figure…": the window knows every chart, so it does the writing.
    export_all_requested = Signal()

    def __init__(self, t, parent=None, *, threads: QThreadPool | None = None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self.spec, rows, letters = _remembered()
        if self.spec.width != "custom":                # the width is one number here
            self.spec = fx.with_changes(self.spec, width="custom",
                                        width_mm=self.spec.width_value())
        self.model = Composition(rows=rows, letters=letters)
        self._threads = threads or QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.drawn.connect(self._drawn)
        self._generation = 0
        self._running = False
        self._again = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._render)
        self._rows: list[_PanelRow] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        # The page narrows with the window (the preview shrinks); it never scrolls sideways.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page = QWidget()
        page.setStyleSheet("background:transparent;")
        scroll.setWidget(page)
        outer.addWidget(scroll)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        intro = label("Charts added from their Export dialogs, put together as one figure: "
                      "lettered for the journal, one Source Data sheet per panel.",
                      ty.ui("sm"), t.muted_foreground)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        body = QHBoxLayout()
        body.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(self._settings_card(t))
        left.addWidget(self._panels_card(t))
        left.addStretch(1)
        body.addLayout(left, 0)
        body.addWidget(self._preview_card(t), 1, Qt.AlignTop)
        lay.addLayout(body)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.hint = label("", ty.ui("xs"), t.muted_foreground)
        self.hint.setWordWrap(True)
        footer.addWidget(self.hint, 1)
        self.export_all_button = Button(t, "Export every figure…", glyph="download")
        self.export_all_button.setToolTip(
            "Every chart computed in this workspace, each with the settings it was last "
            "exported with (and this composition, if it has panels), into one folder.")
        self.export_all_button.clicked.connect(self.export_all_requested.emit)
        footer.addWidget(self.export_all_button)
        self.copy_button = Button(t, "Copy image", glyph="copy")
        self.copy_button.clicked.connect(self.copy_image)
        footer.addWidget(self.copy_button)
        self.export_button = Button(t, "Export…", variant="default", glyph="download")
        self.export_button.clicked.connect(self.choose_and_export)
        footer.addWidget(self.export_button)
        lay.addLayout(footer)
        lay.addStretch(1)
        self._sync_controls()
        self._refresh()

    # -- building -------------------------------------------------------------------- #
    def _grid_row(self, grid, r, name, widget, extra=None):
        grid.addWidget(label(name, ty.ui("sm", "medium"), self.t.foreground), r, 0,
                       Qt.AlignLeft | Qt.AlignVCenter)
        cell = QHBoxLayout()
        cell.setSpacing(8)
        cell.addWidget(widget)
        if extra is not None:
            cell.addWidget(extra)
        cell.addStretch(1)
        grid.addLayout(cell, r, 1)

    def _settings_card(self, t):
        card = Card(t)
        grid = QGridLayout(card)
        grid.setContentsMargins(20, 16, 20, 16)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        s = self.spec
        self.preset_choice = Segmented(t, [fx.PRESET_LABELS[p] for p in fx.PRESETS],
                                       current=fx.PRESETS.index(s.preset), size="sm")
        self.preset_choice.changed.connect(self._preset_picked)
        self._grid_row(grid, 0, "For", self.preset_choice)
        self.width_mm_field = FloatField(t, 20, 600, s.width_value(), step=1, decimals=0,
                                         suffix=" mm", width=128)
        # THE FIGURE'S WIDTH, one box (asked: the 1 column / 1.5 / 2 columns choice "doesnt
        # do much" here): choosing a journal fills its full width, and a width past what
        # it prints is named under the preview.
        self.width_mm_field.setToolTip("The figure's full width; each panel takes its fraction of "
                                       "it (1/2, 2/3, ...). Choosing a journal fills its full width.")
        self.width_mm_field.changed.connect(
            lambda v: self._set(width="custom", width_mm=float(v)))
        self.height_mm_field = FloatField(t, 20, 600, 120.0, step=1, decimals=0,
                                          suffix=" mm", width=128)
        self.height_mm_field.changed.connect(
            lambda v: self._set(height_mm=float(v)) if not self.auto_height.checked else None)
        self.auto_height = Checkbox(t, checked=s.height_mm is None)
        self.auto_height.toggled.connect(self._auto_height)
        size = QHBoxLayout()
        size.setSpacing(6)
        size.addWidget(self.width_mm_field)
        size.addWidget(label("×", ty.ui("sm"), t.muted_foreground))
        size.addWidget(self.height_mm_field)
        size.addWidget(self.auto_height)
        size.addWidget(label("auto height", ty.ui("xs"), t.muted_foreground))
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        holder.setLayout(size)
        self._grid_row(grid, 1, "Size", holder)
        self.rows_choice = Segmented(t, ROW_LABELS,
                                     current=fx.COMPOSE_ROWS.index(self.model.rows_fixed),
                                     size="sm")
        self.rows_choice.setToolTip("Auto: as many rows as the panels need. A number fixes "
                                    "the grid; its empty cells stay empty.")
        self.rows_choice.changed.connect(self._rows_picked)
        self._grid_row(grid, 2, "Rows", self.rows_choice)
        self.letters_choice = Segmented(
            t, [fx.LETTER_CASE_LABELS[c] for c in fx.LETTER_CASES],
            current=fx.LETTER_CASES.index(self.model.letters), size="sm")
        self.letters_choice.setToolTip(
            "Journal: Nature's own rule is 8 pt bold a, b, c; Cell and eLife letter A, B, C "
            "(not confirmed from their own pages).")
        self.letters_choice.changed.connect(self._letters_picked)
        self._grid_row(grid, 3, "Letters", self.letters_choice)
        self.font_choice = Segmented(t, list(fx.FONTS), current=fx.FONTS.index(s.font),
                                     size="sm")
        self.font_choice.changed.connect(lambda i: self._set(font=fx.FONTS[i]))
        self.font_pt_field = FloatField(t, 4, 40, s.pt(), step=0.5, decimals=1, suffix=" pt",
                                        width=100)
        self.font_pt_field.changed.connect(lambda v: self._set(font_pt=float(v)))
        self._grid_row(grid, 4, "Font", self.font_choice, self.font_pt_field)
        self.format_boxes = {}
        files = QHBoxLayout()
        files.setSpacing(6)
        for fmt in fx.FORMATS:
            box = Checkbox(t, checked=fmt in s.formats)
            box.toggled.connect(lambda _on: self._formats_changed())
            self.format_boxes[fmt] = box
            files.addWidget(box)
            files.addWidget(label(fmt.upper(), ty.ui("sm"), t.foreground))
            files.addSpacing(6)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        holder.setLayout(files)
        self._grid_row(grid, 5, "Files", holder)
        self.dpi_choice = Segmented(t, [f"{d} dpi" for d in fx.DPIS],
                                    current=fx.DPIS.index(s.dpi), size="sm")
        self.dpi_choice.changed.connect(lambda i: self._set(dpi=fx.DPIS[i]))
        self._grid_row(grid, 6, "Images", self.dpi_choice)
        return card

    def _panels_card(self, t):
        card = Card(t)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(label("Panels", ty.ui("sm", "semibold"), t.foreground))
        head.addStretch(1)
        # THE COMPOSITION FILE: the figure whole (panels, data, settings), to open again and change
        self.open_button = Button(t, "Open…", size="small", glyph="folder-open")
        self.open_button.setToolTip("Open a composition file (.composer): its panels, their data and the "
                                    "figure's settings replace what is here, ready to edit and export again. "
                                    "Open only files you made.")
        self.open_button.clicked.connect(self.choose_and_open)
        head.addWidget(self.open_button)
        self.save_button = Button(t, "Save…", size="small", glyph="save")
        self.save_button.setToolTip("Save this composition (.composer) without exporting: every panel with its "
                                    "data and the figure's settings. Export writes one beside the figure too.")
        self.save_button.clicked.connect(self.choose_and_save)
        head.addWidget(self.save_button)
        lay.addLayout(head)
        self.empty_note = label("Nothing added yet. Press Add to composer in any chart's Export.",
                           ty.ui("xs"), t.muted_foreground)
        self.empty_note.setWordWrap(True)
        lay.addWidget(self.empty_note)
        self._list = QVBoxLayout()
        self._list.setSpacing(2)
        lay.addLayout(self._list)
        self.clear_button = Button(t, "Remove all", size="small", glyph="trash")
        self.clear_button.clicked.connect(self.clear)
        clear_row = QHBoxLayout()
        clear_row.addStretch(1)
        clear_row.addWidget(self.clear_button)
        lay.addLayout(clear_row)
        card.setMinimumWidth(rem(32))
        return card

    def _preview_card(self, t):
        card = Card(t)
        col = QVBoxLayout(card)
        col.setContentsMargins(16, 16, 16, 16)
        col.setSpacing(8)
        self.preview = QLabel()
        self.preview.setMinimumSize(rem(20), rem(14))
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet(f"background:{t.muted}; border-radius:{t.radius_md:g}px;"
                                   f" color:{t.muted_foreground};")
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.preview.setFixedHeight(rem(20))
        col.addWidget(self.preview)
        self.info = label("", ty.ui("xs", tabular=True), t.foreground)
        self.info.setWordWrap(True)
        col.addWidget(self.info)
        self.findings = label("", ty.ui("xs"), t.muted_foreground)
        self.findings.setWordWrap(True)
        col.addWidget(self.findings)
        self._pixmap = QPixmap()
        return card

    def _redrawing(self) -> None:
        """The figure taken away and "Redrawing…" at its place, its height kept (`export_dialog.show_redrawing`): a
        resize while it is drawn shows nothing old either. Nothing to redraw, nothing said."""
        if self._pixmap.isNull() or not self.model.panels:
            return
        self._pixmap = QPixmap()
        show_redrawing(self.preview)

    def _show_pixmap(self) -> None:
        """The drawn figure at the width the page leaves it, its height following."""
        if self._pixmap.isNull():
            return
        width = max(1, min(self.preview.width(), self._pixmap.width()))
        shown = self._pixmap.scaledToWidth(width, Qt.SmoothTransformation)
        if shown.height() > PREVIEW_MAX_H:
            shown = self._pixmap.scaledToHeight(PREVIEW_MAX_H, Qt.SmoothTransformation)
        self.preview.setFixedHeight(shown.height())
        self.preview.setPixmap(shown)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._show_pixmap()

    # -- panels ---------------------------------------------------------------------- #
    def add_panel(self, record: dict) -> str:
        """Add a chart handed over by an Export dialog; returns its letter."""
        panel = self.model.add(record["kind"], record["payload"], record.get("words"),
                               record.get("title"), width=record.get("width", "1/2"))
        self._after_grid_change()
        self.panels_changed.emit(len(self.model.panels))
        return self.model.rows(self.spec.preset)[-1]["letter"] if panel else ""

    def panel_count(self) -> int:
        return len(self.model.panels)

    def say(self, text: str) -> None:
        self.hint.setText(text)

    def composed_job(self):
        """This composition as one more figure for "export every figure", or None."""
        if not self.model.panels:
            return None
        records, spec, rows, letters = self.model.build_inputs(self.spec)
        return {"kind": "composed", "name": "composed_figure", "spec": spec,
                "payload": {"panels": records, "rows": rows, "letters": letters},
                "provenance": None}

    def clear(self) -> None:
        self.model.clear()
        self._refresh()
        self.panels_changed.emit(0)

    def _on_width(self, uid: int, width: str) -> None:
        self.model.set_width(uid, width)
        self._after_grid_change()

    def _on_row_span(self, uid: int, rows: int) -> None:
        self.model.set_row_span(uid, rows)
        self._after_grid_change()

    def _on_match(self, uid: int, key: str) -> None:
        dimension, _, target = key.partition(":")
        try:
            self.model.set_match(uid, dimension or None, int(target) if target else None)
        except ComposerError as exc:
            self.hint.setText(str(exc))
            self._refresh()
            return
        self.hint.setText("")
        self._refresh()

    def _on_scale(self, uid: int, percent: int) -> None:
        self.model.set_scale(uid, percent)
        self._redrawing()
        self._debounce.start()

    def _after_grid_change(self) -> None:
        """A change that may have grown the grid: the Rows control follows and the note is
        said."""
        self._show_rows()
        if self.model.note:
            self.hint.setText(self.model.note)
        self._remember()
        self._refresh()

    def _show_rows(self) -> None:
        self.rows_choice.blockSignals(True)
        self.rows_choice.select(fx.COMPOSE_ROWS.index(self.model.rows_fixed))
        self.rows_choice.blockSignals(False)

    def edit_panel(self, uid: int):
        """Panel ``uid``'s own Export dialog, filled with its words (`Composition.edit_spec`), in
        its panel mode: Apply to panel writes them back to that panel only. Returns the dialog, or
        None when the panel cannot be edited (said in the hint)."""
        why = self.model.edit_refusal(uid)
        if why:
            self.hint.setText(why)
            return None
        panel = self.model.panel(uid)
        window = self.window()
        content = window.centralWidget() if hasattr(window, "centralWidget") else None
        dialog = ExportDialog(self.t, panel.kind, panel.payload, content=content,
                              parent=window if window is not self else None,
                              spec=self.model.edit_spec(uid, self.spec),
                              edit_panel=self._label_of(uid),
                              threads=self._threads)
        dialog.closed.connect(dialog.deleteLater)
        dialog.panel_applied.connect(lambda words, u=uid: self._on_edited(u, words))
        self._edit_dialog = dialog
        dialog.open()
        return dialog

    def _on_edited(self, uid: int, words: dict) -> None:
        """Apply to panel: that panel's words, the figure drawn again."""
        try:
            self.model.set_words(uid, words)
        except ComposerError as exc:
            self.hint.setText(str(exc))
            return
        self.hint.setText(f"Panel {self._label_of(uid)} changed.")
        self._refresh()

    def _label_of(self, uid: int) -> str:
        """The panel's name in the list: its letter, or its place when letters are off."""
        return next(r["letter"] for r in self.model.rows(self.spec.preset) if r["uid"] == uid)

    def _on_move(self, uid: int, step: int) -> None:
        self.model.move(uid, step)
        self._refresh()

    def _on_remove(self, uid: int) -> None:
        self.model.remove(uid)
        self._refresh()
        self.panels_changed.emit(len(self.model.panels))

    def _refresh(self) -> None:
        """The list from the model, then a new preview."""
        for row in self._rows:
            self._list.removeWidget(row)
            discard(row)
        self._rows = []
        max_rows = self.model.rows_fixed or min(4, max(2, self.model.n_rows()))
        tips = width_tip(self.spec)
        for data in self.model.rows(self.spec.preset):
            row = _PanelRow(self.t, data, max_rows,
                            self.model.match_options(data["uid"], self.spec.preset), tips)
            row.match_changed.connect(self._on_match)
            row.width_changed.connect(self._on_width)
            row.row_span_changed.connect(self._on_row_span)
            row.scale_changed.connect(self._on_scale)
            row.moved.connect(self._on_move)
            row.removed.connect(self._on_remove)
            row.edit_requested.connect(self.edit_panel)
            self._list.addWidget(row)
            self._rows.append(row)
        has = bool(self.model.panels)
        self.empty_note.setVisible(not has)
        self.clear_button.setVisible(has)
        self.save_button.setEnabled(has)
        self.copy_button.setEnabled(has)
        self.export_button.setEnabled(has)
        self._redrawing()
        self._debounce.start()

    # -- settings ---------------------------------------------------------------------- #
    def _remember(self) -> None:
        try:
            theme.remember(REMEMBER_KEY, {"spec": self.spec.remembered_record(),
                                          "rows": self.model.rows_fixed,
                                          "letters": self.model.letters})
        except OSError:
            pass

    def _set(self, **changes) -> None:
        candidate = fx.with_changes(self.spec, **changes)
        problem = candidate.problem()
        if problem:
            self.hint.setText(problem)
            return
        self.spec = candidate
        self.hint.setText("")
        self._sync_controls()
        self._remember()
        self._refresh()

    def _preset_picked(self, index: int) -> None:
        preset = fx.PRESETS[index]
        full = fx.figstyle.JOURNAL_PRESETS[preset].get("double_mm") or self.spec.width_value()
        self._set(preset=preset, font_pt=None, height_mm=None, width="custom",
                  width_mm=float(full))

    def _auto_height(self, on: bool) -> None:
        self._set(height_mm=None if on else float(self.height_mm_field.value()))

    def _rows_picked(self, index: int) -> None:
        try:
            self.model.set_rows(fx.COMPOSE_ROWS[index])
        except ComposerError as exc:
            self.hint.setText(str(exc))
            self._show_rows()
            return
        self.hint.setText("")
        self._remember()
        self._refresh()

    def _letters_picked(self, index: int) -> None:
        self.model.set_letters(fx.LETTER_CASES[index])
        self._remember()
        self._refresh()

    def _formats_changed(self) -> None:
        chosen = tuple(f for f in fx.FORMATS if self.format_boxes[f].checked)
        if not chosen:
            self.hint.setText("Choose at least one file type.")
            return
        self._set(formats=chosen)

    def _sync_controls(self) -> None:
        s = self.spec
        for field, value, editable in ((self.width_mm_field, s.width_value(), True),
                                       (self.font_pt_field, s.pt(), True)):
            field.blockSignals(True)
            field.setValue(value)
            field.blockSignals(False)
            field.setEnabled(editable)
        if s.height_mm is not None:
            self.height_mm_field.blockSignals(True)
            self.height_mm_field.setValue(s.height_mm)
            self.height_mm_field.blockSignals(False)
        self.height_mm_field.setEnabled(s.height_mm is not None)
        self.dpi_choice.setEnabled(any(f in s.formats for f in ("png", "tiff")))

    # -- the preview, on a worker -------------------------------------------------------- #
    def _render(self) -> None:
        if not self.model.panels:
            self._png = b""
            self._pixmap = QPixmap()
            self.preview.setPixmap(QPixmap())
            self.preview.setFixedHeight(rem(20))
            self.preview.setText(EMPTY_TEXT)
            self.info.setText("")
            self.findings.setText("")
            return
        self._redrawing()
        if self._running:
            self._again = True
            return
        self._running = True
        self._generation += 1
        records, spec, rows, letters = self.model.build_inputs(self.spec)
        self._threads.start(_PreviewJob(
            self._generation, partial(fx.compose, records, spec, rows=rows, letters=letters),
            self._signals, width_px=PREVIEW_W * 2))

    def _drawn(self, generation, png, findings, notes, error) -> None:
        self._running = False
        if self._again:
            self._again = False
            self._render()
            return
        if error:
            self._pixmap = QPixmap()
            self.preview.setPixmap(QPixmap())
            self.preview.setFixedHeight(rem(20))
            self.preview.setText(f"Could not draw this figure:\n{error}")
            return
        self._png = png
        image = QImage.fromData(png)
        self.preview.setText("")
        self._pixmap = QPixmap.fromImage(image)
        self._show_pixmap()
        width_mm = self.spec.width_value()
        height_mm = image.height() / max(1, image.width()) * width_mm
        self.info.setText(f"{self.model.summary(self.spec)}  ·  {width_mm:g} × "
                          f"{height_mm:.0f} mm  ·  {self.spec.font} {self.spec.pt():g} pt")
        lines = [f"• {n}" for n in notes] + [f"⚠ {f}" for f in findings]
        journal = fx.PRESET_LABELS[self.spec.preset]
        self.findings.setText("\n".join(lines) if lines else
                              f"No problems found at this size: every text "
                              f"{self.spec.min_pt():g} pt or more ({journal}'s minimum), "
                              f"nothing overlapping, nothing cut off.")

    def wait(self, seconds: float = 20.0) -> None:
        """Until the preview in flight has landed (for tests and a scripted export)."""
        import time

        end = time.monotonic() + seconds
        while (self._running or self._debounce.isActive()) and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)

    # -- output -------------------------------------------------------------------------- #
    def copy_image(self) -> None:
        built = self.model.build(fx.with_changes(self.spec, dpi=300))
        png = fx.preview_png(built, round(built.size_mm()[0] / 25.4 * 300))
        QApplication.clipboard().setImage(QImage.fromData(png))
        self.hint.setText("Copied at 300 dpi. Paste it into a slide or a message.")

    def export_to(self, path) -> list:
        """Write the composed figure and everything beside it to ``path``, the composition file
        (``<name>.composer``) with it."""
        from ...buildstamp import build_stamp

        records, spec, rows, letters = self.model.build_inputs(self.spec)
        built = fx.compose(records, spec, rows=rows, letters=letters)
        composed = {"panels": records, "rows": rows, "letters": letters}
        try:
            written = fx.write_figure(built, path, stamp=build_stamp(),
                                      composition={"composed": composed, "spec": spec})
            said = f"Wrote {len(written)} files to {Path(path).parent}, with the composition file to open it again"
        except ComposerError as exc:
            # the figure and its numbers are written; only the composition file is not: said
            base = fx._base(path)
            written = sorted(base.parent.glob(f"{base.name}.*"))
            said = f"Wrote {len(written)} files to {Path(path).parent}; {exc}"
        try:
            theme.remember(REMEMBER_FOLDER, str(Path(path).parent))
        except OSError:
            pass
        self.hint.setText(said)
        self.exported.emit([str(p) for p in written])
        return written

    # -- the composition file -------------------------------------------------------------- #
    def save_composition_to(self, path):
        """This composition, with the figure's settings, as a composition file at ``path`` (no export).
        Returns the path written, or None (the hint says why)."""
        from ...buildstamp import build_stamp

        try:
            written = self.model.write(path, self.spec, build=build_stamp())
        except ComposerError as exc:
            self.hint.setText(str(exc))
            return None
        self.hint.setText(f"Saved {written.name}: {len(self.model.panels)} panels and the figure's settings.")
        return written

    def open_composition_from(self, path):
        """The composition in ``path`` in place of this one -- panels, order, grid, letters and the
        figure's settings -- built whole first: a file refused (`composer.read_composition`) leaves this
        page as it was. Returns the composition, or None (the hint says why)."""
        try:
            model, spec, _manifest = read_composition(path)
        except ComposerError as exc:
            self.hint.setText(str(exc))
            return None
        if spec.width != "custom":                             # the width is one number here
            spec = fx.with_changes(spec, width="custom", width_mm=spec.width_value())
        replaced = len(self.model.panels)
        self.model, self.spec = model, spec
        self._sync_all()
        self._refresh()
        self.panels_changed.emit(len(self.model.panels))
        n = len(model.panels)
        self.hint.setText(f"Opened {Path(path).name}: {n} panel{'s' if n != 1 else ''}"
                          + (f" (in place of the {replaced} here)." if replaced else "."))
        return model

    def _sync_all(self) -> None:
        """Every control back on the composition and its settings (an opened file), without
        re-entering their handlers."""
        s = self.spec
        for choice, index in ((self.preset_choice, fx.PRESETS.index(s.preset)),
                              (self.letters_choice, fx.LETTER_CASES.index(self.model.letters)),
                              (self.font_choice, fx.FONTS.index(s.font)),
                              (self.dpi_choice, fx.DPIS.index(s.dpi))):
            choice.blockSignals(True)
            choice.select(index)
            choice.blockSignals(False)
        self._show_rows()
        for fmt, box in self.format_boxes.items():
            box.set_checked(fmt in s.formats, animate=False)
        self.auto_height.set_checked(s.height_mm is None, animate=False)
        self._sync_controls()

    def choose_and_save(self) -> None:
        folder = theme.remembered(REMEMBER_FOLDER) or str(Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save composition", str(Path(folder) / f"figure{COMPOSITION_SUFFIX}"),
            f"Composition (*{COMPOSITION_SUFFIX})")
        if path:
            self.save_composition_to(path)

    def choose_and_open(self) -> None:
        folder = theme.remembered(REMEMBER_FOLDER) or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open composition", folder, f"Composition (*{COMPOSITION_SUFFIX})")
        if path:
            self.open_composition_from(path)

    def choose_and_export(self) -> None:
        folder = theme.remembered(REMEMBER_FOLDER) or str(Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", str(Path(folder) / "figure"),
            "Figure (*.pdf *.svg *.png *.tiff)")
        if not path:
            return
        written = self.export_to(path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(written[0]).parent)))
