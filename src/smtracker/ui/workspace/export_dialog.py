"""The Export dialog: a figure at its printed size, previewed live, then written.

THIN QT. Every figure, size, font, check and file comes from `smtracker.figure_export`
(headless); this only holds the controls, shows the preview it is handed, and asks where
to write. The preview is drawn on a WORKER (`_PreviewJob`): a rebuild is ~0.3 s, the
first ~0.9 s, and a dialog that froze for that on every keystroke would read as broken.
Only the latest request is drawn -- one superseded while running is dropped.

The last settings are remembered per user (`ui.json`, key ``figure_export``), so a lab
that submits to one journal sets it once -- all but the condition names, which belong to
the project in hand.

THE TEXT ROWS (asked for: "custom field for axis titles and condition names"): the Y and X
axis titles and one field per condition, each pre-filled with what the screen shows;
emptying a condition's field restores its own name. A GROUPED bubble chart (fits side by side,
a folder design's nesting; 30 Oct 2026) lists its GROUP HEADERS and its COLUMN LABELS instead, each
once under a small heading (`figure_export.name_sections`): a rename reaches every place that text
is drawn (`figure_export.with_part_names`). The names stay ``spec.names`` pairs, one route; a
rename of a name no field shows (a full column key, renamed before) is kept, not dropped.

THE FIT ROW (a fitted figure only, `figure_export.fit_choices`): which fit is drawn with its
components and residuals (`FigureSpec.chosen_fit`), starting on the card's. On the jump-distance
figures (`figure_export.FIT_SET_KINDS`; 30 Oct 2026) it is TOGGLES, as the card's Show: each
count on or off, the last one on staying on, what a click does asked of `figure_export.fit_pressed`
and the pills lit from `figure_export.shown_fit` -- the dialog decides nothing.

THE SHOW ROW is `figure_export.show_options`: only the boxes that change this chart (the
jump-distance figure's include its two residual plots, asked 30 Oct 2026: "also make it optional
to export ecdf and residuals in export window and in composer", and both jump figures' Components,
which starts as the card has it: `figure_export.show_checked`). Four boxes or more take two
lines; Add to composer carries them with the panel (`figure_export.PANEL_WORDS`).

THE SIGNIFICANCE ROW (the bubble charts only; ruled 13 Oct 2026, moved 14 Oct 2026): READ-ONLY.
The brackets are chosen on the CHART (its Significance tool on the right rail) and exported as
shown ("it exports how u see it depending on the choices on the plot"); the row says what will
be drawn -- the test, whose fraction, each pair's p (adjusted and raw) or why it has none, all from
the headless `smtracker.significance` -- and where to change it. A mixed-model p not yet in the
cache is computed by the PREVIEW WORKER (its build runs `significance.compare`), and the lines
are refreshed when the preview lands. A chart with no per-day values behind
its bubbles (Uncertainty None or Cells, an overlay, the apparent-D mixture) says why beside a
box that cannot be ticked.

EDITING A COMPOSER PANEL (30 Oct 2026, asked: "in COMPOSER; edit button for single images - for
example if title u wanna edit, etc basically it gets u to the same page u would have on sole export
screen for this image to change stuff"): ``edit_panel`` (the panel's letter) opens THIS dialog for
one Composer panel, on `composer.Composition.edit_spec` -- the same rows its chart's Export offers,
filled with the panel's words, the preview the panel at the width it has in the figure. The footer
is Cancel and **Apply to panel** (``panel_applied``: `figure_export.panel_words`); Add to composer,
Copy image and Export are not offered. The journal, width, size, font, files and resolution are the
COMPOSED figure's: off and left out (`COMPOSITION_ROWS`), and one line says so (`EDIT_NOTE`). Nothing is remembered as the next
export's settings from this mode.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import composer
from ... import figure_export as fx
from ... import significance as sig
from .. import theme
from ..theme import typography as ty
from ..widgets.controls import Button, FloatField, Segmented, Toggles
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import Checkbox, FitScroll, discard, fit_label, label
from .ramp_dialog import _combo_css

__all__ = ["COMPOSITION_ROWS", "EDIT_NOTE", "ExportDialog", "remembered_spec"]

PREVIEW_W, PREVIEW_H = 560, 400
#: Said over the settings when the dialog edits a Composer panel: why some rows are not there.
EDIT_NOTE = ("Journal, width, size, font and files belong to the composed figure: set them in the "
             "Composer. Here, this panel's own words and options.")
#: The rows that are the COMPOSED figure's, not a panel's: left out when editing a panel.
COMPOSITION_ROWS = ("For", "Width", "Size", "Font", "Files", "Images")
REMEMBER_KEY = "figure_export"
REMEMBER_FOLDER = "figure_export_folder"
#: The Width pills, said (3 Nov 2026: named as fractions of the full width, `figure_export.WIDTHS`).
WIDTH_TIP = ("A fraction of the journal's full figure width (Nature 183, Cell 174, eLife 170 mm). 1/2 and Full are "
             "one and two columns; 2/3 is Cell's 1.5 columns (114 mm). Other fractions use the Composer's gap.")


def remembered_spec() -> fx.FigureSpec:
    return fx.FigureSpec.from_dict(theme.remembered(REMEMBER_KEY))


#: Said in a preview while it is drawn again (7 Nov 2026, asked: "when exporter and composer is redrawing something,
#: can we let the image disappear and indicate a redraw action").
REDRAWING_TEXT = "Redrawing…"


def show_redrawing(preview: QLabel) -> None:
    """The preview's figure taken away and "Redrawing…" in its place, at its size, the moment a change asks for a new
    one: the old figure is never shown beside settings it was not drawn with. The new figure replaces the words."""
    preview.setPixmap(QPixmap())
    preview.setText(REDRAWING_TEXT)


class _TextField(QLineEdit):
    """A one-line text input in the shadcn Input styling."""

    def __init__(self, t, text: str, placeholder: str = "", width: int = 240):
        super().__init__(text)
        self.setFont(ty.ui("sm"))
        self.setPlaceholderText(placeholder)
        self.setFixedSize(width, t.control_height)
        self.setStyleSheet(
            f"QLineEdit {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:0px 10px; }}")


#: The Export dialog's scrolling columns: `primitives.FitScroll`, now shared with the Spatial Map's dialog.
_FitScroll = FitScroll


class _Signals(QObject):
    #: generation, PNG bytes (or b""), findings, notes, error
    drawn = Signal(int, bytes, list, list, str)


class _PreviewJob(QRunnable):
    """Draws ``build()`` (a single chart, or the composer's whole figure) off the GUI thread."""

    def __init__(self, generation, build, signals, width_px: int = PREVIEW_W * 2):
        super().__init__()
        self.generation, self.build, self.signals = generation, build, signals
        self.width_px = width_px

    def run(self):
        try:
            built = self.build()
            found = fx.check_figure(built)
            # A text set smaller to fit its room is a NOTE (•), not a problem (⚠): `fx.FigureNote`.
            notes = list(built.notes) + [f for f in found if isinstance(f, fx.FigureNote)]
            result = (self.generation, fx.preview_png(built, self.width_px),
                      [str(f) for f in found if not isinstance(f, fx.FigureNote)], [str(n) for n in notes], "")
        except Exception as exc:                            # noqa: BLE001 - shown
            result = (self.generation, b"", [], [], f"{type(exc).__name__}: {exc}")
        try:
            self.signals.drawn.emit(*result)
        except RuntimeError:
            # The dialog (or page) closed while this drew: nobody is left to show it to.
            pass


class ExportDialog(OverlayDialog):
    """Settings left, the figure right; Copy image and Export… below."""

    exported = Signal(list)
    #: "Add to composer": ``{"kind", "payload", "words", "title", "width"}`` (`composer.Composition.add`).
    composer_requested = Signal(dict)
    #: "Apply to panel" (editing a Composer panel): the dialog's words (`figure_export.panel_words`).
    panel_applied = Signal(dict)

    def __init__(self, t, kind: str, payload: dict, *, provenance=None, stamp: str = "",
                 content=None, parent=None, threads: QThreadPool | None = None,
                 spec: fx.FigureSpec | None = None, edit_panel: str | None = None):
        super().__init__(t, content, parent=parent)
        self.t, self.kind, self.payload = t, kind, payload
        #: The Composer panel this dialog edits (its letter), or None: an ordinary export.
        self.edit_panel = edit_panel
        self.provenance, self.stamp = provenance, stamp
        #: ``spec``: THIS chart's settings as last exported (titles and names included),
        #: so an export after a re-run starts where the last one ended.
        self.spec = spec if spec is not None else remembered_spec()
        #: The conditions the payload draws, in order: one name field each.
        self.conditions = fx.column_names(kind, payload)
        self._threads = threads or QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.drawn.connect(self._drawn)
        self._generation = 0
        self._running = False
        self._again = False
        self._png = b""
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._render)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        heading = (f"Edit panel {edit_panel}: {fx.KINDS.get(kind, kind)}" if edit_panel is not None
                   else f"Export figure: {fx.KINDS.get(kind, kind)}")
        head.addWidget(label(heading, ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        self.edit_note = None
        if edit_panel is not None:
            self.edit_note = label(EDIT_NOTE, ty.ui("xs"), t.muted_foreground)
            self.edit_note.setWordWrap(True)
            self.card_lay.addWidget(self.edit_note)

        body = QHBoxLayout()
        body.setSpacing(24)
        settings = QWidget()
        settings.setStyleSheet("background:transparent;")
        settings.setLayout(self._settings(t))
        self.settings_scroll = _FitScroll()
        self.settings_scroll.setWidget(settings)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QScrollArea.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        body.addWidget(self.settings_scroll, 0)
        preview = QWidget()
        preview.setStyleSheet("background:transparent;")
        preview.setLayout(self._preview_column(t))
        self.preview_scroll = _FitScroll()
        self.preview_scroll.setWidget(preview)
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFrameShape(QScrollArea.NoFrame)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        body.addWidget(self.preview_scroll, 1)
        self.card_lay.addLayout(body)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.hint = label("", ty.ui("xs"), t.muted_foreground)
        self.hint.setWordWrap(True)
        footer.addWidget(self.hint, 1)
        self.compose_button = Button(t, "Add to composer", glyph="plus")
        self.compose_button.setToolTip("Keep this chart, as set here, as a panel of a "
                                       "multi-panel figure: arrange and export it in the "
                                       "Composer section.")
        self.compose_button.clicked.connect(self.add_to_composer)
        footer.addWidget(self.compose_button)
        self.copy_button = Button(t, "Copy image", glyph="copy")
        self.copy_button.setToolTip("The figure as a 300 dpi image on the clipboard, "
                                    "to paste into slides or a message.")
        self.copy_button.clicked.connect(self.copy_image)
        footer.addWidget(self.copy_button)
        self.export_button = Button(t, "Export…", variant="default", glyph="download")
        self.export_button.clicked.connect(self.choose_and_export)
        footer.addWidget(self.export_button)
        self.cancel_button = self.apply_button = None
        if edit_panel is not None:
            # A PANEL'S OWN DIALOG: its words go back to the panel, or nowhere
            for button in (self.compose_button, self.copy_button, self.export_button):
                button.hide()
            self.cancel_button = Button(t, "Cancel")
            self.cancel_button.setToolTip("Close; the panel stays as it was.")
            self.cancel_button.clicked.connect(self._dismiss)
            footer.addWidget(self.cancel_button)
            self.apply_button = Button(t, "Apply to panel", variant="default", glyph="check")
            self.apply_button.setToolTip(f"Panel {edit_panel} takes these words and options; the "
                                         f"Composer draws the figure again.")
            self.apply_button.clicked.connect(self.apply_to_panel)
            footer.addWidget(self.apply_button)
        self.card_lay.addLayout(footer)
        self._sync_controls()
        # THE FIRST PREVIEW WAITS FOR THE DIALOG TO HAVE OPENED (the UI designer, 21 Oct 2026: "the dialog
        # is laggy when it appears"). Started here, its matplotlib build on the worker held the
        # interpreter through the 200 ms zoom-in and half the frames were dropped (6-7 frames,
        # gaps up to 71 ms; 13 frames at 16 ms without it, measured). Opened (`open`), it starts
        # when the intro lands (`opened`); a dialog built and never opened (headless, tests)
        # starts on the next turn of the event loop, or in `wait`.
        self._first_pending = True
        self._opening = False
        self.opened.connect(self._first_preview)
        # (the dialog as the timer's context: deleted first, the timer is dropped with it)
        QTimer.singleShot(0, self, lambda: None if self._opening else self._first_preview())

    def open(self) -> None:
        self._opening = True
        super().open()

    def _first_preview(self) -> None:
        if self._first_pending:
            self._render()

    # -- the settings column --------------------------------------------------------- #
    def _row(self, grid, r, name, widget, extra=None):
        title = label(name, ty.ui("sm", "medium"), self.t.foreground)
        grid.addWidget(title, r, 0, Qt.AlignLeft | Qt.AlignVCenter)
        cell = QHBoxLayout()
        cell.setSpacing(8)
        cell.addWidget(widget)
        if extra is not None:
            cell.addWidget(extra)
        cell.addStretch(1)
        grid.addLayout(cell, r, 1)
        #: Each row's widgets by its title (a panel's edit hides the composition's rows).
        self.row_widgets = getattr(self, "row_widgets", {})
        self.row_widgets[name] = [title, widget] + ([extra] if extra is not None else [])

    def _settings(self, t):
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        s = self.spec
        self.preset_choice = Segmented(t, [fx.PRESET_LABELS[p] for p in fx.PRESETS],
                                current=fx.PRESETS.index(s.preset), size="sm")
        self.preset_choice.changed.connect(lambda i: self._set(preset=fx.PRESETS[i],
                                                        font_pt=None, height_mm=None))
        self._row(grid, 0, "For", self.preset_choice)
        self.width_choice = Segmented(t, [fx.WIDTH_LABELS[w] for w in fx.WIDTHS],
                               current=fx.WIDTHS.index(s.width), size="sm")
        self.width_choice.setToolTip(WIDTH_TIP)
        self.width_choice.changed.connect(self._width_picked)
        self._row(grid, 1, "Width", self.width_choice)
        self.width_mm_field = FloatField(t, 20, 600, s.width_value(), step=1, decimals=0,
                                   suffix=" mm", width=128)
        self.width_mm_field.changed.connect(lambda v: self._set(width="custom", width_mm=float(v))
                                      if self.spec.width == "custom" else None)
        self.height_mm_field = FloatField(t, 20, 600, s.size_mm()[1], step=1, decimals=0,
                                    suffix=" mm", width=128)
        self.height_mm_field.changed.connect(lambda v: self._set(height_mm=float(v))
                                       if not self.auto_height.checked else None)
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
        self._row(grid, 2, "Size", holder)
        self.font_choice = Segmented(t, list(fx.FONTS), current=fx.FONTS.index(s.font), size="sm")
        self.font_choice.changed.connect(lambda i: self._set(font=fx.FONTS[i]))
        self.font_pt_field = FloatField(t, 4, 40, s.pt(), step=0.5, decimals=1, suffix=" pt",
                                  width=100)
        self.font_pt_field.changed.connect(lambda v: self._set(font_pt=float(v)))
        self._row(grid, 3, "Font", self.font_choice, self.font_pt_field)
        self.shows = {}
        # TWO LINES when there are four boxes (the Spatial Map's: Numbers, Scale bar and its length, Colour bar,
        # Labels -- one line cut "Labels" to "Labe" in the settings column), two and two; five (the jump
        # distances': Numbers, Grid, Legend, then its two residual plots), three and two; one line otherwise.
        options = fx.show_options(self.kind, self.payload)
        second_line = 3 if len(options) >= 5 else 2
        show_lines = QVBoxLayout()
        show_lines.setContentsMargins(0, 0, 0, 0)
        show_lines.setSpacing(6)
        show = QHBoxLayout()
        show.setSpacing(6)
        show_lines.addLayout(show)
        # Only the boxes that change THIS chart (`figure_export.show_options`).
        for index, (key, text) in enumerate(options):
            if len(options) >= 4 and index == second_line:
                show.addStretch(1)
                show = QHBoxLayout()
                show.setSpacing(6)
                show_lines.addLayout(show)
            # ticked as the chart shows it (Components follows the card until ticked here)
            box = Checkbox(t, checked=fx.show_checked(self.kind, self.payload, s, key))
            box.toggled.connect(lambda on, k=key: self._set(**{k: bool(on)}))
            self.shows[key] = box
            show.addWidget(box)
            show.addWidget(label(text, ty.ui("sm"), t.foreground))
            if key == "scale_bar":
                # Its LENGTH, typed (asked for: "let the user change the number and length of
                # the bar -> type 2"); empty follows the zoom, and says what it would pick.
                auto = fx.scale_bar_auto(self.payload)
                self.bar_length = _TextField(
                    t, "" if s.scale_bar_um is None else f"{s.scale_bar_um:g}",
                    f"auto, {auto:g} µm", width=110)
                self.bar_length.setToolTip("The scale bar's length in µm. Empty: a round "
                                           "length that follows the zoom.")
                self.bar_length.textChanged.connect(
                    lambda text: self._set(scale_bar_um=fx.parse_length(text)))
                box.toggled.connect(self.bar_length.setEnabled)
                self.bar_length.setEnabled(s.scale_bar)
                show.addWidget(self.bar_length)
                show.addWidget(label("µm", ty.ui("sm"), t.muted_foreground))
            show.addSpacing(8)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        show.addStretch(1)
        holder.setLayout(show_lines)
        if self.shows:                       # a table has nothing to show or hide
            self._row(grid, 4, "Show", holder)
        self.format_boxes = {}
        files = QHBoxLayout()
        files.setSpacing(6)
        for fmt in fx.FORMATS:
            box = Checkbox(t, checked=fmt in s.formats)
            box.toggled.connect(lambda _on: self._formats_changed())
            self.format_boxes[fmt] = box
            files.addWidget(box)
            files.addWidget(label(fmt.upper(), ty.ui("sm"), t.foreground))
            files.addSpacing(8)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        holder.setLayout(files)
        self._row(grid, 5, "Files", holder)
        self.dpi_choice = Segmented(t, [f"{d} dpi" for d in fx.DPIS],
                             current=fx.DPIS.index(s.dpi), size="sm")
        self.dpi_choice.changed.connect(lambda i: self._set(dpi=fx.DPIS[i]))
        self._row(grid, 6, "Images", self.dpi_choice)
        own_y, own_x = fx.default_titles(self.kind, self.payload)
        # THE FIGURE'S TITLE AND WHERE EACH AXIS STARTS AND ENDS (asked for: "change x/y
        # axis label, title, and x y axis SCALE, where it starts and ends"). Empty is the
        # chart's own; only the axes that have a numeric scale get the two boxes.
        # AUTOMATIC UNTIL TYPED IN: the field shows the project's name (and the condition's,
        # for a chart of one condition), `fx.auto_title`; the spec keeps None until the user
        # edits, so the figure follows a renamed condition. Emptied, the figure has none.
        self.title_field = _TextField(
            t, fx.auto_title(self.kind, self.payload) if s.title is None else s.title,
            "no title", width=300)
        self.title_field.setToolTip(
            "Centred over the figure, in bold. It starts as the project's folder name, with "
            "the condition's name for a chart of one condition. Type to change it; empty "
            "for none.")
        self.title_field.textChanged.connect(lambda text: self._set(title=text.strip()))
        self._row(grid, 7, "Title", self.title_field)
        self.y_title = _TextField(t, own_y if s.y_title is None else s.y_title, "no title",
                                  width=250)
        self.y_title.textChanged.connect(lambda text: self._set(y_title=text))
        self.x_title = _TextField(t, own_x if s.x_title is None else s.x_title, "no title",
                                  width=250)
        self.x_title.textChanged.connect(lambda text: self._set(x_title=text))
        self.range_fields = {}
        ranges = {axis: self._range_box(t, axis) for axis in fx.axis_ranges(self.kind)}
        if self.kind in fx.NO_TITLES:
            # A table has no axis: neither title is offered.
            for field in (self.y_title, self.x_title):
                field.setEnabled(False)
                field.setVisible(False)
        elif self.kind in fx.NO_Y_TITLE:
            # No y axis, and the one title names the colour bar: said so, rather than an
            # "X title" beside a greyed "Y title".
            self.y_title.setEnabled(False)
            self.y_title.setToolTip("This figure has no y axis.")
            self.y_title.setVisible(False)
            self._row(grid, 8, "Bar title", self.x_title)
        else:
            self.y_title.setToolTip("The y axis's title. Empty: none.")
            self.x_title.setToolTip("The x axis's title. Empty: none.")
            self._row(grid, 8, "Y axis", self.y_title, ranges.get("y"))
            self._row(grid, 9, "X axis", self.x_title, ranges.get("x"))
        names = QWidget()
        names.setStyleSheet("background:transparent;")
        rows = QVBoxLayout(names)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)
        self.name_fields = {}
        #: A box beside each name on a figure whose conditions can be left out (`figure_export.PICK_KINDS`; 5 Nov 2026,
        #: asked: "first i think we need to select which conditions to take, for example if i just wanna show two of
        #: them"): ticked, drawn; the spec's `left_out` is the unticked.
        self.pick_boxes = {}
        pick = self.kind in fx.PICK_KINDS
        #: The Names rows' headings, by their words (a grouped chart's "Group headers", "Column labels").
        self.name_headings = {}
        sections = fx.name_sections(self.kind, self.payload)
        heights = 0
        for heading, section in sections:
            if heading:
                words = label(heading, ty.ui("xs", "medium"), t.muted_foreground)
                self.name_headings[heading] = words
                rows.addWidget(words)
                heights += 20
            for condition in section:
                field = _TextField(t, s.shown_name(condition), condition, width=272 if pick else 300)
                field.setCursorPosition(0)
                field.setToolTip(
                    f"How {condition} is named in the figure. Empty: its own name. The numbers written beside it "
                    f"keep the real name." + (" Every place it is drawn follows: the headers, the labels under every "
                                              "group, the source data and the legend." if heading else ""))
                field.textChanged.connect(lambda _text: self._names_changed())
                self.name_fields[condition] = field
                if pick:
                    box = Checkbox(t, checked=condition not in s.left_out)
                    box.setToolTip(f"Draw {condition} in the figure. Unticked: left out (its numbers too).")
                    box.toggled.connect(lambda _on: self._picked())
                    self.pick_boxes[condition] = box
                    line = QHBoxLayout()
                    line.setContentsMargins(0, 0, 0, 0)
                    line.setSpacing(8)
                    line.addWidget(box)
                    line.addWidget(field)
                    line.addStretch(1)
                    rows.addLayout(line)
                else:
                    rows.addWidget(field)
        scroll = QScrollArea()
        scroll.setWidget(names)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        shown = min(len(self.name_fields), 4 if not self.name_headings else 5)
        scroll.setFixedHeight(max(1, shown) * (t.control_height + 6) + min(heights, 40))
        scroll.setFixedWidth(320)
        if self.name_fields:
            # A chart naming no condition (a cell image, a single map) shows no empty row.
            self._row(grid, 10, "Names", scroll)
        # WHICH FITS a fitted figure is drawn around (`FigureSpec.chosen_fit`): it starts on
        # the card's own (its Show, else BIC's); only those charts have the row.
        self.fit_choice = None
        choices = fx.fit_choices(self.kind, self.payload)
        self._fit_counts = [k for k, _words in choices]
        if choices and self.kind in fx.FIT_SET_KINDS:
            # TOGGLES, as the card's Show (30 Oct 2026): each count on or off, the last one on staying on
            self.fit_choice = Toggles(t, [words for _k, words in choices], size="sm", on=self._fit_lit())
            self.fit_choice.setToolTip("The fits drawn: click a count to turn it on or off. One fit shows its "
                                       "components and residuals; two or more are drawn each in its own "
                                       "colour (Cards: each condition as its card shows it). It starts as the "
                                       "card shows it.")
            self.fit_choice.pressed.connect(self._fit_pressed)
            self._row(grid, 11, "Fit", self.fit_choice)
        elif choices:
            counts = self._fit_counts
            current = fx.shown_fit(self.kind, self.payload, s)
            self.fit_choice = Segmented(
                t, [words for _k, words in choices], size="sm",
                current=counts.index(current) if current in counts else 0)
            self.fit_choice.setToolTip("The fit drawn, with its components and residuals. It starts as the "
                                       "card shows it.")
            self.fit_choice.changed.connect(lambda i, c=counts: self._set(chosen_fit=c[i]))
            self._row(grid, 11, "Fit", self.fit_choice)
        # THE Y UNIT of a jump-distance figure (30 Oct 2026, `figure_export.y_mode_choices`): it starts as the card
        # shows it; a change drops a typed y range (the unit changed) and puts the y title in the new unit
        self.y_mode_choice = None
        units = fx.y_mode_choices(self.kind)
        if units:
            self._y_modes = [m for m, _w in units]
            current = fx.shown_y_mode(self.kind, self.payload, s)
            self.y_mode_choice = Segmented(t, [w for _m, w in units], size="sm",
                                           current=self._y_modes.index(current))
            self.y_mode_choice.setToolTip("Probability per bin (as in SMTracker 2 and Spot-On), jumps per bin, or "
                                          "probability density per nm. It starts as the card shows it.")
            self.y_mode_choice.changed.connect(self._y_mode_picked)
            self._row(grid, 13, "Y unit", self.y_mode_choice)
        # THE GRID of the HMM state diagrams (5 Nov 2026, asked: "Option to order in x times x grid ?";
        # `figure_export.grid_choices`): Auto, or the columns chosen, the rows following. No such figure has a Fit row.
        # Its pills name rows x columns for the conditions KEPT, so a box ticked or unticked builds them again
        # (`_build_grid_choice`), in a holder of their own.
        self.grid_choice = None
        self._grid_values = []
        self._grid_holder = None
        if self.kind in fx.GRID_KINDS:
            self._grid_holder = QWidget()
            self._grid_holder.setStyleSheet("background:transparent;")
            holder = QHBoxLayout(self._grid_holder)
            holder.setContentsMargins(0, 0, 0, 0)
            self._row(grid, 11, "Grid", self._grid_holder)
            self._build_grid_choice()
        # THE MAP SCALE of the Spatial Map (5 Nov 2026, asked: "the cells get downscaled differently depending on the
        # text label size and other stuff? its hard to compare", then "one fixed size"; `FigureSpec.map_scale`): empty
        # is the standard (`figure_export.map_scale_standard`), typed that many mm per µm; the figure keeps its width
        # and height either way. The check line says the scale drawn. No such figure has a Fit or Grid row.
        self.map_scale = None
        if self.kind in fx.MAP_SCALE_KINDS:
            self.map_scale = _TextField(t, "" if s.map_scale is None else f"{s.map_scale:g}",
                                        "standard (6 at a journal column)", width=240)
            self.map_scale.setToolTip(
                "The cells' size on paper, in mm per µm of cell. Empty: the standard, 6 at an 89 mm "
                "journal column. If the columns do not fit the width, cells shrink to fit; the line under "
                "the preview gives the scale used.")
            self.map_scale.textChanged.connect(lambda text: self._set(map_scale=fx.parse_map_scale(text)))
            self._row(grid, 11, "Map scale", self.map_scale, label("mm per µm", ty.ui("sm"), t.muted_foreground))
        self._significance_rows(t, grid, 12)
        grid.setRowStretch(14, 1)
        if self.edit_panel is not None:
            # the COMPOSED figure's settings are not this panel's: off and out of sight (greyed, the size and
            # font fields and the file boxes still read as live); `EDIT_NOTE` says where they are set
            for widget in (self.preset_choice, self.width_choice, self.width_mm_field,
                           self.height_mm_field, self.auto_height, self.font_choice,
                           self.font_pt_field, self.dpi_choice, *self.format_boxes.values()):
                widget.setEnabled(False)
            for name in COMPOSITION_ROWS:
                for widget in self.row_widgets.get(name, ()):
                    widget.hide()
        return grid

    def _y_mode_picked(self, index: int) -> None:
        """A new y unit: a typed y range is dropped (its numbers meant the old unit), and the y title, unless typed,
        names the new one."""
        self._set(y_mode=self._y_modes[index], y_min=None, y_max=None)
        for field in self.range_fields.get("y", ()):
            field.blockSignals(True)
            field.setText("")
            field.blockSignals(False)
        if self.spec.y_title is None:
            self.y_title.blockSignals(True)
            self.y_title.setText(fx.jumps_y_title(self.kind, self.payload, self.spec))
            self.y_title.blockSignals(False)

    def _fit_lit(self) -> list:
        """The Fit toggles to light: the counts the spec draws, or "Cards" while the cards differ
        (`figure_export.shown_fit`)."""
        shown = fx.shown_fit(self.kind, self.payload, self.spec)
        lit = shown if isinstance(shown, tuple) else (shown,)
        return [self._fit_counts.index(k) for k in lit if k in self._fit_counts]

    def _fit_pressed(self, index: int) -> None:
        """A Fit toggle clicked: the spec's fits as `figure_export.fit_pressed` says, the pills lit again."""
        if 0 <= index < len(self._fit_counts):
            self._set(chosen_fit=fx.fit_pressed(self.kind, self.payload, self.spec, self._fit_counts[index]))
        self.fit_choice.set_on(self._fit_lit())

    # -- significance brackets (ruled 13 Oct 2026) ------------------------------------ #
    def _significance_rows(self, t, grid, row: int) -> None:
        """The Significance row, READ-ONLY: the brackets as the chart shows them.

        Asked for 14 Oct 2026: "add in the viewing options next to the plots a way to show
        significance in the plot ... then during export, it exports how u see it depending on
        the choices on the plot". The choices are set in the chart's Visibility options
        (Significance) and arrive here in the spec (`window._open_export`); this row says what
        will be drawn (the test, whose fraction, each pair's p) and where to change it. A chart
        that cannot carry a p says why, and the spec is then exported without brackets."""
        self.significance_words = None
        self.significance_status = None
        if not sig.offered(self.kind):
            return
        s = self.spec
        why = sig.unavailable(self.kind, self.payload, s.significance_test)
        if why and s.significance:
            # Asked for on the chart, but this chart cannot carry it (a headless export of
            # the same settings is refused by name, `figure_export._significance_plan`).
            self.spec = fx.with_changes(self.spec, significance=False)
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        # FitLabels (never a wrapped QLabel): a wrapped label here was given less height than its lines and drawn
        # over the row under it (found by the overnight tester, 4 Nov 2026)
        self.significance_words = fit_label("", ty.ui("sm"), t.foreground)
        self.significance_words.setMaximumWidth(330)
        lay.addWidget(self.significance_words)
        self.significance_status = fit_label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.significance_status.setMaximumWidth(330)
        lay.addWidget(self.significance_status)
        where = label("Set on the chart: Significance (right rail).", ty.ui("xs"),
                      t.muted_foreground)
        where.setWordWrap(True)
        where.setMaximumWidth(330)
        lay.addWidget(where)
        self._significance_why = why
        self._row(grid, row, "Significance", box)
        self._sync_significance()

    def _sync_significance(self) -> None:
        """The row's words and p lines back on what the spec holds."""
        if getattr(self, "significance_words", None) is None:
            return
        s = self.spec
        pairs = tuple(tuple(p) for p in s.significance_pairs)
        if self._significance_why:
            self.significance_words.setText("No brackets: " + self._significance_why + ".")
            self.significance_status.setText("")
            return
        if not (s.significance and pairs):
            self.significance_words.setText("No brackets on the chart.")
            self.significance_status.setText("")
            return
        test = sig.TEST_LABELS.get(s.significance_test, s.significance_test)
        if s.significance_test == "tost":
            test += f" within ±{s.significance_margin:g} pp"
        elif s.significance_test == "met":
            test += f" beyond ±{s.significance_margin:g} pp"
        says = {"stars": "stars", "p": "the exact p", "both": "stars and the exact p"}.get(
            s.significance_label, s.significance_label)
        if s.significance_test == "tost":
            says = says.replace("stars", "the equivalence mark")
        key = (f" ({sig.stars_sentence(s.significance_stars, s.significance_test, s.significance_margin)})"
               if s.significance_label != "p" else "")
        whose = sig.whose_words(self.kind, s.significance_population)
        self.significance_words.setText(
            f"As on the chart: {test}, {whose}, "
            f"{sig.correction_name(s.significance_correction)}-adjusted over {len(pairs)} pair(s); "
            f"the brackets show {says}{key}.")
        self.significance_status.setText("\n".join(sig.status_lines(
            self.kind, self.payload, pairs, s.significance_population, s.significance_test,
            stars=s.significance_stars, margin_pp=s.significance_margin,
            correction=s.significance_correction)))

    def _range_box(self, t, axis: str) -> QWidget:
        """"from [ ] to [ ]" for one axis: where it starts and ends, empty = the chart's."""
        s = self.spec
        low, high = (s.y_min, s.y_max) if axis == "y" else (s.x_min, s.x_max)
        fields = []
        for value, word in ((low, "start"), (high, "end")):
            field = _TextField(t, "" if value is None else f"{value:g}", "auto", width=74)
            field.setToolTip(f"Where the {axis} axis {word}s. Empty: the chart's own.")
            field.textChanged.connect(lambda _text, a=axis: self._range_typed(a))
            fields.append(field)
        self.range_fields[axis] = tuple(fields)
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(label("from", ty.ui("xs"), t.muted_foreground))
        lay.addWidget(fields[0])
        lay.addWidget(label("to", ty.ui("xs"), t.muted_foreground))
        lay.addWidget(fields[1])
        return box

    def _range_typed(self, axis: str) -> None:
        low, high = self.range_fields[axis]
        try:
            values = fx.parse_number(low.text()), fx.parse_number(high.text())
        except ValueError as exc:
            self.hint.setText(f"{axis} axis: {exc}")
            return
        self._set(**{f"{axis}_min": values[0], f"{axis}_max": values[1]})

    def _preview_column(self, t):
        col = QVBoxLayout()
        col.setSpacing(8)
        self.preview = QLabel()
        self.preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(f"background:{t.muted}; border-radius:{t.radius_md:g}px;"
                                   f" color:{t.muted_foreground};")
        self.preview.setText("Drawing…")
        self.preview.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        col.addWidget(self.preview)
        self.info = label("", ty.ui("xs", tabular=True), t.foreground)
        col.addWidget(self.info)
        # A FitLabel, its height pinned to its lines (found by the overnight tester, 4 Nov 2026: "No problems found at
        # this size ..." was half hidden behind the size line, a wrapped QLabel given one line's height)
        self.findings = fit_label("", ty.ui("xs"), t.muted_foreground)
        self.findings.setMaximumWidth(PREVIEW_W)
        col.addWidget(self.findings)
        col.addStretch(1)
        # Under the preview, not under the settings: with a few name fields the settings
        # column fills a laptop window and the note was cut off there.
        also = fit_label("Written beside the figure: its Source Data (an Excel workbook and a "
                         "CSV, the plotted values as numbers), a drafted legend, the analysis "
                         "provenance and the export settings.", ty.ui("xs"), t.muted_foreground)
        also.setMaximumWidth(PREVIEW_W)
        col.addWidget(also)
        if self.edit_panel is not None:
            also.hide()                    # editing a panel writes no file: the Composer's Export does
        return col

    # -- state ------------------------------------------------------------------- #
    def _set(self, **changes) -> None:
        candidate = fx.with_changes(self.spec, **changes)
        problem = candidate.problem()
        if problem:
            self.hint.setText(problem)
            # A refused pill must not stay lit (seen: "1.5 columns" raised beside a 183 mm
            # figure, the hint the only sign it was refused).
            QTimer.singleShot(0, self._sync_choices)
            return
        self.spec = candidate
        self.hint.setText("")
        self._sync_controls()
        if self.edit_panel is None:
            # (a panel's edit is that panel's: never the next export's settings)
            try:
                theme.remember(REMEMBER_KEY, self.spec.remembered_record())
            except OSError:
                pass
        show_redrawing(self.preview)
        self._debounce.start()

    def _build_grid_choice(self) -> None:
        """The Grid row's pills for the conditions the spec keeps (`figure_export.grid_choices`): Auto, then rows x
        columns; none, and the row hidden, where one condition is left."""
        if self._grid_holder is None:
            return
        if self.grid_choice is not None:
            discard(self.grid_choice)
            self.grid_choice = None
        grids = fx.grid_choices(self.kind, self.payload, self.spec)
        self._grid_values = [v for v, _w in grids]
        for widget in self.row_widgets.get("Grid", ()):
            widget.setVisible(bool(grids))
        if not grids:
            return
        current = self.spec.grid_columns if self.spec.grid_columns in self._grid_values else None
        self.grid_choice = Segmented(self.t, [w for _v, w in grids], size="sm",
                                     current=self._grid_values.index(current))
        self.grid_choice.setToolTip("Rows × columns. Auto: as many across as the width holds (four in a row at full "
                                    "width, two by two in a journal's column).")
        self.grid_choice.changed.connect(lambda i: self._set(grid_columns=self._grid_values[i]))
        self._grid_holder.layout().addWidget(self.grid_choice)
        self._grid_holder.setVisible(True)

    def _picked(self) -> None:
        """A box beside a name ticked or unticked: the unticked are left out of the figure; the last ticked one stays
        (a figure of nothing is no figure), and the Grid row counts the conditions kept."""
        left_out = tuple(c for c, box in self.pick_boxes.items() if not box.checked)
        if len(left_out) == len(self.pick_boxes):
            self.hint.setText("Keep at least one condition in the figure.")
            for c, box in self.pick_boxes.items():
                if c not in self.spec.left_out:
                    box.set_checked(True, animate=False)          # quietly: `set_checked` emits nothing
            return
        # a column count past the conditions kept falls back to Auto
        kept = len(self.pick_boxes) - len(left_out)
        columns = self.spec.grid_columns if (self.spec.grid_columns or 0) <= kept else None
        self._set(left_out=left_out, grid_columns=columns)
        self._build_grid_choice()

    def _names_changed(self) -> None:
        # a rename of a name no field shows (a grouped chart's full column key, renamed before) is kept
        kept = tuple((real, shown) for real, shown in self.spec.names if real not in self.name_fields)
        self._set(names=kept + tuple((c, f.text().strip()) for c, f in self.name_fields.items()
                                     if f.text().strip() and f.text().strip() != c))

    def _width_picked(self, index: int) -> None:
        width = fx.WIDTHS[index]
        if width == "custom":
            self._set(width="custom", width_mm=self.spec.width_value())
        else:
            self._set(width=width, width_mm=None)

    def _auto_height(self, on: bool) -> None:
        self._set(height_mm=None if on else float(self.spec.size_mm()[1]))

    def _formats_changed(self) -> None:
        chosen = tuple(f for f in fx.FORMATS if self.format_boxes[f].checked)
        if not chosen:
            self.hint.setText("Choose at least one file type.")
            return
        self._set(formats=chosen)

    def _sync_choices(self) -> None:
        """The journal and width pills back on what the spec holds."""
        for choice, value, options in ((self.preset_choice, self.spec.preset, fx.PRESETS),
                                       (self.width_choice, self.spec.width, fx.WIDTHS)):
            if choice.current() != options.index(value):
                choice.blockSignals(True)
                choice.select(options.index(value))
                choice.blockSignals(False)

    def _sync_controls(self) -> None:
        """Show what the spec now says, without re-entering `_set`."""
        s = self.spec
        width, height = s.size_mm()
        own = getattr(self, "edit_panel", None) is None      # editing a panel: the composition's, greyed
        for field, value, editable in ((self.width_mm_field, width, s.width == "custom"),
                                       (self.height_mm_field, height, s.height_mm is not None),
                                       (self.font_pt_field, s.pt(), True)):
            field.blockSignals(True)
            field.setValue(value)
            field.blockSignals(False)
            field.setEnabled(editable and own)
        self.dpi_choice.setEnabled(any(f in s.formats for f in ("png", "tiff")) and own)
        self._say_size(width, height)
        self._sync_significance()

    def _say_size(self, width: float, height: float) -> None:
        s = self.spec
        self.info.setText(f"{width:g} × {height:g} mm  ·  {s.font} {s.pt():g} pt  ·  "
                          + ", ".join(f.upper() for f in s.formats))

    # -- the preview, on a worker ---------------------------------------------------- #
    def _render(self) -> None:
        self._first_pending = False
        if not self.preview.pixmap().isNull():
            show_redrawing(self.preview)          # a redraw from anywhere: never the old figure beside new settings
        if self._running:
            self._again = True
            return
        self._running = True
        self._generation += 1
        self.preview.setToolTip("")
        self._threads.start(_PreviewJob(
            self._generation, partial(fx.build_figure, self.kind, self.payload, self.spec),
            self._signals))

    def _drawn(self, generation, png, findings, notes, error) -> None:
        self._running = False
        # The preview's build computed (and cached) any mixed-model p the lines were waiting
        # for: show it.
        self._sync_significance()
        if self._again:
            self._again = False
            self._render()
            return
        if error:
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"Could not draw this figure:\n{error}")
            return
        self._png = png
        image = QImage.fromData(png)
        if self.spec.height_mm is None and image.width():
            # AUTO HEIGHT IS WHAT WAS DRAWN: a table, a cell image or the maps set their own
            # height, and the line read the preset's default until now.
            width = self.spec.width_value()
            height = round(width * image.height() / image.width(), 1)
            self._say_size(width, height)
            self.height_mm_field.blockSignals(True)
            self.height_mm_field.setValue(height)
            self.height_mm_field.blockSignals(False)
        pixmap = QPixmap.fromImage(image).scaled(
            PREVIEW_W - 16, PREVIEW_H - 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pixmap)
        lines = [f"• {n}" for n in notes] + [f"⚠ {f}" for f in findings]
        journal = fx.PRESET_LABELS[self.spec.preset]
        self.findings.setText("\n".join(lines) if lines else
                              f"No problems found at this size: every text "
                              f"{self.spec.min_pt():g} pt or more ({journal}'s minimum), "
                              f"nothing overlapping, nothing cut off.")

    def wait(self, seconds: float = 10.0) -> None:
        """Until the preview in flight has landed AND the dialog is laid out around it (for tests and a scripted
        export). The landing sets the size line and the check line, and their layout requests are posted, not run:
        a grab straight after the loop drew the check line squeezed under "Written beside the figure" (3 Nov 2026,
        reported; the screen never shows that state, since Qt lays out before it paints)."""
        import time

        end = time.monotonic() + seconds
        if self._first_pending and not self._opening:
            self._first_preview()
        while (self._first_pending or self._running or self._debounce.isActive()) and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        # the chain of layout requests: the column, its scroll area, the card, the card's place in the window
        for _ in range(4):
            QApplication.processEvents()

    # -- output -------------------------------------------------------------------- #
    def add_to_composer(self) -> None:
        """Hand this chart, with this dialog's words, to the composer (the window adds it
        and says where it went)."""
        self.composer_requested.emit({"kind": self.kind, "payload": self.payload,
                                      "words": fx.panel_words(self.spec),
                                      "title": fx.panel_title(self.kind, self.payload),
                                      # set at 2/3 here, it is 2/3 there (`composer.width_for`)
                                      "width": composer.width_for(self.spec)})

    def apply_to_panel(self) -> None:
        """Editing a Composer panel: hand its new words back (the Composer writes them to that
        panel only and draws again), then close."""
        if self.edit_panel is None:
            raise RuntimeError("Apply to panel: this dialog is not editing a Composer panel")
        problem = self.spec.problem()
        if problem:
            self.hint.setText(f"Not applied: {problem}")
            return
        self.panel_applied.emit(fx.panel_words(self.spec))
        self._acted = True
        self._dismiss()

    def say(self, text: str) -> None:
        self.hint.setText(text)

    def copy_image(self) -> None:
        built = fx.build_figure(self.kind, self.payload, fx.with_changes(self.spec, dpi=300))
        png = fx.preview_png(built, round(built.size_mm()[0] / 25.4 * 300))
        QApplication.clipboard().setImage(QImage.fromData(png))
        self.hint.setText("Copied at 300 dpi. Paste it into a slide or a message.")

    def export_to(self, path) -> list:
        """Write the figure and everything beside it to ``path`` (no extension needed)."""
        built = fx.build_figure(self.kind, self.payload, self.spec)
        written = fx.write_figure(built, path, provenance=self.provenance, stamp=self.stamp)
        try:
            theme.remember(REMEMBER_FOLDER, str(Path(path).parent))
        except OSError:
            pass
        self._last_folder = Path(path).parent
        self.hint.setText(f"Wrote {len(written)} files to {Path(path).parent}")
        self.exported.emit([str(p) for p in written])
        return written

    def choose_and_export(self) -> None:
        folder = theme.remembered(REMEMBER_FOLDER) or str(Path.home())
        default = str(Path(folder) / self.payload.get("file_name", "figure"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", default, "Figure (*.pdf *.svg *.png *.tiff)")
        if not path:
            return
        written = self.export_to(path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(written[0]).parent)))
        self._acted = True
        self._dismiss()
