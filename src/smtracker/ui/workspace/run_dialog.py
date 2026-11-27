"""The apparent-D and HMM Run dialogs' shared frame: SQD's two pages, General and Fits (4 Nov 2026).

Asked 4 Nov 2026: "ok for appd and hmm the population fit picker per condition is there. but its scrollable with only 2
visible conditions per screen. maybe we can improve that? either a 2nd page, or a better way to fit more conditions onto
one page". MEASURED before (offscreen, ten conditions, a maximised window's client area): 1 condition in view at
1366 x 768, 2 at 1920 x 1080 at 125 %, 3 at 1920 x 1080. Two causes, both of layout:

* the rows sat in a `QScrollArea`, whose size hint Qt caps at 36 x 24 font heights, so the card was 626 px wide whatever
  the rows needed and every row WRAPPED its Force same D across pills onto a second line (76 px a condition);
* the settings, each with its help line, took the card's height above the rows, and the rows' scroll area, held under
  256 px, was squeezed below that by a window of 697 px.

NOW SQD's DESIGN (`compute.ComputeDialog`): two pages behind SQD's line tabs (`compute.LineTabs`, General · Fits, the
same glyphs), the settings on General and, on Fits, ONE LINE PER CONDITION -- [include box,] name, its models (or state
count) and its own Force same D across -- under a line of column titles and the Every condition line, which sets every
row (its Force same D across too, as SQD's does). The rows show as many as the window holds (`FitsPage.set_room`, from
`RunDialog._fit_window` at `open` and on the window's resize, as SQD's `ComputeDialog._fit_window`), `MIN_VISIBLE_ROWS`
at least, and scroll the rest. The scroll area answers its rows' own width (`_RowsScroll`), so the card is as wide as
a row's one line. Every line is a `WrapRow` (CLAUDE.md: a row of controls wraps, never cuts) built from the same widths
-- every name as wide as the widest (`align_names`), the same trays, the same gaps -- so its columns stand over each
other by construction, and a window too narrow for one line wraps every line alike rather than cutting it.

THE PAGES' HEIGHT IS THE PAGE IN FRONT'S (`RunDialog.show_page`): the page behind is `Ignored` vertically, so the
`QStackedWidget` sizes to the one shown and the card recentres; both stay `Preferred` sideways, so the card keeps one
width on both tabs. Every open starts on General, as SQD's does.

The rules stay where they were -- `appd_fits.run_config` / `hmm_run.run_config` and `workspace_view`'s toggles, read
by each dialog's `_sync` and `config`. Nothing here decides a value.
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QSize, Qt, Signal
from qtpy.QtWidgets import (QHBoxLayout, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from ...workspace_view import ROW_SHARED_D, SHARED_D_NAME
from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import Checkbox, FitScroll, fit_label, label
from ..widgets.settings import InfoLink
from ..widgets.wraprow import WrapRow, align_names
from .compute import ComputePanel, ConditionRow, LineTabs, SharedDChoice
from .toolbar import rem

__all__ = ["FitRow", "FitsPage", "MIN_VISIBLE_ROWS", "NAME_CHARS", "PAGES", "RunDialog", "one_line_height",
           "short_name"]

#: The dialog's pages, in order: (key, tab text, glyph) -- SQD's tabs, SQD's glyphs (`compute.ComputePanel`)
PAGES = (("general", "General", "sliders-horizontal"), ("fits", "Fits", "flask-conical"))
#: Condition rows shown whatever the window's room (SQD's)
MIN_VISIBLE_ROWS = ComputePanel.MIN_VISIBLE_ROWS
#: Between two condition rows
ROW_SPACING = 6
#: A name longer than this is cut, whole on hover
NAME_CHARS = 28
#: Between the columns: SQD's row's gaps (`ConditionRow`), the include box to the name, the name to the models, the
#: models to the Force same D across
BOX_GAP, LEAD_GAP, COLUMN_GAP = ConditionRow.BOX_GAP, ConditionRow.LEAD_GAP, ConditionRow.COLUMN_GAP


def short_name(name: str) -> str:
    """A condition's name as its row prints it: whole up to `NAME_CHARS`, else cut with an ellipsis."""
    name = str(name)
    return name if len(name) <= NAME_CHARS else name[:NAME_CHARS - 1] + "…"


def _transparent() -> QWidget:
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    return w


def one_line_height(widget: QWidget) -> int:
    """A `WrapRow` line's height on ONE line: its tallest item. Never the `WrapRow`'s own height, which follows the
    breaks of its LAST geometry -- read just after a line widened (a set-wide value lifted, two pills back for one) it
    counted every line as wrapped, and the Fits page showed 4 rows of 10, not 9."""
    lay = widget.layout()
    return max((lay.itemAt(i).sizeHint().height() for i in range(lay.count())), default=0)


def _indented(widget: QWidget, left: int) -> QWidget:
    """``widget`` behind ``left`` px: the wider gap before the Force same D across column, inside a `WrapRow` whose
    spacing is one number."""
    holder = _transparent()
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(left, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(widget)
    return holder


class FitRow(QWidget):
    """One line of the Fits page: [include box,] the name, a choice tray (the models' `Toggles`, the state count's
    `Segmented`: the caller's, connected by the caller) and its own Force same D across (`compute.SharedDChoice`,
    SQD's row control). The Every condition line is one too (``muted``, its name in the titles' ink)."""

    #: ``(key, on)``: the include box moved (a user's click)
    included = Signal(str, bool)
    #: ``(key, index in ROW_SHARED_D)``: the user picked this row's own Force same D across
    shared_changed = Signal(str, int)

    def __init__(self, t, key: str, name: str, choice: QWidget, *, include: bool | None = None,
                 shared_d: str | None = ROW_SHARED_D[0], muted: bool = False, parent=None):
        super().__init__(parent)
        self.key = key
        self.setStyleSheet("background:transparent;")
        row = WrapRow(self, spacing=LEAD_GAP, line_spacing=rem(0.4))
        self._row = row
        #: THE LEAD, ONE GROUP (it never wraps inside): the include box and the name
        self.lead = _transparent()
        lead = QHBoxLayout(self.lead)
        lead.setContentsMargins(0, 0, 0, 0)
        lead.setSpacing(BOX_GAP)
        self.include = None
        if include is not None:
            self.include = Checkbox(t, checked=bool(include))
            self.include.toggled.connect(lambda on: self.included.emit(self.key, bool(on)))
            lead.addWidget(self.include)
        self.title = label(short_name(name), ty.ui("sm", "medium"), t.muted_foreground if muted else t.foreground)
        if short_name(name) != str(name):
            self.title.setToolTip(str(name))
        lead.addWidget(self.title)
        row.addWidget(self.lead)
        self.choice = choice
        row.addWidget(choice)
        self.shared = SharedDChoice(t, shared_d)
        self.shared.setToolTip(f"This condition's own {SHARED_D_NAME}" if not muted
                               else f"Every condition's own {SHARED_D_NAME}")
        self.shared.changed.connect(lambda v: self.shared_changed.emit(self.key, ROW_SHARED_D.index(v)))
        row.addWidget(_indented(self.shared, COLUMN_GAP - LEAD_GAP))

    def line_height(self) -> int:
        """The row's height on one line (`one_line_height`): what the Fits page budgets."""
        return one_line_height(self)


class _RowsScroll(FitScroll):
    """The rows' scroll area: `primitives.FitScroll` (its width hint is its rows', not Qt's cap of 36 font heights,
    which held the card at 626 px and wrapped every row), with the vertical scroll bar's width counted only while it
    shows (a card with no bar keeps no empty strip at its right, as SQD's Fits page), and its HEIGHT the budget
    `FitsPage.set_room` gives it -- whole rows -- its minimum the first `MIN_VISIBLE_ROWS` of them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { background:transparent; }")
        self._budget: int | None = None
        self._least: int = 0
        self._scrolls = False

    def set_budget(self, budget: int | None, least: int, scrolls: bool) -> None:
        self._budget = None if budget is None else int(budget)
        self._least = int(least)
        self._scrolls = bool(scrolls)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded if scrolls else Qt.ScrollBarAlwaysOff)
        self.updateGeometry()

    def _bar(self) -> int:
        return super()._bar() if self._scrolls else 0

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return hint if self._budget is None else QSize(hint.width(), self._budget)

    def minimumSizeHint(self) -> QSize:
        least = super().minimumSizeHint()
        return QSize(least.width(), self._least if self._budget is None else min(self._least, self._budget))


class _GeneralScroll(FitScroll):
    """The General page in a scroll area (13 Nov 2026): the GUI sweep found Clustering's General page, nine settings
    each with its help line, 555 px tall in a card that needs 726 against a 1366 x 768 laptop's 697, so its rows were
    squeezed. Only the Fits page scrolled. Now the page's hint is the whole page (a tall window shows all of it, no
    bar) and its minimum `FitScroll.MIN_HEIGHT`: a short window shrinks this and the settings scroll. The bar's width
    is counted only while it shows, as on the Fits page, so the card keeps its width in a window that holds it all."""

    def __init__(self, page: QWidget, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("QScrollArea { background:transparent; }")
        self.setWidget(page)

    def _bar(self) -> int:
        return super()._bar() if self.verticalScrollBar().isVisible() else 0


class FitsPage(QWidget):
    """The Fits page: its help line, the line saying why the rows are disabled (a set-wide Force same D across), the
    column titles, the Every condition line (``every``, a `FitRow`), a hairline, and the condition rows (`set_rows`)
    in `_RowsScroll`, as many shown as `set_room` allows."""

    def __init__(self, t, *, help_line: str, help_more: str, choice_title: str, every_choice: QWidget,
                 include: bool = False, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.info = InfoLink(t, help_line, help_more)
        lay.addWidget(self.info)
        lay.addSpacing(10)
        #: Why every row is disabled: a set-wide Force same D across (`workspace_view.set_wide_note`); hidden otherwise,
        #: its space with it (its gap is its own bottom margin). A `FitLabel`: a plain wrapped label would make the
        #: pages height-for-width, which a `QStackedWidget` asks of every page (`primitives.FitLabel`)
        self.note = fit_label("", ty.ui("xs"), t.muted_foreground)
        self.note.setContentsMargins(0, 0, 0, 8)
        self.note.hide()
        lay.addWidget(self.note)
        # THE COLUMN TITLES, each over its column: an empty lead as wide as the rows' (`align`), the choice's title as
        # wide as its tray, then the Force same D across title behind the same wider gap as the rows' pills
        self._titles = _transparent()
        titles = WrapRow(self._titles, spacing=LEAD_GAP, line_spacing=rem(0.4))
        self._titles_lead = _transparent()
        titles.addWidget(self._titles_lead)
        self.choice_title = label(choice_title, ty.ui("sm", "medium"), t.muted_foreground)
        titles.addWidget(self.choice_title)
        self.shared_title = label(SHARED_D_NAME, ty.ui("sm", "medium"), t.muted_foreground)
        titles.addWidget(_indented(self.shared_title, COLUMN_GAP - LEAD_GAP))
        lay.addWidget(self._titles)
        lay.addSpacing(4)
        #: THE EVERY CONDITION LINE: a row of the table that sets the others (SQD's)
        self.every = FitRow(t, "", "Every condition", every_choice, include=True if include else None,
                            shared_d=None, muted=True)
        lay.addWidget(self.every)
        lay.addSpacing(8)
        rule = QWidget()
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"background:{t.border}; border:none;")
        lay.addWidget(rule)
        lay.addSpacing(8)
        self._host = _transparent()
        self._rows_lay = QVBoxLayout(self._host)
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        self._rows_lay.setSpacing(ROW_SPACING)
        self._rows_lay.addStretch(1)
        self.scroll = _RowsScroll()
        self.scroll.setWidget(self._host)
        lay.addWidget(self.scroll, 1)
        self.rows: list = []
        #: The most height this page may take (`set_room`), None for no limit
        self._room: int | None = None

    # -- the rows ------------------------------------------------------------------------------------------------- #
    def set_rows(self, rows) -> None:
        """The condition rows, in order (built once, with the dialog); every name column then as wide as the widest."""
        self.rows = list(rows)
        for row in self.rows:
            self._rows_lay.insertWidget(self._rows_lay.count() - 1, row)
        self.align()
        self._fit()

    def align(self) -> None:
        """ONE NAME COLUMN (SQD's Fits page; the overnight tester, 4 Nov 2026: a long name pushed its row's pills out
        of line): every row's name and the Every condition line's as wide as the widest, so every line's columns
        start at one x; the titles' lead as wide as a row's lead, the choice's title as wide as its tray."""
        align_names([r.title for r in self.rows] + [self.every.title])
        self._titles_lead.setFixedWidth(self.every.lead.sizeHint().width())
        self.choice_title.setFixedWidth(max(self.every.choice.sizeHint().width(),
                                            self.choice_title.sizeHint().width()))

    def add_set_wide(self, control: QWidget) -> None:
        """FORCE SAME D ACROSS ON TOP, SQD's (5 Nov 2026, asked: "in hmm (and maybe appd fit) on the conditions tab,
        there is not the simultaneous fit across all conditions like in sqd"): the General page's setting -- Whole data
        set the fit of every condition together -- one value in two places, as SQD's Fits page has it
        (`ComputePanel.fits_shared_d`). ``control`` is the dialog's second tray of the General page's pills; the dialog
        keeps the two on one value. Set before the page is shown; the line counts in `chrome_h` as one line."""
        self.wide_line = _transparent()
        row = QHBoxLayout(self.wide_line)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(LEAD_GAP)
        self.wide_title = label(SHARED_D_NAME, ty.ui("sm", "medium"), self.t.muted_foreground)
        row.addWidget(self.wide_title)
        row.addWidget(control)
        row.addStretch(1)
        lay = self.layout()
        lay.insertWidget(0, self.wide_line)
        lay.insertSpacing(1, 12)

    def set_note(self, text: str) -> None:
        """The line over the titles (`workspace_view.set_wide_note`), hidden when empty; the rows' budget follows."""
        self.note.setText(text)
        if self.note.isHidden() == (not text):
            return
        self.note.setVisible(bool(text))
        self.layout().invalidate()
        self._fit()

    # -- the height ----------------------------------------------------------------------------------------------- #
    def chrome_h(self) -> int:
        """This page's fixed height: every item of its layout but the rows' scroll area, the gaps included."""
        lay = self.layout()
        total = lay.contentsMargins().top() + lay.contentsMargins().bottom()
        for i in range(lay.count()):
            item = lay.itemAt(i)
            if item.widget() is self.scroll:
                continue
            if item.widget() in (self._titles, self.every):
                total += one_line_height(item.widget())
                continue
            if item.widget() is self.note and not self.note.isHidden():
                # wrapped at the rows' width, the page's (its margin included: `QLabel.heightForWidth` counts it)
                total += self.note.heightForWidth(max(self.scroll.sizeHint().width(), 200))
                continue
            total += item.sizeHint().height()
        return total

    def set_room(self, room) -> None:
        """THE ROWS FIT THE WINDOW: ``room`` is the most height this page may take in the window the dialog is open
        in (`RunDialog._fit_window`), None for no limit; the page then shows as many whole rows as it holds,
        `MIN_VISIBLE_ROWS` at least, and scrolls the rest."""
        self._room = None if room is None else int(room)
        self._fit()

    def visible_rows(self) -> int:
        """How many rows the budget shows whole."""
        return getattr(self, "_visible", len(self.rows))

    def _fit(self) -> None:
        n = len(self.rows)
        if not n:
            self._visible = 0
            self.scroll.set_budget(0, 0, False)
            return
        heights = [r.line_height() for r in self.rows]
        pitch = max(heights) + ROW_SPACING
        everything = sum(heights) + ROW_SPACING * (n - 1)
        visible = n
        if self._room is not None:
            fit = (self._room - self.chrome_h() + ROW_SPACING) // pitch
            visible = max(min(n, int(fit)), min(n, MIN_VISIBLE_ROWS))
        self._visible = visible
        least = sum(heights[:min(n, MIN_VISIBLE_ROWS)]) + ROW_SPACING * (min(n, MIN_VISIBLE_ROWS) - 1)
        if visible >= n:
            self.scroll.set_budget(everything, least, False)
        else:
            self.scroll.set_budget(sum(heights[:visible]) + ROW_SPACING * (visible - 1), least, True)
        self.updateGeometry()


class RunDialog(OverlayDialog):
    """The frame: the title and its close button, the line under it, SQD's tabs over two pages -- General
    (``general_lay``, filled with `add_setting`) and Fits (`set_fits`) -- and the footer: the refusal (``hint``),
    Cancel and Run. A subclass builds its controls into the pages and keeps its rules; transactional like every Run
    dialog here, Cancel and the corner X discard the draft."""

    run = Signal(dict)

    #: The room kept between the card and its window's top and bottom edges (SQD's `ComputeDialog._WINDOW_ROOM`)
    _WINDOW_ROOM = 8

    def __init__(self, t, title: str, subtitle: str, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label(title, ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        self.subtitle = label(subtitle, ty.ui("sm"), t.muted_foreground)
        self.subtitle.setWordWrap(True)
        self.card_lay.addWidget(self.subtitle)
        self.tabs = LineTabs(t, PAGES, current=PAGES[0][0])
        self.tabs.changed.connect(self.show_page)
        self.card_lay.addWidget(self.tabs)
        self.general = _transparent()
        self.general_lay = QVBoxLayout(self.general)
        self.general_lay.setContentsMargins(0, 0, 0, 0)
        self.general_lay.setSpacing(0)
        self.fits: FitsPage | None = None
        self._pages = QStackedWidget()
        self._pages.addWidget(_GeneralScroll(self.general))
        self.card_lay.addWidget(self._pages)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        self.hint = label("", ty.ui("xs"), t.muted_foreground)
        self.hint.setWordWrap(True)
        footer.addWidget(self.hint, 1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self._run_btn = Button(t, "Run", variant="default")
        self._run_btn.clicked.connect(self._on_run)
        footer.addWidget(cancel)
        footer.addWidget(self._run_btn)
        self.card_lay.addLayout(footer)

    # -- the pages ------------------------------------------------------------------------------------------------ #
    def add_setting(self, name: str, control: QWidget, help_line: str, help_more: str, extra=()) -> InfoLink:
        """One General setting: its name, its control (and ``extra`` beside it, e.g. a field) on a line that wraps,
        its help line under it. Returns the help line."""
        if self.general_lay.count():
            self.general_lay.addSpacing(12)
        holder = _transparent()
        row = WrapRow(holder, spacing=rem(0.75), line_spacing=rem(0.4))
        name_label = label(name, ty.ui("sm", "medium"), self.t.muted_foreground)
        name_label.setMinimumWidth(rem(9))
        row.addWidget(name_label)
        row.addWidget(control)
        for widget in extra:
            row.addWidget(widget)
        self.general_lay.addWidget(holder)
        self.general_lay.addSpacing(4)
        link = InfoLink(self.t, help_line, help_more)
        self.general_lay.addWidget(link)
        return link

    def set_fits(self, fits: FitsPage) -> None:
        self.fits = fits
        self._pages.addWidget(fits)
        self.show_page(self.page())

    def page(self) -> str:
        """The page in front: "general" or "fits"."""
        return self.tabs.current() or PAGES[0][0]

    def show_page(self, key: str) -> None:
        """``key``'s page in front, its tab lit; the pages as tall as that page (the other `Ignored` vertically), as
        wide as the wider."""
        keys = [k for k, _text, _glyph in PAGES]
        index = keys.index(key) if key in keys else 0
        index = min(index, self._pages.count() - 1)
        for i in range(self._pages.count()):
            self._pages.widget(i).setSizePolicy(
                QSizePolicy.Preferred, QSizePolicy.Preferred if i == index else QSizePolicy.Ignored)
        self._pages.setCurrentIndex(index)
        self.tabs.select(keys[index])
        self._pages.updateGeometry()
        if self.isVisible():
            self._resync_after_resize()

    def open(self) -> None:
        """Every open starts on General (SQD's), with the card fitted to its window (`_fit_window`)."""
        self.show_page(PAGES[0][0])
        self._fit_window()
        super().open()

    def _fit_window(self) -> None:
        """THE CARD FITS THE WINDOW IT OPENS IN (SQD's `ComputeDialog._fit_window`): the Fits page may take the
        window's height less the card's own chrome (title, line, tabs, footer, margins) and `_WINDOW_ROOM` above and
        below, so its rows scroll rather than push the card past the window. Without a parent window nothing is
        limited."""
        if self.fits is None:
            return
        window = self.parentWidget()
        if window is None or window.height() <= 0:
            self.fits.set_room(None)
            return
        self._card.layout().activate()
        chrome = self._card.sizeHint().height() - self._pages.sizeHint().height()
        self.fits.set_room(window.height() - 2 * self._WINDOW_ROOM - chrome)

    def eventFilter(self, obj, event):
        """The window resized while the card is open: the card fits it again (`_fit_window`)."""
        handled = super().eventFilter(obj, event)
        if obj is self.parentWidget() and event.type() == QEvent.Resize and self.isVisible():
            self._fit_window()
        return handled

    def _on_run(self) -> None:                  # pragma: no cover - every subclass has its own
        raise NotImplementedError
