"""The Confirm screen's fixed bands: counts, acquisition, days, title, footer.

Every figure here is DERIVED from :mod:`shadcn.project`, never typed. The counts
row sums what the project says is included; the day strip reads the day records;
the footer counts what is unresolved. Two numbers on this screen cannot disagree,
because there is only ever one of them.

That matters more than it sounds. The first version of this prototype carried
its numbers as literals copied off the Paper artboard, and it read 28/28 and
151/151 forever -- which led to deleting two progress bars on the grounds that
they could never move. They move as soon as the data is real.
"""

from __future__ import annotations

from typing import ClassVar

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from ..theme import icons
from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.primitives import (
    Badge,
    Card,
    HoverFill,
    MiddleElided,
    Progress,
    breathe,
    col,
    day_badge,
    fit_label,
    icon,
    label,
    mix,
    row,
)
from ..widgets.primitives import discard

__all__ = [
    "FACTS",
    "ConfirmFooter",
    "CountsRow",
    "ParamRow",
    "ReplicateDays",
    "TitleRow",
    "band",
    "divider",
]

#: Acquisition settings. NOT in ``project`` because they are properties of the
#: microscope, not of the project: every condition on this screen was recorded
#: on the same instrument, and a per-project copy would invite them to drift.
#: The UNIT is a column of its own, not part of the value string. That is what
#: lets an acquisition value be set exactly like a count -- a big tabular figure
#: with a small muted qualifier beside it -- instead of rendering "106.0 nm" at
#: 24 px and shouting the unit as loudly as the number.
#: THE FIVE FACTS, and what a real project can answer. The mock typed values in
#: here; a loaded project reads them off `AcquisitionParams`, which comes from
#: the tracker's own files. Two of the mock's five had no counterpart and were
#: replaced by the fields that do:
#:
#:   Objective NA     -> PSF width. `psf_sigma_px` is u-track's own fitted PSF,
#:                       read from movieData.mat. NA is not metadata this
#:                       project reads, and a plausible 1.49 on a screen whose
#:                       job is to be CHECKED is the worst place to invent one.
#:   Min track length -> Localization precision, with its provenance in the
#:                       caption -- the σ a project opened with is load-bearing
#:                       here in a way a filter u-track already applied is not.
#:
#: TITLES SPELL THEIR SYMBOLS -- now a PREFERENCE, and it used to be forced.
#: Under Geist there was no Greek, so "PSF σ" rendered as an empty box. Inter
#: carries σ and τ as real glyphs (measured, see `theme.typography`), so the
#: words are kept because they read better to someone who does not already
#: know the notation -- not because the font cannot draw the symbol.
FACTS = [
    ("grid-2x2", "Pixel size", "pixel_size_nm", "nm", "{n} series, one value"),
    ("clock", "Frame interval", "frame_interval_ms", "ms",
     "the time between frames, used for every D"),
    ("aperture", "PSF width", "psf_sigma_px", "px",
     "u-track's own fit"),
    ("crosshair", "Search radius", "search_radius_px", "px",
     "Caps the largest D that can be measured"),
    ("ruler", "Localization precision", "localization_precision_nm", "nm",
     "source: {source}"),
]

#: What a fact shows when the files did not record it. NEVER a default: this
#: screen exists to be checked before a number is trusted.
UNRECORDED = "–"

#: The two card titles on this screen -- "Replicate days" and, in `groups`,
#: "Analysis groups". 16 px is `text-base`, the step the registry gives a
#: `CardTitle`; 18 was tried first and read as a page heading rather than a
#: section one, competing with "Confirm project" above it.
#:
#: 600 is `font-semibold`, which is what the registry gives a `CardTitle` -- so
#: the pair now lands exactly on the class rather than near it. A named step, so
#: the axis value and the `weight` argument agree rather than one overriding the
#: other (the axis wins when they differ, which is why 475 needed it). Kept on
#: the axis anyway, so both titles have one knob to turn.
SECTION_PX = 16
SECTION_WGHT = 600


def section_font():
    """The section-title font, with its FALLBACK NAME derived from the number.

    `typography.ui` takes both a named step and an axis value, and the axis wins
    when they disagree -- so a hard-coded ``"medium"`` beside ``wght=600``
    renders correctly here and one step light on any machine whose Inter is not
    the variable build, which is exactly where nobody would look. Picking the
    nearest named step keeps the fallback honest whatever `SECTION_WGHT` becomes.
    """
    name = min(ty.WEIGHTS, key=lambda k: abs(int(ty.WEIGHTS[k]) - SECTION_WGHT))
    return ty.ui("base", name, px=SECTION_PX, wght=SECTION_WGHT)

#: How many figures each fact is worth. A pixel size is a calibration and gets
#: its tenth; a fitted PSF gets two; a precision in nanometres gets none, because
#: nothing here resolves a fraction of one.
_DIGITS = {
    "pixel_size_nm": "{:.1f}",
    "psf_sigma_px": "{:.2f}",
    "localization_precision_nm": "{:.0f}",
}


def _fact(project, field: str, note: str):
    """One acquisition value and its caption, from the project's own params.

    `None` for the project (or for the field) is not an error: it is a screen
    drawn before a load, and the honest answer is a dash.
    """
    params = getattr(project, "params", None) if project is not None else None
    value = getattr(params, field, None) if params is not None else None
    if value is None:
        return UNRECORDED, "not recorded by the tracker"
    # ROUNDED TO WHAT THE NUMBER MEANS, not to what the float holds. A PSF fitted
    # to 0.931681 px is a fit, not a measurement to six figures, and a
    # localization precision of 14.6645 nm invites a reader to trust a decimal
    # the microscope never resolved.
    text = _DIGITS.get(field, "{:g}").format(value)
    source = str(getattr(params, "localization_sigma_source", "")).replace("_", " ")
    return text, note.format(n=getattr(project, "n_series", ""), source=source)


def band(widget: QWidget, name: str, css: str) -> QWidget:
    """Style one widget WITHOUT the rule reaching its children.

    An unscoped ``setStyleSheet("border-bottom: 1px solid ...")`` cascades: Qt
    applies it to the widget and to every descendant, so a header's rule draws
    an underline beneath each of its own labels. Scoping by object name confines
    it, and it is easy to get wrong because the unscoped version looks right
    until the widget has children.

    NO ``:hover`` here. A stylesheet pseudo-state was how the row highlights
    worked first, and it worked -- but a Qt stylesheet cannot TRANSITION, and
    ``TableRow`` is ``transition-colors``. :class:`primitives.HoverFill` owns
    that now, and owning it in one place is the point: two ways to give a widget
    a hover fill is how one of them ends up animated and the other does not.
    """
    widget.setObjectName(name)
    widget.setStyleSheet(f"QWidget#{name} {{ {css} }}")
    return widget


def divider(t, vertical: bool = True) -> QFrame:
    """A 1 px rule. A styled QFrame, not ``QFrame.VLine``, because the built-in
    line draws a two-tone bevel that no shadcn border has."""
    w = QFrame()
    if vertical:
        w.setFixedWidth(1)
    else:
        w.setFixedHeight(1)
    w.setStyleSheet(f"background:{t.border}; border:none;")
    return w


def spaced(n: int) -> str:
    """``5264`` -> ``5 264``. A thin space, so a four-figure count stays one
    word and does not wrap between its thousands and its hundreds."""
    return f"{n:,}".replace(",", " ")


class _CountCard(Card):
    """One derived count: head, value + bar + percent, note.

    Three lines, always. The bar rides the value row rather than taking a fourth
    line of its own -- a row is as tall as its tallest card, so a card that grows
    a line when its data dips below 100 % would resize the whole row under the
    user as they exclude series.
    """

    def __init__(self, t, glyph: str, name: str, parent=None):
        super().__init__(t, parent=parent)
        self.t = t
        inner = QVBoxLayout(self)
        inner.setContentsMargins(16, 16, 16, 16)
        inner.setSpacing(10)
        inner.addWidget(row(icon(glyph, icons.MD, t.muted_foreground),
                            label(name, ty.ui("sm", "medium"),
                                  t.muted_foreground),
                            None, spacing=8))

        self.value = label("", ty.value(), t.foreground)
        self.of = label("", ty.ui("sm"), t.muted_foreground)
        # shadcn's Progress is ``w-full``; inline there is no "full" to take, so
        # it takes the SLACK -- the stretch that would otherwise be empty.
        self.bar = Progress(t, 0.0, height=8)
        # Tabular too: this one also rewrites itself as series are ticked.
        self.pct = label("", ty.ui("sm", "medium", tabular=True), t.foreground)
        value_row = row(self.value, self.of, self.bar, self.pct, spacing=8)
        value_row.layout().setAlignment(Qt.AlignBottom)
        inner.addWidget(value_row)

        self.note = label("", ty.ui("xs"), t.muted_foreground)
        inner.addWidget(self.note)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def show_values(self, value: int, of: int, of_text: str, note: str,
                    show_pct: bool) -> None:
        frac = value / of if of else 0.0
        self.value.setText(spaced(value))
        self.of.setText(of_text)
        self.bar.value = max(0.0, min(1.0, frac))
        self.bar.update()
        self.pct.setText(f"{frac:.1%}" if show_pct else "")
        self.pct.setVisible(show_pct)
        self.note.setText(note)


class CountsRow(QWidget):
    """What the tool counted, against what is on disk.

    The denominator is always ``on_disk`` -- the whole folder -- so excluding a
    series makes the bar retreat rather than silently rescaling to a new total.
    A bar whose denominator moves with its numerator can only ever read 100 %.
    """

    def __init__(self, t, project, parent=None):
        super().__init__(parent)
        self.t, self.p = t, project
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self.series = _CountCard(t, "rows-3", "Series")
        self.cells = _CountCard(t, "scan", "Segmented cells")
        self.tracks = _CountCard(t, "activity", "Tracks kept")
        # The tracks card carries a percentage and the longest note, so it gets
        # twice the width. The stretch is hierarchy, not decoration.
        for card, weight in ((self.series, 1), (self.cells, 1), (self.tracks, 2)):
            lay.addWidget(card, weight)
        self.refresh()

    def refresh(self) -> None:
        p = self.p
        got, disk = p.totals(), p.on_disk()
        held, loose = len(p.excluded), len(p.unplaced())

        if not held and not loose:
            note = "Every series is in a group and included"
        else:
            parts = []
            if loose:
                parts.append(f"{loose} not in a group")
            if held:
                parts.append(f"{held} held out")
            note = " · ".join(parts)
        self.series.show_values(got["series"], disk["series"],
                                f"of {disk['series']} in the project", note,
                                show_pct=False)

        missing = disk["cells"] - got["cells"]
        self.cells.show_values(
            got["cells"], disk["cells"], f"of {spaced(disk['cells'])} outlined",
            "Every outlined cell sits in an included series" if not missing
            else f"{spaced(missing)} in series that are not included",
            show_pct=False)

        self.tracks.show_values(
            got["kept"], disk["found"], f"of {spaced(disk['found'])} found",
            f"{spaced(disk['found'] - got['kept'])} outside a cell outline",
            show_pct=True)


class ParamRow(Card):
    """The five acquisition facts, as ONE card with divided columns.

    One card rather than five, deliberately. These facts are read-only and
    cannot be acted on, where the counts decide whether you proceed; giving each
    of them its own card, as the counts have, flattened a hierarchy that exists.

    **The values are set exactly like the counts, and that changed when the
    views split.** They used to run 18 px against the counts' 24, and the
    smaller size WAS the hierarchy: read-only facts sitting quietly under the
    numbers that carry the decision. That argument depended on both being on one
    screen. Since the Record / Groups split they never appear together -- the
    counts are on the other view -- so all 18 px did was make the only numbers
    on this screen the smallest ones in the app, subordinated to a neighbour
    that is not there. The card itself still carries the hierarchy: one card
    with dividers, not five, and no progress bar or percentage.
    """

    def __init__(self, t, project=None, parent=None):
        super().__init__(t, parent=parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        outer.addLayout(lay)
        # THIS CONDITION'S OWN VALUES where the load gave them (`onboarding_view.condition_cards`, tester B1, 3 Nov
        # 2026): the project's merged ``params`` read 5 px for a condition linked at 8, 25 ms for one at 30
        cards = getattr(project, "cards", None) if project is not None else None
        for i, (glyph, name, field, unit, note) in enumerate(FACTS):
            value, note = cards[i] if cards and i < len(cards) else _fact(project, field, note)
            if i:
                lay.addWidget(divider(t))
            # Baseline-aligned, like the counts' value row: a 14 px unit
            # centred against a 24 px figure floats, and the eye reads the
            # mismatch before it reads either.
            value_row = row(label(value, ty.value(), t.foreground),
                            label(unit, ty.ui("sm"), t.muted_foreground),
                            None, spacing=6)
            value_row.layout().setAlignment(Qt.AlignBottom)
            lay.addWidget(col(
                row(icon(glyph, icons.SM, t.muted_foreground),
                    # 14 px muted -- `CardDescription`, which is what a metric's
                    # NAME is in shadcn's own stat card. Same for the counts.
                    label(name, ty.ui("sm", "medium"), t.muted_foreground),
                    None, spacing=7),
                value_row,
                # wrapped: a caption naming the other conditions' values must not widen the card
                fit_label(note, ty.ui("xs"), t.muted_foreground),
                spacing=5, margins=(20, 16, 20, 16)), 1)
        #: Where the conditions were not recorded alike (`onboarding_view.acquisition_differences`): said here, at
        #: the door, and no longer only in QC (asked by the tester, 3 Nov 2026).
        self.note = fit_label(getattr(project, "acquisition_note", "") or "", ty.ui("xs"), t.foreground)
        self.note.setVisible(bool(self.note.text()))
        # `band`, so the rule above the line does not cascade onto the label inside
        note_box = band(QWidget(), "paramNote", f"background:transparent; border-top:1px solid {t.border};")
        nl = QVBoxLayout(note_box)
        nl.setContentsMargins(20, 10, 20, 12)
        nl.addWidget(self.note)
        outer.addWidget(note_box)
        note_box.setVisible(bool(self.note.text()))


class ReplicateDays(Card):
    """What the files recorded. Read-only, and the strip says so.

    Days are ORDINAL, so their badges take a sequential ramp -- one hue at
    varying lightness, derived from the condition's own colour. shadcn has no
    ordinal scale (its badges are nominal variants), so this is the one element
    on the screen that is not drawn from shadcn's palette. That is a finding
    about the system's reach, not a compromise in the port.
    """

    #: Emitted after the table folds or unfolds, so a parent can re-fit.
    table_requested = Signal()

    #: DAY / SERIES / IMAGING DATE / FRAMES / CELLS / KEPT-FOUND / NOTE.
    #: Fixed slots so the rows form real columns; NOTE takes what is left.
    COLS: ClassVar = (86, 104, 118, 76, 58, 208, 0)
    #: IMAGING DAY, between REPLICATE and SERIES, only when the rows are saved replicates (`Project.record_days`):
    #: the replicate is what the analysis uses, the imaging day what dates it and proposed it.
    DAY_COL: ClassVar = 96

    #: `h-10` on the head and the same rhythm on a row, struck for 14 px type.
    #: FIXED, both of them: a row's height is a property of the design, not of
    #: how many rows the experiment happened to produce.
    HEAD_H: ClassVar = 42
    ROW_H: ClassVar = 41

    def __init__(self, t, project, fills, inks, parent=None):
        super().__init__(t, parent=parent)
        self.t, self.p = t, project
        self.fills, self.inks = fills, inks
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        head.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(head)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(28)

        # THE SAVED REPLICATES where there are any (`Project.record_rows`): two replicates imaged on one date are two
        # rows with that date. Else the imaging days, which are what the board proposes as replicates.
        note = ("What the files recorded, in the replicates you saved · read-only"
                if project.replicates is not None else "What the files recorded · read-only")
        lay.addWidget(col(
            row(icon("calendar", icons.MD, t.muted_foreground),
                label("Replicates", section_font(), t.foreground),
                Badge(t, str(len(project.record_rows())), t.muted, t.foreground),
                None, spacing=8),
            label(note, ty.ui("xs"), t.muted_foreground),
            spacing=3))

        # NO DAY STRIP IN THE HEADER.
        #
        # A scrolling row of D1..D7 chips used to live here, from when this card
        # was folded by default and the strip was the only place the days
        # appeared. Since the Record view opens the table to full height it is
        # always visible underneath, so the strip was showing the same seven
        # tags twice, 34 px apart, and the header had to scroll sideways to do
        # it. The count badge stays: that is a summary, not a repeat.
        lay.addStretch(1)

        self._link = label("Show full table", ty.ui("control", "medium"),
                           t.foreground)
        self._link.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._link.setCursor(Qt.PointingHandCursor)
        self._link.mouseReleaseEvent = lambda _e: self.flip()
        lay.addWidget(self._link)
        outer.addWidget(head)

        # THE STRIP, UNFOLDED IN PLACE -- not a view switch.
        #
        # This is a different table from the one in Analysis groups, and the
        # difference is the row: one row per DAY here, one per SERIES there.
        # The strip already shows every day; what it cannot show is what each
        # day is made of, so the table opens under it rather than replacing
        # anything or sending you to another panel.
        self._table_host = QWidget()
        self._table_host.setStyleSheet("background:transparent;")
        self._table_lay = QVBoxLayout(self._table_host)
        self._table_lay.setContentsMargins(0, 0, 0, 0)
        self._table_lay.setSpacing(0)
        self._open = False

        # THE COLUMN HEADER SITS OUTSIDE THE SCROLL AREA, as a sibling above
        # it, exactly for `groups.GroupsPanel`'s reason: in FILL MODE (below)
        # this table can genuinely grow taller than the window and scroll, and
        # a header built at the top of `_table_host` -- the widget the scroll
        # area scrolls -- would disappear with the first row scrolled past it.
        # `_build_table` fills this once; folded, `flip`/`set_fill` hide it in
        # step with `_table_scroll` so a closed table shows no header either.
        self._table_head_host = QWidget()
        self._table_head_host.setStyleSheet("background:transparent;")
        self._table_head_lay = QVBoxLayout(self._table_head_host)
        self._table_head_lay.setContentsMargins(0, 0, 0, 0)
        self._table_head_lay.setSpacing(0)
        self._table_head_host.setVisible(False)
        outer.addWidget(self._table_head_host)

        # The table goes in a scroll area so FILL MODE can give it the rest of
        # the window without the day count deciding the panel's height. Folded,
        # the area is sized to its content and never scrolls, so the inline
        # behaviour is unchanged.
        self._table_scroll = QScrollArea()
        self._table_scroll.setWidget(self._table_host)
        self._table_scroll.setWidgetResizable(True)
        self._table_scroll.setFrameShape(QScrollArea.NoFrame)
        self._table_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table_scroll.setStyleSheet(
            "QScrollArea { background:transparent; }")
        # THE SAME EASED WHEEL THE LANES HAVE. `smoothscroll.attach` is what
        # gives a scroll area its eased wheel and eased scrollbar; the groups
        # board attaches it to all three of its areas, and this one was the odd
        # area out -- one table in the flow that jumped where everything else
        # glided.
        from ..widgets import smoothscroll

        self.table_scroller = smoothscroll.attach(self._table_scroll,
                                                  background=t.card)
        self._table_scroll.setVisible(False)
        self._fill = False
        outer.addWidget(self._table_scroll)
        self._outer = outer
        self._build_table()
        self._fit_table()

    def set_fill(self, on: bool) -> None:
        """Fill mode: the table is ALWAYS open and takes the rest of the view.

        The read-only view has nothing under the day record, so a fold control
        there would collapse the table to reveal empty space -- the toggle stops
        meaning anything, and the link is hidden. In the mixed page the fold is
        the point, because Analysis groups is underneath.
        """
        self._fill = on
        self._link.setVisible(not on)
        if on:
            self._open = True
            self._table_scroll.setVisible(True)
            self._table_head_host.setVisible(True)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        else:
            self._table_scroll.setVisible(self._open)
            self._table_head_host.setVisible(self._open)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._fit_table()

    def _fit_table(self) -> None:
        """Sized to content when folded-in-place, free to stretch when filling.

        Without the first half the scroll area would report a sizeHint of its
        own and the panel would grow a scrollbar for a table that fits.
        """
        if self._fill:
            self._table_scroll.setMinimumHeight(120)
            self._table_scroll.setMaximumHeight(16777215)
            self._table_scroll.setSizePolicy(QSizePolicy.Preferred,
                                             QSizePolicy.Expanding)
        else:
            # ARITHMETIC, NOT `sizeHint()`.
            #
            # CORRECTED, and the original note is kept because its NUMBER was
            # right while its CAUSE was wrong. It read: "that one call cost 3.8
            # SECONDS the first time it ran on a real window -- profiled, a
            # single `QWidget.sizeHint` at 3.842 s of a 3.86 s construction, and
            # 25 ms on every later one", and concluded that laying out the
            # subtree was inherently expensive. It was not. `sizeHint` forces
            # layout, layout shapes text, and the text held a U+2009 the bundled
            # face had no glyph for -- so the 3.8 s was Qt walking 2 942 font
            # families for a fallback. The face is Inter now and the SAME call
            # measures 7.2 ms. See `theme.typography`.
            #
            # The arithmetic stays anyway, on its own merits: 0.1 ms against
            # 7.2, and deterministic.
            #
            # It is also unnecessary now. The head and the rows have FIXED
            # heights (`HEAD_H`, `ROW_H`), so the table's height is a sum rather
            # than a question -- and a number this widget already knows beats
            # asking Qt to lay out the whole subtree to find out.
            #
            # HEAD_H is no longer part of THIS sum -- the header sits in
            # `_table_head_host`, a sibling above the scroll area rather than
            # its first row, so it is not part of what this height covers.
            hint = self.ROW_H * len(self.p.record_rows()) + self._footer_height()
            self._table_scroll.setFixedHeight(hint)
            self._table_scroll.setSizePolicy(QSizePolicy.Preferred,
                                             QSizePolicy.Fixed)

    def _footer_height(self) -> int:
        """The summary strip under the rows, measured once from its own font."""
        from qtpy.QtGui import QFontMetricsF

        return int(QFontMetricsF(ty.ui("xs")).height()) + 24

    def flip(self) -> None:
        """Open or close the table.

        The state is a FLAG, not ``isVisible()``. A widget on a non-current page
        of a QStackedWidget reports invisible however it was set, so reading
        visibility back means the toggle inverts itself the moment this
        condition is not the tab you are looking at -- and it reads correctly
        while you test it on the first tab.
        """
        self._open = not self._open
        self._table_scroll.setVisible(self._open)
        self._table_head_host.setVisible(self._open)
        self._link.setText("Hide table" if self._open else "Show full table")
        self.table_requested.emit()

    def recolour(self, fills, inks) -> None:
        """Point the day tags at a new ramp. The tags belong to the condition,
        so they follow its colour -- a tag left on the old hue would be claiming
        a condition it no longer matches."""
        self.fills, self.inks = fills, inks
        # REPAINT, do not rebuild. See ``Badge.set_colours``.
        for badge in self.findChildren(Badge):
            if badge.day_index is not None:
                i = min(badge.day_index - 1, len(fills) - 1)
                badge.set_colours(fills[i], inks[i])

    # -- the full table -------------------------------------------------------
    def _build_table(self) -> None:
        t, p = self.t, self.p
        while self._table_lay.count():
            item = self._table_lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                discard(widget)
        while self._table_head_lay.count():
            item = self._table_head_lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                discard(widget)

        heads = []
        names = ["Replicate", "Series", "Imaging date", "Frames", "Cells", "Tracks kept / found", "Note"]
        widths = list(self.COLS)
        if p.replicates is not None:
            names.insert(1, "Imaging day")
            widths.insert(1, self.DAY_COL)
        for name, w in zip(names, widths):
            # `TableHead` is `font-medium text-foreground` -- FOREGROUND, not
            # muted. Worth checking rather than remembering: older shadcn did
            # set `text-muted-foreground` here, and a header greyed out like a
            # caption is what made this table read as small print.
            lab = label(name, ty.ui("sm", "medium"), t.foreground)
            if w:
                lab.setFixedWidth(w)
            heads.append(lab)
        # `h-10` on the head, `p-2` on the cells -- the registry's own rhythm,
        # re-struck for 14 px type.
        head = row(*heads[:-1], heads[-1], spacing=14,
                   margins=(20, 11, 20, 11))
        head.setFixedHeight(self.HEAD_H)
        # `_table_head_lay`, NOT `_table_lay` -- see the note on
        # `_table_head_host` in `__init__`. This is the fixed sibling above the
        # scroll area, so the column names stay on screen while the day rows
        # scroll underneath them.
        self._table_head_lay.addWidget(band(head, "daythead",
                                            f"background:{t.subtle};"
                                            f" border-top:1px solid {t.border};"
                                            f" border-bottom:1px solid {t.border};"))
        for i, d in enumerate(p.record_rows()):
            if i and not i % 6:                        # see `groups.Lane.rebuild`
                breathe()
            self._table_lay.addWidget(self._day_row(d))
        # THE ROWS KEEP THEIR HEIGHT AND THE SURPLUS GOES BELOW THEM. A
        # QVBoxLayout hands spare height to whatever will take it, so on a
        # project with three days each row grew to ~100 px -- a table whose row
        # height is a function of how many rows it has. The stretch is what the
        # surplus lands on instead; the footer stays where it was, at the foot
        # of the card.
        self._table_lay.addStretch(1)
        self._table_lay.addWidget(self._table_footer())

    def _day_row(self, d) -> QWidget:
        t, p = self.t, self.p
        tot = p.record_totals(d.index)
        rows = p.record_series(d.index)
        span = (f"{rows[0].name[-3:]} – {rows[-1].name[-3:]}" if rows else "–")
        day_cell = row(day_badge(t, self.fills, self.inks, d.tag, d.index),
                       None, spacing=0)
        day_cell.setFixedWidth(self.COLS[0])

        cells = [day_cell]
        if p.replicates is not None:
            # BOTH LABELS: the replicate (what the analysis uses) and the imaging day(s) it was taken on
            days = p.record_days(d.index)
            on = row(*[day_badge(t, self.fills, self.inks, day.tag, day.index) for day in days], None, spacing=4) \
                if days else label("–", ty.ui("sm"), t.muted_foreground)
            on.setFixedWidth(self.DAY_COL)
            cells.append(on)
        for text, w, colour in (
                (span, self.COLS[1], t.foreground),
                # THE DATE, AND WHETHER IT IS THE REAL ONE. The raw `.nd2` /
                # `.stk` and `movieData.mat`'s `acquisitionDate_` say when a
                # movie was taken; `createTime_` -- the import time -- says when
                # u-track read it, and `daymap` treats it as a PROXY. So does
                # this: a proxy date is drawn muted and says what it is on hover.
                # It is not even right about the day: protein A/Malate's three
                # sessions, eight months apart, were all imported on one date.
                (d.on.isoformat() if d.on else "–", self.COLS[2],
                 t.muted_foreground if (d.date_is_proxy or not d.on)
                 else t.foreground),
                (spaced(d.frames), self.COLS[3], t.muted_foreground),
                (str(tot["cells"]), self.COLS[4], t.muted_foreground)):
            lab = label(text, ty.ui("sm"), colour)
            lab.setFixedWidth(w)
            if w == self.COLS[2] and d.on and d.date_is_proxy:
                lab.setToolTip("The time u-track imported the movie, not the acquisition date. "
                               "The raw .nd2/.stk was not read for this replicate.")
            cells.append(lab)

        # Kept / found gets its own bar -- the same mark the counts row and the
        # series cards use. One meaning per mark, across the whole screen.
        share = tot["kept"] / tot["found"] if tot["found"] else 0.0
        kept = row(Progress(t, share, height=8, width=56),
                   label(f"{tot['kept']} / {tot['found']}", ty.ui("sm"),
                         t.muted_foreground),
                   label(f"{share:.0%}", ty.ui("sm", tabular=True),
                         t.muted_foreground),
                   None, spacing=9)
        kept.setFixedWidth(self.COLS[5])
        cells.append(kept)
        cells.append(label(d.note or "–", ty.ui("sm"), t.muted_foreground))

        line = row(*cells, spacing=14, margins=(20, 9, 20, 9))
        line.setFixedHeight(self.ROW_H)
        name = f"dayrow{d.index}"
        line.setObjectName(name)
        # `TableRow`: `border-b transition-colors hover:bg-muted/50`.
        HoverFill(line,
                  lambda fill, n=name: (f"QWidget#{n} {{ background:{fill};"
                                        f" border-bottom:1px solid {t.border}; }}"),
                  # `t.card`, NOT "transparent", and the difference is real:
                  # a fade from rgba(0,0,0,0) interpolates through ALPHA, so the
                  # row passes through semi-transparent greys instead of between
                  # two solid colours. CSS transitions the resolved colour, and
                  # the resolved colour of a transparent row on this card IS the
                  # card. Same pixels at rest, correct ones in between.
                  rest=t.card, hover=mix(t.muted, t.card, 0.5))
        return line

    def _table_footer(self) -> QWidget:
        """Says what the table IS, and totals it."""
        t, p = self.t, self.p
        disk = p.on_disk()
        rates = [d for d in (p.record_totals(x.index) for x in p.record_rows())
                 if d["found"]]
        spread = [r["kept"] / r["found"] for r in rates]
        lo, hi = (min(spread), max(spread)) if spread else (0.0, 0.0)
        line = row(label(f"{len(p.record_rows())} replicates · {disk['series']} series · "
                         f"{spaced(disk['cells'])} cells", ty.ui("xs"),
                         t.muted_foreground),
                   None,
                   label(f"keep rate {lo:.0%}–{hi:.0%} across replicates",
                         ty.ui("xs"), t.muted_foreground),
                   spacing=14, margins=(20, 10, 20, 12))
        return band(line, "dayfoot", "background:transparent;")

class TitleRow(QWidget):
    """The project's identity, and the palette it is drawn in."""

    palette_requested = Signal()

    def __init__(self, t, project, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.title = label("Confirm project", ty.ui("2xl", "semibold"),
                           t.foreground)
        # SHORTENED IN THE MIDDLE, never a minimum width: a long project path (no spaces to wrap at) held the window
        # at 1661 px on a 1366 px screen (the GUI sweep, 5 Nov 2026); the whole path is the hover
        self.path = MiddleElided(project.path, ty.ui("sm"), t.muted_foreground)
        lay.addWidget(self.title)
        lay.addWidget(self.path, 1)
        # THE PALETTE PICKER ENDS THE ROW, hard right against the margin.
        # A "Choose a different folder" button used to sit outboard of it and
        # was removed: its `folder_requested` signal was declared and emitted
        # and CONNECTED TO NOTHING, so the button did nothing when pressed --
        # the way back is Cancel in the footer, which does work. The signal
        # went with it rather than being left for someone to wire up later.
        lay.addWidget(label("Palette", ty.ui("sm"), t.muted_foreground))
        self.palette = self._palette()
        lay.addWidget(self.palette)

    def _palette(self) -> QWidget:
        t = self.t
        field = QFrame()
        field.setObjectName("select")
        field.setStyleSheet(
            f"QFrame#select {{ background:{t.card};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px; }}")
        field.setFixedHeight(t.control_height)
        fl = QHBoxLayout(field)
        fl.setContentsMargins(12, 0, 12, 0)
        fl.setSpacing(10)
        swatches = QWidget()
        swatches.setStyleSheet("background:transparent;")
        sw = QHBoxLayout(swatches)
        sw.setContentsMargins(0, 0, 0, 0)
        sw.setSpacing(3)
        self._swatch_lay = sw
        self._palette_name = label("Okabe-Ito", ty.ui("sm"), t.foreground)
        fl.addWidget(swatches)
        fl.addWidget(self._palette_name)
        fl.addWidget(icon("chevron-down", icons.SM, t.muted_foreground))
        field.setCursor(Qt.PointingHandCursor)
        field.mouseReleaseEvent = lambda _e: self.palette_requested.emit()
        return field

    def show_palette(self, colours, name: str) -> None:
        """Preview the active palette in the field itself.

        A picker that does not show what it currently holds makes you open it to
        find out, which is the one thing the field is there to save.
        """
        while self._swatch_lay.count():
            item = self._swatch_lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                discard(widget)
        for hexcode in colours:
            dot = QWidget()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background:{hexcode}; border-radius:2px;")
            self._swatch_lay.addWidget(dot)
        self._palette_name.setText(name)

    def set_project(self, project) -> None:
        self.path.set_full_text(project.path)


class ConfirmFooter(QWidget):
    """What opening the workspace will do, and whether anything is unresolved.

    The status line is DERIVED, not decorative: it counts series with no group
    and days with a recorded note. "Nothing needs your attention" is a claim the
    footer has to earn on every refresh.
    """

    cancelled = Signal()
    opened = Signal()

    def __init__(self, t, project, parent=None):
        super().__init__(parent)
        self.t, self.p = t, project
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(24)
        self.lead = label("", ty.ui("sm"), t.foreground)
        self.sub = MiddleElided("", ty.ui("xs"), t.muted_foreground)     # the cache's path: as the title row's
        lay.addWidget(col(self.lead, self.sub, spacing=3), 1)

        self.status_icon = icon("check", icons.MD, t.muted_foreground)
        self.status = label("", ty.ui("sm"), t.muted_foreground)
        lay.addWidget(row(self.status_icon, self.status, spacing=7))

        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self.cancelled.emit)
        lay.addWidget(cancel)
        open_btn = Button(t, "Open workspace", variant="default")
        open_btn.clicked.connect(self.opened.emit)
        lay.addWidget(open_btn)
        self.refresh()

    def refresh(self) -> None:
        p = self.p
        self.lead.setText(
            "Nothing is fitted yet. Opening the workspace loads the tables and "
            "writes a cache into the project folder.")
        # THE CACHE PATH, AND NOTHING ELSE. The mock said "numbers are
        # invented" here, which was true of it and is a LIE about a loaded
        # project: every figure on it is read from the files.
        # The OPENED folder's, where `load_project` writes it; it named the condition's folder, where nothing is
        # written (tester B15, 3 Nov 2026)
        self.sub.set_full_text(getattr(p, "cache_dir", "") or (f"{p.path}" + chr(92) + ".smtracker_warm"))
        loose = len(p.unplaced())
        noted = [d for d in p.record_rows() if d.note]
        if loose:
            self.status.setText(f"{loose} series not in a group")
        elif noted:
            # the record view's rows: the saved replicates, else the imaging days (B14: "2 replicate with a note")
            what = "replicate" if p.replicates is not None else "imaging day"
            self.status.setText(f"{len(noted)} {what}{'s' if len(noted) != 1 else ''} with a note")
        else:
            self.status.setText("Nothing needs your attention")
        glyph = "check" if not loose else "alert"
        self.status_icon.name = glyph
        self.status_icon.update()
