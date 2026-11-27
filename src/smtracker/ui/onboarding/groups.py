"""Analysis groups: the one thing on this screen you are here to decide.

Everything else on the Confirm screen reports; this panel edits. It carries the
two views of the same data (cards and table), the selection that spans them, and
the held-out tick that takes a series out of the fit without forgetting where it
was placed.

**One selection, two views.** ``Selection`` is a plain QObject holding names, so
switching between cards and table cannot lose what you picked -- the views are
two renderings of one model rather than two models that must be kept in step.
That is also why the selection bar lives on the panel and not inside either
view.

**Pair replicates, opt-in** (5 Nov 2026, asked: "optional pair ... in day
picker?", "a tick box for activate pairing in day picker"). One switch for the
whole project (`Pairing`), shown under every condition's board header; while it
is ticked each lane carries a Pair row, its tag ("A", ...) naming its partner
replicate in the other conditions (`model.Group.pair`, `free_tags`), and
"Propose from imaging days" fills the tags from shared imaging dates
(`model.propose_pairs`) for a person to keep or clear. The rules live in the
model; this draws and edits them.
"""

from __future__ import annotations

from typing import ClassVar

from qtpy.QtCore import QMimeData, QObject, QPoint, QRectF, Qt, QVariantAnimation, Signal
from qtpy.QtGui import QColor, QDrag, QPainter
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...onboarding_model import free_tags, pair_note
from ...onboarding_view import composition_words
from .. import theme
from ..theme import icons
from ..widgets import smoothscroll
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.dialogs import RegroupDialog
from ..widgets.menus import (FilterChip, Panel, menu_note, menu_row,
                             menu_section, retriggered, styled_menu)
from .panels import band, section_font, spaced
from ..widgets.primitives import (
    breathe,
    TRANSITION_MS,
    Badge,
    Card,
    Checkbox,
    EndElided,
    HoverFill,
    Progress,
    Surface,
    day_badge,
    draw_shadow,
    icon,
    label,
    mix,
    row,
    tw_ease,
)
from ..widgets.primitives import discard

__all__ = [
    "Checkbox",
    "GroupsPanel",
    "Pairing",
    "GroupsTable",
    "Lane",
    "Selection",
    "SelectionBar",
    "SeriesCard",
    "ShadowCell",
]


class Selection(QObject):
    """Which series are picked, as names.

    Names rather than widgets, because the same series is drawn by a card in one
    view and a table row in the other; a selection of widgets would belong to
    whichever view built it.
    """

    #: Emitted with the set of names whose state actually CHANGED.
    #:
    #: The payload is what keeps a click cheap. Repainting every card on every
    #: toggle cost 127 ms with 83 cards alive, because a stylesheet write forces
    #: Qt to recompute that widget's style -- and 82 of those cards had not
    #: changed. Telling listeners WHICH names moved lets them touch only those.
    changed = Signal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.names: set[str] = set()

    def toggle(self, name: str) -> None:
        self.names.symmetric_difference_update({name})
        self.changed.emit({name})

    def clear(self) -> None:
        if self.names:
            gone = set(self.names)
            self.names.clear()
            self.changed.emit(gone)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)


class Pairing(QObject):
    """The board's Pair replicates switch: ONE for every condition's board (the tags pair replicates ACROSS
    conditions), holding the conditions' models so a lane can say which other conditions use a tag. Off unless a
    saved tag says it was on."""

    #: The switch moved: every board shows or hides its pair rows.
    toggled = Signal(bool)
    #: A tag changed somewhere (a lane's menu, or a proposal): every lane redraws its tag.
    tags_changed = Signal()
    #: "Propose from imaging days" was pressed on a board: the page applies `model.propose_pairs` to every condition.
    propose_requested = Signal()

    def __init__(self, projects=(), parent=None):
        super().__init__(parent)
        self.projects = list(projects)
        self.on = any(g.pair for p in self.projects for g in p.groups)

    def set_on(self, on: bool) -> None:
        on = bool(on)
        if on != self.on:
            self.on = on
            self.toggled.emit(on)

    def users(self, tag: str, condition: str) -> list:
        """The other conditions holding a replicate tagged ``tag``."""
        return [p.condition for p in self.projects
                if p.condition != condition and any(g.pair == tag for g in p.groups)]


class ShadowCell(Surface):
    """A container whose only job is to paint one Checkbox's ``shadow-xs``.

    A shadow is painted by the nearest Surface ANCESTOR, never by the widget
    itself -- Qt clips a child to its own rect and the shadow lives outside it.
    In a :class:`SeriesCard` that ancestor is the card, which sits right above
    the tick and works unaided. In the TABLE the nearest one was
    ``GroupsPanel``, four levels and a ``QScrollArea`` up: geometrically it
    still maps, but the panel paints long before the row does, so the row's own
    background covers the shadow completely. Measured -- 255 under the box in
    every state.

    So the table's tick gets a Surface of its own, directly around it. It is
    ``shadowed=False``: it draws the shadows of what is INSIDE it, and has none
    of its own.
    """

    def __init__(self, t, child, parent=None):
        super().__init__(t, radius=0.0, shadowed=False, parent=parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(child)


class SeriesCard(Surface):
    """One series: identity above, quantity below.

    The fixed 64 px bar keeps the percent and the counts in one vertical lane
    across every row -- gap alone would let a wider series name push them out of
    alignment, which is the usual way a list of rows stops looking like a list.
    """

    clicked = Signal(str)
    held = Signal(str, bool)

    def __init__(self, t, project, series, fills, inks, selection, parent=None):
        # A series row is shadcn's ITEM, not a Card: ``rounded-md`` with a
        # border, which the registry gives every size of Item. It was drawn at
        # ``rounded-sm`` -- the MENU-ITEM radius -- one step tighter than the
        # primitive it actually is.
        super().__init__(t, radius=t.radius_md, shadowed=False, parent=parent)
        self.t, self.p, self.s, self.sel = t, project, series, selection
        self.setObjectName("srow")
        self.setCursor(Qt.PointingHandCursor)
        self._press = None
        #: Built on the first `_paint_state`, then re-pointed rather than
        #: rebuilt -- a new animation per repaint would drop the one running.
        self._hover_fill = None
        self._edge_now = t.border
        self._paint_state()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 9, 10, 9)
        lay.setSpacing(7)

        day = self.p.day(series.day) if series.day else None
        # The replicate tag always; the calendar date only when the files gave
        # one, which a loaded project never does -- see `adapt`.
        tag = ("no date" if day is None else
               f"{day.tag} · {day.on.day} {day.on.strftime('%b')}" if day.on
               else day.tag)
        chip = (day_badge(t, fills, inks, tag, series.day, weight="medium")
                if day else
                Badge(t, tag, t.muted, t.muted_foreground, weight="medium"))
        self.tick = Checkbox(t, checked=series.name not in project.excluded)
        self.tick.toggled.connect(lambda on: self.held.emit(series.name, on))
        lay.addWidget(row(self.tick,
                          label(series.name, ty.ui("sm", "medium"),
                                t.foreground),
                          None,
                          chip, spacing=8))
        lay.addWidget(row(Progress(t, series.keep_rate, height=8, width=64),
                          label(f"{series.keep_rate:.0%}",
                                ty.ui("xs", "medium"), t.foreground),
                          None,
                          label(f"{series.kept} / {series.found} · "
                                f"{series.cells} cells",
                                ty.ui("xs"), t.muted_foreground),
                          spacing=8))

    def _paint_state(self) -> None:
        """Fill, edge, and the hover the registry gives an INTERACTIVE Item.

        `itemVariants` carries `[a]:hover:bg-accent/50` -- guarded on the Item
        being a link, i.e. on it being something you can act on. This card is:
        it selects, it drags, it holds a series out. So it takes that hover, and
        a static Item elsewhere would not.

        Not applied while SELECTED, for the reason `GroupsTable._paint_row`
        gives: a picked card dropping to a half-strength fill under the pointer
        reads as letting go of it.
        """
        t = self.t
        picked = self.s.name in self.sel
        out = self.s.name in self.p.excluded
        edge = t.primary if picked else t.border
        rest = t.muted if picked else t.card
        hover = rest if picked else mix(t.accent, t.card, 0.5)
        if self._hover_fill is None:
            # `itemVariants` is `transition-colors duration-100` -- FASTER than
            # the 150 ms default a Button gets, and the registry says so
            # explicitly. A row in a list is crossed in passing, so it settles
            # sooner than a control you aim at.
            self._hover_fill = HoverFill(
                self,
                lambda fill: (f"QFrame#srow {{ background:{fill};"
                              f" border:1px solid {self._edge_now};"
                              f" border-radius:{t.radius_sm:g}px; }}"),
                rest=rest, hover=hover, ms=100)
        self._edge_now = edge
        self._hover_fill.set_colours(rest, hover)
        self.setWindowOpacity(1.0)
        for child in self.findChildren(QWidget):
            child.setEnabled(True)
        self.setProperty("held", out)

    def refresh(self) -> None:
        """Repaint, and re-read the tick from the model.

        The tick matters because the two views share one project: holding a
        series out from the table has to show on its card, and the card is not
        the widget that was clicked.
        """
        self._paint_state()
        want = self.s.name not in self.p.excluded
        self.tick.set_checked(want)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):
        """Start a drag once the pointer has actually travelled.

        Guarded by ``startDragDistance`` because without it every click becomes
        a one-pixel drag and the card can never simply be selected -- the two
        gestures begin identically and are told apart only by how far they go.
        """
        if not (e.buttons() & Qt.LeftButton) or self._press is None:
            return
        travelled = (e.position().toPoint() - self._press).manhattanLength()
        if travelled < QApplication.startDragDistance():
            return
        # Dragging a card that is part of the selection drags the WHOLE
        # selection; dragging an unselected one drags just it. Anything else
        # makes a multi-select you built silently do nothing.
        names = (sorted(self.sel.names) if self.s.name in self.sel
                 else [self.s.name])
        mime = QMimeData()
        mime.setText(SERIES_MIME + ",".join(names))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.setHotSpot(e.position().toPoint())
        drag.exec(Qt.MoveAction)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit(self.s.name)


#: The drag payload prefix. A prefix rather than a custom MIME type so the
#: payload is legible in a debugger and cannot be confused with dropped text.
SERIES_MIME = "smtracker/series:"


class _PairNote(EndElided):
    """The Pair row's note: its whole width where the lane has it, else shortened in the MIDDLE, so the date
    verdict at its end ("· dates differ") stays in sight."""

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, max(1, self.width())))


class Lane(Card):
    """One analysis group, as a column of series.

    Bordered but NOT shadowed: shadcn nests cards without nesting elevation, and
    two stacked shadows read as a lifted thing on a lifted thing -- a claim
    about hierarchy this screen does not make.
    """

    picked = Signal(str)
    held = Signal(str, bool)
    dropped = Signal(list, str)

    #: The visible height of one lane's card list. Everything else about the
    #: panel's height is derived from it, so there is one number to tune.
    #: The card list's FLOOR, reached only at the smallest window the layout
    #: allows. It is not a height and not a preference -- the list takes
    #: whatever the window gives it above this.
    #:
    #: The number sets the window's own minimum height, because with no page
    #: scroll area the view's minimum propagates straight to the window: at 250
    #: the window could not be shorter than 799 px, which is taller than a
    #: 768-line laptop screen. 122 is two cards, and puts the floor at 671.
    FLOOR_H = 122

    #: A lane never narrows past this. Named because the board's SCROLL STEPS
    #: are derived from it: the unit of travel on a board of lanes is a lane,
    #: not a pixel count carried over from a vertical page.
    MIN_W = 300

    #: The gap between lanes, matching ``_lanes_lay``'s spacing. ``MIN_W + GAP``
    #: is the PITCH -- MEASURED at exactly 316 px at every window size tried
    #: (1200 / 1456 / 1900), because the lanes hold their minimum and the board
    #: scrolls rather than squeezing them.
    GAP = 16

    def __init__(self, t, project, group, fills, inks, selection, parent=None, pairing=None):
        # A lane IS a Card, so it takes the Card radius -- ``rounded-xl``.
        # ``shadowed=False`` stays: shadcn nests cards without nesting
        # elevation, and that is a claim about SHADOW, not about corners.
        super().__init__(t, shadowed=False, parent=parent)
        self.t, self.p, self.g = t, project, group
        self.fills, self.inks, self.sel = fills, inks, selection
        # FIXED, NOT A MINIMUM. A minimum is only a size when there are enough
        # lanes to press against it: six over a 1456 px window hit it and the row
        # scrolls, but THREE share the surplus and each lane comes out 150 px
        # wider than the design, which makes the number of groups change how a
        # group looks. Fixed keeps a lane one width whatever the arrangement,
        # and `_lanes_lay`'s trailing stretch takes the leftover.
        self.setFixedWidth(self.MIN_W)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.tags = QWidget()
        self.tags.setStyleSheet("background:transparent;")
        self._tags_lay = QHBoxLayout(self.tags)
        self._tags_lay.setContentsMargins(0, 0, 0, 0)
        self._tags_lay.setSpacing(4)
        self._composition = label("", ty.ui("xs"), t.muted_foreground)
        # 14, NOT the 16 a `CardTitle` takes. The one place this design steps
        # DOWN from the registry, and deliberately: a lane is a card nested
        # inside "Analysis groups", six-across in a ~300 px column. At 16 it
        # ranked level with the section that contains it, so the board read as
        # six sections rather than one section holding six lanes. The top-level
        # card titles ("Analysis groups", "Replicate days", "Recent projects")
        # keep the registry's 16; a title INSIDE one of them gives way to it.
        head = row(label(group.name, ty.ui("sm", "semibold"), t.foreground),
                   self.tags, None, self._composition,
                   spacing=8, margins=(12, 11, 12, 11))
        # BANDS TAKE THEIR HINT; the slack goes to one explicit stretch.
        #
        # Without this a lane stretched to fill its column divided the height
        # EQUALLY between head, rows and foot -- three 314 px thirds, with a
        # 42 px header floating in the middle of one of them. A QVBoxLayout
        # distributes surplus among whatever will accept it, so the way to
        # control where surplus lands is to make everything else refuse it.
        head.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lay.addWidget(band(head, "lanehead",
                           f"background:transparent;"
                           f" border-bottom:1px solid {t.border};"))
        # THE PAIR ROW (Pair replicates ticked): this replicate's tag, picked from a menu of the tags still free in
        # its condition; shown only while the board's switch is on
        self.pairing = pairing
        self.pair_button = None
        self.pair_row = None
        if pairing is not None:
            self.pair_button = Button(t, "", size="small")
            self.pair_button.setToolTip("Replicates of other conditions with the same tag are this replicate's "
                                        "partners, for example one culture split or one imaging session. Only "
                                        "the paired significance test uses them.")
            self.pair_button.clicked.connect(self._open_pair_menu)
            # the partners and whether they share its imaging date (`model.pair_note`), shortened in the middle to the
            # lane's width so a long note never widens the board; the dates on hover
            self._pair_note = _PairNote("", ty.ui("xs"), t.muted_foreground)
            pair = row(label("Pair", ty.ui("xs", "medium"), t.muted_foreground), self.pair_button,
                       self._pair_note, None, spacing=8, margins=(12, 6, 12, 6))
            pair.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.pair_row = band(pair, "lanepair", f"background:transparent; border-bottom:1px solid {t.border};")
            lay.addWidget(self.pair_row)
            self.pair_row.setVisible(pairing.on)
            self.refresh_pair()

        # A LANE IS A FIXED WINDOW ONTO ITS SERIES, not a column that grows.
        #
        # Groups do not hold equal numbers -- six groups over 28 series is
        # 5·5·5·5·5·3 -- and a lane sized to its own contents makes the row as
        # tall as the largest group, leaving the others padded with dead space
        # and the panel's height a function of the arrangement you happen to be
        # trying. Fixing the window and scrolling inside it keeps the panel one
        # size while you regroup.
        self.rows = QWidget()
        self.rows.setStyleSheet("background:transparent;")
        self._rows_lay = QVBoxLayout(self.rows)
        self._rows_lay.setContentsMargins(10, 10, 10, 10)
        self._rows_lay.setSpacing(6)
        self._scroll = QScrollArea()
        self._scroll.setWidget(self.rows)
        self._scroll.setWidgetResizable(True)
        # IGNORED vertically, and no minimum of its own.
        #
        # A `widgetResizable` QScrollArea reports its CONTENT's minimum height,
        # so a lane holding four cards claimed ~290 px and the column grew past
        # the board's viewport -- clipping its own footer, which is the line
        # that says how many series are in the group. The same trap the panel's
        # own stack documents. The height floor lives ONCE, on the stack
        # (`VIEW_MIN`), and everything below it takes what it is given.
        self._scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background:transparent; }")
        # Eased like the page. A lane list is the cheapest of the three to
        # scroll (measured 3.8 ms/px against the board's 21.3) but it is also
        # the one you land on first, and a mixed feel reads as a bug.
        self.scroller = smoothscroll.attach(self._scroll, background=t.card)
        lay.addWidget(self._scroll)

        self._foot = label("", ty.ui("xs"), t.muted_foreground)
        foot = row(self._foot, None, spacing=8, margins=(12, 9, 12, 9))
        foot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lay.addWidget(band(foot, "lanefoot",
                           f"background:transparent;"
                           f" border-top:1px solid {t.border};"))
        self.rebuild()

    # -- drop target ---------------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasText() and e.mimeData().text().startswith(SERIES_MIME):
            e.acceptProposedAction()
            self._highlight(True)

    def dragLeaveEvent(self, _e):
        self._highlight(False)

    def dropEvent(self, e):
        """Move every dragged series into THIS group.

        ``Project.assign`` removes a name from whatever group held it before
        adding it here, so a move out of six lanes is still one call per name
        and cannot leave a series in two groups.
        """
        self._highlight(False)
        names = e.mimeData().text()[len(SERIES_MIME):].split(",")
        self.dropped.emit([n for n in names if n], self.g.name)
        e.acceptProposedAction()

    def _highlight(self, on: bool) -> None:
        t = self.t
        edge = t.primary if on else t.border
        self.setStyleSheet(
            f"QFrame#card {{ background:{t.card}; border:1px solid {edge};"
            f" border-radius:{self.radius:g}px; }}")

    def rebuild(self) -> None:
        _discard(self._rows_lay)
        for i, name in enumerate(self.g.members):
            # EVERY OTHER CARD, not every card and not every fourth: `breathe`
            # is a full trip through the event loop, so per-widget costs more
            # than the widget -- and MEASURED at four, a chunk was 450 ms, which
            # is a visible hitch. Two lands the worst chunk near a tenth of a
            # second, which reads as motion rather than as a stop.
            if i and not i % 2:
                breathe()
            self._rows_lay.addWidget(self._new_card(name))
        self._seal()

        self.refresh_totals()

    def _seal(self) -> None:
        """Put a stretch after the cards, so THEY never absorb the slack.

        The card list lives in a ``setWidgetResizable`` scroll area, so its
        content widget is stretched to the viewport whenever the cards do not
        fill it -- and a ``QVBoxLayout`` hands that slack to whatever accepts
        it, which is every card. Measured on a lane holding two of them:
        **112 px each against a 63 px sizeHint**, which is the "short groups
        have giant cards" this fixes. One explicit stretch takes the slack
        instead, and the cards keep their natural height whether a lane holds
        two or twenty.
        """
        lay = self._rows_lay
        last = lay.itemAt(lay.count() - 1) if lay.count() else None
        if last is None or last.widget() is not None:
            lay.addStretch(1)

    def _new_card(self, name: str):
        card = SeriesCard(self.t, self.p, self.p.series_by_name(name),
                          self.fills, self.inks, self.sel)
        card.clicked.connect(self.picked.emit)
        card.held.connect(self.held.emit)
        return card

    def take_card(self, name: str) -> bool:
        """Drop one card out of this lane. True if it was here."""
        for i in range(self._rows_lay.count()):
            item = self._rows_lay.itemAt(i)
            # A stretch item has no widget -- `widget()` returns None, so the
            # isinstance check below skips it without a special case.
            w = item.widget() if item is not None else None
            if isinstance(w, SeriesCard) and w.s.name == name:
                # The widget FIRST: after ``takeAt`` the item no longer owns it
                # and a second ``widget()`` returns None.
                self._rows_lay.takeAt(i)
                discard(w)
                return True
        return False

    def insert_card(self, name: str):
        """Add one card, at its position in ``g.members``."""
        card = self._new_card(name)
        try:
            index = self.g.members.index(name)
        except ValueError:
            index = self._rows_lay.count()
        # BEFORE the trailing stretch -- inserting after it would put the card
        # below the slack and leave a gap above it.
        cards = sum(1 for i in range(self._rows_lay.count())
                    if self._rows_lay.itemAt(i).widget() is not None)
        self._rows_lay.insertWidget(min(index, cards), card)
        self._seal()
        return card

    def refresh_pair(self) -> None:
        """The Pair row's tag, the conditions sharing it and whether they were imaged on its date."""
        if self.pair_button is None:
            return
        self.pair_button.set_text(self.g.pair or "none")
        note, tip = pair_note(self.pairing.projects, self.p, self.g)
        self._pair_note.set_full_text(note)
        self._pair_note.setToolTip("\n".join(x for x in (note, tip) if x))

    def _open_pair_menu(self) -> None:
        if retriggered(self.pair_button):
            return
        t = self.t
        menu = Panel(t, self.window())
        menu.set_trigger(self.pair_button)
        menu_section(t, menu, f"{self.g.name}: pair tag")
        menu_row(t, menu, "No pair", current=not self.g.pair, on_click=lambda: self._set_pair(""))
        menu.addSeparator()
        for tag in free_tags(self.p, self.g):
            users = self.pairing.users(tag, self.p.condition)
            menu_row(t, menu, tag, current=tag == self.g.pair, count=len(users) or None,
                     on_click=lambda tag=tag: self._set_pair(tag))
        menu_note(t, menu, "The number is how many other conditions use the tag.")
        at = self.pair_button.mapToGlobal(self.pair_button.rect().bottomLeft())
        menu.popup(at + QPoint(0, 4))

    def _set_pair(self, tag: str) -> None:
        self.g.pair = str(tag)
        self.pairing.tags_changed.emit()

    def refresh_totals(self) -> None:
        """The day strip and the footer line -- everything a HOLD can change.

        Split out of :meth:`rebuild` because holding one series out changes a
        lane's counts and possibly its day strip, but never which cards are in
        it: the lane lists ``g.members`` whether they are held or not. Rebuilding
        to update a footer destroyed and recreated every card in the group.
        """
        t, p = self.t, self.p
        _discard(self._tags_lay)
        tot = p.group_totals(self.g)
        for index in tot["days"]:
            self._tags_lay.addWidget(
                day_badge(t, self.fills, self.inks, f"D{index}", index))
        n_days = len(tot["days"])
        # IT COUNTS IMAGING DAYS (the D badges beside it), never replicates (tester B9, 3 Nov 2026): "2 replicates
        # merged" on a group that IS one replicate read as two independent cultures pooled (Rule 0)
        self._composition.setText(composition_words(n_days))
        held = tot["held_out"]
        units = (f"{tot['series']} of {len(self.g.members)} series"
                 if held else f"{tot['series']} series")
        self._foot.setText(f"{units} · {tot['cells']} cells · "
                           f"{spaced(tot['kept'])} tracks")


def _discard(layout) -> None:
    """Empty ``layout``, and make the widgets stop painting NOW.

    ``deleteLater`` defers to the event loop, and a widget removed from a layout
    is still a child of its parent until then -- so it keeps painting over its
    replacement. ``setParent(None)`` is the line that makes removal immediate.
    """
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            discard(w)


class GroupsTable(QWidget):
    """Every series in one flat list -- the same data the lanes show.

    A table rather than a QTableWidget: the rows carry a checkbox, a badge and a
    progress bar, and Qt's item views want delegates for that. At 28 rows the
    widget-per-row cost is irrelevant and the code is the same code the cards
    use.
    """

    picked = Signal(str)
    held = Signal(str, bool)

    #: (heading, width). Fixed widths, not a stretch per column: every cell in
    #: a column has to start on the same x or the table stops reading as one.
    COLUMNS: ClassVar = [("", 34), ("Series", 130), ("Replicate", 118),
                         ("Group", 96), ("Keep rate", 150), ("Tracks", 130),
                         ("Cells", 64)]

    def __init__(self, t, project, fills, inks, selection, owner, parent=None):
        super().__init__(parent)
        self.t, self.p = t, project
        self.fills, self.inks, self.sel = fills, inks, selection
        #: WHO OWNS THE FILTER. It used to live here, and that is what made this
        #: table impossible to build lazily: `GroupsPanel` read and wrote
        #: `table._filter` from seven places, so a table that did not exist yet
        #: was a crash the first time anybody ticked a series. The filter is a
        #: property of the PANEL -- it drives the chip, the "showing n of m"
        #: line and the menu's ticks, none of which are this widget's -- so it
        #: belongs there and this asks for it.
        self.owner = owner
        self._rows: dict[str, QWidget] = {}
        self._ticks: dict[str, Checkbox] = {}
        self._group_cells: dict[str, QWidget] = {}
        #: The visible row names as of the last rebuild -- what `after_hold`
        #: compares against to decide whether a rebuild is needed at all.
        self._shown: list[str] = []
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # NOT added to `lay`. This widget IS what `GroupsPanel` puts inside the
        # scrolling `QScrollArea`, so a header row added here would scroll away
        # with the body the moment there were enough rows to need scrolling --
        # reported as exactly that, the column names disappearing on scroll.
        # `GroupsPanel._build_table` places `self.head` OUTSIDE the scroll area
        # instead, where it stays put while only the rows underneath it move.
        self.head = self._header()
        self.body = QWidget()
        self.body.setStyleSheet("background:transparent;")
        self._body_lay = QVBoxLayout(self.body)
        self._body_lay.setContentsMargins(0, 0, 0, 0)
        self._body_lay.setSpacing(0)
        lay.addWidget(self.body)
        lay.addStretch(1)
        self.rebuild()

    def _header(self) -> QWidget:
        t = self.t
        cells = []
        for name, width in self.COLUMNS:
            # `TableHead`: `font-medium text-foreground`, on a `text-sm` root.
            lab = label(name, ty.ui("sm", "medium"), t.foreground)
            lab.setFixedWidth(width)
            cells.append(lab)
        head = row(*cells, None, spacing=8, margins=(16, 11, 16, 11))
        return band(head, "thead", f"background:{t.subtle};"
                                   f" border-bottom:1px solid {t.border};")

    def shown(self) -> list:
        """The rows this table draws -- the panel decides which."""
        return self.owner.shown()

    def _paint_row(self, line, name: str) -> None:
        """`TableRow`: `transition-colors hover:bg-muted/50`,
        `data-[state=selected]:bg-muted`.

        Hover is attached only when the row is NOT selected. In CSS the two
        utilities have equal specificity and the outcome depends on source
        order, which is not a decision worth inheriting: a selected row that
        goes half-strength under the pointer reads as deselecting itself. A
        selected row is given the same colour for both, so its `HoverFill` stays
        in place and simply has nothing to fade to.
        """
        t = self.t
        picked = name in self.sel
        # `t.card` rather than "transparent" -- see the note in `panels._day_row`:
        # fading from a zero-alpha colour interpolates through alpha.
        rest = t.muted if picked else t.card
        hover = rest if picked else mix(t.muted, t.card, 0.5)
        fade = getattr(line, "_hover_fill", None)
        if fade is None:
            key = f"tr_{name}"
            line.setObjectName(key)
            line._hover_fill = HoverFill(
                line,
                lambda fill, n=key: (f"QWidget#{n} {{ background:{fill};"
                                     f" border-bottom:1px solid {t.border}; }}"),
                rest=rest, hover=hover)
        else:
            fade.set_colours(rest, hover)

    def refresh_selection(self, names=None) -> None:
        """Repaint the affected rows IN PLACE.

        The first version called ``rebuild()`` on every selection change, which
        recreated 28 rows -- each with a checkbox, a badge and a progress bar --
        for a fill change, whether or not the table was even the visible view.
        Measured: 467 ms per click, then 127 ms once it repainted instead of
        rebuilding, then ~0 once it repainted only what moved.
        """
        for name in (names if names is not None else self._rows):
            line = self._rows.get(name)
            if line is not None:
                self._paint_row(line, name)

    def rebuild(self) -> None:
        t, p = self.t, self.p
        _discard(self._body_lay)
        rows = self.shown()
        self._rows = {}
        self._ticks = {}
        self._group_cells = {}
        self._shown = [x.name for x in rows]
        if not rows:
            self._body_lay.addWidget(
                row(label("No series match this filter.", ty.ui("sm"),
                          t.muted_foreground), None,
                    spacing=8, margins=(16, 14, 16, 14)))
            return
        for i, s in enumerate(rows):
            if i and not i % 4:                        # see `Lane.rebuild`
                breathe()
            line = self._row(t, p, s)
            self._rows[s.name] = line
            self._body_lay.addWidget(line)

    def after_hold(self, name: str) -> None:
        """A series was held out or brought back. Repaint; rebuild only if the
        set of VISIBLE rows actually changed.

        It only changes under a filter that selects on inclusion -- "Held out".
        Everywhere else the same 28 rows stay, one of them differently ticked,
        and rebuilding them is the cost this avoids.
        """
        if [x.name for x in self.shown()] != self._shown:
            self.rebuild()
            return
        tick = self._ticks.get(name)
        if tick is not None:
            want = name not in self.p.excluded
            tick.set_checked(want)
        line = self._rows.get(name)
        if line is not None:
            self._paint_row(line, name)

    def after_move(self, names) -> None:
        """Series changed group. Retext the Group cell; rebuild only if the set
        of VISIBLE rows changed, which a group filter can do."""
        if [x.name for x in self.shown()] != self._shown:
            self.rebuild()
            return
        t = self.t
        for name in names:
            cell = self._group_cells.get(name)
            if cell is None:
                continue
            g = self.p.group_of(name)
            cell.setText(g.name if g else "unplaced")
            cell.setStyleSheet(
                f"color:{t.foreground if g else t.muted_foreground};"
                f" background:transparent;")

    def _row(self, t, p, s) -> QWidget:
        widths = [w for _n, w in self.COLUMNS]
        tick = Checkbox(t, checked=s.name not in p.excluded)
        self._ticks[s.name] = tick
        tick.toggled.connect(lambda on, n=s.name: self.held.emit(n, on))
        tick_cell = ShadowCell(t, tick)
        tick_cell.setFixedWidth(widths[0])

        name = label(s.name, ty.ui("sm", "medium"), t.foreground)
        name.setFixedWidth(widths[1])

        if s.day:
            d = p.day(s.day)
            # THE BADGE ALWAYS, THE DATE ONLY IF THERE IS ONE. A day is a
            # replicate index and always exists; the calendar date beside it
            # comes from the files, and a loaded project has none -- see
            # `adapt`. The row keeps its shape either way.
            when = (f"{d.on.day} {d.on.strftime('%b')}" if d.on else "no date")
            day_cell = row(day_badge(t, self.fills, self.inks, d.tag, s.day),
                           label(when, ty.ui("sm"), t.muted_foreground), spacing=6)
        else:
            day_cell = row(Badge(t, "–", t.muted, t.muted_foreground),
                           label("no date", ty.ui("sm"), t.muted_foreground),
                           spacing=6)
        day_cell.setFixedWidth(widths[2])

        g = p.group_of(s.name)
        group = label(g.name if g else "unplaced", ty.ui("sm"),
                      t.foreground if g else t.muted_foreground)
        group.setFixedWidth(widths[3])
        self._group_cells[s.name] = group

        rate = row(Progress(t, s.keep_rate, height=8, width=72),
                   label(f"{s.keep_rate:.0%}", ty.ui("sm", "medium",
                                              tabular=True),
                         t.foreground), spacing=8)
        rate.setFixedWidth(widths[4])

        tracks = label(f"{s.kept} / {s.found}", ty.ui("sm"), t.muted_foreground)
        tracks.setFixedWidth(widths[5])
        cells = label(str(s.cells), ty.ui("sm"), t.muted_foreground)
        cells.setFixedWidth(widths[6])

        line = row(tick_cell, name, day_cell, group, rate, tracks, cells, None,
                   spacing=8, margins=(16, 7, 16, 7))
        self._paint_row(line, s.name)
        line.setCursor(Qt.PointingHandCursor)
        line.mouseReleaseEvent = lambda _e, n=s.name: self.picked.emit(n)
        return line


class SelectionBar(QWidget):
    """A floating action bar: appears only when something is picked.

    Hidden rather than disabled when empty. A permanently visible bar offering
    "Move to..." with nothing selected is a control that is wrong most of the
    time.

    **It is drawn as a floating card**, not as a row: ``bg-popover`` with a
    border, ``rounded-md`` and ``shadow-lg`` -- the same elevation shadcn gives
    a Dialog. It overlays the lanes, and an overlay that looks like a row reads
    as a layout bug rather than as something hovering above the content. The
    shadow is painted INSIDE the widget's own rect, with the contents inset to
    leave room, because a widget cannot paint outside itself and this one has
    no ``ShadowHost`` to do it for it.
    """

    cleared = Signal()
    move_requested = Signal()

    #: Room reserved inside the widget for the shadow to fall into.
    PAD = 12

    def __init__(self, t, selection, parent=None):
        super().__init__(parent)
        self.t, self.sel = t, selection
        self.count = label("", ty.ui("sm", "medium"), t.foreground)
        move = Button(t, "Move to…", size="small", glyph="arrow-right")
        move.clicked.connect(self.move_requested.emit)
        clear = Button(t, "Clear", size="small", variant="ghost")
        clear.clicked.connect(self.cleared.emit)
        inner = row(icon("check", icons.SM, t.muted_foreground), self.count,
                    24, move, clear, spacing=8, margins=(16, 10, 12, 10))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(self.PAD, self.PAD, self.PAD, self.PAD)
        lay.setSpacing(0)
        lay.addWidget(inner)
        selection.changed.connect(lambda _n: self.refresh())
        self.refresh()

    def paintEvent(self, _e):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(self.rect()).adjusted(self.PAD, self.PAD,
                                           -self.PAD, -self.PAD)
        draw_shadow(p, box, t.radius_md, t, t.shadow_lg)
        p.setPen(QColor(t.border))
        p.setBrush(QColor(t.popover))
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5),
                          t.radius_md, t.radius_md)
        p.end()

    def refresh(self) -> None:
        n = len(self.sel)
        self.count.setText(f"{n} series selected" if n != 1
                           else "1 series selected")


#: ui.json: the Analysis groups board shown (True, the standard) or folded to its header line
BOARD_OPEN_KEY = "groups_board_open"


class GroupsPanel(Card):
    """The panel: a toolbar, two views of one model, and a selection bar.

    FOLDABLE (asked 25 Oct 2026: "a button to toggle ... if some people dont care"): Hide folds the board to its
    header line and Show brings it back. Folding changes nothing about the fit -- the groups the board holds (the
    replicates, unless someone regrouped) are what the analysis uses either way -- and the choice is remembered for
    every later project (`BOARD_OPEN_KEY`)."""

    changed = Signal()
    #: Emitted with the new state after Hide / Show, so the page can give the freed height away.
    folded = Signal(bool)

    def __init__(self, t, project, fills, inks, parent=None, pairing=None):
        super().__init__(t, parent=parent)
        self.t, self.p = t, project
        self.pairing = pairing
        self.fills, self.inks = fills, inks
        self.sel = Selection(self)
        self._cards: dict[str, SeriesCard] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.caption = label("", ty.ui("sm"), t.muted_foreground)
        self.view = Segmented(t, ["Cards", "Table"], current=0,
                              glyphs=["grid", "list"], size="small")
        self.view.changed.connect(self._show_view)
        self.chip = FilterChip(t)
        self.chip.setVisible(False)
        self.chip.dismissed.connect(lambda: self._set_filter(None))
        #: ``None`` | ``("day", i)`` | ``("group", name)`` | ``("unplaced",
        #: None)`` | ``("held", None)``. A filter HIDES ROWS and changes nothing
        #: about the fit -- which is why no count is refreshed when it is set,
        #: and why the menu says so out loud. It lives HERE rather than on the
        #: table because it drives the chip, the "showing n of m" line and the
        #: menu's ticks, none of which the table owns -- and because a table
        #: that owns it cannot be built lazily.
        self._filter = None
        self.showing = label("", ty.ui("xs", "medium"), t.muted_foreground)
        self.showing.setVisible(False)
        self._filter_button = Button(t, "Filter", size="small", glyph="filter")
        self._filter_button.clicked.connect(self._open_filter)
        regroup = Button(t, "Regroup…", size="small", glyph="chart")
        self._regroup_button = regroup
        regroup.clicked.connect(self._open_regroup)
        self._reset_b = self._reset_button()
        self._open = True
        self.folded_note = label("Hidden. The analysis still uses these groups.", ty.ui("xs"),
                                 t.muted_foreground)
        self.fold = Button(t, "Hide", size="small")
        self.fold.setToolTip("Hide or show the board. Hiding it does not change the analysis.")
        self.fold.clicked.connect(lambda: self.set_open(not self._open))
        head = row(icon("columns-3", icons.MD, t.muted_foreground),
                   label("Analysis groups", section_font(),
                         t.foreground),
                   self.caption, self.chip, self.showing, self.folded_note, None,
                   self._filter_button, regroup,
                   self._reset_b,
                   self.view, self.fold,
                   spacing=8, margins=(20, 16, 20, 16))
        lay.addWidget(band(head, "panelhead",
                           f"background:transparent;"
                           f" border-bottom:1px solid {t.border};"))
        # PAIR REPLICATES (5 Nov 2026, opt-in): one switch for the project, its own line so the header keeps its width
        self.pair_band = None
        if pairing is not None:
            self.pair_box = Checkbox(t, checked=pairing.on)
            self.pair_box.setToolTip("Declare which replicates of different conditions belong together, for a "
                                     "paired significance test. Off: every replicate stands alone and only "
                                     "Welch is offered.")
            self.pair_box.toggled.connect(pairing.set_on)
            self.propose = Button(t, "Propose from imaging days", size="small", glyph="rotate")
            self.propose.setToolTip("Give replicates of different conditions imaged on the same date the same tag. "
                                    "This is a proposal: check each tag and change or clear it.")
            self.propose.clicked.connect(pairing.propose_requested.emit)
            self.propose.setVisible(pairing.on)
            self._pair_hint = label("Tag the replicates of different conditions that belong together.",
                                    ty.ui("xs"), t.muted_foreground)
            self._pair_hint.setVisible(pairing.on)
            line = row(self.pair_box, label("Pair replicates", ty.ui("sm", "medium"), t.foreground),
                       self._pair_hint, None, self.propose, spacing=8, margins=(20, 8, 20, 8))
            line.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.pair_band = band(line, "pairband", f"background:transparent; border-bottom:1px solid {t.border};")
            lay.addWidget(self.pair_band)
            pairing.toggled.connect(self._pairing_toggled)
            pairing.tags_changed.connect(self._pairs_changed)

        # ONE HEIGHT for the stack and both of its pages.
        #
        # Each page is a QScrollArea that would otherwise report its content's
        # height and grow the panel -- 28 table rows are far taller than six
        # lanes. Ignored HORIZONTALLY for the same class of reason: six 300 px
        # lanes make the viewport report ~1 900 px, that width propagates up
        # through the page, and the counts row above gets pushed off the right
        # edge. The lanes scroll sideways INSIDE the panel; they do not widen
        # the screen.
        # ONE MINIMUM for the stack and both of its pages -- a minimum now,
        # not a fixed height.
        #
        # It was fixed, and the cost was measured: the Analysis groups card
        # grows with the window, but the lane list inside it did not, so at a
        # 1400 px window the card was 831 px tall with a 250 px list inside it
        # -- 463 px of dead space, and three cards visible no matter how large
        # the window got. Each page still has to be told a minimum, because a
        # QScrollArea reports its CONTENT's height otherwise and 28 table rows
        # would drive the panel taller than the screen.
        self.VIEW_MIN = Lane.FLOOR_H + 118
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background:transparent;")
        self.stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.stack.setMinimumHeight(self.VIEW_MIN)
        self.stack.setMinimumWidth(0)
        self.lanes_host = QWidget()
        self.lanes_host.setStyleSheet(f"background:{t.subtle}; border:none;")
        self._lanes_lay = QHBoxLayout(self.lanes_host)
        self._lanes_lay.setContentsMargins(Lane.GAP, Lane.GAP, Lane.GAP, Lane.GAP)
        self._lanes_lay.setSpacing(Lane.GAP)
        # SIX lanes do not fit a 1456 px window, and squeezing them until they
        # do is how a group column stops being readable. They keep a minimum
        # width and the row scrolls sideways instead -- which is also what the
        # data means: the number of groups is the user's choice, so the panel
        # cannot assume any particular number of them fits.
        lanes_scroll = QScrollArea()
        lanes_scroll.setWidget(self.lanes_host)
        lanes_scroll.setWidgetResizable(True)
        lanes_scroll.setFrameShape(QScrollArea.NoFrame)
        lanes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lanes_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lanes_scroll.setStyleSheet("QScrollArea { background:transparent; }")
        lanes_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        lanes_scroll.setMinimumHeight(self.VIEW_MIN)
        self._lanes_scroll = lanes_scroll
        # SIDEWAYS, and on its own step. Qt's horizontal singleStep here is
        # content-derived and gives 202 px a detent -- 2.6 of them across the
        # whole board. 100 px is about a third of a lane and crosses in 5.2,
        # which is what the page does vertically.
        # THE STEPS ON THIS BOARD ARE MEASURED IN LANES.
        #
        # A wheel notch of 100 px crossed the whole board in 4.3 detents at a
        # wide window and 7.7 at a narrow one -- coarser than the page's own
        # tuning ever was (5.3 across the common state, 20 at its tallest), on a
        # surface where the content is cards you are trying to read.
        #
        # A FIFTH OF A LANE is the replacement, because it is predictable rather
        # than merely smaller: five detents move exactly one card, so the board
        # advances in units you can see. 63 px gives 6.8 detents at a wide
        # window and 12.2 at a narrow one, which lands inside the page's range.
        #
        # ONE LANE for the bar's page step, and that is a deliberate departure
        # from the platform, where a page click travels a viewport. Here the
        # whole range is 431-770 px against a viewport of 1142-1481 -- LESS THAN
        # ONE PAGE -- so "a viewport" and "jump to the far end" are the same
        # click at every window size this ships at. Paging by a card is the only
        # reading of the gesture that leaves anywhere to land.
        self.lanes_scroller = smoothscroll.attach(
            lanes_scroll, background=t.subtle,
            notch_px=(Lane.MIN_W + Lane.GAP) / 5,
            page_px=Lane.MIN_W + Lane.GAP)
        self.stack.addWidget(lanes_scroll)

        # THE TABLE IS BUILT WHEN IT IS FIRST OPENED. `Cards` is the view you
        # land on, so on a 30-series project this was 0.25 s of a 0.64 s panel
        # spent on a pane nobody had asked for -- the same argument that defers
        # the whole groups view out of `ConfirmPage`.
        #
        # It could not be done while the table owned `_filter`; now the panel
        # does, and every other site that reaches into the table is guarded by
        # `if self.table is not None`. The scroll area exists from the start so
        # the stack has two pages and `_show_view(1)` has somewhere to go; only
        # its CONTENT waits.
        self.table = None
        table_scroll = QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setFrameShape(QScrollArea.NoFrame)
        table_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        table_scroll.setMinimumHeight(self.VIEW_MIN)
        table_scroll.setStyleSheet("QScrollArea { background:transparent; }")
        self._table_scroll = table_scroll
        self.table_scroller = smoothscroll.attach(table_scroll, background=t.card)

        # THE HEADER SITS OUTSIDE THE SCROLL AREA, as a sibling above it, so it
        # stays on screen while the rows scroll underneath -- reported as the
        # column names disappearing on scroll, because `GroupsTable` used to
        # put its own header at the top of the very widget the scroll area
        # scrolls. `_build_table` fills this once the table (and its header)
        # actually exist; empty until then, since a page with only a lane list
        # ever opened has nothing to show here.
        table_page = QWidget()
        table_page.setStyleSheet("background:transparent;")
        table_page_lay = QVBoxLayout(table_page)
        table_page_lay.setContentsMargins(0, 0, 0, 0)
        table_page_lay.setSpacing(0)
        self._table_head_host = QWidget()
        self._table_head_host.setStyleSheet("background:transparent;")
        self._table_head_lay = QVBoxLayout(self._table_head_host)
        self._table_head_lay.setContentsMargins(0, 0, 0, 0)
        self._table_head_lay.setSpacing(0)
        table_page_lay.addWidget(self._table_head_host)
        table_page_lay.addWidget(table_scroll, 1)
        self.stack.addWidget(table_page)
        lay.addWidget(self.stack)

        # THE BAR IS AN OVERLAY, not a row in this layout.
        #
        # As a row it did two things wrong at once. It appeared at the panel's
        # bottom -- measured 82 px BELOW the page viewport, so the control
        # became invisible exactly when it became relevant. And it took its
        # 52 px out of the lane view, so selecting a card re-scrolled the lanes
        # you were reading.
        #
        # It is parented and positioned by :class:`~shadcn.screens.ConditionBody`,
        # which owns the scroll area and is therefore the only thing that knows
        # which part of this panel is on screen.
        self.bar = SelectionBar(t, self.sel)
        self.bar.cleared.connect(self.sel.clear)
        self.bar.move_requested.connect(self._open_move)

        self.sel.changed.connect(self._repaint_picked)
        # LANES ONLY, because `GroupsTable.__init__` has already built its own
        # rows. Calling the full `rebuild` here built the 30-row table a SECOND
        # time -- profiled: `_row` ran 60 times for 30 series, 0.19 s of a
        # 0.85 s panel, discarding thirty freshly-made rows to make thirty
        # identical ones. Every LATER rebuild still does both, which is what
        # regrouping needs.
        self._rebuild_lanes()
        self._refresh_caption()
        self.folded_note.setVisible(False)
        if theme.remembered(BOARD_OPEN_KEY) is False:
            self.set_open(False, remember=False)

    def set_open(self, on: bool, *, remember: bool = True) -> None:
        """Show the board, or fold it to the header line (the name and the count of series and groups).

        The tools that act on the board fold with it, and a selection is let go, since nothing it could be moved
        to is on screen. The vertical policy goes to Maximum while folded, as `ReplicateDays` does when its table
        is shut, so the card is its header's height and not a stretched empty frame."""
        on = bool(on)
        if on == self._open:
            return
        self._open = on
        if not on:
            self.sel.clear()
        for widget in (self.stack, self._filter_button, self._regroup_button, self._reset_b, self.view):
            widget.setVisible(on)
        if self.pair_band is not None:
            self.pair_band.setVisible(on)
        self.chip.setVisible(on and self._filter is not None)
        self.showing.setVisible(on and self._filter is not None)
        self.folded_note.setVisible(not on)
        self.fold.set_text("Hide" if on else "Show")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred if on else QSizePolicy.Maximum)
        self.updateGeometry()
        if remember:
            theme.remember(BOARD_OPEN_KEY, on)
        self.folded.emit(not on)

    def is_open(self) -> bool:
        return self._open

    def _reset_button(self):
        b = Button(self.t, "Reset to replicates", size="small", glyph="rotate")
        b.clicked.connect(self._reset)
        return b

    # -- views --------------------------------------------------------------- #
    def _show_view(self, index: int) -> None:
        if index == 1:
            self._build_table()
        self.stack.setCurrentIndex(index)

    def _build_table(self) -> None:
        """Build the table pane. Idempotent, on first sight of it."""
        if self.table is not None:
            return
        self.table = GroupsTable(self.t, self.p, self.fills, self.inks,
                                 self.sel, self)
        self.table.picked.connect(self._pick)
        self.table.held.connect(self._hold)
        self._table_head_lay.addWidget(self.table.head)
        self._table_scroll.setWidget(self.table)

    def rebuild(self) -> None:
        """Rebuild both presentations of the groups: the lanes and the table."""
        self._rebuild_lanes()
        # A table nobody has opened has nothing to rebuild: it reads the current
        # groups when it is built. Every reach into it below is guarded the same
        # way, which is the price of building it late and is one line each.
        if self.table is not None:
            self.table.rebuild()
        self._refresh_caption()

    def _rebuild_lanes(self) -> None:
        _discard(self._lanes_lay)
        self.lanes: list[Lane] = []
        self._cards: dict[str, SeriesCard] = {}
        for g in self.p.groups:
            # A LANE AT A TIME, with the loop allowed to paint between them.
            # Each lane is one series card per member -- five to ten widgets
            # each -- and this is the single most expensive build on the screen
            # (1.6 s measured over 30 series). `breathe` keeps the spinner
            # turning through it without letting a click re-enter the build.
            breathe()
            lane = Lane(self.t, self.p, g, self.fills, self.inks, self.sel, pairing=self.pairing)
            lane.picked.connect(self._pick)
            lane.held.connect(self._hold)
            lane.dropped.connect(self._drop)
            self._lanes_lay.addWidget(lane)
            self.lanes.append(lane)
            for card in lane.findChildren(SeriesCard):
                self._cards[card.s.name] = card
        # Surplus width pools to the RIGHT of the last lane rather than being
        # shared between them -- see `Lane.__init__`.
        self._lanes_lay.addStretch(1)

    def _pairing_toggled(self, on: bool) -> None:
        """The switch moved (here or on another condition's board): the box, the line and every lane's Pair row."""
        if self.pair_box.checked != on:
            self.pair_box.set_checked(on, animate=False)
        self.propose.setVisible(on)
        self._pair_hint.setVisible(on)
        for lane in getattr(self, "lanes", ()):
            if lane.pair_row is not None:
                lane.pair_row.setVisible(on)
        self.changed.emit()

    def _pairs_changed(self) -> None:
        for lane in getattr(self, "lanes", ()):
            lane.refresh_pair()
        self.changed.emit()

    def _refresh_caption(self) -> None:
        p = self.p
        self.caption.setText(f"{p.condition} · {len(p.series)} series in "
                             f"{len(p.groups)} groups")

    # -- edits --------------------------------------------------------------- #
    def _pick(self, name: str) -> None:
        self.sel.toggle(name)

    def _hold(self, name: str, on: bool) -> None:
        """Hold a series out, or bring it back. IN PLACE.

        This used to rebuild every lane and the whole table for one checkbox --
        six lanes, 83 cards and 28 rows destroyed and recreated to change one
        tick and some footers. That is the freeze; it is also where the crash
        came from, because ``_cards`` still pointed at the SeriesCards those
        rebuilds had just deleted, and the next selection change called
        ``refresh()`` on them.

        Nothing structural changes here: a held-out series stays in its lane and
        stays in the table. Only its own card, the lane footers and the totals
        move.
        """
        self.p.set_included(name, on)
        card = self._cards.get(name)
        if card is not None:
            card.refresh()
        # ONLY the lane that holds it. ``group_totals`` is per group, so the
        # other five cannot have changed -- and refreshing all six was 28 ms of
        # the 56 ms this used to cost.
        for lane in self.lanes:
            if name in lane.g.members:
                lane.refresh_totals()
                break
        if self.table is not None:
            self.table.after_hold(name)
        self.changed.emit()

    def recolour(self, fills, inks) -> None:
        """Point every day badge in the panel at a new ramp -- IN PLACE.

        One pass of stylesheet writes. The first version called ``rebuild()``,
        which meant destroying and recreating six lanes, every series card in
        them and 28 table rows to change a fill: 948 ms for one colour and
        2 728 ms for a palette switch, measured, on 83 cards.
        """
        self.fills, self.inks = fills, inks
        # A table built LATER reads these off the panel, so a recolour before it
        # exists is not lost -- it is simply the state it will be born with.
        if self.table is not None:
            self.table.fills, self.table.inks = fills, inks
        for badge in self.findChildren(Badge):
            if badge.day_index is not None:
                i = min(badge.day_index - 1, len(fills) - 1)
                badge.set_colours(fills[i], inks[i])

    # -- filter, regroup, move ------------------------------------------------
    def _open_filter(self) -> None:
        """Sections, submenus, and counts beside the two that matter.

        "A day" and "A group" open submenus rather than spilling seven days and
        six groups into one list: the menu should be the SHAPE of the choice,
        and the choice is "by what" first, "which one" second.
        """
        # A second click on the button CLOSES the menu instead of reopening it.
        if retriggered(self._filter_button):
            return
        t, p = self.t, self.p
        # `Panel`, in the page. Its submenus come from `styled_menu(t, menu)`
        # below, which yields a Panel because this parent is one -- see
        # :func:`menus.styled_menu`.
        menu = Panel(t, self.window())
        menu.set_trigger(self._filter_button)
        menu_section(t, menu, "Show only")
        menu_row(t, menu, "Everything", current=self._filter is None,
                 on_click=lambda: self._set_filter(None))

        # BUILT ON HOVER, not here. Twelve of this menu's seventeen rows live
        # in these two submenus, and neither is on screen until you hover the
        # row that owns it -- so building them up front was most of the delay
        # between pressing Filter and seeing anything.
        def days_menu():
            days = styled_menu(t, menu)
            for d in p.days:
                k = min(d.index - 1, len(self.fills) - 1)
                menu_row(t, days,
                         d.on.strftime("%d %b %Y").lstrip("0") if d.on
                         else f"Replicate {d.index}",
                         badge=(self.fills[k], self.inks[k], d.tag),
                         on_click=lambda i=d.index: self._set_filter(
                             ("day", i)))
            return days

        spec = self._filter
        active = spec[1] if spec and spec[0] == "day" else None
        shown = next((d for d in p.days if d.index == active), None)
        k = min((active or 1) - 1, len(self.fills) - 1)
        menu_row(t, menu, "A replicate", submenu_factory=days_menu,
                 badge=((self.fills[k], self.inks[k], shown.tag) if shown
                        else None))

        def groups_menu():
            groups = styled_menu(t, menu)
            for g in p.groups:
                menu_row(t, groups, g.name, count=len(g.members),
                         on_click=lambda n=g.name: self._set_filter(
                             ("group", n)))
            return groups

        menu_row(t, menu, "A group", submenu_factory=groups_menu)

        menu_section(t, menu, "Not in the fit")
        menu_row(t, menu, "No group yet", count=len(p.unplaced()),
                 on_click=lambda: self._set_filter(("unplaced", None)))
        menu_row(t, menu, "Held out", count=len(p.excluded),
                 on_click=lambda: self._set_filter(("held", None)))
        menu_note(t, menu, "Only hides rows on this board; the fit is unchanged.")

        below = self._filter_button.mapToGlobal(
            self._filter_button.rect().bottomLeft())
        # `aria-expanded`: the trigger holds its fill for as long as the menu is
        # up. Cleared from the menu's own signal rather than from anything here,
        # so it is right however the menu goes away -- a pick, a click outside,
        # Escape.
        self._filter_button.set_expanded(True)
        # `closed`, not `aboutToHide`: a Panel is a widget, not a QMenu.
        menu.closed.connect(
            lambda: self._filter_button.set_expanded(False))
        menu.popup(below + QPoint(0, 6))

    def shown(self) -> list:
        """The series the current filter leaves visible, in project order."""
        return [s for s in self.p.series if self._matches(s)]

    def _matches(self, s) -> bool:
        if not self._filter:
            return True
        kind, value = self._filter
        if kind == "day":
            return s.day == value
        if kind == "group":
            g = self.p.group_of(s.name)
            return g is not None and g.name == value
        if kind == "unplaced":
            return self.p.group_of(s.name) is None
        if kind == "held":
            return s.name in self.p.excluded
        return True

    def _set_filter(self, spec) -> None:
        self._filter = spec
        if self.table is not None:
            self.table.rebuild()
        on = spec is not None
        self.chip.setVisible(on)
        self.showing.setVisible(on)
        if on:
            kind, value = spec
            names = {"day": f"Replicate · D{value}", "group": str(value),
                     "unplaced": "No group yet", "held": "Held out"}
            self.chip.set_text(names[kind])
            self.showing.setText(f"showing {len(self.shown())} of "
                                 f"{len(self.p.series)} rows")
        # A filter is a VIEW: deliberately NOT refreshing any count here.
        if on:
            self.view.select(1)

    def _open_regroup(self) -> None:
        # A dialog trigger is `aria-haspopup` too (Base UI's `DialogTrigger`
        # sets `aria-haspopup="dialog"`), so it holds its fill while the dialog
        # is up -- which shows when the pointer moves off it.
        self._regroup_button.set_expanded(True)
        # `self.window()` -- already this file's own idiom for reaching the
        # top-level `OnboardingWindow` (`_open_move`'s `Panel(t,
        # self.window())`, a few lines below). `window.pages` (the
        # `QStackedWidget` filling the whole window) is what this overlay
        # blurs and disables -- the structural analog of `WorkspaceWindow
        # .centralWidget()`, which `OnboardingWindow` (a plain `QWidget`)
        # has no equivalent API for. See `widgets.dialogs.OverlayDialog`'s
        # own docstring on why `content` is a required, explicit argument
        # rather than something guessed from the parent.
        window = self.window()
        dialog = RegroupDialog(self.t, self.p, self.fills, self.inks,
                               content=window.pages, parent=window)

        def _on_applied() -> None:
            self.p.apply_regroup(dialog.blocks())
            self.sel.clear()
            self.rebuild()
            self.changed.emit()

        dialog.applied.connect(_on_applied)
        # Deferred to `closed` (the outro has actually finished), not fired
        # immediately on Apply/Cancel -- the non-blocking analog of the old
        # `try/finally` around a blocking `exec()`: the trigger should read
        # "expanded" for as long as the dialog is visibly on screen,
        # including its ~200ms close fade, on either exit path.
        dialog.closed.connect(lambda: self._regroup_button.set_expanded(False))
        # Constructed fresh per open (`self._groups` re-reads `len(self.p.
        # groups)` live, so a cached instance would go stale the moment a
        # regroup actually changes that count) -- `closed` -> `deleteLater`
        # is what stops those fresh instances from piling up as hidden
        # widgets still holding a live `eventFilter` on the window.
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _open_move(self) -> None:
        """Where the selection should go. Groups, plus out of any group."""
        move_to = self.bar.move_button if hasattr(self.bar, "move_button") else self.bar
        if retriggered(move_to):
            return
        t, p = self.t, self.p
        # `Panel`, in the page -- see :class:`menus.Panel`. The last of the four
        # popovers to move; none opens as a native popup now.
        menu = Panel(t, self.window())
        menu.set_trigger(move_to)
        menu_section(t, menu, f"Move {len(self.sel)} to")
        for g in p.groups:
            menu_row(t, menu, g.name, count=len(g.members),
                     on_click=lambda n=g.name: self._move(n))
        menu.addSeparator()
        menu_row(t, menu, "No group", on_click=lambda: self._move(None))
        at = self.bar.mapToGlobal(self.bar.rect().topRight())
        menu.popup(at + QPoint(-220, -8))

    def _move(self, group_name) -> None:
        """The selection bar's "Move to". The SAME path as a drag-and-drop.

        It used to call ``rebuild()`` -- the 961 ms whole-panel rebuild that
        made a drop feel stuck, and made this button feel the same way. There is
        no reason for the two to differ: dropping is moving one series with the
        pointer, this is moving several from a menu.
        """
        self._drop(sorted(self.sel.names), group_name)

    def _drop(self, names, group_name: str) -> None:
        """Move series into ``group_name``. Only the lanes that changed.

        This used to rebuild the whole panel -- measured at **961 ms** for one
        card, of which the table alone was 468 ms and the six lanes ~313 ms.
        That is the pause where the drag pixmap sits on screen after the drop:
        the cursor is not lagging, the UI is busy rebuilding everything.

        A move touches exactly two lanes -- the one it left and the one it
        joined -- and changes one cell per row in the table.

        Shared with the selection bar's "Move to" (:meth:`_move`), which is the
        same operation started a different way, and with ``group_name=None`` for
        "No group", where there is no destination lane at all.
        """
        by_lane = {lane.g.name: lane for lane in self.lanes}
        target = by_lane.get(group_name)
        touched = {group_name}
        moving = []
        for name in names:
            source = next((l for l in self.lanes if name in l.g.members), None)
            if source is None or source is target:
                continue
            touched.add(source.g.name)
            moving.append((name, source))

        for name in names:
            self.p.assign(name, group_name)
        self.sel.clear()

        # ONE card created per moved series, not one per series in two columns.
        for name, source in moving:
            source.take_card(name)
            # Re-point the registry in the same breath: the old entry is
            # dangling the moment `take_card` deletes it, and a dangling entry
            # is what took the window down on the next repaint. With no target
            # -- "Move to -> No group" -- there is no lane to hold the card, so
            # the entry is DROPPED rather than left pointing at a dead widget.
            if target is not None:
                self._cards[name] = target.insert_card(name)
            else:
                self._cards.pop(name, None)
        for lane in self.lanes:
            if lane.g.name in touched:
                lane.refresh_totals()
        if self.table is not None:
            self.table.after_move(names)
        self.changed.emit()

    def _reset(self) -> None:
        self.p.reset_to_days()
        self.sel.clear()
        self.rebuild()
        self.changed.emit()

    def _repaint_picked(self, names=None) -> None:
        """One stylesheet write per CHANGED widget. No rebuild, no tree walk."""
        if names is None:
            names = set(self._cards)
        for name in list(names):
            card = self._cards.get(name)
            if card is None:
                continue
            try:
                card.refresh()
            except RuntimeError:
                # The C++ object is gone while the Python wrapper is not -- the
                # same lifetime trap ShadowHost documents. Drop the entry rather
                # than let a stale cache take the window down.
                self._cards.pop(name, None)
        if self.table is not None:
            self.table.refresh_selection(names)


