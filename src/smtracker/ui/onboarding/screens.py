"""The two screens and the shell that switches them.

``OnboardingWindow`` is a stack of exactly two pages, because the flow is two
decisions: *which folder*, then *is this what you meant*. Everything that looked
at first like a third step turned out to be a state of the second one.
"""

from __future__ import annotations

from typing import ClassVar

from qtpy.QtCore import QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme import hcl
from ..theme import icons
from ..theme import typography as ty
from ... import condition_colours
from .colour import ColourChoice
from .condition_tabs import ConditionTabs
from ..widgets.controls import Button, Segmented
from .groups import GroupsPanel, Pairing
from ..widgets.menus import (Panel, menu_note, menu_row, menu_section, retriggered,
                             warm_backdrop)
from .panels import ConfirmFooter, CountsRow, ParamRow, ReplicateDays, TitleRow, band
from ..widgets.primitives import (
    Card,
    FitScroll,
    ShadowHost,
    breathe,
    col,
    icon,
    label,
    row,
)
from ..widgets.swatches import PaletteRow, SwatchRow
from ..theme.tokens import day_ramp

__all__ = ["ConditionBody", "ConfirmPage"]


class ConditionBody(QWidget):
    """One condition's whole body: counts, parameters, days, groups.

    A body PER CONDITION rather than one body that is re-pointed. Switching a
    tab must not lose the arrangement you made in the tab you left, and the only
    way to be sure of that is for each condition to own its widgets and its
    Project -- nothing to reset, and nothing to forget to reset.
    """

    changed = Signal()

    def __init__(self, t, project, base, parent=None, pairing=None):
        super().__init__(parent)
        self.t, self.p = t, project
        self.pairing = pairing
        self.setStyleSheet("background:transparent;")
        self.base = base
        # one shade per imaging day AND per saved replicate (the record view's R tags), whichever are more
        fills, inks = day_ramp(t.color_scheme, base, max(len(project.days), len(project.record_rows())))

        # TWO VIEWS, not one scrolling page.
        #
        # The page had a scroll range of 159 px -- too short to read as
        # navigation, long enough to clip the Analysis groups card mid-row --
        # and it was the OUTERMOST of three scroll regions stacked under the
        # pointer inside a lane (list 40, board 524, page 159). A wheel there
        # could mean any of the three depending on whether the inner one still
        # had room. Splitting the page removes the outer region entirely: two
        # scroll contexts in the working view, one in the record view.
        #
        # The split is not "general vs detailed" -- the counts cards are the
        # most general thing here and they live in the WORKING view, because
        # `groups.changed -> counts.refresh` makes them the live feedback for
        # grouping. It is what the files SAY versus what you DECIDE.
        self.counts = CountsRow(t, project)
        breathe()
        self.params = ParamRow(t, project)
        breathe()
        self.days = ReplicateDays(t, project, fills, inks)
        self.days.set_fill(True)
        breathe()

        # THE GROUPS VIEW IS BUILT WHEN IT IS FIRST LOOKED AT, and that is not a
        # micro-optimisation. RE-MEASURED on a 30-series project after the face
        # changed to Inter: the record view arrives in 523 ms and `_build_groups`
        # costs a further 826 ms -- thirty series cards, a thirty-row table and
        # three lanes -- and every millisecond of it is spent on the GUI thread,
        # where it is a FREEZE. (It read 1.61 s under Geist. The drop is the
        # glyph-fallback search leaving with that font, not the widgets getting
        # cheaper, which is why this deferral survives and `window._prewarm`
        # did not: 826 ms of real building does not amortise away.) The load itself
        # runs on a worker and stalls the event loop by 48 ms at its worst; this
        # was two full seconds of a stopped spinner at the end of it, which is
        # the part that reads as a crash.
        #
        # It costs one thing, recorded rather than discovered: the counts row
        # refreshes from `groups.changed`, so the wiring below cannot be done
        # until the panel exists. `_build_groups` does both, once.
        self._fills, self._inks = fills, inks
        self.groups = None

        self.views = QStackedWidget()
        self.views.setStyleSheet("background:transparent;")
        self._record = self._page(t, [(self.params, 0), (self.days, 1)])
        self._working = None
        self.views.addWidget(self._record)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.views)

    def warm(self, after_ms: int = 300) -> None:
        """Build the working view LATER, while the record view is being read.

        The deferral below keeps the screen from freezing on arrival; this keeps
        the Groups tab from freezing on the first click. The cost has to be paid
        on the GUI thread wherever it lands -- Qt widgets cannot be built off
        it -- so it is paid at the one moment nobody is waiting on a result:
        just after the record view has painted.

        Idempotent, and harmless if the user gets there first: `_build_groups`
        returns immediately once the panel exists.
        """
        QTimer.singleShot(after_ms, self._build_groups)

    def _build_groups(self) -> None:
        """Build the working view. Idempotent, and called on first sight of it."""
        if self.groups is not None:
            return
        t = self.t
        breathe()
        self.groups = GroupsPanel(t, self.p, self._fills, self._inks, pairing=self.pairing)
        breathe()
        self.groups.changed.connect(self.counts.refresh)
        self.groups.changed.connect(self.changed.emit)
        self._working = self._page(t, [(self.counts, 0), (self.groups, 1)])
        # FOLDED, the board's height goes to a spacer under it, not to a stretched empty card: with a stretch on
        # the board the spacer (stretch 0) gets nothing, and without one it takes what is left
        self._working.layout().addStretch(0)
        self.views.addWidget(self._working)

        # The selection bar floats over the WORKING view -- the only one that
        # has a selection to act on.
        self._bar = self.groups.bar
        self._bar.setParent(self._working)
        self._bar.raise_()
        self.groups.sel.changed.connect(lambda _n: self._place_bar())
        self.groups.folded.connect(self._on_folded)
        self._on_folded(not self.groups.is_open())

    def _page(self, t, rows):
        """One view. NO scroll area -- that is the point.

        A page scroll area was here as a safety net, and it is exactly what a
        safety net costs: below the view's own minimum it grew a page scrollbar
        instead of refusing to be that small. Measured, the working view needs
        611 px of content while the window floor allowed 493, so the bar
        appeared for every window between them.

        Without it, the view's minimum propagates to the WINDOW through the
        ordinary layout, and Qt simply will not let the window be smaller than
        its content -- so there is no size at which a page scrollbar can exist.
        The regions that legitimately scroll are all INSIDE this: the day
        table, the lane lists, the board, the series table.
        """
        inner = ShadowHost(t)
        inner.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(28, 18, 28, 24)
        lay.setSpacing(14)
        for widget, stretch in rows:
            lay.addWidget(widget, stretch)
        return inner

    def _on_folded(self, folded: bool) -> None:
        self._working.layout().setStretchFactor(self.groups, 0 if folded else 1)
        self._place_bar()

    def show_view(self, index: int) -> None:
        """Switch views, building the working one the first time it is asked for."""
        if index == 1:
            self._build_groups()
        self.views.setCurrentIndex(index)
        self._place_bar()

    def current_view(self) -> int:
        return self.views.currentIndex()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_bar()

    def _place_bar(self) -> None:
        """Pin the bar to the bottom of the panel's VISIBLE region.

        Not the bottom of the panel -- the panel is 434 px tall inside a 656 px
        viewport that already has the counts, parameters and day strip above
        it, so its own bottom sits off-screen and a bar anchored there is a
        control you cannot see. Not the bottom of the window either: that would
        detach it from the thing it acts on.

        The intersection is the honest answer. It reads as "the bottom of the
        Analysis groups section" whenever the section fits, and stays reachable
        when it does not.
        """
        if self.groups is None:            # the working view has never been shown
            return
        bar = self._bar
        if not len(self.groups.sel):
            bar.hide()
            return
        viewport = self._working
        top_left = self.groups.mapTo(viewport, QPoint(0, 0))
        panel = QRect(top_left, self.groups.size())
        visible = panel.intersected(viewport.rect())
        if visible.height() < 60:
            bar.hide()
            return
        height = bar.sizeHint().height()
        # Content width, CENTRED -- a floating bar that spans the full panel
        # reads as a row again. The widget carries its own shadow padding, so
        # the visual card is inset from this geometry by SelectionBar.PAD.
        # The widget is TALLER than the card it draws -- it reserves PAD on
        # every side for its own shadow. Positioning by the widget's bottom is
        # what keeps that padding inside the viewport too; anchoring the card
        # flush to the bottom pushed 6 px of shadow past the edge, where it is
        # clipped and the card looks cut off.
        width = bar.sizeHint().width()
        bar.setGeometry(visible.center().x() - width // 2,
                        visible.bottom() - height,
                        width, height)
        bar.show()
        bar.raise_()

    def set_base(self, colour: str) -> None:
        """Point this condition's ramp at a new colour.

        The day tags belong to the CONDITION, so they follow its colour -- a tag
        left on the old hue would be claiming a condition it no longer matches,
        which is the one thing deriving the ramp from the condition exists to
        prevent.
        """
        if colour == self.base:
            return
        self.base = colour
        fills, inks = day_ramp(self.t.color_scheme, colour, max(len(self.p.days), len(self.p.record_rows())))
        self.days.recolour(fills, inks)
        # THE RAMP IS REMEMBERED EVEN WHEN THE PANEL IS NOT BUILT, so a palette
        # chosen while the record view is up survives to the working view --
        # `_build_groups` reads these, and a recolour that only reached the live
        # widgets would leave the groups on the old hue the first time they were
        # opened.
        self._fills, self._inks = fills, inks
        if self.groups is not None:
            self.groups.recolour(fills, inks)


class ConfirmPage(QWidget):
    """Everything the tool read, and the one thing it asks you to decide."""

    cancelled = Signal()
    opened = Signal()

    def __init__(self, t, projects, parent=None):
        super().__init__(parent)
        self.t = t
        # THE ORDER THE WORKSPACE WILL DRAW THEM IN (`condition_order`, remembered per project), changed here by
        # dragging a tab (6 Nov 2026); `order_changed` tells the window to remember a new one on Open workspace
        from ... import condition_order

        listed = list(projects.values())
        self.order_key = condition_order.project_key([p.path for p in listed if getattr(p, "path", None)])
        rank = {c: i for i, c in enumerate(condition_order.recall(self.order_key, [p.condition for p in listed]))}
        self.projects = sorted(listed, key=lambda p: rank.get(p.condition, len(rank)))
        self.order_changed = False
        self.p = self.projects[0]
        self.setStyleSheet(f"background:{t.background};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        head.setStyleSheet("background:transparent;")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(28, 28, 28, 0)
        hl.setSpacing(14)
        self.title = TitleRow(t, self.p)
        hl.addWidget(self.title)

        # REMEMBERED PER PROJECT (per user, in ui.json, never beside the data), and
        # handed on to the workspace by `OnboardingWindow._hand_on` -- the choice
        # made here is the one the workspace draws with.
        names = [p.condition for p in self.projects]
        self.colour_key = condition_colours.project_key(
            [p.path for p in self.projects if getattr(p, "path", None)])
        self.colours = ColourChoice(
            names, model=condition_colours.recall(self.colour_key, names),
            dark=bool(getattr(t, "_dark", False)))
        self.colours.changed.connect(self._recolour)
        self.colours.changed.connect(
            lambda: condition_colours.remember(self.colour_key, self.colours.model))
        # WRAPPED onto as many rows as the width needs (`ConditionTabs`): one fixed row of
        # many long names pushed the page past the screen (asked 15 Oct 2026).
        self.tabs = ConditionTabs(
            t, [f"{p.condition}\t{len(p.series)}" for p in self.projects],
            current=0,
            swatches=[self.colours.colour_of(i)
                      for i in range(len(self.projects))])
        self.tabs.changed.connect(self._switch)
        self.tabs.swatch_clicked.connect(self._pick_colour)
        self.tabs.moved.connect(self._reorder)

        # WHICH VIEW, on the right, mirroring which CONDITION on the left.
        #
        # Icon-only because the pair is a place, not a setting with values to
        # read: two words there would compete with the condition names for the
        # same glance. The record view is FIRST and is the default -- this
        # screen is called Confirm project, and its job is that somebody sees
        # the acquisition parameters before opening a workspace. Defaulting to
        # the grouping view would let a project be confirmed without them ever
        # being on screen.
        # An icon-only control has to say what it is somewhere. shadcn's own
        # Tooltip, per the registry -- inverted, and with NO delay.
        # SAME SIZE as the condition tabs. The two groups sit on one row and
        # mirror each other; at `small` this one was 32 px against their 40 and
        # the pair read as a control and an afterthought. At the default size a
        # `size-4` glyph fills the pill the way shadcn's `icon-sm` button does.
        #
        # `layers`, not `columns-3`: that one already labels the Analysis groups
        # section, and an icon that means two things on one screen means neither.
        self.view_tabs = Segmented(
            t, ["", ""], glyphs=["file-text", "layers"],
            tips=["Record", "Groups"])
        self.view_tabs.changed.connect(self._switch_view)
        tab_row = QWidget()
        tab_row.setStyleSheet("background:transparent;")
        tl = QHBoxLayout(tab_row)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)
        tl.addWidget(self.tabs, 1)
        tl.addWidget(self.view_tabs, 0, Qt.AlignTop)
        hl.addWidget(tab_row)
        # THE SAFEGUARD (6 Nov 2026): more conditions than the palette's colours, or two hard to tell apart, is said
        # here, where the colours are chosen (`condition_colours.crowding_text`)
        from ..widgets.primitives import fit_label

        self.crowding = fit_label("", ty.ui("xs"), t.muted_foreground)
        hl.addWidget(self.crowding)
        self._sync_crowding()
        outer.addWidget(head)

        # PAIR REPLICATES (5 Nov 2026, opt-in): ONE switch for every condition's board, since a pair spans conditions
        self.pairing = Pairing(self.projects, self)
        self.pairing.propose_requested.connect(self._propose_pairs)
        self.bodies = QStackedWidget()
        self.bodies.setStyleSheet("background:transparent;")
        for i, project in enumerate(self.projects):
            # ONE BREATH PER CONDITION. Each body is tens of widgets and a
            # project with eight conditions builds eight of them; letting the
            # loop paint between keeps the spinner behind this turning.
            breathe()
            body = ConditionBody(t, project, self.colours.colour_of(i), pairing=self.pairing)
            body.changed.connect(self._refresh_footer)
            self.bodies.addWidget(body)

        # THE BODIES SCROLL WHERE THE WINDOW IS SHORTER THAN THEM, and only there (30 Oct 2026). The working view
        # asks for 478 px (counts, and the groups board at its floor), the page 664 with its title and footer: 23 px
        # taller than a 1280 x 720 laptop's maximised window, which then drew the board over the footer. `FitScroll`:
        # no scroll range on a screen that holds the page, the footer outside it either way.
        self.bodies_scroll = FitScroll()
        self.bodies_scroll.setWidget(self.bodies)
        self.bodies_scroll.setWidgetResizable(True)
        self.bodies_scroll.setFrameShape(QFrame.NoFrame)
        self.bodies_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.bodies_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.bodies_scroll.viewport().setStyleSheet("background:transparent;")
        outer.addWidget(self.bodies_scroll, 1)
        self.title.palette_requested.connect(self._pick_palette)
        self.title.show_palette(self.colours.palette_swatches(),
                                hcl.PALETTES[self.colours.palette].label)

        # OUTSIDE the scroll area: the primary action must not scroll away.
        self.footer = ConfirmFooter(t, self.p)
        self.footer.cancelled.connect(self.cancelled.emit)
        self.footer.opened.connect(self.opened.emit)
        wrap = row(self.footer, margins=(28, 16, 28, 16))
        outer.addWidget(band(wrap, "confirmFooter",
                             f"background:{t.subtle};"
                             f" border-top:1px solid {t.border};"))

    def warm(self, after_ms: int = 300) -> None:
        """Warm every condition's groups view, once this page is on screen.

        SCHEDULED BY THE WINDOW, not by each body as it is built. A timer armed
        inside `ConditionBody.__init__` could fire while `ConfirmPage` was still
        being constructed -- `breathe` runs the event loop mid-build, which is
        exactly when a 300 ms timer comes due -- so the groups board was
        sometimes built INSIDE the page build it was meant to follow. That is
        how the handover measured 450 ms on one run and 2 300 ms on the next.

        Staggered, so several conditions do not land on one another.
        """
        for i in range(self.bodies.count()):
            self.bodies.widget(i).warm(after_ms + 400 * i)

    def _propose_pairs(self) -> None:
        """"Propose from imaging days" (`model.propose_pairs`): every condition's replicates tagged by shared imaging
        dates, replacing the tags there were; every board redraws."""
        from ...onboarding_model import propose_pairs

        proposal = propose_pairs(self.projects)
        for project in self.projects:
            tags = proposal.get(project.condition, {})
            for group in project.groups:
                group.pair = tags.get(group.name, "")
        self.pairing.tags_changed.emit()

    def _switch_view(self, index: int) -> None:
        """Every condition, not just the visible one.

        The view is a property of what you are DOING, not of which condition you
        are looking at; leaving each body on its own view would mean switching
        condition could also switch view, which is a second effect nobody asked
        that tab for.
        """
        for i in range(self.bodies.count()):
            self.bodies.widget(i).show_view(index)

    def _switch(self, index: int) -> None:
        self.p = self.projects[index]
        self.bodies.setCurrentIndex(index)
        self.title.set_project(self.p)
        self.footer.p = self.p
        self.footer.refresh()

    def _refresh_footer(self) -> None:
        self.footer.refresh()

    def condition_order(self) -> list:
        """The conditions in the order the tabs show."""
        return [p.condition for p in self.projects]

    def _reorder(self, from_index: int, to_index: int) -> None:
        """A tab dragged before the tab at ``to_index``: the conditions, their pages and their colours follow (a
        colour stays with its condition); the one on show stays on show. Remembered on Open workspace."""
        n = len(self.projects)
        if not (0 <= from_index < n and 0 <= to_index <= n) or to_index in (from_index, from_index + 1):
            return
        order = list(range(n))
        order.insert(to_index - 1 if to_index > from_index else to_index, order.pop(from_index))
        shown = self.projects[self.bodies.currentIndex()]
        bodies = [self.bodies.widget(i) for i in range(n)]
        self.projects = [self.projects[i] for i in order]
        self.colours.names = [self.colours.names[i] for i in order]
        for body in bodies:
            self.bodies.removeWidget(body)
        for i in order:
            self.bodies.addWidget(bodies[i])
        current = self.projects.index(shown)
        self.bodies.setCurrentIndex(current)
        self.tabs.set_order([f"{p.condition}\t{len(p.series)}" for p in self.projects],
                            [self.colours.colour_of(i) for i in range(n)], current)
        self.order_changed = True

    # -- colour ---------------------------------------------------------------
    def _sync_crowding(self) -> None:
        text = condition_colours.crowding_text(self.colours.model, dark=bool(getattr(self.t, "_dark", False)))
        self.crowding.setText(text or "")
        self.crowding.setVisible(bool(text))

    def _recolour(self) -> None:
        self._sync_crowding()
        self.tabs.set_swatches([self.colours.colour_of(i)
                                for i in range(len(self.projects))])
        self.title.show_palette(self.colours.palette_swatches(),
                                hcl.PALETTES[self.colours.palette].label)
        for i in range(len(self.projects)):
            self.bodies.widget(i).set_base(self.colours.colour_of(i))

    def _pick_palette(self) -> None:
        if retriggered(self.title.palette):
            return
        t = self.t
        # `Panel`, not `styled_menu`: drawn IN THE PAGE rather than in a native
        # popup, because DWM shadows a popup window on its own terms and that
        # shadow lands on the border. See :class:`menus.Panel`. The animation is
        # `Menu`'s, ported rather than approximated.
        menu = Panel(t, self.window())
        # SUPPORTED palettes have a colour for every condition and can be chosen;
        # UNSUPPORTED ones are too small and are shown disabled, with why.
        n = len(self.projects)
        for heading, keys in (("Supported", self.colours.model.supported()),
                              ("Unsupported", self.colours.model.unsupported())):
            if not keys:
                continue
            menu_section(t, menu, heading)
            for key in keys:
                palette = hcl.PALETTES[key]
                ok = heading == "Supported"
                # The palette's OWN colours, not just its name -- a menu offering
                # two names asks you to already know which set is which.
                item = PaletteRow(
                    t, palette.label, self.colours.colours(key)[:5],
                    current=key == self.colours.palette, disabled=not ok,
                    note="" if ok else f"{len(palette.order)} colours, {n} conditions")
                if ok:
                    item.triggered.connect(
                        lambda k=key: (self.colours.set_palette(k), menu.close_chain()))
                menu.add_widget(item)
        # YOUR OWN (6 Nov 2026): make one, or edit one made before (`PaletteMaker`)
        menu_row(t, menu, "Make a palette\u2026", glyph="plus",
                 on_click=lambda: QTimer.singleShot(0, self._make_palette))
        menu_note(t, menu, "A palette is a set. Choosing one resets every "
                           "condition to its own slot.")
        field = self.title.palette
        # END-ALIGNED. The picker sits against the right margin, so a surface
        # growing rightward from its left edge runs off the screen and Qt
        # clamps it -- landing at an arbitrary offset from the control that
        # opened it. Radix's own `align="end"`; see `Menu.popup_under`.
        menu.popup_under(field, align="end")

    def _pick_colour(self, index: int, at: QRect) -> None:
        """Which slot this condition holds. Taking a held one SWAPS.

        Anchored under the DOT that was clicked, not under the tab strip: the
        popover should appear where the pointer already is.
        """
        # The swatch dot has no lasting widget of its own -- the pill is rebuilt
        # on every tab change -- so its trigger is the RECT it was clicked in,
        # which `retriggered` accepts as readily as a widget.
        if retriggered(at):
            return
        t = self.t
        name = self.projects[index].condition
        # `Panel`, in the page -- see `_pick_palette` and :class:`menus.Panel`.
        menu = Panel(t, self.window())
        menu.set_trigger(at)
        row_widget = SwatchRow(
            t, self.colours.colours(), self.colours.colour_names(),
            current=self.colours.slots[index],
            holders=[self.colours.holder_of(s)
                     for s in range(len(self.colours.colours()))],
            owner=name)
        row_widget.picked.connect(
            lambda slot: (self.colours.assign(index, slot), menu.close_chain()))
        menu.add_widget(row_widget)
        # A COLOUR BY HAND (6 Nov 2026, "or to manually adjust some colors"): any colour, over the palette's
        menu_row(t, menu, "Custom colour\u2026", glyph="pencil",
                 on_click=lambda: QTimer.singleShot(0, lambda: self._custom_colour(index)))
        if name in self.colours.model.custom:
            menu_row(t, menu, "Use the palette's colour", glyph="undo",
                     on_click=lambda: self.colours.set_custom(index, None))

        # HORIZONTAL from the dot, VERTICAL from the tab strip.
        #
        # The dot sits inside the pill, so anchoring the popover to the dot's
        # own bottom opened it overlapping the tabs. Only the horizontal
        # position should follow the thing you clicked; the vertical one has to
        # clear the control the thing lives in.
        #
        # 4 px is shadcn's own ``sideOffset`` -- Popover and DropdownMenu both
        # default to it, fetched from the registry rather than guessed.
        width = row_widget.sizeHint().width()
        x = at.center().x() - width // 2
        edge = self.mapToGlobal(self.rect().topRight()).x() - width - 12
        below = self.tabs.bottom_of(index)
        menu.popup(QPoint(max(12, min(x, edge)), below + 4))

    def _custom_colour(self, index: int) -> None:
        """The system colour picker for condition ``index``; its colour by hand on OK."""
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog

        chosen = QColorDialog.getColor(QColor(self.colours.colour_of(index)), self.window(),
                                       f"Colour for {self.projects[index].condition}")
        if chosen.isValid():
            self.colours.set_custom(index, chosen.name())

    def _make_palette(self) -> None:
        """The palette maker (`PaletteMaker`), starting from the palette in use; a palette saved there is chosen when
        it has a colour for every condition."""
        from ..widgets.palette_maker import PaletteMaker

        maker = PaletteMaker(self.t, start=self.colours.palette, content=self, parent=self.window())
        maker.saved.connect(self._palette_made)
        maker.closed.connect(maker.deleteLater)
        self.palette_maker = maker
        maker.open()

    def _palette_made(self, key: str) -> None:
        if self.colours.supported(key):
            self.colours.set_palette(key)
        self._recolour()

    # -- what the launcher and the shot script reach for --------------------- #
    @property
    def body(self):
        return self.bodies.currentWidget()

    @property
    def counts(self):
        return self.body.counts

    @property
    def days(self):
        return self.body.days

    @property
    def groups(self):
        return self.body.groups
