"""The Confirm screen's condition tabs, WRAPPED onto as many rows as the width needs.

Asked 15 Oct 2026: with many conditions "u only see a few of it and the rest are not
visible cause the text is too long". The tabs were ONE `Segmented` of fixed size: fourteen
of one user's condition names made it 3 017 px wide, the page was forced to 3 161 px and most
tabs sat off the screen. Now each row is its own `Segmented` tray, packed by
`onboarding_view.tab_rows`; one selection runs across all of them (a tray without it shows
none raised, `Segmented.select(-1)`, the Compute dialog's two-row precedent).

The WIDTH comes from the page, never from the names (horizontal policy Ignored, and the
layout sets no minimum), so the tabs can no longer force the window wider. The rows are
repacked when the width changes, and only when the packing does.

DRAG A TAB TO REORDER (6 Nov 2026, asked: "also in day picker already -> drag and drop to change order of the
conditions"): pressing a tab and moving past `DRAG_SLOP` drags it -- a bar shows where it lands, between tabs and
across rows -- and the release emits ``moved(from_index, to_index)`` (``to_index`` the place BEFORE which it lands,
0 to the tab count) instead of selecting it; a press that does not move is the click it always was. The page
reorders its conditions (`ConfirmPage._reorder`) and hands the tabs back (`set_order`). The pills are rebuilt on every
selection (`Segmented.select`), so the drag's event filter is put back on every rebuild (`_watch`).
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QPoint, QRect, Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLayout, QSizePolicy, QVBoxLayout, QWidget

from ...onboarding_view import tab_rows
from ..widgets.controls import Segmented
from ..widgets.primitives import discard

__all__ = ["ConditionTabs"]

#: Between two rows of trays.
ROW_GAP = 6
#: `Segmented`'s own tray padding and pill spacing at the default size.
_PAD, _GAP = 4, 4
#: A press on a tab becomes a drag after this many pixels (a click is not a drag).
DRAG_SLOP = 6


class ConditionTabs(QWidget):
    """``changed(index)`` when another condition is picked; ``swatch_clicked(index, rect)``
    when a condition's colour dot is, both with the index over ALL conditions."""

    changed = Signal(int)
    swatch_clicked = Signal(int, QRect)
    #: ``(from_index, to_index)``: a tab dragged and dropped before the tab now at ``to_index`` (the count: last)
    moved = Signal(int, int)

    def __init__(self, t, items, *, current: int = 0, swatches=None, parent=None):
        super().__init__(parent)
        self.t = t
        self._items = list(items)
        self._swatches = list(swatches) if swatches else None
        self._current = current
        self.setStyleSheet("background:transparent;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(ROW_GAP)
        # NO MINIMUM from the trays: the page decides the width (see the module docstring)
        self._lay.setSizeConstraint(QLayout.SetNoConstraint)
        self._widths = self._measure()
        self._rows: list = []          # [(start, stop, tray, holder)]
        self._packing: list = []
        self._syncing = False
        #: the drag: the index pressed, where, whether it moved far enough, where it would land
        self._press_index: int | None = None
        self._press_at: QPoint | None = None
        self._dragging = False
        self.drop_target: int | None = None
        self.marker = QWidget(self)
        self.marker.setStyleSheet(f"background:{t.primary}; border-radius:1px;")
        self.marker.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.marker.hide()
        self._pack(1200)

    # -- what the page reads --------------------------------------------------------- #
    def current(self) -> int:
        return self._current

    def rows(self) -> list:
        """``[(start, stop), ...]``: which conditions each row holds."""
        return list(self._packing)

    def tray_of(self, index: int):
        """The `Segmented` holding condition ``index``."""
        for start, stop, tray, _holder in self._rows:
            if start <= index < stop:
                return tray
        return None

    def bottom_of(self, index: int) -> int:
        """The global y of the bottom of the row holding ``index``: a popover opened from a
        tab's dot clears the row the dot is in."""
        tray = self.tray_of(index) or self
        return tray.mapToGlobal(tray.rect().bottomLeft()).y()

    def select(self, index: int) -> None:
        if not 0 <= index < len(self._items) or index == self._current:
            return
        self._current = index
        self._sync()
        self.changed.emit(index)

    def set_swatches(self, colours) -> None:
        """Re-point the dots without touching the selection."""
        self._swatches = list(colours)
        self._rebuild()

    def set_order(self, items, swatches=None, current: int = 0) -> None:
        """The tabs again in a new order (after a drag): their words, their dots and the one raised, nothing emitted."""
        self._items = list(items)
        self._swatches = list(swatches) if swatches else None
        self._current = int(current)
        self._widths = self._measure()
        self._packing = []
        self._pack(max(self.width(), 1))

    # -- building -------------------------------------------------------------------- #
    def _measure(self) -> list:
        """Each tab's width, read off one tray holding them all (never shown)."""
        probe = Segmented(self.t, self._items, current=-1, swatches=self._swatches)
        try:
            return [probe.pill(i).sizeHint().width() for i in range(len(self._items))]
        finally:
            discard(probe)

    def _pack(self, width: int) -> None:
        packing = tab_rows(self._widths, max(int(width), 1), gap=_GAP, pad=_PAD)
        if packing == self._packing:
            return
        self._packing = packing
        self._rebuild()

    def _rebuild(self) -> None:
        for _start, _stop, _tray, holder in self._rows:
            self._lay.removeWidget(holder)
            discard(holder)
        self._rows = []
        for start, stop in self._packing:
            tray = Segmented(self.t, self._items[start:stop],
                             current=self._current - start if start <= self._current < stop else -1,
                             swatches=self._swatches[start:stop] if self._swatches else None)
            tray.changed.connect(lambda i, s=start: self._picked(s + i) if i >= 0 else None)
            tray.changed.connect(lambda _i, tr=tray: self._watch(tr))
            tray.swatch_clicked.connect(lambda i, rect, s=start: self.swatch_clicked.emit(s + i, rect))
            holder = QWidget()
            holder.setStyleSheet("background:transparent;")
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(tray)
            hl.addStretch(1)
            self._lay.addWidget(holder)
            self._rows.append((start, stop, tray, holder))
            self._watch(tray)
        n = len(self._rows)
        height = self._rows[0][2].minimumHeight() if n else 0
        self.setFixedHeight(n * height + max(n - 1, 0) * ROW_GAP)

    def _picked(self, index: int) -> None:
        if self._syncing or index == self._current:
            return
        self._current = index
        self._sync()
        self.changed.emit(index)

    def _sync(self) -> None:
        """One raised tab across all rows."""
        self._syncing = True
        try:
            for start, stop, tray, _holder in self._rows:
                tray.select(self._current - start if start <= self._current < stop else -1)
        finally:
            self._syncing = False

    def resizeEvent(self, e) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(e)
        if e.size().width() != e.oldSize().width():
            self._pack(e.size().width())

    # -- dragging a tab to another place ------------------------------------------------------------------------ #
    def _watch(self, tray) -> None:
        """The drag's filter on every pill of ``tray`` and everything in it (its dot, its words): a pill is rebuilt
        on each selection, so this runs after every rebuild."""
        for i in range(len(getattr(tray, "_pills", ()))):
            pill = tray.pill(i)
            for w in (pill, *pill.findChildren(QWidget)):
                w.removeEventFilter(self)
                w.installEventFilter(self)

    def _index_of(self, widget) -> int | None:
        """The condition index of the pill ``widget`` is (or is in)."""
        while widget is not None and widget is not self:
            for start, _stop, tray, _holder in self._rows:
                pills = getattr(tray, "_pills", ())
                if widget in pills:
                    return start + pills.index(widget)
            widget = widget.parentWidget()
        return None

    def target_at(self, pos: QPoint) -> int:
        """Where a tab dropped at ``pos`` (this widget's coordinates) lands: before the first tab of the row under it
        whose middle is right of it, else after that row's last (0 to the tab count)."""
        best = None
        for start, stop, tray, _holder in self._rows:
            top = tray.mapTo(self, QPoint(0, 0)).y()
            distance = 0 if top <= pos.y() <= top + tray.height() else min(abs(pos.y() - top),
                                                                           abs(pos.y() - top - tray.height()))
            if best is None or distance < best[0]:
                best = (distance, start, stop)
        if best is None:
            return 0
        _d, start, stop = best
        for index in range(start, stop):
            pill = self.pill_at(index)
            if pill.mapTo(self, pill.rect().center()).x() > pos.x():
                return index
        return stop

    def _show_marker(self, target: int | None) -> None:
        self.drop_target = target
        if target is None or not self._items:
            self.marker.hide()
            return
        index = min(target, len(self._items) - 1)
        pill = self.pill_at(index)
        if pill is None:
            self.marker.hide()
            return
        corner = pill.mapTo(self, QPoint(0, 0))
        x = corner.x() - 3 if target < len(self._items) else corner.x() + pill.width() + 1
        self.marker.setGeometry(max(0, x), corner.y(), 2, pill.height())
        self.marker.raise_()
        self.marker.show()

    def eventFilter(self, obj, e):              # noqa: N802 - Qt's name
        kind = e.type()
        if kind == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
            if self._press_index is None:
                self._press_index = self._index_of(obj)
                self._press_at = obj.mapTo(self, e.pos())
                self._dragging = False
            return False
        if kind == QEvent.MouseMove and self._press_index is not None and (e.buttons() & Qt.LeftButton):
            pos = obj.mapTo(self, e.pos())
            if not self._dragging and (pos - self._press_at).manhattanLength() >= DRAG_SLOP:
                self._dragging = True
                self.setCursor(Qt.ClosedHandCursor)
            if self._dragging:
                self._show_marker(self.target_at(pos))
                return True
            return False
        if kind == QEvent.MouseButtonRelease and e.button() == Qt.LeftButton:
            pressed, dragged, target = self._press_index, self._dragging, self.drop_target
            self._press_index, self._press_at, self._dragging = None, None, False
            self._show_marker(None)
            self.unsetCursor()
            if dragged:
                if pressed is not None and target is not None and target not in (pressed, pressed + 1):
                    # after this release is handled: the page rebuilds the tabs, this pill among them
                    from qtpy.QtCore import QTimer

                    QTimer.singleShot(0, lambda a=pressed, b=target: self.moved.emit(a, b))
                return True                     # a drag is not a click: nothing is selected, no colour asked
            return False
        return False

    def pill_at(self, index: int) -> QWidget | None:
        """Condition ``index``'s tab, wherever it was packed."""
        for start, stop, tray, _holder in self._rows:
            if start <= index < stop:
                return tray.pill(index - start)
        return None

    def centre_of(self, index: int) -> QPoint:
        pill = self.pill_at(index)
        return pill.mapToGlobal(pill.rect().center()) if pill is not None else QPoint()
