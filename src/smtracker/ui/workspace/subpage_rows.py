"""The sidebar's SUBPAGE controls: the "+" at a section row's right end and the list of added pages under the row
(`smtracker.subpages` holds the rules; this draws them). Thin Qt: no rule lives here.

* `PlusAction` -- shadcn's `SidebarMenuAction`: a small "+" at the row's right end, shown while the pointer is over the
  row (`attach_hover`), left of the HMM-style chevron where the section has one.
* `SubpageRow` -- one added page: a small row of the page's name; click selects its first page; double-click renames in
  place; a hover "x" deletes; a right-click menu offers Rename, Move up, Move down, Delete; dragging the row up or down
  puts it elsewhere in the list.
* `PageChoice` -- the drop-down that picks WHICH page of a section something takes from (the Spatial Map dialog's
  "Use the same D as the SQD fit"): a `QComboBox` in the workspace's combo style with a chevron at its right end.
* `SubpageList` -- the rows of one section, indented under the section's row with the hairline `_PageList` draws, and
  the signals the sidebar forwards: ``add``-less (the "+" is the row's), ``picked(view)``, ``renamed(number, text)``,
  ``deleted(number)``, ``moved(number, step)``, ``dropped(number, index)``.

Invariants: a row emits what the user asked and changes nothing in the model; the sidebar's `set_subpages` redraws the
list from the model after every change, so what is drawn is always what the model holds.
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QPoint, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import QComboBox, QLineEdit, QVBoxLayout, QWidget

from ... import subpages as _sp
from ..theme import typography as ty
from ..widgets.menus import Panel, menu_row
from ..widgets.primitives import discard

#: The drag must travel this many pixels before a press becomes a drag (a click is not a drag).
DRAG_SLOP = 6


class _Glyph(QWidget):
    """A small clickable glyph at a row's right end."""

    clicked = Signal()

    def __init__(self, t, name: str, size: int, tip: str = "", parent=None):
        super().__init__(parent)
        self.t = t
        self.name = name
        self._hot = False
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tip)
        self.setAttribute(Qt.WA_Hover, True)

    def enterEvent(self, e):                         # noqa: N802 - Qt's name
        self._hot = True
        self.update()

    def leaveEvent(self, e):                         # noqa: N802 - Qt's name
        self._hot = False
        self.update()

    def mousePressEvent(self, e):                    # noqa: N802 - Qt's name
        e.accept()

    def mouseReleaseEvent(self, e):                  # noqa: N802 - Qt's name
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()
        e.accept()

    def paintEvent(self, _e):                        # noqa: N802 - Qt's name
        from ..theme import icons

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._hot:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self.t.sidebar_accent))
            p.drawRoundedRect(QRectF(self.rect()), self.t.radius_sm, self.t.radius_sm)
        inner = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        icons.paint(p, self.name, inner, self._ink(), stroke=2.0)
        p.end()

    def _ink(self) -> str:
        return self.t.sidebar_accent_foreground if self._hot else self.t.sidebar_foreground


class PlusAction(_Glyph):
    """The "+" of a section row (shadcn's `SidebarMenuAction`). ALWAYS SHOWN on the wide rail (asked 4 Nov 2026:
    "i thought about a + next to SQD", a hover-only "+" was not found), faint until the row is pointed at."""

    def __init__(self, t, size: int, tip: str, parent=None):
        super().__init__(t, "plus", size, tip, parent)
        self.row_hot = False

    def set_row_hot(self, on: bool) -> None:
        if bool(on) != self.row_hot:
            self.row_hot = bool(on)
            self.update()

    def _ink(self) -> str:
        if self._hot or self.row_hot:
            return super()._ink()
        return self.t.muted_foreground


class _RowInk(QObject):
    """Tells ``plus`` whether the pointer is over ``row`` (or the "+" itself), so it can darken."""

    def __init__(self, row, plus):
        super().__init__(row)
        self.row, self.plus = row, plus
        row.installEventFilter(self)
        plus.installEventFilter(self)

    def eventFilter(self, watched, event):           # noqa: N802 - Qt's name
        kind = event.type()
        if kind == QEvent.Enter:
            self.plus.set_row_hot(True)
        elif kind == QEvent.Leave and watched is self.row:
            from qtpy.QtGui import QCursor

            if not self.row.rect().contains(self.row.mapFromGlobal(QCursor.pos())):
                self.plus.set_row_hot(False)
        return False


def attach_row_ink(row, plus) -> _RowInk:
    """Darken ``plus`` while ``row`` is hovered (it is shown all the time)."""
    return _RowInk(row, plus)


class _HoverWatch(QObject):
    """Shows ``widgets`` while the pointer is over ``row`` (or one of them) and hides them otherwise."""

    def __init__(self, row, widgets, enabled=lambda: True):
        super().__init__(row)
        self.row, self.widgets, self.enabled = row, list(widgets), enabled
        row.installEventFilter(self)
        for w in self.widgets:
            w.installEventFilter(self)
            w.hide()

    def set_shown(self, on: bool) -> None:
        for w in self.widgets:
            w.setVisible(bool(on) and self.enabled())

    def eventFilter(self, watched, event):           # noqa: N802 - Qt's name
        kind = event.type()
        if kind == QEvent.Enter:
            self.set_shown(True)
        elif kind == QEvent.Leave and watched is self.row:
            # the pointer may have moved onto one of the actions, which are the row's children
            from qtpy.QtGui import QCursor

            if not self.row.rect().contains(self.row.mapFromGlobal(QCursor.pos())):
                self.set_shown(False)
        return False


def attach_hover(row, widgets, enabled=lambda: True) -> _HoverWatch:
    """Make ``widgets`` appear on ``row`` only while it is hovered."""
    return _HoverWatch(row, widgets, enabled)


def make_row(nav_button_class):
    """`SubpageRow` built on the sidebar's `NavButton` (passed in: this module cannot import the sidebar, which
    imports it)."""

    class SubpageRow(nav_button_class):
        """One added page's row."""

        picked = Signal(str)
        renamed = Signal(int, str)
        deleted = Signal(int)
        moved = Signal(int, int)
        dragging = Signal(int, int)             # number, the pointer's y in the list's coordinates
        dropped = Signal(int)                   # number

        def __init__(self, t, number: int, view: str, name: str, glyph_px: int, parent=None):
            super().__init__(t, name, size="sm", bar=True, parent=parent)
            self.number, self.view, self.page_name = number, view, name
            self._press: QPoint | None = None
            self._drag = False
            self._edit: QLineEdit | None = None
            self.delete_glyph = _Glyph(t, "x", glyph_px, "Delete this page")
            self.delete_glyph.clicked.connect(lambda: self.deleted.emit(self.number))
            self.layout().addWidget(self.delete_glyph)
            self._watch = attach_hover(self, [self.delete_glyph])
            self.setToolTip("")
            self.clicked.connect(lambda: self.picked.emit(self.view))

        # -- the mouse ------------------------------------------------------------------------------------- #
        def mousePressEvent(self, e):            # noqa: N802 - Qt's name
            if e.button() == Qt.LeftButton:
                self._press = e.pos()
                self._drag = False
            super().mousePressEvent(e)

        def mouseMoveEvent(self, e):             # noqa: N802 - Qt's name
            if self._press is not None and (e.buttons() & Qt.LeftButton):
                if not self._drag and abs(e.pos().y() - self._press.y()) >= DRAG_SLOP:
                    self._drag = True
                    self.setCursor(Qt.ClosedHandCursor)
                if self._drag:
                    parent = self.parentWidget()
                    y = self.mapTo(parent, e.pos()).y() if parent is not None else e.pos().y()
                    self.dragging.emit(self.number, y)
                    return
            super().mouseMoveEvent(e)

        def mouseReleaseEvent(self, e):          # noqa: N802 - Qt's name
            was_drag = self._drag
            self._press, self._drag = None, False
            self.setCursor(Qt.PointingHandCursor)
            if was_drag:
                self.dropped.emit(self.number)
                return
            super().mouseReleaseEvent(e)

        def mouseDoubleClickEvent(self, e):      # noqa: N802 - Qt's name
            if e.button() == Qt.LeftButton:
                self.begin_rename()

        def contextMenuEvent(self, e):           # noqa: N802 - Qt's name
            menu = Panel(self.t, self.window())
            menu.set_trigger(self)
            menu_row(self.t, menu, "Rename", glyph="pencil", on_click=self.begin_rename)
            menu_row(self.t, menu, "Move up", glyph="chevron-up", on_click=lambda: self.moved.emit(self.number, -1))
            menu_row(self.t, menu, "Move down", glyph="chevron-down", on_click=lambda: self.moved.emit(self.number, 1))
            menu_row(self.t, menu, "Delete", glyph="trash", on_click=lambda: self.deleted.emit(self.number))
            menu.popup_under(self)
            self._menu = menu

        # -- the name, edited in place --------------------------------------------------------------------- #
        def begin_rename(self) -> None:
            """Replace the name by a field holding it; Enter keeps the new name, Escape or leaving the field keeps
            the old one."""
            if self._edit is not None:
                return
            edit = QLineEdit(self.page_name, self)
            edit.setFont(ty.ui("sm"))
            edit.setStyleSheet(
                f"QLineEdit {{ background:{self.t.background}; color:{self.t.foreground};"
                f" border:1px solid {self.t.ring}; border-radius:{self.t.radius_sm:g}px; padding:0 4px; }}")
            m = self.layout().contentsMargins()
            edit.setGeometry(m.left(), 2, max(40, self.width() - m.left() - m.right()), self.height() - 4)
            edit.selectAll()
            edit.returnPressed.connect(lambda: self._finish_rename(True))
            edit.editingFinished.connect(lambda: self._finish_rename(False))
            edit.installEventFilter(self)
            edit.show()
            edit.setFocus()
            self._edit = edit

        def _finish_rename(self, accept: bool) -> None:
            edit, self._edit = self._edit, None
            if edit is None:
                return
            text = edit.text().strip()
            discard(edit)
            if accept and text and text != self.page_name:
                self.renamed.emit(self.number, text)

        def eventFilter(self, watched, event):   # noqa: N802 - Qt's name
            if watched is self._edit and event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
                edit, self._edit = self._edit, None
                if edit is not None:
                    discard(edit)
                return True
            return False

        def is_editing(self) -> bool:
            return self._edit is not None

    return SubpageRow


class SubpageList(QWidget):
    """A section's added pages, one `SubpageRow` each, indented under the section's row with the pages list's
    hairline down the left. ``rows`` maps number -> row."""

    picked = Signal(str)
    renamed = Signal(int, str)
    deleted = Signal(int)
    moved = Signal(int, int)
    dropped = Signal(int, int)                  # number, the index it was dropped at

    def __init__(self, t, row_class, indent: int, pad: int, gap: int, row_h: int, glyph_px: int, parent=None):
        super().__init__(parent)
        self.t = t
        self._row_class, self._glyph_px = row_class, glyph_px
        self.indent, self.row_h, self.gap = indent, row_h, gap
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(indent + pad, gap // 2, 0, gap // 2)
        lay.setSpacing(gap)
        self.rows: dict = {}
        self._order: list = []
        self._drag_target: int | None = None

    def set_pages(self, entries, current_view: str = "") -> None:
        """Redraw: ``entries`` = ``[(number, view, name)]`` in order."""
        for row in list(self.rows.values()):
            discard(row)
        self.rows.clear()
        self._order = [n for n, _v, _name in entries]
        lay = self.layout()
        for number, view, name in entries:
            row = self._row_class(self.t, number, view, name, self._glyph_px)
            row.picked.connect(self.picked)
            row.renamed.connect(self.renamed)
            row.deleted.connect(self.deleted)
            row.moved.connect(self.moved)
            row.dragging.connect(self._on_dragging)
            row.dropped.connect(self._on_dropped)
            self.rows[number] = row
            lay.addWidget(row)
        self.set_current(current_view)
        self.setVisible(bool(entries))
        self.setFixedHeight(self.natural())

    def natural(self) -> int:
        m = self.layout().contentsMargins()
        n = len(self.rows)
        return 0 if not n else m.top() + m.bottom() + n * self.row_h + (n - 1) * self.gap

    def set_current(self, view: str) -> None:
        for row in self.rows.values():
            row.set_active(_sp.page_of_view(row.view) == _sp.page_of_view(view))

    def squeeze(self, width: int) -> None:
        inner = max(0, width - self.indent - 2 * self.gap)
        for row in self.rows.values():
            row.setMaximumWidth(inner)
            row.squeeze(inner)

    # -- dragging a row to another place ---------------------------------------------------------------------- #
    def index_at(self, y: int) -> int:
        """The place in the list a pointer at ``y`` (this list's coordinates) is over."""
        m = self.layout().contentsMargins()
        step = self.row_h + self.gap
        return max(0, min(len(self._order) - 1, int((y - m.top()) // step))) if self._order else 0

    def _on_dragging(self, number: int, y: int) -> None:
        self._drag_target = self.index_at(y)
        self.update()

    def _on_dropped(self, number: int) -> None:
        target, self._drag_target = self._drag_target, None
        self.update()
        if target is not None and number in self._order and target != self._order.index(number):
            self.dropped.emit(number, target)

    def paintEvent(self, e):                     # noqa: N802 - Qt's name
        super().paintEvent(e)
        p = QPainter(self)
        w = 1.0 / self.devicePixelRatioF()
        m = self.layout().contentsMargins()
        p.fillRect(QRectF(self.indent, m.top(), w, max(0.0, self.height() - m.top() - m.bottom())),
                   QColor(self.t.sidebar_border))
        if self._drag_target is not None:
            y = m.top() + self._drag_target * (self.row_h + self.gap)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self.t.sidebar_primary))
            p.drawRoundedRect(QRectF(self.indent + 4, y - 1, self.width() - self.indent - 6, 2), 1, 1)
        p.end()


class PageChoice(QComboBox):
    """A drop-down of a section's pages, drawn as the workspace's other combo boxes with a chevron at its right end
    (`ramp_dialog._combo_css` has none, so a closed box read as a text field)."""

    ROOM = 26

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        from .ramp_dialog import _combo_css

        self.setFont(ty.ui("sm"))
        self.setStyleSheet(_combo_css(t) + f"QComboBox {{ padding-right:{self.ROOM}px; }}")
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

    def paintEvent(self, e):                     # noqa: N802 - Qt's name
        super().paintEvent(e)
        from ..theme import icons

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        size = 14.0
        icons.paint(p, "chevron-down", QRectF(self.width() - self.ROOM + 6, (self.height() - size) / 2, size, size),
                    self.t.muted_foreground, stroke=2.0)
        p.end()
