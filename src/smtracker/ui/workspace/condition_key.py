"""A chart's key of conditions, each with an EYE that hides it (asked 17 Oct 2026: "how about we do an eye whenever there
is a condition like msd, sqd map etc, where u can hide it").

One entry per condition (or per day on a day page): the eye, the condition's mark in its colour, its name. A click on
the eye hides or shows that condition on the chart beside it (`hidden_changed`, the set of hidden keys); a click on the
name, where the owner offers it (`pickable`), is `picked` -- the MSD overview opens that condition's replicate days.
Fixed entries (an even-distribution reference, say) follow, without an eye. The entries flow onto as many rows as the
width needs; the widget's height is its rows'. Thin Qt: it holds which keys are hidden, nothing else.

GROUPED BY THE FACTORS (`set_grouping`, folder designs, 25 Oct 2026): with a nesting of two or more factors and
entries that are conditions carrying them, the entries stand in the nesting's order, each named by its inner values,
after a header with its outer value (`factor_layout`). Entries that are not such conditions (a day page's days) are
drawn as given.
"""
from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

from ..theme import icons
from ..theme import typography as ty
from .toolbar import text_px

__all__ = ["ConditionKey"]

_EYE = 14.0
_MARK = 14.0
_GAP = 16.0


class ConditionKey(QWidget):
    """``set_entries([(key, name, colour)], fixed=[(name, colour, "dashed" | "line")], mark="dot" | "line")``."""

    #: The keys now hidden.
    hidden_changed = Signal(object)
    #: A name was clicked (only when `pickable`).
    picked = Signal(str)

    def __init__(self, t, parent=None, *, pickable: bool = False):
        super().__init__(parent)
        self.t = t
        self.pickable = pickable
        self._entries: list = []
        self._fixed: list = []
        self._mark = "dot"
        self._bold: set = set()
        self._hidden: set = set()
        self._picked: str | None = None
        self._hits: list = []
        self._grouping: tuple = ({}, ())
        self._font = ty.ui("xs", px=text_px(0.8))
        self._bold_font = ty.ui("xs", "semibold", px=text_px(0.8))
        self._row_h = QFontMetricsF(self._font).height() + 8
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(int(self._row_h))

    # -- state ------------------------------------------------------------------------------------------------- #
    def set_entries(self, entries: list, *, fixed: list | None = None, mark: str = "dot",
                    bold: set | None = None) -> None:
        """``entries`` ``[(key, name, colour)]``; a key no longer present stops being hidden."""
        self._entries = [(str(k), str(n), str(c)) for k, n, c in entries]
        self._fixed = list(fixed or [])
        self._mark = mark
        self._bold = set(bold or ())
        keys = {k for k, _n, _c in self._entries}
        if not self._hidden <= keys:
            self._hidden &= keys
            self.hidden_changed.emit(set(self._hidden))
        self._relayout()
        self.update()

    def set_grouping(self, factors: dict, nesting) -> None:
        """Group the entries by the folder design's factors (module docstring); ``({}, ())`` ungroups."""
        self._grouping = (dict(factors or {}), tuple(nesting or ()))
        self._relayout()
        self.update()

    def _grouped_entries(self) -> list:
        """``[("group", label) | ("entry", key, name, colour)]``: the entries, grouped where they can be."""
        from ...factor_layout import columns, nested

        factors, nesting = self._grouping
        keys = [k for k, _n, _c in self._entries]
        if len(nesting) < 2 or not keys or not all(
                all(n in (factors.get(k) or {}) for n in nesting) for k in keys):
            return [("entry", k, n, c) for k, n, c in self._entries]
        by_key = {k: (n, c) for k, n, c in self._entries}
        order = nested(keys, factors, nesting)
        layout = columns(order, factors, nesting)
        out = []
        for g in layout["groups"]:
            out.append(("group", g["label"]))
            for i in range(g["first"], g["last"] + 1):
                k = order[i]
                out.append(("entry", k, layout["labels"][i], by_key[k][1]))
        return out if layout["groups"] else [("entry", k, n, c) for k, n, c in self._entries]

    def hidden(self) -> set:
        return set(self._hidden)

    def set_hidden(self, keys) -> None:
        self._hidden = {str(k) for k in keys} & {k for k, _n, _c in self._entries}
        self.update()
        self.hidden_changed.emit(set(self._hidden))

    def set_picked(self, key: str | None) -> None:
        """The name shown as picked (the MSD overview's condition whose days are open), or None."""
        self._picked = key
        self.update()

    def toggle(self, key: str) -> None:
        self._hidden ^= {key}
        self.update()
        self.hidden_changed.emit(set(self._hidden))

    # -- layout ------------------------------------------------------------------------------------------------ #
    def _items(self) -> list:
        fm = QFontMetricsF(self._font)
        bold = QFontMetricsF(self._bold_font)
        out = []
        for item in self._grouped_entries():
            if item[0] == "group":
                out.append(("group", None, item[1], None, bold.horizontalAdvance(item[1]) + 2))
            else:
                _kind, k, n, c = item
                out.append(("entry", k, n, c, _EYE + 4 + _MARK + 5 + fm.horizontalAdvance(n)))
        out += [("fixed", None, n, c, _MARK + 5 + fm.horizontalAdvance(n), s) for n, c, s in self._fixed]
        return out

    def _place(self, width: float) -> list:
        """``[(item, x, row)]`` flowing over ``width``."""
        x, row, placed = 0.0, 0, []
        for item in self._items():
            w = item[4]
            if x > 0 and x + w > width:
                x, row = 0.0, row + 1
            placed.append((item, x, row))
            x += w + _GAP
        return placed

    def _relayout(self) -> None:
        rows = 1 + max((r for _i, _x, r in self._place(max(50.0, float(self.width())))), default=0)
        height = int(rows * self._row_h)
        if height != self.height():
            self.setFixedHeight(height)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._relayout()

    # -- painting and clicks ----------------------------------------------------------------------------------- #
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        muted = QColor(self.t.muted_foreground)
        self._hits = []
        for item, x, row in self._place(float(self.width())):
            y = row * self._row_h
            cy = y + self._row_h / 2.0
            kind, key, name, colour = item[0], item[1], item[2], item[3]
            if kind == "group":
                p.setFont(self._bold_font)
                p.setPen(muted)
                p.drawText(QRectF(x, y, item[4], self._row_h), Qt.AlignLeft | Qt.AlignVCenter, name)
                p.setFont(self._font)
                continue
            if kind == "entry":
                hidden = key in self._hidden
                eye = QRectF(x, cy - _EYE / 2, _EYE, _EYE)
                icons.paint(p, "eye-off" if hidden else "eye", eye, muted.name() if hidden else self.t.foreground)
                self._hits.append(("eye", key, eye.adjusted(-3, -3, 3, 3)))
                mx = x + _EYE + 4
                ink = QColor(colour)
                if hidden:
                    ink.setAlphaF(0.3)
                if self._mark == "line":
                    pen = QPen(ink)
                    pen.setWidthF(2.4 if key in self._bold else 1.4)
                    p.setPen(pen)
                    p.drawLine(QPointF(mx, cy), QPointF(mx + _MARK, cy))
                else:
                    p.setPen(Qt.NoPen)
                    p.setBrush(ink)
                    p.drawEllipse(QPointF(mx + _MARK / 2, cy), 4.0, 4.0)
                tx = mx + _MARK + 5
                text = QRectF(tx, y, fm.horizontalAdvance(name) + 2, self._row_h)
                p.setFont(self._bold_font if key == self._picked else self._font)
                p.setPen(muted if hidden else QColor(self.t.foreground))
                p.drawText(text, Qt.AlignLeft | Qt.AlignVCenter, name)
                p.setFont(self._font)
                if hidden:
                    p.drawLine(QPointF(text.left(), cy), QPointF(text.right(), cy))
                if self.pickable:
                    self._hits.append(("name", key, QRectF(mx, y, text.right() - mx, self._row_h)))
            else:
                style = item[5]
                pen = QPen(QColor(colour or self.t.muted_foreground))
                pen.setWidthF(1.4)
                if style == "dashed":
                    pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.drawLine(QPointF(x, cy), QPointF(x + _MARK, cy))
                p.setPen(QColor(self.t.foreground))
                p.drawText(QRectF(x + _MARK + 5, y, fm.horizontalAdvance(name) + 2, self._row_h),
                           Qt.AlignLeft | Qt.AlignVCenter, name)
        p.end()

    def _hit(self, pos: QPointF):
        return next(((what, key) for what, key, rect in self._hits if rect.contains(pos)), None)

    def mouseMoveEvent(self, e) -> None:
        self.setCursor(Qt.PointingHandCursor if self._hit(e.position()) else Qt.ArrowCursor)
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e) -> None:
        hit = self._hit(e.position()) if e.button() == Qt.LeftButton else None
        if hit is not None:
            what, key = hit
            if what == "eye":
                self.toggle(key)
            else:
                self.picked.emit(key)
            return
        super().mousePressEvent(e)
