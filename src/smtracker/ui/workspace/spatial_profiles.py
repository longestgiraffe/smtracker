"""Where each population's jumps sit along and across the cell: the Spatial Map's Profiles card (asked 17 Oct 2026: "A
distribution graph or histogram clickable for the maps to measure distance from poles like the manuscript scripts (x
and y direction)"; made the manuscript's the same day: "can we not do it more like it is in manuscript version?").

As the manuscript's short-axis figure (`manuscript/sqdmap`): ONE ROW PER POPULATION, colour = the condition (on a day
page the condition bold and each day a shade of it), the even-through-the-volume reference dashed. Two columns: ACROSS
the cell (the relative short-axis position, -1..1, cylinder only -- the manuscript's own computation,
`spatialsqd.short_axis_profile`) and ALONG it (distance from the nearer pole, nm), in the same style. The key has an EYE
per condition (`condition_key.ConditionKey`, asked the same day: "an eye whenever there is a condition"): a hidden one
leaves every row and the export. (A Bars view whose bins marked the maps was built and taken out the same evening:
"the clickable histogram is not needed".) Thin Qt: every number is `spatial_map_view`'s (`profile_card_payload`,
`day_profile_payload`, `PROFILE_NOTE`). SINCE 23 NOV 2026 ALSO THE SDA MAPS' x AND y GRAPHS (`spatialsda.sda_profiles`):
a row per map, the localisations as a density, speed, D and acceleration as the mean value per bin in their unit, with
no even reference and a gap where a bin has too few steps; the payload carries the heading, the note and the y label.
"""
from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ... import chart_text, figstyle
from ...spatial_map_view import PROFILE_NOTE
from ..theme import typography as ty
from ..widgets.primitives import discard, label
from ..widgets.settings import InfoLink
from .condition_key import ConditionKey
from .jumps_chart import _Table
from .qc_chart import _DATA, _PanelCard
from .toolbar import rem, text_px

__all__ = ["ProfileChart", "ProfilesCard"]


def _round_ticks(top: float) -> list:
    """Round ticks from 0 up to ``top``, three or four (1, 2 or 5 times a power of ten apart)."""
    import math

    if not top > 0:
        return [0.0]
    raw = top / 3.0
    power = 10.0 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1.0, 2.0, 5.0, 10.0) if m * power >= raw)
    return [k * step for k in range(int(top // step) + 1)]


def _tick_text(v: float) -> str:
    """A y number, as every scale's end is written (`chart_text.scale_number`: "0.05", "600", "1,200")."""
    return chart_text.scale_number(v)


#: The widest y number a row is given room for, so every row of a card starts its plot at one x.
TICK_ROOM = "0,000.0"


class ProfileChart(QWidget):
    """One population's profile on one axis: a density line per entry (bold or thin, in its colour) over the dashed
    even reference. ``axis`` is "long" (distance from the nearer pole, nm; the cap's edge dotted) or "short" (the
    signed relative short-axis position)."""

    def __init__(self, t, axis: str, parent=None):
        super().__init__(parent)
        self.t = t
        self.axis = axis
        self.setMinimumHeight(170)
        self._payload: dict | None = None
        self._colours: dict = {}
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))
        #: Room kept left of the plot for y numbers, set by the card when ANY of its rows carries them (an SDA value
        #: map), so every row's plot starts at one x.
        self.reserve_px = 0.0

    def set_payload(self, payload: dict | None, colours: dict) -> None:
        self._payload, self._colours = payload, dict(colours)
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pl = self._payload
        if not pl:
            p.end()
            return
        key = self.axis
        # a value map's profile (an SDA speed, D or acceleration) has no even reference and a gap where a bin has too
        # few steps (None)
        centres, reference = pl[f"{key}_centres"], pl.get(f"{key}_reference") or []
        lines = [ln for ln in pl["lines"] if ln.get(key)]
        top = max([*reference, *[v for ln in lines for v in ln[key] if v is not None], 1e-9]) * 1.1
        lo, hi = (0.0, pl["half_length_um"]) if key == "long" else (-1.0, 1.0)
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        # A VALUE MAP'S NUMBERS ARE READ (an SDA speed in um/s): its y axis carries ticks; a density's has none (its
        # scale is area 1, the shape is what is compared)
        value = bool(pl.get("unit"))
        y_ticks = _round_ticks(top) if value else []
        left = 40.0 + max(self.reserve_px,
                          max([fm.horizontalAdvance(_tick_text(v)) for v in y_ticks] + [0.0]) + 4.0 if value else 0.0)
        box = QRectF(left, 6.0, max(10.0, self.width() - left - 8.0),
                     max(10.0, self.height() - 12.0 - 2.4 * fm.height()))
        muted = QColor(self.t.muted_foreground)

        def px(v):
            return box.left() + (v - lo) / (hi - lo) * box.width()

        def py(v):
            return box.bottom() - v / top * box.height()

        frame = QPen(QColor(self.t.border))
        frame.setWidthF(0.8)
        p.setPen(frame)
        p.setBrush(Qt.NoBrush)
        p.drawRect(box)
        if key == "long":
            cap = QPen(muted)
            cap.setStyle(Qt.DotLine)
            p.setPen(cap)
            p.drawLine(QPointF(px(pl["cap_um"]), box.top()), QPointF(px(pl["cap_um"]), box.bottom()))
            p.drawText(QRectF(px(pl["cap_um"]) + 3, box.top() + 1, 60, fm.height()), Qt.AlignLeft, "cap")
        ref = QPen(muted)
        ref.setWidthF(1.3)
        ref.setStyle(Qt.DashLine)
        p.setPen(ref)
        pts = [QPointF(px(c), py(v)) for c, v in zip(centres, reference)]
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        for ln in sorted(lines, key=lambda ln: ln.get("bold", False)):        # the bold ones on top
            pen = QPen(QColor(self._colours.get(ln["key"], _DATA)))
            pen.setWidthF(2.0 if ln.get("bold") else 1.1)
            p.setPen(pen)
            pts = [None if v is None else QPointF(px(c), py(v)) for c, v in zip(centres, ln[key])]
            for a, b in zip(pts, pts[1:]):
                if a is not None and b is not None:
                    p.drawLine(a, b)
        p.setPen(muted)
        for v in y_ticks:
            if v <= top:
                p.drawText(QRectF(box.left() - 60.0 - 4.0, py(v) - fm.height() / 2, 60.0, fm.height()),
                           Qt.AlignRight | Qt.AlignVCenter, _tick_text(v))
        ticks = ([-1.0, -0.5, 0.0, 0.5, 1.0] if key == "short"
                 else [v for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0) if v <= hi + 1e-9])
        for v in ticks:
            p.drawText(QRectF(px(v) - 30, box.bottom() + 3, 60, fm.height()), Qt.AlignHCenter,
                       f"{v:g}" if key == "short" else f"{1000 * v:.0f}")
        p.drawText(QRectF(box.left(), box.bottom() + 3 + fm.height(), box.width(), fm.height()), Qt.AlignRight,
                   "from the nearer pole, nm" if key == "long" else "relative short-axis position")
        p.save()
        p.translate(9, box.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-70, -fm.height() / 2, 140, fm.height()), Qt.AlignCenter, pl.get("y_label") or "density")
        p.restore()
        p.end()


class ProfilesCard(_PanelCard):
    """The Profiles card: the key (an eye per condition), one row per population (its two charts side by side) and
    the table. Fed by a ``build(kind)`` that returns one population's plain payload (or None), so the overview and
    the day page share it."""

    def __init__(self, t, title: str, parent=None):
        self.t = t
        body = QWidget()
        body.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        lay.setSpacing(rem(0.5))
        top = QHBoxLayout()
        top.setSpacing(rem(0.75))
        self.key = ConditionKey(t)
        top.addWidget(self.key, 1)
        self.note = InfoLink(t, "How to read the profiles", PROFILE_NOTE)
        top.addWidget(self.note, 0, Qt.AlignTop)
        lay.addLayout(top)
        heads = QHBoxLayout()
        heads.setSpacing(rem(1.0))
        for text in ("Along the cell", "Across the cell"):
            heads.addWidget(label(text, ty.ui("xs", "medium"), t.foreground), 1)
        lay.addLayout(heads)
        self._rows_host = QVBoxLayout()
        self._rows_host.setSpacing(rem(0.5))
        lay.addLayout(self._rows_host)
        self.table = _Table(t)
        lay.addWidget(self.table)
        #: The SDA value maps' means (speed, D, acceleration), a table of their own: their columns are not the
        #: distributions'.
        self.value_table = _Table(t)
        lay.addWidget(self.value_table)
        self.value_table.hide()
        super().__init__(t, title, body, parent=parent, flush=False)
        #: What the title says after its heading ("Where the jumps sit · <this>"): the heading follows the payload
        #: (SDA's x and y graphs read "Along and across the cell").
        self._suffix = title.partition(" · ")[2]
        self.enable_export()
        self._rows: dict = {}
        self._titles: list = []
        self._build = None
        self._kinds: list = []
        self._colours: dict = {}
        self._current: str | None = None
        #: The population was picked by a click on a map (its row says so).
        self._marked = False
        self.key.hidden_changed.connect(lambda _h: self._draw())

    def set_source(self, build, kinds, titles, colours: dict, select: str | None = None) -> None:
        """What to draw: ``build(kind)`` -> the plain payload or None; ``kinds``/``titles`` the populations, one row
        each; ``colours`` key -> colour; ``select`` the population clicked on a map (its row's title is marked).
        Hidden when no population has a profile (a method without populations)."""
        self._build, self._kinds, self._colours = build, list(kinds), dict(colours)
        self._titles = [str(t) for t in titles]
        self._current = select if select in self._kinds else (self._kinds[0] if self._kinds else None)
        self._marked = select in self._kinds
        for row in self._rows.values():
            discard(row["host"])
        self._rows = {}
        for kind, title in zip(self._kinds, self._titles):
            host = QWidget()
            host.setStyleSheet("background:transparent;")
            col = QVBoxLayout(host)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(rem(0.25))
            name = label(title, ty.ui("xs", "medium"), self.t.foreground)
            col.addWidget(name)
            charts = QHBoxLayout()
            charts.setSpacing(rem(1.0))
            long_chart, short_chart = ProfileChart(self.t, "long"), ProfileChart(self.t, "short")
            charts.addWidget(long_chart, 1)
            charts.addWidget(short_chart, 1)
            col.addLayout(charts)
            self._rows_host.addWidget(host)
            self._rows[kind] = {"host": host, "name": name, "long": long_chart, "short": short_chart}
        self._draw()

    def select(self, kind: str) -> None:
        if kind in self._kinds:
            self._current, self._marked = kind, True
            self._draw()

    def current_kind(self) -> str | None:
        return self._current

    def marked_kind(self) -> str | None:
        """The population a click on a map picked, or None when none was."""
        return self._current if self._marked else None

    def export_payload(self) -> dict | None:
        """What the card's Export draws: every population's payload (one row each) without the hidden lines, the
        colours and the names -- as shown."""
        panels = []
        for kind, title in zip(self._kinds, self._titles):
            payload = self._shown(self._build(kind) if self._build else None)
            if payload is not None:
                panels.append({"kind": kind, "title": title, "payload": payload})
        if not panels:
            return None
        return {"panels": panels, "colours": self._line_colours(panels[0]["payload"])}

    def _shown(self, payload: dict | None) -> dict | None:
        """``payload`` without the lines the key hides."""
        if payload is None:
            return None
        hidden = self.key.hidden()
        return {**payload, "lines": [ln for ln in payload["lines"] if str(ln["key"]) not in hidden]}

    def _line_colours(self, payload: dict) -> dict:
        colours = dict(self._colours)
        for ln in payload["lines"]:
            if "day_index" in ln:
                colours[ln["key"]] = figstyle.day_shade(self._colours.get("condition", _DATA), ln["day_index"],
                                                        ln["n_days"])
        return colours

    def _draw(self) -> None:
        payloads = {k: (self._build(k) if self._build else None) for k in self._kinds}
        shown = [k for k in self._kinds if payloads[k] is not None]
        self.setVisible(bool(shown))
        if not shown:
            return
        first = payloads[shown[0]]
        colours = self._line_colours(first)
        heading = first.get("heading") or "Where the jumps sit"
        # never the heading twice ("Along and across the cell · along and across the cell" on SDA's grid)
        self.set_title(f"{heading} · {self._suffix}" if self._suffix and self._suffix.lower() != heading.lower()
                       else heading)
        self.note.set_tip(first.get("note") or PROFILE_NOTE)
        # the even reference in the key where some row draws one (an SDA value map has none)
        even = any(payloads[k].get("long_reference") for k in shown)
        self.key.blockSignals(True)
        self.key.set_entries([(str(ln["key"]), ln["name"], colours.get(ln["key"], _DATA)) for ln in first["lines"]],
                             fixed=[("even distribution", None, "dashed")] if even else [], mark="line",
                             bold={str(ln["key"]) for ln in first["lines"] if ln.get("bold")})
        self.key.blockSignals(False)
        hidden = self.key.hidden()
        rows, value_rows, value_columns = [], [], None
        # one left edge for every row where some row has y numbers (an SDA value map)
        any_row = next(iter(self._rows.values()), None)
        reserve = (QFontMetricsF(any_row["long"]._font).horizontalAdvance(TICK_ROOM) + 4.0
                   if any_row is not None and any(payloads[k].get("unit") for k in shown) else 0.0)
        for kind, title in zip(self._kinds, self._titles):
            row, payload = self._rows.get(kind), payloads[kind]
            if row is None:
                continue
            row["host"].setVisible(payload is not None)
            if payload is None:
                continue
            row["name"].setText(title + ("  ·  clicked on the map" if self._marked and kind == self._current
                                         and len(shown) > 1 else ""))
            for axis in ("long", "short"):
                row[axis].reserve_px = reserve
                row[axis].set_payload(self._shown(payload), colours)
            names = {ln["name"] for ln in payload["lines"] if str(ln["key"]) in hidden}
            if payload.get("unit"):
                # an SDA value map: its means in its own table, the unit with its name
                value_columns = payload["columns"]
                value_rows += [[f"{title} ({payload['unit']})", *r] for r in payload["rows"] if r[0] not in names]
            else:
                rows += [[title, *r] for r in payload["rows"] if r[0] not in names]
        name = first.get("row_name") or "Population"
        distribution = next((payloads[k] for k in shown if not payloads[k].get("unit")), None)
        self.table.setVisible(distribution is not None)
        if distribution is not None:
            self.table.set_table([name, *distribution["columns"]], rows)
        self.value_table.setVisible(value_columns is not None)
        if value_columns is not None:
            self.value_table.set_table([name, *value_columns], value_rows)
