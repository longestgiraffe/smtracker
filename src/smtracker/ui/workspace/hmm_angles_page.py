"""The HMM section's Jump angles page (22 Nov 2026): turning back per HMM state, beside chance at the same places.

Asked: "what about the angle map we planned where u see how tracks change in their angle? ... HMM? appd?", then
"both ja". The page reads the HMM section's Run: once a condition's HMM fit lands and the page is on show, the window
computes its angles (`hmm_angles.compute_hmm_angles`, a pool job through the section cache) and hands them here.

Thin Qt: every number and word is `hmm_angles_view`'s; nothing here computes. On top the strip across conditions
(turning back beyond chance, one panel a state, each replicate a dot, the mean +- SD, 0 and the band free molecules
read within), then each state's Welch table, then one card a condition with each state's angle histogram beside its
chance.
"""
from __future__ import annotations

import math

from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ... import chart_text
from ...condition_order import apply as order_apply
from ...hmm_angles_view import (CHANCE_BOX, CHANCE_TIP, READING_NOTES, card_caption, card_notes, comparison_tables,
                                export_payload, histogram, mode, state_line, strip_payload, strip_title, waiting_text)
from ..theme import typography as ty
from ..widgets.cardgrid import CardGrid
from ..widgets.primitives import Checkbox, discard, fit_label, label
from ..widgets.settings import InfoLink
from .jumps_chart import _Table
from .qc_chart import _DATA, _PanelCard
from .toolbar import rem

__all__ = ["AngleHistogram", "AnglesStrip", "AnglesCard", "HMMAnglesPage"]


def _ticks(lo: float, hi: float, n: int = 4) -> list:
    """Round ticks inside ``lo``..``hi``."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    step = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * step >= raw:
            step *= m
            break
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out


class AngleHistogram(QWidget):
    """One state's angle density as a HALF CIRCLE (asked 22 Nov 2026: "an angular map instead of bars ... something
    like a half circle", then "make 0° left and 180° right? back and forward?"): 0 degrees, forward, to the LEFT; 180,
    turned back, to the RIGHT; each 10-degree bin a wedge as long as that angle is common, in the condition's colour,
    and -- against chance only -- chance at the same places as a dashed outline over them.
    Payload: `hmm_angles_view.histogram` (its ``chance`` None for the data alone)."""

    H = 176
    _TOP, _LABEL, _BOTTOM, _SIDE = 20, 16, 18, 26

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._data: dict | None = None
        self._colour = _DATA
        self.setFixedHeight(self.H)
        self.setMinimumWidth(rem(12))
        #: The half circle as last drawn, ``(centre x, centre y, radius)`` (tests).
        self.drawn: tuple | None = None

    def set_data(self, data: dict | None, colour: str) -> None:
        self._data, self._colour = data, colour
        self.update()

    def _key(self, p: QPainter, width: float) -> None:
        """The state's name at the top left, and at the top right what the wedges and the dashed line are."""
        small = ty.ui("xs")
        p.setFont(ty.ui("xs", "medium"))
        p.setPen(QColor(self.t.foreground))
        p.drawText(QRectF(0, 0, width, self._TOP - 2), Qt.AlignLeft | Qt.AlignVCenter,
                   str(self._data["name"]).capitalize())
        fm = QFontMetricsF(small)
        p.setFont(small)
        observed_w = fm.horizontalAdvance("observed")
        mid = (self._TOP - 2) / 2
        right = width
        if self._data.get("chance") is not None:
            chance_w = fm.horizontalAdvance("chance")
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(width - chance_w, 0, chance_w, self._TOP - 2), Qt.AlignRight | Qt.AlignVCenter,
                       "chance")
            key_pen = QPen(QColor(self.t.foreground), 1.5)
            key_pen.setStyle(Qt.DashLine)
            p.setPen(key_pen)
            p.drawLine(QPointF(width - chance_w - 22, mid), QPointF(width - chance_w - 4, mid))
            right = width - chance_w - 34
        x_obs = right - observed_w
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(QRectF(x_obs, 0, observed_w, self._TOP - 2), Qt.AlignLeft | Qt.AlignVCenter, "observed")
        swatch = QColor(self._colour)
        swatch.setAlphaF(0.55)
        p.fillRect(QRectF(x_obs - 14, mid - 4, 10, 8), swatch)

    def paintEvent(self, _e) -> None:
        if not self._data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        edges = [float(e) for e in self._data["edges"]]
        observed = [float(v) if v is not None else math.nan for v in self._data["observed"]]
        chance = [float(v) if v is not None else math.nan for v in (self._data.get("chance") or [])]
        finite = [v for v in observed + chance if math.isfinite(v)]
        top = max(finite) * 1.08 if finite else 1.0
        self._key(p, float(self.width()))
        # THE HALF CIRCLE: as large as the height and the width allow, its centre on the baseline
        radius = max(10.0, min(self.height() - self._TOP - self._LABEL - self._BOTTOM,
                               (self.width() - 2 * self._SIDE) / 2.0))
        cx, cy = self.width() / 2.0, self._TOP + self._LABEL + radius
        self.drawn = (cx, cy, radius)

        def ring(r):
            return QRectF(cx - r, cy - r, 2 * r, 2 * r)

        def at(deg, r):                                 # forward (0) on the left, back (180) on the right
            return QPointF(cx - r * math.cos(math.radians(deg)), cy - r * math.sin(math.radians(deg)))

        def qt(deg):                                    # Qt's angles run anticlockwise from three o'clock
            return 180.0 - deg

        # the grid: two rings, the diameter and the upright, in the border's colour (a density has no number a
        # reader uses, so the rings are unlabelled)
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.setBrush(Qt.NoBrush)
        for share in (0.5, 1.0):
            p.drawArc(ring(radius * share), 0, 180 * 16)
        p.drawLine(at(0, radius), at(180, radius))
        p.drawLine(QPointF(cx, cy), at(90, radius))
        # the wedges: each bin from the centre out to its density
        fill = QColor(self._colour)
        fill.setAlphaF(0.55)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        for i, v in enumerate(observed):
            if math.isfinite(v) and v > 0:
                r = v / top * radius
                wedge = QPainterPath(QPointF(cx, cy))
                wedge.arcTo(ring(r), qt(edges[i + 1]), edges[i + 1] - edges[i])
                wedge.closeSubpath()
                p.drawPath(wedge)
        # chance: an arc over each bin at its density, joined bin to bin along the bin edges
        path = QPainterPath()
        started = False
        for i, v in enumerate(chance):
            if not math.isfinite(v):
                started = False
                continue
            r = v / top * radius
            if not started:
                path.moveTo(at(edges[i], r))
                started = True
            path.arcTo(ring(r), qt(edges[i]), -(edges[i + 1] - edges[i]))
        pen = QPen(QColor(self.t.foreground), 1.5)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        # the angles, outside the half circle: 0 forward (left), 90 across (top), 180 turned back (right)
        p.setFont(ty.ui("xs"))
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(QRectF(cx - radius - 30, cy + 2, 60, self._BOTTOM - 2), Qt.AlignHCenter | Qt.AlignTop, "0°")
        p.drawText(QRectF(cx + radius - 30, cy + 2, 60, self._BOTTOM - 2), Qt.AlignHCenter | Qt.AlignTop, "180°")
        p.drawText(QRectF(cx - 30, cy - radius - self._LABEL, 60, self._LABEL), Qt.AlignHCenter | Qt.AlignBottom,
                   "90°")
        p.drawText(QRectF(cx - radius * 0.95, cy + 2, radius * 0.6, self._BOTTOM - 2), Qt.AlignHCenter | Qt.AlignTop,
                   "forward")
        p.drawText(QRectF(cx + radius * 0.35, cy + 2, radius * 0.6, self._BOTTOM - 2), Qt.AlignHCenter | Qt.AlignTop,
                   "back")
        p.end()


class AnglesStrip(QWidget):
    """Turning back beyond chance across conditions: one panel a state, one column a condition, each replicate an open
    circle, the mean a filled one with +- 1 SD, 0 and the band free molecules read within shaded.
    Payload: `hmm_angles_view.strip_payload`."""

    H = 230
    _TOP, _BOTTOM, _LEFT, _GAP, _NAME_LINE = 22, 34, 44, 22.0, 15

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._payload: dict | None = None
        self.setFixedHeight(self.H)
        #: What each panel drew its names as (tests): ``{state: lines}``.
        self.drawn_names: dict = {}

    def set_payload(self, payload: dict | None) -> None:
        self._payload = payload
        self.update()

    def paintEvent(self, _e) -> None:
        if not self._payload or not self._payload.get("panels"):
            return
        panels = self._payload["panels"]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = (self.width() - self._GAP * (len(panels) - 1)) / len(panels)
        for i, panel in enumerate(panels):
            self._panel(p, QRectF(i * (w + self._GAP), 0, w, self.height()), panel, first=i == 0)
        p.end()

    def _panel(self, p, box: QRectF, panel: dict, *, first: bool) -> None:
        lo, hi = float(self._payload["lo"]), float(self._payload["hi"])
        band = self._payload.get("resolution")
        reference = float(self._payload.get("reference", 0.0))
        signed = reference == 0.0
        columns = panel["columns"]
        p.setFont(ty.ui("xs", "medium"))
        p.setPen(QColor(self.t.foreground))
        p.drawText(QRectF(box.left(), box.top(), box.width(), self._TOP - 4), Qt.AlignLeft | Qt.AlignVCenter,
                   f"{str(panel['name']).capitalize()} state")
        small = ty.ui("xs")
        fm = QFontMetricsF(small)
        plot_w = box.width() - self._LEFT - 4
        slot = plot_w / max(1, len(columns))
        names = [chart_text.name_lines(str(c["label"]), fm.horizontalAdvance, max(8.0, slot - 4), max_lines=2)
                 for c in columns]
        self.drawn_names[panel["name"]] = names
        extra = (max((len(n) for n in names), default=1) - 1) * self._NAME_LINE
        plot = QRectF(box.left() + self._LEFT, box.top() + self._TOP + 4, plot_w,
                      box.height() - self._TOP - 4 - self._BOTTOM - extra)

        def py(v):
            return plot.bottom() - (v - lo) / (hi - lo) * plot.height()

        if band is not None:
            # against chance: the band free molecules read within, around 0
            p.fillRect(QRectF(plot.left(), py(float(band)), plot.width(), py(-float(band)) - py(float(band))),
                       QColor(self.t.muted))
        p.setFont(small)
        for v in _ticks(lo, hi, 4):
            y = py(v)
            p.setPen(QPen(QColor(self.t.border), 1.0))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(box.left(), y - 8, self._LEFT - 6, 16), Qt.AlignRight | Qt.AlignVCenter,
                       (f"{v:+g}" if abs(v) > 1e-12 else "0") if signed else f"{v:g}")
        # THE REFERENCE: no preference (1) for the data, a free molecule (0) against chance
        p.setPen(QPen(QColor(self.t.muted_foreground), 1.6))
        p.drawLine(QPointF(plot.left(), py(reference)), QPointF(plot.right(), py(reference)))
        for i, column in enumerate(columns):
            cx = plot.left() + slot * (i + 0.5)
            colour = QColor(column["colour"])
            reps = column["replicates"]
            for j, v in enumerate(reps):
                dx = (j - (len(reps) - 1) / 2) * min(7.0, slot / (2 * max(1, len(reps))))
                p.setPen(QPen(colour, 1.4))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(cx - slot * 0.12 + dx, py(v)), 3.2, 3.2)
            mean, sd = column["mean"], column["sd"]
            mx = cx + slot * 0.14
            if math.isfinite(mean):
                if math.isfinite(sd):
                    p.setPen(QPen(QColor(self.t.foreground), 1.3))
                    p.drawLine(QPointF(mx, py(mean - sd)), QPointF(mx, py(mean + sd)))
                    for edge in (mean - sd, mean + sd):
                        p.drawLine(QPointF(mx - 3, py(edge)), QPointF(mx + 3, py(edge)))
                p.setPen(Qt.NoPen)
                p.setBrush(colour)
                p.drawEllipse(QPointF(mx, py(mean)), 4.0, 4.0)
            p.setPen(QColor(self.t.foreground))
            for k, line in enumerate(names[i]):
                p.drawText(QRectF(cx - slot / 2, plot.bottom() + 4 + k * self._NAME_LINE, slot, 16),
                           Qt.AlignHCenter | Qt.AlignTop, line)
        p.setBrush(Qt.NoBrush)


class _AnglesBody(QWidget):
    """A condition's card body: each state's histogram with its line, then the notes."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        self._lay.setSpacing(rem(0.5))
        self._parts: list = []
        self.status = fit_label("Waiting for the HMM…", ty.ui("sm"), t.muted_foreground)
        self.status.setWordWrap(True)
        self._lay.addWidget(self.status)
        self._result: dict | None = None
        self._colour = _DATA

    def _clear(self) -> None:
        for w in self._parts:
            discard(w)
        self._parts = []

    def set_status(self, text: str) -> None:
        self._clear()
        self._result = None
        self.status.setText(text)
        self.status.show()

    def set_result(self, result: dict, colour: str) -> None:
        self._clear()
        self._result, self._colour = result, colour
        self.status.hide()
        for i, state in enumerate(result["states"]):
            plot = AngleHistogram(self.t)
            self._lay.addWidget(plot)
            plot.set_data(histogram(result, i), colour)
            line = fit_label(state_line(state, bool(result.get("chance", True))), ty.ui("xs", tabular=True),
                             self.t.foreground)
            line.setWordWrap(True)
            self._lay.addWidget(line)
            self._parts += [plot, line]
        for text in card_notes(result):
            note = fit_label(text, ty.ui("xs"), self.t.muted_foreground)
            note.setWordWrap(True)
            self._lay.addWidget(note)
            self._parts.append(note)

    def set_colour(self, colour: str) -> None:
        if self._result is not None:
            self.set_result(self._result, colour)


class AnglesCard(_PanelCard):
    """One condition's jump angles."""

    def __init__(self, t, title: str, parent=None):
        self.body = _AnglesBody(t)
        super().__init__(t, title, self.body, parent=parent, flush=False)

    def set_result(self, result: dict, colour: str) -> None:
        if "error" in result:
            self.set_caption("")
            self.body.set_status(f"Not computed: {result['error']}")
            return
        self.set_caption(card_caption(result))
        self.body.set_result(result, colour)

    def set_status(self, text: str) -> None:
        self.set_caption("")
        self.body.set_status(text)


class HMMAnglesPage(QWidget):
    """The page: the strip across conditions, each state's Welch table, one card a condition."""

    #: The strip card's Export button (kind "hmm_angles", `figure_export`).
    export_requested = Signal()
    #: The Against chance box was ticked or unticked (the section asks for every condition again in that reading).
    chance_changed = Signal(bool)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        page = QWidget()
        page.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, rem(0.5))
        lay.setSpacing(rem(1.0))
        self.status = fit_label(waiting_text(False), ty.ui("sm"), t.muted_foreground)
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.reading = InfoLink(t, "How to read these numbers", READING_NOTES)
        lay.addWidget(self.reading)
        # AGAINST CHANCE (22 Nov 2026, asked: "make comparison with simulation optional, and also just show pure
        # data"): off, the data alone; ticked, the simulation at the same places is computed and the page reads
        # beyond it
        chance_row = QHBoxLayout()
        chance_row.setContentsMargins(0, 0, 0, 0)
        chance_row.setSpacing(rem(0.5))
        self.chance_box = Checkbox(t, checked=False)
        self.chance_box.setToolTip(CHANCE_TIP)
        self.chance_box.toggled.connect(self._chance_toggled)
        chance_row.addWidget(self.chance_box)
        chance_row.addWidget(label(CHANCE_BOX, ty.ui("sm"), t.foreground))
        chance_row.addStretch(1)
        lay.addLayout(chance_row)
        self.strip = AnglesStrip(t)
        strip_body = QWidget()
        strip_body.setStyleSheet("background:transparent;")
        strip_lay = QVBoxLayout(strip_body)
        strip_lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        strip_lay.setSpacing(rem(0.75))
        strip_lay.addWidget(self.strip)
        self._tables_host = QWidget()
        self._tables_host.setStyleSheet("background:transparent;")
        self._tables_lay = QVBoxLayout(self._tables_host)
        self._tables_lay.setContentsMargins(0, 0, 0, 0)
        self._tables_lay.setSpacing(rem(0.5))
        strip_lay.addWidget(self._tables_host)
        self._table_parts: list = []
        self.strip_card = _PanelCard(t, strip_title("data"), strip_body, flush=False)
        self.strip_card.enable_export()
        self.strip_card.export_requested.connect(self.export_requested.emit)
        self.strip_card.hide()
        lay.addWidget(self.strip_card)
        grid_host = QWidget()
        grid_host.setStyleSheet("background:transparent;")
        self._grid = CardGrid(grid_host, min_width=rem(26), max_columns=3, spacing=rem(1.0))
        lay.addWidget(grid_host)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:transparent; }")
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.addWidget(page)
        host_lay.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll)
        self._cards: dict = {}
        self._results: dict = {}
        self._labels: dict = {}
        self._colours: dict = {}
        self._order: list = []

    # -- what the section hands it ------------------------------------------------------------------------------- #
    def begin(self, conditions: list, labels: dict, colours: dict) -> None:
        """A new HMM Run: one card a condition, each waiting for its fit."""
        for card in self._cards.values():
            discard(card)
        self._cards, self._results = {}, {}
        self._labels, self._colours = dict(labels), dict(colours)
        self._order = list(conditions)
        for key in conditions:
            card = AnglesCard(self.t, f"Jump angles · {self._labels.get(key, key)}")
            self._cards[key] = card
            self._grid.addWidget(card)
        self.status.setText(waiting_text(False))
        self.status.show()
        self._refresh()

    def chance_on(self) -> bool:
        """Whether the page reads against chance (the box)."""
        return bool(self.chance_box.checked)

    def _chance_toggled(self, on: bool) -> None:
        """The reading changed: every condition's numbers are made again in it (the cache keeps both)."""
        self._results = {}
        for card in self._cards.values():
            card.set_status("Computing…")
        self.status.setText(waiting_text(True))
        self.status.show()
        self._refresh()
        self.chance_changed.emit(bool(on))

    def set_waiting(self, condition: str, have_hmm: bool) -> None:
        card = self._cards.get(condition)
        if card is not None and condition not in self._results:
            card.set_status("Computing…" if have_hmm else "Waiting for the HMM fit…")
        if have_hmm:
            self.status.setText(waiting_text(True))

    def set_result(self, condition: str, result: dict) -> None:
        card = self._cards.get(condition)
        if card is None:
            return
        self._results[condition] = result
        card.set_result(result, self._colours.get(condition, _DATA))
        pending = [k for k in self._cards if k not in self._results]
        self.status.setVisible(bool(pending))
        self._refresh()

    def set_order(self, order) -> None:
        self._order = order_apply(list(self._cards), order)
        self._cards = {k: self._cards[k] for k in self._order}
        self._grid.reorder(list(self._cards.values()))
        self._refresh()

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        for key, card in self._cards.items():
            if key in self._results and "error" not in self._results[key]:
                card.body.set_colour(self._colours.get(key, _DATA))
        self._refresh()

    def results(self) -> dict:
        return dict(self._results)

    def cards(self) -> dict:
        return dict(self._cards)

    def export_payload(self) -> dict | None:
        """What the strip card's Export draws (`hmm_angles_view.export_payload`), or None before any result."""
        good = {k: r for k, r in self._results.items() if "error" not in r}
        if not good:
            return None
        return export_payload(good, [k for k in self._order if k in good], self._labels, self._colours)

    # -- drawing ------------------------------------------------------------------------------------------------- #
    def _refresh(self) -> None:
        good = {k: r for k, r in self._results.items() if "error" not in r}
        order = [k for k in self._order if k in good]
        if not good:
            self.strip_card.hide()
            return
        self.strip_card.set_title(strip_title(mode(good)))
        self.strip.set_payload(strip_payload(good, order, self._labels, self._colours))
        for w in self._table_parts:
            discard(w)
        self._table_parts = []
        for table in comparison_tables(good, order, self._labels):
            caption = fit_label(table["caption"], ty.ui("xs"), self.t.muted_foreground)
            caption.setWordWrap(True)
            grid = _Table(self.t)
            grid.set_table(table["columns"], table["rows"])
            self._tables_lay.addWidget(caption)
            self._tables_lay.addWidget(grid)
            self._table_parts += [caption, grid]
        self.strip_card.show()
