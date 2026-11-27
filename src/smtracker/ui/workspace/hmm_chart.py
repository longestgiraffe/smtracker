"""The HMM sidebar section: the diffusive-state HMM, one card per condition.

WHAT IT IS (asked 13 Oct 2026: "then the hmm tab"). Each card draws the fitted model as a
STATE DIAGRAM -- one node per state, its area the occupancy, labelled with the state's
apparent D, and an arrow each way between two states carrying the switching rate -- beside
the rates with their intervals (they lead), and under it one compact table of the states
(D, occupancy, the track-start pi and the stationary mix, the model-derived dwell), the BIC
line when the count was chosen, and only the flags that change how a number may be quoted.

THE PAGE (13 Oct 2026: "too much text and the 2x2 grid is not well supported ... check also
the scripts python for the manuscript"): ABOVE the cards, the manuscript's comparison
across conditions (`HMMComparison`, `hmm_view.comparison_payload`): slow-state occupancy,
the rates into and out of it and its apparent D, each replicate day an open circle. The
explanations every card repeated are said once, in the "How to read" hover card
(`hmm_view.READING_NOTES`).

NO CARD IS GIVEN A MINIMUM HEIGHT. An explicit `setMinimumHeight` REPLACES the layout's own
minimum in Qt; the first version set 26 rem on a card whose content needed more, and its
box layout drew the diagram over its sentences (measured on the jumps card the same way:
content 919 px held at 640, the charts placed 88 px apart at 240 px tall).

The numbers, words and rules come from `smtracker.hmm_view` (headless, picklable, run on the
process pool); this module reads only the plain lists and strings it returns and imports no
numpy, pandas or scipy (`tests/test_ui_thin_qt.py`). The Run lives in `window.py`, this
workspace's convention; `run_requested` asks for it. Each card's Export button
(`_PanelCard.enable_export`) asks the window through `export_requested(key)`; the figure is
`figure_export`'s "hmm_condition", drawn from `export_entries(key)` -- the card's own result.
The comparison card's Export (13 Oct 2026: "HMM across-conditions strip as an exportable
figure") asks through `comparison_export_requested`; the figure is "hmm_comparison", drawn
from `comparison_export()` -- the very payload the strip paints, nothing recomputed, with each
panel's typed y range as ``"axes"``: the strip's Axes button (30 Oct 2026, `smtracker.axes_view`, the
shared popover) sets each panel's y on its own; the section keeps them through a new Run until Auto
(`comparison_axes`, `set_comparison_axes`). "Export all diagrams…" over the cards (5 Nov 2026: "we need to be able
to export multiple conditions as one image for hmm") asks through `overview_export_requested`; the figure is
"hmm_overview", every card's diagram side by side, drawn from `overview_entries()` -- the conditions the comparison's
eyes show, in the section's order.
"""
from __future__ import annotations

from ...condition_order import apply as order_apply
from ...axes_view import apply_range, axis_rows, check
from ... import chart_text
from ..widgets.axes_popover import AxesPopover

import math

from qtpy.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from ...hmm_view import (
    DEFAULTS,
    SIGMA_LABELS,
    SIGMA_MODES,
    STATE_CHOICES,
    STATE_LABELS,
    UNCERTAINTY_LABELS,
    UNCERTAINTY_MODES,
    READING_NOTES,
    PAIR_LINES_WORDS,
    card_caption,
    card_tables,
    comparison_payload,
    comparison_tip,
    condition_table,
    CONDITION_TABLE_NOTE,
    STATES,
    config_problem,
    hmm_config,
    pair_lines,
    setting_help,
)
from ... import significance as sig
from ...map_fit_source import MAP_THIS_FIT, map_this_fit_tip
from ..theme import typography as ty
from ..widgets.cardgrid import CardGrid
from ..widgets.controls import Button, NumberField, Segmented
from ..widgets.primitives import Checkbox, discard, fit_label, label
from ..widgets.settings import InfoLink
from ...hmm_run import (
    BEST_MAX,
    HELP as RUN_HELP,
    STATE_MODEL_LABELS,
    STATE_MODELS,
    UNCERTAINTY as RUN_UNCERTAINTY,
    UNCERTAINTY_LABELS as RUN_UNCERTAINTY_LABELS,
    HMMRunError,
    run_config,
    states_of,
)
from ...workspace_view import (
    GROUP_PREFIX,
    ROW_SHARED_D,
    SHARED_D_NAME,
    SHARED_D_ORDER,
    SHARED_D_PILLS,
    factor_order,
    group_label,
    is_set_wide,
    set_wide_note,
)
from .condition_key import ConditionKey
from .jumps_chart import _Table
from .condition_list import ConditionList as _ConditionList
from .qc_chart import _DATA, _FLAG, _Caption, _PanelCard
from .run_dialog import FitRow, FitsPage, RunDialog
from .section_placeholder import SectionPlaceholder
from .toolbar import rem, text_px

__all__ = ["HMMCard", "HMMComparison", "HMMConfigDialog", "HMMDiagram", "HMMSection"]


class HMMDiagram(QWidget):
    """The fitted model as a state diagram: nodes sized by occupancy (AREA, so the radius
    goes as its square root), arrows carrying the rates. Two states sit side by side with
    one arrow above (slow to fast) and one below; three sit on a triangle."""

    H = 210

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._payload: dict | None = None
        self._colour = _DATA
        #: the card's own diagram carries the rate alone; the rates beside it carry the
        #: interval (the export draws both on the arrows, `figure_export`)
        self.short_labels = False
        #: The Test page's cards: the chosen state's ring (an index, or None), and the page's
        #: ONE rate ceiling the arrows are weighed against (None: this diagram's own top).
        self.highlight = None
        self.scale = None
        self.setFixedHeight(self.H)

    def set_payload(self, payload: dict | None, colour: str) -> None:
        self._payload, self._colour = payload, colour
        # three states on a triangle need more room than two side by side
        n = len(payload["nodes"]) if payload else 2
        self.setFixedHeight(self.H if n <= 2 else self.H + 50)
        self.update()

    def _centres(self, n: int, box: QRectF) -> list:
        if n == 1:
            return [box.center()]
        if n == 2:
            y = box.center().y()
            return [QPointF(box.left() + box.width() * 0.27, y),
                    QPointF(box.left() + box.width() * 0.73, y)]
        return [QPointF(box.left() + box.width() * 0.22, box.bottom() - box.height() * 0.28),
                QPointF(box.center().x(), box.top() + box.height() * 0.26),
                QPointF(box.left() + box.width() * 0.78, box.bottom() - box.height() * 0.28)]

    def paintEvent(self, _e) -> None:
        if not self._payload:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        nodes, arrows = self._payload["nodes"], self._payload["arrows"]
        n = len(nodes)
        box = QRectF(8.0, 8.0, self.width() - 16.0, self.height() - 16.0)
        centres = self._centres(n, box)
        r_max = min(box.height() * (0.34 if n <= 2 else 0.2), box.width() * 0.16)
        radii = [max(12.0, r_max * math.sqrt(max(node["occupancy"], 0.0))) for node in nodes]
        ink = QColor(self._colour)
        muted = QColor(self.t.muted_foreground)
        font = ty.ui("xs", px=text_px(0.72))
        fm = QFontMetricsF(font)
        p.setFont(font)

        # the arrows, under the nodes. On the card with three states the rates table beside
        # the diagram carries every rate: six labels inside a triangle collided, so there the
        # arrows carry their rate as WEIGHT instead (the export labels them, `figure_export`).
        weighted = self.short_labels and n > 2
        top_rate = self.scale or max((a["rate"] for a in arrows), default=1.0) or 1.0
        for arrow in arrows:
            a, b = centres[arrow["from"]], centres[arrow["to"]]
            ra, rb = radii[arrow["from"]], radii[arrow["to"]]
            dx, dy = b.x() - a.x(), b.y() - a.y()
            dist = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dist, dy / dist
            nx, ny = -uy, ux                       # the side this direction runs on
            bend = 18.0 if n == 2 else 9.0
            start = QPointF(a.x() + ux * ra + nx * bend * 0.5, a.y() + uy * ra + ny * bend * 0.5)
            end = QPointF(b.x() - ux * rb + nx * bend * 0.5, b.y() - uy * rb + ny * bend * 0.5)
            mid = QPointF((start.x() + end.x()) / 2 + nx * bend,
                          (start.y() + end.y()) / 2 + ny * bend)
            path = QPainterPath(start)
            path.quadTo(mid, end)
            pen = QPen(muted)
            pen.setWidthF(1.4 if not weighted
                          else 1.0 + 3.0 * min(arrow["rate"], top_rate) / top_rate)
            if self._payload.get("not_identifiable"):
                pen.setDashPattern([4, 3])
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            # the head, along the curve's last tangent
            tx, ty_ = end.x() - mid.x(), end.y() - mid.y()
            tl = math.hypot(tx, ty_) or 1.0
            tx, ty_ = tx / tl, ty_ / tl
            head = QPainterPath(end)
            head.lineTo(end.x() - 8 * tx + 4 * ty_, end.y() - 8 * ty_ - 4 * tx)
            head.lineTo(end.x() - 8 * tx - 4 * ty_, end.y() - 8 * ty_ + 4 * tx)
            head.closeSubpath()
            p.setPen(Qt.NoPen)
            p.setBrush(muted)
            p.drawPath(head)
            if weighted:
                continue
            text = (f"{arrow['rate']:.3g} /s" if self.short_labels else arrow["text"])
            w = fm.horizontalAdvance(text) + 6
            spot = QPointF(mid.x() + nx * 10.0, mid.y() + ny * 10.0)
            plate = QRectF(spot.x() - w / 2, spot.y() - 8, w, 16)
            # a backing in the card's colour, so a label stays legible over another arrow
            back = QColor(self.t.card)
            back.setAlpha(225)
            p.setPen(Qt.NoPen)
            p.setBrush(back)
            p.drawRoundedRect(plate, 3.0, 3.0)
            p.setPen(QColor(self.t.foreground))
            p.drawText(plate, Qt.AlignCenter, text)

        # the nodes
        for index, (node, c, r) in enumerate(zip(nodes, centres, radii)):
            fill = QColor(ink)
            fill.setAlpha(70)
            pen = QPen(ink)
            pen.setWidthF(1.6)
            p.setPen(pen)
            p.setBrush(fill)
            p.drawEllipse(c, r, r)
            if index == self.highlight:
                ring = QPen(QColor(self.t.foreground))
                ring.setWidthF(2.0)
                p.setPen(ring)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, r + 4, r + 4)
            lines = node["label"].split("\n")
            h = fm.height()
            widest = max(fm.horizontalAdvance(text) for text in lines)
            p.setPen(QColor(self.t.foreground))
            if n == 3 and index == 1 and not (widest <= 1.75 * r and h * len(lines) <= 1.7 * r):
                # the triangle's apex, too small for its words: BESIDE it, clear of the
                # arrows that run down from it (under it, they crossed them)
                top = c.y() - h * len(lines) / 2
                for k, text in enumerate(lines):
                    p.drawText(QRectF(c.x() + r + 5.0, top + k * h, widest + 4, h),
                               Qt.AlignLeft | Qt.AlignVCenter, text)
                continue
            if widest <= 1.75 * r and h * len(lines) <= 1.7 * r:
                top = c.y() - h * len(lines) / 2          # inside the node
            else:
                top = c.y() + r + 2.0                       # too small: under it
            for k, text in enumerate(lines):
                w = fm.horizontalAdvance(text) + 2
                p.drawText(QRectF(c.x() - w / 2, top + k * h, w, h), Qt.AlignCenter, text)
        p.end()


class HMMComparison(QWidget):
    """The manuscript's comparison across conditions (`scripts/figure_hmm_four_panel.py`):
    four panels -- slow-state occupancy, the rate into it, the rate out of it, its apparent
    D -- each condition a column: replicate days as open circles (spread sideways, no order
    implied), their mean filled with a +-1 SD bar; or the pooled fit with a 95 % movie or
    cell interval; or the pooled fit alone. Axes start at zero. Payload:
    `hmm_view.comparison_payload`."""

    H = 236
    _TOP = 22          # the panel's title
    _BOTTOM = 34       # the condition names
    _LEFT = 42         # the y tick labels
    _NAME_LINE = 15    # each further line of a wrapped condition name
    _GAP = 22.0        # between panels

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._payload: dict | None = None
        self.setFixedHeight(self.H)
        # TOOLTIPS (the UI designer, 21 Oct 2026): hovering a condition's column in a panel shows shadcn's
        # chart tooltip (`dynamics_chart._ChartTooltip`, the app's one) with what that column
        # draws (`hmm_view.comparison_tip`), and the muted cursor band behind the column. The
        # hit areas are RECORDED by the paint (`_hits`), never recomputed, as `_HoverChart` does.
        from .dynamics_chart import _ChartTooltip

        self.setMouseTracking(True)
        self._hits: list = []
        self._hover: tuple | None = None
        self._tip = _ChartTooltip(t, self)
        #: Typed ranges per panel (the card's Axes, 30 Oct 2026: `axes_view`), ``{panel key: (from, to)}``, and the
        #: ranges as last drawn.
        self.axes: dict = {}
        self.drawn: dict = {}
        self.drawn_names: dict = {}
        #: The PAIR LINES (5 Nov 2026, off by default): a thin line through the open circles of replicates declared
        #: as pairs on the Groups board (`hmm_view.pair_lines`); ``drawn_lines`` what each panel drew (tests).
        self.pair_lines = False
        self.drawn_lines: dict = {}
        #: THE BRACKETS (5 Nov 2026, the Significance tool's, off by default): ``[{"a", "b", "population" (the
        #: panel's key), "text"}]`` (`significance.bracket_marks`); ``drawn_brackets`` each panel's as drawn (tests).
        self._marks: list = []
        self.drawn_brackets: dict = {}

    #: The row the group headers take under the names when the conditions are grouped (folder designs).
    _GROUP_ROW = 20

    def set_payload(self, payload: dict | None) -> None:
        self._set_hover(None)
        self._payload = payload
        self._fit_height()
        self.update()

    def set_pair_lines(self, on: bool) -> None:
        self.pair_lines = bool(on)
        self.update()

    #: A bracket's tier on screen (px): its line, its legs and its text over it.
    _TIER = 16

    def set_brackets(self, marks) -> None:
        self._marks = [dict(m) for m in marks or ()]
        self.update()

    def brackets(self) -> list:
        return [dict(m) for m in self._marks]

    #: The narrowest a condition's column may be before the panels stack two across (4 Nov 2026, 1366 px: five
    #: conditions in four panels a row left 31 px a name, every one elided to "mN..."). About the width of "simulated" at 12 px.
    MIN_NAME_SLOT = 56.0

    def _grouped(self) -> bool:
        return bool(((self._payload or {}).get("layout") or {}).get("groups"))

    def _columns(self) -> int:
        """How many condition columns a panel draws (grouped designs leave gaps between groups)."""
        payload = self._payload or {}
        panels = payload.get("panels") or []
        rows = panels[0]["rows"] if panels else []
        layout = payload.get("layout") or {}
        if layout.get("groups") and len(layout.get("x") or ()) == len(rows) and rows:
            xs_ = [float(x) for x in layout["x"]]
            return int(round(xs_[-1] - xs_[0] + 1.0))
        return max(1, len(rows))

    def _across(self) -> int:
        """Panels per line: all of them, else two where four would give a condition less than `MIN_NAME_SLOT`."""
        count = len((self._payload or {}).get("panels") or ())
        if count <= 2:
            return max(1, count)
        slot = ((self.width() - self._GAP * (count - 1)) / count - self._LEFT - 4) / self._columns()
        return count if slot >= self.MIN_NAME_SLOT else (count + 1) // 2

    def _lines(self) -> int:
        count = len((self._payload or {}).get("panels") or ())
        return -(-count // self._across()) if count else 1

    def _fit_height(self) -> None:
        # ONLY WHERE THE HEIGHT IS FIXED (the section's card; the retired Test page freed it to fill its page)
        if self.maximumHeight() < 16777215:
            self.setFixedHeight(self._lines() * (self.H + (self._GROUP_ROW if self._grouped() else 0)))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit_height()                  # the stacking depends on the width: idempotent, it moves only the height

    # -- hover ----------------------------------------------------------------------------- #
    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        hit = next((h for h in self._hits if h[0].contains(pos)), None)
        self._set_hover(hit)
        super().mouseMoveEvent(e)

    def leaveEvent(self, e) -> None:
        self._set_hover(None)
        super().leaveEvent(e)

    def hover_key(self) -> tuple | None:
        return self._hover

    def _set_hover(self, hit) -> None:
        key = None if hit is None else hit[1]
        same = key == self._hover
        self._hover = key
        try:
            if hit is None:
                self._tip.hide()
            else:
                _rect, _key, anchor, tip = hit
                self._tip.set_content(heading=tip["heading"], rows=tip["rows"],
                                      colour=tip["colour"] or _DATA)
                window = self.window()
                if self._tip.parentWidget() is not window:
                    self._tip.setParent(window)
                self._place(anchor)
                self._tip.raise_()
                self._tip.show()
        except RuntimeError:
            # the tooltip lives on the WINDOW, which tears it down first (`_HoverChart`'s note)
            return
        if not same:
            self.update()

    def _place(self, anchor: QPointF) -> None:
        """Centred above ``anchor``, clamped to the window, below it when there is no room --
        `_HoverChart._place`'s rule."""
        window = self.window()
        w, h = self._tip.width(), self._tip.height()
        pos = self.mapTo(window, QPoint(round(anchor.x()), round(anchor.y())))
        gap = 6
        x = max(4, min(pos.x() - w // 2, max(4, window.width() - w - 4)))
        y = pos.y() - gap - h
        if y < 4:
            y = pos.y() + gap
        self._tip.move(x, y)

    @staticmethod
    def _ticks(top: float) -> list:
        if top <= 0:
            return [0.0]
        raw = top / 4.0
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
        return [i * step for i in range(int(top / step + 1e-9) + 1)]

    @classmethod
    def _ticks_between(cls, lo: float, hi: float) -> list:
        """Round ticks inside a typed range ``[lo, hi]`` (about four steps), its ends when none fall inside."""
        raw = (hi - lo) / 4.0
        if raw <= 0:
            return [lo]
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
        first = math.ceil(lo / step - 1e-9)
        ticks = [k * step for k in range(first, int(math.floor(hi / step + 1e-9)) + 1)]
        return ticks or [lo, hi]

    def paintEvent(self, _e) -> None:
        if not self._payload:
            return
        panels = self._payload["panels"]
        self._hits = []
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        gap = self._GAP
        across, lines = self._across(), self._lines()
        w = (self.width() - gap * (across - 1)) / across
        line_h = self.height() / lines
        for i, panel in enumerate(panels):
            row, col = divmod(i, across)
            self._panel(p, QRectF(col * (w + gap), row * line_h, w, float(line_h)), panel)
        p.end()

    def _panel(self, p, box: QRectF, panel: dict) -> None:
        rows = panel["rows"]
        title_font = ty.ui("xs", "medium")
        p.setFont(title_font)
        p.setPen(QColor(self.t.foreground))
        p.drawText(QRectF(box.left(), box.top(), box.width(), self._TOP - 4),
                   Qt.AlignLeft | Qt.AlignVCenter, f"{panel['title']} ({panel['unit']})")
        layout = (self._payload or {}).get("layout") or {}
        grouped = bool(layout.get("groups")) and len(layout.get("x") or ()) == len(rows)
        # THE CONDITION NAMES, WRAPPED onto up to three lines at their column's width (`chart_text.wrap_text`), the
        # plot giving up the room the extra lines take: elided to one line, "mNG, measured", "mNG, D2|D5 simulated"
        # and "mNG, D2|D5 truth" all read "mNG, ..." at 1366 px (found by the overnight tester, 4 Nov 2026)
        name_font = QFontMetricsF(ty.ui("xs"))
        plot_w = box.width() - self._LEFT - 4
        if grouped:
            xs_ = [float(x) for x in layout["x"]]
            name_slot = plot_w / (xs_[-1] - xs_[0] + 1.0)
            raw_names = [str(v) for v in layout["labels"]]
        else:
            name_slot = plot_w / max(1, len(rows))
            raw_names = [str(r["name"]) for r in rows]
        # Broken at spaces, a word wider than the column ended with "…" (`chart_text.name_lines`); the column's hover
        # carries the whole name (`hmm_view.comparison_tip`'s heading).
        name_lines = [chart_text.name_lines(n, name_font.horizontalAdvance, max(8.0, name_slot - 4), max_lines=3)
                      for n in raw_names]
        extra = (max((len(lines) for lines in name_lines), default=1) - 1) * self._NAME_LINE
        self.drawn_names[panel["key"]] = (name_lines, name_slot - 4)        # what the names were drawn as (tests)
        # THE BRACKETS' BAND over the plot: their tiers by the one rule the export keeps (`significance.bracket_tiers`)
        mine = [dict(m) for m in self._marks if m.get("population") == panel["key"]]
        units = [float(x) for x in layout["x"]] if grouped else [float(k) for k in range(len(rows))]
        tiers = []
        if mine:
            bracket_fm = QFontMetricsF(ty.ui("xs"))
            for m in mine:
                m["x0"], m["x1"] = units[m["a"]], units[m["b"]]
            tiers = sig.bracket_tiers(mine, lambda m: (bracket_fm.horizontalAdvance(m["text"]) / 2 + 4)
                                      / max(1.0, name_slot))
        band = (max(tiers) + 1) * self._TIER if tiers else 0
        plot = QRectF(box.left() + self._LEFT, box.top() + self._TOP + 6 + band,
                      plot_w,
                      box.height() - self._TOP - 6 - band - self._BOTTOM - extra
                      - (self._GROUP_ROW if grouped else 0))
        tops = []
        for r in rows:
            tops += r["days"]
            for v in (r["value"],):
                if v is not None:
                    tops.append(v)
            if r["mean"] is not None and r["sd"] is not None:
                tops.append(r["mean"] + r["sd"])
            if r["band"] is not None:
                tops.append(r["band"][1])
        ticks = self._ticks(max(tops) * 1.12 if tops else 1.0)
        hi = ticks[-1] if ticks[-1] > 0 else 1.0
        if tops and max(tops) > hi:
            hi = max(tops) * 1.05
        lo = 0.0
        typed = (self.axes or {}).get(panel["key"])
        if typed is not None:
            # a typed range (the card's Axes): this panel's y exactly where it was put, its round ticks inside it
            lo, hi = apply_range(0.0, hi, typed)
            ticks = self._ticks_between(lo, hi)
        self.drawn[panel["key"]] = (lo, hi)

        def py(v):                                     # below a typed floor: under the plot, clipped
            return plot.bottom() - ((max(0.0, v) - lo) / (hi - lo)) * plot.height()

        # ticks and condition names at the scale's floor, 12 px (shadcn's charts: `text-xs`);
        # they were 10 (font audit, 21 Oct 2026). The margins (`_LEFT`, `_BOTTOM`) hold them
        small = ty.ui("xs")
        fm = QFontMetricsF(small)
        p.setFont(small)
        for v in ticks:
            y = py(v)
            p.setPen(QPen(QColor(self.t.border), 1.0))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(box.left(), y - 8, self._LEFT - 6, 16),
                       Qt.AlignRight | Qt.AlignVCenter, f"{v:g}")
        if not rows:
            return
        # GROUPED BY THE FACTORS (`factor_layout.columns`): the columns stand at the layout's positions, a gap
        # between groups, each named by its inner values and the group by its outer value under them
        if grouped:
            xs = [float(x) for x in layout["x"]]
            slot = plot.width() / (xs[-1] - xs[0] + 1.0)
            centres = [plot.left() + slot * (x - xs[0] + 0.5) for x in xs]
            for group in layout["groups"]:
                x0 = plot.left() + slot * (group["x0"] - xs[0] + 0.08)
                x1 = plot.left() + slot * (group["x1"] - xs[0] + 0.92)
                line = plot.bottom() + 4 + 18 + extra
                p.setPen(QPen(QColor(self.t.border), 1.0))
                p.drawLine(QPointF(x0, line), QPointF(x1, line))
                p.setPen(QColor(self.t.foreground))
                p.drawText(QRectF(x0, line + 2, x1 - x0, 16), Qt.AlignHCenter | Qt.AlignTop,
                           fm.elidedText(str(group["label"]), Qt.ElideRight, int(max(8.0, x1 - x0))))
        else:
            slot = plot.width() / len(rows)
            centres = [plot.left() + slot * (k + 0.5) for k in range(len(rows))]
        # where each replicate's open circle stands: spread sideways in no order, the mean beside them
        spread = min(slot * 0.32, 22.0)
        circles = {}
        for k, r in enumerate(rows):
            n = len(r["days"])
            for j, v in enumerate(r["days"]):
                circles[(k, j)] = QPointF(centres[k] + (-spread / 2 + spread * j / (n - 1) if n > 1 else 0.0), py(v))
        # THE PAIR LINES, behind the circles (off unless ticked; `hmm_view.pair_lines`)
        lines = pair_lines(rows) if self.pair_lines else []
        self.drawn_lines[panel["key"]] = [(tag, [(k, j) for k, j in points]) for tag, points in lines]
        if lines:
            p.save()
            p.setClipRect(QRectF(plot.left() - 8, plot.top() - 8, plot.width() + 16, plot.height() + 8))
            ink = QColor(self.t.muted_foreground)
            ink.setAlphaF(0.55)
            p.setPen(QPen(ink, 1.0))
            for _tag, points in lines:
                for (k0, j0), (k1, j1) in zip(points, points[1:]):
                    p.drawLine(circles[(k0, j0)], circles[(k1, j1)])
            p.restore()
        for k, r in enumerate(rows):
            cx = centres[k]
            ink = QColor(r["colour"] or _DATA)
            # the column's hit area (the plot and its name), its cursor band, and where the
            # tooltip points: above the highest thing the column draws
            column = QRectF(cx - slot / 2, plot.top(), slot, box.bottom() - plot.top())
            key = (panel["key"], r["key"])
            if key == self._hover:
                from ..widgets.primitives import mix

                p.setPen(Qt.NoPen)
                p.setBrush(QColor(mix(self.t.muted, self.t.background, 0.45)))
                p.drawRect(QRectF(column.left() + 2, plot.top(), column.width() - 4, plot.height()))
            marks = list(r["days"]) + [v for v in (r["value"],) if v is not None]
            if r["mean"] is not None and r["sd"] is not None:
                marks.append(r["mean"] + r["sd"])
            if r["band"] is not None:
                marks.append(r["band"][1])
            anchor = QPointF(cx, py(max(marks)) - 6 if marks else plot.bottom() - 18)
            self._hits.append((column, key, anchor, comparison_tip(panel, r)))
            p.setPen(QColor(self.t.muted_foreground))
            for j, line_text in enumerate(name_lines[k]):
                p.drawText(QRectF(cx - slot / 2, plot.bottom() + 4 + j * self._NAME_LINE, slot, 16),
                           Qt.AlignHCenter | Qt.AlignTop, line_text)
            p.save()
            p.setClipRect(QRectF(plot.left() - 8, plot.top() - 8, plot.width() + 16, plot.height() + 8))
            if r["days"]:
                for j, _v in enumerate(r["days"]):
                    p.setPen(QPen(ink, 1.4))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(circles[(k, j)], 3.6, 3.6)
                self._bar(p, cx + spread * 0.9, r["mean"] - r["sd"], r["mean"] + r["sd"], py,
                          ink)
                p.setPen(Qt.NoPen)
                p.setBrush(ink)
                p.drawEllipse(QPointF(cx + spread * 0.9, py(r["mean"])), 4.4, 4.4)
            elif r["value"] is not None:
                if r["band"] is not None:
                    self._bar(p, cx, r["band"][0], r["band"][1], py, ink)
                p.setPen(Qt.NoPen)
                p.setBrush(ink)
                p.drawEllipse(QPointF(cx, py(r["value"])), 4.4, 4.4)
            else:
                p.setPen(QColor(self.t.muted_foreground))
                p.drawText(QRectF(cx - slot / 2, plot.bottom() - 18, slot, 16),
                           Qt.AlignCenter, "–")
            p.restore()
        self.drawn_brackets[panel["key"]] = [(m["a"], m["b"], level, m["text"]) for m, level in zip(mine, tiers)]
        if mine:
            p.setPen(QPen(QColor(self.t.foreground), 1.0))
            p.setFont(ty.ui("xs"))
            for m, level in zip(mine, tiers):
                x0, x1 = centres[m["a"]], centres[m["b"]]
                y = plot.top() - 3 - level * self._TIER
                p.drawLine(QPointF(x0, y + 4), QPointF(x0, y))
                p.drawLine(QPointF(x0, y), QPointF(x1, y))
                p.drawLine(QPointF(x1, y), QPointF(x1, y + 4))
                p.drawText(QRectF((x0 + x1) / 2 - 80, y - 14, 160, 13), Qt.AlignHCenter | Qt.AlignBottom, m["text"])

    @staticmethod
    def _bar(p, x: float, lo: float, hi: float, py, ink: QColor) -> None:
        pen = QPen(ink, 1.5)
        p.setPen(pen)
        p.drawLine(QPointF(x, py(lo)), QPointF(x, py(hi)))
        for v in (lo, hi):
            p.drawLine(QPointF(x - 4, py(v)), QPointF(x + 4, py(v)))


class _HMMBody(QWidget):
    """A card's content: a message, or the diagram beside the rates, then the states table,
    the BIC line, the spread line and the flags (`hmm_view.card_tables`)."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._result = None
        self.setStyleSheet("background:transparent;")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        self._lay.setSpacing(rem(0.5))
        self.message = _Caption(t, size="sm")
        self.message.setText("computing…")
        self._lay.addWidget(self.message)
        self._content = QWidget()
        self._content.setStyleSheet("background:transparent;")
        content = QVBoxLayout(self._content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(rem(0.5))
        top = QHBoxLayout()
        top.setSpacing(rem(1.0))
        self.diagram = HMMDiagram(t)
        self.diagram.short_labels = True
        self.diagram.setFixedWidth(rem(17))
        top.addWidget(self.diagram, 0, Qt.AlignTop)
        right = QVBoxLayout()
        right.setSpacing(rem(0.3))
        right.addWidget(label("Switching rates (per second)", ty.ui("xs", "medium"), t.foreground))
        self.rates = _Table(t)
        right.addWidget(self.rates)
        right.addStretch(1)
        top.addLayout(right, 1)
        content.addLayout(top)
        self.states = _Table(t)
        content.addWidget(self.states)
        self.bic = _Caption(t, size="xs")
        content.addWidget(self.bic)
        self.spread = _Caption(t, size="xs")
        content.addWidget(self.spread)
        self._flags: list[QWidget] = []
        self._content_lay = content
        self._lay.addWidget(self._content)
        self._lay.addStretch(1)
        self._content.hide()

    def _clear(self) -> None:
        for w in self._flags:
            discard(w)
        self._flags = []

    def set_error(self, text: str) -> None:
        self._result = None
        self._clear()
        self._content.hide()
        self.message.setText(text)
        self.message.show()

    def set_result(self, result: dict, colour: str) -> None:
        self._clear()
        self.message.hide()
        self._result = result
        tables = card_tables(result)
        self.diagram.set_payload(result["diagram"], colour)
        self.rates.set_table(tables["rate_columns"], tables["rates"])
        self.states.set_table(tables["states"]["columns"], tables["states"]["rows"])
        self.bic.setText(tables["bic"] or "")
        self.bic.setVisible(bool(tables["bic"]))
        self.spread.setText(tables["spread"] or "")
        self.spread.setVisible(bool(tables["spread"]))
        for text in tables["flags"]:
            w = _Caption(self.t, size="xs")
            w.setStyleSheet(f"color:{_FLAG}; background:transparent;")
            w.setText(text)
            self._content_lay.addWidget(w)
            self._flags.append(w)
        self._content.show()

    def set_colour(self, colour: str) -> None:
        if self._result is not None:
            self.diagram.set_payload(self._result["diagram"], colour)


class HMMCard(_PanelCard):
    """One condition's HMM."""

    def __init__(self, t, title: str, parent=None):
        self.hmm_body = _HMMBody(t)
        super().__init__(t, title, self.hmm_body, parent=parent, flush=False)

    def set_result(self, result: dict, colour: str) -> None:
        self.set_caption(card_caption(result))
        self.hmm_body.set_result(result, colour)

    def set_colour(self, colour: str) -> None:
        self.hmm_body.set_colour(colour)

    def set_error(self, text: str) -> None:
        self.set_caption("")
        self.hmm_body.set_error(text)

    def result(self) -> dict | None:
        """The fit this card draws, or None (computing, or an error)."""
        return self.hmm_body._result


def _mirror(tray, index: int) -> None:
    """``tray`` lit at ``index`` without saying so (the Fits page's copy of the General page's pills)."""
    if tray.current() != index:
        tray.blockSignals(True)
        tray.select(index)
        tray.blockSignals(False)


class _StateRow(FitRow):
    """One condition's line on the HMM Run dialog's Fits page (`run_dialog.FitRow`): included or not, its state count,
    its own Force same D across."""

    states_changed = Signal(str, int)

    def __init__(self, t, key: str, name: str, on: bool, parent=None):
        states = Segmented(t, list(STATE_MODEL_LABELS), current=0, size="sm")
        states.setToolTip("This condition's state count")
        super().__init__(t, key, name, states, include=on, parent=parent)
        self.states = states
        self.states.changed.connect(lambda i: self.states_changed.emit(self.key, i))


class HMMConfigDialog(RunDialog):
    """The Run's settings, SQD's in SQD's words (3 Nov 2026, asked: "the HMM run window is a bit outdated compared to
    how advanced the sqd run window is"; `hmm_run`), on SQD's two pages (`run_dialog.RunDialog`, 4 Nov 2026) --
    General: Force same D across, Uncertainty (Chunks with its count), σ correction and Best's range; Fits: every
    condition on one line -- included or not, its state count, its own Force same D -- under the Every condition line.
    Every rule is headless (`hmm_run.run_config`); a refusal is said above Run and nothing runs. Transactional like
    every Run dialog here: Cancel and the corner X discard the draft. Seeded from the last run."""

    def __init__(self, t, conditions: list, *, seed: dict | None = None, factors: dict | None = None,
                 content=None, parent=None, factor_names=(), sigma_note: str = ""):
        super().__init__(t, "Configure HMM", "Nothing runs until you press Run.", content=content, parent=parent)
        self._keys = [str(k) for k, _name in conditions]
        self._names = {str(k): str(name) for k, name in conditions}
        self._factors = dict(factors or {})
        try:
            seed = run_config(seed or {}, conditions=self._keys, factors=self._factors)
        except HMMRunError:
            seed = run_config({}, conditions=self._keys, factors=self._factors)
        self.helpers: dict = {}
        self._hint = self.hint
        # FORCE SAME D ACROSS: SQD's pills in SQD's order, a Conditions pill per folder-design factor in the design's
        # order (`workspace_view.factor_order`)
        names = factor_order(self._factors, factor_names)
        self._shared_values = []
        for value in SHARED_D_ORDER:
            self._shared_values += ([GROUP_PREFIX + f for f in names] if value == "group" else [value])
        pill = {v: (group_label(v[len(GROUP_PREFIX):], len(names)) if v.startswith(GROUP_PREFIX)
                    else SHARED_D_PILLS[v]) for v in self._shared_values}
        self._pills = pill
        start = seed["shared_d"] if seed["shared_d"] in self._shared_values else "condition"
        self._shared = Segmented(t, [pill[v] for v in self._shared_values],
                                 current=self._shared_values.index(start), size="sm")
        self._shared.changed.connect(lambda _i: self._on_shared())
        self._uncertainty = Segmented(t, list(RUN_UNCERTAINTY_LABELS),
                                      current=RUN_UNCERTAINTY.index(seed["uncertainty"]), size="sm")
        self._uncertainty.changed.connect(lambda _i: self._sync())
        self._chunks = NumberField(t, 2, 20, int(seed["chunks"]), width=rem(7))
        self._chunks_label = label("chunks", ty.ui("xs"), t.muted_foreground)
        self._sigma = Segmented(t, list(SIGMA_LABELS), current=SIGMA_MODES.index(seed["sigma"]), size="sm")
        self._sigma.changed.connect(lambda _i: self._sync())
        self._best = Segmented(t, [f"{k} states" for k in BEST_MAX], current=BEST_MAX.index(seed["best_max"]),
                               size="sm")
        # THE GENERAL PAGE
        for name, control, key, extra in ((SHARED_D_NAME, self._shared, "shared_d", ()),
                                          ("Uncertainty", self._uncertainty, "uncertainty",
                                           (self._chunks, self._chunks_label)),
                                          ("σ correction", self._sigma, "sigma", ()),
                                          ("Best (BIC) up to", self._best, "best_max", ())):
            line, more = RUN_HELP[key]
            self.helpers[key] = self.add_setting(name, control, line, more, extra)
            if key == "sigma":
                # NO SIGMA RECORDED: the theoretical one Tracker's sigma would use, said before Run (the overnight
                # tester's question 2, 4 Nov 2026; `hmm_view.sigma_note`), shown while Tracker's sigma is picked
                self._sigma_note = fit_label(sigma_note, ty.ui("xs"), _FLAG)
                self._sigma_note.setContentsMargins(0, 4, 0, 0)
                self.general_lay.addWidget(self._sigma_note)
                self._sigma_note.setVisible(False)
        # THE FITS PAGE: the Every condition line, then a line per condition
        every = Segmented(t, list(STATE_MODEL_LABELS), current=-1, size="sm")
        line, more = RUN_HELP["states"]
        fits = FitsPage(t, help_line=line, help_more=more, choice_title="States", every_choice=every, include=True)
        self.helpers["states"] = fits.info
        self._every = every
        self._every.changed.connect(self._on_every_states)
        self._every_include = fits.every.include
        self._every_include.toggled.connect(self._on_every_include)
        self._every_label = fits.every.title
        fits.every.shared_changed.connect(self._on_every_shared)
        self._set_note = fits.note
        # FORCE SAME D ACROSS ON TOP OF THE FITS PAGE TOO, as SQD's (5 Nov 2026): the General page's pills, one value
        # in two places -- a click here selects the General page's pill, whose handler does the rest; `_sync` lights it
        self._fits_shared = Segmented(t, [pill[v] for v in self._shared_values], current=self._shared.current(),
                                      size="sm")
        self._fits_shared.changed.connect(self._shared.select)
        fits.add_set_wide(self._fits_shared)
        self._included = {k: k in seed["conditions"] for k in self._keys}
        self._states = {k: states_of(seed, k) for k in self._keys}
        self._own = {k: str(((seed.get("rows") or {}).get(k) or {}).get("shared_d") or "") for k in self._keys}
        self._rows: dict = {}
        for k in self._keys:
            row = _StateRow(t, k, self._names.get(k, k), self._included[k])
            row.included.connect(self._on_row_include)
            row.states_changed.connect(self._on_row_states)
            row.shared_changed.connect(self._on_row_shared)
            self._rows[k] = row
        fits.set_rows(self._rows.values())
        self.set_fits(fits)
        self._sync()

    # -- the rules, headless (`hmm_run`) ------------------------------------------------------------------------- #
    def _shared_value(self) -> str:
        return self._shared_values[self._shared.current()]

    def _on_shared(self) -> None:
        if is_set_wide(self._shared_value()):
            # one D across conditions: one count, the included rows' first
            first = next((self._states[k] for k in self._keys if self._included[k]), "2")
            self._states = {k: first for k in self._keys}
        self._sync()

    def _on_every_include(self, on: bool) -> None:
        self._included = {k: bool(on) for k in self._keys}
        self._sync()

    def _on_every_states(self, index: int) -> None:
        if index < 0:
            return
        value = STATE_MODELS[index]
        self._states = {k: (value if self._included[k] or is_set_wide(self._shared_value()) else self._states[k])
                        for k in self._keys}
        self._sync()

    def _on_row_include(self, key: str, on: bool) -> None:
        self._included[key] = bool(on)
        self._sync()

    def _on_row_states(self, key: str, index: int) -> None:
        self._states[key] = STATE_MODELS[index]
        self._sync()

    def _on_row_shared(self, key: str, index: int) -> None:
        self._own[key] = ROW_SHARED_D[index]
        self._sync()

    def _on_every_shared(self, _key: str, index: int) -> None:
        """The Every condition line's Force same D across: every row's own, as clicking each row would make it (SQD's
        `ComputePanel._set_every_shared_d`)."""
        self._own = {k: ROW_SHARED_D[index] for k in self._keys}
        self._sync()

    def _row_shared(self, k: str, shared: str) -> str:
        """A row's own Force same D across: its pick, else the General page's (Replicates under a set-wide one)."""
        return self._own.get(k) or (shared if shared in ROW_SHARED_D else ROW_SHARED_D[0])

    def _refresh(self) -> None:
        """(the name the shell tests call) every control back on the draft."""
        self._sync()

    def _sync(self) -> None:
        shared = self._shared_value()
        wide = is_set_wide(shared)
        _mirror(self._fits_shared, self._shared.current())
        lit = {self._states[k] for k in self._keys if self._included[k]}
        self._every.blockSignals(True)
        self._every.select(STATE_MODELS.index(next(iter(lit))) if len(lit) == 1 else -1)
        self._every.blockSignals(False)
        self._every_include.blockSignals(True)
        self._every_include.set_checked(all(self._included.values()), animate=False)
        self._every_include.blockSignals(False)
        for k, row in self._rows.items():
            row.include.blockSignals(True)
            row.include.set_checked(self._included[k], animate=False)
            row.include.blockSignals(False)
            row.states.blockSignals(True)
            row.states.select(STATE_MODELS.index(self._states[k]))
            row.states.blockSignals(False)
            row.states.setEnabled(not wide and self._included[k])
            # under a set-wide value the row SHOWS it (SQD's Fits page), never a lit Replicates of its own
            row.shared.set_wide(self._pills[shared] if wide else None)
            if not wide:
                row.shared.set_value(self._row_shared(k, shared))
            row.shared.setEnabled(not wide and self._included[k])
        # the Every condition line's: the set-wide value, else the rows' common one (none while they differ)
        every = self.fits.every.shared
        every.set_wide(self._pills[shared] if wide else None)
        if not wide:
            found = {self._row_shared(k, shared) for k in self._keys}
            every.set_value(found.pop() if len(found) == 1 else None)
        every.setEnabled(not wide)
        note = set_wide_note(shared)
        self.fits.set_note(note.replace("models", "state count") if note else "")
        chunk = RUN_UNCERTAINTY[self._uncertainty.current()] == "chunk"
        self._chunks.setVisible(chunk)
        self._chunks_label.setVisible(chunk)
        self._sigma_note.setVisible(SIGMA_MODES[self._sigma.current()] == "tracker" and bool(self._sigma_note.text()))
        try:
            self.config()
            self._hint.setText("")
            self._run_btn.setEnabled(True)
        except HMMRunError as exc:
            self._hint.setText(str(exc))
            self._run_btn.setEnabled(False)

    def config(self) -> dict:
        """The draft as `hmm_run.run_config` checks it. RAISES `HMMRunError` on a refusal."""
        shared = self._shared_value()
        wide = is_set_wide(shared)
        chosen = [k for k in self._keys if self._included[k]]
        counts = [self._states[k] for k in chosen]
        every = max(set(counts), key=counts.count) if counts else "2"
        rows = {}
        for k in chosen:
            row = {}
            if self._states[k] != every:
                row["states"] = self._states[k]
            own = self._own.get(k)
            if not wide and own and own != shared:
                row["shared_d"] = own
            if row:
                rows[k] = row
        return run_config({"conditions": chosen, "states": every, "shared_d": shared,
                           "uncertainty": RUN_UNCERTAINTY[self._uncertainty.current()],
                           "sigma": SIGMA_MODES[self._sigma.current()], "best_max": BEST_MAX[self._best.current()],
                           "chunks": int(self._chunks.value()), "rows": rows},
                          conditions=self._keys, factors=self._factors)

    def _on_run(self) -> None:
        try:
            config = self.config()
        except HMMRunError as exc:
            self._hint.setText(str(exc))
            return
        self._acted = True
        self.run.emit(config)
        self._dismiss()


#: The section's pages behind its strip, in order; the sidebar's sub-views are ``hmm.<page>``. Named by what they draw
#: since 23 Nov 2026 (asked: "hmm across and per condition doesnt make much sense. we should either name these
#: different plots. also the bubbles with the rates is tab 1. angles could be tab 3 or 4 we dont need a distinct page
#: for that"): the state diagrams first, the slow state in every condition, the Jump angles of this page's fits (a
#: sidebar section of its own from 22 Nov), the table. The keys stay ("overview" is the slow state's page), so a
#: session's page still opens.
PAGES = ("cards", "overview", "angles", "table")
PAGE_LABELS = ("State diagrams", "Slow state", "Jump angles", "Table")
#: The Table page's states (`hmm_view.condition_table`; a fit without the state shows a dash and says why).
TABLE_STATES = STATES


def _scrolled_page(widget) -> QScrollArea:
    """``widget`` at the top of a scroll area with no sideways bar, the rest of the height empty."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea { background:transparent; }")
    host = QWidget()
    host.setStyleSheet("background:transparent;")
    host_lay = QVBoxLayout(host)
    host_lay.setContentsMargins(0, 0, 0, 0)
    host_lay.addWidget(widget)
    host_lay.addStretch(1)
    scroll.setWidget(host)
    return scroll


class HMMSection(QWidget):
    """The HMM sidebar section: a placeholder until the first result, then one card per
    included condition, filling in as conditions land."""

    run_requested = Signal()
    cancel_requested = Signal()
    #: A card's Export button, with the condition's key: the window opens the dialog
    #: (kind "hmm_condition", `figure_export`).
    export_requested = Signal(str)
    #: The comparison card's Export button: every condition's slow state (kind
    #: "hmm_comparison"), from `comparison_export`.
    comparison_export_requested = Signal()
    #: "Export all diagrams…": every shown condition's state diagram in one figure (kind "hmm_overview",
    #: `figure_export`), from `overview_entries` (5 Nov 2026).
    overview_export_requested = Signal()
    #: The comparison strip was drawn again (a result landed, an eye, the order, the pairs): its brackets follow.
    comparison_changed = Signal()
    #: The strip was pressed: the page's index in `PAGES` (the window moves the sidebar's sub-view).
    page_changed = Signal(int)
    #: "Map this fit" (`map_fit_source.MAP_THIS_FIT`): the window opens the Spatial Map's settings for this page's fit.
    map_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.placeholder = SectionPlaceholder(
            t, title="HMM",
            description="A hidden Markov model of diffusive states: how often molecules "
                        "switch between slow and fast, measured per step, beside how much "
                        "time they spend in each.",
            icon_name="arrow-left-right")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)
        self._page = QWidget()
        self._page.setStyleSheet("background:transparent;")
        page = QVBoxLayout(self._page)
        page.setContentsMargins(0, 0, 0, rem(0.5))
        page.setSpacing(rem(1.0))
        #: Said once for every card (`hmm_view.READING_NOTES`).
        self.reading = InfoLink(t, "How to read these numbers", READING_NOTES)
        page.addWidget(self.reading)
        self.comparison = HMMComparison(t)
        #: An eye per condition (17 Oct 2026, "an eye whenever there is a condition"): a hidden one leaves the
        #: four panels and their export; its own card below stays.
        self.comparison_key = ConditionKey(t)
        self.comparison_key.hidden_changed.connect(lambda _h: self._refresh_comparison())
        comparison_body = QWidget()
        comparison_body.setStyleSheet("background:transparent;")
        comparison_lay = QVBoxLayout(comparison_body)
        comparison_lay.setContentsMargins(0, 0, 0, 0)
        comparison_lay.setSpacing(rem(0.25))
        comparison_lay.addWidget(self.comparison_key)
        # THE DWELL PANEL (12 Nov 2026, asked: "can we get for hmm a dwell time and rate +- error. also for export
        # option to export dwell time as well"): off; a fifth panel after the rate out, and Export starts as shown
        # (`FigureSpec.hmm_dwell`)
        self.dwell_box = Checkbox(t, checked=False)
        self.dwell_box.setToolTip("Draw a fifth panel with the slow state's dwell (1 / its rate out) in ms. It is "
                                  "model-derived, usually longer than the tracks, and an estimate.")
        self.dwell_box.toggled.connect(lambda _on: self._refresh_comparison())
        self._dwell_row = QWidget()
        self._dwell_row.setStyleSheet("background:transparent;")
        dwell_lay = QHBoxLayout(self._dwell_row)
        dwell_lay.setContentsMargins(0, 0, 0, 0)
        dwell_lay.setSpacing(rem(0.5))
        dwell_lay.addWidget(self.dwell_box)
        dwell_lay.addWidget(label("Dwell time", ty.ui("xs"), t.muted_foreground))
        dwell_lay.addStretch(1)
        comparison_lay.addWidget(self._dwell_row)
        # JOIN PAIRED REPLICATES (5 Nov 2026, off): shown only where replicates of two drawn conditions were tagged
        # alike on the Groups board; display only, nothing is tested from it
        self._pairs: dict = {}
        self.pair_box = Checkbox(t, checked=False)
        self.pair_box.setToolTip("Draw a thin line through the replicates declared as pairs on the Groups board (Pair "
                                 "replicates). Display only: the paired test is chosen under Significance.")
        self.pair_box.toggled.connect(self._pair_lines_toggled)
        self._pair_row = QWidget()
        self._pair_row.setStyleSheet("background:transparent;")
        pair_lay = QHBoxLayout(self._pair_row)
        pair_lay.setContentsMargins(0, 0, 0, 0)
        pair_lay.setSpacing(rem(0.5))
        pair_lay.addWidget(self.pair_box)
        pair_lay.addWidget(label("Join paired replicates", ty.ui("xs"), t.muted_foreground))
        pair_lay.addStretch(1)
        comparison_lay.addWidget(self._pair_row)
        self._pair_row.hide()
        comparison_lay.addWidget(self.comparison)
        self.comparison_card = _PanelCard(t, "Across conditions · the slow state",
                                          comparison_body, flush=False)
        self.comparison_card.enable_export()
        self.comparison_card.export_requested.connect(self.comparison_export_requested.emit)
        # THE AXES (30 Oct 2026): each panel's y typed (they have their own units), kept through a new Run
        self.comparison_card.enable_inspect("Axes", glyph="ruler")
        self.comparison_card.inspect_requested.connect(self.open_comparison_axes)
        self._comparison_axes: dict = {}
        self._axes_popover = None
        #: What the strip paints right now (`hmm_view.comparison_payload`), or None.
        self._comparison: dict | None = None
        page.addWidget(self.comparison_card)
        self.comparison_card.hide()
        # PER CONDITION, the strip's second page (22 Nov 2026): the cards
        cards_page = QWidget()
        cards_page.setStyleSheet("background:transparent;")
        cards = QVBoxLayout(cards_page)
        cards.setContentsMargins(0, 0, 0, rem(0.5))
        cards.setSpacing(rem(1.0))
        # EVERY DIAGRAM IN ONE FIGURE (asked 5 Nov 2026: "with the bubbles, fraction sizes, rates, D for all four
        # conditions here, now it only exports one which i cannot build into one image"): the cards' own diagrams side
        # by side, the conditions the comparison's eyes show, in the section's order
        overview_row = QHBoxLayout()
        overview_row.setContentsMargins(0, 0, 0, 0)
        overview_row.addStretch(1)
        self.overview_export = Button(t, "Export all diagrams…", size="small", variant="outline", glyph="download")
        self.overview_export.setToolTip("Every condition's state diagram in one figure, in this order. Pick the "
                                        "conditions with the box beside each name in the dialog (those hidden by the "
                                        "eyes above start unticked).")
        self.overview_export.clicked.connect(self.overview_export_requested.emit)
        overview_row.addWidget(self.overview_export)
        cards.addLayout(overview_row)
        grid_host = QWidget()
        grid_host.setStyleSheet("background:transparent;")
        # As many columns as the width holds (two at most): a card needs its diagram and
        # its rates side by side (`widgets.cardgrid`).
        self._grid = CardGrid(grid_host, min_width=rem(39), max_columns=2,
                              spacing=rem(1.0))
        cards.addWidget(grid_host)
        # TABLE, the strip's third page (22 Nov 2026; the UI designer's table from the retired redesign, `hmm_table`): one state in
        # every condition, its D, dwell, three shares and rates in and out (`hmm_view.condition_table`)
        from .hmm_table import ConditionTable

        table_page = QWidget()
        table_page.setStyleSheet("background:transparent;")
        table_lay = QVBoxLayout(table_page)
        table_lay.setContentsMargins(0, 0, 0, rem(0.5))
        table_lay.setSpacing(rem(0.75))
        state_row = QHBoxLayout()
        state_row.setContentsMargins(0, 0, 0, 0)
        state_row.setSpacing(rem(0.5))
        state_row.addWidget(label("State", ty.ui("sm", "medium"), t.foreground))
        self.table_state = Segmented(t, [s.capitalize() for s in TABLE_STATES], current=0, size="sm")
        self.table_state.changed.connect(lambda _i: self._refresh_table())
        state_row.addWidget(self.table_state)
        state_row.addStretch(1)
        table_lay.addLayout(state_row)
        self.table_note = fit_label("", ty.ui("xs"), t.muted_foreground)
        self.table_note.setWordWrap(True)
        table_lay.addWidget(self.table_note)
        self.table = ConditionTable(t)
        table_lay.addWidget(self.table)
        # JUMP ANGLES (`hmm_angles_page`): turning back per state, read from THIS page's fits; the window asks for the
        # angles while the tab is on show. It scrolls itself.
        from .hmm_angles_page import HMMAnglesPage

        self.angles = HMMAnglesPage(t)
        # PAGES BEHIND A STRIP (22 Nov 2026, asked: "does it make sense to do different tabs for some other analysis
        # tools as well? for example HMM, the bubble rates and the other plots?"); in `PAGES`' order since 23 Nov 2026
        self._content = QStackedWidget()
        pages = {"cards": _scrolled_page(cards_page), "overview": _scrolled_page(self._page),
                 "angles": self.angles, "table": _scrolled_page(table_page)}
        for key in PAGES:
            self._content.addWidget(pages[key])
        self.strip = Segmented(t, list(PAGE_LABELS), current=0, size="sm")
        self.strip.changed.connect(self.page_changed.emit)
        strip_row = QHBoxLayout()
        strip_row.setContentsMargins(0, 0, 0, 0)
        strip_row.addWidget(self.strip)
        strip_row.addStretch(1)
        self.map_button = Button(t, MAP_THIS_FIT, size="small", variant="outline", glyph="grid")
        self.map_button.setToolTip(map_this_fit_tip("hmm"))
        self.map_button.clicked.connect(self.map_requested.emit)
        strip_row.addWidget(self.map_button)
        self._strip_host = QWidget()
        self._strip_host.setStyleSheet("background:transparent;")
        self._strip_host.setLayout(strip_row)
        outer.addWidget(self._strip_host)
        self._pages = QStackedWidget()
        self._pages.addWidget(self.placeholder)
        self._pages.addWidget(self._content)
        #: Each condition's HMM result of this Run: what the Jump angles tab reads (`results`, `hmm_result`).
        self._hmm_results: dict = {}
        outer.addWidget(self._pages)
        self._cards: dict[str, HMMCard] = {}
        self._labels: dict[str, str] = {}
        self._colours: dict[str, str] = {}
        self._ever_shown = False
        self.last_config: dict = {}

    def begin_run(self, conditions: list, labels: dict | None = None,
                  colours: dict | None = None, config: dict | None = None) -> None:
        for card in self._cards.values():
            discard(card)
        self._cards = {}
        self._labels = dict(labels or {})
        self._colours = dict(colours or {})
        for i, key in enumerate(conditions):
            card = HMMCard(self.t, f"HMM · {self._labels.get(key, key)}")
            card.enable_export()
            card.export_requested.connect(lambda k=key: self.export_requested.emit(k))
            self._cards[key] = card
            self._grid.addWidget(card)
        self.last_config = dict(config or {})
        self._hmm_results = {}
        self._refresh_comparison()
        self.placeholder.set_running(True)
        self._show_content() if self._ever_shown else self._pages.setCurrentIndex(0)

    def _show_content(self) -> None:
        self._pages.setCurrentWidget(self._content)

    def show_page(self, sub: str = PAGES[0]) -> None:
        """``sub`` a page of `PAGES` (anything else the first), the strip following without signalling; the page, or
        the placeholder before the first result."""
        index = PAGES.index(sub) if sub in PAGES else 0
        self._content.setCurrentIndex(index)
        if self.strip.current() != index:
            self.strip.blockSignals(True)
            self.strip.select(index)
            self.strip.blockSignals(False)
        if self._ever_shown:
            self._show_content()
        else:
            self._pages.setCurrentIndex(0)

    def page(self) -> str:
        return PAGES[self._content.currentIndex()]

    def _refresh_table(self) -> None:
        """The Table page: the chosen state in every condition with a fit, in the cards' order."""
        state = TABLE_STATES[self.table_state.current()]
        entries = [(key, self._labels.get(key, key), self._colours.get(key, _DATA), card.result())
                   for key, card in self._cards.items() if card.result() is not None]
        if not entries:
            self.table.set(None, None, 0)
            self.table_note.setText("")
            return
        self.table.set(condition_table(entries, state), None, 0)
        self.table_note.setText(CONDITION_TABLE_NOTE)

    def results(self) -> dict:
        """This Run's HMM result of every condition that landed (what a Jump angles page reads)."""
        return dict(self._hmm_results)

    def run_conditions(self) -> tuple:
        """``(conditions, labels, colours)`` of this Run, in the cards' order (a Jump angles page starts from them)."""
        return list(self._cards), dict(self._labels), dict(self._colours)

    def hmm_result(self, condition: str):
        """This Run's HMM result for ``condition`` (what its angles are read from), or None."""
        return self._hmm_results.get(condition)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`, the Conditions tool's arrows, 3 Nov 2026): the cards, the
        slow state's panels and the Jump angles tab move; nothing is refitted or recomputed."""
        self._cards = {k: self._cards[k] for k in order_apply(list(self._cards), order)}
        self._grid.reorder(list(self._cards.values()))
        self._refresh_comparison()
        self.angles.set_order(order)

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        for key, card in self._cards.items():
            if key in colours:
                card.set_colour(colours[key])
        self._refresh_comparison()
        self.angles.set_colours(colours)

    def set_grouping(self, factors: dict, nesting) -> None:
        """The folder design's factors and the nesting chosen (`factor_layout`): the comparison's columns are
        grouped by them, now and at every later Run, here and on the Test page."""
        self._grouping = (dict(factors or {}), tuple(nesting or ()))
        self._refresh_comparison()

    def _refresh_comparison(self) -> None:
        """The strip over every card holding a fit, in the cards' order; hidden until one
        does. The Table page follows (`_refresh_table`)."""
        self._refresh_table()
        entries = [(key, self._labels.get(key, key), self._colours.get(key, _DATA),
                    card.result()) for key, card in self._cards.items()
                   if card.result() is not None]
        if not entries:
            self._comparison = None
            self.comparison_card.hide()
            return
        self.comparison_key.blockSignals(True)
        self.comparison_key.set_entries([(k, n, c) for k, n, c, _r in entries])
        self.comparison_key.blockSignals(False)
        hidden = self.comparison_key.hidden()
        shown = [e for e in entries if e[0] not in hidden]
        if not shown:
            self._comparison = None
            self.comparison.set_payload(None)
            self.comparison_card.set_caption("Every condition is hidden: click an eye above to show it again.")
            return
        payload = comparison_payload(shown, grouping=getattr(self, "_grouping", None), pairs=self._pairs,
                                     dwell=self.dwell_on())
        self._comparison = payload
        self._pair_row.setVisible(bool(payload.get("paired")))
        self.comparison.set_pair_lines(self.pair_lines_on())
        self.comparison.set_payload(payload)
        self._set_comparison_caption()
        self.comparison_card.show()
        self.comparison_changed.emit()

    def bracket_data(self) -> dict | None:
        """What the brackets compare: the payload the comparison strip paints, or None while it paints nothing."""
        return self._comparison

    def set_brackets(self, marks, correction_name: str = "Holm") -> None:
        """The Significance tool's brackets on the strip (`significance.bracket_marks`), and the caption saying so,
        with the correction they were adjusted by (`significance.CORRECTION_NAMES`; Holm unless chosen otherwise)."""
        self._correction_name = str(correction_name or "Holm")
        self.comparison.set_brackets(marks)
        self._set_comparison_caption()

    def _set_comparison_caption(self) -> None:
        payload = self._comparison or {}
        caption = str(payload.get("caption") or "")
        if self.comparison.brackets():
            caption = caption.replace("No statistic between conditions is drawn.",
                                      "Brackets: the Significance tool's test between conditions, "
                                      f"{getattr(self, '_correction_name', 'Holm')}-adjusted.")
        if self.pair_lines_on():
            caption += f" {PAIR_LINES_WORDS[0].upper()}{PAIR_LINES_WORDS[1:]}."
        self.comparison_card.set_caption(caption)

    def set_pairs(self, pairs: dict) -> None:
        """The replicates declared as pairs on the Groups board (`project.Project.replicate_pairs`): what the pair
        lines join and the paired test reads."""
        self._pairs = {str(c): dict(tags) for c, tags in (pairs or {}).items()}
        self._refresh_comparison()

    def dwell_on(self) -> bool:
        """Whether the strip draws its fifth panel, the slow state's model dwell (the "Dwell time" box)."""
        return bool(self.dwell_box.checked)

    def pair_lines_on(self) -> bool:
        """Whether the pair lines are drawn: ticked, and a tag shared by two drawn conditions."""
        return bool(self.pair_box.checked and (self._comparison or {}).get("paired"))

    def _pair_lines_toggled(self, _on: bool) -> None:
        self.comparison.set_pair_lines(self.pair_lines_on())
        self._set_comparison_caption()

    def set_payload(self, condition: str, result: dict) -> None:
        card = self._cards.get(condition)
        if card is None:
            return
        if "error" in result:
            card.set_error(result["error"])
        else:
            card.set_result(result, self._colours.get(condition, _DATA))
            self._hmm_results[condition] = result
        self._refresh_comparison()
        self._ever_shown = True
        self._show_content()

    def set_error(self, condition: str, message: str) -> None:
        card = self._cards.get(condition)
        if card is not None:
            card.set_error(message)
        self._refresh_comparison()
        if self._cards:
            self._ever_shown = True
            self._show_content()

    def has_result(self) -> bool:
        return self._ever_shown

    def cards(self) -> dict:
        return dict(self._cards)

    def comparison_export(self) -> dict | None:
        """What the comparison card's Export draws: the payload the strip paints
        (`hmm_view.comparison_payload`), a copy with each panel's typed y range (``"axes"``), or
        None while no card holds a fit."""
        if self._comparison is None:
            return None
        return {**self._comparison, "axes": dict(self._comparison_axes)}

    # -- typed ranges (30 Oct 2026) ------------------------------------------------------------------------------ #
    def _comparison_rows(self) -> list:
        panels = (self._comparison or {}).get("panels") or []
        return axis_rows("hmm_comparison", panels=[(p["key"], p["title"], p["unit"]) for p in panels])

    def comparison_axes(self) -> dict:
        """Each panel's typed y range (``{panel key: (from, to)}``; empty: automatic)."""
        return dict(self._comparison_axes)

    def set_comparison_axes(self, axes) -> None:
        """Panels' y ranges (`axes_view.check` against the panels shown, refused by name); all None: automatic."""
        rows = self._comparison_rows()
        axes = check(rows, axes or {}) if rows else dict(axes or {})
        self._comparison_axes = {k: v for k, v in axes.items() if v is not None}
        self.comparison.axes = dict(self._comparison_axes)
        self.comparison.update()

    def open_comparison_axes(self) -> None:
        """The Axes popover for the strip: a row per panel."""
        if self._axes_popover is None:
            self._axes_popover = AxesPopover(self.t, self)
            self._axes_popover.applied.connect(self.set_comparison_axes)
        self._axes_popover.show_for(self._comparison_rows(), self._comparison_axes, self.comparison_card._inspect,
                                    note="Each panel its own y. Empty: from 0 to what is drawn.")

    def overview_entries(self) -> list:
        """What "Export all diagrams…" offers (`figure_export`'s "hmm_overview"): every card holding a fit, as its
        Export would draw it (`export_entries`), in the section's order. Which are drawn is the Export dialog's, a box
        beside each name; it starts without those the comparison's eyes hide (`overview_left_out`)."""
        return [e for key in self._cards for e in self.export_entries(key)]

    def overview_left_out(self) -> tuple:
        """The names of the conditions the comparison's eyes hide: unticked when the Export dialog opens."""
        hidden = self.comparison_key.hidden()
        return tuple(e["name"] for e in self.overview_entries() if e["key"] in hidden)

    def export_entries(self, key: str) -> list:
        """What the card ``key``'s Export draws: ``[{"key", "name", "colour", "result"}]``,
        or ``[]`` while it holds no fit (still computing, or an error)."""
        card = self._cards.get(key)
        result = None if card is None else card.result()
        if result is None:
            return []
        return [{"key": key, "name": self._labels.get(key, key),
                 "colour": self._colours.get(key, _DATA), "result": result}]
