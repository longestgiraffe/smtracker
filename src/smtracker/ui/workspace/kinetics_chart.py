"""The Kinetics sidebar section: how fast a population responds to something you did.

WHAT IT IS. `smtracker.kinetics` turns a TRIGGERED experiment (a drug added, a carbon source
switched, a photo-activation) into a rate constant. It was built and tested and reachable
from nothing; the original app never exposed it (its own "Kinetics" tab is the unrelated
HMM switching-rate view, which the workspace already has under Apparent Diffusion). This
module is the door, drawn from `docs/gui_tool_views.md` section 3.

THE CLOCK IS THE DESIGN. Time is SECONDS SINCE THE TRIGGER, so every molecule is on one
clock because one external event started the response. There is one readout, the SLOW
FRACTION: a two-state HMM is fitted ONCE per condition (never per time bin) and only its
per-step posterior is binned, which is the rule `timecourse` and `kinetics` both state.

NOTHING THAT DECIDES WHAT COUNTS AS A MEASUREMENT HAS A DEFAULT. The trigger frame is a
fact about the experiment, so it is typed per condition; the bin width is in SECONDS and
typed (or the curve is left unbinned), and this module always hands the spine either a
width or `binned=False`, so the spine's silent ten-bin fallback can never be the one used.

WHAT THE CARDS REFUSE TO HIDE. Each card carries the untriggered drift the response has to
beat (it is not zero), the half-life flag when the acquisition could not resolve it, a
delta-BIC tie, and, when the user asks to align each unit on its own onset, THE NULL THAT
ALIGNMENT REQUIRES, beside it every time: aligning on a feature found in each unit's own
noise manufactures that feature, so only the excess over the null is evidence.

NO STATISTIC BETWEEN CONDITIONS. Each condition is computed on its own and drawn beside the
others; which statistic answers "are these two different" is the experimenter's decision.

A condition that fails is a card that says why, never an absence.

Layout of the file: the chart, the card, the configuration dialog and the section. The
numbers and sentences come from `smtracker.kinetics_view` (headless, picklable, run on the
process pool); this module reads only the plain lists and strings it returns, and imports no
numpy, pandas or scipy (`tests/test_ui_thin_qt.py`).

Typed ranges (30 Oct 2026, `smtracker.axes_view`, the shared popover): a Populations over time card's Axes sets the
time (both panels), the D panels' y (linear, one typed range for all) and the share panel's y; the section keeps them by condition through a new
Run until Auto (`typed_axes`, `set_typed_axes`, `axes_state`), and `export_entries` hands them to the figure.
"""
from __future__ import annotations

import math

from ...condition_order import apply as order_apply
from ...axes_view import apply_range, axis_rows, check
from ...axes_view import empty as axes_empty
from ..widgets.axes_popover import AxesPopover

from ... import chart_text
from qtpy.QtCore import QLocale, QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QDoubleValidator, QFontMetricsF, QIntValidator, QPainterPath, QPen
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...induction_view import (
    CLOCK_NOTES,
    CLOCK_WORDS,
    DESIGN_EXPLANATIONS,
    DESIGN_WORDS,
    DESIGNS,
    ALTERNATIVES_LABEL,
    INDUCTION_MODEL_HEADINGS,
    d_panels,
    N_POPS,
    POPULATION_READOUTS,
    REFERENCE_LABEL,
    READOUT_WORDS,
    READOUTS,
    TIME_UNIT_WORDS,
    TIME_UNITS,
    convert_range,
    design_of,
    half_time_mark,
    half_time_panels,
    induction_caption,
    induction_model_cells,
    population_hover,
)
from ...population_style import REFERENCE_DASH, population_styles
from ...kinetics_designs import DESIGN_CLOCKS, parse_paste, same_time_fill, samples_from_rows, seed_culture_times
from ...kinetics_mock_view import (BLOCK_HEADING, HELD_OUT_HEADING, LINE_X, held_out_caution, leads,
                                   mock_lines)
from ...kinetics_view import (
    PERMUTATION_WORDS_SAMPLE,
    auto_bin,
    reference_note,
    MOCK_AFTER_CHOICES,
    MOCK_AFTER_NOTE,
    MOCK_AFTER_WORDS,
    MOCK_NONE,
    MOCK_READOUTS,
    MOCK_TIP,
    MOCK_WORDS,
    PRECISION_WORDS,
    READOUT_NOTES,
    SPREAD_WORDS,
    UNITS,
    bin_note,
    card_caption,
    config_problem,
    kinetics_config,
    mobility_preview,
    MODEL_HEADINGS,
    PERMUTATION_WORDS,
    PREVIEW_MAX_CONDITIONS,
    PREVIEW_NOTE,
    PREVIEW_WORDS,
    REFERENCE_TIP,
    REFERENCE_WORDS,
    model_cells,
    response_hover,
)
from ..theme import typography as ty
from .kinetics_design_art import KineticsDesignArt
from ..widgets.cardgrid import CardGrid
from ..widgets.controls import Button, Segmented
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.options_dialog import _ChoiceBox
from ..widgets.primitives import Checkbox, FitScroll, label
from .qc_chart import _DATA, _FLAG, _Caption, _PanelCard, _QCChart, draw_marker
from .section_placeholder import SectionPlaceholder
from .toolbar import rem
from ..widgets.primitives import discard

__all__ = [
    "KineticsConfigDialog",
    "KineticsSection",
    "PopulationChart",
    "ResponseChart",
]

# --------------------------------------------------------------------------- #
# The chart
# --------------------------------------------------------------------------- #


class ResponseChart(_QCChart):
    """Slow fraction (%) against seconds since the trigger.

    ``payload``: ``{"result": compute_kinetics's dict, "colour": hex}``. Binned, every
    unit is a thin pale curve and the mean across units is the bold line; unbinned, every
    TRACK is a dot. The recommended fit is dashed over either. The y axis follows the
    data (a few points of change would vanish on a 0-100 axis) and says what it is.
    """

    _FULL_BOTTOM = 40
    _ROOMY_H = 160
    _PLOT_LEFT = 66

    def _paint(self, p, box: QRectF, pl) -> None:
        result, colour = pl["result"], pl.get("colour") or _DATA
        data = result["chart"]
        if data["y_range"] is None:
            return
        lo, hi = data["y_range"]
        t_max = max(data["t_max"], 1e-9)
        pad = max((hi - lo) * 0.08, 1.0)
        axis_lo, axis_hi = self._grid(p, box, max(0.0, lo - pad), min(100.0, hi + pad),
                                      chart_text.SLOW_PCT_LABEL)

        def px(t):
            return box.left() + float(t) / t_max * box.width()

        def py(v):
            return self._map_y(box, float(v), axis_lo, axis_hi)

        colour_q = QColor(colour)
        self._geom = {"box": box, "t_max": t_max, "binned": data["binned"],
                      "unit": data["unit"], "mean": data["mean"], "points": data["points"]}
        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))      # nothing is drawn outside the axes
        if data["binned"]:
            pale = QColor(colour_q)
            pale.setAlpha(70)
            p.setBrush(Qt.NoBrush)
            for pts in data["units"]:
                if len(pts) < 2:
                    continue
                path = QPainterPath()
                for i, (t, v) in enumerate(pts):
                    (path.moveTo if i == 0 else path.lineTo)(px(t), py(v))
                p.setPen(QPen(pale, 1.0))
                p.drawPath(path)
            if data["mean"]:
                path = QPainterPath()
                for i, (t, v, _sd, _n) in enumerate(data["mean"]):
                    (path.moveTo if i == 0 else path.lineTo)(px(t), py(v))
                p.setPen(QPen(colour_q, 2.2))
                p.drawPath(path)
                p.setPen(Qt.NoPen)
                p.setBrush(colour_q)
                for t, v, _sd, _n in data["mean"]:
                    p.drawEllipse(QPointF(px(t), py(v)), 2.6, 2.6)
        else:
            dots = QColor(colour_q)
            dots.setAlpha(80)
            p.setPen(Qt.NoPen)
            p.setBrush(dots)
            for t, v in data["points"]:
                p.drawEllipse(QPointF(px(t), py(v)), 1.7, 1.7)
        if data["fit"]:
            path = QPainterPath()
            for i, (t, v) in enumerate(data["fit"]):
                (path.moveTo if i == 0 else path.lineTo)(px(t), py(v))
            line = QColor(self.t.foreground)
            line.setAlpha(220)
            pen = QPen(line, 1.6)
            pen.setStyle(Qt.DashLine)
            p.setBrush(Qt.NoBrush)
            p.setPen(pen)
            p.drawPath(path)
        p.restore()
        self._x_numeric(p, box, 0.0, t_max, chart_text.TRIGGER_TIME_LABEL)

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        t = (pos.x() - g["box"].left()) / g["box"].width() * g["t_max"]
        return response_hover(g, t)


def dash_pen(ink, width: float, dash) -> QPen:
    """A pen of ``width`` px in ``ink``: solid, or the on / off lengths (in line widths) of a population's line."""
    pen = QPen(QColor(ink), width)
    if dash:
        pen.setDashPattern([float(d) for d in dash])
    return pen


class PopulationChart(_QCChart):
    """One panel of "Populations over time": each population's D (its own linear axis, `d_panels`) or share (%) against seconds since
    the induction.

    ``payload``: ``{"panel": induction_view's chart["D"] or ["fractions"], "t_range", "colours", "y_label",
    "kind": "D" | "share"}``. The model is a line with its 95 % band (whole units resampled); the free per-bin fits
    are hollow dots beside it -- what the data look like, never the estimate. A dashed rule marks the induction.
    """

    _FULL_BOTTOM = 40
    _ROOMY_H = 150
    _PLOT_LEFT = 66
    _TOP_MARGIN = 22                                   # room for the legend above the plot

    def _paint(self, p, box: QRectF, pl) -> None:
        panel, styles = pl["panel"], pl["styles"]
        colours = [st["ink"] for st in styles]
        typed = pl.get("axes") or {}
        k_t = float(pl.get("t_scale") or 1.0)            # seconds per unit of the time axis (seconds, or minutes)
        t0, t1 = apply_range(pl["t_range"][0] / k_t, pl["t_range"][1] / k_t, typed.get("x"))
        span = max(t1 - t0, 1e-9)
        lo, hi = apply_range(*panel["y_range"], typed.get("y"), log=bool(panel["log"]))
        self.drawn = {"x": (t0, t1), "y": (lo, hi)}
        if panel["log"] and typed.get("y") is not None:
            # a typed range may hold no decade: 1-2-5 ticks inside it, thinned to what the panel's height holds
            ticks = chart_text.log_ticks(lo, hi)
            room = max(2, int(box.height() // 22))
            for keep in ((1, 5), (1,)):
                if len(ticks) > room:
                    ticks = [v for v in ticks if round(v / 10 ** math.floor(math.log10(v) + 1e-9)) in keep] or ticks
            self._draw_grid(p, box, [(self._map_log(box, v, lo, hi), f"{v:g}") for v in ticks], pl["y_label"])
            span_log = math.log10(hi) - math.log10(lo)

            def py(v):                                 # below a typed floor: under the panel (clipped), not on it
                return box.bottom() - (math.log10(max(float(v), 1e-12)) - math.log10(lo)) / span_log * box.height()
        elif panel["log"]:
            self._grid_log(p, box, lo, hi, pl["y_label"])

            def py(v):
                return self._map_log(box, max(float(v), lo), lo, hi)
        elif typed.get("y") is not None:
            axis_lo, axis_hi = self._grid_exact(p, box, lo, hi, pl["y_label"])

            def py(v):
                return self._map_y(box, float(v), axis_lo, axis_hi)
        else:
            axis_lo, axis_hi = self._grid(p, box, lo, hi, pl["y_label"])

            def py(v):
                return self._map_y(box, float(v), axis_lo, axis_hi)

        def px(t):
            return box.left() + (float(t) / k_t - t0) / span * box.width()

        self._geom = {"box": box, "t0": t0, "span": span, "series": panel["series"], "kind": pl["kind"],
                      "time_unit": pl.get("t_unit") or "s"}
        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))      # nothing is drawn outside the axes
        if t0 <= 0.0 <= t1:
            rule = QColor(self.t.muted_foreground)
            rule.setAlpha(150)
            pen = QPen(rule, 1.0)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(px(0.0), box.top()), QPointF(px(0.0), box.bottom()))
        for series, style in zip(panel["series"], styles):
            ink = QColor(style["ink"])
            shape = style["marker"]
            band = series["band"]
            if band:
                path = QPainterPath()
                for i, (t, _low, up) in enumerate(band):
                    (path.moveTo if i == 0 else path.lineTo)(px(t), py(up))
                for t, low, _up in reversed(band):
                    path.lineTo(px(t), py(low))
                path.closeSubpath()
                fill = QColor(ink)
                fill.setAlpha(45)
                p.setPen(Qt.NoPen)
                p.setBrush(fill)
                p.drawPath(path)
            p.setBrush(Qt.NoBrush)
            reference = series.get("reference") or []
            if len(reference) > 1:
                # THE REFERENCE (a file the person chose): a long grey dash, under the model (8 Nov 2026: in the
                # population's ink it read as the fast population's dash-dot)
                path = QPainterPath()
                for i, (t, v) in enumerate(reference):
                    (path.moveTo if i == 0 else path.lineTo)(px(t), py(v))
                p.setPen(dash_pen(QColor(self.t.muted_foreground), 1.5, REFERENCE_DASH))
                p.drawPath(path)
            # EACH REPLICATE'S OWN per-bin fit: small open circles, the app's convention for a replicate
            faint = QColor(ink)
            faint.setAlpha(170)
            p.setPen(QPen(faint, 1.0))
            for _rep, t, v in series.get("replicates") or ():
                draw_marker(p, shape, QPointF(px(t), py(v)), 1.9)
            p.setPen(QPen(ink, 1.3))
            for t, v in series["bins"]:
                draw_marker(p, shape, QPointF(px(t), py(v)), 2.8)
            # THE SHAPES THE DATA CANNOT TELL APART, faint and behind the bold chosen curve (8 Nov 2026)
            for alt in series.get("alternatives") or ():
                if len(alt["curve"]) > 1:
                    path = QPainterPath()
                    for i, (t, v) in enumerate(alt["curve"]):
                        (path.moveTo if i == 0 else path.lineTo)(px(t), py(v))
                    faint_line = QColor(ink)
                    faint_line.setAlpha(95)
                    p.setPen(dash_pen(faint_line, 1.2, style["dash"]))
                    p.drawPath(path)
            if series["curve"]:
                path = QPainterPath()
                for i, (t, v) in enumerate(series["curve"]):
                    (path.moveTo if i == 0 else path.lineTo)(px(t), py(v))
                p.setPen(dash_pen(ink, 2.2, style["dash"]))
                p.drawPath(path)
        mark = pl.get("half_time")
        if mark:
            # THE HALF-TIME (8 Nov 2026, as the figure draws it, `induction_view.half_time_mark`): a light band over its
            # range or interval, a dotted line at its middle, behind nothing the eye needs (drawn last, faint)
            muted = QColor(self.t.muted_foreground)
            shade = QColor(muted)
            shade.setAlpha(40)
            p.setPen(Qt.NoPen)
            p.setBrush(shade)
            left, right = px(mark["lo_s"]), px(mark["hi_s"])
            p.drawRect(QRectF(left, box.top(), max(right - left, 1.0), box.height()))
            p.setBrush(Qt.NoBrush)
            pen = QPen(muted, 1.2)
            pen.setStyle(Qt.DotLine)
            p.setPen(pen)
            p.drawLine(QPointF(px(mark["mid_s"]), box.top()), QPointF(px(mark["mid_s"]), box.bottom()))
        p.restore()
        if mark:
            # its words in the corner the curve has left: the bottom right where it rises, the top right where it falls
            font = ty.ui("xs", tabular=True)
            fm = QFontMetricsF(font)
            p.setFont(font)
            p.setPen(QColor(self.t.muted_foreground))
            h = fm.height() * len(mark["lines"])
            top = box.bottom() - h - 3 if mark["rising"] else box.top() + 3
            p.drawText(QRectF(box.left(), top, box.width() - 5, h), Qt.AlignRight | Qt.AlignTop,
                       "\n".join(mark["lines"]))
        self._x_numeric(p, box, t0, t1, chart_text.induction_time_label(pl.get("t_unit") or "s"))
        # in the strip above the plot, from its left: never over a curve, never past the card's edge
        entries = [(c, s["name"], f"shape:{k}")
                   for k, (s, c) in enumerate(zip(panel["series"], colours), start=int(pl.get("index0") or 0))]
        if any(s.get("reference") for s in panel["series"]):
            entries.append((self.t.muted_foreground, REFERENCE_LABEL, "dash"))
        if any(s.get("alternatives") for s in panel["series"]):
            entries.append((self.t.muted_foreground, ALTERNATIVES_LABEL, "line"))
        self._legend(p, entries, QPointF(box.left() + 4, box.top() - self._TOP_MARGIN + 2), horizontal=True,
                     backdrop=False)

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        t = g["t0"] + (pos.x() - g["box"].left()) / g["box"].width() * g["span"]
        return population_hover(g["series"], t, unit=g["kind"], time_unit=g["time_unit"]) or None


class MobilityPreviewChart(_QCChart):
    """The Run dialog's raw-data preview of ONE condition (8 Nov 2026): each movie's mean squared jump over 4 tau (a
    model-free apparent D) against its time since the addition, one dot per movie, coloured by replicate. No fit.

    ``payload``: ``{"points": [{"movie", "replicate", "rank", "colour", "t_s", "D"}], "unit": "s" | "min"}``
    (`kinetics_view.mobility_preview`). The replicate colours are the condition's own order (Rule 0: the same colour
    in two conditions means nothing)."""

    _FULL_BOTTOM = 34
    _ROOMY_H = 90
    _PLOT_LEFT = 58
    _TOP_MARGIN = 20

    def _paint(self, p, box: QRectF, pl) -> None:
        points = pl["points"]
        k_t = 60.0 if pl.get("unit") == "min" else 1.0
        xs = [pt["t_s"] / k_t for pt in points]
        t0, t1 = min(xs), max(xs)
        pad = 0.05 * (t1 - t0) if t1 > t0 else 1.0
        t0, t1 = t0 - pad, t1 + pad
        top = max(pt["D"] for pt in points)
        axis_lo, axis_hi = self._grid(p, box, 0.0, top * 1.1 if top > 0 else 1.0, chart_text.D_LINEAR_LABEL)

        def px(t):
            return box.left() + (float(t) - t0) / (t1 - t0) * box.width()

        self._geom = {"box": box, "dots": []}
        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))
        if t0 <= 0.0 <= t1:
            rule = QColor(self.t.muted_foreground)
            rule.setAlpha(150)
            pen = QPen(rule, 1.0)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(px(0.0), box.top()), QPointF(px(0.0), box.bottom()))
        for pt in points:
            ink = QColor(pt["colour"])
            centre = QPointF(px(pt["t_s"] / k_t), self._map_y(box, pt["D"], axis_lo, axis_hi))
            p.setPen(QPen(ink, 1.2))
            fill = QColor(ink)
            fill.setAlpha(150)
            p.setBrush(fill)
            p.drawEllipse(centre, 3.4, 3.4)
            self._geom["dots"].append((centre, pt))
        p.restore()
        self._x_numeric(p, box, t0, t1, chart_text.induction_time_label(pl.get("unit") or "s"))
        seen = {}
        for pt in points:
            seen.setdefault(pt["rank"], (pt["colour"], pt["replicate"]))
        entries = [(ink, f"replicate {rep}", "dot") for _rank, (ink, rep) in sorted(seen.items())]
        self._legend(p, entries, QPointF(box.left() + 4, box.top() - self._TOP_MARGIN + 2), horizontal=True,
                     backdrop=False)

    def _hover_text(self, pos: QPointF) -> str | None:
        near = [(math.hypot(c.x() - pos.x(), c.y() - pos.y()), pt) for c, pt in (self._geom or {}).get("dots", ())]
        if not near:
            return None
        distance, pt = min(near, key=lambda item: item[0])
        if distance > 10.0:
            return None
        return f"{pt['movie']} (replicate {pt['replicate']})\n{pt['D']:.3g} µm²/s at {pt['t_s']:.0f} s"


class MockChart(_QCChart):
    """Each replicate BEFORE the addition and AFTER it (8 Nov 2026, `kinetics_mock_view.mock_lines`, as Fig. 6D): the
    treated condition and the mock side by side, a thin line per replicate from its value before to its value after, in
    the readout's own unit (µm²/s, %), the replicates' mean as a bold line; above, each condition's own mean change and
    the p of their difference. The groups wear the CONDITIONS' own colours.

    ``payload``: ``{"lines": mock_lines's dict, "colours": {"treated", "mock"}, "names": {"treated", "mock"}}``. A line
    joins one replicate's two windows; nothing joins a treated replicate to a mock one (Rule 0: the two groups are
    independent samples)."""

    _FULL_BOTTOM = 40
    _ROOMY_H = 90
    _PLOT_LEFT = 58
    _TOP_MARGIN = 40

    def _paint(self, p, box: QRectF, pl) -> None:
        drawn = pl["lines"]
        lo, hi = drawn["y_range"]
        axis_lo, axis_hi = self._grid(p, box, lo, hi, drawn["title"])
        span_x = LINE_X["mock"][1] - LINE_X["treated"][0]

        def px(x):
            return box.left() + box.width() * (0.1 + 0.8 * (x - LINE_X["treated"][0]) / span_x)

        def py(v):
            return self._map_y(box, v, axis_lo, axis_hi)

        self._geom = {"box": box, "dots": []}
        p.save()
        p.setClipRect(box.adjusted(-4, -4, 4, 4))
        for column in drawn["columns"]:
            ink = QColor(pl["colours"][column["role"]])
            x0, x1 = (px(x) for x in column["x"])
            faint = QColor(ink)
            faint.setAlpha(140)
            name = pl["names"][column["role"]] or column["role"]
            for replicate, before, after in column["rows"]:
                a, b = QPointF(x0, py(before)), QPointF(x1, py(after))
                p.setPen(QPen(faint, 1.2))
                p.drawLine(a, b)
                p.setPen(Qt.NoPen)
                p.setBrush(faint)
                for point in (a, b):
                    p.drawEllipse(point, 2.6, 2.6)
                self._geom["dots"] += [(a, f"{name}, replicate {replicate}: before {before:.3g} {drawn['unit']}"),
                                       (b, f"{name}, replicate {replicate}: after {after:.3g} {drawn['unit']}")]
            if all(math.isfinite(v) for v in column["mean"]):
                pen = QPen(ink, 3.0)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(x0, py(column["mean"][0])), QPointF(x1, py(column["mean"][1])))
        p.restore()
        # before / after under each group, the group's name under them in its colour
        small = ty.ui("xs")
        p.setFont(small)
        p.setPen(QColor(self.t.muted_foreground))
        for column in drawn["columns"]:
            for x, word in zip(column["x"], ("before", "after")):
                p.drawText(QRectF(px(x) - 40, box.bottom() + 3, 80, 14), Qt.AlignHCenter | Qt.AlignTop, word)
        p.setFont(ty.ui("xs", "semibold"))
        for column in drawn["columns"]:
            centre = (px(column["x"][0]) + px(column["x"][1])) / 2
            p.setPen(QColor(pl["colours"][column["role"]]))
            name = pl["names"][column["role"]] or column["role"]
            p.drawText(QRectF(centre - 0.24 * box.width(), box.bottom() + 18, 0.48 * box.width(), 14),
                       Qt.AlignHCenter | Qt.AlignTop,
                       QFontMetricsF(ty.ui("xs", "semibold")).elidedText(name, Qt.ElideRight, 0.48 * box.width()))
        # the heading: each condition's own change, then the verdict and p
        p.setFont(ty.ui("sm", "medium"))
        p.setPen(QColor(self.t.foreground))
        p.drawText(QRectF(box.left(), 2, box.width(), self._TOP_MARGIN - 6), Qt.AlignLeft | Qt.AlignVCenter,
                   chr(10).join(drawn["heading"]))

    def _hover_text(self, pos: QPointF) -> str | None:
        near = [(math.hypot(c.x() - pos.x(), c.y() - pos.y()), text) for c, text in (self._geom or {}).get("dots", ())]
        if not near:
            return None
        distance, text = min(near, key=lambda item: item[0])
        return text if distance <= 10.0 else None


# --------------------------------------------------------------------------- #
# The card
# --------------------------------------------------------------------------- #


class _KineticsBody(QWidget):
    """What a condition card holds: a message, or the chart plus its sentences."""

    CHART_H = 230

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        #: The drawn result, kept so a colour change can repaint it.
        self._result = None
        self.setStyleSheet("background:transparent;")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        self._lay.setSpacing(rem(0.5))
        self.message = _Caption(t, size="sm")
        self.message.setText("computing…")
        self._lay.addWidget(self.message)
        self.chart = ResponseChart(t)
        self.chart.setMinimumHeight(self.CHART_H)
        self.chart.hide()
        self._lay.addWidget(self.chart)
        self._models = QGridLayout()
        self._models.setHorizontalSpacing(rem(1.0))
        self._models.setVerticalSpacing(rem(0.15))
        self._models_host = QWidget()
        self._models_host.setLayout(self._models)
        self._models_host.hide()
        self._lay.addWidget(self._models_host)
        self._lines: list[QWidget] = []
        #: Spare card height goes below the text, never between its lines.
        self._tail = self._lay.addStretch(1)

    def _clear_lines(self) -> None:
        for w in self._lines:
            discard(w)
        self._lines = []
        while self._models.count():
            item = self._models.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _at(self, before) -> int:
        """Where a line or chart goes: above ``before`` (a widget in the card's flow), else above the closing stretch."""
        return self._lay.indexOf(before) if before is not None else self._lay.count() - 1

    def _add_line(self, text: str, *, flag: bool = False, muted: bool = True, before=None) -> None:
        w = _Caption(self.t, size="xs" if muted else "sm")
        if flag:
            w.setStyleSheet(f"color:{_FLAG}; background:transparent;")
        w.setText(text)
        self._lay.insertWidget(self._at(before), w)
        self._lines.append(w)

    def set_colour(self, colour: str) -> None:
        """Re-colour the drawn result -- a change in the Conditions tool, no refit."""
        result = getattr(self, "_result", None)
        if result is not None:
            self.chart.set_payload({"result": result, "colour": colour})

    def set_error(self, text: str) -> None:
        self._result = None
        self._clear_lines()
        self.chart.hide()
        self._models_host.hide()
        self.message.setText(text)
        self.message.show()

    def set_result(self, result: dict, colour: str) -> None:
        self._clear_lines()
        self.message.hide()
        self._result = result
        self.chart.set_payload({"result": result, "colour": colour})
        self.chart.show()
        text = result["text"]
        t = self.t
        for col, head in enumerate(MODEL_HEADINGS):
            self._models.addWidget(label(head, ty.ui("xs", "medium"), t.muted_foreground), 0, col)
        for row, cells in enumerate(model_cells(text["models"]), start=1):
            *columns, best = cells
            ink = t.foreground if best else t.muted_foreground
            for col, text_ in enumerate(columns):
                self._models.addWidget(label(text_, ty.ui("sm"), ink), row, col)
        self._models.setColumnStretch(len(MODEL_HEADINGS), 1)
        self._models_host.show()
        for line in text["parameters"]:
            self._add_line(line, muted=False)
        self._add_line(text["change"])
        for line in text["flags"]:
            self._add_line(line, flag=True)
        if text["alignment"] is not None:
            self._add_line("Aligned on each unit's own onset", muted=False)
            for line in text["alignment"]:
                self._add_line(line)


class _PopulationsBody(_KineticsBody):
    """"Populations over time": one D panel PER POPULATION (each its own y range, `induction_view.d_panels`) over the
    share panel, the model table, the sentences."""

    PANEL_H = 190
    #: A D panel's height when several are stacked. 170, not 150 (8 Nov 2026): at 150 the plot box came to about 89 px,
    #: under the 90 px the y title needs (`qc_chart._draw_grid`), so the D panels showed numbers without "D (µm²/s)".
    D_PANEL_H = 170

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        discard(self.chart)                            # the slow-fraction chart is not this readout's
        self.chart = None
        #: Typed ranges (the card's Axes, 30 Oct 2026): time (both panels), D, share.
        self.axes: dict = {}
        #: The time axis' unit ("s" or "min", `induction_view.TIME_UNITS`), set by the section for every card.
        self.time_unit = "s"
        self._colour_now = _DATA
        #: One chart per population, in the order of its series; `d_chart` is the first (the only one of one).
        self.d_charts: list = []
        self.f_chart = PopulationChart(t)
        self.f_chart.setMinimumHeight(self.PANEL_H)
        self._lay.insertWidget(1, self.f_chart)
        self.f_chart.hide()
        self._fit_d_charts(1)

    @property
    def d_chart(self):
        return self.d_charts[0]

    def _fit_d_charts(self, n: int) -> None:
        """Exactly ``n`` D panels above the share panel (made or discarded; a re-run with another population count
        changes the stack)."""
        while len(self.d_charts) > n:
            discard(self.d_charts.pop())
        while len(self.d_charts) < n:
            chart = PopulationChart(self.t)
            self._lay.insertWidget(1 + len(self.d_charts), chart)
            self.d_charts.append(chart)
            chart.hide()
        for chart in self.d_charts:
            chart.setMinimumHeight(self.PANEL_H if n == 1 else self.D_PANEL_H)

    #: The mock condition's colour and this card's condition name, set by the section before a result (the mock chart
    #: wears both conditions' own colours).
    mock_colour = _DATA
    condition_name = ""

    def _add_widget(self, cls, payload: dict, height: int = 190, before=None) -> None:
        """A chart in the card's flow (above ``before``, else above the closing stretch); discarded with the lines on a
        re-draw."""
        chart = cls(self.t)
        chart.setMinimumHeight(height)
        chart.set_payload(payload)
        self._lay.insertWidget(self._at(before), chart)
        self._lines.append(chart)

    def set_colour(self, colour: str) -> None:
        self._colour_now = colour
        if self._result is not None:
            self._draw(self._result, colour)
            if (self._result.get("mock") or None) is not None:
                self.set_result(self._result, colour)             # the mock chart wears the colour too

    def set_error(self, text: str) -> None:
        self._result = None
        self._clear_lines()
        for chart in (*self.d_charts, self.f_chart):
            chart.hide()
        self._models_host.hide()
        self.message.setText(text)
        self.message.show()

    def _draw(self, result: dict, colour: str) -> None:
        chart = result["chart"]
        styles = population_styles(colour, len(chart["D"]["series"]), dark=self.t.color_scheme == "dark")
        typed = self.axes or {}
        unit = self.time_unit
        panels = d_panels(chart)
        self._fit_d_charts(len(panels))
        # the half-time on the panels that changed (8 Nov 2026): every D panel where the D's did, the share panel
        # where the shares did; a result made before carries none and draws as it did
        marked = half_time_panels(result.get("held_out_change"))

        def mark(which, series):
            if which not in marked or not series:
                return None
            return half_time_mark(result, which, series[0]["curve"], time_unit=unit)

        for k, (d_chart, panel) in enumerate(zip(self.d_charts, panels)):
            # the typed D range applies to every D panel (the Axes popover says so)
            d_chart.set_payload({"panel": panel, "t_range": chart["t_range"], "styles": [styles[k]], "index0": k,
                                 "y_label": chart_text.D_LINEAR_LABEL, "kind": "D",
                                 "t_scale": TIME_UNITS[unit], "t_unit": unit,
                                 "half_time": mark("D", panel["series"]),
                                 "axes": {"x": typed.get("x"), "y": typed.get("D")}})
        self.f_chart.set_payload({"panel": chart["fractions"], "t_range": chart["t_range"], "styles": styles,
                                  "y_label": chart_text.SHARE_PCT_LABEL, "kind": "share",
                                  "t_scale": TIME_UNITS[unit], "t_unit": unit,
                                  "half_time": mark("share", chart["fractions"]["series"]),
                                  "axes": {"x": typed.get("x"), "y": typed.get("share")}})

    def set_axes(self, axes) -> None:
        """Typed ranges ``{"x", "D", "share"}`` (the card's Axes): time across both panels, each panel's y."""
        self.axes = dict(axes or {})
        if self._result is not None:
            self._draw(self._result, self._colour_now)

    def set_result(self, result: dict, colour: str) -> None:
        self._clear_lines()
        self.message.hide()
        self._result = result
        self._colour_now = colour
        self._draw(result, colour)
        for d_chart in self.d_charts:
            d_chart.show()
        self.f_chart.setVisible(bool(result["chart"].get("share_panel", True)))
        text = result["text"]
        t = self.t
        if leads(text):
            # THE MOCK COMPARISON LEADS (ruled 8 Nov 2026, `kinetics_mock_view`): its block above the model table, its
            # chart where it could be made, then its sentences (or the refusal in its words); the held-out verdict
            # second, under its heading and the drift it cannot take out
            top = self._models_host
            self._add_line(BLOCK_HEADING, muted=False, before=top)
            names = {"treated": self.condition_name, "mock": result["mock"]["mock"]}
            drawn = mock_lines(result["mock"], names["treated"] or "treated", names["mock"] or "mock")
            if drawn is not None:
                self._add_widget(MockChart, {"lines": drawn, "colours": {"treated": colour, "mock": self.mock_colour},
                                             "names": names}, height=220, before=top)
            for line in text["mock"]:
                self._add_line(line, before=top)
            self._add_line(HELD_OUT_HEADING, muted=False, before=top)
            self._add_line(held_out_caution(result["mock"]), before=top)
        for col, head in enumerate(INDUCTION_MODEL_HEADINGS):
            self._models.addWidget(label(head, ty.ui("xs", "medium"), t.muted_foreground), 0, col)
        for row, cells in enumerate(induction_model_cells(text["models"]), start=1):
            *columns, chosen = cells
            ink = t.foreground if chosen else t.muted_foreground
            for col, text_ in enumerate(columns):
                self._models.addWidget(label(text_, ty.ui("sm"), ink), row, col)
        self._models.setColumnStretch(len(INDUCTION_MODEL_HEADINGS), 1)
        self._models_host.show()
        for line in text["lines"]:
            self._add_line(line, muted=False)
        if text.get("permutation"):
            # BESIDE the held-out verdict, never replacing it (7 Nov 2026)
            self._add_line("Time-label permutation test", muted=False)
            for line in text["permutation"]:
                self._add_line(line)
        for line in text["flags"]:
            self._add_line(line, flag=True)
        if text["spread"] is not None:
            self._add_line("Onsets between movies", muted=False)
            for line in text["spread"]:
                self._add_line(line)
        for line in text.get("notes") or ():
            self._add_line(line)
        if text.get("design"):
            # WHAT WAS OBSERVED (the experiment designs, 7 Nov 2026): the window, the baseline, the acquisition per
            # time window and the within-track statement, on every card
            self._add_line("Design and acquisition", muted=False)
            for line in text["design"]:
                self._add_line(line)


class KineticsCard(_PanelCard):
    """One condition's triggered response, in the readout its Run asked for (`induction_view.READOUTS`)."""

    def __init__(self, t, title: str, parent=None, *, readout: str = "slow_fraction"):
        self.readout = readout
        self.kinetics_body = _PopulationsBody(t) if readout in POPULATION_READOUTS else _KineticsBody(t)
        super().__init__(t, title, self.kinetics_body, parent=parent, flush=False)

    def set_result(self, result: dict, colour: str) -> None:
        self.set_caption(induction_caption(result) if result.get("readout") == "populations"
                         else card_caption(result))
        self.kinetics_body.set_result(result, colour)

    def set_colour(self, colour: str) -> None:
        self.kinetics_body.set_colour(colour)

    def set_error(self, text: str, detail: str = "") -> None:
        """``detail``: the job's whole traceback, on the message's hover (the message alone says what refused, not
        where)."""
        self.set_caption("")
        self.kinetics_body.set_error(text)
        self.kinetics_body.message.setToolTip(detail or "")


# --------------------------------------------------------------------------- #
# The configuration dialog
# --------------------------------------------------------------------------- #


class _TextField(QFrame):
    """A one-line number input with a unit suffix, in the same shadcn Input styling as
    the QC laser field. ``integer`` selects whole frames; otherwise a positive decimal."""

    changed = Signal(str)

    def __init__(self, t, *, integer: bool | None, unit: str, placeholder: str, width: int = 132,
                 parent=None):
        """``integer`` None: free text (a clock time), checked by the dialog's gate rather than a validator."""
        super().__init__(parent)
        self.setObjectName("kfield")
        self.setStyleSheet(
            f"QFrame#kfield {{ background:{t.card}; border:1px solid {t.input};"
            f" border-radius:{t.radius_md:g}px; }}")
        self.setFixedSize(width, t.control_height)
        self._edit = QLineEdit("")
        self._edit.setFrame(False)
        self._edit.setFont(ty.ui("sm"))
        self._edit.setPlaceholderText(placeholder)
        self._edit.setStyleSheet(
            f"QLineEdit {{ background:transparent; color:{t.foreground};"
            f" border:none; padding:0px; }}")
        if integer is not None:
            if integer:
                validator = QIntValidator(0, 10_000_000, self._edit)
            else:
                validator = QDoubleValidator(0.0, 1e6, 3, self._edit)
                validator.setNotation(QDoubleValidator.StandardNotation)
            validator.setLocale(QLocale.c())
            self._edit.setValidator(validator)
        self._edit.textChanged.connect(self.changed.emit)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(8)
        lay.addWidget(self._edit, 1)
        lay.addWidget(label(unit, ty.ui("sm"), t.muted_foreground))

    def text(self) -> str:
        return self._edit.text().strip()

    def set_text(self, text: str) -> None:
        self._edit.setText(text)


class KineticsConfigDialog(OverlayDialog):
    """The settings no one else can know: which condition got a trigger, and when.

    TRANSACTIONAL, like the QC comparator's dialog: nothing runs until Run, and Cancel
    and the corner X discard the draft. Run stays disabled until at least one condition
    is included and every included one has a trigger frame (and, when binned, a bin width
    in seconds). Neither has a default; entries from the last run seed the fields so a
    re-run does not have to retype them.
    """

    #: The raw-data preview chart of one condition: tall enough for its x axis and the replicate key.
    PREVIEW_H = 130

    run = Signal(dict)

    def __init__(self, t, conditions: list, *, seed: dict | None = None, content=None,
                 parent=None, days: dict | None = None, movies: dict | None = None, saved: dict | None = None,
                 design_art_reduced: bool = False, saved_times: dict | None = None, extents_for=None,
                 mobility_for=None,
                 found_reference: list | None = None):
        """``days`` ``{condition: [(day, n movies, first movie)]}`` (`induction_view.condition_days`): the rows for
        the times of the addition, under the "Time after the addition" clock. ``movies`` ``{condition: [(movie, the
        name its file wears, replicate)]}`` (`induction_view.condition_movies`) and ``saved`` ``{condition: {sample:
        {"movies", "time"}}}`` (`kinetics_designs.load_additions`): the table of the design "Added to each sample",
        seeded from the last Run, else from what was saved. ``saved_times`` ``{condition: {replicate: typed time}}``
        (`kinetics_designs.load_culture_times`): the addition times typed for the replicates on an earlier Run, the
        starting text of their fields when the last Run of this session holds none. ``extents_for(condition,
        camera)`` gives `induction_view.movie_extents` of a condition (the window's, read when Auto first needs it);
        without it Auto cannot guess and says so. ``found_reference``: the ``reference*.csv`` files in the project
        root (`induction_view.find_reference`); the first is used unless the last Run named a file."""
        super().__init__(t, content, parent=parent)
        self.t = t
        self._card.setMinimumWidth(rem(32))
        seed = seed or {}
        days = days or {}
        movies = movies or {}
        saved = saved or {}
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label("Configure kinetics", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        self.card_lay.addWidget(label("Nothing runs until you press Run.", ty.ui("sm"),
                                      t.muted_foreground))

        # THE SETTINGS SCROLL on a short window, the title and Run stay (found by `scripts/gui_sweep.py`, 25 Oct
        # 2026: at 1280 x 720 the dialog was taller than the window and its rows were drawn over each other). The
        # Spatial Map dialog's column (`primitives.FitScroll`): a tall window shows every setting with no scroll bar.
        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(self.card_lay.spacing())
        body.addWidget(label("Readout", ty.ui("sm", "medium"), t.muted_foreground))
        readout_row = QHBoxLayout()
        self._readout = Segmented(t, [READOUT_WORDS[r] for r in READOUTS],
                                  current=READOUTS.index(seed.get("readout", READOUTS[0])))
        self._readout.changed.connect(lambda _i: self._refresh())
        readout_row.addWidget(self._readout)
        readout_row.addStretch(1)
        body.addLayout(readout_row)
        self._readout_note = label("", ty.ui("xs"), t.muted_foreground)
        self._readout_note.setWordWrap(True)
        body.addWidget(self._readout_note)
        body.addSpacing(rem(0.5))

        # THE EXPERIMENT DESIGN (7 Nov 2026, `kinetics_designs`): what "time since the addition" means, what is
        # independent and what the permutation test may deal. Each name is explained under it.
        self._design_title = label("How was it added?", ty.ui("sm", "medium"), t.muted_foreground)
        body.addWidget(self._design_title)
        self._design = Segmented(t, [DESIGN_WORDS[d] for d in DESIGNS], current=DESIGNS.index(design_of(seed)))
        self._design.changed.connect(lambda _i: self._refresh())
        design_row = QHBoxLayout()
        design_row.addWidget(self._design)
        design_row.addStretch(1)
        body.addLayout(design_row)
        # the name and its two lines (from the headless module) beside the loop that shows them (about 5 s, still
        # under reduced motion)
        self._design_name = label("", ty.ui("sm", "semibold"), t.foreground)
        self._design_notes = [label("", ty.ui("xs"), t.muted_foreground) for _ in range(2)]
        words = QVBoxLayout()
        words.setSpacing(rem(0.25))
        words.addWidget(self._design_name)
        for note in self._design_notes:
            note.setWordWrap(True)
            words.addWidget(note)
        words.addStretch(1)
        self._design_art = KineticsDesignArt(t, DESIGNS[self._design.current()], reduced_motion=design_art_reduced)
        art_row = QHBoxLayout()
        art_row.setSpacing(rem(0.75))
        art_row.addLayout(words, 1)
        art_row.addWidget(self._design_art, 0, Qt.AlignTop)
        body.addLayout(art_row)
        body.addSpacing(rem(0.5))

        self._clock_title = label("Clock", ty.ui("sm", "medium"), t.muted_foreground)
        body.addWidget(self._clock_title)
        clock_row = QHBoxLayout()
        self._micro_clocks = DESIGN_CLOCKS["microscope"]
        self._clock = Segmented(t, [CLOCK_WORDS[c] for c in self._micro_clocks],
                                current=self._micro_clocks.index(seed.get("clock"))
                                if seed.get("clock") in self._micro_clocks else 0)
        self._clock.changed.connect(lambda _i: self._refresh())
        clock_row.addWidget(self._clock)
        clock_row.addStretch(1)
        body.addLayout(clock_row)
        body.addSpacing(rem(0.5))

        self._conditions_title = label("", ty.ui("sm", "medium"), t.muted_foreground)
        body.addWidget(self._conditions_title)
        # ONE TIME FOR EVERY REPLICATE (7 Nov 2026): typed once, it fills every replicate field of every ticked
        # condition (`kinetics_designs.same_time_fill`); each field stays editable
        self._same_title = label("Same time for every replicate", ty.ui("sm"), t.foreground)
        self._same = _TextField(t, integer=None, unit="", placeholder="e.g. 10:00", width=140)
        self._same.changed.connect(lambda _text: self._fill_same())
        same_row = QHBoxLayout()
        same_row.setSpacing(rem(0.75))
        same_row.addWidget(self._same_title)
        same_row.addWidget(self._same)
        same_row.addStretch(1)
        body.addLayout(same_row)
        self._same_hosts = [self._same_title, self._same]
        self._rows: dict = {}
        #: ``{condition: {day: field}}``: the time of each day's addition (the "addition" clock)
        self._day_fields: dict = {}
        self._day_hosts: list = []
        #: ``{condition: {movie: (sample name field, time field)}}``: the per-sample table (design "sample")
        self._sample_fields: dict = {}
        self._sample_hosts: list = []
        self._movies = {key: [m for m, _o, _d in rows] for key, rows in movies.items()}
        self._aliases = {key: {o: m for m, o, _d in rows if o != m} for key, rows in movies.items()}
        self._paste_note = label("", ty.ui("xs"), _FLAG)
        self._paste_note.setWordWrap(True)
        seeded_samples = seed.get("samples") or {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(rem(0.75))
        grid.setVerticalSpacing(rem(0.4))
        included = seed.get("conditions", {})
        seeded_times = seed.get("additions") or {}
        saved_times = saved_times or {}
        r = 0
        for key, name in conditions:
            box = Checkbox(t, checked=key in included)
            field = _TextField(t, integer=True, unit="frame", placeholder="e.g. 250")
            if key in included and included[key] is not None:
                field.set_text(str(included[key]))
            box.toggled.connect(lambda _on: self._ticked_changed())
            field.changed.connect(lambda text, b=box: self._on_frame_typed(b, text))
            grid.addWidget(box, r, 0)
            grid.addWidget(label(name, ty.ui("sm"), t.foreground), r, 1)
            grid.addWidget(field, r, 2)
            self._rows[key] = (box, field)
            condition_row = r
            r += 1
            fields = {}
            for day, n_movies, first in days.get(key, []):
                words = label(f"replicate {day} · {n_movies} movie{'' if n_movies == 1 else 's'}"
                              + (f", from {first}" if first else ""), ty.ui("xs"), t.muted_foreground)
                time_field = _TextField(t, integer=None, unit="", placeholder="e.g. 10:30")
                time_field.set_text(seed_culture_times(seeded_times, saved_times, key).get(day, ""))
                time_field.changed.connect(lambda text, b=box: self._on_frame_typed(b, text))
                grid.addWidget(words, r, 1)
                grid.addWidget(time_field, r, 2)
                self._day_hosts += [words, time_field]
                fields[day] = time_field
                r += 1
            self._day_fields[key] = fields
            # the per-sample table: each movie is its own sample unless given a sample name; a time each
            table = seeded_samples.get(key) or saved.get(key) or {}
            by_movie = {m: (name, e["time"]) for name, e in table.items() for m in e["movies"]}
            paste = Button(t, "Paste from Excel…", variant="outline", size="small")
            paste.clicked.connect(lambda _c=False, k=key: self._paste(k))
            paste.setToolTip("Paste times copied from Excel: one tab-separated line per movie, "
                             "movie name then time, or just a column of times in the order shown.")
            grid.addWidget(paste, condition_row, 3)
            self._sample_hosts.append(paste)
            movie_fields = {}
            for movie, original, day in movies.get(key, []):
                name, time = by_movie.get(movie, ("", ""))
                words = label(f"replicate {day} · {original}", ty.ui("xs"), t.muted_foreground)
                name_field = _TextField(t, integer=None, unit="", placeholder="own sample", width=120)
                name_field.set_text("" if name == movie else name)
                time_field = _TextField(t, integer=None, unit="", placeholder="e.g. 10:30")
                time_field.set_text(time)
                time_field.changed.connect(lambda text, b=box: self._on_frame_typed(b, text))
                name_field.changed.connect(lambda _text: self._refresh())
                time_field._edit.installEventFilter(self)
                grid.addWidget(words, r, 1)
                grid.addWidget(time_field, r, 2)
                grid.addWidget(name_field, r, 3)
                self._sample_hosts += [words, name_field, time_field]
                movie_fields[movie] = (name_field, time_field)
                r += 1
            self._sample_fields[key] = movie_fields
        grid.setColumnStretch(1, 1)
        body.addLayout(grid)
        self._clock_note = label("", ty.ui("xs"), t.muted_foreground)
        self._clock_note.setWordWrap(True)
        body.addWidget(self._clock_note)
        body.addWidget(self._paste_note)
        self._paste_note.hide()

        self._extents_for = extents_for
        self._extents: dict = {}
        # THE RAW DATA BEFORE ANY FIT (8 Nov 2026, `kinetics_view.mobility_preview`): one dot per movie, its mean
        # squared jump over 4 tau against its time since the addition, coloured by replicate; updates with the times
        # and the design. Hidden where the dialog was given no way to read the movies.
        self._mobility_for = mobility_for
        self._mobility: dict = {}
        self._preview_title = label(PREVIEW_WORDS, ty.ui("sm", "medium"), t.muted_foreground)
        self._preview_note = label(PREVIEW_NOTE, ty.ui("xs"), t.muted_foreground)
        self._preview_note.setWordWrap(True)
        self._preview_rows: dict = {}                    # condition -> (name label, message label, chart)
        self._preview_more = label("", ty.ui("xs"), t.muted_foreground)
        self._preview_more.setWordWrap(True)
        body.addSpacing(rem(0.5))
        body.addWidget(self._preview_title)
        body.addWidget(self._preview_note)
        for key, name in conditions:
            title = label(name, ty.ui("xs", "medium"), t.foreground)
            message = label("", ty.ui("xs"), t.muted_foreground)
            message.setWordWrap(True)
            chart = MobilityPreviewChart(t)
            chart.setMinimumHeight(self.PREVIEW_H)
            chart.setMaximumHeight(self.PREVIEW_H)
            for widget in (title, message, chart):
                body.addWidget(widget)
                widget.hide()
            self._preview_rows[key] = (title, message, chart)
        body.addWidget(self._preview_more)

        body.addSpacing(rem(0.5))
        body.addWidget(label("Time resolution", ty.ui("sm", "medium"), t.muted_foreground))
        bin_row = QHBoxLayout()
        bin_row.setSpacing(rem(0.75))
        # AUTO is the default (7 Nov 2026): the width is guessed from the movies (`kinetics_view.auto_bin`)
        if seed.get("bin_auto"):
            mode = 0
        elif seed.get("bin_seconds") is None and "unit" in seed:
            mode = 2
        else:
            mode = 1 if seed.get("bin_seconds") is not None else 0
        self._bin_mode = Segmented(t, ["Auto", "Width in seconds", "Unbinned"], current=mode)
        self._bin_mode.changed.connect(lambda _i: self._refresh())
        bin_row.addWidget(self._bin_mode)
        self._width = _TextField(t, integer=False, unit="s", placeholder="e.g. 2")
        if seed.get("bin_seconds") is not None and mode == 1:
            self._width.set_text(f"{seed['bin_seconds']:g}")
        self._width.changed.connect(lambda _text: self._refresh())
        bin_row.addWidget(self._width)
        bin_row.addStretch(1)
        body.addLayout(bin_row)
        self._bin_note = label("", ty.ui("xs"), t.muted_foreground)
        self._bin_note.setWordWrap(True)                  # Auto's sentence is long: a one-line label pinned 1021 px
        body.addWidget(self._bin_note)

        body.addSpacing(rem(0.5))
        body.addWidget(label("Independent unit", ty.ui("sm", "medium"), t.muted_foreground))
        self._unit = Segmented(t, [u.capitalize() for u in UNITS],
                               current=UNITS.index(seed.get("unit", "movie")))
        self._unit.changed.connect(lambda _i: self._refresh())
        unit_row = QHBoxLayout()
        unit_row.addWidget(self._unit)
        unit_row.addStretch(1)
        body.addLayout(unit_row)

        self._pops_title = label("Populations", ty.ui("sm", "medium"), t.muted_foreground)
        body.addSpacing(rem(0.5))
        body.addWidget(self._pops_title)
        # a Mean mobility Run keeps n_pop 1 (one population IS that readout, `kinetics_view.kinetics_config`): the
        # count offered here is the populations readout's, so a count it does not offer opens on the first (the
        # dialog raised on reopening after a mobility Run until 8 Nov 2026; found by the subpage tests)
        seeded_n = seed.get("n_pop")
        self._n_pop = Segmented(t, [str(n) for n in N_POPS],
                                current=N_POPS.index(seeded_n) if seeded_n in N_POPS else 0)
        pops_row = QHBoxLayout()
        pops_row.addWidget(self._n_pop)
        pops_row.addStretch(1)
        body.addLayout(pops_row)
        self._spread = Checkbox(t, checked=bool(seed.get("spread", False)))
        spread_row = QHBoxLayout()
        spread_row.setSpacing(rem(0.5))
        spread_row.addWidget(self._spread)
        self._spread_label = label(SPREAD_WORDS, ty.ui("sm"), t.foreground)
        self._spread_label.setWordWrap(True)
        spread_row.addWidget(self._spread_label, 1)
        body.addSpacing(rem(0.25))
        body.addLayout(spread_row)
        self._precision = Checkbox(t, checked=bool(seed.get("precision", False)))
        precision_row = QHBoxLayout()
        precision_row.setSpacing(rem(0.5))
        precision_row.addWidget(self._precision)
        self._precision_label = label(PRECISION_WORDS, ty.ui("sm"), t.foreground)
        self._precision_label.setWordWrap(True)
        precision_row.addWidget(self._precision_label, 1)
        body.addSpacing(rem(0.25))
        body.addLayout(precision_row)
        self._permutation = Checkbox(t, checked=bool(seed.get("permutation", False)))
        permutation_row = QHBoxLayout()
        permutation_row.setSpacing(rem(0.5))
        permutation_row.addWidget(self._permutation)
        self._permutation_label = label(PERMUTATION_WORDS, ty.ui("sm"), t.foreground)
        self._permutation_label.setWordWrap(True)
        permutation_row.addWidget(self._permutation_label, 1)
        body.addSpacing(rem(0.25))
        body.addLayout(permutation_row)
        # THE REFERENCE CURVE is optional and looks it (7 Nov 2026): one collapsed row with a Browse button, and a
        # reference*.csv in the project root is picked up on its own and said ("Found reference_SynK.csv")
        self._found_reference = [str(f) for f in (found_reference or [])]
        self._reference_path = str(seed.get("reference") or "") or (self._found_reference[0]
                                                                   if self._found_reference else "")
        self._reference_title = label(REFERENCE_WORDS, ty.ui("sm", "medium"), t.muted_foreground)
        self._reference_title.setToolTip(REFERENCE_TIP)
        self._reference_browse = Button(t, "Browse…", variant="outline", size="small")
        self._reference_browse.clicked.connect(self._browse_reference)
        self._reference_clear = Button(t, "Clear", variant="outline", size="small")
        self._reference_clear.clicked.connect(lambda _c=False: self._set_reference(""))
        self._reference_note = label("", ty.ui("xs"), t.muted_foreground)
        self._reference_note.setWordWrap(True)
        self._reference_row = [self._reference_title, self._reference_browse, self._reference_clear,
                               self._reference_note]
        reference_row = QHBoxLayout()
        reference_row.setSpacing(rem(0.5))
        reference_row.addWidget(self._reference_title)
        reference_row.addWidget(self._reference_browse)
        reference_row.addWidget(self._reference_clear)
        reference_row.addWidget(self._reference_note, 1)
        body.addSpacing(rem(0.5))
        body.addLayout(reference_row)

        # A MOCK ADDITION (8 Nov 2026, `kinetics_mock`): a condition imaged on the same schedule with nothing added;
        # every other ticked condition is read against it. A drop-down of the conditions ("None" the default), and
        # under it the after window: Auto, or a start typed in seconds.
        self._mock_keys = [key for key, _name in conditions]
        mock_seed = seed.get("mock") if seed.get("mock") in self._mock_keys else None
        self._mock_title = label(MOCK_WORDS, ty.ui("sm", "medium"), t.muted_foreground)
        self._mock_title.setToolTip(MOCK_TIP)
        self._mock = _ChoiceBox(t, [MOCK_NONE] + [name for _key, name in conditions],
                                current=0 if mock_seed is None else 1 + self._mock_keys.index(mock_seed))
        self._mock.currentIndexChanged.connect(lambda _i: self._refresh())
        mock_row = QHBoxLayout()
        mock_row.setSpacing(rem(0.5))
        mock_row.addWidget(self._mock_title)
        mock_row.addWidget(self._mock)
        mock_row.addStretch(1)
        self._mock_after_title = label(MOCK_AFTER_WORDS, ty.ui("sm"), t.foreground)
        seeded_after = seed.get("mock_after_s")
        self._mock_after = Segmented(t, list(MOCK_AFTER_CHOICES), current=0 if seeded_after is None else 1)
        self._mock_after.changed.connect(lambda _i: self._refresh())
        self._mock_after_field = _TextField(t, integer=False, unit="s", placeholder="e.g. 300")
        if seeded_after is not None:
            self._mock_after_field.set_text(f"{float(seeded_after):g}")
        self._mock_after_field.changed.connect(lambda _text: self._refresh())
        mock_after_row = QHBoxLayout()
        mock_after_row.setSpacing(rem(0.5))
        mock_after_row.addWidget(self._mock_after_title)
        mock_after_row.addWidget(self._mock_after)
        mock_after_row.addWidget(self._mock_after_field)
        mock_after_row.addStretch(1)
        self._mock_note = label(MOCK_AFTER_NOTE, ty.ui("xs"), t.muted_foreground)
        self._mock_note.setWordWrap(True)
        self._mock_widgets = [self._mock_after_title, self._mock_after, self._mock_after_field, self._mock_note]
        body.addSpacing(rem(0.5))
        body.addLayout(mock_row)
        body.addLayout(mock_after_row)
        body.addWidget(self._mock_note)

        self._align = Checkbox(t, checked=bool(seed.get("align", False)))
        align_row = QHBoxLayout()
        align_row.setSpacing(rem(0.5))
        align_row.addWidget(self._align)
        self._align_label = label("Also align each unit to its own onset (slower; the null is "
                                  "always shown with it)", ty.ui("sm"), t.foreground)
        self._align_label.setWordWrap(True)
        align_row.addWidget(self._align_label, 1)
        body.addSpacing(rem(0.5))
        body.addLayout(align_row)
        body_host = QWidget()
        body_host.setStyleSheet("background:transparent;")
        body_host.setLayout(body)
        self.body_scroll = FitScroll()
        self.body_scroll.setWidget(body_host)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QScrollArea.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.card_lay.addWidget(self.body_scroll, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, rem(0.5), 0, 0)
        footer.setSpacing(8)
        self._hint = label("", ty.ui("xs"), t.muted_foreground)
        self._hint.setWordWrap(True)
        footer.addWidget(self._hint)
        footer.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self._run_btn = Button(t, "Run", variant="default")
        self._run_btn.clicked.connect(self._on_run)
        footer.addWidget(cancel)
        footer.addWidget(self._run_btn)
        self.card_lay.addLayout(footer)
        self._refresh()

    # -- the draft -------------------------------------------------------------- #

    def _on_frame_typed(self, box: Checkbox, text: str) -> None:
        """Typing a frame includes the condition: nobody types a trigger for a condition
        they mean to leave out. The checkbox stays for excluding one again."""
        if text.strip() and not box.checked:
            box.set_checked(True, animate=False)
        self._refresh()

    def _fill_same(self, only_blank: bool = False) -> None:
        """The "same time" field into the replicate fields of every ticked condition (`same_time_fill`)."""
        ticked = [key for key, (box, _f) in self._rows.items() if box.checked]
        current = self._additions_text()
        filled = same_time_fill(self._same.text(), current, ticked, only_blank=only_blank)
        for key, days in filled.items():
            for day, text in days.items():
                if text != current[key][day]:
                    self._day_fields[key][day].set_text(text)
        self._refresh()

    def _ticked_changed(self) -> None:
        """A condition ticked after the time was typed takes it into its blank fields."""
        self._fill_same(only_blank=True)

    def _binned(self) -> bool:
        return self._bin_mode.current() != 2

    def _auto(self) -> bool:
        return self._bin_mode.current() == 0

    def _auto_guess(self) -> tuple:
        """Auto's ``(width or None, text)`` for the draft (`kinetics_view.auto_bin`); the movies of a condition are
        read once, when Auto first needs them."""
        if self._extents_for is None:
            return None, "Auto: the movies' times are not available here; type a width"
        clock = self._clock_key()
        rows = self._rows_text()
        extents = {}
        for key, (on, _text) in rows.items():
            if on:
                if (key, clock != "trigger") not in self._extents:
                    self._extents[(key, clock != "trigger")] = self._extents_for(key, clock != "trigger")
                extents[key] = self._extents[(key, clock != "trigger")]
        try:
            samples = self._samples_draft()
        except ValueError as exc:
            return None, f"Auto: {exc}"
        return auto_bin(extents, clock=clock, rows=rows, additions=self._additions_text(), samples=samples)

    def _width_text(self) -> str:
        """The width the draft means: Auto's guess, or what was typed."""
        if self._auto():
            width, _text = self._auto_guess()
            return "" if width is None else f"{width:g}"
        return self._width.text()

    def _rows_text(self) -> dict:
        """What the draft says: each condition's checkbox and typed trigger frame."""
        return {key: (box.checked, field.text()) for key, (box, field) in self._rows.items()}

    def _additions_text(self) -> dict:
        """Each condition's typed time of the addition, per day."""
        return {key: {day: f.text() for day, f in fields.items()} for key, fields in self._day_fields.items()}

    def _design_key(self) -> str:
        return DESIGNS[self._design.current()]

    def _clock_key(self) -> str:
        """The clock in force: the design fixes it, except under the microscope (a frame, or an addition time per
        replicate); the designs belong to the populations readout only."""
        if self._readout_key() not in POPULATION_READOUTS:
            return "trigger"
        clocks = DESIGN_CLOCKS[self._design_key()]
        return clocks[0] if len(clocks) == 1 else self._micro_clocks[self._clock.current()]

    def _samples_draft(self) -> dict:
        """``{condition: {sample: {"movies", "time"}}}`` from the table; RAISES ValueError where a sample's movies
        carry two different times."""
        return {key: samples_from_rows({m: (nf.text(), tf.text()) for m, (nf, tf) in fields.items()})
                for key, fields in self._sample_fields.items()}

    def _problem(self) -> str:
        readout = self._readout_key()
        try:
            samples = self._samples_draft()
        except ValueError as exc:
            return str(exc)
        auto = self._binned() and self._auto()
        # what is missing in the draft comes first; Auto's own reason only when that is complete (its guess needs it)
        problem = config_problem(self._rows_text(), self._binned(), "1" if auto else self._width.text(),
                                 clock=self._clock_key(), additions=self._additions_text(),
                                 unit=UNITS[self._unit.current()], readout=readout,
                                 reference=self._reference_path if readout == "populations" else None,
                                 samples=samples, movies=self._movies, mock=self._mock_key(),
                                 mock_after_text=self._mock_after_text())
        if problem or not auto:
            return problem
        width, text = self._auto_guess()
        return "" if width is not None else text

    def eventFilter(self, obj, event):
        """A paste (Ctrl+V) of several lines into a time field is a paste of the whole column from Excel."""
        from qtpy.QtCore import QEvent
        from qtpy.QtGui import QKeySequence
        from qtpy.QtWidgets import QApplication

        if event.type() == QEvent.KeyPress and event.matches(QKeySequence.Paste):
            text = QApplication.clipboard().text()
            if "\n" in text.strip() or "\t" in text:
                for key, fields in self._sample_fields.items():
                    if any(tf._edit is obj for _nf, tf in fields.values()):
                        self._paste(key, text)
                        return True
        return super().eventFilter(obj, event)

    def _paste(self, key: str, text: str | None = None) -> None:
        """Times from the clipboard into ``key``'s table (`kinetics_designs.parse_paste`); what does not read is
        said under the table, never skipped."""
        from qtpy.QtWidgets import QApplication

        text = QApplication.clipboard().text() if text is None else text
        fields = self._sample_fields.get(key, {})
        got, problems = parse_paste(text, list(fields), aliases=self._aliases.get(key))
        for movie, time in got.items():
            fields[movie][1].set_text(time)
        self._paste_note.setText("; ".join(problems))
        self._paste_note.setVisible(bool(problems))
        self._refresh()

    def _set_reference(self, path: str) -> None:
        """The reference file in force ("" none); the row says which."""
        self._reference_path = str(path or "")
        self._refresh()

    def _browse_reference(self) -> None:
        """The reference CSV (`induction_view.reference_curves` reads it at the Run, and refuses it by name)."""
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(self, "Compare with a known curve", self._reference_path,
                                                    "CSV files (*.csv);;All files (*)")
        if path:
            self._set_reference(path)

    def _readout_key(self) -> str:
        return READOUTS[self._readout.current()]

    def _mock_key(self) -> str | None:
        """The condition chosen as the mock addition, None for "None" (or for a readout that offers none)."""
        index = self._mock.current()
        return self._mock_keys[index - 1] if index > 0 and self._readout_key() in MOCK_READOUTS else None

    def _mock_after_text(self) -> str:
        """The typed start of the after window; "" under Auto."""
        return self._mock_after_field.text() if self._mock_after.current() == 1 else ""

    def _refresh(self) -> None:
        binned = self._binned()
        readout = self._readout_key()
        populations = readout in POPULATION_READOUTS
        mobility = readout == "mobility"
        clock = self._clock_key()
        addition, sample = clock == "addition", clock == "sample"
        design = self._design_key()
        for widget in (self._design_title, self._design, self._design_name, self._design_art, *self._design_notes):
            widget.setVisible(populations)
        self._design_name.setText(DESIGN_WORDS[design])
        self._design_art.set_design(design)
        for note, words in zip(self._design_notes, DESIGN_EXPLANATIONS[design]):
            note.setText(words)
        for widget in (self._clock_title, self._clock):
            widget.setVisible(populations and design == "microscope")
        for _box, field in self._rows.values():
            field.setVisible(not addition and not sample)
        for widget in self._day_hosts + self._same_hosts:
            widget.setVisible(addition)
        for widget in self._sample_hosts:
            widget.setVisible(sample)
        self._paste_note.setVisible(sample and bool(self._paste_note.text()))
        self._conditions_title.setText("Time each sample was added, per movie" if sample
                                       else "Time of the addition, per replicate" if addition
                                       else "Trigger frame, per condition")
        self._clock_note.setText(CLOCK_NOTES[clock])
        self._permutation_label.setText(PERMUTATION_WORDS_SAMPLE if sample else PERMUTATION_WORDS)
        self._readout_note.setText(READOUT_NOTES[readout])
        self._width.setVisible(self._bin_mode.current() == 1)
        self._bin_note.setText((self._auto_guess()[1] + ". " if binned and self._auto() else "")
                               + bin_note(binned, readout))
        for widget in (self._spread, self._spread_label, self._precision, self._precision_label, self._permutation,
                       self._permutation_label):
            widget.setVisible(populations)
        # ONE population has no count to choose, and a reference curve names populations (slow, fast)
        for widget in (self._pops_title, self._n_pop, *self._reference_row):
            widget.setVisible(readout == "populations")
        self._reference_note.setText(reference_note(self._reference_path, self._found_reference))
        self._reference_clear.setVisible(readout == "populations" and bool(self._reference_path))
        self._reference_note.setVisible(readout == "populations" and bool(self._reference_path))
        offers_mock = readout in MOCK_READOUTS
        chosen = offers_mock and self._mock.current() > 0
        for widget in (self._mock_title, self._mock):
            widget.setVisible(offers_mock)
        for widget in self._mock_widgets:
            widget.setVisible(chosen)
        self._mock_after_field.setVisible(chosen and self._mock_after.current() == 1)
        for widget in (self._align, self._align_label):
            widget.setVisible(not populations)
        self._align.setEnabled(binned)
        self._align_label.setEnabled(binned)
        self._update_preview()
        problem = self._problem()
        self._hint.setText(problem)
        self._run_btn.setEnabled(not problem)

    def _update_preview(self) -> None:
        """The raw-data preview for the draft: the first `PREVIEW_MAX_CONDITIONS` ticked conditions, each its chart or
        the line saying what is missing. Nothing is fitted; a condition's movies are read once."""
        if not hasattr(self, "_preview_rows"):             # a refresh while the dialog is still being built
            return
        available = self._mobility_for is not None and self._extents_for is not None
        for widget in (self._preview_title, self._preview_note):
            widget.setVisible(available)
        rows = self._rows_text()
        ticked = [key for key, (on, _text) in rows.items() if on]
        shown = ticked[:PREVIEW_MAX_CONDITIONS] if available else []
        clock = self._clock_key()
        camera = clock != "trigger"
        try:
            samples = self._samples_draft()
        except ValueError:
            samples = {}
        additions = self._additions_text()
        for key, (title, message, chart) in self._preview_rows.items():
            drawn = key in shown
            title.setVisible(drawn)
            message.setVisible(False)
            chart.setVisible(False)
            if not drawn:
                continue
            if (key, camera) not in self._extents:
                self._extents[(key, camera)] = self._extents_for(key, camera)
            if key not in self._mobility:
                self._mobility[key] = self._mobility_for(key)
            got = mobility_preview(self._mobility[key], self._extents[(key, camera)], clock=clock,
                                   trigger_text=rows[key][1], additions=additions.get(key, {}),
                                   samples=samples.get(key, {}))
            if got["points"]:
                span = max(p["t_s"] for p in got["points"]) - min(p["t_s"] for p in got["points"])
                chart.set_payload({"points": got["points"], "unit": "min" if span > 600.0 else "s"})
                chart.setVisible(True)
            else:
                message.setText(got["problem"])
                message.setVisible(True)
        extra = len(ticked) - len(shown) if available else 0
        self._preview_more.setText(f"…and {extra} more condition{'s' if extra != 1 else ''} not drawn here."
                                   if extra > 0 else "")
        self._preview_more.setVisible(extra > 0)

    def config(self) -> dict:
        """The committed settings. RAISES while the draft is incomplete: a missing width
        must never read back as "unbinned"."""
        populations = self._readout_key() in POPULATION_READOUTS
        return kinetics_config(self._rows_text(), self._binned(), self._width_text(),
                               UNITS[self._unit.current()], self._align.checked,
                               readout=self._readout_key(), n_pop=N_POPS[self._n_pop.current()],
                               spread=self._spread.checked, clock=self._clock_key(),
                               additions=self._additions_text(), permutation=self._permutation.checked,
                               reference=self._reference_path or None, precision=self._precision.checked,
                               design=self._design_key() if populations else None,
                               samples=self._samples_draft(), movies=self._movies,
                               bin_auto=self._binned() and self._auto(), mock=self._mock_key(),
                               mock_after_text=self._mock_after_text())

    def _on_run(self) -> None:
        if self._problem():
            return
        self._acted = True
        self.run.emit(self.config())
        self._dismiss()


# --------------------------------------------------------------------------- #
# The section
# --------------------------------------------------------------------------- #


class KineticsSection(QWidget):
    """The Kinetics sidebar section: a placeholder until the first result, then one
    card per included condition, filling in as conditions land.

    The same idle -> content shape the other sections use, with the same one-way door:
    once a card has been drawn, a later failure is reported on the page rather than by
    returning to an empty tray. The Run itself lives in `window.py`, this workspace's
    convention for every Run; `run_requested` asks for it.
    """

    run_requested = Signal()
    cancel_requested = Signal()
    #: A condition card's Export button, with the condition's key.
    export_requested = Signal(str)
    #: The every-condition figure's button (shown once two Populations over time results are in).
    overview_export_requested = Signal()
    #: The figure of every treated condition against its mock addition ("Export against the mock…", 8 Nov 2026).
    mock_export_requested = Signal()
    #: The narrowest a card is drawn: its two panels and its model table, measured 560 px (3 Nov 2026).
    CARD_MIN_W = rem(35)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.placeholder = SectionPlaceholder(
            t, title="Kinetics",
            description="How fast the population responds after something you did, "
                        "against time since the trigger.",
            icon_name="clock")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)

        self._page = QWidget()
        self._page.setStyleSheet("background:transparent;")
        # CARDS IN AS MANY COLUMNS AS THE WIDTH HOLDS (`CardGrid`, as every other section): a fixed two-column grid ran
        # the second card off the window's right edge on a 1366 px laptop (3 Nov 2026, reported: its axis ending "1",
        # the model table's last column gone, the verdict cut mid-word). A card's own content needs `CARD_MIN_W`.
        self._grid = CardGrid(self._page, min_width=self.CARD_MIN_W, max_columns=2, spacing=rem(1.0))
        self._grid.setContentsMargins(0, 0, 0, rem(0.5))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:transparent; }")
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        # EVERY CONDITION IN ONE FIGURE (7 Nov 2026): the conditions as columns on one time axis, one y range per row
        overview_row = QHBoxLayout()
        overview_row.setContentsMargins(0, 0, 0, 0)
        overview_row.addStretch(1)
        self.overview_export = Button(t, "Export all conditions…", size="small", variant="outline", glyph="download")
        self.overview_export.setToolTip("Every condition's populations over time in one figure: the conditions as "
                                        "columns on one time axis, one D range and one share range for all.")
        self.overview_export.clicked.connect(self.overview_export_requested.emit)
        overview_row.addWidget(self.overview_export)
        # EVERY TREATED CONDITION AGAINST ITS MOCK (8 Nov 2026, `figure_export`'s "kinetics_mock"): the cards' mock
        # charts as one figure, shown where some card was read against a mock
        self.mock_export = Button(t, "Export against the mock…", size="small", variant="outline", glyph="download")
        self.mock_export.setToolTip("Every treated condition against its mock addition in one figure: one panel each, "
                                    "a dot per replicate, the change beyond the mock and its p above.")
        self.mock_export.clicked.connect(self.mock_export_requested.emit)
        overview_row.addWidget(self.mock_export)
        host_lay.addLayout(overview_row)
        self.overview_export.hide()
        self.mock_export.hide()
        host_lay.addWidget(self._page)
        host_lay.addStretch(1)
        scroll.setWidget(host)

        self._pages = QStackedWidget()
        self._pages.addWidget(self.placeholder)
        self._pages.addWidget(scroll)
        outer.addWidget(self._pages)
        self._cards: dict[str, KineticsCard] = {}
        self._colours: dict[str, str] = {}
        self._ever_shown = False
        #: "Populations over time"'s typed ranges by condition, kept through a new Run until Auto.
        self._typed: dict = {}
        self._time_unit = "s"
        self._axes_popover = None
        #: The last committed configuration, to seed the next dialog.
        self.last_config: dict = {}

    # -- the run ---------------------------------------------------------------- #
    def begin_run(self, conditions: list, labels: dict | None = None,
                  colours: dict | None = None, config: dict | None = None) -> None:
        """A new Run over ``conditions``: drops every previous card."""
        for card in self._cards.values():
            discard(card)
        self._cards = {}
        self._colours = dict(colours or {})
        labels = labels or {}
        self._labels = dict(labels)
        readout = (config or {}).get("readout", "slow_fraction")
        for i, key in enumerate(conditions):
            card = KineticsCard(self.t, f"Kinetics · {labels.get(key, key)}", readout=readout)
            card.enable_export()
            card.export_requested.connect(lambda k=key: self.export_requested.emit(k))
            if readout in POPULATION_READOUTS:
                # THE AXES (30 Oct 2026): time across both panels, D and share each its own y; kept through a Run
                card.enable_inspect("Axes", glyph="ruler")
                card.inspect_requested.connect(lambda k=key: self.open_axes(k))
                card.kinetics_body.axes = dict(self._typed.get(key) or {})
                card.kinetics_body.time_unit = self._time_unit
            self._cards[key] = card
            self._grid.addWidget(card)
        self.last_config = dict(config or {})
        self.overview_export.hide()
        self.mock_export.hide()
        self.placeholder.set_running(True)
        if not self._ever_shown:
            self._pages.setCurrentIndex(0)
        else:
            self._pages.setCurrentIndex(1)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`, the Conditions tool's arrows, 3 Nov 2026): the cards move;
        nothing is refitted or recomputed."""
        keys = order_apply(list(self._cards), order)
        if keys == list(self._cards):
            return
        self._cards = {k: self._cards[k] for k in keys}
        self._grid.reorder(list(self._cards.values()))

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    def set_colours(self, colours: dict) -> None:
        """Re-colour every drawn card -- a change in the Conditions tool, no refit."""
        self._colours.update(colours)
        for key, card in self._cards.items():
            if key in colours:
                card.set_colour(colours[key])
            else:                                           # a card read against a mock whose colour changed
                body = card.kinetics_body
                mock = ((getattr(body, "_result", None) or {}).get("mock") or {}).get("mock")
                if mock in colours:
                    body.mock_colour = colours[mock]
                    body.set_colour(body._colour_now)

    def set_payload(self, condition: str, result: dict) -> None:
        card = self._cards.get(condition)
        if card is None:
            return
        if "error" in result:
            card.set_error(result["error"], result.get("traceback", ""))
        else:
            body = card.kinetics_body
            if hasattr(body, "mock_colour"):                # the mock chart wears both conditions' own colours
                body.condition_name = self._labels.get(condition, condition)
                body.mock_colour = self._colours.get((result.get("mock") or {}).get("mock"), _DATA)
            card.set_result(result, self._colours.get(condition, _DATA))
        self._ever_shown = True
        self._pages.setCurrentIndex(1)
        self.overview_export.setVisible(len(self.overview_entries()) > 1)
        self.mock_export.setVisible(bool(self.mock_entries()))

    def set_error(self, condition: str, message: str) -> None:
        """A failure reported IN ITS OWN CARD once there is a page to report it on."""
        card = self._cards.get(condition)
        if card is not None:
            card.set_error(message)
        if self._cards:
            self._ever_shown = True
            self._pages.setCurrentIndex(1)

    def has_result(self) -> bool:
        return self._ever_shown

    def cards(self) -> dict:
        return dict(self._cards)

    def export_entries(self, key: str) -> list:
        """What an export draws: ``[{"key", "name", "colour", "result", "axes"}]`` for ``key`` -- ``axes`` its typed
        ranges (Populations over time: ``{"x", "D", "share"}``)."""
        card = self._cards.get(key)
        result = None if card is None else card.kinetics_body._result
        if result is None:
            return []
        return [{"key": key, "name": getattr(self, "_labels", {}).get(key, key),
                 "colour": self._colours.get(key, _DATA), "result": result,
                 "axes": dict(self._typed.get(key) or {}), "time_unit": self._time_unit}]

    def overview_entries(self) -> list:
        """What the every-condition figure draws (`figure_export`'s "kinetics_overview"): every card holding a
        Populations over time result, in the cards' order, each as `export_entries` gives it."""
        out = []
        for key, card in self._cards.items():
            result = card.kinetics_body._result
            if card.readout in POPULATION_READOUTS and result is not None and "error" not in result:
                out += self.export_entries(key)
        return out

    def mock_entries(self) -> list:
        """What the figure against the mock draws (`figure_export`'s "kinetics_mock"): every card read against a mock
        addition whose comparison could be made, in the cards' order, each as `export_entries` gives it with the mock's
        shown name and colour (``mock_name``, ``mock_colour``)."""
        out = []
        for key, card in self._cards.items():
            result = card.kinetics_body._result
            block = (result or {}).get("mock") or {}
            if "welch" not in block:
                continue
            mock = block.get("mock")
            for entry in self.export_entries(key):
                out.append(dict(entry, mock_name=getattr(self, "_labels", {}).get(mock, mock),
                                mock_colour=self._colours.get(mock, _DATA)))
        return out

    # -- typed ranges (30 Oct 2026) ------------------------------------------------------------------------------ #
    def typed_axes(self, key: str) -> dict:
        """The ranges typed for ``key``'s Populations over time (``{"x", "D", "share"}``; empty: automatic). The time
        range is in the section's time unit (`time_unit`)."""
        return dict(self._typed.get(key) or {})

    def time_unit(self) -> str:
        """The unit of the time axis of every Populations card: "s" (seconds, the default) or "min"."""
        return self._time_unit

    def set_time_unit(self, unit: str) -> None:
        """Seconds or minutes on every card's time axis and in the export. DISPLAY only: a typed time range is moved
        to the new unit (a range typed 0 to 300 s reads 0 to 5 min), nothing is refitted."""
        if unit not in TIME_UNITS:
            raise ValueError(f"time unit {unit!r}: choose one of {tuple(TIME_UNITS)}")
        if unit == self._time_unit:
            return
        for key, axes in list(self._typed.items()):
            if axes.get("x") is not None:
                self._typed[key] = dict(axes, x=convert_range(axes["x"], self._time_unit, unit))
        self._time_unit = unit
        for key, card in self._cards.items():
            if card.readout in POPULATION_READOUTS:
                card.kinetics_body.time_unit = unit
                card.kinetics_body.set_axes(self._typed.get(key) or {})

    def set_typed_axes(self, key: str, axes) -> None:
        """Ranges for ``key`` (`axes_view.check`, refused by name), drawn now; all None: automatic. A ``"time_unit"``
        in ``axes`` (the popover's pills) sets the section's unit first, so a time range typed with it is read in it."""
        axes = dict(axes or {})
        if "time_unit" in axes:
            self.set_time_unit(axes.pop("time_unit"))
        axes = check(axis_rows("kinetics_populations"), axes)
        if axes_empty(axes):
            self._typed.pop(key, None)
        else:
            self._typed[key] = axes
        card = self._cards.get(key)
        if card is not None and card.readout in POPULATION_READOUTS:
            card.kinetics_body.set_axes(self._typed.get(key) or {})

    def open_axes(self, key: str) -> None:
        if self._axes_popover is None:
            self._axes_popover = AxesPopover(self.t, self)
            self._axes_popover.applied.connect(lambda axes: self.set_typed_axes(self._axes_key, axes))
        self._axes_key = key
        card = self._cards.get(key)
        choices = [{"key": "time_unit", "label": "time axis in", "value": self._time_unit,
                    "options": [(u, TIME_UNIT_WORDS[u].capitalize()) for u in TIME_UNITS]}]
        self._axes_popover.show_for(axis_rows("kinetics_populations"), self.typed_axes(key),
                                    None if card is None else card._inspect, choices=choices)

    def axes_state(self) -> dict:
        state = {k: dict(v) for k, v in self._typed.items()}
        if self._time_unit != "s":
            state["_time_unit"] = self._time_unit
        return state

    def set_axes_state(self, state: dict) -> None:
        state = dict(state or {})
        unit = state.pop("_time_unit", "s")
        self._typed = {}
        self._time_unit = "s"
        self.set_time_unit(unit)                          # before the ranges: a saved range is in the saved unit
        for key, axes in state.items():
            self.set_typed_axes(key, axes)

