"""The Spatial Map's replicate page: one condition's maps, a row per replicate, and what
makes a clicked spot. Tried behind the rail's Test button from 13 Oct 2026 (asked then:
"don't change the main page of Spatial Map for now"); THE replicate page since 23 Nov 2026
(asked: "I like the redes map page can we finish the idea and include"), with everything
the page it replaced offered: the eye that hides a track from the map, the profiles, both
viewers, the value maps' whole bins, the splits by hand.

Two cards, one question each (mockup 4 of the UI designer's spot-card notes):
- LEFT, where: the days' maps (`SpatialMapMatrix`, so the Visibility tool reaches it
  unchanged), then under a hairline the clicked spot -- its heading, its position, its
  bin, and Cells | Tracks side by side: what the spot is worth (n_eff of N, a meter), its
  contributors as one neutral share bar and an aligned list, each track with its EYE
  (hidden from the map, it stays listed under its spot, struck through, until shown again).
- RIGHT, what it looks like: the spot's largest tracks as JUMP STRIPS (every jump a tick
  on one shared nm axis, the immobile band shaded once behind every row, the median a
  taller mark), then a player that plays the picked track's own frames, zoomed to its
  cell: the whole path faint, the travelled part highlighted, a ringed dot on the
  molecule, loop on, ¼× to 4×, the recorded frames or a still picture behind it, and the
  track handed on to Images & tracks or napari.
- UNDER BOTH, for a method that has them: where the jumps sit, this condition and each of
  its replicates (`spatial_profiles.ProfilesCard`).

One selection: picking a track in the strips or in the Tracks list picks it in both,
thickens it on the map and plays it.

Thin Qt: every word, number and rule is `spot_page_view`'s (and the spot is
`spatial_map_view.spot_contributors`'); the frames are `trackclip.clip_frames`, read on a
worker thread. Tooltips are the styled ones only.
"""
from __future__ import annotations

import traceback

from qtpy.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from qtpy.QtGui import QColor, QCursor, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPolygonF
from qtpy.QtWidgets import (QApplication, QHBoxLayout, QLayout, QSizePolicy, QVBoxLayout,
                            QWidget)

from ... import images_view as iv
from ... import spot_page_view as pv
from ...spatial_map_view import (N_EFF_EXPLAINED, PROFILE_METHODS, SPOT_TOP_TRACKS, day_label,
                                 day_profile_payload, kinds_for, panel_titles_for, spot_contributors)
from ..theme import typography as ty
from ..widgets import tooltip
from ..widgets.controls import Button, Segmented, Slider
from ..widgets.primitives import Card, Checkbox, discard, label
from ..widgets.settings import InfoLink
from ..widgets.wraprow import WrapRow
from .bubble_chart import CARD_GAP, CARD_PAD
from .images_page import _ImageCanvas
from .spatial_map_chart import SpatialMapMatrix, _scrolled
from .toolbar import ToolButton, rem

#: The player's highlight: the travelled part and the molecule. Not a condition colour,
#: not a rank shade: the one saturated mark on the right card.
TRACK_INK = "#FF5A1F"
#: Below this width the right card moves under the left one.
SIDE_BY_SIDE_MIN = 920
#: The narrowest the map card gets when it shrinks to fit the window's height: below this the
#: maps stop being readable, so a window too short for it scrolls instead.
LEFT_MIN_PX = 420


def _clear(widget: QWidget) -> QWidget:
    widget.setStyleSheet("background:transparent;")
    return widget


class _Hairline(QWidget):
    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setFixedHeight(1)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.t.border))
        p.end()


class _Swatch(QWidget):
    """A small rounded square in one rank shade."""

    def __init__(self, colour: str, size: int = 8, parent=None):
        super().__init__(parent)
        self.colour = colour
        self.setFixedSize(size, size)

    def set_colour(self, colour: str) -> None:
        self.colour = colour
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.colour))
        p.drawRoundedRect(QRectF(self.rect()), 2, 2)
        p.end()


class _Meter(QWidget):
    """What a spot is worth against how many touch it: a thin bar, filled to n_eff / N."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.fill = 0.0
        self.setFixedHeight(6)

    def set_fill(self, fill: float) -> None:
        self.fill = max(0.0, min(1.0, float(fill)))
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        r = QRectF(self.rect())
        p.setBrush(QColor(self.t.muted))
        p.drawRoundedRect(r, 3, 3)
        if self.fill > 0:
            p.setBrush(QColor(self.t.foreground))
            p.drawRoundedRect(QRectF(r.x(), r.y(), max(r.height(), r.width() * self.fill),
                                     r.height()), 3, 3)
        p.end()


class _ShareBar(QWidget):
    """The contributors as segments in the rank shades, the rest as one "other" segment."""

    GAP = 2

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.shares: list[float] = []
        self.colours: list[str] = []
        self.other = ""
        self.setFixedHeight(12)

    def set_shares(self, shares, colours, other) -> None:
        self.shares, self.colours, self.other = list(shares), list(colours), other
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        w, h = float(self.width()), float(self.height())
        x = 0.0
        for share, colour in zip(self.shares, self.colours):
            seg = max(3.0, w * share - self.GAP)
            p.setBrush(QColor(colour))
            p.drawRoundedRect(QRectF(x, 0, seg, h), 3, 3)
            x += seg + self.GAP
        if x < w:
            p.setBrush(QColor(self.other or self.t.muted))
            p.drawRoundedRect(QRectF(x, 0, w - x, h), 3, 3)
        p.end()


class _ListRow(QWidget):
    """One contributor: swatch, name, where, share -- its columns sized by the list so
    every row lines up. A track row picks its track, and carries the EYE that hides it from
    the map (asked 16 Oct 2026: "an eye symbol, if u click on it it gets a / on the eye and
    the track is excluded from the map")."""

    clicked = Signal(int)
    #: The eye was clicked: hide this track from the map, or show it again.
    hide_toggled = Signal(int)

    def __init__(self, t, index: int, colour: str, cells: list[tuple[str, str]],
                 pct: str, *, clickable: bool, muted_name: bool = False, eye: bool = False,
                 hidden: bool = False, parent=None):
        super().__init__(parent)
        self.t = t
        self.index = index
        self.selected = False
        self.hovered = False
        self.clickable = clickable
        self._muted_name = muted_name
        if clickable:
            self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        # a row with an eye is the eye's height, no taller than one without (the list fits a laptop's height)
        lay.setContentsMargins(4, 0 if eye else 2, 4, 0 if eye else 2)
        lay.setSpacing(8)
        self.swatch = _Swatch(colour)
        lay.addWidget(self.swatch, 0, Qt.AlignVCenter)
        self.columns = []
        self._roles = []
        for k, (text, role) in enumerate(cells):
            w = label(text, ty.ui("xs", "normal", tabular=True), t.muted_foreground)
            lay.addWidget(w)
            self.columns.append(w)
            self._roles.append(role)
        lay.addStretch(1)
        self.pct = label(pct, ty.ui("xs", tabular=True), t.foreground)
        self.pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.pct)
        self.eye = None
        if eye:
            self.eye = ToolButton(t, "eye", "eye", size=rem(1.125), rest=t.card)
            self.eye.clicked.connect(lambda _k: self.hide_toggled.emit(self.index))
            self._eye_tip = tooltip.attach(t, self.eye, "", side="left")
            lay.addWidget(self.eye, 0, Qt.AlignVCenter)
        self._hidden = False
        self.set_hidden(hidden)

    def is_hidden(self) -> bool:
        return self._hidden

    def set_hidden(self, on: bool) -> None:
        """The eye struck through and the name greyed while the track is hidden from the map."""
        self._hidden = bool(on)
        muted = self._muted_name or self._hidden
        for w, role in zip(self.columns, self._roles):
            name = role == "name" and not muted
            w.setFont(ty.ui("xs", "medium" if name else "normal", tabular=True))
            w.setStyleSheet(f"color:{self.t.foreground if name else self.t.muted_foreground};"
                            " background:transparent;")
        self.pct.setStyleSheet(f"color:{self.t.muted_foreground if muted else self.t.foreground};"
                               " background:transparent;")
        if self.eye is not None:
            self.eye.glyph.name = "eye-off" if on else "eye"
            self.eye.glyph.colour = self.t.muted_foreground if on else self.t.foreground
            self.eye.glyph.update()
            self._eye_tip.text = pv.EYE_TIPS[1 if on else 0]

    def set_selected(self, on: bool) -> None:
        self.selected = bool(on)
        self.update()

    def enterEvent(self, e) -> None:
        self.hovered = self.clickable
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.hovered = False
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self.clickable and e.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mouseReleaseEvent(e)

    def paintEvent(self, _e) -> None:
        if not (self.selected or self.hovered):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.muted if self.selected else self.t.accent))
        p.drawRoundedRect(QRectF(self.rect()), 5, 5)
        p.end()


class _Column(QWidget):
    """Cells or Tracks under a clicked spot: its worth, its contributors."""

    picked = Signal(int)
    hide_toggled = Signal(int)

    def __init__(self, t, unit: str, parent=None):
        super().__init__(parent)
        _clear(self)
        self.t = t
        self.unit = unit
        self.rows: list[_ListRow] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.title = label("CELLS" if unit == "cells" else "TRACKS",
                           ty.ui("xs", "medium"), t.muted_foreground)
        lay.addWidget(self.title)
        worth = QHBoxLayout()
        worth.setSpacing(6)
        self.value = label("", ty.value("lg", "semibold"), t.foreground)
        self.what = label("", ty.ui("xs"), t.foreground)
        self.of = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        worth.addWidget(self.value, 0, Qt.AlignBaseline)
        worth.addWidget(self.what, 0, Qt.AlignBaseline)
        worth.addStretch(1)
        worth.addWidget(self.of, 0, Qt.AlignBaseline)
        lay.addLayout(worth)
        self.meter = _Meter(t)
        lay.addWidget(self.meter)
        lay.addSpacing(8)
        lay.addWidget(label("CONTRIBUTORS", ty.ui("xs", "medium"), t.muted_foreground))
        self.bar = _ShareBar(t)
        lay.addWidget(self.bar)
        self._list = QVBoxLayout()
        self._list.setContentsMargins(0, 2, 0, 0)
        self._list.setSpacing(0)
        lay.addLayout(self._list)

    def set_spot(self, spot: dict) -> None:
        t = self.t
        shades, other = pv.ink_ramp(t.color_scheme)
        m = pv.meter(spot, self.unit)
        self.value.setText(m["value"])
        self.what.setText(m["what"])
        self.of.setText(m["of"])
        self.meter.set_fill(m["fill"])
        data = pv.contributor_rows(spot, self.unit)
        self.bar.set_shares([r["share"] for r in data["rows"]],
                            shades[:len(data["rows"])], other)
        for row in self.rows:
            discard(row)
        self.rows = []
        tracks = self.unit == "tracks"
        for i, r in enumerate(data["rows"]):
            cells = [(r["name"], "name"), (r["movie"], "where")]
            if tracks and r.get("cell"):
                cells.append((r["cell"], "where"))
            hidden = bool(r.get("hidden"))
            row = _ListRow(t, i, other if hidden else shades[i % len(shades)], cells, r["pct"],
                           clickable=tracks, eye=tracks, hidden=hidden)
            row.clicked.connect(self.picked.emit)
            row.hide_toggled.connect(self.hide_toggled.emit)
            self._list.addWidget(row)
            self.rows.append(row)
        if data["other"]:
            row = _ListRow(t, -1, other, [(data["other"], "name")], data["other_pct"],
                           clickable=False, muted_name=True)
            self._list.addWidget(row)
            self.rows.append(row)
        self._align()

    def _align(self) -> None:
        """Every row's k-th column as wide as the widest k-th column: the list reads as a
        table without being one."""
        named = [r for r in self.rows if r.index >= 0]
        n = max((len(r.columns) for r in named), default=0)
        for k in range(n):
            width = max(r.columns[k].sizeHint().width() for r in named if len(r.columns) > k)
            for r in named:
                if len(r.columns) > k:
                    r.columns[k].setFixedWidth(width)

    def set_selected(self, index: int) -> None:
        for row in self.rows:
            row.set_selected(row.index == index and self.unit == "tracks")

    def set_hidden(self, index: int, on: bool) -> None:
        """Track ``index``'s eye, struck through or open, before the redrawn map lists it again."""
        for row in self.rows:
            if row.index == index:
                row.set_hidden(on)


class _SpotSummary(QWidget):
    """Under the map: the clicked spot, or the hint before one is clicked."""

    track_picked = Signal(int)
    hide_toggled = Signal(int)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        _clear(self)
        self.t = t
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.hint = label(pv.SPOT_HINT, ty.ui("xs"), t.muted_foreground)
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.heading = label("", ty.ui("sm", "semibold"), t.foreground)
        self.position = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        #: The bin whose tracks are listed (a value map's whole value bin) and why a grey one is grey.
        self.notes = label("", ty.ui("xs"), t.muted_foreground)
        self.notes.setWordWrap(True)
        lay.addWidget(self.heading)
        lay.addWidget(self.position)
        lay.addWidget(self.notes)
        lay.addSpacing(6)
        columns = QHBoxLayout()
        columns.setSpacing(rem(1.5))
        self.cells = _Column(t, "cells")
        self.tracks = _Column(t, "tracks")
        self.tracks.picked.connect(self.track_picked.emit)
        self.tracks.hide_toggled.connect(self.hide_toggled.emit)
        columns.addWidget(self.cells, 1, Qt.AlignTop)
        columns.addWidget(self.tracks, 1, Qt.AlignTop)
        lay.addLayout(columns)
        self.n_eff_note = InfoLink(t, "What is n_eff?", N_EFF_EXPLAINED)
        lay.addWidget(self.n_eff_note)
        self.clear()

    def clear(self, hint: str = pv.SPOT_HINT) -> None:
        self.hint.setText(hint)
        self.hint.show()
        for w in (self.heading, self.position, self.notes, self.cells, self.tracks, self.n_eff_note):
            w.hide()

    def set_spot(self, spot: dict) -> None:
        self.hint.hide()
        self.heading.setText(pv.spot_heading(spot))
        self.position.setText(pv.spot_position(spot))
        notes = pv.spot_notes(spot)
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))
        self.cells.set_spot(spot)
        self.tracks.set_spot(spot)
        for w in (self.heading, self.position, self.cells, self.tracks, self.n_eff_note):
            w.show()

    def set_selected(self, index: int) -> None:
        self.tracks.set_selected(index)

    def set_hidden(self, index: int, on: bool) -> None:
        self.tracks.set_hidden(index, on)


class _JumpStrips(QWidget):
    """The spot's largest tracks, one row each: every jump a tick on one shared distance
    axis, the immobile band shaded once behind every row, the median a taller mark."""

    picked = Signal(int)

    ROW_H = 40
    AXIS_H = 34
    KEY_H = 22
    PAD_X = 8

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.rows: list[dict] = []
        self.floor_nm = None
        self.sigma_nm = None
        self.axis_max = 300.0
        self.selected = 0
        self.hovered = -1
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._f_title = ty.ui("xs", "semibold")
        self._f_small = ty.ui("xs", tabular=True)

    def set_rows(self, rows: list[dict], floor_nm, sigma_nm) -> None:
        self.rows, self.floor_nm, self.sigma_nm = list(rows), floor_nm, sigma_nm
        self.axis_max = pv.axis_max_nm(self.rows, floor_nm)
        self.selected, self.hovered = 0, -1
        self.updateGeometry()
        self.update()

    def set_selected(self, index: int) -> None:
        self.selected = index
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(rem(24), len(self.rows) * self.ROW_H + self.AXIS_H + self.KEY_H)

    def minimumSizeHint(self) -> QSize:
        return QSize(rem(18), self.sizeHint().height())

    # -- geometry ----------------------------------------------------------------------- #
    def _columns(self) -> tuple[float, float, float, float]:
        """(name column x, strip x0, strip x1, inside-text column right edge)."""
        fm_t = QFontMetrics(self._f_title)
        fm_s = QFontMetrics(self._f_small)
        name_w = max([fm_t.horizontalAdvance(r["title"]) for r in self.rows]
                     + [fm_s.horizontalAdvance(r["where"]) for r in self.rows] + [40])
        inside_w = max([fm_s.horizontalAdvance(r["inside_text"]) for r in self.rows] + [0])
        x_name = self.PAD_X + 8 + 10
        x0 = x_name + name_w + 14
        x1 = self.width() - self.PAD_X - inside_w - (14 if inside_w else 0)
        return x_name, x0, max(x0 + 40, x1), self.width() - self.PAD_X

    def _x(self, nm: float, x0: float, x1: float) -> float:
        return x0 + (x1 - x0) * min(max(nm, 0.0), self.axis_max) / self.axis_max

    def row_at(self, y: float) -> int:
        k = int(y // self.ROW_H)
        return k if 0 <= k < len(self.rows) and y >= 0 else -1

    # -- events ------------------------------------------------------------------------- #
    def mouseMoveEvent(self, e) -> None:
        pos = e.position() if hasattr(e, "position") else e.pos()
        k = self.row_at(pos.y())
        if k != self.hovered:
            self.hovered = k
            self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e) -> None:
        self.hovered = -1
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        pos = e.position() if hasattr(e, "position") else e.pos()
        k = self.row_at(pos.y())
        if e.button() == Qt.LeftButton and k >= 0:
            self.picked.emit(k)
        super().mouseReleaseEvent(e)

    # -- painting ----------------------------------------------------------------------- #
    def paintEvent(self, _e) -> None:
        if not self.rows:
            return
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        shades, _other = pv.ink_ramp(t.color_scheme)
        x_name, x0, x1, x_right = self._columns()
        rows_h = len(self.rows) * self.ROW_H
        fg = QColor(t.foreground)
        muted = QColor(t.muted_foreground)
        # the selected row and the hovered one, behind everything
        for k, colour in ((self.hovered, t.accent), (self.selected, None)):
            if 0 <= k < len(self.rows):
                r = QRectF(1, k * self.ROW_H + 1, self.width() - 2, self.ROW_H - 2)
                if colour is None:
                    p.setPen(QPen(QColor(t.border), 1))
                    p.setBrush(QColor(t.muted))
                else:
                    if k == self.selected:
                        continue
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(colour))
                p.drawRoundedRect(r, 7, 7)
        # the band, ONCE behind every row
        if self.floor_nm:
            xb = self._x(float(self.floor_nm), x0, x1)
            band = QColor(t.foreground)
            band.setAlpha(14)
            p.setPen(Qt.NoPen)
            p.setBrush(band)
            p.drawRect(QRectF(x0, 0, xb - x0, rows_h))
            dash = QPen(muted, 1, Qt.DashLine)
            dash.setDashPattern([3, 3])
            p.setPen(dash)
            p.drawLine(QPointF(xb, 0), QPointF(xb, rows_h))
        for k, row in enumerate(self.rows):
            top = k * self.ROW_H
            mid = top + self.ROW_H / 2.0
            # a track hidden from the map: listed, greyed, its jumps faint
            hidden = bool(row.get("hidden"))
            ink = muted if hidden else fg
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(_other if hidden else shades[k % len(shades)]))
            p.drawRoundedRect(QRectF(self.PAD_X + 2, mid - 4, 8, 8), 2, 2)
            p.setFont(self._f_title)
            p.setPen(ink)
            p.drawText(QRectF(x_name, top + 4, x0 - x_name, self.ROW_H / 2.0 - 2),
                       Qt.AlignLeft | Qt.AlignBottom, row["title"])
            p.setFont(self._f_small)
            p.setPen(muted)
            p.drawText(QRectF(x_name, mid, x0 - x_name, self.ROW_H / 2.0 - 4),
                       Qt.AlignLeft | Qt.AlignTop, row["where"])
            # the strip: a baseline, every jump, the median
            p.setPen(QPen(QColor(t.border), 1))
            p.drawLine(QPointF(x0, mid), QPointF(x1, mid))
            tick = QColor(t.foreground)
            tick.setAlpha(45 if hidden else 115)
            p.setPen(QPen(tick, 1.5))
            for j in row["jumps_nm"]:
                x = self._x(j, x0, x1)
                p.drawLine(QPointF(x, mid - 6), QPointF(x, mid + 6))
            if row["median_nm"] is not None:
                x = self._x(float(row["median_nm"]), x0, x1)
                p.setPen(QPen(ink, 2.5, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(x, mid - 11), QPointF(x, mid + 11))
            if row["inside_text"]:
                p.setFont(self._f_small)
                p.setPen(ink)
                p.drawText(QRectF(x1 + 8, top, x_right - x1 - 8, self.ROW_H),
                           Qt.AlignRight | Qt.AlignVCenter, row["inside_text"])
        # the shared axis
        y = rows_h + 2
        p.setPen(QPen(QColor(t.border), 1))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        p.setFont(self._f_small)
        p.setPen(muted)
        for v in pv.axis_ticks(self.axis_max):
            x = self._x(v, x0, x1)
            p.drawText(QRectF(x - 30, y + 2, 60, 14), Qt.AlignHCenter | Qt.AlignTop, f"{v:g}")
        p.drawText(QRectF(x0, y + 16, x1 - x0, 14), Qt.AlignRight | Qt.AlignTop,
                   "jump distance, nm")
        # the key
        ky = rows_h + self.AXIS_H + 4
        x = float(self.PAD_X + 2)
        tick = QColor(t.foreground)
        tick.setAlpha(115)
        items = [("one jump", tick, 1.5, 10), ("median", fg, 2.5, 14)]
        for text, colour, width, h in items:
            p.setPen(QPen(colour, width))
            p.drawLine(QPointF(x + 1, ky + 7 - h / 2), QPointF(x + 1, ky + 7 + h / 2))
            p.setPen(muted)
            w = QFontMetrics(self._f_small).horizontalAdvance(text)
            p.drawText(QRectF(x + 8, ky, w + 4, 14), Qt.AlignLeft | Qt.AlignVCenter, text)
            x += w + 26
        if self.floor_nm:
            band = QColor(t.foreground)
            band.setAlpha(14)
            p.setPen(Qt.NoPen)
            p.setBrush(band)
            p.drawRect(QRectF(x, ky + 2, 14, 10))
            dash = QPen(muted, 1, Qt.DashLine)
            dash.setDashPattern([2, 2])
            p.setPen(dash)
            p.drawLine(QPointF(x + 14, ky + 2), QPointF(x + 14, ky + 12))
            text = f"immobile band, {self.floor_nm:.0f} nm"
            if self.sigma_nm:
                text += f" at σ {self.sigma_nm:.0f} nm"
            p.setPen(muted)
            p.drawText(QRectF(x + 20, ky, 400, 14), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.end()


class _ClipCanvas(_ImageCanvas):
    """One track's frames, zoomed to its cell: the recorded frame (or a still picture),
    the cell's outline, the track's whole path faint, the travelled part highlighted and
    a ringed dot on the molecule while it is on the frame."""

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self.setCursor(Qt.ArrowCursor)
        self.setMinimumSize(rem(16), rem(18))
        self._message = pv.PICK_HINT
        self._clip_images: list[QImage] = []
        self._clip_first = 1
        self._movie = True
        self._others = False
        self._frame: int | None = None
        self._pick: dict | None = None

    def set_clip(self, clip: dict | None) -> None:
        self._clip_images = []
        if clip:
            self._clip_first = int(clip["first"])
            table = list(iv.grey_table())
            for raw in clip["frames"]:
                img = QImage(raw, clip["width"], clip["height"], clip["width"],
                             QImage.Format_Indexed8).copy()
                img.setColorTable(table)
                self._clip_images.append(img)
        self.update()

    def has_clip(self) -> bool:
        return bool(self._clip_images)

    def set_pick(self, track: dict | None) -> None:
        self._pick = track
        self.update()

    def set_frame(self, frame) -> None:
        self._frame = None if frame is None else int(frame)
        self.update()

    def set_movie(self, on: bool) -> None:
        self._movie = bool(on)
        self.update()

    def set_others(self, on: bool) -> None:
        self._others = bool(on)
        self.update()

    def background(self) -> QImage | None:
        """The picture behind the track now: the frame on screen, else the still."""
        if self._movie and self._clip_images and self._frame is not None:
            k = self._frame - self._clip_first
            if 0 <= k < len(self._clip_images):
                return self._clip_images[k]
        return self._image

    def mouseReleaseEvent(self, e) -> None:
        QWidget.mouseReleaseEvent(self, e)        # no cell or track picking in the player

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.t.muted))
        image = self.background()
        if self._payload is None or self._view is None or image is None:
            p.setPen(QColor(self.t.muted_foreground))
            p.setFont(ty.ui("sm"))
            p.drawText(self.rect().adjusted(24, 24, -24, -24),
                       Qt.AlignCenter | Qt.TextWordWrap, self._message or pv.PICK_HINT)
            p.end()
            return
        fit = self._fit()
        x0, y0, x1, y1 = self._shown_view()
        p.setClipRect(QRectF(self._to_screen(x0, y0, fit), self._to_screen(x1, y1, fit)))
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(QRectF(self._to_screen(-0.5, -0.5, fit),
                           self._to_screen(image.width() - 0.5, image.height() - 0.5, fit)),
                    image)
        p.setRenderHint(QPainter.Antialiasing, True)
        payload = self._payload
        for outline in payload["outlines"]:
            if outline["cell"] != self._cell:
                continue
            p.setPen(QPen(QColor(payload["outline_ink"]), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF([self._to_screen(x, y, fit) for x, y in outline["points"]]))
        if self._others and self._frame is not None:
            view = iv.frame_view(payload, self._frame, cell=self._cell)
            grey = QColor("#d4d4d8")
            grey.setAlpha(150)
            for trail in view["trails"]:
                if self._pick is not None and trail["uid"] == self._pick["uid"]:
                    continue
                self._paint_trail(p, fit, trail, grey, 1.2)
        if self._pick is not None:
            self._paint_pick(p, fit)
        self._paint_scale_bar(p, fit, (x0, y0, x1, y1))
        p.end()

    def _paint_pick(self, p, fit) -> None:
        points, frames = self._pick["points"], self._pick["frames"]
        faint = QColor("#ffffff")
        faint.setAlpha(80)
        if len(points) > 1:
            p.setPen(QPen(faint, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(self._path(points, fit))
        p.setPen(Qt.NoPen)
        dots = QColor("#ffffff")
        dots.setAlpha(110)
        p.setBrush(dots)
        for x, y in points:
            p.drawEllipse(self._to_screen(x, y, fit), 1.8, 1.8)
        if self._frame is None:
            return
        k, on = pv.travelled(frames, self._frame)
        if k >= 1:
            p.setPen(QPen(QColor(TRACK_INK), 2.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(self._path(points[:k + 1], fit))
        if on:
            centre = self._to_screen(points[k][0], points[k][1], fit)
            # a dark halo under the white ring: a molecule on its frame is itself white
            p.setPen(QPen(QColor(0, 0, 0, 170), 4.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(centre, 9.0, 9.0)
            p.setPen(QPen(QColor("#ffffff"), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(centre, 9.0, 9.0)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(TRACK_INK))
            p.drawEllipse(centre, 3.4, 3.4)


class _ClipPlayer(QWidget):
    """The caption, the canvas and the controls that keep the clip's time."""

    open_requested = Signal()
    napari_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        _clear(self)
        self.t = t
        self._tips: list = []
        self._payload = None
        self._row: dict | None = None
        self._lo, self._hi = 1, 1
        self._frame: float | None = None
        self._speed = pv.DEFAULT_SPEED
        self._loop = True
        self._syncing = False
        self._clock = QTimer(self)
        self._clock.setInterval(iv.TICK_MS)
        self._clock.timeout.connect(self._tick)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        cap = QHBoxLayout()
        cap.setSpacing(8)
        self.swatch = _Swatch("#18181B")
        self.title = label("", ty.ui("xs", "semibold"), t.foreground)
        self.where = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        cap.addWidget(self.swatch, 0, Qt.AlignVCenter)
        cap.addWidget(self.title)
        cap.addWidget(self.where)
        cap.addStretch(1)
        self.caption = QWidget()
        _clear(self.caption)
        self.caption.setLayout(cap)
        lay.addWidget(self.caption)
        self.canvas = _ClipCanvas(t)
        lay.addWidget(self.canvas, 1)

        clock = QHBoxLayout()
        clock.setSpacing(6)
        self.play_button = ToolButton(t, "play", "play", size=rem(1.75), rest=t.card)
        self.play_button.clicked.connect(lambda _k: self.play())
        self.pause_button = ToolButton(t, "pause", "pause", size=rem(1.75), rest=t.card)
        self.pause_button.clicked.connect(lambda _k: self.pause())
        self.loop_button = ToolButton(t, "loop", "rotate", size=rem(1.75), rest=t.card)
        self.loop_button.clicked.connect(lambda _k: self.set_loop(not self._loop))
        self._tips += [tooltip.attach(t, self.play_button, "Play the track's frames", side="top"),
                       tooltip.attach(t, self.pause_button, "Pause", side="top"),
                       tooltip.attach(t, self.loop_button, "Loop: start again after the last frame",
                                      side="top")]
        self.slider = Slider(t, 1, 2, 1)
        self.slider.setMinimumWidth(rem(4))
        self.slider.changed.connect(self._slid)
        self.counter = label("", ty.ui("xs", tabular=True), t.foreground)
        self.counter.setMinimumWidth(QFontMetrics(ty.ui("xs", tabular=True))
                                     .horizontalAdvance("frame 00,000") + 4)
        self.speed = Segmented(t, list(pv.SPEED_LABELS), current=pv.SPEEDS.index(self._speed),
                               size="sm", tips=list(pv.SPEED_TIPS))
        self.speed.changed.connect(lambda i: setattr(self, "_speed", pv.SPEEDS[i]))
        for w in (self.play_button, self.pause_button, self.loop_button):
            clock.addWidget(w)
        clock.addWidget(self.slider, 1)
        clock.addWidget(self.counter)
        clock.addWidget(self.speed)
        lay.addLayout(clock)

        # ONE ROW THAT WRAPS (`WrapRow`): the two hand-on buttons beside the background where the card is wide, on a
        # line of their own where it is narrow -- a row of their own always made the card too tall for a laptop
        more = WrapRow(spacing=10, line_spacing=6)
        self.background = Segmented(t, list(pv.BACKGROUNDS), current=0, size="sm",
                                    tips=list(pv.BACKGROUND_TIPS))
        self.background.changed.connect(lambda i: self.canvas.set_movie(i == 0))
        self.others = Checkbox(t, checked=False)
        self.others.toggled.connect(self.canvas.set_others)
        others_label = label("Other tracks in this cell", ty.ui("xs"), t.muted_foreground)
        self.open_button = Button(t, "Open in Images & tracks", variant="ghost", size="small")
        self.open_button.clicked.connect(self.open_requested.emit)
        self._tips.append(tooltip.attach(
            t, self.open_button, "The whole movie at this track, with every other track",
            side="top"))
        # BOTH VIEWERS, by ruling ("both as options would be best"): napari loads the condition from its folder
        self.napari_button = Button(t, "Open in napari", variant="ghost", size="small")
        self.napari_button.clicked.connect(self.napari_requested.emit)
        self._tips.append(tooltip.attach(
            t, self.napari_button, "The condition in napari, opened at this track (it loads the folder first)",
            side="top"))
        more.addWidget(self.background)
        others = QHBoxLayout()                       # the box and its words stay together
        others.setSpacing(6)
        others.addWidget(self.others, 0, Qt.AlignVCenter)
        others.addWidget(others_label, 0, Qt.AlignVCenter)
        more.addLayout(others)
        more.addStretch(1)
        hand_on = QHBoxLayout()                      # the two viewers stay together
        hand_on.setSpacing(6)
        hand_on.addWidget(self.open_button)
        hand_on.addWidget(self.napari_button)
        more.addLayout(hand_on)
        lay.addLayout(more)
        self.message = label("", ty.ui("xs"), t.muted_foreground)
        self.message.setWordWrap(True)
        self.message.hide()
        lay.addWidget(self.message)
        self.set_loop(True)
        self.clear()

    # -- state ------------------------------------------------------------------------ #
    def clear(self, hint: str = pv.PICK_HINT) -> None:
        self._clock.stop()
        self._payload, self._row, self._frame = None, None, None
        self.caption.hide()
        self.message.hide()
        self.canvas.set_pick(None)
        self.canvas.set_clip(None)
        self.canvas.set_message(hint)
        self._sync()

    def show_loading(self, row: dict, colour: str) -> None:
        self._clock.stop()
        self._show_caption(row, colour)
        self.message.hide()
        self._payload, self._frame = None, None
        self.canvas.set_pick(None)
        self.canvas.set_clip(None)
        self.canvas.set_message(pv.READING)
        self._sync()

    def show_error(self, row: dict, colour: str, text: str) -> None:
        self._clock.stop()
        self._show_caption(row, colour)
        self._payload = None
        self.canvas.set_message(text)
        self._sync()

    def _show_caption(self, row: dict, colour: str) -> None:
        self._row = row
        title, rest = pv.caption(row)
        self.title.setText(title)
        self.where.setText(rest)
        self.swatch.set_colour(colour)
        self.caption.show()

    def show_clip(self, row: dict, colour: str, result: dict) -> None:
        """A loaded clip: zoom to the track's cell and play it, looping."""
        payload = result["payload"]
        self._show_caption(row, colour)
        self._payload = payload
        self._lo, self._hi = result["lo"], result["hi"]
        self.canvas.set_view(payload, view=result["view"], cell=result["cell"],
                             show_outlines=True, show_tracks=False)
        self.canvas.set_clip(result.get("clip"))
        self.canvas.set_pick(result["track"])
        if result.get("clip_error"):
            self.message.setText(f"The recorded frames could not be shown: "
                                 f"{result['clip_error']} The still picture is shown instead.")
            self.message.show()
        else:
            self.message.hide()
        self.background.setEnabled(self.canvas.has_clip())
        if not self.canvas.has_clip() and self.background.current() == 0:
            self.background.select(1)
            self.canvas.set_movie(False)
        self._frame = float(self._lo)
        self._clock.start()
        self._sync()

    def set_loop(self, on: bool) -> None:
        self._loop = bool(on)
        self.loop_button.set_active(self._loop)

    def playing(self) -> bool:
        return self._clock.isActive()

    def frame(self):
        return None if self._frame is None else int(self._frame)

    def play(self) -> None:
        if self._payload is None:
            return
        if self._frame is None or int(self._frame) >= self._hi:
            self._frame = float(self._lo)
        self._clock.start()
        self._sync()

    def pause(self) -> None:
        self._clock.stop()
        self._sync()

    def _tick(self) -> None:
        if self._payload is None or self._frame is None:
            self._clock.stop()
            return
        step = self._speed * iv.TICK_MS / max(1e-6, self._payload["time"]["frame_interval_ms"])
        self._frame, going = pv.advance(self._frame, step, self._lo, self._hi, self._loop)
        if not going:
            self._clock.stop()
        self._sync()

    def _slid(self, value: int) -> None:
        if self._syncing or self._payload is None:
            return
        self._frame = float(value)
        self._sync()

    def _sync(self) -> None:
        has = self._payload is not None
        playing = self._clock.isActive()
        self.play_button.setVisible(not playing)
        self.pause_button.setVisible(playing)
        for w in (self.play_button, self.pause_button, self.loop_button, self.slider,
                  self.speed, self.others):
            w.setEnabled(has)
        # a picked track can be handed on even where its frames could not be read
        for w in (self.open_button, self.napari_button):
            w.setEnabled(self._row is not None)
        if not has:
            self.counter.setText("")
            return
        self._syncing = True
        try:
            self.slider.lo, self.slider.hi = self._lo, self._hi
            self.slider.set_value(int(self._frame) if self._frame is not None else self._lo)
            self.slider.update()
        finally:
            self._syncing = False
        self.canvas.set_frame(self._frame)
        self.counter.setText(pv.frame_text(self._frame, self._lo, self._hi))

    def hideEvent(self, e) -> None:
        self._clock.stop()
        self._sync()
        super().hideEvent(e)


class _Signals(QObject):
    loaded = Signal(int, object, str)


class _ClipJob(QRunnable):
    """Everything the player needs for one track, read off the GUI thread: which movie
    and cell it is (`trackviewer.track_target`), the movie's payload (outlines, tracks,
    still picture) and the track's recorded frames. A stack that cannot be read is said,
    and the still is shown; anything else fails the job with its traceback."""

    def __init__(self, generation, project, condition, uid, signals):
        super().__init__()
        self.args = (generation, project, condition, uid)
        self.signals = signals

    def run(self) -> None:
        generation, project, condition, uid = self.args
        try:
            result, error = pv.clip_result(project, condition, uid), ""
        except Exception:                                  # noqa: BLE001 - said in the player
            result, error = None, traceback.format_exc(limit=4)
        try:
            self.signals.loaded.emit(generation, result, error)
        except RuntimeError:                               # the page closed meanwhile
            pass


class _Columns(QLayout):
    """Two cards side by side when there is room, the right under the left when not.
    Sized here, in `setGeometry` (never in a `resizeEvent`).

    FITTED TO THE WINDOW'S HEIGHT TOO (asked 14 Oct 2026: with the sidebar collapsed the
    page scrolled). The maps keep the cell's proportions, so a wider map card is a taller
    one; side by side, the map card is half the page but never wider than the widest width
    at which it still fits `fit_height()` (bisection over its height-for-width, which grows
    with the width), never narrower than `LEFT_MIN_PX`; the right card takes the rest."""

    def __init__(self, left: QWidget, right: QWidget, gap: int, parent=None, *,
                 below: QWidget | None = None):
        super().__init__(parent)
        #: The height the page may use (the scroll area's viewport), or None: no fitting.
        self.fit_height = None
        self._items = []
        self.gap = gap
        self.left, self.right = left, right
        #: A card the page's full width under both (the profiles), while it is shown; the two
        #: cards still fit the window's height, so it is scrolled to.
        self.below = below
        self.addWidget(left)
        self.addWidget(right)
        if below is not None:
            self.addWidget(below)

    def _below_h(self, width: int) -> int:
        if self.below is None or self.below.isHidden():
            return 0
        return self.gap + self._h(self.below, width)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def side_by_side(self, width: int) -> bool:
        return width >= SIDE_BY_SIDE_MIN

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        a, b = self.left.minimumSizeHint(), self.right.minimumSizeHint()
        return QSize(max(a.width(), b.width()), max(a.height(), b.height()))

    def hasHeightForWidth(self) -> bool:
        return True

    def left_width(self, width: int) -> int:
        """The map card's width side by side: half the page, or less to fit the height."""
        half = (width - self.gap) // 2
        limit = self.fit_height() if self.fit_height is not None else None
        if not limit or limit <= 0 or self._h(self.left, half) <= limit:
            return half
        lo, hi = min(LEFT_MIN_PX, half), half
        if self._h(self.left, lo) > limit:
            return lo                           # too short even at the floor: scroll
        while hi - lo > 2:
            mid = (lo + hi) // 2
            if self._h(self.left, mid) <= limit:
                lo = mid
            else:
                hi = mid
        return lo

    def heightForWidth(self, width: int) -> int:
        if self.side_by_side(width):
            w = self.left_width(width)
            return (max(self._h(self.left, w), self._h(self.right, width - w - self.gap))
                    + self._below_h(width))
        return (self._h(self.left, width) + self.gap + self._h(self.right, width)
                + self._below_h(width))

    @staticmethod
    def _h(widget: QWidget, width: int) -> int:
        lay = widget.layout()
        if lay is not None and lay.hasHeightForWidth():
            return lay.totalHeightForWidth(width)
        return max(widget.sizeHint().height(), widget.minimumSizeHint().height())

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        if self.side_by_side(rect.width()):
            w = self.left_width(rect.width())
            hl, hr = self._h(self.left, w), self._h(self.right, rect.width() - w - self.gap)
            # the right card as tall as the left: its player takes the extra height
            self.left.setGeometry(QRect(rect.x(), rect.y(), w, hl))
            self.right.setGeometry(QRect(rect.x() + w + self.gap, rect.y(),
                                         rect.width() - w - self.gap, max(hl, hr)))
            bottom = rect.y() + max(hl, hr)
        else:
            hl = self._h(self.left, rect.width())
            hr = self._h(self.right, rect.width())
            self.left.setGeometry(QRect(rect.x(), rect.y(), rect.width(), hl))
            self.right.setGeometry(QRect(rect.x(), rect.y() + hl + self.gap, rect.width(), hr))
            bottom = rect.y() + hl + self.gap + hr
        if self.below is not None and not self.below.isHidden():
            self.below.setGeometry(QRect(rect.x(), bottom + self.gap, rect.width(),
                                         self._h(self.below, rect.width())))


class _Loupe(QWidget):
    """A round magnifying lens over the Spatial Map (asked 14 Oct 2026: "a magnifying glass
    ... shows the grids underneath, bigger"). It follows the pointer over any capsule of the
    page's map, offset below-right (flipped near the page's edges) so it never covers what
    is pointed at, nor the value tooltip above the bin. It shows the map EXACTLY as drawn --
    the chart's own drawing at the lens's magnification -- with the bin under the pointer
    outlined.

    IT GLIDES (asked 14 Oct 2026: "moves disruptedly"). Measured before: each move repainted
    what the lens uncovered on the page (~15 paints a move) and redrew every bin of the map,
    and a 1 px pointer step jumped the 4x picture 4 px. So:
    - the lens is its OWN small window (`Qt.ToolTip`, like the styled tooltips' `top_level`):
      the window system moves it and nothing under it is repainted;
    - the magnified map is drawn ONCE into a picture (`paint_map` at the zoom, at the screen's
      pixel ratio, so still sharp) and only re-drawn when the map's `content_version`, the
      zoom or the size changes; a move only shifts which part of it shows;
    - the shown position GLIDES to the pointer, `LOUPE_GLIDE` of the way per ~16 ms frame,
      so steps of whole pixels become continuous motion, and bursts of mouse events paint one
      frame per screen refresh at most."""

    def __init__(self, t, page):
        # The page is its PARENT (so it is destroyed with the page, never after the
        # application -- a parentless window crashed the interpreter's shutdown, measured 3
        # in 8 runs) and the flags make it its own window all the same.
        super().__init__(page, Qt.ToolTip | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.t = t
        self._page = page
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        #: THE LENS FOLLOWS THE LAST MOVE (asked 14 Oct 2026, no modes): by the pointer after
        #: a mouse move, on the selected bin after an arrow key. True while it is the keys'.
        self._by_keys = False
        self._chart = None
        #: The pointer (the target) and what is shown (gliding to it), in the chart's pixels.
        self._at = QPointF()
        self._shown = QPointF()
        self._bin = None
        self._zoom = 1.0
        #: (key, picture): the chart drawn whole at the zoom, re-drawn when the key changes.
        self._cache = (None, None)
        self._glide = QTimer(self)
        self._glide.setInterval(pv.LOUPE_FRAME_MS)
        self._glide.timeout.connect(self._step)
        self.setFixedSize(pv.LOUPE_SIZE + 8, pv.LOUPE_SIZE + 8)
        self.hide()

    def follow(self, chart, pos) -> None:
        self._by_keys = False
        if not chart._grid:
            self.hide()
            return
        if not chart._capsule(chart._box()).contains(QPointF(pos)):
            self.hide()
            return
        spot = chart.bin_at(pos)
        if spot is None:
            self.hide()
            return
        self._bin = chart._bin_rect(*spot)
        # THE LENS DECIDES FOR ITSELF (asked 14 Oct 2026): offered only while a bin is smaller
        # on screen than a comfortable target, and enlarged just enough to make it one -- so
        # 100 nm bins (~10 px) get it, 345 nm bins (~25 px) do not, and a resize decides again.
        # Logical pixels: at 150 % Windows scaling a bin LOOKS as big as at 100 %.
        bin_px = pv.bin_side(self._bin.width(), self._bin.height())
        if not pv.loupe_wanted(bin_px):
            self.hide()
            return
        zoom = pv.loupe_zoom(bin_px)
        fresh = not self.isVisible() or chart is not self._chart or zoom != self._zoom
        self._zoom = zoom
        self._chart, self._at = chart, QPointF(pos)
        if fresh:
            self._shown = QPointF(pos)          # a new lens starts where it is, no slide in
        page = self._page
        at = chart.mapTo(page, QPointF(pos).toPoint())
        x, y = at.x() + pv.LOUPE_OFFSET, at.y() + pv.LOUPE_OFFSET
        if x + self.width() > page.width():
            x = at.x() - pv.LOUPE_OFFSET - self.width()
        if y + self.height() > page.height():
            y = at.y() - pv.LOUPE_OFFSET - self.height()
        self.move(page.mapToGlobal(QPoint(max(0, x), max(0, y))))
        if not self.isVisible():
            self.show()
        if not self._glide.isActive():
            self._glide.start()
        self.update()

    def _step(self) -> None:
        """One frame: the shown position a step closer to the pointer; stop once there."""
        d = self._at - self._shown
        if abs(d.x()) < 0.02 and abs(d.y()) < 0.02:
            self._shown = QPointF(self._at)
            self._glide.stop()
        else:
            self._shown = self._shown + d * pv.LOUPE_GLIDE
        self.update()

    def hide(self) -> None:
        self._glide.stop()
        super().hide()

    def leave(self, _chart=None) -> None:
        if not self._by_keys:               # the keys' lens stays until the mouse moves
            self.hide()

    def clear(self) -> None:
        self._by_keys = False
        self.hide()

    def show_at_bin(self, chart, ix: int, iy: int) -> None:
        """The lens on a bin the KEYBOARD selected: the same lens, centred on that bin and
        placed beside it as it is beside the pointer, gliding there from the last bin. It
        stays until the mouse moves."""
        self.follow(chart, chart._bin_rect(ix, iy).center())
        self._by_keys = self.isVisible()

    def _picture(self):
        """The chart drawn whole at the zoom (cached; see the class docstring)."""
        chart, k = self._chart, self._zoom
        ratio = self.devicePixelRatioF()
        key = (id(chart), chart.content_version, k, chart.width(), chart.height(), ratio)
        if self._cache[0] == key:
            return self._cache[1]
        w, h = chart.width() * k, chart.height() * k
        image = QImage(max(1, round(w * ratio)), max(1, round(h * ratio)),
                       QImage.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(QColor(self.t.card))
        q = QPainter(image)
        box = chart._box()
        chart.paint_map(q, QRectF(box.x() * k, box.y() * k, box.width() * k, box.height() * k),
                        hover=False, faded=True)
        q.end()
        self._cache = (key, image)
        return image

    def paintEvent(self, _e) -> None:
        chart = self._chart
        if chart is None or not chart._grid:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        d = float(pv.LOUPE_SIZE)
        lens = QRectF(4, 4, d, d)
        # a soft shadow, then the lens
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawEllipse(lens.translated(0, 2))
        round_ = QPainterPath()
        round_.addEllipse(lens)
        p.save()
        p.setClipPath(round_)
        p.fillRect(lens, QColor(self.t.card))
        k = self._zoom
        cx, cy = lens.center().x(), lens.center().y()
        # the cached picture, slid so the SHOWN position sits at the lens's centre
        p.drawImage(QPointF(cx - self._shown.x() * k, cy - self._shown.y() * k), self._picture())
        if self._bin is not None:
            r = QRectF(cx + (self._bin.x() - self._shown.x()) * k,
                       cy + (self._bin.y() - self._shown.y()) * k,
                       self._bin.width() * k, self._bin.height() * k)
            for colour, width, inset in (("#111111", 3.0, 1.5), ("#ffffff", 1.5, 1.5)):
                pen = QPen(QColor(colour), width)
                pen.setJoinStyle(Qt.MiterJoin)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(r.adjusted(inset, inset, -inset, -inset))
        p.restore()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 120), 3.0))
        p.drawEllipse(lens)
        p.setPen(QPen(QColor("#ffffff"), 1.6))
        p.drawEllipse(lens)
        p.end()


class SpotPage(QWidget):
    """The replicate page: one condition's payload with its `days`; plus the project, for
    the player.

    **Nothing is computed here.** Every replicate's maps arrive inside the condition's own
    payload (`payload["days"]`, built by `spatial_map_view` in the same job as the
    condition's maps), drawn under the condition's OWN fit -- one rule for every replicate,
    so a difference between two rows is where that replicate's jumps fell, never a different
    rule. The grid's rows are the replicates, so the colour scale is one per population
    across them (a fainter replicate reads paler). Only one condition's replicates are ever
    on this page (Rule 0: a replicate is a within-condition ordinal, never paired across
    conditions)."""

    #: A track was handed to a viewer: (condition, project uid, "fast" | "napari"). "fast" is
    #: Images & tracks, at the track (`WorkspaceWindow._open_track`).
    track_requested = Signal(str, str, str)
    #: A track's eye was clicked: (condition, project uid, hidden). The window redraws that
    #: condition's map without the hidden ones (`spatial_map_view.compute_spatial_map(hidden=)`)
    #: and hands it back through `refresh_condition`.
    hidden_changed = Signal(str, str, bool)

    def __init__(self, t, parent=None, *, threads: QThreadPool | None = None):
        super().__init__(parent)
        _clear(self)
        self.t = t
        self._project = None
        self._condition: str | None = None
        #: (day key, kind, ix, iy) of the spot on screen -- a value map's WHOLE value bin, by
        #: its first small bin -- so a second click on it clears.
        self._spot_key: tuple | None = None
        self._spot: dict | None = None
        self._rows: list[dict] = []
        self._overlay: dict | None = None
        self._selected = 0
        #: condition -> {uid: the track's record, with the spot it was hidden from}: a hidden
        #: track is no longer in its spot once the map is redrawn without it, so the page keeps
        #: listing it there, struck through, and one click on its eye brings it back.
        self._hidden: dict[str, dict[str, dict]] = {}
        #: How many small bins a side the open spot spans (a value map's value bin), for the keys.
        self._span = 1
        self._threads = threads or QThreadPool.globalInstance()
        self._generation = 0
        self._cache: dict = {}
        self._signals = _Signals()
        self._signals.loaded.connect(self._loaded)

        # LEFT: where. The map and, under a hairline, the clicked spot -- in ONE card (the
        # matrix is the card; the summary joins its layout, so no card nests in a card).
        self.matrix = SpatialMapMatrix(t, row_unit="day")
        self.matrix.spot_clicked.connect(self._on_spot)
        self.summary = _SpotSummary(t)
        self.summary.track_picked.connect(self.select_track)
        self.summary.hide_toggled.connect(self._toggle_hidden)
        self.matrix.layout().addWidget(_Hairline(t))
        self.matrix.layout().addWidget(self.summary)

        # RIGHT: what it looks like. The strips, then the player.
        self.right = Card(t)
        right = QVBoxLayout(self.right)
        right.setContentsMargins(CARD_PAD, CARD_PAD, CARD_PAD, CARD_PAD)
        right.setSpacing(CARD_GAP)
        self.strips_title = label(pv.STRIPS_TITLE,
                                  ty.ui("xs", "medium"), t.muted_foreground)
        right.addWidget(self.strips_title)
        self.strips = _JumpStrips(t)
        self.strips.picked.connect(self.select_track)
        right.addWidget(self.strips)
        self.right_hint = label(pv.SPOT_HINT, ty.ui("xs"), t.muted_foreground)
        self.right_hint.setWordWrap(True)
        right.addWidget(self.right_hint)
        right.addWidget(_Hairline(t))
        self.player = _ClipPlayer(t)
        self.player.open_requested.connect(lambda: self._open_requested("fast"))
        self.player.napari_requested.connect(lambda: self._open_requested("napari"))
        right.addWidget(self.player, 1)

        # UNDER BOTH: where the jumps sit, this condition and each replicate (17 Oct 2026)
        from .spatial_profiles import ProfilesCard

        self.profiles = ProfilesCard(t, "Where the jumps sit · this condition and each replicate")
        self.profiles.hide()

        page = QWidget()
        _clear(page)
        self._columns = _Columns(self.matrix, self.right, CARD_GAP, below=self.profiles)
        page.setLayout(self._columns)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = _scrolled(page)
        self._scroll = scroll
        self._columns.fit_height = lambda: scroll.viewport().height()
        outer.addWidget(scroll)
        #: The hover lens over the map (the page's own child, above the cards).
        self.loupe = _Loupe(t, self)
        #: True while the arrow keys lead and the pointer is hidden; the chart they hover.
        self._pointer_hidden = False
        self._keyed = None
        self._rest = QPoint()
        QApplication.instance().applicationStateChanged.connect(self._app_state)
        # THE KEYBOARD MOVES THE SPOT (asked 14 Oct 2026: not every hand can land on a 9 px
        # bin): after a rough click, the arrow keys step the selection one bin at a time.
        self.setFocusPolicy(Qt.StrongFocus)
        self.matrix.set_loupe(self.loupe)
        self._clear_spot()

    # -- inputs ------------------------------------------------------------------------ #
    def set_project(self, project) -> None:
        self._project = project
        self._cache.clear()

    def show_condition(self, condition: str, payload: dict, *, name: str | None = None,
                       colour: str | None = None, kind: str | None = None) -> None:
        method, n_pop, ranges = payload["method"], payload.get("n_pop") or 2, payload.get("ranges")
        if method in PROFILE_METHODS and payload.get("profiles"):
            self.profiles.set_source(lambda k: day_profile_payload(payload, k, name or condition),
                                     kinds_for(method, n_pop, ranges),
                                     panel_titles_for(method, n_pop, ranges),
                                     {"condition": colour or "#2b6cb0"}, select=kind)
        else:
            self.profiles.hide()
        days = payload["days"]
        keys = [str(d["day"]) for d in days]
        self.matrix.set_method(method, n_pop, ranges)
        self.matrix.set_conditions(keys, {str(d["day"]): day_label(d["day"]) for d in days})
        self.matrix.clear_payloads()
        for key, day in zip(keys, days):
            self.matrix.set_payload(key, day)
        self._condition = condition
        self._clear_spot()
        # a payload from before the spot record existed cannot be clicked: say so rather than
        # offering a click that does nothing
        if not any("spots" in d for d in days):
            self.summary.clear(pv.no_spots_text(payload))

    def condition(self) -> str | None:
        return self._condition

    def refresh_condition(self, condition: str, payload: dict) -> None:
        """The condition's map came back without (or with) a hidden track: redraw its
        replicates and reopen the spot that was open, so the card stays where it was."""
        if condition != self._condition:
            return
        spot = self._spot_key
        self.show_condition(condition, payload, kind=self.profiles.marked_kind())
        if spot is not None:
            self._on_spot(*spot)

    def hidden_uids(self, condition: str) -> list[str]:
        """The tracks of ``condition`` hidden from the map on this page."""
        return sorted(self._hidden.get(condition, {}))

    def _toggle_hidden(self, index: int) -> None:
        """Hide track ``index`` from the map, or show it again; the window redraws the map."""
        if not 0 <= index < len(self._rows) or self._condition is None or self._spot is None:
            return
        track = self._spot["tracks"][index]
        hidden = not bool(self._rows[index].get("hidden"))
        mine = self._hidden.setdefault(self._condition, {})
        if hidden:
            mine[track["uid"]] = {**track, "hidden": True, "spot": self._spot_key}
        else:
            mine.pop(track["uid"], None)
        # marked at once, before the redrawn map lists the spot again
        tracks = list(self._spot["tracks"])
        tracks[index] = {**track, "hidden": hidden}
        self._spot = {**self._spot, "tracks": tracks}
        self._rows = pv.strip_rows(self._spot)
        self.summary.set_hidden(index, hidden)
        self.strips.set_rows(self._rows, self._spot.get("floor_nm"), self._spot.get("sigma_nm"))
        self.strips.set_selected(self._selected)
        self.hidden_changed.emit(self._condition, track["uid"], hidden)

    def hideEvent(self, e) -> None:
        self._away()                        # its own window: it does not hide with the page
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().hideEvent(e)

    # -- the spot ---------------------------------------------------------------------- #
    def _clear_spot(self) -> None:
        self._spot_key, self._spot, self._rows, self._overlay, self._span = None, None, [], None, 1
        self.matrix.set_overlay("", "", None)
        self.matrix.set_tip(pv.TIP_SELECT)
        if hasattr(self, "loupe"):
            self.loupe.clear()
            self._pointer_leads()
        self.summary.clear()
        self.strips.set_rows([], None, None)
        self.strips.hide()
        self.strips_title.hide()
        self.right_hint.show()
        self.player.clear()

    def _on_spot(self, key: str, kind: str, ix: int, iy: int) -> None:
        payload = self.matrix.payload(key)
        spot = None if payload is None else spot_contributors(payload, kind, ix, iy,
                                                              top=SPOT_TOP_TRACKS)
        if spot is None:
            return                  # outside the capsule: nothing is drawn there to name
        # A value map's spot is its whole value bin: the ring sits on that bin's centre and
        # spans it, and a second click anywhere in it clears (`spot_contributors`' bin_*).
        span = int(spot.get("bin_factor", 1))
        anchor = (int(spot.get("bin_ix", ix)), int(spot.get("bin_iy", iy)))
        if self._spot_key == (key, kind, *anchor):
            self._clear_spot()
            return
        mine = self._hidden.get(self._condition or "", {})
        here = [tr for tr in mine.values() if tr.get("spot") == (key, kind, *anchor)]
        spot = pv.listed(spot, here, mine)
        self._spot_key, self._spot, self._span = (key, kind, *anchor), spot, span
        self._rows = pv.strip_rows(spot)
        self._overlay = {"paths": [tr["path"] for tr in spot["tracks"] if not tr.get("hidden")],
                         "bin": (anchor[0] + (span - 1) / 2.0, anchor[1] + (span - 1) / 2.0),
                         "span": span, "selected": 0}
        self.matrix.set_overlay(key, kind, self._overlay)
        self.matrix.set_tip(pv.TIP_MOVE)
        self.summary.set_spot(spot)
        self.strips.set_rows(self._rows, spot.get("floor_nm"), spot.get("sigma_nm"))
        has = bool(self._rows)
        self.strips.setVisible(has)
        self.strips_title.setVisible(has)
        self.right_hint.setVisible(not has)
        if has:
            self.select_track(0)
        else:
            self.player.clear("No track of this map reaches this spot.")
        self.setFocus(Qt.OtherFocusReason)        # the arrow keys move it from here

    _STEPS = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, 1), Qt.Key_Down: (0, -1)}

    # -- who leads: the pointer, or the arrow keys ------------------------------------ #
    def _charts(self):
        return [c for row in self.matrix._rows.values() for c in row["charts"].values()]

    def _keys_lead(self, chart, ix: int, iy: int) -> None:
        """THE KEYS TAKE THE POINTER'S PLACE (asked 14 Oct 2026): the pointer is hidden until
        the mouse next moves (where it was; the macOS hide-while-typing rule), and the selected
        bin is the map's hover -- its tooltip, its outline, the tracks faded -- and the lens's."""
        for other in self._charts():
            if other is not chart:
                other.clear_hover()
        chart.hover_bin(ix, iy)
        self._keyed = chart
        self.loupe.show_at_bin(chart, ix, iy)
        if not self._pointer_hidden:
            QApplication.setOverrideCursor(Qt.BlankCursor)
            self._pointer_hidden = True
            # where the pointer rests: a "move" that did not move it (the window system's, when
            # something moves under a still pointer) must not hand the lead back
            self._rest = QCursor.pos()

    def _pointer_leads(self) -> None:
        """The mouse moved (or the app lost focus): the pointer back where it was, and the
        keys' hover handed back -- the chart under the pointer takes it from its own move."""
        if self._pointer_hidden:
            QApplication.restoreOverrideCursor()
            self._pointer_hidden = False
        if self._keyed is not None:
            try:
                self._keyed.clear_hover()
            except RuntimeError:                # rebuilt meanwhile
                pass
            self._keyed = None

    def eventFilter(self, obj, ev) -> bool:
        """Watches the whole application while the page is shown: a mouse move hands the lead
        back to the pointer; leaving the application or the window never leaves the lens up."""
        kind = ev.type()
        if kind == QEvent.MouseMove and (self._pointer_hidden or self._keyed is not None):
            if QCursor.pos() != self._rest:
                self._pointer_leads()
        elif kind in (QEvent.WindowDeactivate, QEvent.ApplicationDeactivate) and \
                (obj is self.window() or kind == QEvent.ApplicationDeactivate):
            self._away()
        return False

    def _away(self) -> None:
        self.loupe.clear()
        self._pointer_leads()

    def _app_state(self, state) -> None:
        if state != Qt.ApplicationActive:
            self._away()

    def showEvent(self, e) -> None:
        QApplication.instance().installEventFilter(self)
        super().showEvent(e)

    def keyPressEvent(self, e) -> None:
        key = e.key()
        if key == Qt.Key_Escape and self._spot_key is not None:
            self._clear_spot()                   # one meaning: no mode to leave
            return
        if key in self._STEPS and self._spot_key is not None:
            map_key, kind, ix, iy = self._spot_key
            payload = self.matrix.payload(map_key)
            # a value map steps a whole value bin, so a step always reaches the next spot
            dx, dy = self._STEPS[key]
            nxt = None if payload is None else pv.step_bin(payload, ix, iy, dx * self._span,
                                                           dy * self._span)
            if nxt is not None:
                self._on_spot(map_key, kind, *nxt)
                self._keys_lead(self.matrix.chart(map_key, kind), *nxt)
            return
        super().keyPressEvent(e)

    def select_track(self, index: int) -> None:
        """One selection for the strips, the Tracks list, the map and the player."""
        if not 0 <= index < len(self._rows):
            return
        self._selected = index
        self.strips.set_selected(index)
        self.summary.set_selected(index)
        if self._overlay is not None and self._spot_key is not None:
            # a hidden track is not on the redrawn map, so no path is thickened for it
            shown = len(self._overlay["paths"])
            self._overlay = {**self._overlay, "selected": index if index < shown else -1}
            self.matrix.set_overlay(self._spot_key[0], self._spot_key[1], self._overlay)
        self._load(self._rows[index])

    def selected_uid(self) -> str | None:
        return self._rows[self._selected]["uid"] if self._rows else None

    def _colour(self, row: dict) -> str:
        shades, _ = pv.ink_ramp(self.t.color_scheme)
        return shades[row["rank"] % len(shades)]

    # -- the player's input ------------------------------------------------------------- #
    def _load(self, row: dict) -> None:
        self._generation += 1
        if self._project is None or self._condition is None:
            self.player.show_error(row, self._colour(row),
                                   "The project is not loaded, so the track cannot be played.")
            return
        cached = self._cache.get((self._condition, row["uid"]))
        if cached is not None:
            self.player.show_clip(row, self._colour(row), cached)
            return
        self.player.show_loading(row, self._colour(row))
        self._threads.start(_ClipJob(self._generation, self._project, self._condition,
                                     row["uid"], self._signals))

    def _loaded(self, generation: int, result, error: str) -> None:
        if generation != self._generation or not self._rows:
            return
        row = self._rows[self._selected]
        if result is None:
            self.player.show_error(row, self._colour(row),
                                   "This track could not be played:\n"
                                   + error.strip().splitlines()[-1])
            return
        self._cache[(self._condition, row["uid"])] = result
        self.player.show_clip(row, self._colour(row), result)

    def _open_requested(self, viewer: str) -> None:
        uid = self.selected_uid()
        if uid is not None and self._condition is not None:
            self.track_requested.emit(self._condition, uid, viewer)

    def show_track_message(self, text: str) -> None:
        """What became of a viewer request or a redraw -- said under the player, never in a
        modal dialog."""
        self.player.message.setText(text)
        self.player.message.setVisible(bool(text))

    # -- display settings, as the current page takes them -------------------------------- #
    def set_colormap(self, name: str) -> None:
        self.matrix.set_colormap(name)

    def set_rendering(self, mode: str) -> None:
        self.matrix.set_rendering(mode)

    def set_scale(self, level: str) -> None:
        self.matrix.set_scale(level)

    def set_panel_visible(self, kind: str, on: bool) -> None:
        self.matrix.set_panel_visible(kind, on)
