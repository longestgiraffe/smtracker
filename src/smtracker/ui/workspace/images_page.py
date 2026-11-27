"""The Images & tracks section: the dataset's sample size, then each condition's movies --
the image, the cell outlines and the tracks -- one cell at a time, one track at a time, and
what is shown as a figure.

Asked for: "the start screen in matlab had like something where u can cycle thorugh the
images, track view etc." v2 opened on exactly this. Then, looking at one user's d2: "most cells
here have no tracks or less? maybe in the images and tracks on the right we can also show
how many tracks in the selected cell, or in the image ... also with a table and list of this
cells tracks? we also need a list of how many cells, replicates, tracks per cell (per 1000
frames or something)".

TWO PAGES, as QC and the Spatial Map have: the OVERVIEW is the sample size of every
condition and its replicate days (`images_view.inventory`: outlined cells, a cell with no
track included, and tracks per cell per 1000 frames and per minute); a click on a row opens
that condition (a day row at its first movie). The breadcrumb's Overview crumb comes back.
In a condition, the side card lists the movie's cells (a click zooms to one) or, with a cell
zoomed, that cell's tracks (a click, or a click on the track in the image, picks one out).

THIN QT. Every number, word, row and colour comes from `smtracker.images_view` as plain
data; this paints them. A movie and the inventory are read on workers (`_LoadJob`,
`_InventoryJob`), only the latest request drawn. Nothing is drawn over a missing image --
the page says which file is missing instead.
"""
from __future__ import annotations

import traceback

from qtpy.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from qtpy.QtGui import (QColor, QFontMetrics, QImage, QPainter, QPainterPath, QPen,
                        QPolygonF)
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ... import images_view as iv
from ..onboarding.panels import section_font
from ..theme import icons
from ..theme import typography as ty
from ..widgets import tooltip
from ..widgets.controls import Button, Segmented, Slider
from ..widgets.primitives import (SCROLLBAR_W, Card, Checkbox, col, discard, fit_label, icon, mix,
                                  label, row)
from ..widgets.table import TableBody
from .toolbar import SelectButton, ToolButton, rem

__all__ = ["ImagesSection"]

#: Remembered between sessions: where the sample-size table was last saved.
REMEMBER_TABLE_FOLDER = "images_inventory_folder"


class _Signals(QObject):
    #: generation, payload (or None), error
    loaded = Signal(int, object, str)
    #: generation, inventory payload (or None), error
    counted = Signal(int, object, str)
    #: data key (condition, movie, background, colour), look (the drawing's settings),
    #: thumbnail data (or None), its picture (a QImage, or None), error -- a gallery tile,
    #: keyed by what it was READ and DRAWN for, not by what is on screen when it lands
    thumbed = Signal(object, object, object, object, str)
    #: generation, a block of the movie's frames (`images_view.frame_block`) or None, error
    framed = Signal(int, object, str)
    #: generation, the movie's levels for its display (`images_view.movie_levels`) or None, error
    levelled = Signal(int, object, str)


#: The Movie background's frames are read this many at a time, on a worker, ahead of play.
FRAME_BLOCK = 48
#: At most this many bytes of one movie's frames are kept; the farthest go first.
FRAME_CACHE_BYTES = 256 * 2 ** 20


def _emit(signal, *args) -> None:
    try:
        signal.emit(*args)
    except RuntimeError:
        # The window closed while this read: nobody is left to show it to.
        pass


class _LoadJob(QRunnable):
    def __init__(self, generation, project, condition, movie, background, colour, signals, channels=None):
        super().__init__()
        self.args = (project, condition, movie)
        self.generation, self.background, self.colour = generation, background, colour
        self.signals = signals
        self.channels = dict(channels or {})

    def run(self):
        project, condition, movie = self.args
        try:
            payload = iv.movie_payload(project, condition, movie, background=self.background,
                                       colour=self.colour, channels=self.channels)
            result = (self.generation, payload, "")
        except iv.ImagesError as exc:
            result = (self.generation, None, str(exc))
        except Exception as exc:                            # noqa: BLE001 - shown AND printed
            traceback.print_exc()
            result = (self.generation, None, f"{type(exc).__name__}: {exc}")
        _emit(self.signals.loaded, *result)


class _FrameJob(QRunnable):
    """A block of the movie's recorded frames, read and greyed off the GUI thread."""

    def __init__(self, generation, project, condition, movie, lo, hi, signals, display=None):
        super().__init__()
        self.generation, self.args, self.signals = generation, (project, condition, movie,
                                                                lo, hi), signals
        self.display = display

    def run(self):
        try:
            result = (self.generation, iv.frame_block(*self.args, display=self.display), "")
        except iv.ImagesError as exc:
            result = (self.generation, None, str(exc))
        except Exception as exc:                            # noqa: BLE001 - shown AND printed
            traceback.print_exc()
            result = (self.generation, None, f"{type(exc).__name__}: {exc}")
        _emit(self.signals.framed, *result)


class _LevelsJob(QRunnable):
    """The counts the Movie background's black and white sliders run over, off the GUI thread."""

    def __init__(self, generation, project, condition, movie, display, signals):
        super().__init__()
        self.generation, self.args, self.signals = generation, (project, condition, movie, display), signals

    def run(self):
        try:
            result = (self.generation, iv.movie_levels(*self.args), "")
        except iv.ImagesError as exc:
            result = (self.generation, None, str(exc))
        except Exception as exc:                            # noqa: BLE001 - shown AND printed
            traceback.print_exc()
            result = (self.generation, None, f"{type(exc).__name__}: {exc}")
        _emit(self.signals.levelled, *result)


class _Histogram(QWidget):
    """The movie's counts as Fiji's B&C draws them: a histogram (log heights, the background would flatten the
    rest) with the black and white levels marked."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._levels, self._black, self._white = None, None, None
        self.setFixedHeight(64)
        self.setMinimumWidth(rem(16))

    def set_levels(self, levels, black, white) -> None:
        self._levels, self._black, self._white = levels, black, white
        self.update()

    def paintEvent(self, _event) -> None:
        import math

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(QColor(self.t.border), 1))
        painter.drawRect(rect)
        levels = self._levels
        if not levels or not levels.get("counts"):
            painter.end()
            return
        counts = levels["counts"]
        top = max(math.log1p(c) for c in counts) or 1.0
        width = (rect.width() - 1) / len(counts)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(self.t.muted_foreground))
        for i, c in enumerate(counts):
            h = (rect.height() - 2) * math.log1p(c) / top
            painter.drawRect(QRectF(rect.left() + 1 + i * width, rect.bottom() - h, max(width, 1.0), h))
        low, high = levels["low"], levels["high"]
        painter.setPen(QPen(QColor(self.t.foreground), 1))
        for value in (self._black, self._white):
            if value is not None and high > low:
                x = rect.left() + 1 + (rect.width() - 2) * (min(max(value, low), high) - low) / (high - low)
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.end()


class _ThumbJob(QRunnable):
    """One gallery tile, off the GUI thread: its data (`images_view.movie_thumbnail`, unless
    already read) and its picture, DRAWN here too -- a QPainter on a QImage is safe on a
    worker -- so the click that opens the gallery only shows what is ready."""

    def __init__(self, project, key, look, thumb, signals, channels=None):
        super().__init__()
        self.project, self.key, self.look, self.thumb = project, key, look, thumb
        self.signals = signals
        self.channels = dict(channels or {})

    def run(self):
        condition, movie, background, colour = self.key
        thumb, image, error = self.thumb, None, ""
        try:
            if thumb is None:
                thumb = iv.movie_thumbnail(self.project, condition, movie,
                                           background=background, colour=colour, channels=self.channels)
            image = render_thumbnail(thumb, **dict(self.look))
        except iv.ImagesError as exc:
            error = str(exc)
        except Exception as exc:                            # noqa: BLE001 - shown AND printed
            traceback.print_exc()
            error = f"{type(exc).__name__}: {exc}"
        _emit(self.signals.thumbed, self.key, self.look, thumb, image, error)


class _InventoryJob(QRunnable):
    def __init__(self, generation, project, conditions, signals):
        super().__init__()
        self.generation, self.project, self.conditions = generation, project, conditions
        self.signals = signals

    def run(self):
        try:
            result = (self.generation, iv.inventory(self.project, self.conditions), "")
        except Exception as exc:                            # noqa: BLE001 - shown AND printed
            traceback.print_exc()
            result = (self.generation, None, f"{type(exc).__name__}: {exc}")
        _emit(self.signals.counted, *result)


def _screen(x, y, fit) -> QPointF:
    k, ox, oy = fit
    return QPointF(x * k + ox, y * k + oy)


def _track_path(points, fit) -> QPainterPath:
    path = QPainterPath(_screen(points[0][0], points[0][1], fit))
    for x, y in points[1:]:
        path.lineTo(_screen(x, y, fit))
    return path


def paint_movie(p, fit, payload, image, *, cell=None, track=None, show_outlines=True,
                show_tracks=True, outline_width=1.4, track_width=1.3, track_alpha=255) -> None:
    """The movie as the page draws it: the grey image, the cell outlines and every track
    (one cell's when ``cell`` is zoomed, ``track`` picked out). ONE drawing for the page's
    canvas and the gallery's tiles, which pass thinner, fainter tracks so the cells still
    read at thumbnail size. ``fit`` is ``(scale, x offset, y offset)`` from data pixels to
    the painter; the caller sets the clip."""
    p.setRenderHint(QPainter.SmoothPixmapTransform, False)
    meta = payload["image"]
    p.drawImage(QRectF(_screen(-0.5, -0.5, fit),
                       _screen(meta["width"] - 0.5, meta["height"] - 0.5, fit)), image)
    p.setRenderHint(QPainter.Antialiasing, True)
    if show_outlines:
        for outline in payload["outlines"]:
            chosen = outline["cell"] == cell
            p.setPen(QPen(QColor(payload["outline_ink"]),
                          outline_width + 1.0 if chosen else outline_width))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF([_screen(x, y, fit) for x, y in outline["points"]]))
    if not show_tracks:
        return
    picked = None
    for one in payload["tracks"]:
        if cell is not None and one["cell"] != cell:
            continue
        if one["uid"] == track:
            picked = one
            continue
        colour = QColor(one["colour"])
        colour.setAlpha(80 if track is not None else track_alpha)   # the picked one stands out
        p.setPen(QPen(colour, track_width))
        p.drawPath(_track_path(one["points"], fit))
    if picked is not None:
        path = _track_path(picked["points"], fit)
        p.setPen(QPen(QColor("#ffffff"), 4.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
        p.setPen(QPen(QColor(picked["colour"]), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
        start = _screen(picked["points"][0][0], picked["points"][0][1], fit)
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.setBrush(QColor(picked["colour"]))
        p.drawEllipse(start, 4.0, 4.0)          # where it starts


def render_thumbnail(thumb: dict, *, size: tuple, dpr: float, table: tuple, outlines: bool,
                     tracks: bool, background: str) -> QImage:
    """A gallery tile's picture: the movie fitted into ``size`` (logical pixels, drawn at
    ``dpr``) by `paint_movie`, with thinner, fainter tracks so its cells still read at
    thumbnail size. A QImage, so a worker thread can draw it."""
    w, h = max(1, size[0]), max(1, size[1])
    out = QImage(max(1, round(w * dpr)), max(1, round(h * dpr)), QImage.Format_ARGB32_Premultiplied)
    out.setDevicePixelRatio(dpr)
    out.fill(QColor(background))
    meta = thumb.get("image")
    if not meta:
        return out
    image = QImage(meta["pixels"], meta["width"], meta["height"], meta["width"],
                   QImage.Format_Indexed8).copy()
    image.setColorTable(list(table))
    k = min(w / meta["width"], h / meta["height"])
    fit = (k, (w - meta["width"] * k) / 2.0 + 0.5 * k, (h - meta["height"] * k) / 2.0 + 0.5 * k)
    p = QPainter(out)
    p.setClipRect(QRectF(0, 0, w, h))
    paint_movie(p, fit, thumb, image, show_outlines=outlines, show_tracks=tracks,
                outline_width=1.0, track_width=0.6, track_alpha=150)
    p.end()
    return out


class _ImageCanvas(QWidget):
    """The movie (or one cell of it) at the widget's size, its own proportions kept: the
    grey image pixel for pixel, outlines, tracks. A click says which cell it landed in; with
    a cell zoomed, a click on one of its tracks says which track."""

    cell_clicked = Signal(str)
    track_clicked = Signal(str)
    #: A click inside the zoomed cell while Mark foci is on, in image pixels (17 Oct 2026, v2's distances).
    focus_clicked = Signal(float, float)

    #: How near a click must land to a track to pick it, in screen pixels.
    PICK_PX = 8.0
    # NO ZOOM ANIMATION (14 Oct 2026, the UI designer and a colleague): a cell click, the cell arrows
    # and a movie step jump straight to the new view. The 320 ms zoom repainted the whole
    # canvas -- image, outlines, every track -- each frame, a cost feared on older laptops.

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setMinimumSize(rem(18), rem(16))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        # A click gives the image the keyboard, so Space plays and pauses.
        self.setFocusPolicy(Qt.ClickFocus)
        self._payload = None
        self._image = None
        self._view = None
        self._cell = None
        self._track = None
        self._show_outlines = True
        self._show_tracks = True
        self._table = iv.grey_table()
        #: `images_view.frame_view` while playing, None for every track at once.
        self._time = None
        #: The Movie background's frame on screen (a QImage), or None for the still.
        self._frame = None
        self._message = "Pick a condition."
        #: The movie's marked foci, ``[(cell, x, y)]`` in image pixels, and whether a click marks one.
        self._foci: list = []
        self._foci_mode = False

    def set_foci(self, foci: list, mode: bool) -> None:
        """The foci to draw on this movie, and whether a click in the zoomed cell places one."""
        self._foci, self._foci_mode = list(foci), bool(mode)
        self.setCursor(Qt.CrossCursor if mode and self._cell is not None else Qt.PointingHandCursor)
        self.update()

    def focus_pick_px(self) -> float:
        """How near, in image pixels, a click must land to a focus to take it away again."""
        return self.PICK_PX / self._fit()[0] if self._view else 1.0

    def set_message(self, text: str) -> None:
        self._payload, self._image, self._message = None, None, text
        self.update()

    def set_display(self, table: list) -> None:
        """The grey colour table the image is shown through (brightness and contrast)."""
        self._table = list(table)
        for image in (self._image, self._frame):
            if image is not None:
                image.setColorTable(self._table)
        self.update()

    def set_frame(self, frame) -> None:
        """Draw one recorded frame, ``(width, height, bytes)``, in place of the still; None
        for the still again. Same colour table, so brightness and contrast hold."""
        if frame is None:
            self._frame = None
        else:
            width, height, pixels = frame
            self._frame = QImage(pixels, width, height, width, QImage.Format_Indexed8).copy()
            self._frame.setColorTable(self._table)
        self.update()

    def set_time(self, view) -> None:
        """Draw the movie at one frame (`images_view.frame_view`), or every track (None)."""
        self._time = view
        self.update()

    def set_view(self, payload, *, view, cell, show_outlines, show_tracks,
                 track=None) -> None:
        """Show ``view`` of ``payload``, at once."""
        if payload is not self._payload:
            image = payload.get("image")
            if image is None:
                self._image, self._message = None, payload.get("image_error") or "No image."
            else:
                # INDEXED, so brightness and contrast are a 256-entry colour table and a
                # slider never touches a pixel (`images_view.grey_table`).
                self._image = QImage(image["pixels"], image["width"], image["height"],
                                     image["width"], QImage.Format_Indexed8).copy()
                self._image.setColorTable(self._table)
                self._message = ""
            self._frame = None
        self._payload, self._view, self._cell, self._track = payload, view, cell, track
        self._show_outlines, self._show_tracks = show_outlines, show_tracks
        self.update()

    def _shown_view(self):
        """The view drawn: the one set (there is no zoom in between)."""
        return self._view

    # -- geometry ---------------------------------------------------------------------- #
    def _fit(self):
        x0, y0, x1, y1 = self._shown_view()
        w, h = float(self.width()), float(self.height())
        k = min(w / (x1 - x0), h / (y1 - y0))
        ox = (w - (x1 - x0) * k) / 2.0 - x0 * k
        oy = (h - (y1 - y0) * k) / 2.0 - y0 * k
        return k, ox, oy

    def _to_screen(self, x, y, fit) -> QPointF:
        k, ox, oy = fit
        return QPointF(x * k + ox, y * k + oy)

    def _to_data(self, pos, fit) -> tuple:
        k, ox, oy = fit
        return (pos.x() - ox) / k, (pos.y() - oy) / k

    # -- painting ---------------------------------------------------------------------- #
    def _path(self, points, fit) -> QPainterPath:
        path = QPainterPath(self._to_screen(points[0][0], points[0][1], fit))
        for x, y in points[1:]:
            path.lineTo(self._to_screen(x, y, fit))
        return path

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.t.muted))
        if self._image is None or self._payload is None or self._view is None:
            p.setPen(QColor(self.t.muted_foreground))
            p.setFont(ty.ui("sm"))
            p.drawText(self.rect().adjusted(24, 24, -24, -24),
                       Qt.AlignCenter | Qt.TextWordWrap, self._message)
            p.end()
            return
        fit = self._fit()
        x0, y0, x1, y1 = self._shown_view()
        clip = QRectF(self._to_screen(x0, y0, fit), self._to_screen(x1, y1, fit))
        p.setClipRect(clip)
        image = self._frame if self._frame is not None else self._image
        paint_movie(p, fit, self._payload, image, cell=self._cell, track=self._track,
                    show_outlines=self._show_outlines,
                    show_tracks=self._show_tracks and self._time is None)
        if self._show_tracks and self._time is not None:
            self._paint_time(p, fit)
        self._paint_foci(p, fit)
        self._paint_scale_bar(p, fit, (x0, y0, x1, y1))
        p.end()

    def _paint_foci(self, p, fit) -> None:
        """Each marked focus a yellow star with a dark rim (v2's yellow pentagram), the zoomed cell's or all."""
        import math

        for cell, x, y in self._foci:
            if self._cell is not None and cell != self._cell:
                continue
            c = self._to_screen(x, y, fit)
            star = QPainterPath()
            for i in range(10):
                r = 9.0 if i % 2 == 0 else 4.0
                a = -math.pi / 2 + i * math.pi / 5
                pt = QPointF(c.x() + r * math.cos(a), c.y() + r * math.sin(a))
                star.moveTo(pt) if i == 0 else star.lineTo(pt)
            star.closeSubpath()
            p.setPen(QPen(QColor("#1a1a1a"), 1.2))
            p.setBrush(QColor("#ffd23f"))
            p.drawPath(star)

    def _paint_scale_bar(self, p, fit, shown) -> None:
        """The scale bar, from the pixel size the analysis used (`images_view.scale_bar_um`),
        bottom left of the view ``shown`` = (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = shown
        per_px = float(self._payload["um_per_pixel"])
        bar_um = iv.scale_bar_um((x1 - x0) * per_px)
        start = self._to_screen(x0 + 0.05 * (x1 - x0), y1 - 0.06 * (y1 - y0), fit)
        end = self._to_screen(x0 + 0.05 * (x1 - x0) + bar_um / per_px, y1 - 0.06 * (y1 - y0),
                              fit)
        # On a dark backing: white alone vanishes on a light phase image.
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        p.drawRoundedRect(QRectF(start.x() - 6, start.y() - 24,
                                 max(end.x() - start.x(), 40) + 12, 32), 4, 4)
        p.setPen(QPen(QColor("#ffffff"), 3.0, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(start, end)
        p.setFont(ty.ui("xs", "medium"))
        p.drawText(QRectF(start.x(), start.y() - 22, max(end.x() - start.x(), 40), 18),
                   Qt.AlignHCenter | Qt.AlignBottom, iv.scale_bar_text(bar_um))

    def _paint_time(self, p, fit) -> None:
        """One frame of the movie: each molecule's recent path, a dot where it is now."""
        picked = None
        for trail in self._time["trails"]:
            if trail["uid"] == self._track:
                picked = trail
                continue
            colour = QColor(trail["colour"])
            if self._track is not None:
                colour.setAlpha(90)
            self._paint_trail(p, fit, trail, colour, 1.4)
        if picked is not None:
            p.setPen(QPen(QColor("#ffffff"), 4.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if len(picked["points"]) > 1:
                p.drawPath(self._path(picked["points"], fit))
            self._paint_trail(p, fit, picked, QColor(picked["colour"]), 2.2)

    def _paint_trail(self, p, fit, trail, colour, width) -> None:
        points = trail["points"]
        if len(points) > 1:
            p.setPen(QPen(colour, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(self._path(points, fit))
        if trail["on_frame"]:
            p.setPen(QPen(QColor("#ffffff"), 1.2))
            p.setBrush(colour)
            p.drawEllipse(self._to_screen(points[-1][0], points[-1][1], fit), 3.6, 3.6)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self._payload is not None and self._view:
            fit = self._fit()
            x, y = self._to_data(e.position() if hasattr(e, "position") else e.pos(), fit)
            hit = ""
            for outline in self._payload["outlines"]:
                polygon = QPolygonF([QPointF(px, py) for px, py in outline["points"]])
                if polygon.containsPoint(QPointF(x, y), Qt.OddEvenFill):
                    hit = outline["cell"]
                    break
            if self._foci_mode and self._cell is not None and hit == self._cell:
                # Mark foci is on: a click in the zoomed cell places a focus (or takes one away).
                self.focus_clicked.emit(float(x), float(y))
            elif self._cell is not None and hit == self._cell:
                # Inside the zoomed cell: a click picks the track drawn under it, or none.
                uid = iv.nearest_track(self._payload, x, y, cell=self._cell,
                                       max_distance=self.PICK_PX / fit[0])
                self.track_clicked.emit(uid or "")
            else:
                self.cell_clicked.emit(hit)
        super().mouseReleaseEvent(e)


#: A `Segmented` at size "sm": the grey tray every option above the image sits in.
TRAY_H, TRAY_PAD = 32, 3


class _Tray(QWidget):
    """The grey box a `Segmented` draws, holding something else: the movie button's
    face, the arrow pair, the Show checkboxes. One box per option group, so a group's
    parts read as one control (asked for: "it looks like Outlines and Tracks don't belong
    to Show")."""

    def __init__(self, t, *, pad=(TRAY_PAD, TRAY_PAD), spacing=2, parent=None):
        super().__init__(parent)
        self.t = t
        self._fill = t.muted
        self.setFixedHeight(TRAY_H)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(pad[0], TRAY_PAD, pad[1], TRAY_PAD)
        self.lay.setSpacing(spacing)

    def set_fill(self, colour: str) -> None:
        self._fill = colour
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._fill))
        p.drawRoundedRect(QRectF(self.rect()), self.t.radius_md, self.t.radius_md)
        p.end()


class _MovieButton(_Tray):
    """The movie's name and a grid glyph in one grey box; a click opens the gallery of
    every movie in the condition. The name keeps the WIDEST name's width
    (`images_view.movie_position_widest`), so the arrows beside it never move."""

    clicked = Signal()

    def __init__(self, t, name_label, parent=None):
        super().__init__(t, pad=(12, 10), spacing=10, parent=parent)
        self.name = name_label
        self.lay.addWidget(self.name)
        self.lay.addStretch(1)
        self.lay.addWidget(icon("grid", 15, t.muted_foreground, stroke=2.0))
        self.setCursor(Qt.PointingHandCursor)
        self._tip = tooltip.attach(t, self, "Show all movies")

    def set_name_width(self, text: str) -> None:
        self.name.setMinimumWidth(QFontMetrics(self.name.font()).horizontalAdvance(text) + 2)

    def enterEvent(self, e) -> None:
        if self.isEnabled():
            self.set_fill(mix(self.t.foreground, self.t.muted, 0.05))
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.set_fill(self.t.muted)
        super().leaveEvent(e)

    def changeEvent(self, e) -> None:
        if not self.isEnabled():
            self.set_fill(self.t.muted)
        super().changeEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.isEnabled() and self.rect().contains(
                e.position().toPoint() if hasattr(e, "position") else e.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class _TrayButton(QWidget):
    """A glyph inside a tray that raises to a white pill on hover, as a `Segmented`
    pill does: a ghost button's own hover is the tray's grey, so it would not show."""

    clicked = Signal(str)

    def __init__(self, t, key: str, glyph: str, parent=None):
        super().__init__(parent)
        self.t, self.key, self._glyph = t, key, glyph
        self._hover = False
        side = TRAY_H - 2 * TRAY_PAD
        self.setFixedSize(side, side)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, e) -> None:
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def changeEvent(self, e) -> None:
        self.update()
        super().changeEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.isEnabled() and self.rect().contains(
                e.position().toPoint() if hasattr(e, "position") else e.pos()):
            self.clicked.emit(self.key)
        super().mouseReleaseEvent(e)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        on = self._hover and self.isEnabled()
        if on:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self.t.background))
            p.drawRoundedRect(QRectF(self.rect()), self.t.radius_sm, self.t.radius_sm)
        ink = (self.t.foreground if on else self.t.muted_foreground) if self.isEnabled() \
            else mix(self.t.muted_foreground, self.t.muted, 0.4)
        g = 15
        icons.paint(p, self._glyph, QRectF((self.width() - g) / 2, (self.height() - g) / 2, g, g),
                    ink, stroke=2.0)
        p.end()


class _RampBar(QWidget):
    """The apparent-D colour ramp with its ends named (`images_view`'s ramp and range)."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._ramp, self._ends = (), ("", "")
        self.setFixedHeight(rem(2.2))

    def set_ramp(self, ramp, ends) -> None:
        self._ramp, self._ends = tuple(ramp), tuple(ends)
        self.update()

    def paintEvent(self, _e) -> None:
        if not self._ramp:
            return
        p = QPainter(self)
        w = self.width()
        steps = 96
        for i in range(steps):
            p.fillRect(QRectF(i * w / steps, 0, w / steps + 1, 10),
                       QColor(iv.ramp_colour(i / (steps - 1), self._ramp)))
        p.setPen(QColor(self.t.muted_foreground))
        p.setFont(ty.ui("xs", tabular=True))
        box = QRectF(0, 12, w, self.height() - 12)
        p.drawText(box, Qt.AlignLeft | Qt.AlignVCenter, self._ends[0])
        p.drawText(box, Qt.AlignRight | Qt.AlignVCenter, self._ends[1])
        p.end()


class _NameCell(QWidget):
    """A row's name: a condition's beside its colour dot, a day's or a movie's indented and
    muted under it. A name longer than ``max_px`` is shortened IN THE MIDDLE (the end is
    what tells "...d2.1" from "...d2.2") with the whole name on hover, so a long condition
    name cannot make the table wider than the window."""

    def __init__(self, t, text: str, colour: str | None = None, *, max_px: int,
                 parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0 if colour else rem(1.1), 0, 0, 0)
        lay.setSpacing(rem(0.5))
        if colour:
            dot = QWidget()
            dot.setFixedSize(9, 9)
            dot.setStyleSheet(f"background:{colour}; border-radius:2px;")
            lay.addWidget(dot, 0, Qt.AlignVCenter)
        font = ty.ui("sm", "medium" if colour else "normal")
        shown = text
        if QFontMetrics(font).horizontalAdvance(text) > max_px:
            shown = QFontMetrics(font).elidedText(text, Qt.ElideMiddle, max_px)
            self._tip = tooltip.attach(t, self, text)
        lay.addWidget(label(shown, font, t.foreground if colour else t.muted_foreground),
                      0, Qt.AlignVCenter)
        lay.addStretch(1)


class _HScroll(QScrollArea):
    """The table alone scrolls sideways, and only when the window is narrower than the
    table: the card, its header and its caption always fit the page. The bar's height is
    kept below the table whether it shows or not, so the card does not jump."""

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.setWidget(table)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedHeight(table.natural_height() + SCROLLBAR_W)


class _OverviewPage(QWidget):
    """The sample size of every condition and its replicate days, as one table."""

    #: The longest a row's name is drawn before it is shortened in the middle.
    NAME_PX = rem(13)

    #: condition, day (None for the condition), movie (None but for a movie row)
    row_clicked = Signal(str, object, object)
    #: "Export table": the table as a figure, or into the Composer.
    table_export_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        #: The styled (shadcn) tooltips, kept so they live as long as the page.
        self._tips: list = []
        self._payload = None
        self._rows: list = []
        self._labels: dict = {}
        self._colours: dict = {}
        self._level = iv.LEVELS[0]
        self._table: TableBody | None = None
        self._table_box: _HScroll | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # BOTH WAYS: a table of long condition names (one user's are 32 characters) is wider
        # than a laptop window, and its natural width must not become the window's minimum.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        # The page is the window's width, never wider: the TABLE scrolls sideways inside
        # its card when it must (`_HScroll`), so the header and the caption never run off
        # (reported on a window that was not full screen).
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page = QWidget()
        page.setStyleSheet("background:transparent;")
        scroll.setWidget(page)
        outer.addWidget(scroll)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self.card = Card(t)
        card_lay = QVBoxLayout(self.card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)
        self.save_button = ToolButton(t, "save-table", "save", size=rem(2), rest=t.card)
        self._tips.append(tooltip.attach(self.t, self.save_button, "Save this table, with every movie, as a CSV"))
        self.save_button.clicked.connect(lambda _k: self.save_table())
        # THE TABLE AS A FIGURE (asked for: "a table exporter also copyable to composer"):
        # the Export dialog every chart has, so Add to composer comes with it.
        self.export_button = ToolButton(t, "export-table", "download", size=rem(2),
                                        rest=t.card)
        self._tips.append(tooltip.attach(self.t, self.export_button, "Export this table as a figure (PDF, SVG, PNG, TIFF),\n"
                                      "or add it to the Composer"))
        self.export_button.clicked.connect(lambda _k: self.table_export_requested.emit())
        self.level_choice = Segmented(t, [iv.LEVEL_LABELS[k] for k in iv.LEVELS], current=0,
                                      size="sm")
        self.level_choice.changed.connect(self._level_picked)
        head = col(
            row(icon("table", icons.MD, t.muted_foreground),
                label("Sample size", section_font(), t.foreground),
                None, self.level_choice, self.save_button, self.export_button, spacing=8),
            label("Cells, replicates and tracks per cell per condition · click a row "
                  "to see its movies", ty.ui("xs"), t.muted_foreground),
            spacing=3, margins=(20, 16, 20, 12))
        card_lay.addWidget(head)
        self._host = QVBoxLayout()
        self._host.setContentsMargins(0, 0, 0, 0)
        card_lay.addLayout(self._host)
        self.status = label("Counting cells…", ty.ui("sm"), t.muted_foreground)
        self.status.setContentsMargins(20, 4, 20, 16)
        self._host.addWidget(self.status)
        self.caption = fit_label("", ty.ui("xs"), t.muted_foreground)
        self.caption.setContentsMargins(20, 10, 20, 16)
        card_lay.addWidget(self.caption)
        lay.addWidget(self.card)
        lay.addStretch(1)
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)

    def set_labels(self, labels: dict) -> None:
        self._labels = dict(labels)

    def set_colours(self, colours: dict) -> None:
        self._colours = dict(colours)
        if self._payload is not None:
            self._build()

    def set_message(self, text: str) -> None:
        self._payload = None
        if self._table_box is not None:
            discard(self._table_box)
            self._table_box = self._table = None
        self.status.setText(text)
        self.status.setVisible(True)
        self.caption.setText("")
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)

    def set_payload(self, payload: dict) -> None:
        self._payload = payload
        self._build()

    def payload(self):
        return self._payload

    def level(self) -> str:
        """"day" or "movie": what the table lists under each condition."""
        return self._level

    def _level_picked(self, index: int) -> None:
        self._level = iv.LEVELS[index]
        if self._payload is not None:
            self._build()

    def _build(self) -> None:
        payload = self._payload
        self._rows = iv.inventory_rows(payload, self._level)
        cells = []
        for item in self._rows:
            first = item["cells"][0]
            if item["kind"] in ("day", "movie"):
                name = _NameCell(self.t, first, max_px=self.NAME_PX)
            else:
                name = _NameCell(self.t, self._labels.get(first, first),
                                 self._colours.get(first, self.t.muted_foreground),
                                 max_px=self.NAME_PX)
            cells.append([name] + list(item["cells"][1:]))
        if self._table_box is not None:
            discard(self._table_box)
        self._table = TableBody(self.t, list(iv.INVENTORY_COLUMNS), cells, "",
                                clickable=True)
        self._table.row_clicked.connect(self._clicked)
        self._table_box = _HScroll(self._table)
        self._host.addWidget(self._table_box)
        self.status.setVisible(False)
        self.caption.setText(iv.inventory_caption(payload))
        self.save_button.setEnabled(True)
        self.export_button.setEnabled(True)

    def _clicked(self, index: int) -> None:
        item = self._rows[index]
        self.row_clicked.emit(item["condition"], item["day"], item.get("movie"))

    def save_table(self, path=None):
        """Write the inventory (condition, day and movie rows) to a CSV; returns the path."""
        if self._payload is None:
            return None
        if path is None:
            from pathlib import Path

            from ..theme import remember, remembered

            start = remembered(REMEMBER_TABLE_FOLDER) or str(Path.home())
            path, _ = QFileDialog.getSaveFileName(
                self, "Save the sample size", str(Path(start) / "sample_size.csv"),
                "CSV (*.csv)")
            if not path:
                return None
            remember(REMEMBER_TABLE_FOLDER, str(Path(path).parent))
        return iv.write_inventory_csv(self._payload, path)


class ImagesSection(QWidget):
    """The section: the Overview (sample size), or one condition's movies."""

    #: The Export button: `export_payload` is what it draws.
    export_requested = Signal()
    #: The Sample size table's Export.
    table_export_requested = Signal()
    #: The page or the condition on screen changed (the window re-titles its breadcrumb).
    condition_changed = Signal(str)
    #: condition, the PROJECT's uid of the picked track, "fast" or "napari" -- the same
    #: request the Spatial Map's spot card makes, answered by the same window method.
    track_requested = Signal(str, str, str)

    def __init__(self, t, parent=None, *, threads: QThreadPool | None = None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        #: The styled (shadcn) tooltips -- the toolbar's, never Qt's native one.
        self._tips: list = []
        self._project = None
        self._labels: dict = {}
        self._condition = None
        self._movie = None
        self._background = iv.BACKGROUNDS[0]
        self._colour = iv.COLOUR_MODES[0]
        self._cell = None
        self._track = None
        self._payload = None
        #: background -> (brightness, contrast): phase and fluorescence want their own.
        self._display = {k: (0, 0) for k in iv.BACKGROUNDS}
        #: background -> its colour table (`images_view.LUTS`), grey until picked.
        self._luts = {k: "grey" for k in iv.BACKGROUNDS}
        #: THE MOVIE BACKGROUND'S DISPLAY (asked 22 Oct 2026, against Fiji's B&C): black and white in the
        #: frames' raw counts (None = Auto), Auto per frame, smoothing and a background subtraction
        #: (`trackclip.FrameDisplay`), and the counts its sliders run over (`images_view.movie_levels`).
        self._movie_display = iv.MovieDisplay()
        self._levels = None
        self._levels_generation = 0
        self._levels_key = None
        self._filters_popover = None
        #: The experimenter's own channels, ``{name: folder}`` (`images_view.CHANNEL_PREFIX`; 17 Oct 2026), each
        #: one more Background pill. Set by the window, which remembers them per project.
        self._channels: dict = {}
        #: Marked foci (17 Oct 2026, v2's distances), ``{condition: {movie: {cell: [[x, y], ...]}}}`` in image
        #: pixels; the window remembers them per project (`foci_changed`).
        self._foci: dict = {}
        self._background_keys = iv.background_keys()
        self._generation = 0
        self._loading = False
        self._counting = False
        self._count_generation = 0
        self._list_key = None
        self._list_table: TableBody | None = None
        self._list_keys: list = []
        self._threads = threads or QThreadPool.globalInstance()
        self._syncing_display = False
        #: The frame shown while playing (float, so slow speeds advance smoothly), or None
        #: for every track at once; the speed; the clock; a track to open once loaded.
        self._time = None
        self._speed = 1.0
        self._syncing_time = False
        self._pending = None
        self._clock = QTimer(self)
        self._clock.setInterval(iv.TICK_MS)
        self._clock.timeout.connect(self._tick)
        self._signals = _Signals()
        self._signals.loaded.connect(self._loaded)
        self._signals.counted.connect(self._counted)
        self._signals.thumbed.connect(self._thumbed)
        self._signals.framed.connect(self._framed)
        self._signals.levelled.connect(self._levelled)
        # a slider dragged re-reads the frames once it rests, not at every step
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(150)
        self._display_timer.timeout.connect(self._apply_movie_display)
        #: THE MOVIE BACKGROUND's frames, by the tracker's frame number: read FRAME_BLOCK at a
        #: time on workers and kept for the movie on screen; the blocks read and in flight;
        #: a generation so a block for another movie is dropped when it lands.
        self._frames: dict = {}
        self._blocks_done: set = set()
        self._blocks_reading: set = set()
        self._frame_generation = 0
        self._frames_key = None
        #: The open gallery (or None); every tile read so far, by (condition, movie,
        #: background, colour), a failure kept as {"error": ...}; the keys being read; and
        #: whether a click is waiting for the last of them. The tiles are READ AHEAD, as
        #: soon as a condition's movie is on screen (asked for: "when the dialog appears,
        #: all series must be ready"), so the gallery opens complete.
        self._gallery = None
        self._thumbs: dict = {}
        #: Tile pictures for the CURRENT look only (data key -> QImage or None), the jobs in
        #: flight, the dialog built ahead with the signature it was built for, and whether
        #: a click is waiting for it.
        self._tile_images: dict = {}
        self._tile_look = None
        self._thumbs_reading: set = set()
        self._ready = None
        self._gallery_waiting = False
        #: A resize, a slider drag or a Show toggle redraws the tiles once it SETTLES: per
        #: tick it would redraw thirty pictures and rebuild the dialog on every step.
        self._look_timer = QTimer(self)
        self._look_timer.setSingleShot(True)
        self._look_timer.setInterval(250)
        self._look_timer.timeout.connect(self._prefetch_thumbs)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages)
        self.overview = _OverviewPage(t)
        self.overview.row_clicked.connect(self._overview_clicked)
        self.overview.table_export_requested.connect(self.table_export_requested.emit)
        self.pages.addWidget(self.overview)
        self.viewer = QWidget()
        self.viewer.setStyleSheet("background:transparent;")
        self.pages.addWidget(self.viewer)
        self._build_viewer(t)
        self._show_display()
        self._sync()
        self._show_clock()

    def _build_viewer(self, t) -> None:
        outer = QVBoxLayout(self.viewer)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        # THE OPTIONS: every group a muted label and ONE boxed control (asked for, after
        # "Movie" read as three blocks, Background and Show did not line up, and the
        # checkboxes did not read as Show's). Labels in two aligned columns, Export top right.
        # WIDTH: a 1280 px Windows window (150 % scaling) leaves the options ~721 px
        # (`test_qc_widgets`' width test). "Each track | One colour" made the grid 747 px;
        # "Per track | Single" (with a hover) makes it ~705.
        self.movie_label = label("", ty.ui("sm", "medium"), t.foreground)
        self.movie_button = _MovieButton(t, self.movie_label)
        self.movie_button.clicked.connect(self.open_gallery)
        self.prev_movie = _TrayButton(t, "prev", "chevron-left")
        self.next_movie = _TrayButton(t, "next", "chevron-right")
        self._tips.append(tooltip.attach(self.t, self.prev_movie, "Previous movie"))
        self._tips.append(tooltip.attach(self.t, self.next_movie, "Next movie"))
        self.prev_movie.clicked.connect(lambda _k: self.step_movie(-1))
        self.next_movie.clicked.connect(lambda _k: self.step_movie(1))
        arrows = _Tray(t)
        arrows.lay.addWidget(self.prev_movie)
        arrows.lay.addWidget(self.next_movie)
        movie = QHBoxLayout()
        movie.setContentsMargins(0, 0, 0, 0)
        movie.setSpacing(6)
        movie.addWidget(self.movie_button)
        movie.addWidget(arrows)
        movie.addStretch(1)
        self.background_choice = self._make_background_choice()
        self.channels_button = ToolButton(t, "channels", "plus", size=rem(2), rest=t.background)
        self._tips.append(tooltip.attach(self.t, self.channels_button,
                                         "Fluorescence channels: add, name and point to their folders"))
        self.channels_button.clicked.connect(lambda _k: self.open_channels_dialog())
        self._background_row = QHBoxLayout()
        self._background_row.setContentsMargins(0, 0, 0, 0)
        self._background_row.setSpacing(6)
        self._background_row.addWidget(self.background_choice)
        self._background_row.addWidget(self.channels_button)
        self.colour_choice = Segmented(
            t, [iv.COLOUR_LABELS[k] for k in iv.COLOUR_MODES], current=0, size="sm",
            tips=[iv.COLOUR_TIPS[k] for k in iv.COLOUR_MODES])
        self.colour_choice.changed.connect(self._colour_picked)
        self.outlines_box = Checkbox(t, checked=True)
        self.outlines_box.toggled.connect(lambda _on: (self._redraw(), self._look_changed()))
        self.tracks_box = Checkbox(t, checked=True)
        self.tracks_box.toggled.connect(lambda _on: (self._redraw(), self._look_changed()))
        self.show_tray = _Tray(t, pad=(12, 12), spacing=8)
        self.show_tray.lay.addWidget(self.outlines_box)
        self.show_tray.lay.addWidget(label("Outlines", ty.ui("sm"), t.foreground))
        self.show_tray.lay.addSpacing(10)
        self.show_tray.lay.addWidget(self.tracks_box)
        self.show_tray.lay.addWidget(label("Tracks", ty.ui("sm"), t.foreground))
        self.export_button = ToolButton(t, "export", "download", size=rem(2), rest=t.background)
        self._tips.append(tooltip.attach(self.t, self.export_button, "Export what is shown as a figure (PDF, SVG, PNG, TIFF)"))
        self.export_button.clicked.connect(lambda _k: self.export_requested.emit())
        self.movie_name = label("Movie", ty.ui("sm"), t.muted_foreground)
        self.colour_name = label("Colour by", ty.ui("sm"), t.muted_foreground)
        self.background_name = label("Background", ty.ui("sm"), t.muted_foreground)
        self.show_name = label("Show", ty.ui("sm"), t.muted_foreground)
        options = QGridLayout()
        options.setContentsMargins(0, 0, 0, 0)
        options.setHorizontalSpacing(8)
        options.setVerticalSpacing(10)
        left = Qt.AlignLeft | Qt.AlignVCenter
        options.addWidget(self.movie_name, 0, 0, left)
        options.addLayout(movie, 0, 1)
        options.addWidget(self.background_name, 0, 3, left)
        options.addLayout(self._background_row, 0, 4, left)
        options.addWidget(self.export_button, 0, 6, Qt.AlignRight | Qt.AlignVCenter)
        options.addWidget(self.colour_name, 1, 0, left)
        options.addWidget(self.colour_choice, 1, 1, left)
        options.addWidget(self.show_name, 1, 3, left)
        options.addWidget(self.show_tray, 1, 4, left)
        options.setColumnMinimumWidth(2, 8)               # with the spacing: 24 px between groups
        options.setColumnStretch(5, 1)
        outer.addLayout(options)

        body = QHBoxLayout()
        body.setSpacing(16)
        image_card = Card(t)
        image_lay = QVBoxLayout(image_card)
        image_lay.setContentsMargins(12, 12, 12, 12)
        self.canvas = _ImageCanvas(t)
        self.canvas.cell_clicked.connect(self._cell_clicked)
        self.canvas.track_clicked.connect(self._track_clicked)
        self.canvas.focus_clicked.connect(self._focus_clicked)
        image_lay.addWidget(self.canvas)
        # PLAYBACK (asked for: "i dont see controls, i think we need to redesign the track
        # viewer"): the movie in order of time on its own image -- each molecule a dot on
        # its frame with its recent path behind it. Every frame is drawn from
        # `images_view.frame_view`; this bar only keeps time.
        clock = QHBoxLayout()
        clock.setSpacing(6)
        self.play_button = ToolButton(t, "play", "play", size=rem(1.75), rest=t.card)
        self._tips.append(tooltip.attach(self.t, self.play_button, "Play the movie in order of time (Space)"))
        self.play_button.clicked.connect(lambda _k: self.play())
        self.pause_button = ToolButton(t, "pause", "pause", size=rem(1.75), rest=t.card)
        self._tips.append(tooltip.attach(self.t, self.pause_button, "Pause"))
        self.pause_button.clicked.connect(lambda _k: self.pause())
        self.pause_button.setVisible(False)
        self.stop_button = ToolButton(t, "stop", "square", size=rem(1.75), rest=t.card)
        self._tips.append(tooltip.attach(self.t, self.stop_button, "Stop: every track again, all frames at once"))
        self.stop_button.clicked.connect(lambda _k: self.stop_time())
        self.time_slider = Slider(t, 1, 2, 1)
        self.time_slider.setMinimumWidth(rem(4))
        self.time_slider.changed.connect(self._slid)
        self.speed_choice = Segmented(t, list(iv.SPEED_LABELS),
                                      current=iv.SPEEDS.index(1.0), size="sm",
                                      tips=list(iv.SPEED_TIPS))
        self.speed_choice.changed.connect(self._speed_picked)
        clock.addWidget(self.play_button)
        clock.addWidget(self.pause_button)
        clock.addWidget(self.stop_button)
        clock.addWidget(self.time_slider, 1)
        clock.addWidget(self.speed_choice)
        image_lay.addLayout(clock)
        self.time_label = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        # THE COLOUR TABLE (asked: "a contrast and brightness picker (and color picker maybe,
        # LUT)"): how each level is coloured, one choice per background like the sliders.
        # On the readout's line: the sliders' row already sets the page's narrowest width
        # (721 px on a 1280 px Windows laptop at 150 %, `TestOptionsGrid`).
        self.lut_choice = SelectButton(t, [(k, iv.LUT_LABELS[k]) for k in iv.LUTS],
                                       current="grey")
        self.lut_choice.setFixedWidth(rem(6.5))
        self._tips.append(tooltip.attach(t, self.lut_choice,
                                         "Colour table: how each brightness level is coloured "
                                         "(display only, printed with the figure)", side="top"))
        self.lut_choice.changed.connect(self._lut_picked)
        readout = QHBoxLayout()
        readout.setSpacing(8)
        readout.addWidget(self.time_label, 1)
        readout.addWidget(label("LUT", ty.ui("xs"), t.muted_foreground))
        readout.addWidget(self.lut_choice)
        image_lay.addLayout(readout)
        # BRIGHTNESS AND CONTRAST (asked for: "bar for adjusting brightness and contrast of
        # images"), under the image they change, one pair per background since phase and
        # fluorescence want different settings. Linear, whole image, and printed with the
        # figure (`images_view.display_lut`).
        adjust = QHBoxLayout()
        adjust.setSpacing(8)
        lo, hi = iv.ADJUST_RANGE
        self.brightness = Slider(t, lo, hi, 0)
        self.contrast = Slider(t, lo, hi, 0)
        # ABOVE: the sliders sit at the bottom of the window, where a tooltip below was cut
        self._brightness_tip = tooltip.attach(t, self.brightness, "Image display", side="top")
        self._contrast_tip = tooltip.attach(t, self.contrast, "Image display", side="top")
        self._tips += [self._brightness_tip, self._contrast_tip]
        for name, slider in (("Brightness", self.brightness), ("Contrast", self.contrast)):
            slider.setMinimumWidth(rem(3))
            slider.changed.connect(lambda _v: self._adjusted())
            adjust.addWidget(label(name, ty.ui("xs"), t.muted_foreground))
            adjust.addWidget(slider, 1)
            adjust.addSpacing(4)
        self.reset_display = ToolButton(t, "reset-display", "rotate", size=rem(1.75),
                                        rest=t.card)
        self._tips.append(tooltip.attach(self.t, self.reset_display,
                                         "Back to the automatic stretch (0.5-99.5 %)",
                                         side="top"))
        self.reset_display.clicked.connect(lambda _k: self.set_display(0, 0))
        adjust.addWidget(self.reset_display)
        adjust.setContentsMargins(0, 0, 0, 0)
        self._bc_row = QWidget()
        self._bc_row.setStyleSheet("background:transparent;")
        self._bc_row.setLayout(adjust)
        image_lay.addWidget(self._bc_row)
        # THE MOVIE'S OWN ROW, in the sliders' place while the Movie background is on: black and white in
        # the recorded counts, as Fiji's B&C sets them (`_build_movie_row`)
        self._movie_row = self._build_movie_row()
        image_lay.addWidget(self._movie_row)
        self._movie_row.setVisible(False)
        body.addWidget(image_card, 1)

        side = Card(t)
        side.setFixedWidth(rem(21))
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(18, 16, 18, 16)
        side_lay.setSpacing(8)
        side_lay.addWidget(label("This movie", ty.ui("sm", "semibold"), t.foreground))
        self.caption = fit_label("", ty.ui("xs"), t.muted_foreground)
        side_lay.addWidget(self.caption)
        self.ramp = _RampBar(t)
        side_lay.addWidget(self.ramp)
        self.legend = fit_label("", ty.ui("xs"), t.muted_foreground)
        side_lay.addWidget(self.legend)
        side_lay.addSpacing(4)
        cells = QHBoxLayout()
        cells.setSpacing(6)
        self.prev_cell = ToolButton(t, "prev-cell", "chevron-left", size=rem(1.75), rest=t.card)
        self.next_cell = ToolButton(t, "next-cell", "chevron-right", size=rem(1.75), rest=t.card)
        self._tips.append(tooltip.attach(self.t, self.prev_cell, "Previous cell"))
        self._tips.append(tooltip.attach(self.t, self.next_cell, "Next cell"))
        self.prev_cell.clicked.connect(lambda _k: self.step_cell(-1))
        self.next_cell.clicked.connect(lambda _k: self.step_cell(1))
        self.cell_label = label("Whole field", ty.ui("sm", "semibold"), t.foreground)
        cells.addWidget(self.prev_cell)
        cells.addWidget(self.cell_label, 1)
        cells.addWidget(self.next_cell)
        side_lay.addLayout(cells)
        self.cell_note = fit_label("Click a cell, here or in the image, to zoom to it.",
                                   ty.ui("xs"), t.muted_foreground)
        side_lay.addWidget(self.cell_note)
        # DISTANCES TO FOCI (17 Oct 2026, v2's Active Tools > Distances measurements): mark points of interest in a
        # zoomed cell; Distances... measures every localisation of the marked cells to its nearest focus, beside
        # chance (`distances_view`).
        side_lay.addSpacing(4)
        side_lay.addWidget(label("Distances to foci", ty.ui("sm", "semibold"), t.foreground))
        mark = QHBoxLayout()
        mark.setSpacing(8)
        self.foci_box = Checkbox(t, checked=False)
        self.foci_box.toggled.connect(lambda _on: self._redraw())
        mark.addWidget(self.foci_box)
        mark.addWidget(label("Mark foci (click in the zoomed cell)", ty.ui("sm"), t.foreground))
        mark.addStretch(1)
        side_lay.addLayout(mark)
        self.foci_note = fit_label("", ty.ui("xs"), t.muted_foreground)
        side_lay.addWidget(self.foci_note)
        foci_actions = QHBoxLayout()
        foci_actions.setSpacing(6)
        self.clear_foci = Button(t, "Clear this cell", size="small", variant="outline")
        self.clear_foci.clicked.connect(self.clear_cell_foci)
        self.distances_button = Button(t, "Distances\u2026", size="small")
        self.distances_button.clicked.connect(self.distances_requested.emit)
        foci_actions.addWidget(self.clear_foci)
        foci_actions.addWidget(self.distances_button)
        foci_actions.addStretch(1)
        side_lay.addLayout(foci_actions)
        # THE PICKED TRACK IN A VIEWER (asked for: "a way to open it in track viewer /
        # napari"): the fast track view on the track's own cell, or napari at the track.
        self.track_actions = QWidget()
        self.track_actions.setStyleSheet("background:transparent;")
        actions = QHBoxLayout(self.track_actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self.open_fast = Button(t, "Play this track", size="small")
        self._tips.append(tooltip.attach(self.t, self.open_fast, "Play the movie from just before this track starts,\n"
                                  "the track drawn over the others"))
        self.open_fast.clicked.connect(self.play_track)
        self.open_napari = Button(t, "Open in napari", size="small")
        self._tips.append(tooltip.attach(self.t, self.open_napari, "napari on this movie, at this track\n"
                                    "(it loads the condition, which takes a moment)"))
        self.open_napari.clicked.connect(lambda: self._request_track("napari"))
        actions.addWidget(self.open_fast)
        actions.addWidget(self.open_napari)
        actions.addStretch(1)
        side_lay.addWidget(self.track_actions)
        self.track_message = fit_label("", ty.ui("xs"), t.muted_foreground)
        side_lay.addWidget(self.track_message)
        self.track_actions.setVisible(False)
        self.track_message.setVisible(False)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.list_title = label("Cells in this movie", ty.ui("sm", "semibold"), t.foreground)
        head.addWidget(self.list_title, 1)
        self.whole_button = Button(t, "Whole field", size="small")
        self.whole_button.clicked.connect(lambda: self.set_cell(None))
        head.addWidget(self.whole_button)
        side_lay.addSpacing(4)
        side_lay.addLayout(head)
        self._list_host = QVBoxLayout()
        self._list_host.setContentsMargins(0, 0, 0, 0)
        side_lay.addLayout(self._list_host, 1)
        # where the list is empty ("No tracks in this cell") the spare height goes here, never between the headings
        side_lay.addStretch(0)
        body.addWidget(side, 0)
        outer.addLayout(body, 1)

    # -- what is shown ------------------------------------------------------------------ #
    def set_project(self, project, conditions) -> None:
        """``conditions``: ``[(key, label)]`` in the workspace's order."""
        self._project = project
        self._labels = {key: lab for key, lab in conditions}
        self._condition, self._movie, self._payload = None, None, None
        self._cell = self._track = None
        iv.forget(project)
        self._thumbs, self._thumbs_reading, self._gallery_waiting = {}, set(), False
        self._tile_images, self._tile_look = {}, None
        self._drop_ready()
        self.overview.set_labels(self._labels)
        self.overview.set_message("Counting cells…")
        self._count_generation += 1
        self._counting = False
        self.canvas.set_message("Pick a condition.")
        self.pages.setCurrentWidget(self.overview)
        self._sync()

    def set_colours(self, colours: dict) -> None:
        """Each condition's colour, the workspace's own (`window._condition_colours`)."""
        self.overview.set_colours(colours)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`): the list every picker offers and the Overview's rows;
        nothing is counted again."""
        from ...condition_order import apply, sort_by

        self._labels = {k: self._labels[k] for k in apply(list(self._labels), order)}
        self.overview.set_labels(self._labels)
        payload = self.overview.payload()
        if payload is not None and payload.get("conditions"):
            self.overview.set_payload(dict(payload, conditions=sort_by(payload["conditions"], order,
                                                                       lambda e: e["condition"])))

    def conditions(self) -> list:
        return list(self._labels)

    def condition(self):
        """The condition on screen, or None on the Overview."""
        return None if self.showing_overview() else self._condition

    def showing_overview(self) -> bool:
        return self.pages.currentWidget() is self.overview

    def show_overview(self) -> None:
        self.pages.setCurrentWidget(self.overview)
        self._count()
        self.condition_changed.emit("")

    def show_condition(self, condition: str, movie: str | None = None) -> None:
        if self._project is None:
            return
        changed = condition != self._condition or self.showing_overview()
        if condition != self._condition or movie is not None:
            self._condition, self._movie = condition, movie
            self._cell = self._track = None
        self.pages.setCurrentWidget(self.viewer)
        if changed:
            self.condition_changed.emit(condition)
        self._load()

    def ensure_shown(self) -> None:
        """The first visit opens on the Overview and counts the cells."""
        if self._project is not None and self.showing_overview():
            self._count()

    def step_movie(self, step: int) -> None:
        if not self._payload:
            return
        movies = self._payload["movies"]
        index = max(0, min(len(movies) - 1, movies.index(self._payload["movie"]) + step))
        if movies[index] != self._payload["movie"]:
            self._movie, self._cell, self._track = movies[index], None, None
            self._load()

    def show_movie(self, movie: str) -> None:
        """Open ``movie`` of the condition on screen, whole field, nothing picked."""
        if not self._payload or movie == self._payload["movie"]:
            return
        self._movie, self._cell, self._track = movie, None, None
        self._load()

    # -- the gallery: every movie of the condition as a picture -------------------------- #
    # NOTHING HEAVY ON THE CLICK (asked for: "300 ms is too much"). Measured on a MacBook
    # Air, a 29-movie condition: building the dialog 46 ms, drawing its 29 pictures 59 ms, the
    # overlay's snapshot 43 ms. So the tiles are read AND drawn ahead on worker threads,
    # the dialog is built ahead and kept, and a click only re-points it and opens it.

    def _thumb_key(self, movie: str) -> tuple:
        return (self._condition, movie, self._background, self._colour)

    def _gallery_window(self):
        window = self.window()
        return window if window is not self else self

    def _look(self) -> tuple:
        """How the tiles are drawn now: their size in this window, the screen's ratio, the
        display and the Show toggles -- everything a picture depends on, as a hashable."""
        from .movie_gallery import tile_picture_size

        image = (self._payload or {}).get("image")
        aspect = image["height"] / image["width"] if image else 1.0
        brightness, contrast = self._display[self._background]
        return (("size", tile_picture_size(self._gallery_window().width(), aspect)),
                ("dpr", float(self.devicePixelRatioF())),
                ("table", tuple(iv.display_table(brightness, contrast,
                                                 self._luts[self._background]))),
                ("outlines", bool(self.outlines_box.checked)),
                ("tracks", bool(self.tracks_box.checked)),
                ("background", self.t.muted))

    def _signature(self) -> tuple:
        payload = self._payload
        return (self._condition, self._background, self._colour, tuple(payload["movies"]),
                self._gallery_window().width(), self._look())

    def _prefetch_thumbs(self) -> None:
        """Read and draw every tile of the condition on screen in the background (current
        movie first, then outward), then build the dialog. Skips what is ready or in flight;
        a change of look redraws from the data already read, never re-reads."""
        payload = self._payload
        if payload is None or self._project is None:
            return
        look = self._look()
        if look != self._tile_look:              # pictures for another look are stale
            self._tile_look, self._tile_images = look, {}
        movies, here = payload["movies"], payload["index"]
        for i in sorted(range(len(movies)), key=lambda i: (abs(i - here), i)):
            key = self._thumb_key(movies[i])
            if key in self._tile_images or (key, look) in self._thumbs_reading:
                continue
            thumb = self._thumbs.get(key)
            if thumb is not None and "error" in thumb:
                self._tile_images[key] = None
                continue
            self._thumbs_reading.add((key, look))
            self._threads.start(_ThumbJob(self._project, key, look, thumb, self._signals, self._channels))
        self._build_if_ready()

    def _missing_tiles(self) -> list:
        if self._payload is None:
            return []
        return [m for m in self._payload["movies"]
                if self._thumb_key(m) not in self._tile_images]

    def _thumbed(self, key, look, thumb, image, error: str) -> None:
        if (key, look) not in self._thumbs_reading:
            return                                      # from a project since replaced
        self._thumbs_reading.discard((key, look))
        # a failure is kept, named on its tile, so the gallery is never waiting on it
        self._thumbs[key] = thumb if thumb is not None else {"error": error}
        if look == self._tile_look:
            self._tile_images[key] = image
        self._build_if_ready()

    def _build_if_ready(self) -> None:
        """Every tile drawn: build the dialog now, hidden, so the click has nothing to do."""
        if self._payload is None or self._missing_tiles():
            return
        signature = self._signature()
        if self._ready is not None and self._ready[0] == signature:
            dialog = self._ready[1]
        elif self._gallery is not None:
            return                      # the open one is replaced once it closes
        else:
            dialog = self._build_gallery()
            self._drop_ready()
            self._ready = (signature, dialog)
        if self._gallery_waiting:
            self._open(dialog)

    def _build_gallery(self):
        from .movie_gallery import MovieGalleryDialog

        payload = self._payload
        window = self._gallery_window()
        content = window.centralWidget() if hasattr(window, "centralWidget") else None
        image = payload.get("image")
        aspect = image["height"] / image["width"] if image else 1.0
        title, count = iv.gallery_title(self._labels.get(self._condition, self._condition),
                                        len(payload["movies"]))
        dialog = MovieGalleryDialog(self.t, title, count, payload["movies"], payload["movie"],
                                    aspect=aspect, content=content, parent=window)
        dialog.picked.connect(self.show_movie)
        dialog.closed.connect(self._gallery_closed)
        for movie in payload["movies"]:
            key = self._thumb_key(movie)
            thumb = self._thumbs.get(key) or {}
            dialog.set_picture(movie, self._tile_images.get(key),
                               thumb.get("caption") or thumb.get("error", ""))
        return dialog

    def _drop_ready(self) -> None:
        if self._ready is not None and self._ready[1] is not self._gallery:
            self._ready[1].deleteLater()
        self._ready = None

    def open_gallery(self) -> None:
        """The movie button: the dialog built ahead, re-pointed at this movie and opened. If
        a click beats the read-ahead it opens the moment the last tile is drawn -- never
        half-filled."""
        if self._payload is None or self._gallery is not None:
            return
        if self._ready is not None and self._ready[0] == self._signature():
            self._open(self._ready[1])
            return
        self._gallery_waiting = True
        self._prefetch_thumbs()

    def _open(self, dialog) -> None:
        self._gallery_waiting = False
        self._gallery = dialog
        dialog.set_current(self._payload["movie"])
        dialog.open()

    def _gallery_closed(self) -> None:
        self._gallery = None
        self._prefetch_thumbs()        # anything that changed while it was open, rebuilt now

    def _look_changed(self) -> None:
        if getattr(self, "_payload", None) is not None:
            self._look_timer.start()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._look_changed()

    def step_cell(self, step: int) -> None:
        if not self._payload or not self._payload["outlines"]:
            return
        cells = [o["cell"] for o in self._payload["outlines"]]
        if self._cell not in cells:
            index = 0 if step > 0 else len(cells) - 1
        else:
            index = (cells.index(self._cell) + step) % len(cells)
        self.set_cell(cells[index])

    def set_cell(self, cell) -> None:
        """Show ``cell`` (None: the whole field)."""
        if (cell or None) != self._cell:
            self._track = None
        self._cell = cell or None
        self._redraw()

    def set_track(self, uid) -> None:
        """Pick one track out (None puts them all back)."""
        self._track = uid or None
        self._redraw()

    def _cell_clicked(self, cell: str) -> None:
        self.set_cell(cell or None)

    # -- playback -------------------------------------------------------------------------- #
    def playing(self) -> bool:
        return self._clock.isActive()

    def frame(self):
        """The frame on screen while playing or paused, None for every track at once."""
        return None if self._time is None else int(self._time)

    def play(self) -> None:
        if self._payload is None:
            return
        axis = self._payload["time"]
        if self._time is None or int(self._time) >= axis["last"]:
            start = (iv.track_start(self._payload, self._track) if self._track
                     else axis["first"])
            self._time = float(start)
        self._clock.start()
        self._show_clock()

    def pause(self) -> None:
        self._clock.stop()
        self._show_clock()

    def stop_time(self) -> None:
        """Every track again, all frames at once."""
        self._clock.stop()
        self._time = None
        self._show_clock()

    def play_track(self) -> None:
        """Play from just before the picked track starts."""
        if self._payload is None or self._track is None:
            return
        self._time = float(iv.track_start(self._payload, self._track))
        self.play()

    def show_track(self, condition: str, movie: str, cell, uid) -> None:
        """Open ``movie`` of ``condition`` at ``cell`` with ``uid`` picked and playing: the
        track viewer, reached from another section (the Spatial Map's spot card)."""
        self._pending = (cell, uid)
        self.show_condition(condition, movie)

    def _tick(self) -> None:
        if self._payload is None or self._time is None:
            self._clock.stop()
            return
        axis = self._payload["time"]
        step = self._time + self._speed * iv.TICK_MS / max(1e-6, axis["frame_interval_ms"])
        step = min(step, float(axis["last"]))
        if self._movie_shown() and int(step) not in self._frames:
            # THE CLOCK WAITS FOR THE FRAME: a track is never drawn over another frame.
            self._want_frame(int(step))
            return
        self._time = step
        if self._time >= axis["last"]:
            self._clock.stop()
        self._show_clock()

    # -- the Movie background's frames -------------------------------------------------------- #
    def _movie_shown(self) -> bool:
        payload = self._payload
        return (payload is not None and payload["background"] == "movie"
                and payload["image"] is not None)

    def _reset_frames(self) -> None:
        self._frame_generation += 1
        self._frames, self._blocks_done, self._blocks_reading = {}, set(), set()
        self.canvas.set_frame(None)

    def _want_frame(self, frame: int) -> None:
        """Read the block holding ``frame`` unless it is read or on its way."""
        axis = self._payload["time"]
        if not axis["first"] <= frame <= axis["last"]:
            return
        start = axis["first"] + ((frame - axis["first"]) // FRAME_BLOCK) * FRAME_BLOCK
        if start in self._blocks_done or start in self._blocks_reading:
            return
        self._blocks_reading.add(start)
        self._threads.start(_FrameJob(self._frame_generation, self._project, self._condition,
                                      self._payload["movie"], start, start + FRAME_BLOCK - 1,
                                      self._signals, self._movie_display))

    def _framed(self, generation: int, block, error: str) -> None:
        if generation != self._frame_generation:
            return
        if block is None:
            self._clock.stop()
            self._blocks_reading.clear()
            self.time_label.setText(f"The movie's frames could not be read: {error}")
            self._show_clock(read=False)
            return
        self._blocks_reading.discard(block["first"])
        self._blocks_done.add(block["first"])
        for k, pixels in enumerate(block["frames"]):
            self._frames[block["first"] + k] = pixels
        per = max(1, block["width"] * block["height"])
        if len(self._frames) * per > FRAME_CACHE_BYTES and self._time is not None:
            # the farthest from the frame on screen go first, a block at a time
            keep = int(self._time)
            for start in sorted(self._blocks_done, key=lambda b: -abs(b - keep)):
                if len(self._frames) * per <= FRAME_CACHE_BYTES:
                    break
                self._blocks_done.discard(start)
                for f in range(start, start + FRAME_BLOCK):
                    self._frames.pop(f, None)
        if self._payload is not None:
            self._show_clock()

    def _slid(self, value: int) -> None:
        if self._syncing_time:
            return
        self._time = float(value)
        self._show_clock()

    def _speed_picked(self, index: int) -> None:
        self._speed = iv.SPEEDS[index]

    def _show_clock(self, read: bool = True) -> None:
        """The time bar and the canvas, for the frame on screen (or every track)."""
        payload = self._payload
        playing = self._clock.isActive()
        self.play_button.setVisible(not playing)
        self.pause_button.setVisible(playing)
        has = payload is not None
        for widget in (self.play_button, self.pause_button, self.time_slider,
                       self.speed_choice):
            widget.setEnabled(has)
        self.stop_button.setEnabled(has and self._time is not None)
        if not has:
            self.time_label.setText("")
            self.canvas.set_time(None)
            return
        axis = payload["time"]
        self._syncing_time = True
        try:
            self.time_slider.lo, self.time_slider.hi = axis["first"], axis["last"]
            self.time_slider.set_value(axis["first"] if self._time is None
                                       else int(self._time))
            self.time_slider.update()
        finally:
            self._syncing_time = False
        if self._time is None:
            self.canvas.set_time(None)
            # EVERY TRACK AT ONCE, over the first frame as the display shows it (the still was
            # stretched once on loading; the black and white set here must hold on it too)
            first = int(axis["first"])
            pixels = self._frames.get(first) if self._movie_shown() else None
            if self._movie_shown() and pixels is None and read:
                self._want_frame(first)
            image = payload["image"]
            self.canvas.set_frame((image["width"], image["height"], pixels) if pixels is not None else None)
            self.time_label.setText(iv.time_text(payload))
            return
        frame = int(self._time)
        view = iv.frame_view(payload, frame, cell=self._cell)
        text = iv.time_text(payload, frame, view)
        if self._movie_shown():
            pixels = self._frames.get(frame)
            if pixels is None:
                # the tracks wait for their frame: nothing is drawn on another one
                if read:
                    self._want_frame(frame)
                self.time_label.setText(text + " · reading the frames…")
                return
            image = payload["image"]
            self.canvas.set_frame((image["width"], image["height"], pixels))
            if self._clock.isActive():
                self._want_frame(frame + FRAME_BLOCK // 2)      # the next block, ahead of play
        self.canvas.set_time(view)
        self.time_label.setText(text)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and self._payload is not None:
            self.pause() if self.playing() else self.play()
            return
        super().keyPressEvent(e)

    def _request_track(self, viewer: str) -> None:
        if self._project is None or self._condition is None or self._track is None:
            return
        self.track_requested.emit(
            self._condition, iv.project_uid(self._project, self._condition, self._track),
            viewer)

    def show_track_message(self, text: str) -> None:
        """What became of the last open request (the window says it here, never in a
        modal dialog)."""
        self.track_message.setText(text)
        self.track_message.setVisible(bool(text))

    def _track_clicked(self, uid: str) -> None:
        self.set_track(uid or None)

    # -- foci (17 Oct 2026, v2's distances) --------------------------------------------------------------------- #
    #: Every change of the marked foci (the window remembers them per project); Distances... pressed.
    foci_changed = Signal(dict)
    distances_requested = Signal()

    def foci(self) -> dict:
        """``{condition: {movie: {cell: [[x, y], ...]}}}``, cells without a focus left out."""
        return {c: {m: {k: [list(p) for p in v] for k, v in cells.items() if v} for m, cells in movies.items()}
                for c, movies in self._foci.items()}

    def set_foci(self, foci: dict) -> None:
        """The foci remembered for this project (the window's), drawn where their movie is shown."""
        self._foci = {c: {m: {k: [list(p) for p in v] for k, v in cells.items()} for m, cells in movies.items()}
                      for c, movies in (foci or {}).items()}
        self._redraw()

    def _cell_foci(self) -> list:
        if self._condition is None or self._movie is None or self._cell is None:
            return []
        return self._foci.get(self._condition, {}).get(self._movie, {}).get(self._cell, [])

    def _focus_clicked(self, x: float, y: float) -> None:
        """A click in the zoomed cell with Mark foci on: a focus there, or the one under it taken away."""
        if self._condition is None or self._movie is None or self._cell is None:
            return
        cells = self._foci.setdefault(self._condition, {}).setdefault(self._movie, {})
        points = cells.setdefault(self._cell, [])
        reach = self.canvas.focus_pick_px()
        near = [i for i, (fx, fy) in enumerate(points) if (fx - x) ** 2 + (fy - y) ** 2 <= reach ** 2]
        if near:
            points.pop(near[0])
        else:
            points.append([round(x, 2), round(y, 2)])
        self._redraw()
        self.foci_changed.emit(self.foci())

    def clear_cell_foci(self) -> None:
        if self._cell_foci():
            self._foci[self._condition][self._movie][self._cell] = []
            self._redraw()
            self.foci_changed.emit(self.foci())

    def _show_foci(self) -> None:
        """The canvas's foci for the movie on screen, and the side panel's line."""
        movie = (self._foci.get(self._condition, {}).get(self._movie, {})
                 if self._condition is not None and self._movie is not None else {})
        self.canvas.set_foci([(cell, x, y) for cell, pts in movie.items() for x, y in pts],
                             self.foci_box.checked)
        n_here = len(self._cell_foci())
        n_cells = sum(1 for cells in self._foci.get(self._condition, {}).values() for v in cells.values() if v)
        name = self._labels.get(self._condition, self._condition) if self._condition is not None else ""
        if self._cell is None:
            words = ("Zoom to a cell, switch on Mark foci and click its foci." if not n_cells else
                     f"{n_cells} cell{'s' * (n_cells != 1)} marked in {name}.")
        else:
            words = (f"{n_here} focus in this cell" if n_here == 1 else f"{n_here} foci in this cell") + (
                f"; {n_cells} cell{'s' * (n_cells != 1)} marked in {name}." if n_cells else ".")
        self.foci_note.setText(words)
        self.clear_foci.setEnabled(bool(n_here))

    def _overview_clicked(self, condition: str, day, movie=None) -> None:
        if movie is None and day is not None and self.overview.payload() is not None:
            movie = iv.first_movie_of_day(self.overview.payload(), condition, day)
        self.show_condition(condition, movie)

    def _background_picked(self, index: int) -> None:
        self._background = self._background_keys[index]
        self._show_display()
        self._load()

    # -- the experimenter's own channels (17 Oct 2026) --------------------------------------------- #
    #: Asked when a channel set is saved in the dialog: ``[{"name", "folder"}]`` -- the window remembers it.
    channels_changed = Signal(list)

    def _make_background_choice(self):
        keys = self._background_keys
        choice = Segmented(self.t, [iv.background_short(k) for k in keys],
                           current=keys.index(self._background) if self._background in keys else 0, size="sm",
                           tips=[iv.background_tip(k, self._channels) for k in keys])
        choice.changed.connect(self._background_picked)
        return choice

    def channels(self) -> list:
        return [{"name": n, "folder": f} for n, f in self._channels.items()]

    def set_channels(self, channels: list) -> None:
        """Use ``[{"name", "folder"}]`` as the extra Backgrounds: the pills are rebuilt, a removed channel on screen
        falls back to Phase, and every tile read with the old set is dropped."""
        if iv.channels_problem(channels):
            raise ValueError(iv.channels_problem(channels))
        self._channels = iv.channel_map(channels)
        self._background_keys = iv.background_keys(self._channels)
        for key in self._background_keys:
            self._display.setdefault(key, (0, 0))
            self._luts.setdefault(key, "grey")
        if self._background not in self._background_keys:
            self._background = iv.BACKGROUNDS[0]
        old = self.background_choice
        self.background_choice = self._make_background_choice()
        self._background_row.insertWidget(0, self.background_choice)
        discard(old)
        self._thumbs, self._thumbs_reading = {}, set()
        self._tile_images, self._tile_look = {}, None
        self._show_display()
        if self._payload is not None:
            self._load()

    def open_channels_dialog(self):
        """The dialog of channels (`image_channels_dialog`); Save applies the set and asks the window to keep it."""
        from .image_channels_dialog import ImageChannelsDialog

        window = self.window()
        content = window.centralWidget() if hasattr(window, "centralWidget") else None
        roots = list((getattr(self._project, "roots", None) or {}).values())
        dialog = ImageChannelsDialog(self.t, self.channels(), roots, content=content, parent=window)
        dialog.saved.connect(self._channels_saved)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()
        self._channels_dialog = dialog
        return dialog

    def _channels_saved(self, channels: list) -> None:
        self.set_channels(channels)
        self.channels_changed.emit(list(channels))

    # -- brightness and contrast ------------------------------------------------------------ #
    def display(self) -> tuple:
        """``(brightness, contrast)`` for the background on screen."""
        return self._display[self._background]

    def set_display(self, brightness: int, contrast: int) -> None:
        self._display[self._background] = (int(brightness), int(contrast))
        self._show_display()

    def lut(self) -> str:
        """The colour table of the background on screen."""
        return self._luts[self._background]

    def set_lut(self, lut: str) -> None:
        if lut not in iv.LUTS:
            raise ValueError(f"LUT must be one of {iv.LUTS}")
        self._luts[self._background] = lut
        self._show_display()

    def _lut_picked(self, key: str) -> None:
        self.set_lut(key)

    def _adjusted(self) -> None:
        if self._syncing_display:
            return
        self._display[self._background] = (self.brightness.value(), self.contrast.value())
        self._show_display()

    def _show_display(self) -> None:
        movie = self._background == "movie"
        if movie:
            # the movie's levels are set in counts (its own row); brightness and contrast stay neutral on it
            self._display["movie"] = (0, 0)
        if self._movie_row.isHidden() == movie:
            self._movie_row.setVisible(movie)
            self._bc_row.setVisible(not movie)
        brightness, contrast = self._display[self._background]
        self._syncing_display = True
        try:
            self.brightness.set_value(brightness)
            self.contrast.set_value(contrast)
        finally:
            self._syncing_display = False
        lut = self._luts[self._background]
        if self.lut_choice.current != lut:
            self.lut_choice.set_current(lut)
        self.canvas.set_display(iv.display_table(brightness, contrast, lut))
        self.reset_display.setEnabled(bool(brightness or contrast))
        tip = iv.display_words(brightness, contrast, lut)
        # styled tooltips whose words follow the setting
        for watch in (self._brightness_tip, self._contrast_tip):
            watch.text = f"Image display: {tip}"
        self._sync_movie_row()
        self._look_changed()

    # -- the Movie background's display: black and white in counts, and two filters ------------------ #
    def _build_movie_row(self) -> QWidget:
        """Black and white in the recorded counts (Fiji's B&C), Auto, Reset and the Filters popover."""
        t = self.t
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.black_slider = Slider(t, 0, 1, 0)
        self.white_slider = Slider(t, 0, 1, 1)
        self.black_value = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.white_value = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        for name, slider, value in (("Black", self.black_slider, self.black_value),
                                    ("White", self.white_slider, self.white_value)):
            slider.setMinimumWidth(rem(3))
            slider.changed.connect(lambda _v: self._levels_moved())
            value.setMinimumWidth(rem(2.5))
            lay.addWidget(label(name, ty.ui("xs"), t.muted_foreground))
            lay.addWidget(slider, 1)
            lay.addWidget(value)
        self._black_tip = tooltip.attach(t, self.black_slider, "The count shown black", side="top")
        self._white_tip = tooltip.attach(t, self.white_slider, "The count shown white", side="top")
        self._tips += [self._black_tip, self._white_tip]
        self.auto_levels = Button(t, "Auto", size="sm")
        self.auto_levels.clicked.connect(self._auto_levels)
        self.reset_levels = Button(t, "Reset", size="sm")
        self.reset_levels.clicked.connect(self._reset_levels)
        self.filters_button = Button(t, "Filters", size="sm")
        self.filters_button.clicked.connect(self._open_filters)
        self._tips += [
            tooltip.attach(t, self.auto_levels, "Black and white from the movie: 0.35 % of its counts saturated, "
                                                "as Fiji's Auto", side="top"),
            tooltip.attach(t, self.reset_levels, "Black and white at the movie's lowest and highest count",
                           side="top"),
            tooltip.attach(t, self.filters_button, "The counts' histogram, Auto per frame, smoothing and a "
                                                   "background subtraction (display only)", side="top")]
        for button in (self.auto_levels, self.reset_levels, self.filters_button):
            lay.addWidget(button)
        return row

    def movie_display(self):
        """The Movie background's display (`trackclip.FrameDisplay`)."""
        return self._movie_display

    def set_movie_display(self, display) -> None:
        """Show the Movie background as ``display`` says; a change of filter puts black and white back on Auto
        (the counts mean something else once a background is subtracted) and reads the movie's levels again."""
        filters = display.filters != self._movie_display.filters
        if filters:
            from dataclasses import replace

            display = replace(display, black=None, white=None)
            self._levels = None
        self._movie_display = display
        if filters:
            self._request_levels()
        self._sync_movie_row()
        self._display_timer.start()

    def _levels_moved(self) -> None:
        if self._syncing_display or self._levels is None:
            return
        from dataclasses import replace

        black, white = float(self.black_slider.value()), float(self.white_slider.value())
        if white <= black:
            white = black + 1.0
        self.set_movie_display(replace(self._movie_display, black=black, white=white))

    def _auto_levels(self) -> None:
        from dataclasses import replace

        self.set_movie_display(replace(self._movie_display, black=None, white=None))

    def _reset_levels(self) -> None:
        if self._levels is None:
            return
        from dataclasses import replace

        self.set_movie_display(replace(self._movie_display, black=float(self._levels["low"]),
                                       white=float(self._levels["high"])))

    def _sync_movie_row(self) -> None:
        """The sliders, numbers, histogram and words, from the display and the movie's levels."""
        levels, display = self._levels, self._movie_display
        enabled = levels is not None
        for widget in (self.black_slider, self.white_slider, self.reset_levels):
            widget.setEnabled(enabled)
        if not enabled:
            self.black_value.setText("")
            self.white_value.setText("")
        else:
            import math

            low, high = math.floor(levels["low"]), math.ceil(levels["high"])
            auto = levels["auto"]
            black = display.black if display.black is not None else auto[0]
            white = display.white if display.white is not None else auto[1]
            self._syncing_display = True
            try:
                for slider, value in ((self.black_slider, black), (self.white_slider, white)):
                    slider.lo, slider.hi = low, max(high, low + 1)
                    slider.set_value(int(round(value)))
                    slider.update()
            finally:
                self._syncing_display = False
            per_frame = display.per_frame and (display.black is None or display.white is None)
            self.black_value.setText("frame" if per_frame and display.black is None else f"{black:.0f}")
            self.white_value.setText("frame" if per_frame and display.white is None else f"{white:.0f}")
        words = iv.movie_display_words(display, levels, self._luts.get("movie", "grey"))
        for watch in (self._black_tip, self._white_tip):
            watch.text = f"Movie display: {words}"
        if self._filters_popover is not None and getattr(self, "levels_histogram", None) is not None:
            if levels is not None:
                self.levels_histogram.set_levels(
                    levels, display.black if display.black is not None else levels["auto"][0],
                    display.white if display.white is not None else levels["auto"][1])
            self.filters_words.setText(words)

    def _open_filters(self) -> None:
        """The Filters popover: the counts' histogram with black and white marked, Auto per frame, smoothing at the
        spot's size and a background subtraction."""
        from ..widgets.menus import Panel

        t = self.t
        if self._filters_popover is not None:
            self._filters_popover.close_chain()
            self._filters_popover = None
        menu = Panel(t, self.window())
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        lay.addWidget(label("Counts in the movie", ty.ui("sm", "medium"), t.foreground))
        self.levels_histogram = _Histogram(t)
        lay.addWidget(self.levels_histogram)
        display = self._movie_display

        def check_row(text, checked, on_toggle):
            line = QHBoxLayout()
            line.setSpacing(8)
            box = Checkbox(t, checked)
            box.toggled.connect(on_toggle)
            line.addWidget(box)
            line.addWidget(label(text, ty.ui("sm"), t.foreground), 1)
            lay.addLayout(line)
            return box

        self.per_frame_box = check_row("Auto per frame (a bleaching movie stays visible)", display.per_frame,
                                       self._per_frame_toggled)
        self.smooth_box = check_row(f"Smooth at the spot's size ({iv.SMOOTH_PX:g} px)", display.smooth_px > 0,
                                    self._smooth_toggled)
        lay.addWidget(label("Subtract background", ty.ui("sm", "medium"), t.foreground))
        keys = list(iv.MOVIE_FILTER_LABELS)
        self.background_filter = Segmented(t, [iv.MOVIE_FILTER_LABELS[k] for k in keys],
                                           current=keys.index(display.background), size="sm",
                                           tips=[iv.MOVIE_FILTER_TIPS[k] for k in keys])
        self.background_filter.changed.connect(self._background_filter_picked)
        lay.addWidget(self.background_filter)
        self.filters_words = label("", ty.ui("xs"), t.muted_foreground)
        self.filters_words.setWordWrap(True)
        self.filters_words.setFixedWidth(rem(20))
        lay.addWidget(self.filters_words)
        menu.add_widget(holder)
        menu.set_minimum_visible_width(rem(21))

        def _closed() -> None:
            if self._filters_popover is menu:
                self._filters_popover = None
                self.levels_histogram = None

        menu.closed.connect(_closed)
        self._filters_popover = menu
        self._sync_movie_row()
        menu.popup_under(self.filters_button)

    def _per_frame_toggled(self, on: bool) -> None:
        from dataclasses import replace

        self.set_movie_display(replace(self._movie_display, per_frame=bool(on)))

    def _smooth_toggled(self, on: bool) -> None:
        from dataclasses import replace

        self.set_movie_display(replace(self._movie_display, smooth_px=iv.SMOOTH_PX if on else 0.0))

    def _background_filter_picked(self, index: int) -> None:
        from dataclasses import replace

        self.set_movie_display(replace(self._movie_display, background=list(iv.MOVIE_FILTER_LABELS)[index]))

    def _request_levels(self) -> None:
        payload = self._payload
        if payload is None or not self._movie_shown():
            return
        self._levels_generation += 1
        self._levels_key = (payload["condition"], payload["movie"], self._movie_display.filters)
        self._threads.start(_LevelsJob(self._levels_generation, self._project, payload["condition"],
                                       payload["movie"], self._movie_display, self._signals))

    def _levelled(self, generation: int, levels, error: str) -> None:
        if generation != self._levels_generation:
            return
        self._levels = levels
        if levels is None:
            self.black_value.setText("")
            self.white_value.setText(error[:40])
        self._sync_movie_row()

    def _apply_movie_display(self) -> None:
        """Read the frames on screen again under the display (the slider has come to rest)."""
        if not self._movie_shown():
            return
        self._reset_frames()
        self._show_clock()

    def _colour_picked(self, index: int) -> None:
        self._colour = iv.COLOUR_MODES[index]
        self._load()

    # -- loading, on workers ---------------------------------------------------------------- #
    def _count(self) -> None:
        if self._project is None or self._counting or self.overview.payload() is not None:
            return
        self._count_generation += 1
        self._counting = True
        self.overview.set_message("Counting cells…")
        self._threads.start(_InventoryJob(self._count_generation, self._project,
                                          list(self._labels), self._signals))

    def _counted(self, generation: int, payload, error: str) -> None:
        if generation != self._count_generation:
            return
        self._counting = False
        if payload is None:
            self.overview.set_message(error)
            return
        self.overview.set_payload(payload)

    def _load(self) -> None:
        if self._project is None or self._condition is None:
            return
        self._generation += 1
        self._loading = True
        self.movie_label.setText("loading…")
        self._threads.start(_LoadJob(self._generation, self._project, self._condition,
                                     self._movie, self._background, self._colour,
                                     self._signals, self._channels))

    def _loaded(self, generation: int, payload, error: str) -> None:
        if generation != self._generation:
            return
        self._loading = False
        if payload is None:
            self._payload = None
            self._pending = None
            self._clock.stop()
            self._time = None
            self.canvas.set_message(error)
            self.caption.setText(error)
            self._sync()
            return
        moved = self._payload is None or payload["movie"] != self._payload["movie"]
        key = (payload["condition"], payload["movie"], payload["background"])
        if key != self._frames_key:
            self._frames_key = key
            self._reset_frames()
        self._payload = payload
        self._movie = payload["movie"]
        if (payload["background"] == "movie" and payload["image"] is not None
                and self._levels_key != (payload["condition"], payload["movie"], self._movie_display.filters)):
            self._levels = None                       # this movie's own counts, read on a worker
            self._request_levels()
        if self._cell is not None and self._cell not in {o["cell"] for o in payload["outlines"]}:
            self._cell = None
        if self._track is not None and self._track not in {t["uid"] for t in payload["tracks"]}:
            self._track = None
        if moved:
            self._clock.stop()
            self._time = None
        pending, self._pending = self._pending, None
        if pending is not None:
            cell, uid = pending
            outlines = {o["cell"] for o in payload["outlines"]}
            self._cell = cell if cell in outlines else None
            self._track = uid if uid in {t["uid"] for t in payload["tracks"]} else None
        self._redraw()
        self._prefetch_thumbs()
        if pending is not None and self._track is not None:
            self.play_track()

    def wait(self, seconds: float = 20.0) -> None:
        """Until the movie and the count in flight have landed (for tests)."""
        import time

        end = time.monotonic() + seconds
        while (self._loading or self._counting) and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)

    # -- drawing ---------------------------------------------------------------------------- #
    def _view(self):
        payload = self._payload
        if self._cell is not None:
            return iv.cell_box(payload, self._cell)
        image = payload["image"]
        if image is None:
            return None
        return [-0.5, -0.5, image["width"] - 0.5, image["height"] - 0.5]

    def _redraw(self) -> None:
        payload = self._payload
        if payload is None:
            self._sync()
            return
        if payload["image"] is None:
            self.canvas.set_message(payload["image_error"])
        else:
            self.canvas.set_view(payload, view=self._view(), cell=self._cell,
                                 show_outlines=self.outlines_box.checked,
                                 show_tracks=self.tracks_box.checked, track=self._track)
        self._show_foci()
        self.caption.setText(iv.movie_line(payload))
        self.legend.setText("Tracks: " + payload["legend"] + ".")
        lo, hi = payload["appd_range"]
        self.ramp.setVisible(payload["colour_mode"] == "appd")
        self.ramp.set_ramp(payload["ramp"], iv.ramp_ends(lo, hi))
        self._sync()
        self._show_clock()

    def _rebuild_list(self) -> None:
        """The side card's table: the movie's cells, or the zoomed cell's tracks. Rebuilt
        only when the movie or the cell changes; picking a track only lights its row."""
        payload = self._payload
        key = (id(payload), self._cell) if payload is not None else None
        if key != self._list_key:
            self._list_key = key
            if self._list_table is not None:
                discard(self._list_table)
                self._list_table = None
            self._list_keys = []
            if payload is not None:
                if self._cell is None:
                    rows, keys = iv.cell_rows(payload)
                    columns = iv.CELL_COLUMNS
                else:
                    rows, keys = iv.track_rows(payload, self._cell)
                    columns = iv.TRACK_COLUMNS
                self._list_keys = keys
                if rows:
                    table = TableBody(self.t, list(columns), rows, "", scrollable=True,
                                      row_keys=keys, clickable=True)
                    table.row_clicked.connect(self._list_clicked)
                    self._list_host.addWidget(table)
                    # SHOWN HERE, in the layout first (`primitives.discard`'s rule): the show Qt queues for a widget
                    # added to a visible parent never ran for a table built in the turn that discarded the one
                    # before it (a movie's frames landing just after a cell was picked), and the card's sections
                    # spread over the empty space (3 Nov 2026, reported)
                    table.show()
                    self._list_table = table
        if payload is None:
            self.list_title.setText("")
        elif self._cell is None:
            n = len(payload["outlines"])
            self.list_title.setText("Cells in this movie" if n else "No cells outlined")
        else:
            n = sum(1 for t in payload["tracks"] if t["cell"] == self._cell)
            self.list_title.setText(f"Tracks in this cell ({n})" if n
                                    else "No tracks in this cell")
        if self._list_table is not None:
            if self._cell is not None and self._track is not None:
                self._list_table.highlight_key(self._track)
            else:
                self._list_table.clear_highlight()

    def _list_clicked(self, index: int) -> None:
        key = self._list_keys[index]
        if self._cell is None:
            self.set_cell(key)
        else:
            self.set_track(None if key == self._track else key)

    def _sync(self) -> None:
        payload = self._payload
        has = payload is not None
        if has:
            self.movie_label.setText(iv.movie_position(payload))
            self.movie_button.set_name_width(iv.movie_position_widest(payload))
            self.cell_label.setText(iv.cell_position(payload, self._cell))
            if self._cell is None:
                self.cell_note.setText("Click a cell, here or in the image, to zoom to it.")
            else:
                note = iv.cell_summary(payload, self._cell)
                if self._track is not None:
                    note += f" · track {self._track.rsplit(':', 1)[-1]} picked out"
                else:
                    note += " · click a track to pick it out"
                self.cell_note.setText(note)
        elif not self._loading:
            self.movie_label.setText("")
        for widget in (self.prev_movie, self.next_movie, self.prev_cell, self.next_cell,
                       self.export_button, self.whole_button, self.movie_button):
            widget.setEnabled(has)
        self.whole_button.setEnabled(has and self._cell is not None)
        self.export_button.setEnabled(has and payload["image"] is not None)
        picked = has and self._track is not None
        # isHidden, not isVisible: the section may be off screen while this runs.
        if picked == self.track_actions.isHidden():
            self.track_actions.setVisible(picked)
        if not picked:
            self.show_track_message("")
        self._rebuild_list()

    # -- export ----------------------------------------------------------------------------- #
    def export_payload(self):
        """What the Export draws: this movie as shown -- the cell zoomed, what is visible."""
        if not self._payload or self._payload["image"] is None:
            return None
        view = self._view()
        brightness, contrast = self.display()
        out = dict(self._payload, crop=view if self._cell is not None else None,
                   cell=self._cell, show_outlines=self.outlines_box.checked,
                   show_tracks=self.tracks_box.checked,
                   display={"brightness": brightness, "contrast": contrast,
                            "lut": self._luts[self._background]})
        if self._movie_shown():
            # the movie's frames were rendered under its display: the legend says which
            out["display"]["words"] = iv.movie_display_words(self._movie_display, self._levels,
                                                             self._luts[self._background])
        # AS SHOWN: a paused movie prints its frame and the tracks as they stood at it.
        if self._time is not None:
            frame = int(self._time)
            out["tracks"] = iv.tracks_at(self._payload, frame, cell=self._cell)
            out["trail_frame"] = frame
            pixels = self._frames.get(frame) if self._movie_shown() else None
            if pixels is not None:
                out["image"] = dict(self._payload["image"], pixels=pixels)
                out["frame"] = frame
        if self._movie_shown() and "frame" not in out:
            out["frame"] = int(self._payload["time"]["first"])
        return out
