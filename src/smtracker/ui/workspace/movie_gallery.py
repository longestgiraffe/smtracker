"""The movie gallery: every movie of a condition as a picture, to pick one by sight.

Asked for after a drop-down list of thirty near-identical names ("series017") proved not
meaningful: people recognise a field by what it looks like. The Images & tracks section's
movie button opens it; a tile is the movie as the page draws it (image, outlines, thin
tracks), its name, and its cell and track counts, which is also a quick check before
opening one. Picking a tile opens that movie and closes the dialog.

THIN QT. The pictures arrive as finished `QPixmap`s from the section, which renders them
with the SAME drawing the page uses (`images_page.paint_movie`), and the words come from
`images_view`. Tiles are built at once and fill in as each movie's picture lands, so the
dialog is never an empty wait.

ESCAPE CLOSES IT, unlike the other dialogs (`widgets.dialogs`, "ESCAPE, THE ONE THING..."):
those hold settings a stray key would throw away; a picker holds nothing to lose.
"""
from __future__ import annotations

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QScrollArea, QWidget

from ..theme import typography as ty
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import SCROLLBAR_W, label, mix, scrollbar_css

__all__ = ["MovieGalleryDialog", "TILE_MIN", "gallery_layout", "tile_picture_size"]

#: The narrowest a tile gets; the card holds as many columns as fit.
TILE_MIN = 190
#: The card's widest, and the room it leaves around itself inside the window.
CARD_MAX, CARD_MARGIN = 900, 48
GAP = 16
CARD_PAD = 24
#: The ring around a tile's picture.
RING = 3


def gallery_layout(window_width: int) -> tuple:
    """``(card width, columns, tile width)`` for a window this wide -- one arithmetic for the
    dialog and for the section that draws the tiles BEFORE the dialog exists."""
    card_w = max(TILE_MIN + 2 * CARD_PAD, min(CARD_MAX, window_width - CARD_MARGIN))
    # room for the scrollbar, so it never covers the last column
    inner_w = card_w - 2 * CARD_PAD - SCROLLBAR_W - 6
    columns = max(1, (inner_w + GAP) // (TILE_MIN + GAP))
    return card_w, columns, (inner_w - GAP * (columns - 1)) // columns


def tile_picture_size(window_width: int, aspect: float) -> tuple:
    """The pixels (logical) a tile's picture fills inside its ring, for a window this wide."""
    _card, _columns, tile_w = gallery_layout(window_width)
    return tile_w - 2 * RING, round(tile_w * aspect) - 2 * RING


class _MovieTile(QWidget):
    """One movie: its picture (a placeholder until it lands), name and counts."""

    clicked = Signal(str)
    hovered = Signal(str)

    NAME_H, CAPTION_H, TEXT_GAP = 20, 18, 8

    def __init__(self, t, movie: str, width: int, aspect: float, parent=None):
        super().__init__(parent)
        self.t, self.movie = t, movie
        self.pixmap = None
        self.caption = ""
        self.current = self.focused = self.hover = False
        self._aspect = aspect
        self.setFixedWidth(width)
        self._resize()
        self.setCursor(Qt.PointingHandCursor)

    def _resize(self) -> None:
        pic = round(self.width() * self._aspect)
        self.setFixedHeight(pic + self.TEXT_GAP + self.NAME_H + self.CAPTION_H)

    def picture_rect(self) -> QRectF:
        return QRectF(0, 0, self.width(), round(self.width() * self._aspect))

    def set_picture(self, image, caption: str) -> None:
        """A `QImage` drawn ahead, off the GUI thread (or None for a tile that failed)."""
        self.pixmap, self.caption = image, caption
        self.update()

    def set_state(self, *, current=None, focused=None) -> None:
        if current is not None:
            self.current = current
        if focused is not None:
            self.focused = focused
        self.update()

    def enterEvent(self, e) -> None:
        self.hover = True
        self.hovered.emit(self.movie)
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self.hover = False
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.rect().contains(
                e.position().toPoint() if hasattr(e, "position") else e.pos()):
            self.clicked.emit(self.movie)
        super().mouseReleaseEvent(e)

    def paintEvent(self, _e) -> None:
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.picture_rect()
        radius = t.radius_md
        inner = rect.adjusted(RING, RING, -RING, -RING)    # room for the ring outside the picture
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.muted))
        p.drawRoundedRect(inner, radius, radius)
        if self.pixmap is not None:
            p.save()
            clip = QPainterPath()
            clip.addRoundedRect(inner, radius, radius)
            p.setClipPath(clip)
            # drawn at the size it will be shown; scaled smoothly if the window has changed
            # since (the section draws it again at the new size for next time)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(inner, self.pixmap)
            p.restore()
        else:
            p.setPen(QColor(t.muted_foreground))
            p.setFont(ty.ui("xs"))
            p.drawText(inner, Qt.AlignCenter, "loading…")
        # the rings: current dark, keyboard focus or hover grey; never both colours at once
        ring = None
        if self.current:
            ring = QPen(QColor(t.foreground), 2.0)
        elif self.focused or self.hover:
            ring = QPen(QColor(mix(t.foreground, t.background, 0.35)), 2.0)
        if ring is not None:
            p.setPen(ring)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius + 2, radius + 2)
        else:
            p.setPen(QPen(QColor(t.border), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(inner.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        # the name, a "current" tag, the counts
        top = rect.bottom() + self.TEXT_GAP
        p.setFont(ty.ui("sm", "medium"))
        p.setPen(QColor(t.foreground))
        p.drawText(QRectF(0, top, self.width(), self.NAME_H), Qt.AlignLeft | Qt.AlignVCenter,
                   self.movie)
        if self.current:
            name_w = QFontMetrics(ty.ui("sm", "medium")).horizontalAdvance(self.movie)
            tag_font = ty.ui("xs", "semibold")
            tag_w = QFontMetrics(tag_font).horizontalAdvance("current") + 14
            tag = QRectF(name_w + 8, top + 2, tag_w, self.NAME_H - 4)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(t.foreground))
            p.drawRoundedRect(tag, tag.height() / 2, tag.height() / 2)
            p.setPen(QColor(t.background))
            p.setFont(tag_font)
            p.drawText(tag, Qt.AlignCenter, "current")
        p.setFont(ty.ui("xs"))
        p.setPen(QColor(t.muted_foreground))
        p.drawText(QRectF(0, top + self.NAME_H, self.width(), self.CAPTION_H),
                   Qt.AlignLeft | Qt.AlignVCenter, self.caption)
        p.end()


class MovieGalleryDialog(OverlayDialog):
    """Every movie of a condition as a tile; `picked(movie)` fires and the dialog closes.

    Arrow keys move a focus ring (by row and column), Enter or Space opens the focused
    movie, Home / End jump to the first / last, Escape or the corner X closes."""

    picked = Signal(str)

    def __init__(self, t, title: str, count: str, movies: list, current: str, *,
                 aspect: float = 1.0, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self.movies = list(movies)
        self.current = current
        self.focus_index = self.movies.index(current) if current in self.movies else 0
        window_w = parent.width() if parent is not None else CARD_MAX + CARD_MARGIN
        window_h = parent.height() if parent is not None else 800
        card_w, self.columns, tile_w = gallery_layout(window_w)
        self._card.setFixedWidth(card_w)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self.title_label = label(title, ty.ui("base", "semibold"), t.foreground)
        head.addWidget(self.title_label)
        head.addWidget(label(f"· {count}", ty.ui("sm"), t.muted_foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)

        body = QWidget()
        body.setStyleSheet("background:transparent;")
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(GAP)
        grid.setVerticalSpacing(GAP + 4)
        self.tiles: dict = {}
        for i, movie in enumerate(self.movies):
            tile = _MovieTile(t, movie, tile_w, aspect)
            tile.clicked.connect(self._pick)
            tile.hovered.connect(lambda m: self._set_focus(self.movies.index(m), scroll=False))
            tile.set_state(current=movie == current)
            grid.addWidget(tile, i // self.columns, i % self.columns, Qt.AlignTop)
            self.tiles[movie] = tile
        grid.setRowStretch(len(self.movies) // self.columns + 1, 1)
        self.scroll = QScrollArea()
        self.scroll.setWidget(body)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { background:transparent; }" + scrollbar_css(t))
        self.scroll.viewport().setStyleSheet("background:transparent;")
        # As tall as the tiles, up to 80 % of the window; past that it scrolls.
        rows = -(-len(self.movies) // self.columns)
        tile_h = next(iter(self.tiles.values())).height() if self.tiles else 0
        natural = rows * tile_h + (rows - 1) * (GAP + 4) + 4
        self.scroll.setFixedHeight(max(tile_h, min(natural, int(window_h * 0.8) - 90)))
        self.card_lay.addWidget(self.scroll)
        self._set_focus(self.focus_index)

    # -- pictures ------------------------------------------------------------------- #
    def picture_size(self) -> tuple:
        """The pixels a tile's picture fills (inside its ring), to render it at."""
        tile = next(iter(self.tiles.values()), None)
        if tile is None:
            return 0, 0
        r = tile.picture_rect().adjusted(RING, RING, -RING, -RING)
        return int(r.width()), int(r.height())

    def set_picture(self, movie: str, pixmap, caption: str) -> None:
        tile = self.tiles.get(movie)
        if tile is not None:
            tile.set_picture(pixmap, caption)

    def set_current(self, movie: str) -> None:
        """Re-point a dialog that is kept and reopened: the ring, and the focus on it."""
        self.current = movie
        for name, tile in self.tiles.items():
            tile.set_state(current=name == movie)
        if movie in self.movies:
            self._set_focus(self.movies.index(movie))

    # -- focus and picking ---------------------------------------------------------- #
    def _set_focus(self, index: int, *, scroll: bool = True) -> None:
        index = max(0, min(len(self.movies) - 1, index))
        for i, movie in enumerate(self.movies):
            self.tiles[movie].set_state(focused=i == index)
        self.focus_index = index
        if scroll and self.movies:
            self.scroll.ensureWidgetVisible(self.tiles[self.movies[index]], 0, 8)

    def _pick(self, movie: str) -> None:
        self._acted = True
        self.picked.emit(movie)
        self._dismiss()

    def keyPressEvent(self, e) -> None:
        key = e.key()
        step = {Qt.Key_Right: 1, Qt.Key_Left: -1,
                Qt.Key_Down: self.columns, Qt.Key_Up: -self.columns}.get(key)
        if key == Qt.Key_Escape:
            e.accept()
            self._dismiss()
        elif key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            e.accept()
            if self.movies:
                self._pick(self.movies[self.focus_index])
        elif key == Qt.Key_Home:
            e.accept()
            self._set_focus(0)
        elif key == Qt.Key_End:
            e.accept()
            self._set_focus(len(self.movies) - 1)
        elif step is not None:
            e.accept()
            self._set_focus(self.focus_index + step)
        else:
            super().keyPressEvent(e)
