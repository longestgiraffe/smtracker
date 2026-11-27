"""The Metadata tool's panel: a section's settings printed as a receipt.

Thin Qt. Every word and every grouping comes from `smtracker.metadata_view` (the
payload) -- this module only paints it, in IBM Plex Mono (`typography.receipt`), and
puts `metadata_view.receipt_text` on the clipboard when Copy is pressed.

**Each line is its own small widget**, not one painted page, so it can know where its
underlined value is drawn and its height is plain arithmetic -- fixed width, known number
of wrapped lines -- never a height-for-width layout (the `FitLabel` lesson in CLAUDE.md).

**The explanation answers only on the underlined value** (the UI designer: help shows when it is
sought), with the "?" cursor, in the chart tooltip the Spatial Map's bins use
(`dynamics_chart._ChartTooltip`, as a sentence-only card), not the rail's one-line pill.

**A PANEL, NOT A POPOVER** (the UI designer: "I meant a panel, not a tooltip"). `MetadataDock` is a
full-height column left of the toolbar's rail, as in the prototype: a "Metadata" header
over a bottom border, the receipt scrolling under it, a border against the page. The rail
button toggles it; it slides open and shut by animating its WIDTH (`toast.MOTION_MS`, the
workspace's one motion speed) while its content keeps its own fixed width, so the receipt
is uncovered rather than reflowed.

**BESIDE THE PAGE WHERE THERE IS ROOM, OVER IT WHERE THERE IS NOT** (30 Oct 2026, on a
1536 x 816 laptop: "when u open for example the meta data tab even in full screen mode, it
goes out of monitor as well"). An open slot is part of the window's minimum, so opening the
panel raised the workspace's minimum width from 1077 to 1412 px and a maximised window grew
past a smaller screen. `MetadataDock` now opens its slot only where the window has the
panel's width to spare (`screenfit_view.panel_docks`); elsewhere the panel lies over the
page's right edge at its full width and the window's minimum does not move.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from qtpy.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import QApplication, QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ...metadata_view import NO_SECTION_TEXT, receipt_text
from ...screenfit_view import panel_docks
from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.primitives import discard, label
from ..widgets.toast import MOTION_MS, motion_curve

__all__ = ["DOCK_W", "MetadataBody", "MetadataDock", "RECEIPT_W"]

#: The receipt's width in pixels: the panel less its padding. The prototype's panel was
#: 372 px; the UI designer asked for it 10 % narrower, so 335 px (299 + 2 x 18).
RECEIPT_W = 299
#: Receipt text size, and the line pitch it is set on.
_PX = 12
_PITCH = 19
#: The stamp's first line, SMTRACKER: the design's 16 px.
STAMP_PX = 16
#: How long "Copied" shows before the button reads "Copy" again.
COPIED_MS = 1200
#: The panel's padding around the receipt, and so its width (335 px).
DOCK_PAD = 18
DOCK_W = RECEIPT_W + 2 * DOCK_PAD


def _font(weight: str = "normal"):
    return ty.receipt("xs", weight, px=_PX)


#: THE SPOTLIGHT (the UI designer: hover a row and only it stays clear, to walk the settings one at a
#: time). Everything else in the receipt -- rows, stamp, headings, rules, footer -- fades to
#: this; the pointer on blank space or the Copy button leaves everything clear.
FADE_OPACITY = 0.3
#: The fade in and out: a quick hover effect, faster than the 280 ms panel motion.
FADE_MS = 120
#: Leaving a row waits this long before everything brightens, so sweeping across a heading
#: or a rule between two rows never flashes the whole receipt.
REST_DELAY_MS = 80


def _opacity(widget) -> float:
    """How strongly ``widget`` paints: asked of the `ReceiptView` it sits in (1.0 outside one)."""
    view = widget.parentWidget()
    while view is not None and not isinstance(view, ReceiptView):
        view = view.parentWidget()
    return 1.0 if view is None else view.opacity_for(widget)


class _Rule(QWidget):
    """A dashed rule (between a heading and its lines) or a double one (stamp, total)."""

    def __init__(self, t, double: bool = False, parent=None):
        super().__init__(parent)
        self.t, self.double = t, double
        self.setFixedSize(RECEIPT_W, 9)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setOpacity(_opacity(self))
        pen = QPen(QColor(self.t.muted_foreground))
        pen.setWidthF(1.0)
        if not self.double:
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([3, 3])
        p.setPen(pen)
        y = self.height() / 2.0
        if self.double:
            # One pixel apart, CSS `border-top: 3px double` -- the prototype's rule. Three
            # apart read as two separate lines rather than one double rule.
            top = int(y) - 1
            p.drawLine(0, top + 0.5, self.width(), top + 0.5)
            p.drawLine(0, top + 2.5, self.width(), top + 2.5)
        else:
            p.drawLine(0, round(y) + 0.5, self.width(), round(y) + 0.5)
        p.end()


class _Text(QWidget):
    """One line of plain receipt text: the stamp (centred) or a section heading."""

    def __init__(self, t, text: str, *, centred: bool, colour: str, spacing: float = 0.0,
                 weight: str = "normal", px: int = _PX, parent=None):
        super().__init__(parent)
        self.t, self.text, self.centred, self.colour = t, text, centred, colour
        self.font_ = ty.receipt("xs", weight, px=px)
        if spacing:
            self.font_.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
        self.setFixedSize(RECEIPT_W, round(_PITCH * px / _PX))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setOpacity(_opacity(self))
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.setFont(self.font_)
        p.setPen(QColor(self.colour))
        align = (Qt.AlignHCenter if self.centred else Qt.AlignLeft) | Qt.AlignVCenter
        p.drawText(QRectF(0, 0, self.width(), self.height()), align, self.text)
        p.end()


class ReceiptLine(QWidget):
    """``label . . . . . VALUE``: the label left, the value right, dot leaders between.

    A value too wide for the room left of the label wraps onto further lines, right
    aligned -- never elided, since these are the settings a figure was made with. A
    value with a longer explanation (`detail`) is underlined with dots and shows the
    explanation as the styled tooltip on hover.
    """

    def __init__(self, t, line: dict, parent=None):
        super().__init__(parent)
        self.t = t
        self.label_text = line["label"]
        self.value_text = line["value"]
        self.detail = line.get("detail") or ""
        self.font_ = _font()
        fm = QFontMetricsF(self.font_)
        self._label_w = fm.horizontalAdvance(self.label_text)
        self._gap = fm.horizontalAdvance("  ")
        room = RECEIPT_W - self._label_w - 3 * self._gap
        self.value_lines = self._balanced(self.value_text, fm, max(room, RECEIPT_W * 0.35))
        self.setFixedSize(RECEIPT_W, _PITCH * len(self.value_lines))
        #: Where the underlined VALUE text is drawn, one rect per wrapped line: the only
        #: place the explanation answers (the UI designer: help shows when it is sought, not across
        #: the whole line). Recorded by `paintEvent`, which is what draws it.
        self._value_rects: list = []
        self._over_value = False
        if self.detail:
            self.setMouseTracking(True)

    def _view(self):
        view = self.parentWidget()
        while view is not None and not isinstance(view, ReceiptView):
            view = view.parentWidget()
        return view

    def enterEvent(self, e):
        view = self._view()
        if view is not None:
            view.focus_row(self)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._set_over_value(False)
        view = self._view()
        if view is not None:
            view.focus_row(None)
        super().leaveEvent(e)

    def hideEvent(self, e):
        self._set_over_value(False)
        super().hideEvent(e)

    def mouseMoveEvent(self, e):
        if self.detail:
            pos = e.position() if hasattr(e, "position") else QPointF(e.pos())
            self._set_over_value(any(r.contains(pos) for r in self._value_rects))
        super().mouseMoveEvent(e)

    def _set_over_value(self, on: bool) -> None:
        """The "?" cursor and the explanation, together, only over the underlined value."""
        if on == self._over_value:
            return
        self._over_value = on
        view = self._view()
        if on:
            self.setCursor(Qt.WhatsThisCursor)
            if view is not None:
                view.show_detail(self)
        else:
            self.unsetCursor()
            if view is not None:
                view.hide_detail(self)

    @classmethod
    def _balanced(cls, text: str, fm, room: float) -> list[str]:
        """Wrapped into as FEW lines as `room` allows, and then as EVENLY: the
        narrowest width that keeps that line count, so a wrap never leaves one
        orphaned word ("days") on a line of its own."""
        lines = cls._wrap(text, fm, room)
        if len(lines) == 1:
            return lines
        lo, hi = 0.0, room
        for _ in range(16):
            mid = (lo + hi) / 2.0
            if len(cls._wrap(text, fm, mid)) == len(lines):
                hi = mid
            else:
                lo = mid
        return cls._wrap(text, fm, hi)

    @staticmethod
    def _wrap(text: str, fm, room: float) -> list[str]:
        lines, cur = [], ""
        for word in text.split(" "):
            trial = f"{cur} {word}" if cur else word
            if cur and fm.horizontalAdvance(trial) > room:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        return lines

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setOpacity(_opacity(self))
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.setFont(self.font_)
        fm = QFontMetricsF(self.font_)
        ink, muted = QColor(self.t.foreground), QColor(self.t.muted_foreground)
        w = float(self.width())
        base = (_PITCH + fm.ascent() - fm.descent()) / 2.0
        p.setPen(ink)
        p.drawText(0, round(base), self.label_text)
        first = self.value_lines[0]
        first_w = fm.horizontalAdvance(first)
        # Dot leaders as the design draws them: starting one space after the label and
        # running on an even pitch up to one space before the value (CSS `overflow:
        # hidden` on a row of ". "). Lining them up in shared columns instead left a gap
        # after short labels ("Fold", "Cells") as wide as several dots.
        dot = fm.horizontalAdvance(".")
        pitch = fm.horizontalAdvance(". ")
        space = fm.horizontalAdvance(" ")
        start, end = self._label_w + space, w - first_w - space
        p.setPen(muted)
        x = start
        while x + dot <= end:
            p.drawText(QPointF(x, base), ".")
            x += pitch
        p.setPen(ink)
        self._value_rects = []
        for i, text in enumerate(self.value_lines):
            y = base + i * _PITCH
            tw = fm.horizontalAdvance(text)
            self._value_rects.append(QRectF(w - tw, i * _PITCH, tw, _PITCH))
            p.drawText(round(w - tw), round(y), text)
            if self.detail:
                pen = QPen(muted)
                pen.setWidthF(1.0)
                pen.setStyle(Qt.CustomDashLine)
                pen.setDashPattern([1, 2])
                p.setPen(pen)
                uy = round(y + fm.descent()) + 0.5
                p.drawLine(round(w - tw), uy, round(w), uy)
                p.setPen(ink)
        p.end()


class ReceiptView(QWidget):
    """The whole receipt: stamp, double rule, each heading with its rule and lines,
    then the total between double rules."""

    def __init__(self, t, receipt: dict, parent=None):
        super().__init__(parent)
        self.t = t
        #: The row in the spotlight (None at rest), and how far the rest has faded (0..1).
        self.focused: ReceiptLine | None = None
        self.fade = 0.0
        #: The explanation card (created on first use) and the line it is showing for.
        self._tip = None
        self._tip_row = None
        self._fade_anim = QVariantAnimation(self)
        self._fade_anim.setDuration(FADE_MS)
        self._fade_anim.setEasingCurve(motion_curve())
        self._fade_anim.valueChanged.connect(self._on_fade)
        self._fade_anim.finished.connect(self._fade_done)
        self._rest = QTimer(self)
        self._rest.setSingleShot(True)
        self._rest.setInterval(REST_DELAY_MS)
        self._rest.timeout.connect(lambda: self._fade_to(0.0))
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.lines: list[ReceiptLine] = []
        stamp = receipt.get("stamp") or []
        for i, text in enumerate(stamp):
            # The first line is the receipt's NAME, set large as the design has it: 16 px at
            # 0.18 em tracking, regular weight, in the ink; the rest are small and muted.
            lay.addWidget(_Text(t, text, centred=True, px=STAMP_PX if i == 0 else _PX,
                                spacing=0.18 * STAMP_PX if i == 0 else 0.0,
                                colour=t.foreground if i == 0 else t.muted_foreground))
        # The design's air around the rules: a gap under the stamp, more above each heading.
        lay.addSpacing(10)
        lay.addWidget(_Rule(t, double=True))
        for section in receipt.get("sections") or []:
            lay.addSpacing(16)
            # Regular weight, 0.12 em tracking: the design's section heading.
            lay.addWidget(_Text(t, section["heading"], centred=False, spacing=0.12 * _PX,
                                colour=t.foreground))
            lay.addWidget(_Rule(t))
            for line in section["lines"]:
                row = ReceiptLine(t, line)
                self.lines.append(row)
                lay.addWidget(row)
        total = receipt.get("total")
        if total:
            lay.addSpacing(14)
            lay.addWidget(_Rule(t, double=True))
            lay.addSpacing(4)
            row = ReceiptLine(t, total)
            self.lines.append(row)
            lay.addWidget(row)
            lay.addSpacing(4)
            lay.addWidget(_Rule(t, double=True))
        if receipt.get("footer"):
            lay.addSpacing(16)
            self.footer = _Text(t, receipt["footer"], centred=True, colour=t.muted_foreground)
            lay.addWidget(self.footer)
        self.setFixedWidth(RECEIPT_W)


    # -- the explanation ---------------------------------------------------- #
    def show_detail(self, row) -> None:
        """``row``'s explanation in the chart tooltip the Spatial Map's bins use
        (`dynamics_chart._ChartTooltip`, a sentence-only card), centred above the
        hovered value, clamped to the window, below it when there is no room."""
        try:
            if self._tip is None:
                from .dynamics_chart import _ChartTooltip
                self._tip = _ChartTooltip(self.t, self, compact=True, indicator="none")
                # It lives on the WINDOW once shown, so it must go with this receipt, or
                # every redraw of the receipt would leave one behind.
                self.destroyed.connect(self._tip.deleteLater)
            tip_ = self._tip
            tip_.set_content(heading=row.detail, rows=(), colour=self.t.foreground)
            window = self.window()
            if tip_.parentWidget() is not window:
                tip_.setParent(window)
            rect = row._value_rects[0] if row._value_rects else QRectF(row.rect())
            anchor = row.mapTo(window, rect.center().toPoint())
            top = row.mapTo(window, rect.topLeft().toPoint()).y()
            w, h = tip_.width(), tip_.height()
            gap = 4
            x = max(4, min(anchor.x() - w // 2, max(4, window.width() - w - 4)))
            y = top - gap - h
            if y < 4:
                y = top + round(rect.height()) + gap
            tip_.move(x, y)
            tip_.raise_()
            tip_.show()
            self._tip_row = row
        except RuntimeError:
            # the window owns the reparented tip and may tear it down first on close
            self._tip = None

    def hide_detail(self, row=None) -> None:
        if row is not None and row is not self._tip_row:
            return
        self._tip_row = None
        try:
            if self._tip is not None:
                self._tip.hide()
        except RuntimeError:
            self._tip = None

    def hideEvent(self, e):
        self.hide_detail()
        super().hideEvent(e)

    # -- the spotlight ---------------------------------------------------- #
    def focus_row(self, row) -> None:
        """Spotlight ``row`` (everything else fades), or, with None, return to rest after
        `REST_DELAY_MS` -- cancelled if another row is entered first."""
        if row is None:
            self._rest.start()
            return
        self._rest.stop()
        changed = row is not self.focused
        self.focused = row
        if self.fade < 1.0 or self._fade_anim.state() == QVariantAnimation.Running:
            self._fade_to(1.0)
        elif changed:
            self._repaint()

    def opacity_for(self, widget) -> float:
        if widget is self.focused:
            return 1.0
        return 1.0 - self.fade * (1.0 - FADE_OPACITY)

    def _fade_to(self, target: float) -> None:
        if self._fade_anim.state() == QVariantAnimation.Running and \
                self._fade_anim.endValue() == target:
            return
        self._fade_anim.stop()
        if self.fade == target:
            self._fade_done()
            return
        self._fade_anim.setStartValue(self.fade)
        self._fade_anim.setEndValue(target)
        self._fade_anim.start()

    def _on_fade(self, value) -> None:
        self.fade = float(value)
        self._repaint()

    def _fade_done(self) -> None:
        if self._fade_anim.endValue() == 0.0 or self.fade == 0.0:
            self.fade = 0.0
            self.focused = None
        self._repaint()

    def _repaint(self) -> None:
        for child in self.findChildren(QWidget):
            child.update()


class MetadataBody(QWidget):
    """The Metadata tool's body: title, the receipt (or why there is none), Copy."""

    #: The body's height changed (a new receipt): an OPEN panel must re-fit.
    resized = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.receipt: dict | None = None
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self._slot = QVBoxLayout()
        self._slot.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self._slot)
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.addStretch(1)
        self.copy_button = Button(t, "Copy", variant="outline", size="small", glyph="copy")
        self.copy_button.clicked.connect(self.copy)
        foot.addWidget(self.copy_button)
        lay.addLayout(foot)
        self.view: QWidget | None = None
        self.empty_label = None
        self.setFixedWidth(RECEIPT_W)
        self._copied = QTimer(self)
        self._copied.setSingleShot(True)
        self._copied.timeout.connect(lambda: self.copy_button.set_text("Copy"))
        self.set_receipt(None, NO_SECTION_TEXT)

    def set_receipt(self, receipt: dict | None, empty_text: str = NO_SECTION_TEXT) -> None:
        """Show ``receipt``, or ``empty_text`` when there is none to print."""
        self.receipt = receipt
        if self.view is not None:
            discard(self.view)
            self.view = None
        if self.empty_label is not None:
            discard(self.empty_label)
            self.empty_label = None
        if receipt is None:
            self.empty_label = label(empty_text, ty.receipt("xs", px=_PX), self.t.muted_foreground)
            self.empty_label.setWordWrap(False)
            self._slot.addWidget(self.empty_label)
        else:
            self.view = ReceiptView(self.t, receipt)
            self._slot.addWidget(self.view)
        self.copy_button.setVisible(receipt is not None)
        self.adjustSize()
        self.resized.emit()

    def copy(self) -> None:
        """The receipt as aligned plain text, onto the clipboard."""
        if self.receipt is None:
            return
        QApplication.clipboard().setText(receipt_text(self.receipt))
        self.copy_button.set_text("Copied")
        self._copied.start(COPIED_MS)


class MetadataDock(QWidget):
    """The Metadata PANEL: a full-height column between the page and the toolbar's rail.

    TWO WIDGETS. This one is the panel's SLOT in the toolbar's row: closed, it is zero wide and takes no room. The
    panel itself (`panel`: the "Metadata" header, the receipt scrolling under it, a border against the page) is a
    child of the window's content area (`attach`), not of any layout, and lies against the rail's left edge.

    BESIDE THE PAGE OR OVER IT (30 Oct 2026, on a laptop: "when u open for example the meta data tab even in full
    screen mode, it goes out of monitor as well"). The slot used to open to `DOCK_W` always, and a slot's width is
    part of the window's MINIMUM: the workspace needs 1077 px without it and 1412 px with it, so on a 1366 or 1280 px
    screen the window grew past the screen's edge, maximised or not. Now, on opening and on every change of the
    window's width, `screenfit_view.panel_docks` decides: where the window has `DOCK_W` to spare above what the
    sidebar, page and rail need, the slot opens and the page makes room (as before); where it has not, the slot stays
    shut and the panel lies OVER the page's right edge. Either way the receipt keeps its full width -- it is printed
    to `RECEIPT_W` and its right-aligned values would be the first thing a narrower panel cut -- and opening it never
    makes the window wider than it is. Chosen over a narrower panel that scrolls sideways for that reason. While it
    sits beside the page the window cannot be dragged narrower than both, as before; a smaller screen holds the
    window's minimum (`ui.widgets.screenfit`), the window narrows, and the panel moves over the page.

    `set_open` slides the panel to `DOCK_W` (or back) by animating its width -- through the slot when it sits beside
    the page, directly when it lies over it -- while its content keeps `DOCK_W` and is anchored left, so opening
    uncovers the receipt instead of re-wrapping it every frame.
    """

    #: The panel finished opening (True) or closing (False).
    opened = Signal(bool)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._open = False
        #: Whether the panel sits BESIDE the page (its slot open) rather than over it.
        self._docked = True
        #: How wide the panel is drawn now, 0..`DOCK_W` (the animation's value).
        self._shown = 0
        self._host = None
        self._watched = []
        self._recheck_due = False
        self.setStyleSheet("background:transparent;")
        self.setFixedWidth(0)
        # THE PANEL: parented to this slot until `attach` gives it the content area to lie over.
        self.panel = QWidget(self)
        self.panel.setAttribute(Qt.WA_StyledBackground, True)
        self.panel.setObjectName("metadataDock")
        self.panel.setStyleSheet(f"#metadataDock {{ background:{t.background};"
                                 f" border-left:1px solid {t.border}; }}")
        self.panel.hide()
        self.page = QWidget(self.panel)
        self.page.setObjectName("metadataPage")
        self.page.setStyleSheet("#metadataPage { background:transparent; }")
        col = QVBoxLayout(self.page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        head = QWidget()
        head.setObjectName("metadataHead")
        head.setAttribute(Qt.WA_StyledBackground, True)
        head.setStyleSheet(f"#metadataHead {{ background:transparent;"
                           f" border-bottom:1px solid {t.border}; }}")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(DOCK_PAD, 16, DOCK_PAD, 12)
        from .toolbar import _title  # the panel headings' one look; imported late (toolbar imports this)
        self.title = _title(t, "Metadata")
        hl.addWidget(self.title)
        col.addWidget(head)
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { background:transparent; }")
        holder = QWidget()
        holder.setObjectName("metadataHolder")
        holder.setStyleSheet("#metadataHolder { background:transparent; }")
        inner = QVBoxLayout(holder)
        # no right margin: the receipt is anchored left at its own width, and a margin there made the holder wider
        # than the view whenever the vertical scroll bar showed (the GUI sweep's CLIPPED, 30 Oct 2026)
        inner.setContentsMargins(DOCK_PAD, DOCK_PAD, 0, DOCK_PAD + 6)
        inner.setSpacing(0)
        self.body = MetadataBody(t)
        inner.addWidget(self.body, 0, Qt.AlignTop | Qt.AlignLeft)
        inner.addStretch(1)
        self.scroll.setWidget(holder)
        self.scroll.viewport().setStyleSheet("background:transparent;")
        col.addWidget(self.scroll, 1)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(MOTION_MS)
        self._anim.setEasingCurve(motion_curve())
        self._anim.valueChanged.connect(self._step)
        self._anim.finished.connect(self._landed)

    # -- where the panel lies ------------------------------------------------------------------------------ #
    def attach(self, host) -> None:
        """Lie in ``host`` (the window's content area, which holds the toolbar's row): the panel becomes its child,
        above the page and the rail, and follows the slot from now on. Called once by the window."""
        if host is self._host:
            return
        for w in self._watched:
            w.removeEventFilter(self)
        self._host = host
        self.panel.setParent(host)
        self.panel.hide()
        # ABOVE what the content area holds now (sidebar, page, toolbar); what is made later (the toasts) stays above
        self.panel.raise_()
        self._watched = [host] + ([self.parentWidget()] if self.parentWidget() is not None else [])
        for w in self._watched:
            w.installEventFilter(self)
        self._place()

    def _area(self):
        """The widget the panel lies in: the attached host, else the slot's own parent chain's top."""
        return self._host if self._host is not None else self.parentWidget()

    def others_need(self) -> int:
        """What the content area's row needs at least WITHOUT the panel: its minimum less this slot's width."""
        host = self._area()
        if host is None:
            return 0
        return max(0, host.minimumSizeHint().width() - self.minimumWidth())

    def _page_views(self) -> list:
        """The scroll areas showing a page in the content area -- up and down only, their page sized to the view --
        not the panel's own: where one is narrower than its page, the page is CUT at its right edge."""
        from qtpy.QtWidgets import QScrollArea

        host = self._area()
        if host is None:
            return []
        return [area for area in host.findChildren(QScrollArea)
                if area.widget() is not None and area.widgetResizable() and area.isVisibleTo(host)
                and area.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
                and not self.panel.isAncestorOf(area)]

    def page_slack(self) -> float:
        """How much wider than its content's minimum the narrowest page view is now (no page: unlimited). Each page
        is watched from here on, so a page laid out anew (another view of it) is seen at once."""
        slack = float("inf")
        for area in self._page_views():
            area.widget().installEventFilter(self)          # once per object: Qt moves a repeated filter, never adds
            slack = min(slack, area.viewport().width() - area.widget().minimumSizeHint().width())
        return slack

    def docks(self) -> bool:
        """Whether the panel would sit beside the page at the content area's width now (`panel_docks`)."""
        host = self._area()
        if host is None or self._host is None:
            return True             # not attached (a bare slot): it can only open in its slot
        return panel_docks(host.width(), self.others_need(), DOCK_W, page_slack=self.page_slack(),
                           slot=self.width())

    def is_docked(self) -> bool:
        """Whether the open panel sits BESIDE the page (its slot open) rather than over its edge."""
        return self._docked

    def panel_width(self) -> int:
        """How wide the panel is drawn now: `DOCK_W` once open, beside the page or over it."""
        return self.panel.width() if self.panel.isVisible() or self._open else 0

    def _set_docked(self, docked: bool) -> None:
        if docked == self._docked:
            return
        self._docked = docked
        self.setFixedWidth(self._shown if docked else 0)
        self._place()

    def _place(self) -> None:
        """The panel against the rail: its right edge the slot's, its width the slot's beside the page or the
        animation's over it, the content area's full height."""
        host = self._area()
        if host is None or self.panel.parentWidget() is None:
            return
        if self._host is None:          # still a child of the slot
            self.panel.setGeometry(0, 0, self.width(), self.height())
        else:
            right = self.mapTo(host, QPoint(self.width(), 0))
            width = self.width() if self._docked else self._shown
            self.panel.setGeometry(right.x() - width, right.y(), width, self.height())
        self.page.setGeometry(1, 0, DOCK_W - 1, self.panel.height())
        visible = self.panel.width() > 0
        if visible != self.panel.isVisible():
            self.panel.setVisible(visible)

    def eventFilter(self, watched, event) -> bool:
        if watched in self._watched:
            kind = event.type()
            if kind in (QEvent.Resize, QEvent.Move):
                if self._open and watched is self._host and kind == QEvent.Resize:
                    self._set_docked(self.docks() if not self._docked else self._still_docks())
                self._place()
            elif kind == QEvent.LayoutRequest and self._open and watched is self._host:
                # the page's own minimum may have grown (another section): the panel may no longer fit beside it
                self.recheck()
        elif self._open and event.type() == QEvent.Resize:
            self.recheck()                                  # a page watched by `page_slack` was laid out anew
        return False

    def _still_docks(self) -> bool:
        """Beside the page, the panel stays there while the window still has its width to spare and the page's
        view still holds the page."""
        host = self._area()
        return host is not None and panel_docks(host.width(), self.others_need(), self.width(),
                                                page_slack=self.page_slack(), slot=self.width())

    def recheck(self) -> None:
        """Decide again, after the event in hand (the page shown changed, or its size): beside the page or over it."""
        if self._open and not self._recheck_due:
            self._recheck_due = True
            QTimer.singleShot(0, self, self._recheck)

    def _recheck(self) -> None:
        self._recheck_due = False
        if self._open:
            self._set_docked(self._still_docks() if self._docked else self.docks())

    # -- opening and closing ------------------------------------------------------------------------------- #
    def is_open(self) -> bool:
        return self._open

    def set_open(self, on: bool, animated: bool = True) -> None:
        on = bool(on)
        if on == self._open:
            return
        self._open = on
        self._anim.stop()
        if on:
            self._docked = self.docks()
        target = DOCK_W if on else 0
        if not animated:
            self._step(target)
            self.opened.emit(on)
            return
        self._anim.setStartValue(self._shown)
        self._anim.setEndValue(target)
        self._anim.start()

    def _step(self, value) -> None:
        self._shown = int(value)
        if self._docked:
            self.setFixedWidth(self._shown)     # the layout moves the slot; `_place` follows its resize
        else:
            self.setFixedWidth(0)
        self._place()

    def _landed(self) -> None:
        self.opened.emit(self._open)

    def toggle(self) -> None:
        self.set_open(not self._open)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place()

    def moveEvent(self, e):
        super().moveEvent(e)
        self._place()
