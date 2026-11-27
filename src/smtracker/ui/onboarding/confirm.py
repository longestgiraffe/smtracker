"""The screen between choosing a folder and confirming it.

`screens.ConfirmPage` is the prototype's, and it reads a WHOLE project at
construction — every series, every day, every group. So it cannot exist until
the load lands, and the minutes before that need a screen of their own rather
than an empty frame or a modal progress bar over nothing.

This is that screen: what the glob found, which condition is being read, and a
way out. It says everything it can say for free, which on a first open is the
only thing standing between choosing a folder and *"its loading for 10 years and
nothing is opening"*.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QEasingCurve, QSize, Qt, QVariantAnimation, Signal
from qtpy.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets import smoothscroll
from ..widgets.controls import Button
from ..widgets.primitives import (
    Card,
    Progress,
    ShadowHost,
    Spinner,
    band,
    col,
    icon,
    label,
    row,
    scrollbar_css,
)

__all__ = ["LoadingPage"]


class _FoundList(QScrollArea):
    """The found conditions, as many rows tall as there are up to `LoadingPage.MAX_ROWS`, then scrolling.

    ``sizeHint`` is that height, because a QScrollArea's own is a guess from its font; the minimum is three rows,
    so a short window squeezes the list before it squeezes Cancel off the page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.want = 0
        self.least = 0

    def set_heights(self, want: int, least: int) -> None:
        self.want, self.least = want, least
        self.setMinimumHeight(least)
        self.setMaximumHeight(want)
        self.updateGeometry()

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.want)

    def minimumSizeHint(self):
        return QSize(super().minimumSizeHint().width(), self.least)


class LoadingPage(ShadowHost):
    """What was found, and what is being read. Emits `cancelled`."""

    cancelled = Signal()
    #: the user answered "leave them out" to `ask_leave_out`
    leave_out_chosen = Signal()

    def __init__(self, t, root, parent=None):
        super().__init__(t, parent)
        self.t = t
        self.root = Path(root)
        #: `onboarding_view.load_failure`'s words once the folder was refused, else None
        self.failure: dict | None = None
        self.setStyleSheet(f"background:{t.background};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 40, 48, 40)
        lay.setSpacing(0)
        lay.addStretch(1)

        root = self.root
        head = label(root.name or str(root), ty.ui("2xl", "semibold"), t.foreground)
        head.setAlignment(Qt.AlignHCenter)
        path = label(str(root), ty.ui("xs"), t.muted_foreground)
        path.setAlignment(Qt.AlignHCenter)
        lay.addWidget(col(head, path, spacing=4))
        lay.addSpacing(24)

        card = Card(t)
        card.setMaximumWidth(560)
        card.setMinimumWidth(420)
        self._card_lay = QVBoxLayout(card)
        self._card_lay.setContentsMargins(0, 0, 0, 0)
        self._card_lay.setSpacing(0)
        self._summary = label("looking…", ty.ui("xs"), t.muted_foreground)
        self._card_lay.addWidget(band(
            row(icon("folder-open", 14, t.muted_foreground),
                # `text-sm`, not `text-base`. This card is a summary under the
                # project's own name, not a section of the page; at 16 it was
                # the same step as the heading above it and competed with it.
                label("What is in this folder", ty.ui("sm", "semibold"),
                      t.foreground),
                None, self._summary, spacing=8, margins=(16, 12, 16, 12)),
            "foundHead",
            f"background:transparent; border-bottom:1px solid {t.border};"))
        #: The rows arrive one at a time from `scan.Scout` -- see `add_condition`.
        self._rows: list = []
        # THE ROWS SCROLL (reported 25 Oct 2026: "when i load whole MSD folder on F, the list gets longer and the
        # cancel button disapeears, maybe we can introduce that new row is coming ro scrollking"). Added straight
        # into the card, every condition made the page taller, and a folder of dozens pushed Cancel below the
        # screen. They sit in a list `MAX_ROWS` tall that follows the newest row as it arrives -- unless the user
        # has scrolled up to read, which it leaves alone -- and each new row fades in.
        self._rows_host = QWidget()
        self._rows_host.setStyleSheet("background:transparent;")
        self._rows_lay = QVBoxLayout(self._rows_host)
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        self._rows_lay.setSpacing(0)
        self._rows_lay.addStretch(1)
        self.found_list = _FoundList()
        self.found_list.setWidget(self._rows_host)
        self.found_list.setWidgetResizable(True)
        self.found_list.setFrameShape(QScrollArea.NoFrame)
        self.found_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # shadcn's bar set here too, not only by the window's cascade, so the list is right wherever it is shown
        self.found_list.setStyleSheet("QScrollArea { background:transparent; }" + scrollbar_css(t))
        self.found_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._card_lay.addWidget(self.found_list)
        self.found_list.setVisible(False)                    # no rows yet: the card is its header
        self._scroller = smoothscroll.attach(self.found_list, background=t.card)
        self._follow = QVariantAnimation(self)
        self._follow.setDuration(220)
        self._follow.setEasingCurve(QEasingCurve.OutCubic)
        self._follow.valueChanged.connect(lambda v: self.found_list.verticalScrollBar().setValue(int(v)))
        #: the list was at its end when the last row came, so it follows to the new end (`_on_range`)
        self._pin_end = True
        self.found_list.verticalScrollBar().rangeChanged.connect(self._on_range)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(card, alignment=Qt.AlignHCenter)
        lay.addWidget(holder)
        lay.addSpacing(20)

        # shadcn's `Spinner` -- `Loader2Icon`, `size-4`, `animate-spin` -- ABOVE
        # the line rather than beside it. Centred under a centred card, the two
        # read as one column; on the same row they would shift the text off the
        # axis everything else on this screen is aligned to.
        # STROKE 2, which is lucide's own and what `size-4 animate-spin` gets on
        # the web. `icons.STROKE` is 1.25 here -- a deliberate departure for
        # glyphs sitting beside body text, where Qt's small-grid rasterisation
        # renders the same number heavier than a browser does. A spinner is not
        # beside text, it is the only mark on its line, and at 1.25 it reads as
        # a grey hairline rather than as a thing turning.
        # 28 px, not shadcn's `size-4`. The registry's spinner sits INSIDE a
        # button or beside a line of text, where 16 is right; this one is the
        # only thing moving on an otherwise still screen, and at 16 it was a
        # detail rather than the answer to "is this doing anything".
        self.spinner = Spinner(t.muted_foreground, 28, stroke=2.0)
        spin_row = QWidget()
        spin_row.setStyleSheet("background:transparent;")
        sl = QVBoxLayout(spin_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.spinner, alignment=Qt.AlignHCenter)
        lay.addWidget(spin_row)
        lay.addSpacing(10)

        # THE WORDS WRAP IN A COLUMN AT MOST `TEXT_W` WIDE AND NEVER ASK FOR WIDTH (found 3 Nov 2026, tester B2): a
        # refusal on one unwrapped line, a path with no space to break at, pushed the window to 1428 px on a 1366 px
        # screen with the text cut at both ends, and it stayed that wide.
        self.status = self._words("Reading…", ty.ui("sm"), t.foreground)
        self.bar = Progress(t, 0.0, height=8, width=320)
        self.bar.setVisible(False)
        self.note = self._words(
            "Reading the tracker's tables. A project opened before is served "
            "from its cache and appears at once.", ty.ui("xs"),
            t.muted_foreground)
        #: What to DO about a refusal (`onboarding_view.load_failure`), under its reason; hidden while reading.
        self.advice = self._words("", ty.ui("sm", "medium"), t.foreground)
        self.advice.setVisible(False)
        words = QWidget()
        words.setStyleSheet("background:transparent;")
        words.setMaximumWidth(self.TEXT_W)
        words.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        wl = QVBoxLayout(words)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wl.addWidget(self.status)
        wl.addSpacing(10)
        wl.addWidget(self.bar, alignment=Qt.AlignHCenter)
        wl.addWidget(self.note)
        wl.addSpacing(10)
        wl.addWidget(self.advice)
        centred = QHBoxLayout()
        centred.setContentsMargins(0, 0, 0, 0)
        centred.addStretch(1)
        centred.addWidget(words, 100)
        centred.addStretch(1)
        lay.addLayout(centred)
        lay.addSpacing(16)

        # THE QUESTION'S ANSWER (`ask_leave_out`): added to the layout before it is hidden
        self.leave_out = Button(t, "Leave out and open the rest")
        self.leave_out.clicked.connect(self.leave_out_chosen.emit)
        lay.addWidget(self.leave_out, alignment=Qt.AlignHCenter)
        self.leave_out.setVisible(False)
        lay.addSpacing(8)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self.cancelled.emit)
        lay.addWidget(cancel, alignment=Qt.AlignHCenter)
        lay.addStretch(1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    #: The widest the status, note and advice lines run before they wrap.
    TEXT_W = 640

    @staticmethod
    def _words(text: str, font, colour: str):
        """A centred line that wraps and never asks for width (an ignored width: the column gives it its own)."""
        w = label(text, font, colour)
        w.setAlignment(Qt.AlignHCenter)
        w.setWordWrap(True)
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        w.setMinimumWidth(0)
        return w

    #: The found list's height in rows before it scrolls, and the fewest a short window may squeeze it to.
    MAX_ROWS = 7
    MIN_ROWS = 3

    # -- filling in, as the workers report ----------------------------------- #

    def add_condition(self, name: str, movies: int) -> None:
        """One more row in the list, from `scan.Scout.found`: faded in, and scrolled to when the list was
        following (at its end)."""
        t = self.t
        bar = self.found_list.verticalScrollBar()
        following = not self._rows or bar.value() >= bar.maximum() - 2 or self._follow.state() == \
            QVariantAnimation.Running
        line = band(row(label(name, ty.ui("sm", "medium"), t.foreground), None,
                        label(f"{movies} series", ty.ui("sm", tabular=True),
                              t.muted_foreground),
                        spacing=8, margins=(16, 9, 16, 9)),
                    f"foundRow_{len(self._rows)}",
                    f"background:transparent;"
                    f" border-bottom:1px solid {t.border};")
        self._rows.append(line)
        self._rows_lay.insertWidget(self._rows_lay.count() - 1, line)
        self.found_list.setVisible(True)
        row_h = line.sizeHint().height()
        n = len(self._rows)
        self.found_list.set_heights(min(n, self.MAX_ROWS) * row_h, min(n, self.MIN_ROWS) * row_h)
        self._fade_in(line)
        # the range grows only once the layout has placed the row: `_on_range` glides to the new end then
        self._pin_end = following
        series = sum(1 for _ in self._rows)
        self._summary.setText(f"{len(self._rows)} condition"
                              f"{'s'[:len(self._rows) != 1]} so far")
        self.status.setText(f"Reading {name} …")
        return series

    def _fade_in(self, line: QWidget) -> None:
        effect = QGraphicsOpacityEffect(line)
        effect.setOpacity(0.0)
        line.setGraphicsEffect(effect)
        fade = QVariantAnimation(line)
        fade.setDuration(260)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.OutCubic)
        fade.valueChanged.connect(effect.setOpacity)
        # the effect is dropped at the end: a row left wearing one is drawn through an offscreen pass for good
        fade.finished.connect(lambda: line.setGraphicsEffect(None))
        fade.start(QVariantAnimation.DeleteWhenStopped)

    def _on_range(self, _low: int, high: int) -> None:
        if self._pin_end and high > 0:
            self._glide_to(high)

    def _glide_to(self, target: int) -> None:
        bar = self.found_list.verticalScrollBar()
        self._follow.stop()
        self._follow.setStartValue(bar.value())
        self._follow.setEndValue(target)
        self._follow.start()

    def set_summary(self, text: str) -> None:
        self._summary.setText(text)

    def say(self, text: str) -> None:
        """The status line, VERBATIM. Every step writes its own sentence rather
        than having one prefix bolted onto it -- which is how a progress line
        came out reading "Reading building the screen …"."""
        self.status.setText(text)

    def on_progress(self, name: str, done: int, total: int) -> None:
        """`total` of 0 means a NAMED STEP, printed as it was given.

        A count gets "Reading x … (3 of 30)". A step does not: it already is a
        sentence, and bolting the prefix on produced "Reading assigning tracks
        to cells" -- the second time this screen has done that to a caller's
        text, which is why the two cases are separated here rather than at each
        call site.
        """
        if not total:
            self.status.setText(f"{name[:1].upper()}{name[1:]} …")
        else:
            self.status.setText(f"Reading {name} … ({done} of {total})")
        # THE BAR IS THE PART YOU READ WITHOUT READING. A name and a count tell
        # you the tool is alive; a bar tells you how much is left, which is the
        # question a six-second wait actually asks.
        if total:
            self.bar.setVisible(True)
            self.bar.set_value(done / total)

    def ask_leave_out(self, title: str, body: str) -> None:
        """Movies tracked without a value every D scales with (`onboarding_view.leave_out_question`): the load
        stops and ASKS, rather than "Nothing was loaded"."""
        self.spinner.setVisible(False)
        self.bar.setVisible(False)
        self.status.setText(title)
        self.note.setText(body)
        self.leave_out.setVisible(True)

    def resume(self, text: str = "Reading the tracker's tables…") -> None:
        """Back to reading, after the answer."""
        self.leave_out.setVisible(False)
        self.advice.setVisible(False)
        self.status.setFont(ty.ui("sm"))
        self.spinner.setVisible(True)
        self.status.setText(text)
        self.note.setText("")

    def on_failed(self, message: str) -> None:
        """NOT A MESSAGE BOX: the folder is named on screen above it, and the
        next thing the user wants is Cancel, which is already there."""
        # THE SPINNER STOPS. A thing still turning under a failure message says
        # the work is continuing, which is the one thing that is certainly not
        # happening.
        from ...onboarding_view import holds_acquisitions, load_failure

        self.spinner.setVisible(False)
        self.bar.setVisible(False)
        # THE WHOLE MESSAGE, its cause in plain words and what to do about it (tester B2, 3 Nov 2026): the first line
        # cut at 200 characters dropped the reason itself, and every refusal advised "Choose a different folder"
        nothing_here = "nor any folder under it contains" in message or message.startswith("No conditions found")
        words = load_failure(message, self.root,
                             has_acquisitions=nothing_here and holds_acquisitions(self.root))
        self.status.setFont(ty.ui("base", "semibold"))
        self.status.setText(words["title"])
        self.note.setText(words["detail"])
        self.note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.note.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.advice.setText(words["advice"])
        self.advice.setVisible(bool(words["advice"]))
        self._summary.setText(words["summary"])
        self.failure = words
