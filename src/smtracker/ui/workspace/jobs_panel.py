"""The jobs running in the background, at the foot of the analysis sidebar.

Asked (2 Nov 2026): "maybe if i run multiple things like sqd, hmm, appd etc, we can have on the left bar like an
info that something (what and roughly percent) is running in the background, so people not wondering why its
slow". Sections run side by side, and the only sign of one computing was the rail's Compute mark, which follows the
page on screen: a Run left behind in another section ran unseen.

THIN QT: the words are `run_progress.sidebar_job` (one per job in the workspace's register,
`WorkspaceWindow._running`); this paints them. Hidden while nothing runs.
- Each job a row: a spinner, the job's title and its percent, a bar (none where the size is unknown, QC's one job),
  "2 of 5 conditions · 1:12" with a clock that moves every second. Clicking a row opens its section (``picked``).
- The rail collapsed to icons: one spinner, its tooltip naming every job (`run_progress.sidebar_jobs_words`).
- The rows sit in a scroll area: many jobs at once shrink the list, never grow the window's minimum height past a
  laptop's screen (the trap `ui.widgets.screenfit` exists for). Its SIZE HINT is every row and its minimum one row
  (`_JobsScroll`, 3 Nov 2026, reported: at 768 px one job showed and the rest hid behind a scroll bar, while the
  sidebar had room for all of them -- a plain scroll area does not pass its list's new height on, so the sidebar
  kept the one-row height it measured first).
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QSize, Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ...run_progress import sidebar_jobs_words
from ..theme import typography as ty
from ..widgets import tooltip
from ..widgets.primitives import HoverFill, Progress, Spinner, discard, label, mix

__all__ = ["JobRow", "JobsPanel"]

#: The sidebar's own measures (`sidebar.rem` at its 16 px root), spelled here so this module does not import the
#: sidebar that imports it.
_PAD, _GAP, _ICON = 8, 8, 32
#: The heading, as the sidebar's group labels draw theirs (`sidebar.LABEL_H`, text-xs medium at 70 % ink): Running
#: is one more group of the rail (4 Nov 2026; it was 14 px in a 32 px row, louder than "Data" and "Analyses").
_HEAD_H, _HEAD_PX, _HEAD_WGHT = 28, 12, 410
_SPIN = 14
#: The bar under a job's title: thinner than shadcn's 8 px, a row of a sidebar, not a page's progress.
_BAR_H = 4
_LABEL_ALPHA = 0.70


class JobRow(QWidget):
    """One job: spinner, title, percent; its bar; its count and clock. A click opens its section."""

    clicked = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        #: The view a click opens (`JobsPanel.set_jobs`).
        self.view = ""
        self.setObjectName(f"job{id(self)}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self._fill = HoverFill(
            self, lambda c: (f"QWidget#{self.objectName()} {{ background:{c}; border:none;"
                             f" border-radius:{t.radius_sm:g}px; }}"),
            t.sidebar, t.sidebar_accent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(_PAD, 6, _PAD, 6)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(_GAP)
        self.spinner = Spinner(t.sidebar_foreground, _SPIN)
        head.addWidget(self.spinner)
        self.title = label("", ty.ui("sm", "medium"), t.sidebar_foreground)
        head.addWidget(self.title, 1)
        self.percent = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        head.addWidget(self.percent)
        lay.addLayout(head)
        self.bar = Progress(t, 0.0, height=_BAR_H)
        lay.addWidget(self.bar)
        self.detail = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        lay.addWidget(self.detail)

    def set_job(self, job: dict) -> None:
        self.title.setText(str(job.get("title", "")))
        self.percent.setText(str(job.get("percent", "")))
        self.detail.setText(str(job.get("detail", "")))
        fraction = job.get("fraction")
        self.bar.setVisible(fraction is not None)
        if fraction is not None:
            self.bar.set_value(float(fraction))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class _JobsScroll(QScrollArea):
    """The list's scroll area: as tall as every row where the sidebar has the room (its hint), one row where it has
    not (its minimum), and told when the list grows or shrinks."""

    def setWidget(self, widget) -> None:
        super().setWidget(widget)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.widget() and event.type() == QEvent.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(watched, event)

    def sizeHint(self) -> QSize:
        inner = self.widget()
        if inner is None:
            return super().sizeHint()
        return QSize(super().sizeHint().width(), inner.sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        inner = self.widget()
        rows = inner.findChildren(JobRow) if inner is not None else []
        first = min((r.sizeHint().height() for r in rows), default=0)
        return QSize(super().minimumSizeHint().width(), first)


class JobsPanel(QWidget):
    """The Running list: a heading and a row per job, or one spinner on the collapsed rail."""

    #: The view of the job clicked (`WorkspaceWindow._JOB_VIEW`, else its section's first page).
    picked = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._compact = False
        self._jobs: list = []
        self._rows: dict = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, _PAD)
        lay.setSpacing(2)
        rest = mix(t.sidebar_foreground, t.sidebar, _LABEL_ALPHA)
        self.heading = label("Running", ty.ui("xs", "medium", px=_HEAD_PX, wght=_HEAD_WGHT), rest)
        self.heading.setFixedHeight(_HEAD_H)
        self.heading.setStyleSheet(f"padding-left:{_PAD}px; background:transparent; color:{rest};")
        lay.addWidget(self.heading)
        self._list = QWidget()
        self._list.setStyleSheet("background:transparent;")
        self._list_lay = QVBoxLayout(self._list)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.setSpacing(2)
        self._scroll = _JobsScroll()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # a click on a job leaves the keys with the page, as the rest of the rail does (`sidebar._BodyScroll`)
        self._scroll.setFocusPolicy(Qt.NoFocus)
        self._scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.setWidget(self._list)
        lay.addWidget(self._scroll)
        # the collapsed rail's form: one spinner in an icon-wide cell, every job in its tooltip
        self.compact = QWidget()
        self.compact.setFixedSize(_ICON, _ICON)
        self.compact.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(self.compact)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(Spinner(t.sidebar_foreground, _SPIN), 0, Qt.AlignCenter)
        self._tip = tooltip.attach(t, self.compact, "", side="right", enabled=lambda: self._compact)
        lay.addWidget(self.compact)
        self.compact.setVisible(False)
        self.setVisible(False)

    def jobs(self) -> list:
        return list(self._jobs)

    def set_jobs(self, jobs) -> None:
        """``jobs``: ``[{"key", "view", **run_progress.sidebar_job(...)}]`` in the order they started."""
        jobs = [dict(job) for job in jobs or ()]
        keys = [str(job["key"]) for job in jobs]
        if keys != list(self._rows):
            for row in self._rows.values():
                discard(row)
            self._rows = {}
            for job in jobs:
                row = JobRow(self.t)
                row.clicked.connect(lambda r=row: self.picked.emit(r.view))
                # into the layout BEFORE any visibility is set (`test_no_stray_windows`)
                self._list_lay.addWidget(row)
                self._rows[str(job["key"])] = row
        for job in jobs:
            row = self._rows[str(job["key"])]
            row.view = str(job["view"])
            row.set_job(job)
        self._jobs = jobs
        self._tip.text = sidebar_jobs_words(jobs)
        self.setVisible(bool(jobs))

    def set_compact(self, on: bool) -> None:
        """The rail collapsed to icons (one spinner) or wide again (the list)."""
        self._compact = bool(on)
        self.heading.setVisible(not on)
        self._scroll.setVisible(not on)
        self.compact.setVisible(on)
