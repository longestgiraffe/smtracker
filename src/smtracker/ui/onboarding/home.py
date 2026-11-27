"""The first screen of the analysis flow: choose a folder, or reopen one.

Cold on purpose — nothing is loaded, so nothing is claimed. The two halves say
the same thing at different costs: the drop zone is for a folder the tool has
never seen, the recents card is for one it has.

**The recents are real, and they are the session's.** They live in
`smtracker.session`, beside `last_project`, so there is one per-user JSON file
rather than a second one invented here; `remember_project` moves a reopened
folder to the top instead of adding a row. Each entry carries the conditions and
series counted when it was last opened, which is what lets this screen describe
a project without globbing every drive to draw a list — and which is why the
counts are labelled as of the last open rather than presented as current.

**A folder that has gone is shown, not hidden.** An external drive that is not
plugged in is the ordinary case here, and a list that silently shortened would
leave a user wondering whether the tool had forgotten the project or the disk
had. The row says so and refuses the click.

**A row can be taken off the list** (its x, 14 Oct 2026: "to remove links to recent data
not needed anymore"): `session.forget_project` drops the entry and nothing else -- the
folder is never touched -- and a missing folder's row goes the same way.

**The drop target is real too.** The prototype's zone says "or drop a folder
here", and a label promising something that does nothing is the failure this
project keeps recording, so the zone accepts a drag.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QFontMetrics
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..theme import icons
from ..theme import typography as ty
from ..widgets import tooltip
from ..widgets.controls import Button
from ..widgets.primitives import Card, FitScroll, ShadowHost, band, col, discard, icon, label, row
from .panels import section_font

__all__ = ["HomePage"]


def _ago(iso: str) -> str:
    """`opened yesterday`, from an ISO stamp. Never raises on a bad one."""
    from datetime import datetime, timezone

    try:
        when = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - when).days
    if days <= 0:
        return "opened today"
    if days == 1:
        return "opened yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


class _PathLabel(QLabel):
    """A folder path shortened in the MIDDLE to the room it has, so the drive and the
    folder's own name both stay readable; the whole path is the hover. A path has no
    spaces to wrap at, and cut at the card's edge it lost its end."""

    def __init__(self, text: str, font, colour: str):
        super().__init__()
        self._full = text
        self.setFont(font)
        self.setStyleSheet(f"color:{colour}; background:transparent;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.setText(QFontMetrics(self.font()).elidedText(self._full, Qt.ElideMiddle,
                                                         max(0, self.width())))


class HomePage(ShadowHost):
    """Choose a project folder. Emits `chosen(Path)`; loads nothing itself.

    A `ShadowHost` because the recents card is the one Card on this screen and
    the 48 px page margin is more room than `shadow-sm` needs -- the shadow has
    to be painted by the parent either way, since Qt clips a widget to its own
    rectangle.
    """

    chosen = Signal(object)
    #: The header's buttons (tester B22, 3 Nov 2026: the manual said every window has Options and Back to menu, and
    #: this one had neither, nor Help; closing it was the only way out, and it quit the program).
    help_requested = Signal()
    options_requested = Signal()
    menu_requested = Signal()

    def __init__(self, t, recents=None, parent=None):
        super().__init__(t, parent)
        self.t = t
        #: The styled tooltips (each row's x), held so they live as long as their rows.
        self._tips: list = []
        self.setStyleSheet(f"background:{t.background};")
        self.setAcceptDrops(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 40, 48, 40)
        lay.setSpacing(0)

        # THE OTHER WINDOWS' HEADER BUTTONS, at the top right, in their order
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addStretch(1)
        self.help_button = Button(t, "Help", size="sm", glyph="info")
        self.help_button.setToolTip("The manual: on the website, or the copy in SMTracker.")
        self.help_button.clicked.connect(self.help_requested.emit)
        top.addWidget(self.help_button)
        self.options_button = Button(t, "Options", size="sm", glyph="sliders-horizontal")
        self.options_button.setToolTip("Set the defaults a new analysis starts from, such as the SQD fit settings, "
                                       "the cell filter tolerance and the frame interval.")
        self.options_button.clicked.connect(self.options_requested.emit)
        top.addWidget(self.options_button)
        self.menu_button = Button(t, "Back to menu", size="sm")
        self.menu_button.setToolTip("Go back to the stage menu, for preparation, tracking or cell outlines.")
        self.menu_button.clicked.connect(self.menu_requested.emit)
        top.addWidget(self.menu_button)
        lay.addLayout(top)
        lay.addSpacing(8)

        from ...appicon import logo_pixmap

        # THE WORDMARK (13 Nov 2026), in the theme's own version; the typed name where it did not ship
        logo = logo_pixmap(t.color_scheme == "dark", 34, self.devicePixelRatioF())
        if logo is not None:
            head = QLabel()
            head.setPixmap(logo)
            head.setAccessibleName("SMTracker")
        else:
            head = label("SMTracker", ty.ui("2xl", "semibold"), t.foreground)
        head.setAlignment(Qt.AlignHCenter)
        sub = label("Diffusive-state fractions from single-molecule tracks",
                    ty.ui("sm"), t.muted_foreground)
        sub.setAlignment(Qt.AlignHCenter)
        lay.addWidget(col(head, sub, spacing=5))
        lay.addSpacing(30)

        body = QHBoxLayout()
        body.setSpacing(20)
        body.addWidget(self._drop_zone(), 2)
        self._body = body
        self._recents_card = self._recents(recents)
        body.addWidget(self._recents_card, 1)
        lay.addLayout(body, 1)

    def refresh_recents(self) -> None:
        """Re-read the list: coming BACK to this screen after opening a project must show
        that project at the top, not the list as it was when the window was built."""
        self._tips = []
        fresh = self._recents(None)
        self._body.replaceWidget(self._recents_card, fresh)
        discard(self._recents_card)
        self._recents_card = fresh

    # -- the two halves ------------------------------------------------------ #

    def _drop_zone(self) -> QWidget:
        t = self.t
        zone = QFrame()
        zone.setObjectName("dropZone")
        # Scoped by object name: a bare `QFrame` selector is inherited by every
        # descendant, which draws the dashed border around each label inside it
        # as well as around the zone.
        self._zone = zone
        self._paint_zone(False)
        lay = QVBoxLayout(zone)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(10)
        lay.addStretch(1)

        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addStretch(1)
        hl.addWidget(icon("folder-open", 28, t.muted_foreground))
        hl.addStretch(1)
        lay.addWidget(holder)
        lay.addSpacing(6)

        # 16, NOT `lg`. This screen already carries a 24 px title above it
        # ("SMTracker"), and 18 against that 24 is a ratio of 1.125 -- MEASURED,
        # 1.45 px of cap height, which does not read as a level. `lg` is the top
        # of a MODAL's ladder, where there is no 24; on a screen that has one, a
        # heading drops straight to the section step. See `typography.USED_STEPS`.
        title = label("Open a project folder", section_font(), t.foreground)
        title.setAlignment(Qt.AlignHCenter)
        lay.addWidget(title)

        note = label("Pick the folder holding your tracking results. Each "
                     "subfolder becomes a condition. Your movies and tracks are "
                     "not changed; a replicate map and a cache are added beside them.",
                     ty.ui("sm"), t.muted_foreground)
        note.setAlignment(Qt.AlignHCenter)
        note.setWordWrap(True)
        note.setFixedWidth(430)
        # A wrapped QLabel in a stretch layout is given its SINGLE-LINE height
        # unless told otherwise, so the second line is simply cut off. Its height
        # at its width, not a number: 44 px held two lines, and the text grew to
        # three (cut on a short window, 30 Oct 2026).
        note.setMinimumHeight(note.heightForWidth(430))
        lay.addWidget(note, alignment=Qt.AlignHCenter)
        lay.addSpacing(8)

        choose = Button(t, "Choose folder…", variant="default")
        choose.clicked.connect(self.browse)
        lay.addWidget(choose, alignment=Qt.AlignHCenter)
        drop = label("or drop a folder here", ty.ui("xs"), t.muted_foreground)
        drop.setAlignment(Qt.AlignHCenter)
        lay.addWidget(drop)
        lay.addStretch(1)
        return zone

    def _paint_zone(self, lit: bool) -> None:
        t = self.t
        edge = t.ring if lit else t.input
        fill = t.accent if lit else t.subtle
        self._zone.setStyleSheet(
            f"QFrame#dropZone {{ background:{fill}; border:1px dashed {edge};"
            f" border-radius:{t.radius_lg:g}px; }}")

    def _recents(self, recents) -> QWidget:
        t = self.t
        entries = list(recents if recents is not None else _load_recents())
        card = Card(t)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # `section_font()`, not a second spelling of it. "Analysis groups",
        # "Replicate days" and this are ONE role, and `groups.Lane` names them as
        # one set -- so they resolve through one function rather than through two
        # constants and a literal that can drift apart.
        head = row(label("Recent projects", section_font(), t.foreground),
                   None,
                   label(str(len(entries)), ty.ui("xs"), t.muted_foreground),
                   spacing=8, margins=(16, 12, 16, 12))
        lay.addWidget(band(head, "recentHead", f"background:transparent;"
                           f" border-bottom:1px solid {t.border};"))
        if not entries:
            # AN EMPTY LIST SAYS IT IS EMPTY. A card with a heading and nothing
            # under it reads as a list that failed to load.
            empty = label("Projects you open will appear here.", ty.ui("sm"),
                          t.muted_foreground)
            empty.setWordWrap(True)
            lay.addWidget(col(empty, spacing=0, margins=(16, 14, 16, 14)))
        # THE ROWS SCROLL (30 Oct 2026). Added straight into the card, nine recent
        # projects asked for 688 px and held the whole start screen at 850 px --
        # taller than a 1536 x 816 laptop, so the maximised window left the screen.
        # `FitScroll`: the whole list where there is room, a few rows where there is not.
        holder = QWidget()
        holder.setObjectName("recentRows")
        holder.setStyleSheet("#recentRows { background:transparent; }")
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        for entry in entries:
            rows.addWidget(self._recent_row(entry))
        rows.addStretch(1)
        self.recent_scroll = FitScroll()
        self.recent_scroll.setWidget(holder)
        self.recent_scroll.setWidgetResizable(True)
        self.recent_scroll.setFrameShape(QFrame.NoFrame)
        self.recent_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recent_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.recent_scroll.viewport().setStyleSheet("background:transparent;")
        lay.addWidget(self.recent_scroll, 1)
        # at least its rows' width: a bare 280 REPLACED the layout's own minimum, so the rows were laid out wider
        # than the card's view and the x at each row's end was cut (the GUI sweep's CLIPPED, 30 Oct 2026)
        card.setMinimumWidth(max(280, lay.minimumSize().width()))
        return card

    def _recent_row(self, entry: dict) -> QWidget:
        t = self.t
        path = Path(str(entry.get("path", "")))
        here = path.is_dir()
        ink = t.foreground if here else t.muted_foreground
        top = [label(path.name or str(path), ty.ui("sm", "medium"), ink)]
        if not here:
            top += [icon("alert", icons.XS, t.muted_foreground),
                    label("folder not found", ty.ui("xs"), t.muted_foreground)]
        where = _PathLabel(str(path), ty.ui("xs"), t.muted_foreground)
        self._tips.append(tooltip.attach(t, where, str(path)))
        lines = [row(*top, None, spacing=7), where]
        if here:
            facts = _facts(entry)
            when = _ago(str(entry.get("opened_at", "")))
            tail = " · ".join(x for x in (facts, when) if x)
            if tail:
                lines.append(label(tail, ty.ui("xs"), t.muted_foreground))
        remove = Button(t, "", size="sm", variant="ghost", glyph="x")
        remove.setObjectName("recentRemove")
        remove.setCursor(Qt.PointingHandCursor)
        self._tips.append(tooltip.attach(
            t, remove, "Remove from this list (the folder itself is not touched)"))
        remove.clicked.connect(lambda p=path: self._forget(p))
        item = band(row(col(*lines, spacing=2), None, remove, spacing=8,
                        margins=(16, 10, 10, 10)),
                    f"recent_{abs(hash(str(path)))}",
                    f"background:transparent; border-bottom:1px solid {t.border};")
        item.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        item.remove_button = remove
        if here:
            item.setCursor(Qt.PointingHandCursor)

            def release(e, p=path, item=item, remove=remove):
                # The x's release travels on to this row (`Button` passes it up), and a
                # project must not open on its way off the list.
                at = e.position().toPoint() if hasattr(e, "position") else e.pos()
                hit = item.childAt(at)
                if hit is not None and (hit is remove or remove.isAncestorOf(hit)):
                    return
                self.chosen.emit(p)

            item.mouseReleaseEvent = release
        return item

    def _forget(self, path) -> None:
        """Take one project off the list; the folder itself is not touched."""
        from ...session import forget_project

        forget_project(path)
        # After the click has finished travelling: the row it landed on is replaced.
        QTimer.singleShot(0, self.refresh_recents)

    # -- choosing ------------------------------------------------------------ #

    def browse(self) -> None:
        """The folder picker, opened from INSIDE the window rather than ahead of it.

        The house rule for every stage: a modal dialog in front of an
        application the user has not seen yet is what the window-first design
        removed. Cancelling leaves this screen exactly as it was.
        """
        start = ""
        for entry in _load_recents():
            if Path(str(entry.get("path", ""))).is_dir():
                start = str(Path(str(entry["path"])).parent)
                break
        picked = QFileDialog.getExistingDirectory(
            self, "Choose a project folder", start)
        if picked:
            self.chosen.emit(Path(picked))

    # -- the drop target ----------------------------------------------------- #

    def dragEnterEvent(self, e):
        if self._dropped_folder(e) is not None:
            self._paint_zone(True)
            e.acceptProposedAction()

    def dragLeaveEvent(self, _e):
        self._paint_zone(False)

    def dropEvent(self, e):
        self._paint_zone(False)
        folder = self._dropped_folder(e)
        if folder is not None:
            e.acceptProposedAction()
            self.chosen.emit(folder)

    @staticmethod
    def _dropped_folder(event):
        """The one folder being dragged, or None — files and multi-drops included.

        A drop of three folders has no obvious meaning here (this screen opens
        ONE project), so it is refused at the drag stage, where the cursor still
        says so, rather than accepted and then silently reduced to the first.
        """
        data = event.mimeData()
        if not data.hasUrls():
            return None
        paths = [Path(u.toLocalFile()) for u in data.urls() if u.isLocalFile()]
        folders = [p for p in paths if p.is_dir()]
        return folders[0] if len(folders) == 1 and len(paths) == 1 else None


def _facts(entry: dict) -> str:
    """`3 conditions · 28 series`, from what was counted at the last open."""
    c, s = int(entry.get("conditions", 0) or 0), int(entry.get("series", 0) or 0)
    if not c and not s:
        return ""
    return f"{c} condition{'s'[:c != 1]} · {s} series"


def _load_recents() -> list:
    from ...session import recent_projects

    return recent_projects()
