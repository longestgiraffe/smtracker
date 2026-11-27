"""The legacy window: SMTracker 2.0's tabs on the loaded conditions (`docs/legacy_window_plan.md`).

Opened from the analysis window's right rail (the "history" button, `ToolRail.legacy_pressed`), one per workspace, handed
the workspace's project and condition colours. Thin: every number comes from `smtracker.legacy_view`; this module and
the tab modules only lay out and draw. 2.0's chrome: menus File / Tools / Help, the tabs in 2.0's order, each tab a
column of bold-titled control panels beside its plots, a Console strip at the bottom where every run reports.
"""
from __future__ import annotations

import threading
from pathlib import Path

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (QFileDialog, QGroupBox, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
                            QSizePolicy, QTabWidget, QVBoxLayout, QWidget)

from ...legacy_view import TABS
from .style import WINDOW_QSS, matlab_rc

#: What the window's title bar and Help > About say it is.
WINDOW_TITLE = "SMTracker 2.0 (legacy view)"
ABOUT_TEXT = (
    "This window shows SMTracker 2.0's tabs on the conditions loaded in the analysis window, computed the way "
    "SMTracker 2.0 computed them (its code, ported to Python), with SMTracker 2.0's own labels.\n\n"
    "It is a display of the old program, not a second analysis: its numbers are never shown beside SMTracker 3's "
    "and never go into a paper's deposit. Saved figures carry 'legacy' in their names.\n\n"
    "SMTracker 2.0: Copyright (c) 2016-2019 Thomas Roesch & Luis M. Oviedo-Bocanegra; P. Graumann lab and G. Fritz "
    "lab, Philipps-Universitaet Marburg. GNU General Public License, version 3 or later.")
#: The footnote a saved figure carries (the legacy package's rule: every output is marked).
#: The way back to the analysis window (menu bar corner and File menu).
BACK_TEXT = "Back to SMTracker 3"
SAVED_STAMP = "SMTracker 2.0 method (legacy view of SMTracker 3)"


class _Job(QObject):
    """One background computation's way back to the window's thread."""

    done = Signal(object)
    failed = Signal(str)


class PlotPanel(QGroupBox):
    """A bold-titled panel holding one matplotlib figure, drawn in 2.0's MATLAB style (`style.matlab_rc`)."""

    def __init__(self, title: str, parent=None, *, min_height: int = 220):
        super().__init__(title, parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.title = title
        self.figure = Figure(figsize=(4, 3), dpi=100, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(min_height)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.canvas)

    def draw(self, painter) -> None:
        """Clear the figure and draw ``painter(figure)`` under MATLAB's settings; a failure is drawn as its words."""
        import matplotlib

        with matplotlib.rc_context(matlab_rc()):
            self.figure.clear()
            try:
                painter(self.figure)
            except Exception as exc:                     # noqa: BLE001 - the panel says what failed, loudly
                self.figure.clear()
                self.figure.text(0.5, 0.5, f"Could not draw this panel:\n{type(exc).__name__}: {exc}",
                                 ha="center", va="center", wrap=True, color="#a00000")
            self.canvas.draw()

    def message(self, text: str) -> None:
        """An empty panel with one line (before a run, or where there is nothing to draw)."""
        self.draw(lambda fig: fig.text(0.5, 0.5, text, ha="center", va="center", color="#555555", wrap=True))


class LegacyTab(QWidget):
    """One of 2.0's tabs. Subclasses build their controls and panels; `figures` names what Print saves."""

    title = ""

    def __init__(self, window: "LegacyWindow"):
        super().__init__()
        self.window = window

    @property
    def project(self):
        return self.window.project

    def figures(self) -> dict:
        """``{name: matplotlib Figure}`` of every panel this tab draws, for File > Print figures."""
        return {p.title: p.figure for p in self.findChildren(PlotPanel)}

    def opened(self) -> None:
        """Called the first time the tab is shown (draw what needs no Run)."""


class _Placeholder(LegacyTab):
    """A tab not built yet: says so, never pretends."""

    def __init__(self, window, title: str):
        super().__init__(window)
        self.title = title
        lay = QVBoxLayout(self)
        panel = PlotPanel(title)
        lay.addWidget(panel)
        panel.message(f"The {title} tab is being rebuilt from SMTracker 2.0.")


def _tab_classes() -> dict:
    """``{tab title: class}`` for the tabs built so far (`docs/legacy_window_plan.md`'s order)."""
    from .tab_appd import AppdTab
    from .tab_clustering import ClusteringTab
    from .tab_gmm import GmmTab
    from .tab_import import ImportTab
    from .tab_msd import MsdTab
    from .tab_sda import SdaTab
    from .tab_sla import SlaTab
    from .tab_sqd import SqdTab

    return {"Import": ImportTab, "SLA": SlaTab, "MSD": MsdTab, "Clustering": ClusteringTab, "GMM": GmmTab,
            "SQD": SqdTab, "APPD": AppdTab, "SDA": SdaTab}


class LegacyWindow(QMainWindow):
    """SMTracker 2.0's window on ``project``. ``colours``: ``{condition: hex}`` (the workspace's)."""

    def __init__(self, project, *, colours: dict | None = None, title: str = "", exporters: dict | None = None,
                 home=None, parent=None):
        super().__init__(parent)
        #: The analysis window that opened this one (`go_back` raises it).
        self.home = home
        self.project = project
        self.colours = dict(colours or {})
        self.project_title = title or "SMTracker"
        self._exporters = dict(exporters or {})
        self._jobs: set = set()
        self._closed = False
        self.setWindowTitle(f"{WINDOW_TITLE} – {self.project_title}")
        self.setStyleSheet(WINDOW_QSS)
        root = QWidget()
        root.setObjectName("legacyRoot")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        built = _tab_classes()
        self.tab_widgets: dict = {}
        for name in TABS:
            tab = built[name](self) if name in built else _Placeholder(self, name)
            tab.title = name
            self.tab_widgets[name] = tab
            self.tabs.addTab(tab, name)
        self._opened: set = set()
        self.tabs.currentChanged.connect(self._tab_shown)
        lay.addWidget(self.tabs, 1)
        console_box = QGroupBox("Console")
        cl = QVBoxLayout(console_box)
        cl.setContentsMargins(4, 4, 4, 4)
        self.console = QPlainTextEdit()
        self.console.setObjectName("legacyConsole")
        self.console.setReadOnly(True)
        self.console.setFixedHeight(80)
        cl.addWidget(self.console)
        lay.addWidget(console_box)
        self.setCentralWidget(root)
        self._build_menus()
        self.resize(1280, 860)
        self.log("Data loaded successfully.")
        self.log("Ready!")
        self._tab_shown(self.tabs.currentIndex())

    # -- chrome -------------------------------------------------------------------------- #
    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.menus: dict = {}                            # held, by name
        file_menu = self.menus["File"] = bar.addMenu("File")
        file_menu.addAction(BACK_TEXT, self.go_back)
        file_menu.addSeparator()
        file_menu.addAction("Print figures…", self.print_figures)
        export = file_menu.addMenu("Export")
        for key, label in (("vbspt", "vbSPT…"), ("smmtrack", "SMMTrack…")):
            action = export.addAction(label, self._exporters[key] if key in self._exporters else (lambda: None))
            action.setEnabled(key in self._exporters)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        tools = self.menus["Tools"] = bar.addMenu("Tools")
        tools.addAction("Run & save figures: current tab…", lambda: self.print_figures(all_tabs=False))
        tools.addAction("Run & save figures: all tabs…", lambda: self.print_figures(all_tabs=True))
        help_menu = self.menus["Help"] = bar.addMenu("Help")
        help_menu.addAction("About…", self.about)
        back = QPushButton(BACK_TEXT)
        back.setObjectName("legacyBack")
        back.setCursor(Qt.PointingHandCursor)
        back.clicked.connect(self.go_back)
        self.back_button = back
        bar.setCornerWidget(back, Qt.TopRightCorner)

    def go_back(self) -> None:
        """Back to the analysis window that opened this one: raise it, then close this window (the workspace stays)."""
        home = self.home
        if home is not None:
            try:
                if home.isMinimized():
                    home.showNormal()
                home.show()
                home.raise_()
                home.activateWindow()
            except RuntimeError:                         # its C++ side is gone
                pass
        self.close()

    def about(self) -> None:
        QMessageBox.about(self, WINDOW_TITLE, ABOUT_TEXT)

    def _tab_shown(self, index: int) -> None:
        name = self.tabs.tabText(index)
        if name and name not in self._opened:
            self._opened.add(name)
            self.tab_widgets[name].opened()

    def log(self, text: str) -> None:
        """One line in the Console (2.0's status strip)."""
        self.console.appendPlainText(str(text))

    # -- background work -------------------------------------------------------------------- #
    def run_in_background(self, label: str, compute, on_done) -> None:
        """``compute()`` off the window's thread; ``on_done(result)`` back on it. A failure is said in the Console
        by name, never swallowed."""
        job = _Job()
        self._jobs.add(job)

        def finish(result):
            self._jobs.discard(job)
            if not self._closed:
                on_done(result)

        def fail(message):
            self._jobs.discard(job)
            if not self._closed:
                self.log(f"{label} failed: {message}")

        job.done.connect(finish)
        job.failed.connect(fail)
        self.log(f"{label}…")

        def work():
            try:
                result = compute()
            except Exception as exc:                     # noqa: BLE001 - reported in the Console, by name
                job.failed.emit(f"{type(exc).__name__}: {exc}")
                return
            job.done.emit(result)

        threading.Thread(target=work, name=f"legacy-{label}", daemon=True).start()

    # -- saving ------------------------------------------------------------------------------- #
    def print_figures(self, all_tabs: bool = False) -> list:
        """2.0's File > Print figures: every panel of the current tab (or every tab) as a 300 dpi PNG into a folder,
        each name starting ``legacy_`` and each figure carrying the legacy stamp. Returns the files written."""
        folder = QFileDialog.getExistingDirectory(self, "Save the legacy figures into")
        if not folder:
            return []
        return self.save_figures(Path(folder), all_tabs=all_tabs)

    def save_figures(self, folder: Path, *, all_tabs: bool = False) -> list:
        import re

        tabs = list(self.tab_widgets.values()) if all_tabs else [self.tabs.currentWidget()]
        written = []
        for tab in tabs:
            for name, figure in tab.figures().items():
                if not figure.axes:
                    continue
                safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{tab.title}_{name}").strip("_").lower()
                path = Path(folder) / f"legacy_{safe}.png"
                stamp = figure.text(0.005, 0.005, SAVED_STAMP, fontsize=7, color="#555555", ha="left", va="bottom")
                try:
                    figure.savefig(path, dpi=300, facecolor="white")
                finally:
                    stamp.remove()
                written.append(path)
        self.log(f"Saved {len(written)} figure{'s' * (len(written) != 1)} into {folder}.")
        return written

    def closeEvent(self, event):                         # noqa: N802 - Qt's name
        self._closed = True
        super().closeEvent(event)


def open_legacy_window(project, *, colours=None, title: str = "", exporters=None, home=None, show: bool = True):
    """The window, shown fitted to its screen (`screenfit.show_fitted`)."""
    window = LegacyWindow(project, colours=colours, title=title, exporters=exporters, home=home)
    window.setAttribute(Qt.WA_DeleteOnClose, False)
    if show:
        from ..widgets.screenfit import show_fitted

        show_fitted(window)
    return window
