"""The legacy window's SLA tab: SMTracker 2.0's Stationary Localization Analysis (`SMTracker.m:292-408`; manual Fig. 6).

Left, 2.0's panels: "Choose condition:", "Stationary Localization Analysis:" (the confinement radius, the minimum
number of steps, the "confined tracks" radio with its free / mixed behaviour / dwelling ticks, the "dwell times"
radio, "All Movies"), "Run & Plot" and the "Stationary analysis Table" (a column per condition, the "significantly
different?" rows once every condition has been run for its dwell times). Right: the "Confined Map" (tracks on the
standard cell: free blue, mixed behaviour green, dwelling segments red) with its "Heat map" button, "Number dwell
events & transitions", "MSD for confined tracks"; in the dwell-times view the map panel holds the survival curve of the
dwell times with its fits and the MSD panel is hidden. The "Heat map" button opens 2.0's confinement heat map (jet) with
the two normalised distributions. 2.0's own words (free, mixed behaviour, dwelling, static, mobile) are used here only.
Every number from `smtracker.legacy_view_sla`.
"""
from __future__ import annotations

from qtpy.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox,
                            QHBoxLayout, QHeaderView, QInputDialog, QPushButton, QRadioButton, QScrollArea,
                            QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ... import legacy_draw_sla as lds
from ... import legacy_view_sla as ls
from .window import LegacyTab, PlotPanel

CONTROL_W = 380
RADIUS_WORD = "CUSTOM"
SET_WORDS = {"free": "free", "mixed": "mixed behaviour", "dwelling": "dwelling"}


class SlaTab(LegacyTab):
    title = "SLA"

    def __init__(self, window):
        super().__init__(window)
        self.run = None
        self.radius_nm = ls.default_radius_nm(self.project)
        self.columns = {c: [""] * len(ls.TABLE_ROWS) for c in self.project.conditions}
        self.dwell_times: dict = {}
        self.dwell_done: set = set()
        self._heat = None
        self._gallery = None
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._controls())
        right = QWidget()
        grid = QGridLayout(right)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.map_panel = PlotPanel("Confined Map", min_height=260)
        self.events_panel = PlotPanel("Number dwell events && transitions")      # && : a literal &
        self.msd_panel = PlotPanel("MSD for confined tracks")
        bar = QHBoxLayout()
        self.heat_button = QPushButton("Heat map")
        self.heat_button.clicked.connect(self.open_heatmap)
        self.heat_button.setEnabled(False)
        bar.addWidget(self.heat_button)
        bar.addStretch(1)
        self.map_panel.layout().addLayout(bar)
        grid.addWidget(self.map_panel, 0, 0, 1, 2)
        grid.addWidget(self.events_panel, 1, 0)
        grid.addWidget(self.msd_panel, 1, 1)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        outer.addWidget(right, 1)
        for panel in (self.map_panel, self.events_panel, self.msd_panel):
            panel.message("Press Run & Plot.")

    # -- the left column ----------------------------------------------------------------------- #
    def _controls(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(CONTROL_W)
        lay = QVBoxLayout(column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        cond = QGroupBox("Choose condition:")
        cl = QVBoxLayout(cond)
        self.condition_box = QComboBox()
        self.condition_box.addItems(list(self.project.conditions))
        cl.addWidget(self.condition_box)
        lay.addWidget(cond)

        sla = QGroupBox("Stationary Localization Analysis:")
        sl = QGridLayout(sla)
        self.radius_box = QComboBox()
        self.radius_box.addItem(self._radius_text())
        self.radius_box.activated.connect(self._ask_radius)       # 2.0 asks for the radius on CUSTOM
        self.steps_box = QComboBox()
        self.steps_box.addItems([f"{n} step(s)" for n in ls.MIN_STEPS_CHOICES])
        self.steps_box.setCurrentIndex(ls.MIN_STEPS_CHOICES.index(ls.DEFAULT_MIN_STEPS))
        sl.addWidget(self.radius_box, 0, 0, 1, 2)
        sl.addWidget(self.steps_box, 0, 2, 1, 2)
        self.confined_radio = QRadioButton("confined tracks")
        self.dwell_radio = QRadioButton("dwell times")
        self.confined_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.confined_radio)
        group.addButton(self.dwell_radio)
        self.checks = {}
        sl.addWidget(self.confined_radio, 1, 0, 1, 4)
        ticks = QHBoxLayout()
        ticks.addSpacing(18)
        for key in ("free", "mixed", "dwelling"):
            check = QCheckBox(SET_WORDS[key])
            check.setChecked(key != "dwelling")
            check.toggled.connect(lambda on, k=key: self._set_toggled(k, on))
            self.checks[key] = check
            ticks.addWidget(check)
        ticks.addStretch(1)
        sl.addLayout(ticks, 2, 0, 1, 4)
        sl.addWidget(self.dwell_radio, 3, 0, 1, 2)
        self.all_movies = QCheckBox("All Movies")
        sl.addWidget(self.all_movies, 3, 2, 1, 2)
        lay.addWidget(sla)

        self.run_button = QPushButton("Run && Plot")        # && : a literal &
        self.run_button.clicked.connect(self.start)
        lay.addWidget(self.run_button)

        self.table_box = QGroupBox("Stationary analysis Table")
        tl = QVBoxLayout(self.table_box)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        tl.addWidget(self.table)
        self.table_box.setVisible(False)
        lay.addWidget(self.table_box, 1)
        return column

    def _radius_text(self) -> str:
        return f"{RADIUS_WORD} : {self.radius_nm:g} nm"

    def _ask_radius(self, _index=None) -> None:
        value, ok = QInputDialog.getDouble(self, "Radius", "Enter a value of confinement radius (nm)", self.radius_nm,
                                           1.0, 100000.0, 1)
        if ok:
            self.set_radius_nm(value)

    def set_radius_nm(self, nm: float) -> None:
        self.radius_nm = float(nm)
        self.radius_box.setItemText(0, self._radius_text())

    def _set_toggled(self, key: str, on: bool) -> None:
        """2.0's `showConfinedTracks`: mixed behaviour and dwelling are never ticked together (the one just ticked
        unticks the other); the map redraws with what is ticked."""
        if on and key in ("mixed", "dwelling"):
            other = "dwelling" if key == "mixed" else "mixed"
            blocked = self.checks[other].blockSignals(True)
            self.checks[other].setChecked(False)
            self.checks[other].blockSignals(blocked)
        if self.run is not None and self.run["mode"] == "confined":
            self._draw_map()
        if self._heat is not None:
            self._heat.refresh()

    def shown_sets(self) -> tuple:
        return tuple(k for k, check in self.checks.items() if check.isChecked())

    # -- the run ------------------------------------------------------------------------------- #
    def start(self) -> None:
        condition = self.condition_box.currentText()
        mode = "dwell" if self.dwell_radio.isChecked() else "confined"
        radius_um = self.radius_nm / 1000.0
        min_steps = ls.MIN_STEPS_CHOICES[self.steps_box.currentIndex()]
        want_gallery = self.all_movies.isChecked() and mode == "confined"
        project = self.project
        self.run_button.setEnabled(False)
        for panel in (self.map_panel, self.events_panel, self.msd_panel):
            panel.message("Running…")
        def compute():
            try:
                return ls.sla_run(project, condition, radius_um, min_steps, mode)
            except Exception as exc:                      # noqa: BLE001 - said in the panels and the Console, by name
                return {"error": f"{type(exc).__name__}: {exc}"}

        self.window.run_in_background(f"SLA {mode} analysis of {condition}", compute,
                                      lambda run: self._done(run, want_gallery))

    def _done(self, run: dict, want_gallery: bool = False) -> None:
        self.run_button.setEnabled(True)
        if "error" in run:
            for panel in (self.map_panel, self.events_panel, self.msd_panel):
                panel.message(run["error"])
            self.window.log(f"SLA failed: {run['error']}")
            return
        self.run = run
        condition = run["condition"]
        self.dwell_times[condition] = run["dwell_times_s"]
        if run["mode"] == "dwell":
            self.dwell_done.add(condition)
        self.columns[condition] = ls.sla_table_column(run, self.columns[condition])
        self._fill_table()
        dwell = run["mode"] == "dwell"
        self.heat_button.setVisible(not dwell)
        self.heat_button.setEnabled(not dwell)
        self.msd_panel.setVisible(not dwell)
        if dwell:
            self.map_panel.title = "Dwell times"
            self.map_panel.setTitle("Dwell times")
            self.map_panel.draw(self._draw_dwell_fit)
            if run["dwell_fit_error"]:
                self.window.log(run["dwell_fit_error"])
            for key in ("single_error", "double_error"):
                if run["dwell_fit"] and run["dwell_fit"][key]:
                    self.window.log(run["dwell_fit"][key])
            self.events_panel.draw(self._draw_dwell_events)
        else:
            self.map_panel.title = "Confined Map"
            self.map_panel.setTitle("Confined Map")
            self._draw_map()
            self.events_panel.draw(self._draw_transitions)
            if run["msd_error"]:
                self.msd_panel.message(run["msd_error"])
                self.window.log(run["msd_error"])
            else:
                self.msd_panel.draw(self._draw_msd)
            if self._heat is not None:
                self._heat.refresh()
            if want_gallery:
                self._open_gallery(condition)
        self.window.log("Done!")

    def _fill_table(self) -> None:
        rows = ls.sla_table(self.columns, {c: self.dwell_times[c] for c in self.dwell_done}
                            if self.dwell_done and set(self.columns) <= self.dwell_done else {})
        names = list(self.columns)
        self.table.setColumnCount(len(names) + 1)
        self.table.setHorizontalHeaderLabels([""] + names)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row[: len(names) + 1]):
                self.table.setItem(r, c, QTableWidgetItem(str(value)))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        self.table.setColumnWidth(0, 215)                    # 2.0's `ColumnWidth [200 100 100 ...]`
        for c in range(1, len(names) + 1):
            self.table.setColumnWidth(c, 110)
        self.table_box.setVisible(True)

    # -- the plots ------------------------------------------------------------------------------- #
    def _draw_map(self) -> None:
        self.map_panel.draw(self._paint_map)

    def _paint_map(self, fig) -> None:
        lds.draw_map(fig, self.run, self.shown_sets(), self.window.project_title)

    def _draw_transitions(self, fig) -> None:
        lds.draw_transitions(fig, self.run)

    def _draw_dwell_events(self, fig) -> None:
        lds.draw_dwell_events(fig, self.run)

    def _draw_msd(self, fig) -> None:
        lds.draw_msd(fig, self.run)

    def _draw_dwell_fit(self, fig) -> None:
        lds.draw_dwell_fit(fig, self.run)

    # -- the extra windows ----------------------------------------------------------------------- #
    def open_heatmap(self) -> None:
        """2.0's "Heat map" button: the confinement heat map in its own window."""
        if self.run is None or self.run["mode"] != "confined":
            return
        if self._heat is None:
            self._heat = HeatmapWindow(self)
        self._heat.refresh()
        self._heat.show()
        self._heat.raise_()

    def _open_gallery(self, condition: str) -> None:
        """2.0's "All Movies": the tracks of every movie of the condition, one figure per movie."""
        run = self.run
        project = self.project
        self.window.run_in_background(
            f"Tracks per movie of {condition}", lambda: ls.per_movie_tracks(project, condition, run),
            lambda movies: self._show_gallery(condition, movies))

    def _show_gallery(self, condition: str, movies: list) -> None:
        self._gallery = MoviesWindow(self, condition, movies)
        self._gallery.show()
        self.window.log("Done!")


class HeatmapWindow(QDialog):
    """2.0's confinement heat map (`confinedHeatmap`): jet, from 0 to the maximum density, the standard cell's outline,
    the "Overlay Tracks" tick, and the two normalised distributions of the same points. Opened from the SLA tab's
    "Heat map" button; it follows the tab's run and its free / mixed behaviour / dwelling ticks."""

    def __init__(self, tab: SlaTab):
        super().__init__(tab.window)
        self.tab = tab
        self.setWindowTitle("SMTracker 2.0 (legacy view) - Confined Map, Heat map")
        self.resize(900, 620)
        grid = QGridLayout(self)
        self.map_panel = PlotPanel("Confined Map", min_height=300)
        self.x_panel = PlotPanel("Normalized x-axes distribution")
        self.y_panel = PlotPanel("Normalized y-axes distribution")
        row = QHBoxLayout()
        self.overlay = QCheckBox("Overlay Tracks")
        self.overlay.toggled.connect(self.refresh)
        row.addStretch(1)
        row.addWidget(self.overlay)
        self.map_panel.layout().addLayout(row)
        grid.addWidget(self.map_panel, 0, 0, 2, 1)
        grid.addWidget(self.x_panel, 0, 1)
        grid.addWidget(self.y_panel, 1, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        self.data = None

    def refresh(self, *_args) -> None:
        run = self.tab.run
        if run is None or run["mode"] != "confined":
            return
        try:
            self.data = ls.heatmap(run, self.tab.shown_sets())
        except Exception as exc:                          # noqa: BLE001 - the panels say why, loudly
            self.data = None
            for panel in (self.map_panel, self.x_panel, self.y_panel):
                panel.message(f"{type(exc).__name__}: {exc}")
            self.tab.window.log(f"Heat map: {exc}")
            return
        self.map_panel.draw(self._paint_map)
        self.x_panel.draw(lambda fig: self._paint_distribution(fig, "x_distribution", "distribution long axis"))
        self.y_panel.draw(lambda fig: self._paint_distribution(fig, "y_distribution", "distribution short axis"))

    def _paint_map(self, fig) -> None:
        lds.draw_heat_map(fig, self.data, self.tab.run, self.tab.shown_sets(), self.overlay.isChecked(),
                          self.tab.window.project_title)

    def _paint_distribution(self, fig, key: str, label: str) -> None:
        lds.draw_distribution(fig, self.data[key], key, label, self.tab.window.project_title)


class MoviesWindow(QDialog):
    """2.0's "All Movies": one figure per movie of the condition with the cell outlines and every track (blue), the
    mixed-behaviour tracks (green) and their dwelling segments (red), as `confTracksperCell.m` draws them."""

    def __init__(self, tab: SlaTab, condition: str, movies: list):
        super().__init__(tab.window)
        self.setWindowTitle(f"SMTracker 2.0 (legacy view) - Confined tracks per movie, {condition}")
        self.resize(900, 700)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        grid = QGridLayout(holder)
        self.panels = []
        for i, movie in enumerate(movies):
            panel = PlotPanel(f"{movie['movie']} ({movie['n_tracks']} tracks)", min_height=300)
            panel.draw(lambda fig, m=movie: self._paint(fig, m))
            self.panels.append(panel)
            grid.addWidget(panel, i // 2, i % 2)
        scroll.setWidget(holder)
        outer.addWidget(scroll)

    @staticmethod
    def _paint(fig, movie: dict) -> None:
        lds.draw_movie_tracks(fig, movie)
