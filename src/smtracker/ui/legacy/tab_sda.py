"""The legacy window's SDA tab: SMTracker 2.0's Spatial Distribution Analysis (`SMTracker.m:941-1008`; manual Fig. 16).

A row of "Select options:" across the top (Reset CB limits, the condition, the source: spot location / velocity /
acceleration / diffusion, the motion type: Brownian / Super-diffusive / Sub-diffusive / All, the cell size, 2D / 3D,
Bins - / Bins +, Individual cells), then the heat map of the trajectories' locations on the standard 1 x 3 um cell with
the normalised x distribution beside it, and below the normalised y distribution beside the heat map with the cell
mirrored about both its axes. All in 2.0's reversed hot colour map (`flip(hot(1024))`), the two maps on ONE colour
scale. There is no Run button: as in 2.0 a choice runs the analysis. Every number from `smtracker.legacy_view_sda`.
"""
from __future__ import annotations

from qtpy.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QMessageBox,
                            QPushButton, QRadioButton, QVBoxLayout, QWidget)

from ... import legacy_draw_sda as lds
from ... import legacy_view_sda as sd
from .window import LegacyTab, PlotPanel


class SdaTab(LegacyTab):
    title = "SDA"

    def __init__(self, window):
        super().__init__(window)
        self.result = None
        self.n_bins = 20
        self.clim = self._fresh_clim()
        self._motion_cache: dict = {}
        self._size_cache: dict = {}
        self._busy = False
        self._again = False
        self._quiet = True
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._options())
        grid = QGridLayout()
        grid.setSpacing(6)
        self.map_panel = PlotPanel("Heat map of trajectory locations", min_height=200)
        self.x_panel = PlotPanel("Normalized x-axes distribution", min_height=200)
        self.y_panel = PlotPanel("Normalized y-axes distribution", min_height=200)
        self.mirror_panel = PlotPanel("Heat map with mirroring", min_height=200)
        grid.addWidget(self.map_panel, 0, 0)
        grid.addWidget(self.x_panel, 0, 1)
        grid.addWidget(self.y_panel, 1, 0)
        grid.addWidget(self.mirror_panel, 1, 1)
        outer.addLayout(grid, 1)
        for panel in (self.map_panel, self.x_panel, self.y_panel, self.mirror_panel):
            panel.message("Choose the options above.")
        self._enable()
        self._quiet = False

    @staticmethod
    def _fresh_clim() -> dict:
        """2.0's `params.heatmapCaxis`: one colour limit per source, [0 0] until a map has been drawn."""
        return {i: [0.0, 0.0] for i in range(4)}

    # -- the options row ------------------------------------------------------------------------- #
    def _options(self) -> QWidget:
        box = QGroupBox("Select options:")
        row = QHBoxLayout(box)
        self.reset_button = QPushButton("Reset CB limits")
        self.reset_button.clicked.connect(self.reset_limits)
        self.condition_box = QComboBox()
        self.condition_box.addItems(list(self.project.conditions))
        self.source_box = QComboBox()
        self.source_box.addItems(list(sd.SOURCES))
        self.motion_box = QComboBox()
        self.motion_box.addItems(list(sd.MOTION_TYPES))
        self.motion_box.setCurrentIndex(3)
        self.size_box = QComboBox()
        self.size_box.addItems(list(sd.SIZE_CHOICES))
        self.radio_2d = QRadioButton("2D")
        self.radio_3d = QRadioButton("3D")
        self.radio_2d.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.radio_2d)
        group.addButton(self.radio_3d)
        self.bins_minus = QPushButton("Bins -")
        self.bins_plus = QPushButton("Bins +")
        self.individual_button = QPushButton("Individual cells")
        for w in (self.reset_button, self.condition_box, self.source_box, self.motion_box, self.size_box,
                  self.radio_2d, self.radio_3d, self.bins_minus, self.bins_plus, self.individual_button):
            row.addWidget(w)
        row.addStretch(1)
        self.condition_box.currentIndexChanged.connect(self._condition_changed)
        self.source_box.currentIndexChanged.connect(self._source_changed)
        self.motion_box.currentIndexChanged.connect(self._motion_changed)
        self.size_box.currentIndexChanged.connect(self._changed)
        self.radio_2d.toggled.connect(self._redraw)
        self.bins_minus.clicked.connect(lambda: self.change_bins(-3))
        self.bins_plus.clicked.connect(lambda: self.change_bins(+3))
        self.individual_button.clicked.connect(self.individual_cells)
        return box

    def opened(self) -> None:
        self._fill_sizes()                                   # the cell lengths are measured on first show
        self.start()

    def _enable(self) -> None:
        """2.0's `listDatasource` / `listTypeotion`: the motion type and the size only for the spot location, the size
        only for All."""
        location = self.source_box.currentIndex() == 0
        everything = self.motion_box.currentIndex() == 3
        self.motion_box.setEnabled(location)
        self.size_box.setEnabled(location and everything)
        self.individual_button.setEnabled(location)

    def _fill_sizes(self) -> None:
        condition = self.condition_box.currentText()
        if condition not in self._size_cache:
            try:
                self._size_cache[condition] = sd.size_labels(self.project, condition)
            except Exception as exc:                      # noqa: BLE001 - said in the Console, the plain words stay
                self.window.log(f"Cell sizes of {condition}: {exc}")
                self._size_cache[condition] = list(sd.SIZE_CHOICES)
        blocked = self.size_box.blockSignals(True)
        keep = self.size_box.currentIndex()
        self.size_box.clear()
        self.size_box.addItems(self._size_cache[condition])
        self.size_box.setCurrentIndex(max(keep, 0))
        self.size_box.blockSignals(blocked)

    def _condition_changed(self) -> None:
        self._fill_sizes()
        self._changed()

    def _source_changed(self) -> None:
        self._enable()
        self._changed()

    def _motion_changed(self) -> None:
        blocked = self.size_box.blockSignals(True)
        self.size_box.setCurrentIndex(0)                  # `listTypeotion`: the size goes back to all sizes
        self.size_box.blockSignals(blocked)
        self._enable()
        self._changed()

    def _changed(self) -> None:
        if not self._quiet:
            self.start()

    # -- the run ------------------------------------------------------------------------------- #
    def reset_limits(self) -> None:
        """2.0's "Reset CB limits": every colour limit back to zero, then the maps drawn again."""
        self.clim = self._fresh_clim()
        self.start()

    def start(self) -> None:
        if self._busy:
            self._again = True
            return
        project = self.project
        condition = self.condition_box.currentText()
        source = self.source_box.currentIndex()
        motion = self.motion_box.currentIndex()
        size = self.size_box.currentIndex()
        cache = self._motion_cache
        self._busy = True
        for panel in (self.map_panel, self.x_panel, self.y_panel, self.mirror_panel):
            panel.message("Running…")

        def compute():
            types = None
            if source == 0 and motion != 3:
                if condition not in cache:
                    cache[condition] = sd.track_motion_types(project, condition)
                types = cache[condition]
            run = sd.sda_run(project, condition, source, motion, size, motion_types=types)
            if source == 0:
                plain, mirrored = sd.location_map(run["x"], run["y"], False), sd.location_map(run["x"], run["y"], True)
                dist = None
            else:
                plain = sd.dynamics_map(run["x"], run["y"], run["z"], False)
                mirrored = sd.dynamics_map(run["x"], run["y"], run["z"], True)
                dist = sd.dynamics_distribution(run["x"], run["y"], run["z"])
            return {"run": run, "plain": plain, "mirrored": mirrored, "dist": dist, "condition": condition}

        def safe():
            try:
                return compute()
            except Exception as exc:                      # noqa: BLE001 - said in the panels and the Console, by name
                return {"error": f"{type(exc).__name__}: {exc}"}

        self.window.run_in_background(f"SDA {sd.SOURCES[source]} of {condition}", safe, self._done)

    def _done(self, result: dict) -> None:
        self._busy = False
        if "error" in result:
            self.result = None
            for panel in (self.map_panel, self.x_panel, self.y_panel, self.mirror_panel):
                panel.message(result["error"])
            self.window.log(f"SDA failed: {result['error']}")
            return
        self.result = result
        run = result["run"]
        source = run["source"]
        # one colour limit for both maps of this draw, only ever raised (`plotHeatmap.m:65-70`)
        self.clim[source] = sd.grow_clim(self.clim[source], result["plain"]["vmax"], result["mirrored"]["vmax"])
        if not len(run["x"]):
            self.window.log("No tracks found with those characteristics")
        self._redraw()
        self.window.log("Done!")
        if self._again:
            self._again = False
            self.start()

    def change_bins(self, delta: int) -> None:
        """`binwidthChng`: the two location histograms gain or lose three bins (the dynamics' grid is fixed)."""
        self.n_bins = max(self.n_bins + delta, 1)
        if self.result is not None and self.result["run"]["source"] == 0:
            self._redraw()

    def _redraw(self, *_args) -> None:
        r = self.result
        if r is None:
            return
        run = r["run"]
        source = run["source"]
        clim = self.clim[source]
        title = f"{self.window.project_title} - {r['condition']}"
        three_d = self.radio_3d.isChecked()
        self.map_panel.setTitle(run["title"])
        if source == 0:
            dist = sd.location_distribution(run["x"], run["y"], self.n_bins) if len(run["x"]) else None
        else:
            dist = r["dist"]
            self.map_panel.setTitle(run["title"])
        self.map_panel.draw(lambda fig: lds.paint_heat_map(fig, r["plain"], clim, title, three_d=three_d))
        self.mirror_panel.draw(lambda fig: lds.paint_heat_map(fig, r["mirrored"], clim, title, three_d=three_d))
        self.x_panel.draw(lambda fig: lds.paint_x_distribution(fig, source, dist, f"{self.window.project_title}"))
        self.y_panel.draw(lambda fig: lds.paint_y_distribution(fig, source, dist, f"{self.window.project_title}"))

    # -- Individual cells -------------------------------------------------------------------------- #
    def individual_cells(self) -> None:
        """2.0's "Individual cells": a location heat map for every cell, saved as images (into a folder you choose;
        2.0 wrote them into the condition's own folder)."""
        answer = QMessageBox.question(self, "Individual cells heatmaps",
                                      "Only available for Spot Location of all cells. Do you want to continue?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save the individual heat maps into")
        if folder:
            self.save_individual(folder)

    def save_individual(self, folder: str) -> None:
        project = self.project
        condition = self.condition_box.currentText()
        title = self.window.project_title
        self.window.run_in_background(
            f"Individual heat maps of {condition}", lambda: lds.save_individual_maps(project, condition, folder, title),
            lambda paths: self.window.log(f"Saved {len(paths)} heat map(s) into {folder}."))

    def figures(self) -> dict:
        return {p.title: p.figure for p in (self.map_panel, self.x_panel, self.y_panel, self.mirror_panel)}
