"""The legacy window's Import tab: SMTracker 2.0's first tab (`SMTracker.m:160-280`; manual Fig. 4, p. 8).

Left, 2.0's three panels: where the data came from (pixel size, time interval), "Display movie data" (condition,
movie, cell, track; cells and tracks on or off; phase, sum or white behind them), "Additional options & plots". Right:
the overlay of cells and tracks on the image, the Statistics table (one column per condition), the cell-length
boxplot, and the additional plot picked on the left. Every number from `legacy_view`.
"""
from __future__ import annotations

from qtpy.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGridLayout,
                            QGroupBox, QHBoxLayout, QHeaderView, QLineEdit, QPushButton, QRadioButton,
                            QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ... import legacy_draw_import as ldi
from ... import legacy_view as lv
from .window import LegacyTab, PlotPanel

#: The left column's width (2.0's control column).
CONTROL_W = 270
ADDITIONAL = ("all tracks on one cell", "spots per frame", "length of tracks")
#: How near a click must land to a track's localisation to pick it (px of the image).
TRACK_PICK_PX = 4.0


class ImportTab(LegacyTab):
    title = "Import"

    def __init__(self, window):
        super().__init__(window)
        self._payloads: dict = {}
        self._projected: dict = {}
        self._payload = None
        #: The Track Explorers opened from here (held: CLAUDE.md, hold the windows you open) and each condition's
        #: confinement radius, computed once.
        self.explorers: list = []
        self._radii: dict = {}
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._controls())
        right = QWidget()
        grid = QGridLayout(right)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.overlay_panel = PlotPanel("Overlay cells and tracks on images")
        self.stats_box = QGroupBox("Statistics")
        sl = QVBoxLayout(self.stats_box)
        sl.setContentsMargins(4, 4, 4, 4)
        self.stats_table = QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.verticalHeader().setVisible(False)
        sl.addWidget(self.stats_table)
        self.length_panel = PlotPanel("Cell length")
        self.extra_panel = PlotPanel("Additional plots")
        self.overlay_panel.canvas.mpl_connect("button_press_event", self._overlay_clicked)
        grid.addWidget(self.overlay_panel, 0, 0)
        grid.addWidget(self.stats_box, 0, 1)
        grid.addWidget(self.length_panel, 1, 0)
        grid.addWidget(self.extra_panel, 1, 1)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        outer.addWidget(right, 1)

    # -- the left column ----------------------------------------------------------------------- #
    def _controls(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(CONTROL_W)
        lay = QVBoxLayout(column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        params = self.project.params
        source = QGroupBox("Import data from: analysis window")
        form = QFormLayout(source)
        for label, value in (("Pixel Size [nm]", f"{params.pixel_size_nm:g}"),
                             ("Time interval [ms]", f"{params.frame_interval_ms:g}")):
            field = QLineEdit(value)
            field.setReadOnly(True)
            form.addRow(label, field)
        lay.addWidget(source)

        display = QGroupBox("Display movie data")
        dl = QVBoxLayout(display)
        self.condition_box = QComboBox()
        self.condition_box.addItems([f"{i}-{c}" for i, c in enumerate(self.project.conditions, start=1)])
        self.movie_box = QComboBox()
        self.cell_box = QComboBox()
        self.track_box = QComboBox()
        for box in (self.condition_box, self.movie_box, self.cell_box, self.track_box):
            dl.addWidget(box)
        checks = QHBoxLayout()
        self.cells_check = QCheckBox("Cells")
        self.tracks_check = QCheckBox("Tracks")
        for check in (self.cells_check, self.tracks_check):
            check.setChecked(True)
            check.toggled.connect(self._redraw_overlay)
            checks.addWidget(check)
        checks.addStretch(1)
        dl.addLayout(checks)
        radios = QHBoxLayout()
        self.background_group = QButtonGroup(self)
        for key, text in (("phase", "Phase"), ("sum", "Sum"), ("none", "White")):
            radio = QRadioButton(text)
            radio.setProperty("key", key)
            radio.setChecked(key == "phase")
            self.background_group.addButton(radio)
            radios.addWidget(radio)
        radios.addStretch(1)
        dl.addLayout(radios)
        # 2.0 opened its Track Explorer from a click on a track in the overlay; this button does the same for the
        # track picked above
        self.explorer_button = QPushButton("Track explorer…")
        self.explorer_button.setEnabled(False)
        self.explorer_button.clicked.connect(self.open_explorer)
        dl.addWidget(self.explorer_button)
        lay.addWidget(display)

        extra = QGroupBox("Additional options && plots")  # && : a literal & in a Qt title
        el = QVBoxLayout(extra)
        self.extra_group = QButtonGroup(self)
        for i, text in enumerate(ADDITIONAL):
            radio = QRadioButton(text)
            radio.setChecked(i == 0)
            self.extra_group.addButton(radio, i)
            el.addWidget(radio)
        lay.addWidget(extra)
        lay.addStretch(1)

        self.condition_box.currentIndexChanged.connect(self._condition_changed)
        self.movie_box.currentIndexChanged.connect(self._movie_changed)
        self.cell_box.currentIndexChanged.connect(self._redraw_overlay)
        self.track_box.currentIndexChanged.connect(self._redraw_overlay)
        self.track_box.currentIndexChanged.connect(lambda i: self.explorer_button.setEnabled(i > 0))
        self.background_group.buttonClicked.connect(lambda _b: self._movie_changed())
        self.extra_group.idClicked.connect(lambda _i: self._draw_extra())
        return column

    # -- what is picked ------------------------------------------------------------------------ #
    def condition(self) -> str:
        return self.project.conditions[max(0, self.condition_box.currentIndex())]

    def background(self) -> str:
        button = self.background_group.checkedButton()
        return button.property("key") if button is not None else "phase"

    # -- drawing --------------------------------------------------------------------------------- #
    def opened(self) -> None:
        project, conditions = self.project, list(self.project.conditions)
        self.overlay_panel.message("Loading the first movie…")
        self.length_panel.message("Measuring the cells…")
        self.extra_panel.message("")
        self.window.run_in_background(
            "Statistics",
            lambda: (lv.statistics_table(project, conditions),
                     {c: lv.cell_lengths(project, c) for c in conditions}),
            self._statistics_done)
        self._condition_changed()

    def _statistics_done(self, result) -> None:
        table, lengths = result
        self.stats_table.setColumnCount(len(table["columns"]) + 1)
        self.stats_table.setRowCount(len(table["rows"]))
        self.stats_table.setHorizontalHeaderLabels(
            [""] + [f"{i}-{c}" for i, c in enumerate(table["columns"], start=1)])
        for r, (row, texts) in enumerate(table["rows"]):
            self.stats_table.setItem(r, 0, QTableWidgetItem(row))
            for c, text in enumerate(texts, start=1):
                self.stats_table.setItem(r, c, QTableWidgetItem(text))
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.length_panel.draw(lambda fig: ldi.draw_lengths(fig, lengths, self.window.project_title))
        self.window.log("Statistics done.")

    def _condition_changed(self) -> None:
        condition = self.condition()
        blocked = self.movie_box.blockSignals(True)
        self.movie_box.clear()
        try:
            self.movie_box.addItems(lv.condition_movies(self.project, condition))
        except Exception as exc:                          # noqa: BLE001 - said in the Console
            self.window.log(f"{condition}: no movies to show ({exc})")
        self.movie_box.blockSignals(blocked)
        self._movie_changed()
        self._draw_extra()

    def _movie_changed(self) -> None:
        condition, movie, background = self.condition(), self.movie_box.currentText() or None, self.background()
        key = (condition, movie, background)
        if key in self._payloads:
            self._payload_ready(key, self._payloads[key])
            return
        project = self.project
        self.window.run_in_background(
            f"Loading {movie or condition}",
            lambda: lv.overlay(project, condition, movie, background=background),
            lambda payload: self._payload_ready(key, payload))

    def _payload_ready(self, key, payload) -> None:
        self._payloads[key] = payload
        if key != (self.condition(), self.movie_box.currentText() or None, self.background()):
            return                                         # the user moved on while it loaded
        self._payload = payload
        for box, items in ((self.cell_box, ["All cells"] + [o["label"] for o in payload["outlines"]]),
                           (self.track_box, ["All tracks"] + [f"track {t['number']}" for t in payload["tracks"]])):
            blocked = box.blockSignals(True)
            box.clear()
            box.addItems(items)
            box.blockSignals(blocked)
        self._redraw_overlay()
        if self.extra_group.checkedId() == 1:
            self._draw_extra()

    def _redraw_overlay(self) -> None:
        payload = self._payload
        if payload is None:
            return
        cell_index = self.cell_box.currentIndex() - 1
        track_index = self.track_box.currentIndex() - 1
        show_cells, show_tracks = self.cells_check.isChecked(), self.tracks_check.isChecked()

        self.overlay_panel.draw(
            lambda fig: ldi.draw_overlay(fig, payload, cell_index, track_index, show_cells, show_tracks))

    # -- the Track Explorer ---------------------------------------------------------------------- #
    def _overlay_clicked(self, event) -> None:
        """2.0's way in: a click within a few pixels of a track's localisation picks it and opens the explorer."""
        payload = self._payload
        if payload is None or event.xdata is None or not self.tracks_check.isChecked():
            return
        best = lv.nearest_track(payload, event.xdata, event.ydata, TRACK_PICK_PX)
        if best is not None:
            self.track_box.setCurrentIndex(best + 1)
            self.open_explorer()

    def open_explorer(self):
        """The Track Explorer on the picked track (None when no track is picked)."""
        from .explorer import TrackExplorer

        index = self.track_box.currentIndex() - 1
        payload = self._payload
        if payload is None or not 0 <= index < len(payload["tracks"]):
            return None
        uid = payload["tracks"][index]["uid"]
        explorer = TrackExplorer(self.window, payload["condition"], uid,
                                 projected=self._projected.get(payload["condition"]), radius_cache=self._radii)
        self.explorers.append(explorer)
        explorer.show()
        return explorer

    def _draw_extra(self) -> None:
        which = self.extra_group.checkedId()
        condition = self.condition()
        if which == 0:
            if condition in self._projected:
                self._draw_all_tracks(condition, self._projected[condition])
                return
            project = self.project
            self.extra_panel.message("Projecting the tracks onto their cells…")
            self.window.run_in_background(f"All tracks on one cell, {condition}",
                                          lambda: project.projected(condition),
                                          lambda table: self._draw_all_tracks(condition, table))
        elif which == 1:
            payload = self._payload
            if payload is None:
                return
            self.extra_panel.draw(lambda fig: ldi.draw_spots_per_frame(fig, payload))
        else:
            lengths = lv.track_lengths(self.project, condition)
            self.extra_panel.draw(lambda fig: ldi.draw_track_lengths(fig, lengths, condition))

    def _draw_all_tracks(self, condition: str, table) -> None:
        self._projected[condition] = table
        if self.condition() != condition or self.extra_group.checkedId() != 0:
            return
        self.extra_panel.draw(lambda fig: ldi.draw_all_tracks(fig, table, condition))
