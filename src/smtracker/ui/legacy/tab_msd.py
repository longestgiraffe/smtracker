"""The legacy window's MSD tab: SMTracker 2.0's mean-squared-displacement tab (`SMTracker.m:409-453`; manual Fig. 7, 8).

Left, 2.0's panels: the condition ("Plot ensemble & time-averaged MSD curves"), "Parameters to calculate time-averaged
MSD" (all tau / specific tau, the Linear Fit lag count), "Choose direction", "Axis scaling", "Choose MSD method" (Linear,
Brownian vs Anomalous, Anomalous free exponent; Minimum tau; Exclude last timelag), "Plot curves" and "MSD calculation
results". Right: the mean squared displacement plot (ensemble curve, error bars, the weighted-by-nothing line 2.0 fitted)
and "Localization error estimation" (a condition, "update plot", the tracks to plot, every track's MSD with guide
lines, and the histogram of the intercepts). For the two anomalous methods 2.0 opened a second window: here
`AnomalousWindow`. Every number from `legacy_view_msd`; this module lays out and draws.
"""
from __future__ import annotations

from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QGridLayout, QGroupBox,
                            QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QRadioButton, QTableView,
                            QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ... import legacy_draw_msd as ldm
from ... import legacy_view_msd as lm
from .style import WINDOW_QSS
from .window import LegacyTab, PlotPanel

CONTROL_W = 340
ALL_TAU, SPECIFIC_TAU = "all τ", "specific τ"
class _TracksModel(QAbstractTableModel):
    """2.0's "Tracks to plot" table (`setTable4`): plot (checkbox), D, R^2, Y cut, length, alpha, movie #, cell #,
    track #; the first row, "all", ticks or clears every track."""

    HEADERS = ("plot", "D [µm²/s]", "R²", "Y cut", "length", "alpha", "movie #", "cell #", "track #")

    def __init__(self, tracks: dict):
        super().__init__()
        self.tracks = tracks
        self.ticked = [bool(v) for v in tracks["ticked"]]
        self.all_state = False

    def rowCount(self, parent=QModelIndex()):                 # noqa: N802 - Qt's name
        return 0 if parent.isValid() else len(self.ticked) + 1

    def columnCount(self, parent=QModelIndex()):              # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):   # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return "all" if section == 0 else f"track {section:03d}"

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return base | Qt.ItemIsUserCheckable if index.column() == 0 else base

    def data(self, index, role=Qt.DisplayRole):
        r, c = index.row(), index.column()
        if c == 0:
            if role == Qt.CheckStateRole:
                return Qt.Checked if (self.all_state if r == 0 else self.ticked[r - 1]) else Qt.Unchecked
            return None
        if role != Qt.DisplayRole:
            return None
        if r == 0:
            return "-"
        t, i = self.tracks, r - 1
        return {1: f"{t['D'][i]:.4f}", 2: f"{t['r2'][i]:.4f}", 3: f"{t['ycut'][i]:.4f}", 4: str(t["length"][i]),
                5: f"{t['alpha'][i]:.4f}", 6: str(t["movie"][i]), 7: str(t["cell"][i]), 8: str(t["track"][i])}[c]

    def setData(self, index, value, role=Qt.EditRole):        # noqa: N802
        if role != Qt.CheckStateRole or index.column() != 0:
            return False
        on = value in (Qt.Checked, 2)
        if index.row() == 0:
            self.all_state = on
            self.ticked = [on] * len(self.ticked)
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, 0))
        else:
            self.ticked[index.row() - 1] = on
            self.dataChanged.emit(index, index)
        return True


class MsdTab(LegacyTab):
    title = "MSD"

    def __init__(self, window):
        super().__init__(window)
        self.run = None
        self.tracks = None
        self.model = None
        self.anomalous_window = None
        self._limits = None
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._controls())
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.msd_panel = PlotPanel("Mean squared displacement plot", min_height=240)
        rl.addWidget(self.msd_panel, 3)
        self.loc_box = QGroupBox("Localization error estimation")
        ll = QHBoxLayout(self.loc_box)
        left = QVBoxLayout()
        self.loc_condition = QComboBox()
        self.loc_condition.addItems(["Choose condition"] + list(self.project.conditions))
        self.loc_condition.setCurrentIndex(1 if self.project.conditions else 0)
        self.loc_condition.currentIndexChanged.connect(self._loc_condition_changed)
        left.addWidget(self.loc_condition)
        self.update_button = QPushButton("update plot")
        self.update_button.clicked.connect(self._draw_tracks_panels)
        left.addWidget(self.update_button)
        tracks_box = QGroupBox("Tracks to plot")
        tl = QVBoxLayout(tracks_box)
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tl.addWidget(self.table)
        left.addWidget(tracks_box, 1)
        holder = QWidget()
        holder.setLayout(left)
        holder.setFixedWidth(300)
        ll.addWidget(holder)
        plots = QVBoxLayout()
        self.time_panel = PlotPanel("Per-track MSD curves", min_height=120)
        self.err_panel = PlotPanel("Localization error", min_height=120)
        plots.addWidget(self.time_panel)
        plots.addWidget(self.err_panel)
        ll.addLayout(plots, 1)
        rl.addWidget(self.loc_box, 3)
        outer.addWidget(right, 1)
        for panel in (self.msd_panel, self.time_panel, self.err_panel):
            panel.message("Press Plot curves.")
        self.loc_box.setVisible(False)

    # -- the left column ----------------------------------------------------------------------- #
    def _controls(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(CONTROL_W)
        lay = QVBoxLayout(column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        box = QGroupBox("Plot ensemble && time-averaged MSD curves")
        bl = QVBoxLayout(box)
        self.condition_box = QComboBox()
        self.condition_box.addItems(["all"] + list(self.project.conditions))
        self.condition_box.setCurrentIndex(1 if self.project.conditions else 0)
        self.condition_box.currentIndexChanged.connect(self._redraw_msd)
        bl.addWidget(self.condition_box)
        lay.addWidget(box)

        box = QGroupBox("Parameters to calculate time-averaged MSD")
        row = QHBoxLayout(box)
        self.all_tau = QRadioButton(ALL_TAU)
        self.specific_tau_radio = QRadioButton(SPECIFIC_TAU)
        self.specific_tau_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.all_tau)
        group.addButton(self.specific_tau_radio)
        self.specific_tau = QLineEdit(str(lm.DEFAULT_SPECIFIC_TAU))
        self.specific_tau.setFixedWidth(34)
        self.linear_fit = QLineEdit(str(lm.DEFAULT_LINEAR_FIT))
        self.linear_fit.setFixedWidth(34)
        for w in (self.all_tau, self.specific_tau_radio, self.specific_tau, QLabel("Linear Fit"), self.linear_fit):
            row.addWidget(w)
        lay.addWidget(box)

        box = QGroupBox("Choose direction")
        row = QHBoxLayout(box)
        self.direction_buttons = {}
        group = QButtonGroup(self)
        for name in lm.MSD_DIRECTIONS:
            radio = QRadioButton(name)
            group.addButton(radio)
            self.direction_buttons[name] = radio
            row.addWidget(radio)
        self.direction_buttons["xy"].setChecked(True)
        lay.addWidget(box)

        box = QGroupBox("Axis scaling")
        grid = QGridLayout(box)
        self.axis_fields = {}
        grid.addWidget(QLabel("X axis"), 0, 0)
        grid.addWidget(QLabel("Y axis"), 1, 0)
        for r, axis in enumerate(("X", "Y")):
            for c, edge in enumerate(("min", "max")):
                field = QLineEdit()
                field.setFixedWidth(70)
                field.editingFinished.connect(self._axes_edited)
                self.axis_fields[(axis, edge)] = field
                grid.addWidget(field, r, c + 1)
        lay.addWidget(box)

        box = QGroupBox("Choose MSD method")
        grid = QGridLayout(box)
        self.method_buttons = {}
        group = QButtonGroup(self)
        for i, name in enumerate(lm.MSD_METHODS):
            radio = QRadioButton(name)
            group.addButton(radio)
            self.method_buttons[name] = radio
            grid.addWidget(radio, i, 0)
        self.method_buttons["Linear"].setChecked(True)
        grid.addWidget(QLabel("Minimum τ"), 0, 1)
        self.min_length = QLineEdit(str(lm.DEFAULT_SPECIFIC_TAU))
        self.min_length.setFixedWidth(34)
        self.min_length.setEnabled(False)
        grid.addWidget(self.min_length, 0, 2)
        for radio in self.method_buttons.values():
            radio.toggled.connect(self._pop_msd)
        self.all_tau.toggled.connect(self._pop_msd)
        self.specific_tau.editingFinished.connect(self._pop_msd)
        self.exclude_last = QCheckBox("Exclude last timelag?")
        self.exclude_last.setChecked(True)
        grid.addWidget(self.exclude_last, 1, 1, 1, 2)
        lay.addWidget(box)

        self.run_button = QPushButton("Plot curves")
        self.run_button.clicked.connect(self.start)
        lay.addWidget(self.run_button)

        self.results_box = QGroupBox("MSD calculation results")
        rl = QVBoxLayout(self.results_box)
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        rl.addWidget(self.results_table)
        self.results_box.setVisible(False)
        lay.addWidget(self.results_box, 1)
        return column

    # -- settings ------------------------------------------------------------------------------ #
    def _method(self) -> str:
        return next(name for name, radio in self.method_buttons.items() if radio.isChecked())

    def _direction(self) -> str:
        return next(name for name, radio in self.direction_buttons.items() if radio.isChecked())

    def _int(self, field: QLineEdit, what: str) -> int:
        try:
            return int(float(field.text()))
        except ValueError:
            raise lm.LegacyViewError(f"{what} must be a whole number, got {field.text()!r}") from None

    def _pop_msd(self, *_args) -> None:
        """`popMSD`: in specific tau mode the minimum is the specific tau and fixed; in all tau it is 5 for Linear and
        9 for the others, and editable."""
        specific = self.specific_tau_radio.isChecked()
        self.min_length.setEnabled(not specific)
        if specific:
            self.min_length.setText(self.specific_tau.text())
        else:
            self.min_length.setText(str(lm.default_min_length("all", self._method())))

    def settings(self) -> dict:
        """The controls as `legacy_view_msd.msd_run` takes them (the APPD tab reads the same)."""
        specific = self.specific_tau_radio.isChecked()
        return {"tau_mode": "specific" if specific else "all",
                "specific_tau": self._int(self.specific_tau, "Specific τ"),
                "linear_fit": self._int(self.linear_fit, "Linear Fit"),
                "direction": self._direction(), "method": self._method(),
                "min_length": self._int(self.min_length, "Minimum τ"),
                "exclude_last": self.exclude_last.isChecked()}

    def usable_run(self, specific_tau: int, method: str, exclude_last: bool):
        """The last run where it was made at these settings (for the APPD tab: specific tau, the method and the last-lag
        box), else None."""
        run = self.run
        if run is None:
            return None
        s = run["settings"]
        same = (s["tau_mode"] == "specific" and s["specific_tau"] == specific_tau and s["method"] == method
                and s["exclude_last"] == exclude_last and "xy" in run["directions"])
        return run if same else None

    # -- the run ------------------------------------------------------------------------------- #
    def start(self) -> None:
        try:
            s = self.settings()
        except lm.LegacyViewError as exc:
            self.window.log(f"MSD: {exc}")
            return
        project = self.project
        self.run_button.setEnabled(False)
        self.msd_panel.message("MSD running…")
        self.window.run_in_background("MSD curves", lambda: lm.msd_run(project, **s), self._done)

    def _done(self, run) -> None:
        self.run = run
        self.run_button.setEnabled(True)
        self.tracks = None
        self._fill_results()
        self._limits = None
        self._redraw_msd()
        self.loc_box.setVisible(True)
        self._loc_condition_changed()

    def _fill_results(self) -> None:
        names, rows = lm.msd_table(self.run)
        t = self.results_table
        t.setColumnCount(len(names))
        t.setHorizontalHeaderLabels(names)
        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                t.setItem(r, c, QTableWidgetItem(str(text)))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_box.setVisible(True)

    def _shown(self) -> list:
        text = self.condition_box.currentText()
        return list(self.run["conditions"]) if text == "all" else [text]

    def _axes_edited(self) -> None:
        if self.run is None:
            return
        try:
            self._limits = tuple(float(self.axis_fields[(a, e)].text()) for a in ("X", "Y") for e in ("min", "max"))
        except ValueError:
            self._limits = None
        self.msd_panel.draw(self._draw_msd)

    def _redraw_msd(self) -> None:
        if self.run is not None:
            self.msd_panel.draw(self._draw_msd)

    def _draw_msd(self, fig) -> None:
        title = f"{self.window.project_title} / {self.condition_box.currentText()}"
        lim = ldm.draw_msd(fig, self.run, self._shown(), self.window.colours, self._limits, title)
        if lim is not None:
            lim = {"X": lim[0], "Y": lim[1]}
            for (axis, edge), field in self.axis_fields.items():
                blocked = field.blockSignals(True)
                field.setText(f"{lim[axis][0 if edge == 'min' else 1]:.3g}")
                field.blockSignals(blocked)

    # -- the localisation-error panel ------------------------------------------------------------ #
    def _loc_condition_changed(self, *_args) -> None:
        if self.run is None or self.loc_condition.currentIndex() < 1:
            return
        condition = self.loc_condition.currentText()
        project, run = self.project, self.run
        self.time_panel.message("Fitting each track…")
        self.err_panel.message("")
        self.window.run_in_background(f"Per-track MSD fits ({condition})", lambda: lm.msd_tracks(project, run, condition),
                                      self._tracks_done)

    def _tracks_done(self, tracks) -> None:
        if self.run is None or tracks["condition"] != self.loc_condition.currentText():
            return
        self.tracks = tracks
        self.model = _TracksModel(tracks)
        self.table.setModel(self.model)
        self.model.dataChanged.connect(lambda *_: None)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._draw_tracks_panels()
        if tracks["settings"]["method"] != "Linear":
            self._open_anomalous(tracks)
        self.window.log("Done!")

    def _draw_tracks_panels(self) -> None:
        if self.tracks is None:
            return
        self.time_panel.draw(self._draw_curves)
        self.err_panel.draw(self._draw_loc_error)

    def _draw_curves(self, fig) -> None:
        tracks = self.tracks
        ldm.draw_curves(fig, tracks, self.model.ticked, f"{self.window.project_title} / {tracks['condition']}")

    def _draw_loc_error(self, fig) -> None:
        self.loc_box.setTitle(ldm.draw_loc_error(fig, self.tracks, self.model.ticked))

    # -- the anomalous window ------------------------------------------------------------------- #
    def _open_anomalous(self, tracks: dict) -> None:
        if self.anomalous_window is None:
            self.anomalous_window = AnomalousWindow(self.window)
        self.anomalous_window.show_tracks(tracks)
        self.anomalous_window.show()
        self.anomalous_window.raise_()

    def figures(self) -> dict:
        out = super().figures()
        if self.anomalous_window is not None:
            out.update({f"Anomalous {k}": v for k, v in self.anomalous_window.figures().items()})
        return out


class AnomalousWindow(QWidget):
    """2.0's auxiliary window of anomalous analysis results (manual Fig. 8; `fitMSDTracks.m`): the alpha and R^2
    histograms, the share of each type of motion, the (D, alpha) scatter, its heat map and density scatter, and the
    histogram of D of the tracks taken as Brownian. A separate window held by the MSD tab."""

    def __init__(self, legacy_window):
        super().__init__(legacy_window, Qt.Window)
        self.legacy_window = legacy_window
        self.setWindowTitle("Brownian vs Anomalous")
        self.setStyleSheet(WINDOW_QSS)
        self.resize(900, 1100)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.panel = PlotPanel("Anomalous analysis", min_height=900)
        lay.addWidget(self.panel)

    def figures(self) -> dict:
        return {self.panel.title: self.panel.figure}

    def show_tracks(self, tracks: dict) -> None:
        self.setWindowTitle(tracks["settings"]["method"])
        summary = lm.anomalous_summary(tracks)
        self.panel.draw(lambda fig: self._draw(fig, summary))

    def _draw(self, fig, s: dict) -> None:
        ldm.draw_anomalous(fig, s)
