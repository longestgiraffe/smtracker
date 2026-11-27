"""The legacy window's SQD tab: SMTracker 2.0's square-displacement tab (`SMTracker.m:756-868`; manual Fig. 12, p. 18).

Left, 2.0's panels: "Choose best fit" (automatically by BIC or by hand, simultaneous, number of diffusive groups,
number of time lags, Run & Plot), "Axes parameters", "Plotting conditions" (which population counts, which
conditions, which time lag), "SQD Analysis Results". Right: the cumulative distribution P(r^2, t) on a log axis with
the fits and their residuals; the bubble plot (D on a log axis, area the share); the crossed error bars (share
against D). Every number from `legacy_view.sqd_run`.
"""
from __future__ import annotations

from qtpy.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QGridLayout, QGroupBox,
                            QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QRadioButton, QTableWidget,
                            QTableWidgetItem, QVBoxLayout, QWidget)

from ... import legacy_draw_sqd as lds
from ... import legacy_view as lv
from .window import LegacyTab, PlotPanel

CONTROL_W = 340
#: 2.0's CDF axes defaults (`SMTracker.m:820-830`; manual p. 18): x 1e-4 to 0.5 um^2, y 0 to 1.
CDF_AXES = (1e-4, 0.5, 0.0, 1.0)
POP_LABEL = {1: "1 (dotted)", 2: "2 (line)", 3: "3 (dashed)"}


class SqdTab(LegacyTab):
    title = "SQD"

    def __init__(self, window):
        super().__init__(window)
        self.run = None
        #: The Jump Distance windows opened from here (held: CLAUDE.md, hold the windows you open).
        self.jd_windows: list = []
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._controls())
        right = QWidget()
        grid = QGridLayout(right)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.cdf_panel = PlotPanel("Cumulative distribution function plot")
        self.bubble_panel = PlotPanel("Bubbleplot SQD")
        self.crossed_panel = PlotPanel("Crossed Errorbar SQD")
        grid.addWidget(self.cdf_panel, 0, 0, 1, 2)
        grid.addWidget(self.bubble_panel, 1, 0)
        grid.addWidget(self.crossed_panel, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        outer.addWidget(right, 1)
        for panel in (self.cdf_panel, self.bubble_panel, self.crossed_panel):
            panel.message("Press Run & Plot.")

    # -- the left column ----------------------------------------------------------------------- #
    def _controls(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(CONTROL_W)
        lay = QVBoxLayout(column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        best = QGroupBox("Choose best fit")
        bl = QVBoxLayout(best)
        bl.addWidget(QLabel("Select how to fit displacements"))
        row = QHBoxLayout()
        self.auto_radio = QRadioButton("Automatically")
        self.manual_radio = QRadioButton("Manually")
        self.manual_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.auto_radio)
        group.addButton(self.manual_radio)
        self.simultaneous = QCheckBox("Simultaneous")
        self.simultaneous.setChecked(True)
        for w in (self.auto_radio, self.manual_radio, self.simultaneous):
            row.addWidget(w)
        bl.addLayout(row)
        counts = QGridLayout()
        counts.addWidget(QLabel("#Diffusive groups"), 0, 0)
        self.n_pop_box = QComboBox()
        self.n_pop_box.addItems(["1", "2", "3"])
        self.n_pop_box.setCurrentIndex(1)
        counts.addWidget(self.n_pop_box, 0, 1)
        counts.addWidget(QLabel("#timelags"), 1, 0)
        self.lag_box = QComboBox()
        self.lag_box.addItems(list(lv.SQD_LAG_CHOICES))
        counts.addWidget(self.lag_box, 1, 1)
        bl.addLayout(counts)
        self.auto_radio.toggled.connect(lambda on: self.n_pop_box.setEnabled(not on))
        self.run_button = QPushButton("Run && Plot")  # && : a literal &
        self.run_button.clicked.connect(self.start)
        bl.addWidget(self.run_button)
        # 2.0 popped its Jump Distance window up after a fit at one frame; this reopens it
        self.jd_button = QPushButton("Jump distances && GoF…")
        self.jd_button.setEnabled(False)
        self.jd_button.clicked.connect(self.open_jump_distances)
        bl.addWidget(self.jd_button)
        lay.addWidget(best)

        self.axes_box = QGroupBox("Axes parameters")
        al = QGridLayout(self.axes_box)
        self.axis_fields = {}
        for i, (key, value) in enumerate(zip(("min X", "max X", "min Y", "max Y"), CDF_AXES)):
            field = QLineEdit(f"{value:g}")
            field.editingFinished.connect(self._redraw)
            self.axis_fields[key] = field
            al.addWidget(QLabel(key), i // 2, (i % 2) * 2)
            al.addWidget(field, i // 2, (i % 2) * 2 + 1)
        self.axes_box.setVisible(False)
        lay.addWidget(self.axes_box)

        self.plotting_box = QGroupBox("Plotting conditions")
        pl = QVBoxLayout(self.plotting_box)
        pl.addWidget(QLabel("Choose number of populations:"))
        row = QHBoxLayout()
        self.pop_checks = {}
        for k in lv.SQD_N_POPS:
            check = QCheckBox(POP_LABEL[k])
            check.toggled.connect(self._redraw)
            self.pop_checks[k] = check
            row.addWidget(check)
        pl.addLayout(row)
        pl.addWidget(QLabel("Choose which conditions:"))
        self.condition_checks = {}
        for i, condition in enumerate(self.project.conditions, start=1):
            check = QCheckBox(f"{i}-{condition}")
            check.setChecked(True)
            colour = self.window.colours.get(condition, "#000000")
            check.setStyleSheet(f"QCheckBox {{ color: {colour}; font-weight: bold; }}")
            check.toggled.connect(self._redraw)
            self.condition_checks[condition] = check
            pl.addWidget(check)
        pl.addWidget(QLabel("Select the time lag:"))
        self.shown_lag = QComboBox()
        self.shown_lag.currentIndexChanged.connect(self._redraw)
        pl.addWidget(self.shown_lag)
        self.plotting_box.setVisible(False)
        lay.addWidget(self.plotting_box)

        self.results_box = QGroupBox("SQD Analysis Results")
        rl = QVBoxLayout(self.results_box)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        rl.addWidget(self.table)
        self.results_box.setVisible(False)
        lay.addWidget(self.results_box, 1)
        return column

    # -- the run ------------------------------------------------------------------------------- #
    def start(self) -> None:
        project = self.project
        n_pop = None if self.auto_radio.isChecked() else int(self.n_pop_box.currentText())
        simultaneous = self.simultaneous.isChecked()
        lag_choice = self.lag_box.currentText()
        self.run_button.setEnabled(False)
        for panel in (self.cdf_panel, self.bubble_panel, self.crossed_panel):
            panel.message("Fitting running…")
        self.window.run_in_background(
            "SQD fit", lambda: lv.sqd_run(project, n_pop=n_pop, simultaneous=simultaneous, lag_choice=lag_choice),
            self._done)

    def _done(self, run) -> None:
        self.run_button.setEnabled(True)
        self.run = run
        if run["automatic"]:
            self.n_pop_box.setCurrentIndex(run["n_pop"] - 1)
            bic = ", ".join(f"{c}: {run['best'][c] or '-'}" for c in run["conditions"])
            self.window.log(f"Best model: {bic}")
        for k, check in self.pop_checks.items():
            blocked = check.blockSignals(True)
            check.setChecked(k == run["n_pop"])
            check.blockSignals(blocked)
        blocked = self.shown_lag.blockSignals(True)
        self.shown_lag.clear()
        tau = run["tau_s"]
        self.shown_lag.addItems([f"{lag} frame{'s' * (lag > 1)} ({lag * tau:.3g} s)" for lag in run["lags"]])
        self.shown_lag.blockSignals(blocked)
        for box in (self.axes_box, self.plotting_box, self.results_box):
            box.setVisible(True)
        self._fill_table()
        self._redraw()
        self.jd_button.setEnabled(1 in run["fits"])
        if 1 in run["fits"]:
            self.open_jump_distances()                 # as 2.0 did after a fit at one frame
        self.window.log("Done!")

    def open_jump_distances(self):
        """2.0's "Jump Distance Analysis & GoF" window on the last run, at its population count."""
        from .jd_window import JumpDistanceWindow

        if self.run is None or 1 not in self.run["fits"]:
            return None
        conditions = [c for c in self.run["conditions"] if self.condition_checks[c].isChecked()]
        window = JumpDistanceWindow(self.window, self.run, self.run["n_pop"], conditions, self.window.colours)
        self.jd_windows.append(window)
        window.show()
        return window

    def _lag(self) -> int:
        run = self.run
        index = max(0, self.shown_lag.currentIndex())
        return run["lags"][min(index, len(run["lags"]) - 1)]

    def _fill_table(self) -> None:
        rows = lv.sqd_table(self.run, self._lag())
        conditions = self.run["conditions"]
        self.table.setColumnCount(len(conditions) + 1)
        self.table.setHorizontalHeaderLabels([""] + [f"{i}-{c}" for i, c in enumerate(conditions, start=1)])
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row[: len(conditions) + 1]):
                self.table.setItem(r, c, QTableWidgetItem(str(text)))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _axes(self) -> tuple:
        out = []
        for key, default in zip(("min X", "max X", "min Y", "max Y"), CDF_AXES):
            try:
                out.append(float(self.axis_fields[key].text()))
            except ValueError:
                out.append(default)
        return tuple(out)

    def _shown(self) -> tuple:
        pops = [k for k, check in self.pop_checks.items() if check.isChecked()]
        conditions = [c for c in self.run["conditions"] if self.condition_checks[c].isChecked()]
        return pops, conditions

    def _redraw(self) -> None:
        if self.run is None:
            return
        self._fill_table()
        self.cdf_panel.draw(self._draw_cdf)
        self.bubble_panel.draw(self._draw_bubbles)
        self.crossed_panel.draw(self._draw_crossed)

    # -- the three plots ------------------------------------------------------------------------- #
    def _colour(self, condition: str) -> str:
        return self.window.colours.get(condition, "#000000")

    def _draw_cdf(self, fig) -> None:
        pops, conditions = self._shown()
        lds.draw_cdf(fig, self.run, self._lag(), pops, conditions, self.window.colours, self._axes(),
                     self.window.project_title)

    def _draw_bubbles(self, fig) -> None:
        pops, conditions = self._shown()
        lds.draw_bubbles(fig, self.run, self._lag(), pops, conditions, self.window.colours,
                         self.window.project_title)

    def _draw_crossed(self, fig) -> None:
        pops, conditions = self._shown()
        lds.draw_crossed(fig, self.run, self._lag(), pops, conditions, self.window.colours,
                         self.window.project_title)
