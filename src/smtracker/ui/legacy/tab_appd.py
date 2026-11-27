"""The legacy window's APPD tab: SMTracker 2.0's apparent-diffusion tab (`SMTracker.m:881-955`; manual Fig. 14, 15).

Left, 2.0's panels: "Select fitting parameters" (GAMMA or GAUSSIAN, Min R^2, Simultaneous, Run best fit, Generate Speed
maps), "Axes parameters" (max X, max Y), "Select # Populations" (1, 2, 3, best), "Parameters" (Bins - / Bins +, the
specific tau of the MSD tab) and "Apparent Diffusion calculations" (the significance block, then the GAMMA and GAUSSIAN
blocks). Middle: one histogram per condition with its fitted populations (static red, mobile blue, fast mobile green:
2.0's own words) and "Redo fit for ..." where the fit is not simultaneous. Right: the bubble plot and the crossed error
bars. The "Speed maps" window is `SpeedmapWindow`. Every number from `legacy_view_appd`; this module lays out and draws.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                            QLabel, QLineEdit, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout,
                            QWidget)

from ... import legacy_draw_appd as lda
from ... import legacy_view_appd as la
from ... import legacy_view_msd as lm
from .style import WINDOW_QSS
from .window import LegacyTab, PlotPanel

CONTROL_W = 340

class AppdTab(LegacyTab):
    title = "APPD"

    def __init__(self, window):
        super().__init__(window)
        self.runs: dict = {}
        self.msd_run = None
        self.speedmap_window = None
        self.bin_width = la.DEFAULT_BIN_WIDTH
        self._seed = 0
        self.rows: dict = {}
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._controls())
        self.hist_box = QGroupBox("Apparent Diffusion Histogram")
        hl = QVBoxLayout(self.hist_box)
        hl.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        holder = QWidget()
        self.hist_layout = QVBoxLayout(holder)
        self.hist_layout.setContentsMargins(0, 0, 0, 0)
        for condition in self.project.conditions:
            self.hist_layout.addWidget(self._histogram_row(condition))
        scroll.setWidget(holder)
        hl.addWidget(scroll)
        outer.addWidget(self.hist_box, 3)
        right = QVBoxLayout()
        self.bubble_panel = PlotPanel("Apparent Diffusion Bubbleplot")
        self.crossed_panel = PlotPanel("Apparent Diffusion Bubbleplot (crossed error bars)")
        right.addWidget(self.bubble_panel)
        right.addWidget(self.crossed_panel)
        holder = QWidget()
        holder.setLayout(right)
        outer.addWidget(holder, 2)
        for panel in (self.bubble_panel, self.crossed_panel):
            panel.message("Press Run best fit.")
        self.hist_box.setVisible(False)
        holder.setVisible(True)

    # -- the left column ----------------------------------------------------------------------- #
    def _controls(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(CONTROL_W)
        lay = QVBoxLayout(column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        box = QGroupBox("Select fitting parameters")
        bl = QVBoxLayout(box)
        row = QHBoxLayout()
        self.kind_box = QComboBox()
        self.kind_box.addItems(list(la.APPD_TYPES))
        self.kind_box.currentIndexChanged.connect(self._kind_changed)
        row.addWidget(self.kind_box)
        row.addWidget(QLabel("Min R²"))
        self.r2_edit = QLineEdit(f"{la.DEFAULT_R2_LIMIT:g}")
        self.r2_edit.setFixedWidth(50)
        row.addWidget(self.r2_edit)
        bl.addLayout(row)
        self.simultaneous = QCheckBox("Simultaneous")
        self.simultaneous.setChecked(True)
        if len(self.project.conditions) == 1:                     # `SMTracker.m:6435`: one condition, no joint fit
            self.simultaneous.setChecked(False)
            self.simultaneous.setEnabled(False)
        bl.addWidget(self.simultaneous)
        lay.addWidget(box)
        self.run_button = QPushButton("Run best fit")
        self.run_button.clicked.connect(self.start)
        lay.addWidget(self.run_button)
        self.speed_button = QPushButton("Generate Speed maps")
        self.speed_button.setEnabled(False)
        self.speed_button.clicked.connect(self._open_speedmaps)
        lay.addWidget(self.speed_button)

        self.axes_box = QGroupBox("Axes parameters")
        al = QHBoxLayout(self.axes_box)
        self.max_x = QLineEdit()
        self.max_y = QLineEdit()
        for text, field in (("max X", self.max_x), ("max Y", self.max_y)):
            field.setFixedWidth(60)
            field.editingFinished.connect(self._redraw)
            al.addWidget(QLabel(text))
            al.addWidget(field)
        self.axes_box.setVisible(False)
        lay.addWidget(self.axes_box)

        self.pops_box = QGroupBox("Select # Populations")
        pl = QHBoxLayout(self.pops_box)
        pl.addWidget(QLabel("# Pops"))
        self.pops = QComboBox()
        self.pops.addItems(list(la.APPD_POPS))
        self.pops.setEnabled(False)
        self.pops.currentIndexChanged.connect(self._redraw)
        pl.addWidget(self.pops)
        self.pops_box.setVisible(False)
        lay.addWidget(self.pops_box)

        self.params_box = QGroupBox("Parameters")
        gl = QHBoxLayout(self.params_box)
        minus = QPushButton("Bins -")
        plus = QPushButton("Bins +")
        self.bin_edit = QLineEdit(f"{la.DEFAULT_BIN_WIDTH:g}")
        self.bin_edit.setFixedWidth(46)
        self.bin_edit.editingFinished.connect(self._bin_typed)
        minus.clicked.connect(lambda: self._bins(+0.01))
        plus.clicked.connect(lambda: self._bins(-0.01))
        self.tau_edit = QLineEdit()
        self.tau_edit.setFixedWidth(34)
        self.tau_edit.setEnabled(False)
        for w in (minus, self.bin_edit, plus, QLabel("Specific τ"), self.tau_edit):
            gl.addWidget(w)
        self.params_box.setVisible(False)
        lay.addWidget(self.params_box)

        self.results_box = QGroupBox("Apparent Diffusion calculations")
        rl = QVBoxLayout(self.results_box)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        rl.addWidget(self.table)
        self.results_box.setVisible(False)
        lay.addWidget(self.results_box, 1)
        return column

    def _histogram_row(self, condition: str) -> QWidget:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        button = QPushButton(f"Redo fit for\n{condition}")
        button.setFixedWidth(110)
        button.clicked.connect(lambda _=False, c=condition: self._redo(c))
        left = QVBoxLayout()
        left.addWidget(button)
        left.addStretch(1)
        rl.addLayout(left)
        panel = PlotPanel(condition, min_height=200)
        panel.message("")
        rl.addWidget(panel, 1)
        self.rows[condition] = (button, panel)
        return row

    # -- settings ------------------------------------------------------------------------------ #
    def _kind(self) -> str:
        return self.kind_box.currentText()

    def _choice(self) -> str:
        return self.pops.currentText()

    def _r2(self) -> float:
        try:
            return float(self.r2_edit.text())
        except ValueError:
            raise lm.LegacyViewError(f"Min R² must be a number between 0 and 1, got {self.r2_edit.text()!r}") from None

    def _bins(self, step: float) -> None:
        """`binwidthChng`: Bins + narrows the bin (not below 0.01), Bins - widens it, by 0.01."""
        if step < 0 and self.bin_width <= 0.01 + 1e-9:
            return
        self.bin_width = round(self.bin_width + step, 6)
        self.bin_edit.setText(f"{self.bin_width:g}")
        self._redraw()

    def _bin_typed(self) -> None:
        try:
            value = float(self.bin_edit.text())
        except ValueError:
            return
        if value > 0:
            self.bin_width = value
            self._redraw()

    def _kind_changed(self) -> None:
        if not self.runs:
            return
        if self._kind() in self.runs:
            self._sync_after_run()
            self._redraw()
        else:
            self.start()

    # -- the run ------------------------------------------------------------------------------- #
    def start(self) -> None:
        msd_tab = self.window.tab_widgets.get("MSD")
        try:
            r2 = self._r2()
            ms = msd_tab.settings() if msd_tab is not None and hasattr(msd_tab, "settings") else {}
        except lm.LegacyViewError as exc:
            self.window.log(f"APPD: {exc}")
            return
        specific = ms.get("specific_tau", lm.DEFAULT_SPECIFIC_TAU)
        method = ms.get("method", "Linear")
        exclude_last = ms.get("exclude_last", True)
        linear_fit = ms.get("linear_fit", lm.DEFAULT_LINEAR_FIT)
        reuse = msd_tab.usable_run(specific, method, exclude_last) if msd_tab is not None else None
        kind, simultaneous = self._kind(), self.simultaneous.isChecked()
        project = self.project
        self.tau_edit.setText(str(specific))
        self.run_button.setEnabled(False)
        for _button, panel in self.rows.values():
            panel.message("Fitting running…")

        def work():
            mrun = reuse or lm.msd_run(project, tau_mode="specific", specific_tau=specific, linear_fit=linear_fit,
                                       direction="xy", method=method, exclude_last=exclude_last)
            return mrun, la.appd_run(project, mrun, kind=kind, simultaneous=simultaneous, r2_lim=r2, seed=self._seed)

        self.window.run_in_background("Apparent diffusion fit", work, self._done)

    def _done(self, result) -> None:
        mrun, run = result
        self.run_button.setEnabled(True)
        self.msd_run = mrun
        self.runs[run["kind"]] = run
        if run["kind"] != self._kind():
            blocked = self.kind_box.blockSignals(True)
            self.kind_box.setCurrentIndex(la.APPD_TYPES.index(run["kind"]))
            self.kind_box.blockSignals(blocked)
        blocked = self.pops.blockSignals(True)
        self.pops.setCurrentIndex(3)                        # `bestModel`: the list shows 'best'
        self.pops.blockSignals(blocked)
        self.pops.setEnabled(True)
        for box in (self.axes_box, self.pops_box, self.params_box, self.results_box, self.hist_box):
            box.setVisible(True)
        self.speed_button.setEnabled(True)
        self._sync_after_run()
        self._set_axes_defaults()
        self._redraw()
        self.window.log("Done!")

    def _sync_after_run(self) -> None:
        self.simultaneous.blockSignals(True)
        run = self.runs[self._kind()]
        self.simultaneous.setChecked(run["simultaneous"])
        self.simultaneous.blockSignals(False)
        for condition, (button, _panel) in self.rows.items():
            button.setEnabled(not run["simultaneous"])

    def _set_axes_defaults(self) -> None:
        """2.0 filled max X and max Y from the first draw (the largest across the conditions); here blank means that."""
        for field in (self.max_x, self.max_y):
            field.setText("")

    def _redo(self, condition: str) -> None:
        run = self.runs.get(self._kind())
        if run is None:
            return
        if run["simultaneous"]:
            self.window.log("APPD: Not available when simultaneous fit is selected")
            return
        choice = self._choice()
        n_pop = run["best"][condition] if choice == "best" else int(choice)
        self._seed += 1
        seed = self._seed
        self.window.run_in_background(f"Retry fitting ({condition})",
                                      lambda: la.refit_condition(run, condition, n_pop, seed=seed),
                                      lambda _r: (self._redraw(), self.window.log("Done!")))

    def _redraw(self) -> None:
        run = self.runs.get(self._kind())
        if run is None:
            return
        self._fill_table()
        choice = self._choice()
        x_max = lda.x_axis_max(run)
        for condition in run["conditions"]:
            self.rows[condition][1].draw(lambda fig, c=condition: self._draw_hist(fig, run, c, choice, x_max))
        self.bubble_panel.draw(lambda fig: self._draw_bubbles(fig, run, choice))
        self.crossed_panel.draw(lambda fig: self._draw_crossed(fig, run, choice))

    def _fill_table(self) -> None:
        names, rows = la.appd_table(self.runs, self._kind(), self._choice())
        t = self.table
        t.setColumnCount(len(names))
        t.setHorizontalHeaderLabels(names)
        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                t.setItem(r, c, QTableWidgetItem(str(text)))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    # -- the plots ----------------------------------------------------------------------------- #
    def _colour(self, condition: str) -> str:
        return self.window.colours.get(condition, "#0072bd")

    def _draw_hist(self, fig, run: dict, condition: str, choice: str, x_max: float) -> None:
        x_lim, y_default = lda.draw_hist(fig, run, condition, choice, x_max, colour=self._colour(condition),
                                         bin_width=self.bin_width,
                                         title=f"{self.window.project_title} / {condition}",
                                         max_x_text=self.max_x.text(), max_y_text=self.max_y.text())
        if not self.max_x.text():
            self.max_x.setText(f"{x_lim:g}")
        if not self.max_y.text():
            self.max_y.setText(f"{y_default:.3g}")

    def _draw_bubbles(self, fig, run: dict, choice: str) -> None:
        if lda.draw_bubbles(fig, run, choice, self.window.colours, self.window.project_title):
            self.window.log("Bubbleplot: There are fraction sizes <= 0. They have been excluded in the bubbleplot")

    def _draw_crossed(self, fig, run: dict, choice: str) -> None:
        lda.draw_crossed(fig, run, choice, self.window.colours, self.window.project_title)

    # -- the speed maps -------------------------------------------------------------------------- #
    def _open_speedmaps(self) -> None:
        if self.msd_run is None or self._kind() not in self.runs:
            self.window.log("APPD: No data computed yet. Run best fit first.")
            return
        if self.speedmap_window is None:
            self.speedmap_window = SpeedmapWindow(self)
        self.speedmap_window.refresh()
        self.speedmap_window.show()
        self.speedmap_window.raise_()

    def figures(self) -> dict:
        out = super().figures()
        if self.speedmap_window is not None:
            out.update({f"Speed maps {k}": v for k, v in self.speedmap_window.figures().items()})
        return out


class SpeedmapWindow(QWidget):
    """2.0's "Speed map controls and plots" window (manual Fig. 15; `onSpeedmapGUI`, `plotSpeedmap.m`): a condition, the
    minimum and maximum D, "Plot Speed map!", the apparent-diffusion histogram of the condition, the track map and
    the heat map. A separate window held by the APPD tab."""

    def __init__(self, tab: AppdTab):
        super().__init__(tab.window, Qt.Window)
        self.tab = tab
        self.setWindowTitle("Speed maps")
        self.setStyleSheet(WINDOW_QSS)
        self.resize(1100, 820)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        left = QVBoxLayout()
        left_holder = QWidget()
        left_holder.setFixedWidth(300)
        left_holder.setLayout(left)
        self.condition_box = QComboBox()
        self.condition_box.addItems(list(tab.project.conditions))
        self.condition_box.currentIndexChanged.connect(self.refresh)
        left.addWidget(self.condition_box)
        self.info = QLabel()
        self.info.setWordWrap(True)
        left.addWidget(self.info)
        row = QGridLayout()
        self.min_d = QLineEdit("0")
        self.max_d = QLineEdit("0.1")
        row.addWidget(QLabel("Minimum D"), 0, 0)
        row.addWidget(self.min_d, 0, 1)
        row.addWidget(QLabel("Maximum D"), 1, 0)
        row.addWidget(self.max_d, 1, 1)
        left.addLayout(row)
        self.plot_button = QPushButton("Plot Speed map!")
        self.plot_button.clicked.connect(self.plot)
        left.addWidget(self.plot_button)
        hint = QLabel("(Hint: Set the boundaries according to the intersection of the individual populations.)")
        hint.setWordWrap(True)
        left.addWidget(hint)
        left.addStretch(1)
        outer.addWidget(left_holder)
        right = QVBoxLayout()
        self.hist_panel = PlotPanel("Apparent Diffusion Histogram", min_height=200)
        self.track_panel = PlotPanel("Track map of certain diffusive tracks", min_height=230)
        self.heat_panel = PlotPanel("Heat map of certain diffusive tracks", min_height=230)
        for p in (self.hist_panel, self.track_panel, self.heat_panel):
            right.addWidget(p)
            p.message("")
        outer.addLayout(right, 1)
        self.result = None

    def figures(self) -> dict:
        return {p.title: p.figure for p in (self.hist_panel, self.track_panel, self.heat_panel)}

    def _condition(self) -> str:
        return self.condition_box.currentText()

    def refresh(self) -> None:
        """`speedmap_info` and `speedmap_list_callback`: the conditions on the values, and the condition's histogram."""
        tab = self.tab
        run = tab.runs.get(tab._kind())
        if run is None:
            return
        self.info.setText(f"The current conditions on the apparent diffusion values are:\n"
                          f"Minimum length of tracks: {run['steps'] + 1}\nMinimum R²: {run['r2_lim']:g}")
        condition = self._condition()
        choice = tab._choice()
        x_max = lda.x_axis_max(run)
        self.hist_panel.draw(lambda fig: tab._draw_hist(fig, run, condition, choice, x_max))

    def plot(self) -> None:
        tab = self.tab
        try:
            lower, upper = float(self.min_d.text()), float(self.max_d.text())
        except ValueError:
            tab.window.log("Speed map: Minimum D and Maximum D must be numbers.")
            return
        run = tab.runs.get(tab._kind())
        if run is None or tab.msd_run is None:
            tab.window.log("Speed map: No data computed yet. Run best fit first.")
            return
        project, mrun, condition = tab.project, tab.msd_run, self._condition()
        self.plot_button.setEnabled(False)
        self.track_panel.message("Plotting…")
        tab.window.run_in_background(
            f"Speed map ({condition})", lambda: la.speedmap(project, mrun, run, condition, lower, upper, run["r2_lim"]),
            self._done)

    def _done(self, result) -> None:
        self.plot_button.setEnabled(True)
        self.result = result
        self.track_panel.draw(lambda fig: self._draw_tracks(fig, result))
        self.heat_panel.draw(lambda fig: self._draw_heat(fig, result))

    def _draw_tracks(self, fig, r: dict) -> None:
        lda.draw_speed_tracks(fig, r, self.tab._colour(r["condition"]))

    def _draw_heat(self, fig, r: dict) -> None:
        lda.draw_speed_heat(fig, r)
