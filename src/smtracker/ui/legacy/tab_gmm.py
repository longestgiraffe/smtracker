"""The legacy window's GMM tab: SMTracker 2.0's Gaussian mixture model tab (`SMTracker.m:623-753`; manual Fig. 10, 11).

Top left, 2.0's two panels: "Axes parameters" (min X, max X, min Y, max Y, BinWidth, grid) and "Data & fits"
(reference and comparison condition, projected trajectories, fix pair, single / double / triple fit, Cell Size in
um, the direction xy / x / y, Run & Plot). Top right, the histograms of the reference and the comparison condition's
steps with the fitted Gaussians. Bottom, shown after a run: "Results of gaussian mixture modeling" (2.0's table),
"Bubbleplot GMM" and "Crossed Errorbars GMM". The "GMM & GoF" window opens beside, with the quantile-quantile plot
of the two conditions and one probability-probability plot per condition. Every number from `legacy_view_gmm`.
"""
from __future__ import annotations

import math

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QGridLayout, QGroupBox,
                            QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QRadioButton, QScrollArea,
                            QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ... import legacy_draw_gmm as ldg
from ... import legacy_view_gmm as lg
from .window import LegacyTab, PlotPanel

CONTROL_W = 330

class GmmGofWindow(QWidget):
    """2.0's "GMM & GoF" window: the quantile-quantile plot of the reference against the comparison, then one
    probability-probability plot per condition of the fitted group."""

    def __init__(self, tab: "GmmTab", parent):
        super().__init__(parent, Qt.Window)
        self.tab = tab
        self.setWindowTitle("GMM & GoF")
        self.resize(560, 760)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.holder = QWidget()
        self.rows = QVBoxLayout(self.holder)
        self.rows.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self.holder)
        outer.addWidget(scroll)
        self.panels: list = []

    def show_run(self, run: dict, colours: dict) -> None:
        for panel in self.panels:
            self.rows.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self.panels = []
        if len(run["conditions"]) > 1:
            panel = PlotPanel("Quantile-Quantile Plot", min_height=240)
            panel.draw(lambda fig: self._qq(fig, run))
            self._add(panel)
        n_pop = max(run["fits"])
        group = run["fits"][n_pop]["conditions"]
        for condition in group:
            panel = PlotPanel(f"{condition} - {lg.FIT_NAMES[n_pop]}", min_height=240)
            panel.draw(lambda fig, c=condition: self._pp(fig, run, c, n_pop, colours.get(c, "#0072bd")))
            self._add(panel)
        self.rows.addStretch(1)

    def _add(self, panel) -> None:
        self.panels.append(panel)
        self.rows.addWidget(panel)

    @staticmethod
    def _qq(fig, run: dict) -> None:
        ldg.draw_qq(fig, run)

    @staticmethod
    def _pp(fig, run: dict, condition: str, n_pop: int, colour: str) -> None:
        ldg.draw_pp(fig, run, condition, n_pop, colour)


class GmmTab(LegacyTab):
    title = "GMM"

    def __init__(self, window):
        super().__init__(window)
        self.run = None
        self.blocks: dict = {}                # {(projected, direction): run}: 2.0 kept each direction's fits
        self._steps: dict = {}                # {(projected, cell size): {condition: steps}}
        self.gof_window = None
        self._ready = False                   # the controls fire their callbacks while they are being built
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(self._controls())
        self.ref_panel = PlotPanel("reference", min_height=250)
        self.comp_panel = PlotPanel("comparison", min_height=250)
        top.addWidget(self.ref_panel, 1)
        top.addWidget(self.comp_panel, 1)
        root.addLayout(top, 3)
        self.bottom = QWidget()
        bottom = QHBoxLayout(self.bottom)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(6)
        results = QGroupBox("Results of gaussian mixture modeling")
        rl = QVBoxLayout(results)
        rl.setContentsMargins(4, 4, 4, 4)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)   # the stretch below sets its width
        rl.addWidget(self.table)
        self.bubble_panel = PlotPanel("Bubbleplot GMM", min_height=240)
        self.crossed_panel = PlotPanel("Crossed Errorbars GMM", min_height=240)
        bottom.addWidget(results, 3)
        bottom.addWidget(self.bubble_panel, 2)
        bottom.addWidget(self.crossed_panel, 2)
        self.bottom.setVisible(False)
        root.addWidget(self.bottom, 3)
        self.ref_panel.message("Loading the steps…")
        self.comp_panel.message("")
        self._ready = True

    # -- the controls ------------------------------------------------------------------------------ #
    def _controls(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(CONTROL_W)
        lay = QVBoxLayout(column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        axes_box = QGroupBox("Axes parameters")
        grid = QGridLayout(axes_box)
        d = lg.AXES_DEFAULTS
        self.fields = {}
        layout = (("min X", "max X"), ("min Y", "max Y"), ("BinWidth", "grid"))
        for r, pair in enumerate(layout):
            for c, name in enumerate(pair):
                grid.addWidget(QLabel(name), r, c * 2)
                field = QLineEdit(f"{d[name]:g}")
                if name == "min Y":
                    field.setEnabled(False)                       # 2.0 greyed it: the axis starts at zero
                field.editingFinished.connect(lambda n=name: self._axes_edited(n))
                self.fields[name] = field
                grid.addWidget(field, r, c * 2 + 1)
        lay.addWidget(axes_box)

        data_box = QGroupBox("Data && fits")
        dl = QVBoxLayout(data_box)
        row = QHBoxLayout()
        self.ref_box, self.comp_box = QComboBox(), QComboBox()
        names = list(self.project.conditions)
        for box in (self.ref_box, self.comp_box):
            box.addItems([f"{i}-{c}" for i, c in enumerate(names, start=1)])
            box.currentIndexChanged.connect(self._pair_changed)
            row.addWidget(box)
        self.ref_box.setCurrentIndex(0)
        self.comp_box.setCurrentIndex(1 if len(names) > 1 else 0)
        self.comp_box.setEnabled(len(names) > 1)
        dl.addLayout(row)
        row = QHBoxLayout()
        self.projected_check = QCheckBox("projected trajectories")
        self.projected_check.toggled.connect(self._projected_changed)
        self.fix_check = QCheckBox("fix pair")
        self.fix_check.setEnabled(len(names) > 2)
        self.fix_check.toggled.connect(self._fix_pair)
        row.addWidget(self.projected_check)
        row.addWidget(self.fix_check)
        dl.addLayout(row)
        row = QHBoxLayout()
        self.fit_checks = {}
        for k, text in ((1, "single fit"), (2, "double fit"), (3, "triple fit")):
            check = QCheckBox(text)
            self.fit_checks[k] = check
            row.addWidget(check)
        dl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Cell Size in µm (Min,Max)"))
        self.lo_field, self.hi_field = QLineEdit("0"), QLineEdit("Inf")
        for field in (self.lo_field, self.hi_field):
            field.setFixedWidth(46)
            field.editingFinished.connect(self._cell_size_edited)
            row.addWidget(field)
        dl.addLayout(row)
        dl.addWidget(QLabel("Select direction to display:"))
        row = QHBoxLayout()
        self.direction_group = QButtonGroup(self)
        self.direction_radios = {}
        for key in lg.DIRECTIONS:
            radio = QRadioButton(key)
            self.direction_radios[key] = radio
            self.direction_group.addButton(radio)
            row.addWidget(radio)
            radio.toggled.connect(self._direction_changed)
        dl.addLayout(row)
        self.direction_radios["xy"].setChecked(True)
        self.direction_radios["x"].setEnabled(False)
        self.direction_radios["y"].setEnabled(False)
        self.run_button = QPushButton("Run && Plot")
        self.run_button.clicked.connect(self.start)
        dl.addWidget(self.run_button)
        lay.addWidget(data_box)
        lay.addStretch(1)
        self.fit_checks[2].setChecked(True)            # 2.0 opened with none ticked; the double fit is its usual
        return column

    # -- what is set ------------------------------------------------------------------------------- #
    def _value(self, name: str) -> float:
        try:
            return float(self.fields[name].text())
        except ValueError:
            return float(lg.AXES_DEFAULTS[name])

    def conditions(self) -> list:
        return list(self.project.conditions)

    def reference(self) -> str:
        return self.conditions()[max(0, self.ref_box.currentIndex())]

    def comparison(self) -> str:
        return self.conditions()[max(0, self.comp_box.currentIndex())]

    def projected(self) -> bool:
        return self.projected_check.isChecked()

    def direction(self) -> str:
        for key, radio in self.direction_radios.items():
            if radio.isChecked():
                return key
        return "xy"

    def cell_size(self) -> tuple:
        def number(field, default):
            text = field.text().strip().lower()
            if text in ("inf", "+inf"):
                return math.inf
            try:
                return float(text)
            except ValueError:
                return default

        return (number(self.lo_field, 0.0), number(self.hi_field, math.inf))

    def _key(self) -> tuple:
        return (self.projected(), self.cell_size())

    # -- the small callbacks (2.0's `updateParam`, `projected`, `fixpair`) --------------------------- #
    def _axes_edited(self, name: str) -> None:
        # 2.0 keeps the x axis symmetric about zero: editing one limit sets the other to its negative
        if name == "max X":
            self.fields["min X"].setText(f"{-self._value('max X'):g}")
        elif name == "min X":
            self.fields["max X"].setText(f"{-self._value('min X'):g}")
        self._draw_hists()

    def _projected_changed(self, on: bool) -> None:
        # unprojected: xy only; projected: x and y (the cell's long and short axis), xy off
        self.direction_radios["xy"].setEnabled(not on)
        self.direction_radios["x"].setEnabled(on)
        self.direction_radios["y"].setEnabled(on)
        self.direction_radios["x" if on else "xy"].setChecked(True)
        self._load_steps()

    def _cell_size_edited(self) -> None:
        self._load_steps()

    def _fix_pair(self, on: bool) -> None:
        if on and not (self.fit_checks[2].isChecked() or self.fit_checks[3].isChecked()):
            self.fit_checks[2].setChecked(True)
            self.fit_checks[3].setChecked(True)

    def _pair_changed(self) -> None:
        self._draw_hists()

    def _direction_changed(self, checked: bool) -> None:
        if checked:
            self._draw_hists()

    # -- the steps and the histograms ---------------------------------------------------------------- #
    def opened(self) -> None:
        self._load_steps()

    def _load_steps(self) -> None:
        if not self._ready:
            return
        key = self._key()
        if key in self._steps:
            self._draw_hists()
            return
        project, projected, size = self.project, key[0], key[1]
        self.ref_panel.message("Loading the steps…")
        self.window.run_in_background(
            "Steps of the GMM tab",
            lambda: {c: lg.gmm_steps(project, c, projected=projected, cell_size=size) for c in project.conditions},
            lambda steps: self._steps_loaded(key, steps))

    def _steps_loaded(self, key, steps) -> None:
        self._steps[key] = steps
        self._draw_hists()

    def _current_steps(self):
        if self.run is not None and (self.run["projected"], self.run["cell_size"]) == self._key():
            return self.run["steps"]
        return self._steps.get(self._key())

    def _draw_hists(self) -> None:
        if not self._ready:
            return
        steps = self._current_steps()
        if steps is None:
            return
        direction = self.direction()
        panels = [(self.ref_panel, self.reference(), 0)]
        if len(self.conditions()) > 1:
            panels.append((self.comp_panel, self.comparison(), 1))
        else:
            self.comp_panel.message("One condition only.")
        run = self.run if (self.run is not None and self.run["direction"] == direction
                           and (self.run["projected"], self.run["cell_size"]) == self._key()) else None
        for panel, condition, _i in panels:
            panel.setTitle(condition)
            panel.draw(lambda fig, c=condition: self._paint_hist(fig, c, steps[c][direction], run))

    def _paint_hist(self, fig, condition: str, values, run) -> None:
        ldg.draw_hist(fig, condition, values, run, self.window.colours.get(condition, "#0072bd"),
                      f"{self.window.project_title} / {condition}", bin_width=self._value("BinWidth"),
                      xmin=self._value("min X"), xmax=self._value("max X"), ymax=self._value("max Y"),
                      grid=self._value("grid"))

    # -- the run ------------------------------------------------------------------------------------- #
    def start(self) -> None:
        fits = [k for k, check in self.fit_checks.items() if check.isChecked()]
        if not fits:
            self.window.log("Tick single fit, double fit or triple fit, then Run & Plot.")
            return
        project = self.project
        settings = dict(reference=self.reference(), comparison=self.comparison(), projected=self.projected(),
                        fix_pair=self.fix_check.isChecked(), fits=tuple(fits), cell_size=self.cell_size(),
                        direction=self.direction())
        steps = self._steps.get(self._key())
        self.run_button.setEnabled(False)
        self.ref_panel.message("Fitting running…")

        def compute():
            try:
                return lg.gmm_run(project, steps=steps, **settings)
            except Exception as exc:                     # noqa: BLE001 - named in the Console and the panels
                return {"failed": f"{type(exc).__name__}: {exc}"}

        self.window.run_in_background("Gaussian mixture fits", compute, self._done)

    def _done(self, run) -> None:
        self.run_button.setEnabled(True)
        if "failed" in run:
            self.window.log(f"The mixture fits failed: {run['failed']}")
            self.ref_panel.message("The fits failed:" + chr(10) + run["failed"])
            return
        self.run = run
        self._steps[(run["projected"], run["cell_size"])] = run["steps"]
        self.blocks = {key: r for key, r in self.blocks.items() if key[0] == run["projected"]
                       and r["cell_size"] == run["cell_size"]}
        self.blocks[(run["projected"], run["direction"])] = run
        for k, fit in sorted(run["fits"].items()):
            if k == 1:
                text = "; ".join(f"{c}: D {p['D']:.3g}" for c, p in fit["per"].items())
            else:
                text = "; ".join(f"{name} D {d:.3g}" for name, d in zip(lg.population_names(k), fit["D"]))
            self.window.log(f"{lg.FIT_NAMES[k]} ({run['direction']}): {text} um2/s.")
        if run["bic"]:
            self.window.log("Best model by BIC: " + ", ".join(f"{c}: {v[1]}" for c, v in run["bic"].items())
                            + f"; cross-validation: {run['cv']}")
            if any("(*)" in v[1] for v in run["bic"].values()):
                self.window.log("The BIC decreased less than 5%. The best model predicted could overestimate the "
                                "number of populations.")
        self._fill_table()
        self._draw_hists()
        self.bubble_panel.draw(self._draw_bubbles)
        self.crossed_panel.draw(self._draw_crossed)
        self.bottom.setVisible(True)
        self._open_gof()
        self.window.log("Done!")

    def _open_gof(self) -> None:
        if self.gof_window is None:
            self.gof_window = GmmGofWindow(self, self.window)
        self.gof_window.show_run(self.run, self.window.colours)
        self.gof_window.show()
        self.gof_window.raise_()

    # -- the table ------------------------------------------------------------------------------------- #
    def _fill_table(self) -> None:
        run = self.run
        blocks = {d: r for (p, d), r in self.blocks.items() if p == run["projected"]}
        rows = lg.gmm_table(run, blocks)
        conditions = run["conditions"]
        self.table.setColumnCount(len(conditions) + 1)
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(["Projected tracks" if run["projected"] else "Not Projected tracks"]
                                             + [f"{i}-{c}" for i, c in enumerate(conditions, start=1)])
        bold, italic = QFont(self.table.font()), QFont(self.table.font())
        bold.setBold(True)
        italic.setItalic(True)
        for r, row in enumerate(rows):
            for c, text in enumerate(row[: len(conditions) + 1]):
                item = QTableWidgetItem(str(text))
                if c == 0 and (str(text).startswith("Direction") or text == "significantly different?"):
                    item.setFont(bold)
                elif c == 0 and text in lg.FIT_NAMES.values():
                    item.setFont(italic)
                self.table.setItem(r, c, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    # -- the bubble plot and the crossed error bars ------------------------------------------------------- #
    def _draw_bubbles(self, fig) -> None:
        ldg.draw_bubbles(fig, self.run, self.window.colours, self.window.project_title)

    def _draw_crossed(self, fig) -> None:
        ldg.draw_crossed(fig, self.run, self.window.colours, self.window.project_title)

    def figures(self) -> dict:
        out = super().figures()
        if self.gof_window is not None:
            for panel in self.gof_window.panels:
                out[f"GoF {panel.title}"] = panel.figure
        return out
