"""The SQD tab's "Jump Distance Analysis & GoF" window: SMTracker 2.0's pop-up after an SQD fit at one frame
(`SMTracker.m` `plotPDF`, `ppplot_jd`; manual Fig. 13): per condition, the single-step distances as a histogram with
the fit's populations (after Weimann et al. 2013), and under it the P-P plot with its straight line. Numbers from
`legacy_view.jd_gof`."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QWidget

from ... import legacy_draw_sqd as lds
from ... import legacy_view as lv
from ...legacy_draw_sqd import FIT_NAMES  # noqa: F401
from .style import WINDOW_QSS
from .window import PlotPanel


class JumpDistanceWindow(QWidget):
    """2.0's window for one SQD run: a column per condition."""

    def __init__(self, legacy_window, run: dict, n_pop: int, conditions, colours: dict):
        super().__init__(None, Qt.Window)
        self.setWindowTitle("Jump Distance Analysis & GoF")
        self.setStyleSheet(WINDOW_QSS)
        grid = QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        self.panels = []
        for col, condition in enumerate(conditions):
            index = run["conditions"].index(condition) + 1
            data = lv.jd_gof(run, condition, n_pop)
            colour = colours.get(condition, "#0072bd")
            hist = PlotPanel(f"{index}-{condition}", min_height=230)
            pp = PlotPanel(f"{index}-{condition}, goodness of fit", min_height=230)
            hist.draw(lambda fig, d=data, c=colour, i=index, n=condition: self._histogram(fig, d, c, i, n))
            pp.draw(lambda fig, d=data: self._pp(fig, d))
            grid.addWidget(hist, 0, col)
            grid.addWidget(pp, 1, col)
            self.panels += [hist, pp]
        self.resize(max(420, 360 * len(conditions)), 720)

    @staticmethod
    def _histogram(fig, data, colour, index, name) -> None:
        lds.draw_jd_histogram(fig, data, colour, index, name)

    @staticmethod
    def _pp(fig, data) -> None:
        lds.draw_jd_pp(fig, data)
