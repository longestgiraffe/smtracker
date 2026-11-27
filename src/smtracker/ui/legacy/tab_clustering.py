"""The legacy window's Clustering tab: SMTracker 2.0's clustering tab (`SMTracker.m:553-621`; manual Fig. 9, p. 15).

Left, "Clustering options:" (condition, method k-means, Max 2 to 6 clusters, Specific tau, Run!) and under it a tab
group, "MSD plot" (one line per cluster, its width the cluster's share) and "Table" (2.0's "Characteristics of
cluster"). Right, "Cluster heat maps": one map per cluster on the model cell, in jet, the biggest first, with the
"Overlay Tracks" checkbox beneath. Every number from `legacy_view_clustering`.
"""
from __future__ import annotations

from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                            QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ... import legacy_draw_clustering as ldc
from ... import legacy_view_clustering as lc
from ...legacy_draw_clustering import HEAT_COLS, HEAT_ROWS  # noqa: F401
from .window import LegacyTab, PlotPanel

CONTROL_W = 360


class ClusteringTab(LegacyTab):
    title = "Clustering"

    def __init__(self, window):
        super().__init__(window)
        self.runs: dict = {}                      # {condition: run}, in the order they were run (2.0's table keeps all)
        self.last_run = None
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._left())
        right = QVBoxLayout()
        self.heat_panel = PlotPanel("Cluster heat maps", min_height=300)
        right.addWidget(self.heat_panel, 1)
        self.overlay_check = QCheckBox("Overlay Tracks")
        self.overlay_check.toggled.connect(self._draw_heat)
        right.addWidget(self.overlay_check)
        holder = QWidget()
        holder.setLayout(right)
        outer.addWidget(holder, 1)
        self.heat_panel.message("Press Run!.")
        self.msd_panel.message("Press Run!.")

    # -- the left column ----------------------------------------------------------------------- #
    def _left(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(CONTROL_W)
        lay = QVBoxLayout(column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        options = QGroupBox("Clustering options:")
        ol = QVBoxLayout(options)
        self.condition_box = QComboBox()
        self.condition_box.addItems(list(self.project.conditions))
        self.method_box = QComboBox()
        self.method_box.addItems(["k-means"])
        self.max_box = QComboBox()
        self.max_box.addItems([f"Max {k} clusters" for k in lc.MAX_CLUSTER_CHOICES])
        self.max_box.setCurrentIndex(lc.MAX_CLUSTER_CHOICES.index(lc.DEFAULT_MAX_CLUSTERS))
        for box in (self.condition_box, self.method_box, self.max_box):
            ol.addWidget(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Specific τ"))
        self.tau_field = QLineEdit(str(lc.DEFAULT_SPECIFIC_TAU))
        row.addWidget(self.tau_field)
        ol.addLayout(row)
        self.run_button = QPushButton("Run!")
        self.run_button.clicked.connect(self.start)
        ol.addWidget(self.run_button)
        lay.addWidget(options)

        self.group = QTabWidget()
        self.msd_panel = PlotPanel("MSD plot", min_height=240)
        self.msd_panel.setTitle("")                  # the tab already says it (2.0's panel title was a blank)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnCount(2)
        self.table.setWordWrap(True)
        self.table.setHorizontalHeaderLabels(["", "Characteristics of cluster"])
        self.group.addTab(self.msd_panel, "MSD plot")
        self.group.addTab(self.table, "Table")
        lay.addWidget(self.group, 1)
        return column

    # -- the run ------------------------------------------------------------------------------- #
    def start(self) -> None:
        project = self.project
        condition = self.condition_box.currentText()
        max_clusters = lc.MAX_CLUSTER_CHOICES[max(0, self.max_box.currentIndex())]
        try:
            tau = int(float(self.tau_field.text()))
        except ValueError:
            self.window.log(f"Specific τ must be a whole number of localisations, not {self.tau_field.text()!r}.")
            return
        self.run_button.setEnabled(False)
        self.heat_panel.message("Clustering running…")
        self.msd_panel.message("Clustering running…")

        def compute():
            # the window reports a failure in the Console; this one also hands the Run! button back
            try:
                return lc.clustering_run(project, condition, max_clusters=max_clusters, specific_tau=tau)
            except Exception as exc:                     # noqa: BLE001 - named in the Console and the panels
                return {"failed": f"{type(exc).__name__}: {exc}"}

        self.window.run_in_background(f"Clustering of {condition}", compute, self._done)

    def _done(self, run) -> None:
        self.run_button.setEnabled(True)
        if "failed" in run:
            self.window.log(f"Clustering failed: {run['failed']}")
            for panel in (self.heat_panel, self.msd_panel):
                panel.message("Clustering failed:" + chr(10) + run["failed"])
            return
        self.runs.pop(run["condition"], None)
        self.runs[run["condition"]] = run
        self.last_run = run
        for i, cluster in enumerate(run["clusters"], start=1):
            if not cluster["classified"]:
                self.window.log(f"Cluster {i} ({cluster['n_tracks']} tracks) is not classified: {cluster['reason']}")
        self._fill_table()
        self.msd_panel.draw(self._draw_msd)
        self._draw_heat()
        self.window.log("Done!")

    def _fill_table(self) -> None:
        rows = lc.cluster_table_rows(self.runs.values())
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.resizeRowsToContents()            # 2.0's 320 px column cut the text; here it wraps

    # -- the plots ------------------------------------------------------------------------------- #
    def _draw_msd(self, fig) -> None:
        ldc.draw_msd(fig, self.last_run)

    def _draw_heat(self, *_args) -> None:
        if self.last_run is None:
            return
        self.heat_panel.draw(self._paint_heat)

    def _paint_heat(self, fig) -> None:
        ldc.draw_heat(fig, self.last_run, self.overlay_check.isChecked())
