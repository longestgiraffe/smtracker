"""The legacy window's Track Explorer: SMTracker 2.0's second window (`SMTracker.m:1054-1095`; manual Fig. 5).

Opened from the Import tab (a click on a track in the overlay, as in 2.0, or the Track explorer button). Six panels
under 2.0's blue title bars: the projection of the cell's tracks (the picked one coloured by time), the track's
movement from its start with 2.0's confinement circle, its distance to the origin and step lengths over time with
the circle's radius as a line, its intensity profile, the MSD of every track in its cell (Save MSD in csv...), and
the track's movie frames with a slider. Numbers from `legacy_view.track_explorer`.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QFileDialog, QGridLayout, QPushButton, QSlider, QVBoxLayout, QWidget)

from ... import legacy_draw_import as ldi
from ... import legacy_view as lv
from .style import WINDOW_QSS
from .window import PlotPanel

#: 2.0's explorer panels: `'BackgroundColor', [0 .3 .7]` with white titles.
EXPLORER_BLUE = "#004db3"
EXPLORER_QSS = WINDOW_QSS + f"""
QGroupBox {{ border: 1px solid {EXPLORER_BLUE}; margin-top: 18px; }}
QGroupBox::title {{ background: {EXPLORER_BLUE}; color: #ffffff; padding: 1px 6px; left: 0px; }}
"""


class TrackExplorer(QWidget):
    """One track of the legacy window's project, as 2.0's Track Explorer showed it."""

    def __init__(self, legacy_window, condition: str, uid, *, projected=None, radius_cache: dict | None = None):
        super().__init__(None, Qt.Window)
        self.legacy = legacy_window
        self.project = legacy_window.project
        self.condition = condition
        self.uid = str(uid)
        self._projected = projected
        self._radius_cache = radius_cache if radius_cache is not None else {}
        self.data = None
        self._frames = None
        self._track_px = None
        self.setWindowTitle(f"Track Explorer – {condition}, track {self.uid.rsplit(':', 1)[-1]}")
        self.setStyleSheet(EXPLORER_QSS)
        grid = QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)
        self.projection = PlotPanel("Projection of tracks on cells", min_height=180)
        self.movement = PlotPanel("Track movement", min_height=180)
        self.distance = PlotPanel("Distance to origin", min_height=180)
        self.intensity = PlotPanel("Intensity profile", min_height=180)
        self.msd = PlotPanel("MSD per cell", min_height=180)
        self.movie = PlotPanel("Track Movie", min_height=180)
        msd_box = QWidget()
        ml = QVBoxLayout(msd_box)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.msd, 1)
        self.save_msd = QPushButton("Save MSD in csv…")
        self.save_msd.clicked.connect(self.save_msd_csv)
        ml.addWidget(self.save_msd)
        movie_box = QWidget()
        vl = QVBoxLayout(movie_box)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.movie, 1)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._draw_movie)
        vl.addWidget(self.slider)
        for i, widget in enumerate((self.projection, self.movement, self.distance, self.intensity, msd_box,
                                    movie_box)):
            grid.addWidget(widget, i // 2, i % 2)
        for panel in (self.projection, self.movement, self.distance, self.intensity, self.msd, self.movie):
            panel.message("Loading…")
        self.resize(1000, 880)
        self._load()

    # -- loading --------------------------------------------------------------------------------- #
    def _load(self) -> None:
        project, condition, uid = self.project, self.condition, self.uid
        projected, cache = self._projected, self._radius_cache

        def compute():
            data = lv.track_explorer(project, condition, uid)
            table = projected if projected is not None else project.projected(condition)
            if condition not in cache:
                try:
                    cache[condition] = lv.confinement_radius_um(project, condition)
                except lv.LegacyViewError:
                    cache[condition] = None
            frames, error = None, None
            try:
                from ...images_view import frame_block

                frames = frame_block(project, condition, data["movie_original"], min(data["frames"]),
                                     max(data["frames"]))
            except Exception as exc:                      # noqa: BLE001 - said on the panel
                error = f"No recorded frames for {data['movie_original']}: {exc}"
            return data, table, cache[condition], frames, error

        self.legacy.run_in_background(f"Track Explorer, track {uid.rsplit(':', 1)[-1]}", compute, self._loaded)

    def _loaded(self, result) -> None:
        data, table, radius, frames, error = result
        self.data, self._table, self._radius, self._frames = data, table, radius, frames
        self.projection.draw(self._draw_projection)
        self.movement.draw(self._draw_movement)
        self.distance.draw(self._draw_distance)
        self.intensity.draw(self._draw_intensity)
        self.msd.draw(self._draw_msd)
        if frames and frames.get("frames"):
            self.slider.setRange(0, len(frames["frames"]) - 1)
            self.slider.setEnabled(True)
            self._draw_movie(0)
        else:
            self.movie.message(error or "No recorded frames for this movie.")

    # -- the panels -------------------------------------------------------------------------------- #
    def _draw_projection(self, fig) -> None:
        ldi.draw_projection(fig, self._table, self.data)

    def _draw_movement(self, fig) -> None:
        ldi.draw_movement(fig, self.data, self._radius)

    def _draw_distance(self, fig) -> None:
        ldi.draw_distance(fig, self.data, self._radius)

    def _draw_intensity(self, fig) -> None:
        ldi.draw_intensity(fig, self.data)

    def _draw_msd(self, fig) -> None:
        ldi.draw_msd(fig, self.data)

    def _draw_movie(self, index: int) -> None:
        frames = self._frames
        if not frames or not frames.get("frames"):
            return
        if self._track_px is None:
            self._track_px = lv.track_in_pixels(self.project, self.condition, self.data["uid"])
        track, frame_no = self._track_px, int(frames["first"]) + int(index)
        self.movie.draw(lambda fig: ldi.draw_movie_frame(fig, frames, index, track, frame_no))

    # -- saving -------------------------------------------------------------------------------------- #
    def save_msd_csv(self, path=None):
        """2.0's "Save MSD in csv...": the cell's MSD curves, one column per track and the mean."""
        if self.data is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save MSD per cell", "legacy_msd_per_cell.csv",
                                                  "CSV (*.csv)")
            if not path:
                return None
        lv.save_msd_per_cell(self.data, path)
        self.legacy.log(f"Saved the MSD per cell into {path}.")
        return path
