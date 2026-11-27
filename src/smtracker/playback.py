"""Playable track view without Qt, OpenGL or napari.

Why this exists, measured. The napari viewer's cost is not the data and not the import:
preparing every layer for a whole condition takes 0.09 s and ``import napari`` takes
0.15 s over 204 modules (it is lazy). What the user waits for is ``napari.Viewer()``
building a Qt window and an OpenGL context. A matplotlib canvas skips both.

And this project has no movie to stream. It ships ``phase/`` and ``sum/`` at 126x130,
not the raw 1001-frame stack, so playback here animates tracks over ONE static image.
That is the case matplotlib is good at, and the reason the recommendation is not general:
hand this a real 1001-frame stack and the picture changes, because then the per-frame
cost is an image upload and napari's GPU path is the right tool. Both are kept.

MEASURED on one lab movie (275 tracks, 1001 frames, 126x130 phase image),
blitted, Agg backend: **1699 frames per second, 0.6 ms per frame**, against an
acquisition rate of 50 frames per second. Thirty-four-fold headroom, so playback is
real-time with room to spare and the frame update is not where any future slowness will
be. The update is O(localisations in the movie) per frame, which is about 1.7e3 here;
the crossover where that stops being free sits far above any single-molecule movie this
tool will see, and ``TrackPlayer.headroom`` reports the margin for the data in hand
rather than restating this paragraph.

The compute is outside the GUI, as everywhere else in this package. ``TrackPlayer`` owns
numpy arrays and matplotlib artists and has one method that matters, ``draw_frame``. A Qt
play button calls it on a timer; a script calls it in a loop to write a GIF; a test calls
it headless. Nothing is computed in a Qt class, so a viewer that shows the wrong track
fails in ``test_playback``, not by looking subtly wrong on screen.

Coordinates follow ``smtracker.viewer``: um -> pixels with the SAME pixel size the
analysis used, and ``PIXEL_ORIGIN_OFFSET`` subtracted because u-track and MATLAB are
1-based. Getting that wrong puts every track half a cell off, which is why it is done in
one place and regression-tested against the real project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "PlaybackError",
    "TrackPlayer",
    "DEFAULT_TRAIL_FRAMES",
]

# u-track/MATLAB index from 1; the arrays here index from 0.
PIXEL_ORIGIN_OFFSET = 1

# How many frames of a track stay drawn behind the current position. A trail is a display
# choice, not a measurement, so it is printed on the figure and never silently assumed.
DEFAULT_TRAIL_FRAMES = 20


class PlaybackError(RuntimeError):
    """Raised when playback is asked for something it cannot honestly draw."""


class TrackPlayer:
    """Tracks of one movie, playable over a static background.

    Parameters
    ----------
    canonical:
        Canonical table, already restricted to ONE movie. Restricting is the caller's
        job and is checked, because silently playing two movies' tracks over one movie's
        image would look plausible and be wrong.
    background:
        2-D image array. ``None`` draws on a blank ground and the figure says so; a
        missing image is not invented.
    pixel_size_nm:
        No default. The conversion is the one place a viewer can be wrong by a whole
        cell, so it comes from the acquisition parameters or not at all.
    """

    def __init__(self, canonical, *, pixel_size_nm, background=None,
                 trail_frames=DEFAULT_TRAIL_FRAMES, frame_interval_ms=None):
        required = {"uid", "frame", "x_um", "y_um"}
        missing = sorted(required - set(canonical.columns))
        if missing:
            raise PlaybackError(f"not a canonical table (missing {missing})")
        if canonical.empty:
            raise PlaybackError("no localisations to play")
        if "movie" in canonical.columns and canonical["movie"].nunique() > 1:
            names = sorted(canonical["movie"].unique())
            raise PlaybackError(
                f"{len(names)} movies in one table ({names[:4]}...): playback draws one "
                f"movie's tracks over one movie's image, so restrict the table first. "
                f"Playing them together would look plausible and be wrong.")
        if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
            raise PlaybackError(f"pixel_size_nm must be positive, got {pixel_size_nm!r}")
        if int(trail_frames) < 0:
            raise PlaybackError("trail_frames cannot be negative")

        self.pixel_size_nm = float(pixel_size_nm)
        self.trail_frames = int(trail_frames)
        self.frame_interval_ms = (None if frame_interval_ms is None
                                  else float(frame_interval_ms))
        self.background = None if background is None else np.asarray(background)

        ordered = canonical.sort_values(["uid", "frame"], kind="stable")
        scale = self.pixel_size_nm / 1000.0
        self._x = ordered["x_um"].to_numpy(dtype=float) / scale - PIXEL_ORIGIN_OFFSET
        self._y = ordered["y_um"].to_numpy(dtype=float) / scale - PIXEL_ORIGIN_OFFSET
        self._frame = ordered["frame"].to_numpy(dtype=np.int64)
        self._uid = ordered["uid"].to_numpy()
        # Where one track ends and the next begins, so a trail never joins two molecules.
        self._new_track = np.empty(len(self._uid), dtype=bool)
        self._new_track[0] = True
        self._new_track[1:] = self._uid[1:] != self._uid[:-1]

        self.first_frame = int(self._frame.min())
        self.last_frame = int(self._frame.max())
        self.n_tracks = int(pd.unique(self._uid).size)
        self.n_localisations = int(len(self._x))
        self._artists = None
        self._background_cache = None
        self._axes = None

    # ------------------------------------------------------------------ the frame range
    @property
    def frames(self):
        """Every frame index the player can be asked for, inclusive of both ends."""
        return np.arange(self.first_frame, self.last_frame + 1)

    def visible(self, frame):
        """``(x, y)`` of the localisations ON ``frame``, and the trail behind them.

        Returns ``(head_x, head_y, trail_x, trail_y)``. The trail carries NaN between
        tracks so one polyline draws all of it; joining two molecules with a straight
        line is the failure this guards against.
        """
        frame = int(frame)
        if not (self.first_frame <= frame <= self.last_frame):
            raise PlaybackError(
                f"frame {frame} is outside the movie, which runs "
                f"{self.first_frame} to {self.last_frame}")
        head = self._frame == frame
        if self.trail_frames == 0:
            return self._x[head], self._y[head], np.array([]), np.array([])
        window = (self._frame <= frame) & (self._frame > frame - self.trail_frames)
        tx, ty = self._x[window], self._y[window]
        # THE BREAKS ARE COMPUTED ON THE WINDOW, NOT READ OFF THE FULL ARRAY. A track
        # whose first localisation lies BEFORE the trail window is not a new track
        # globally, so a precomputed flag says False exactly where a break is needed and
        # the trail draws a straight line from one molecule to another. That line looks
        # like a real, very fast step and was visible on the first real frame drawn:
        # lower-left cell to upper-right cell, about 4 um in 20 ms. The first version of
        # the test missed it because every track in the fixture began at frame 0, where
        # the flag happens to be right.
        w_uid = self._uid[window]
        breaks = np.flatnonzero(w_uid[1:] != w_uid[:-1]) + 1
        return (self._x[head], self._y[head],
                np.insert(tx, breaks, np.nan), np.insert(ty, breaks, np.nan))

    # ------------------------------------------------------------------------- drawing
    def attach(self, ax, *, trail_colour="#e53e3e", head_colour="#f6e05a",
               cmap="gray", trail_width=0.8, head_size=3.0):
        """Put the background and the two artists on ``ax``. Call once, then draw_frame."""
        if self.background is not None:
            ax.imshow(self.background, cmap=cmap, interpolation="nearest")
            height, width = self.background.shape[:2]
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)
            ground = "%dx%d image" % (height, width)
        else:
            ax.set_xlim(self._x.min() - 5, self._x.max() + 5)
            ax.set_ylim(self._y.max() + 5, self._y.min() - 5)
            ax.set_facecolor("black")
            ground = "NO IMAGE: positions only, nothing to register them against"
        ax.set_aspect("equal")
        ax.set_axis_off()
        trail, = ax.plot([], [], "-", color=trail_colour, lw=trail_width,
                         solid_capstyle="round", animated=True)
        head, = ax.plot([], [], "o", color=head_colour, ms=head_size, mec="none",
                        animated=True)
        self._artists = (trail, head)
        self._axes = ax
        self.ground = ground
        return trail, head

    def draw_frame(self, frame, *, blit=True):
        """Update the artists to ``frame``. This is the call a play button makes."""
        if self._artists is None:
            raise PlaybackError("attach(ax) before draw_frame")
        trail, head = self._artists
        hx, hy, tx, ty = self.visible(frame)
        trail.set_data(tx, ty)
        head.set_data(hx, hy)
        if not blit:
            return trail, head
        canvas, ax = self._axes.figure.canvas, self._axes
        if self._background_cache is None:
            canvas.draw()
            self._background_cache = canvas.copy_from_bbox(ax.bbox)
        canvas.restore_region(self._background_cache)
        ax.draw_artist(trail)
        ax.draw_artist(head)
        canvas.blit(ax.bbox)
        return trail, head

    def invalidate(self):
        """Forget the cached background. Call after resizing or changing the image."""
        self._background_cache = None

    # ---------------------------------------------------------------------- measurement
    def headroom(self, *, repeats=60, seed=0):
        """Frames per second this player achieves, against the acquisition rate.

        A number measured on the data in hand beats a number quoted from another dataset,
        which is why this is a method and not a docstring. Returns both rates and their
        ratio; the ratio is the honest answer to "will playback keep up".
        """
        import time

        if self._artists is None:
            raise PlaybackError("attach(ax) before headroom")
        rng = np.random.default_rng(seed)
        picks = rng.integers(self.first_frame, self.last_frame + 1, size=int(repeats))
        self.draw_frame(int(picks[0]))
        start = time.perf_counter()
        for frame in picks:
            self.draw_frame(int(frame))
        elapsed = time.perf_counter() - start
        achieved = len(picks) / elapsed if elapsed > 0 else float("inf")
        acquired = (None if not self.frame_interval_ms
                    else 1000.0 / self.frame_interval_ms)
        return {
            "frames_per_second": float(achieved),
            "ms_per_frame": float(1000.0 * elapsed / len(picks)),
            "acquisition_frames_per_second": acquired,
            "headroom": (None if acquired is None else float(achieved / acquired)),
            "n_localisations": self.n_localisations,
            "n_tracks": self.n_tracks,
        }

    def caption(self):
        """What the figure must say about itself, in one line."""
        trail_ms = ("" if self.frame_interval_ms is None else
                    ", %.0f ms" % (self.trail_frames * self.frame_interval_ms))
        return ("N: %d tracks, %d localisations, frames %d to %d   trail %d frames%s   "
                "%s   %.1f nm per pixel"
                % (self.n_tracks, self.n_localisations, self.first_frame,
                   self.last_frame, self.trail_frames, trail_ms,
                   getattr(self, "ground", "not attached"), self.pixel_size_nm))
