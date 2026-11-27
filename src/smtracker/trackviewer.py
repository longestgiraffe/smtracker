"""Open the napari single-track viewer for one condition, from the GUI.

The comparison browser compares conditions; when a molecule looks interesting the
next question is "show me that track on the cell". This bridges the two: given a
loaded multi-condition ``Project`` and a condition, it opens the napari inspector
on that condition's own folder: where the movie images, the cell outlines and
the ORIGINAL (unqualified) movie names live, exactly as ``smtracker --inspect``
expects. So the viewer is reachable from a button, not only a separate CLI run.

It reuses the tested viewer/inspector machinery (``build_bundle`` / ``launch`` /
``dock_inspector`` / ``dock_track_list``): nothing new is computed here. napari
is imported lazily inside ``open_for_condition`` so importing this module stays
Qt-free (and the AST thin-shell guards keep holding).
"""

from __future__ import annotations

from .projectcache import ProjectCache

__all__ = ["TrackViewerError", "movies_for_condition", "selector_state",
           "open_for_condition", "load_condition_data", "clear_condition_cache",
           "pick_condition_movie", "track_target"]


def pick_condition_movie(project, *, condition=None, movie=None, parent=None):
    """A plain-Qt dialog to choose condition + movie BEFORE opening napari.

    Returns ``(condition, movie)`` or ``None`` if cancelled. Deliberately has NO
    GL / napari: the heavy, crash-prone window is opened only once the choice is
    made, so a mis-click never spins up a viewer. The movie list follows the
    chosen condition.
    """
    from qtpy.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
                                QFormLayout, QLabel)

    # Building a widget with no QApplication aborts the process - no exception, no
    # message (see `launcher.choose_stage`). Today every caller is inside the
    # browser, so one already exists; this is exported in `__all__`, so that is a
    # property of the callers rather than of this function.
    QApplication.instance() or QApplication([])
    from .appicon import apply as _apply_icon
    _apply_icon()

    conditions, movies, condition, movie = selector_state(project, condition, movie)
    dialog = QDialog(parent)
    dialog.setWindowTitle("Open track viewer: choose condition / movie")
    form = QFormLayout(dialog)
    form.addRow(QLabel("Pick before opening the viewer (opens one window):"))
    cond_combo = QComboBox(); cond_combo.addItems(conditions)
    cond_combo.setCurrentText(condition)
    movie_combo = QComboBox(); movie_combo.addItems(movies)
    if movie:
        movie_combo.setCurrentText(movie)

    def on_condition(*_):
        _, ms, _, _ = selector_state(project, cond_combo.currentText())
        movie_combo.clear(); movie_combo.addItems(ms)

    cond_combo.currentTextChanged.connect(on_condition)
    form.addRow("condition:", cond_combo)
    form.addRow("movie:", movie_combo)
    buttons = QDialogButtonBox(
        QDialogButtonBox.Open | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None
    return cond_combo.currentText(), movie_combo.currentText()


class TrackViewerError(RuntimeError):
    """Raised when the track viewer cannot be opened for a condition."""


#: Per-condition load cache: the EXPENSIVE, movie-independent work (decode
#: movieData, read every cell mesh, read every movie's tracks, canonicalise, fit
#: the HMM) done ONCE per condition. Switching movie within a condition: the
#: common case that was slow, then rebuilds only the cheap per-movie bundle, and
#: revisiting a condition is instant. Conditions are few (2–8), so we keep them
#: all; ``clear_condition_cache`` frees them (and is called by tests).
#: Per project OBJECT, never per id: an id is reused once a project is freed
#: (`projectcache`).
_CONDITION_CACHE = ProjectCache()


def clear_condition_cache() -> None:
    """Drop all cached per-condition loads (frees their canonical tables)."""
    _CONDITION_CACHE.clear()


def load_condition_data(project, condition, *, hmm_states=2,
                        localization_precision_nm=None):
    """The condition-level load (params, meshes, canonical, HMM fit), CACHED.

    This is everything that does NOT depend on which movie is shown, so it is done
    once per condition and reused when the user switches movies or returns to a
    condition. Returns ``(params, meshes, canonical, fit)``. Raises
    ``TrackViewerError`` (not a napari traceback) on a missing folder / no tracks.
    """
    if condition not in project.roots:
        raise TrackViewerError(
            f"no folder recorded for condition {condition!r}; cannot open the "
            "track viewer")
    if localization_precision_nm is None:
        localization_precision_nm = getattr(
            project.params, "localization_precision_nm", None) or 30.0

    key = (condition, float(localization_precision_nm), int(hmm_states))
    cached = _CONDITION_CACHE.get(project, key)
    if cached is not None:
        return cached

    # THE PROJECT IS ALREADY LOADED, SO NOTHING IS READ AGAIN. This used to re-read the
    # condition from disk with u-track's own readers (`acquisition_params_from_project`,
    # `load_utrack_project`), so a project tracked with TrackMate -- no `movieData.mat` --
    # could not open napari at all: "MetadataError: no movieData.mat". A second read
    # path is also the shape this project keeps paying for (the day map, the cache): the
    # viewer now shows exactly the tables, cells and parameters the analysis uses.
    params = project.params
    canonical, meshes = _original_names(project, condition)
    if canonical.empty:
        raise TrackViewerError(f"condition {condition!r} produced no tracks")
    fit = _try_hmm(canonical, params, hmm_states)

    result = (params, meshes, canonical, fit)
    return _CONDITION_CACHE.put(project, key, result)


def _original_names(project, condition):
    """``(canonical, meshes)`` of one condition AS LOADED, under ORIGINAL movie names.

    The project qualifies every movie (``<condition>_series001``) so conditions cannot
    collide; napari finds a movie's images on disk (``phase/series001.tif``), so the
    viewer works in the original names. ``movie_source`` holds the original for each
    qualified movie; ``uid`` and ``cell`` (``<movie>:<n>``) follow their movie.
    """
    from dataclasses import replace
    from pathlib import Path

    part = project.condition(condition)
    source = getattr(project, "movie_source", None) or {}
    prefix = f"{condition}_"

    def original(qualified):
        qualified = str(qualified)
        if qualified in source:
            return str(source[qualified][1])
        return qualified[len(prefix):] if qualified.startswith(prefix) else qualified

    names = {m: original(m) for m in part["movie"].astype(str).unique()}

    def rekey(value, movie):
        if value is None or value != value:            # NaN: a track with no cell
            return value
        return f"{names[movie]}:{str(value).rsplit(':', 1)[-1]}"

    movies = part["movie"].astype(str)
    out = part.copy()
    out["uid"] = [rekey(u, m) for u, m in zip(part["uid"], movies)]
    if "cell" in out.columns:
        out["cell"] = [rekey(c, m) for c, m in zip(part["cell"], movies)]
    out["movie"] = movies.map(names)
    out.attrs.update(part.attrs)

    # THIS CONDITION'S OUTLINES, by where each movie came from rather than by prefix
    # alone: a condition "Glu" would otherwise claim "Glu_2_series001".
    # RESOLVED on both sides: a project opened by a RELATIVE path keeps its roots relative
    # while `movie_source` holds them absolute, and the two never compared equal -- every
    # outline was dropped, silently (found by the Images & tracks section's first run).
    root = Path(project.roots[condition]).resolve()
    meshes = {}
    for qualified, cells in (getattr(project, "meshes", None) or {}).items():
        qualified = str(qualified)
        belongs = (Path(source[qualified][0]).resolve() == root if qualified in source
                   else qualified.startswith(prefix))
        if belongs:
            short = original(qualified)
            meshes[short] = [replace(cell, movie=short) for cell in cells]
    return out, meshes


def movies_for_condition(project, condition):
    """The ORIGINAL movie names available for a condition (for a movie picker)."""
    from .project import qualified_movie

    part = project.condition(condition)
    qualified = sorted(part["movie"].unique())
    prefix = f"{condition}_"
    return [m[len(prefix):] if m.startswith(prefix) else m for m in qualified]


def selector_state(project, condition, movie=None):
    """The in-window selector's (conditions, movies, condition, movie), PURE.

    Defaults the condition to the one requested (i.e. the picker's current
    selection) falling back to the first, and the movie to the one requested
    falling back to the first available. Pure so the defaults are unit-tested even
    though napari itself cannot run headless.
    """
    conditions = list(project.conditions)
    if condition not in conditions:
        condition = conditions[0]
    movies = movies_for_condition(project, condition)
    if movie not in movies:
        movie = movies[0] if movies else None
    return conditions, movies, condition, movie


def track_target(project, condition, uid, *, projected: bool = True):
    """Everything either track viewer needs to open ONE track of a loaded project.

    The map's spot card names a track by the PROJECT's uid (``<condition>_<movie>:<n>``).
    The fast view reads the project's own tables, so it takes that uid as it is; napari
    loads the condition from its own folder under the ORIGINAL movie names, so it needs
    the prefix stripped (``short_movie``, ``short_uid``) -- the same transformation
    :func:`movies_for_condition` makes. Restricting to the track's movie happens here,
    never in a Qt class. Raises ``TrackViewerError`` for a uid the condition does not hold.
    """
    part = project.condition(condition)
    rows = part[part["uid"].astype(str) == str(uid)]
    if rows.empty:
        raise TrackViewerError(f"no track {uid!r} in condition {condition!r}")
    movie = str(rows["movie"].iloc[0])
    cell = rows["cell"].iloc[0] if "cell" in rows.columns else None
    prefix = f"{condition}_"
    short_movie = movie[len(prefix):] if movie.startswith(prefix) else movie
    track_id = str(uid).rsplit(":", 1)[-1]
    # ``projected=False`` skips the cell-frame projection (seconds on a fresh condition)
    # for a caller that only needs the names: the workspace's viewers.
    projected = project.projected(condition) if projected else None
    params = getattr(project, "params", None)
    return {
        "condition": condition, "uid": str(uid), "movie": movie, "cell": cell,
        "track_id": track_id, "short_movie": short_movie,
        "short_uid": f"{short_movie}:{track_id}",
        "canonical": part[part["movie"].astype(str) == movie],
        "projected": (None if projected is None else
                      projected[projected["movie"].astype(str) == movie]),
        "pixel_size_nm": getattr(params, "pixel_size_nm", None),
        "frame_interval_ms": getattr(params, "frame_interval_ms", None),
    }


def open_for_condition(project, condition, *, movie=None, hmm_states=2,
                       localization_precision_nm=None, inspect=True, uid=None):
    """Load ``condition`` standalone from its own folder and open napari on it.

    ``movie`` picks which movie to show (default: the first), ``uid`` a track in it
    to inspect at once (ORIGINAL movie naming, see :func:`track_target`). Returns the
    napari ``Viewer``. Raises ``TrackViewerError`` with a plain message if the
    condition has no folder or no loadable movie, rather than a napari traceback.
    """
    from .viewer import launch

    # One window for the whole session. The first movie is built, a viewer is
    # launched ONCE, and after that every condition/movie change updates THIS
    # window in place, never a second napari window (that churns the shared GL
    # context and crashes on some drivers).
    session = _TrackViewerSession(
        project, hmm_states=hmm_states,
        localization_precision_nm=localization_precision_nm, inspect=inspect)
    bundle, state = session.prepare(condition, movie)   # raises TrackViewerError
    viewer = launch(bundle)
    session.attach(viewer)
    session.finish_first(bundle, state)
    if uid is not None and session._inspector is not None:    # pragma: no cover - napari
        session._inspector.show_uid(str(uid))
    return viewer


class _TrackViewerSession:
    """Owns the single napari window and switches movie/condition IN PLACE.

    Every change reuses the same window (one GL context): the condition load is
    cached, the layers are replaced via ``viewer.repopulate``, and the inspector /
    track-list docks are rebuilt for the new movie. Reopening a second window,
    the old behaviour: churned the vispy GL context and crashed
    (``GL_INVALID_OPERATION``); this never opens one.
    """

    def __init__(self, project, *, hmm_states=2, localization_precision_nm=None,
                 inspect=True):
        self.project = project
        self.hmm_states = hmm_states
        self.loc_nm = localization_precision_nm
        self.inspect = inspect
        self.viewer = None
        self.condition = None
        self.movie = None
        self._inspector = None
        self._track_list = None
        self._selector = None            # (cond_combo, movie_combo)

    def prepare(self, condition, movie):
        """Load + build the bundle for (condition, movie). Raises loudly."""
        from .viewer import ViewerError, build_bundle

        if condition not in self.project.roots:
            raise TrackViewerError(
                f"no folder recorded for condition {condition!r}")
        params, meshes, canonical, fit = load_condition_data(
            self.project, condition, hmm_states=self.hmm_states,
            localization_precision_nm=self.loc_nm)
        movies = sorted(canonical["movie"].unique())
        movie = movie or movies[0]
        if movie not in movies:
            movie = movies[0]
        try:
            bundle = build_bundle(canonical, movie=movie,
                                  meshes=meshes, root=self.project.roots[condition])
        except ViewerError as exc:
            raise TrackViewerError(
                f"could not build the viewer for {condition}:{movie} ({exc})"
            ) from exc
        return bundle, (params, meshes, canonical, fit, movie, condition)

    def attach(self, viewer):
        self.viewer = viewer
        viewer._smtracker_session = self          # keep the session alive

    def finish_first(self, bundle, state):
        params, meshes, canonical, fit, movie, condition = state
        if self.inspect:
            self._inspector, self._track_list = _dock(
                self.viewer, bundle, canonical, meshes, params, movie, fit)
        self.condition, self.movie = condition, movie
        self._dock_selector()

    def switch(self, condition, movie=None):
        """Change condition/movie IN the existing window: no new viewer."""
        from .viewer import repopulate

        try:
            bundle, state = self.prepare(condition, movie)
        except TrackViewerError:                  # pragma: no cover - napari
            return
        params, meshes, canonical, fit, movie, condition = state
        repopulate(self.viewer, bundle)           # in-place layer swap, one context
        if self.inspect:
            self._rebuild_docks(bundle, canonical, meshes, params, movie, fit)
        self.condition, self.movie = condition, movie
        self._sync_selector()

    def _rebuild_docks(self, bundle, canonical, meshes, params, movie, fit):
        for widget in (self._inspector, self._track_list):
            if widget is not None:
                try:                              # pragma: no cover - napari
                    self.viewer.window.remove_dock_widget(widget)
                except Exception:
                    pass
        self._inspector, self._track_list = _dock(
            self.viewer, bundle, canonical, meshes, params, movie, fit)

    # -- the in-window selector, now wired to switch() (no reopen) ---------- #

    def _dock_selector(self):                     # pragma: no cover - napari
        try:
            from qtpy.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QWidget)
        except Exception:
            return
        conditions, movies, condition, movie = selector_state(
            self.project, self.condition, self.movie)
        panel = QWidget()
        row = QHBoxLayout(panel)
        cond_combo = QComboBox(); cond_combo.addItems(conditions)
        cond_combo.setCurrentText(condition)
        movie_combo = QComboBox(); movie_combo.addItems(movies)
        if movie:
            movie_combo.setCurrentText(movie)
        row.addWidget(QLabel("condition:")); row.addWidget(cond_combo)
        row.addWidget(QLabel("movie:")); row.addWidget(movie_combo)
        self._selector = (cond_combo, movie_combo)

        def on_condition(*_):
            _, ms, _, _ = selector_state(self.project, cond_combo.currentText())
            movie_combo.blockSignals(True)
            movie_combo.clear(); movie_combo.addItems(ms)
            movie_combo.blockSignals(False)
            self.switch(cond_combo.currentText(), movie_combo.currentText())

        def on_movie(*_):
            self.switch(cond_combo.currentText(), movie_combo.currentText())

        cond_combo.currentTextChanged.connect(on_condition)
        movie_combo.activated.connect(on_movie)
        try:
            self.viewer.window.add_dock_widget(
                panel, area="bottom", name="condition / movie")
        except Exception:
            pass

    def _sync_selector(self):                     # pragma: no cover - napari
        if not self._selector:
            return
        cond_combo, movie_combo = self._selector
        for combo, value in ((cond_combo, self.condition), (movie_combo, self.movie)):
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)


def _try_hmm(canonical, params, n_states):
    """Fit an HMM for state colouring; None if it will not fit (not fatal)."""
    from .hmm import HMMError, fit_hmm, step_sequences

    try:
        return fit_hmm(step_sequences(canonical), n_states=n_states,
                       tau_s=params.tau_s, n_starts=6)
    except HMMError:
        return None


def _dock(viewer, bundle, canonical, meshes, params, movie, fit):
    """Dock the inspector + track list; return the two widgets so a later switch
    can remove and rebuild them in place (no new window)."""
    from .qc import track_summary
    from .widgets import dock_inspector, dock_track_list

    outlines_um = {
        f"{m.movie}:{m.cell_id}": m.polygon * params.um_per_pixel
        for values in meshes.values() for m in values
    }
    inspector = dock_inspector(
        viewer, bundle, canonical, hmm_fit=fit, outlines_um=outlines_um,
        confinement_radius_um=0.1,
    )
    movie_tracks = canonical[canonical["movie"] == movie]
    track_list = dock_track_list(
        viewer, track_summary(movie_tracks, hmm_fit=fit),
        on_select=inspector.show_uid,
    )
    return inspector, track_list
