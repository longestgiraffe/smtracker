"""Per-track apparent D and population labels for ``smtracker.spatial``.

The per-track D is the EXISTING estimator, not a reimplementation:

    smtracker.appd.apparent_D(n_steps=4, mode='split')  averaged per track

i.e. the lag-1 mean-squared displacement over consecutive-frame 5-localisation
windows, ``mean(Δx² + Δy²) / (4·τ)`` per window, then the MEAN of a track's
windows: matching ``smtracker.maps.labels_from_appd``. **No 4σ² offset is
removed**, so ``track_D_um2s`` is APPARENT D = D_true + σ²/τ and the ``edges`` live
on the apparent scale. The companion ``track_D_offset_removed_um2s`` (added by the
table builder) subtracts σ²/τ and is NOT clipped at zero, ~7 % of genuinely-low-D
tracks come out negative at 4 steps and a silent clip would hide that.

Populations are named by their D range (``D<0.15``, ``D>=0.15``, …) so the label is
self-documenting in a CSV and a legend. A per-track threshold answers "where are
they", not "how many are there": it does NOT recover the HMM state occupancy and
``population_report`` says so, because a track whose mean-squared displacement is
dominated by its high-D frames reads high-D even if it dwells low-D mid-lifetime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "PopulationError",
    "APPARENT_D_QUALIFIED_NAME",
    "APPARENT_D_FORMULA",
    "per_track_apparent_D",
    "population_labels",
    "assign_populations",
    "population_report",
]


class PopulationError(ValueError):
    """Raised when D cannot be estimated or ``edges`` are malformed."""


#: For the provenance stamp and the report, the estimator, on the record.
APPARENT_D_QUALIFIED_NAME = "smtracker.appd.apparent_D"
APPARENT_D_FORMULA = (
    "per 5-localisation window (n_steps=4): mean(Δx²+Δy²)/(4·τ) over consecutive "
    "frames; per-track D = mean of the track's windows (mode='split'). No 4σ² "
    "offset removed → apparent D = D_true + σ²/τ."
)


def per_track_apparent_D(locs: pd.DataFrame, *, tau_s: float,
                         n_steps: int = 4) -> pd.Series:
    """Per-track apparent D (µm²/s), indexed by ``track_id``.

    Calls :func:`smtracker.appd.apparent_D` on a canonical-shaped adapter of
    ``locs`` (``track_id → uid``, ``tau_s`` in ``attrs['params']``) and averages
    each track's sub-track windows. Tracks with no consecutive 5-localisation
    window (too short or fully gapped) are simply absent from the result.

    Parameters
    ----------
    locs : pandas.DataFrame
        Long-format localisations with ``track_id, frame, x_um, y_um`` (+ the
        pass-through group columns). µm, frame indices.
    tau_s : float
        Frame interval, seconds. Required, never guessed.
    n_steps : int
        Steps per window (legacy 4 = 5 localisations).

    Returns
    -------
    pandas.Series
        ``appd_um2_per_s`` per ``track_id``.
    """
    from ..appd import AppDError, apparent_D
    from ..canonical import AcquisitionParams

    if tau_s is None or tau_s <= 0:
        raise PopulationError(f"tau_s must be > 0, got {tau_s!r}")
    adapter = pd.DataFrame({
        "protein": locs.get("enzyme", "x"),
        "condition": locs.get("condition", "x"),
        "day": locs.get("session", "x"),
        "movie": locs.get("replicate", "x"),
        "cell": locs["cell_id"].to_numpy(),
        "uid": locs["track_id"].to_numpy(),
        "track_id": locs["track_id"].to_numpy(),
        "frame": locs["frame"].to_numpy(),
        "t_s": locs["frame"].to_numpy() * tau_s,
        "x_um": locs["x_um"].to_numpy(),
        "y_um": locs["y_um"].to_numpy(),
        "amp": 1.0,
    })
    adapter.attrs["params"] = AcquisitionParams(
        pixel_size_nm=106.0, frame_interval_ms=tau_s * 1000.0)
    try:
        sub = apparent_D(adapter, n_steps=n_steps, mode="split")
    except AppDError as exc:
        raise PopulationError(str(exc)) from exc
    per_track = sub.groupby("uid")["appd_um2_per_s"].mean()
    per_track.index.name = "track_id"
    return per_track.rename("appd_um2_per_s")


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "inf"
    return f"{x:g}"


def population_labels(edges) -> list[str]:
    """Human labels for the bins defined by ``edges``, named by their D range.

    ``(0, 0.15, inf)`` → ``['D<0.15', 'D>=0.15']``;
    ``(0, 0.1, 0.5, inf)`` → ``['D<0.1', '0.1<=D<0.5', 'D>=0.5']``.
    """
    e = list(edges)
    if len(e) < 2 or any(e[i] >= e[i + 1] for i in range(len(e) - 1)):
        raise PopulationError(f"edges must be strictly increasing, got {edges}")
    labels = []
    for i in range(len(e) - 1):
        lo, hi = e[i], e[i + 1]
        if lo <= 0 and i == 0:
            labels.append(f"D<{_fmt(hi)}")
        elif not np.isfinite(hi):
            labels.append(f"D>={_fmt(lo)}")
        else:
            labels.append(f"{_fmt(lo)}<=D<{_fmt(hi)}")
    return labels


def assign_populations(tracks: pd.DataFrame, edges=(0.0, 0.15, np.inf),
                       min_localisations: int = 5) -> pd.DataFrame:
    """Label each track by its ``D_app_um2s`` range; filter by ``min_localisations``.

    Parameters
    ----------
    tracks : pandas.DataFrame
        One row per track with ``D_app_um2s`` (µm²/s, apparent) and ``n_loc``.
    edges : sequence of float
        Population edges on the APPARENT scale (default ``(0, 0.15, inf)`` → two
        populations). User-supplied; defines the populations everywhere downstream.
    min_localisations : int
        Tracks with fewer localisations are dropped (their D is unreliable), not
        labelled.

    Returns
    -------
    pandas.DataFrame
        The surviving tracks with a ``population`` column (a range label) and a
        ``near_edge`` bool (D within a factor of 2 of an edge → its label is
        arbitrary). ``.attrs['edges']`` / ``['population_labels']`` are set.
    """
    if "D_app_um2s" not in tracks or "n_loc" not in tracks:
        raise PopulationError("tracks needs 'D_app_um2s' and 'n_loc' columns")
    labels = population_labels(edges)
    kept = tracks[(tracks["n_loc"] >= min_localisations)
                  & np.isfinite(tracks["D_app_um2s"])].copy()
    D = kept["D_app_um2s"].to_numpy()
    inner = np.asarray(edges[1:-1], dtype=float)
    idx = np.digitize(D, inner, right=False) if inner.size else np.zeros(len(D), int)
    kept["population"] = pd.Categorical([labels[i] for i in idx], categories=labels)

    finite_edges = np.array([e for e in edges if np.isfinite(e) and e > 0])
    near = np.zeros(len(D), dtype=bool)
    for e in finite_edges:
        near |= (D >= e / 2.0) & (D <= e * 2.0)
    kept["near_edge"] = near
    kept.attrs["edges"] = tuple(edges)
    kept.attrs["population_labels"] = labels
    kept.attrs["min_localisations"] = int(min_localisations)
    return kept


def population_report(tracks: pd.DataFrame, edges=(0.0, 0.15, np.inf)) -> dict:
    """Per-population summary for the run report (NOT a statistic).

    Returns ``{'per_population': DataFrame, 'note': str}``. Per population: n
    tracks, n localisations, median D, and the fraction of tracks within a factor
    of 2 of an edge (labels that are arbitrary). ``note`` states, loudly, that the
    labelled low-D **track** fraction is not the HMM state occupancy and is not
    expected to match it: high purity, low recall is the right trade for a
    "where are they" question and the wrong one for "how many are there".
    """
    labelled = assign_populations(tracks, edges=edges,
                                  min_localisations=tracks.attrs.get(
                                      "min_localisations", 5))
    rows = []
    for label, grp in labelled.groupby("population", observed=True):
        rows.append({
            "population": label,
            "n_tracks": int(len(grp)),
            "n_localisations": int(grp["n_loc"].sum()),
            "median_D_um2s": float(grp["D_app_um2s"].median()),
            "fraction_near_edge": float(grp["near_edge"].mean()) if len(grp) else 0.0,
        })
    note = (
        "The labelled low-D track fraction is not the HMM state occupancy: a track's "
        "mean-squared displacement is dominated by its high-D frames, so a track that "
        "switches reads high-D. It shows where slow molecules are, not how many there are.")
    return {"per_population": pd.DataFrame(rows), "note": note}
