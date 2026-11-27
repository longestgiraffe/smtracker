"""Cell-frame coordinates + ``proj_weight`` for ``smtracker.spatial``.

``to_cell_frame`` rotates every localisation into its cell's spherocylinder frame
(origin = centroid, ``u`` along the long axis) and attaches the geometry-derived
columns of the ``localisations`` schema. It is fully VECTORISED: one merge, then
numpy, no per-localisation Python loop.

The sigma-dependent (``r_norm_sigma``), per-track (``track_D_um2s``,
``population``, ``n_loc_track``) and pass-through/``source`` columns are added by
the table builders in :mod:`smtracker.spatial.io`: they need inputs this function
does not take. This function owns the geometry, and only the geometry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as _g

__all__ = ["CoordError", "ANGLE_UNITS", "to_cell_frame", "cell_to_lab"]


class CoordError(ValueError):
    """Raised when the cell frame cannot be formed (bad angle units, unknown cell)."""


ANGLE_UNITS = ("rad", "deg")
_EPS = 1e-12


def _angle_rad(angle, angle_units: str) -> np.ndarray:
    if angle_units not in ANGLE_UNITS:
        raise CoordError(
            f"angle_units must be one of {ANGLE_UNITS}, got {angle_units!r}: state degrees or "
            "radians; the unit is never inferred from the value range")
    a = np.asarray(angle, dtype=float)
    return np.deg2rad(a) if angle_units == "deg" else a


def cell_to_lab(u, v, centroid_x, centroid_y, angle, angle_units="rad"):
    """Forward map cell-frame ``(u, v)`` → lab-frame ``(x_um, y_um)``. Vectorised.

    ``x = cx + u·cosθ − v·sinθ`` ; ``y = cy + u·sinθ + v·cosθ``. The exact inverse
    of :func:`to_cell_frame`'s rotation, used to plant test points and to place
    null tracks.
    """
    theta = _angle_rad(angle, angle_units)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    x = np.asarray(centroid_x, dtype=float) + u * c - v * s
    y = np.asarray(centroid_y, dtype=float) + u * s + v * c
    return x, y


def to_cell_frame(locs: pd.DataFrame, cells: pd.DataFrame,
                  angle_units: str = "rad") -> pd.DataFrame:
    """Rotate localisations into their cell frames + attach geometry columns.

    Parameters
    ----------
    locs : pandas.DataFrame
        Localisation table with at least ``cell_id, x_um, y_um`` (µm). All other
        columns are carried through unchanged.
    cells : pandas.DataFrame
        Canonical cell table from :func:`smtracker.spatial.io.read_cell_table`
        (``cell_id, centroid_x, centroid_y, angle, length_um, width_um``, µm).
    angle_units : {'rad', 'deg'}
        Units of the cell table's ``angle`` column. The CALLER states it; it is
        never guessed. Default ``'rad'``.

    Returns
    -------
    pandas.DataFrame
        ``locs`` plus the geometry columns of the ``localisations`` schema
        (all µm or dimensionless): ``u_um, v_um`` (signed cell frame),
        ``l_signed, l_abs`` (u / (L/2)), ``halfwidth_um`` = h(u),
        ``r_norm`` = |v|/h clipped to 1, ``d_pole_um`` = L/2 − |u| (axial,
        Cartesian), ``d_membrane_um`` = h − |v| (in-plane, to the projected
        outline), ``in_cap``, ``cell_length_um, cell_width_um``, and
        ``proj_weight`` = chord(u,v) / mean_chord.

    Notes
    -----
    ``r_norm`` is UNRELIABLE where ``halfwidth_um`` is small (deep in the caps): a
    fixed localisation precision is a large fraction of a small half-width, at a
    100 nm half-width, a 31 nm precision is ``r_norm_sigma ≈ 0.3`` (that column is
    added by the localisations builder, which is given ``sigma_um``).
    """
    need = {"cell_id", "x_um", "y_um"}
    missing = need - set(locs.columns)
    if missing:
        raise CoordError(f"localisation table missing {sorted(missing)}")
    geo_cols = ["cell_id", "centroid_x", "centroid_y", "angle", "length_um", "width_um"]
    unknown = set(locs["cell_id"]) - set(cells["cell_id"])
    if unknown:
        raise CoordError(
            f"{len(unknown)} localisation cell_id(s) are not in the cell table: "
            f"{sorted(unknown)[:20]}")

    out = locs.merge(cells[geo_cols], on="cell_id", how="left", validate="many_to_one")
    theta = _angle_rad(out["angle"].to_numpy(), angle_units)
    dx = out["x_um"].to_numpy() - out["centroid_x"].to_numpy()
    dy = out["y_um"].to_numpy() - out["centroid_y"].to_numpy()
    c, s = np.cos(theta), np.sin(theta)

    L = out["length_um"].to_numpy()
    W = out["width_um"].to_numpy()
    u = dx * c + dy * s                                   # along the long axis
    v = -dx * s + dy * c                                  # across it

    h = _g.half_width(u, L, W)
    r_norm = np.clip(np.abs(v) / np.maximum(h, _EPS), 0.0, 1.0)
    mean_chord = _g.mean_chord_um(L, W)

    out["u_um"] = u
    out["v_um"] = v
    out["l_signed"] = u / (L / 2.0)
    out["l_abs"] = np.abs(out["l_signed"].to_numpy())
    out["halfwidth_um"] = h
    out["r_norm"] = r_norm
    out["d_pole_um"] = L / 2.0 - np.abs(u)
    out["d_membrane_um"] = h - np.abs(v)
    out["in_cap"] = _g.in_cap(u, L, W)
    out["cell_length_um"] = L
    out["cell_width_um"] = W
    out["proj_weight"] = _g.chord(u, v, L, W) / mean_chord
    # drop the merge-in geometry helpers that are not schema columns
    return out.drop(columns=["centroid_x", "centroid_y", "angle", "length_um",
                             "width_um"])
