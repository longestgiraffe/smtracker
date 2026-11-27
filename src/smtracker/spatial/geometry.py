"""Spherocylinder cell geometry for ``smtracker.spatial``.

A cell is a spherocylinder: a cylinder of radius ``R = width_um / 2`` capped by two
hemispheres, total tip-to-tip length ``length_um`` along the long axis. The cap
centres sit at ``±(length_um/2 − R)``; the flat cylinder body spans ``|u| ≤ L/2 − R``
and the caps span ``L/2 − R < |u| ≤ L/2``.

All functions are PURE and VECTORISED over numpy arrays; ``u`` and ``v`` are signed
positions in the cell frame (µm), ``u`` along the long axis with ``u = 0`` at
midcell, ``v`` across it with ``v = 0`` on the axis. Every length is µm.

The load-bearing quantity is the CHORD through the shape at a projected point
``(u, v)``. A cell is a 3-D volume imaged in 2-D, so a molecule distributed
uniformly in the volume does NOT project to a flat 2-D sheet: the projected
density is proportional to the chord length along the line of sight. ``coords.py``
divides the chord by ``mean_chord = volume / projected_area`` so the resulting
``proj_weight`` averages to 1 over the projected silhouette (and reproduces the
1.26 / 1.22 / 1.10 / 0.91 / 0.52 radial gradient in equal-width ``r_norm`` bins).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "GeometryError",
    "LENGTH_RANGE_UM",
    "WIDTH_RANGE_UM",
    "radius_um",
    "cap_junction_um",
    "half_width",
    "chord",
    "in_cap",
    "volume_um3",
    "projected_area_um2",
    "mean_chord_um",
    "check_geometry",
]


class GeometryError(ValueError):
    """Raised when a cell's geometry is missing, non-finite or unphysical."""


#: Physically plausible cell length / width (µm). Used by validation, not by the
#: math: a cell outside these is a data error, not a geometry to model.
LENGTH_RANGE_UM = (0.5, 10.0)
WIDTH_RANGE_UM = (0.3, 2.0)


def radius_um(width_um):
    """Cylinder / cap radius ``R = width_um / 2`` (µm)."""
    return np.asarray(width_um, dtype=float) / 2.0


def cap_junction_um(length_um, width_um):
    """Signed axial position ``a = L/2 − R`` (µm) where the cylinder meets the cap.

    ``|u| ≤ a`` is the flat cylinder body; ``a < |u| ≤ L/2`` is a hemispherical cap.
    """
    length_um = np.asarray(length_um, dtype=float)
    return length_um / 2.0 - radius_um(width_um)


def half_width(u, length_um, width_um):
    """Local half-width ``h(u)`` (µm) of the spherocylinder at axial position ``u``.

    ``h(u) = R`` on the cylinder body (``|u| ≤ L/2 − R``) and
    ``h(u) = sqrt(R² − (|u| − (L/2 − R))²)`` in the caps, → 0 exactly at the pole
    ``|u| = L/2`` and → ``R`` exactly at the cap junction (continuous). Vectorised.

    Parameters
    ----------
    u : array-like
        Signed axial position(s), µm.
    length_um, width_um : float or array-like
        Cell geometry, µm.

    Returns
    -------
    numpy.ndarray
        Half-width µm, same broadcast shape as the inputs; 0 outside the pole.
    """
    u = np.asarray(u, dtype=float)
    R = radius_um(width_um)
    a = cap_junction_um(length_um, width_um)
    au = np.abs(u)
    # cap term: R² − (|u| − a)², clipped at 0 so it is 0 at/beyond the pole
    cap = np.sqrt(np.clip(R ** 2 - (au - a) ** 2, 0.0, None))
    return np.where(au <= a, np.broadcast_to(R, au.shape).astype(float), cap)


def chord(u, v, length_um, width_um):
    """Chord length (µm) through the volume along the line of sight at ``(u, v)``.

    ``chord = 2·sqrt(max(h(u)² − v², 0))``: the depth of material a projected point
    sees. 0 outside the silhouette. This is exactly the projected density (up to a
    constant) of a uniform-in-3-D filling, which is why it defines ``proj_weight``.
    """
    v = np.asarray(v, dtype=float)
    h = half_width(u, length_um, width_um)
    return 2.0 * np.sqrt(np.clip(h ** 2 - v ** 2, 0.0, None))


def in_cap(u, length_um, width_um):
    """Boolean: is ``u`` inside a hemispherical cap (``|u| > L/2 − R``)? Vectorised."""
    u = np.asarray(u, dtype=float)
    return np.abs(u) > cap_junction_um(length_um, width_um)


def volume_um3(length_um, width_um):
    """Spherocylinder volume (µm³): cylinder ``πR²(L−2R)`` + one sphere ``4/3·πR³``.

    Also equals ``π·∫ h(u)² du`` (a stack of disks of radius ``h``), which is why
    ``∫∫ chord dA = volume`` and ``mean_chord = volume / projected_area``.
    """
    R = radius_um(width_um)
    L = np.asarray(length_um, dtype=float)
    cyl = np.clip(L - 2.0 * R, 0.0, None)                 # body length (≥0)
    return np.pi * R ** 2 * cyl + (4.0 / 3.0) * np.pi * R ** 3


def projected_area_um2(length_um, width_um):
    """Silhouette area (µm²): cylinder rectangle ``2R(L−2R)`` + two half-disks ``πR²``.

    Equals ``∫ 2·h(u) du``: the 2-D footprint the map bins over.
    """
    R = radius_um(width_um)
    L = np.asarray(length_um, dtype=float)
    cyl = np.clip(L - 2.0 * R, 0.0, None)
    return 2.0 * R * cyl + np.pi * R ** 2


def mean_chord_um(length_um, width_um):
    """Chord averaged over the projected area = ``volume / projected_area`` (µm).

    ``proj_weight = chord / mean_chord`` then averages to 1 over the silhouette.
    """
    return volume_um3(length_um, width_um) / projected_area_um2(length_um, width_um)


def check_geometry(length_um, width_um, *, cell_id=None):
    """Raise :class:`GeometryError` if a single cell's geometry is unusable.

    Checks finiteness, positivity, the physical ranges (``LENGTH_RANGE_UM`` /
    ``WIDTH_RANGE_UM``) and ``length_um ≥ width_um`` (a spherocylinder shorter than
    it is wide is impossible). ``cell_id`` is only for the message.
    """
    tag = "" if cell_id is None else f"cell {cell_id!r}: "
    for name, val in (("length_um", length_um), ("width_um", width_um)):
        if val is None or not np.isfinite(val):
            raise GeometryError(f"{tag}{name} is missing/non-finite ({val!r})")
        if val <= 0:
            raise GeometryError(f"{tag}{name} must be > 0, got {val}")
    lo, hi = LENGTH_RANGE_UM
    if not lo <= length_um <= hi:
        raise GeometryError(
            f"{tag}length_um {length_um} µm outside plausible {lo}–{hi} µm")
    lo, hi = WIDTH_RANGE_UM
    if not lo <= width_um <= hi:
        raise GeometryError(
            f"{tag}width_um {width_um} µm outside plausible {lo}–{hi} µm")
    if length_um < width_um:
        raise GeometryError(
            f"{tag}length_um {length_um} < width_um {width_um}; a spherocylinder "
            "cannot be shorter than it is wide")
