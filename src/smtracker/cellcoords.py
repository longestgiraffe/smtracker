"""Cell coordinates: project positions onto the cell's own axes.

The gateway to everything spatial. A position in camera coordinates says nothing
biological: "x = 63 px" is meaningless, while "80 % of the way to a pole, hard
against the membrane" is a claim. This module converts one into the other, and
because it normalises by each cell's own length and width, positions from cells
of different sizes can finally be **pooled**.

Coordinates
-----------
``L`` (arc length along the cell's centerline, in µm from one pole.
``D``) signed perpendicular distance from the centerline, in µm. The sign
distinguishes the two sides, which matters for asymmetry but is arbitrary in
orientation, so most analyses use ``|D|``.

Normalised (the part that lets cells be pooled):

``l`` (``L / cell_length``, in ``[0, 1]``; 0 and 1 are the poles, 0.5 midcell.
``d``) ``D / half_width(L)``, in ``[-1, 1]``; ±1 is the membrane, 0 the axis.

Normalising ``d`` by the **local** half-width (not a global one) matters: cells
taper toward the poles, so a fixed divisor would push polar molecules
artificially toward the membrane and manufacture polar-membrane localisation.

Method
------
Follows the legacy ``project2Mesh.m`` exactly:

- the centerline is the midpoint of each Oufti rib,
  ``(x_left + x_right)/2``;
- the first and last centerline points are extrapolated far outward so that
  molecules slightly beyond the poles still project onto the axis rather than
  being clamped;
- for each point, the nearest centerline **segment** is found, ``L`` is the
  cumulative arc length to that segment plus the distance along it, and ``D`` is
  the signed perpendicular distance (sign from the cross product).

Units
-----
Outlines are in **pixels** and the canonical table is in **µm**, so the mesh is
scaled to µm here using the table's own ``AcquisitionParams``. Nothing is
converted twice, and nothing is guessed.

Fail loud: an outline with no centerline ribs, a degenerate mesh, or a canonical
table without acquisition parameters all raise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS
from .cellmesh import CellMesh

__all__ = [
    "CellCoordError",
    "CellGeometry",
    "CELL_COORD_COLUMNS",
    "centerline",
    "cell_geometry",
    "project_to_mesh",
    "project_tracks",
]

CELL_COORD_COLUMNS = ["L_um", "D_um", "l_norm", "d_norm"]

#: How far the terminal centerline points are extrapolated, as a multiple of the
#: terminal segment length. Matches the legacy ``f = 1000``.
_EXTRAPOLATION = 1000.0


class CellCoordError(RuntimeError):
    """Raised when positions cannot be projected onto a cell."""


@dataclass(frozen=True)
class CellGeometry:
    """A cell's own measurements, derived from its mesh."""

    cell_id: str
    length_um: float
    max_width_um: float
    area_um2: float
    n_ribs: int

    @property
    def aspect_ratio(self) -> float:
        return self.length_um / self.max_width_um if self.max_width_um > 0 else np.nan

    def summary(self) -> str:
        return (
            f"cell {self.cell_id}: length {self.length_um:.2f} µm, "
            f"max width {self.max_width_um:.2f} µm, area {self.area_um2:.2f} µm²"
        )


def centerline(mesh: np.ndarray) -> np.ndarray:
    """Centerline points from Oufti ribs: the midpoint of each cross-section."""
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 2 or mesh.shape[1] != 4 or len(mesh) < 2:
        raise CellCoordError(
            f"mesh must be (N>=2, 4) [x_left, y_left, x_right, y_right], got {mesh.shape}"
        )
    return np.column_stack(
        [(mesh[:, 0] + mesh[:, 2]) / 2.0, (mesh[:, 1] + mesh[:, 3]) / 2.0]
    )


def _rib_half_widths(mesh: np.ndarray) -> np.ndarray:
    """Half the left-to-right distance at each rib."""
    mesh = np.asarray(mesh, dtype=float)
    return 0.5 * np.hypot(mesh[:, 0] - mesh[:, 2], mesh[:, 1] - mesh[:, 3])


def cell_geometry(cell: CellMesh, *, um_per_pixel: float) -> CellGeometry:
    """Length, width and area of one cell, in µm."""
    if not cell.has_centerline:
        raise CellCoordError(
            f"cell {cell.cell_id} has no mesh ribs, so it has no centerline. "
            "Cell coordinates need Oufti's 'mesh' field, not just the contour."
        )
    if um_per_pixel <= 0:
        raise CellCoordError(f"um_per_pixel must be > 0, got {um_per_pixel}")

    axis = centerline(cell.mesh) * um_per_pixel
    length = float(np.sum(np.hypot(*np.diff(axis, axis=0).T)))
    widths = _rib_half_widths(cell.mesh) * 2.0 * um_per_pixel

    polygon = np.asarray(cell.polygon, dtype=float) * um_per_pixel
    x, y = polygon[:, 0], polygon[:, 1]
    area = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    return CellGeometry(
        cell_id=cell.cell_id,
        length_um=length,
        max_width_um=float(widths.max()),
        area_um2=area,
        n_ribs=len(cell.mesh),
    )


def project_to_mesh(x, y, mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto a cell centerline. Returns ``(L, D)`` in mesh units.

    Faithful to the legacy ``project2Mesh.m``, including the extrapolation of the
    terminal centerline points so molecules beyond a pole project onto the axis
    instead of piling up at the last rib.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise CellCoordError(f"x and y must match: {x.shape} vs {y.shape}")

    axis = centerline(mesh)
    if len(axis) < 3:
        raise CellCoordError("centerline needs at least 3 points to project onto")

    # Segment lengths BEFORE extrapolation define the arc-length scale, exactly
    # as the legacy computes stplng from the un-extended curve.
    step = 0.5 * np.hypot(
        np.diff(mesh[:, 0] + mesh[:, 2]), np.diff(mesh[:, 1] + mesh[:, 3])
    )

    extended = axis.copy()
    delta = _EXTRAPOLATION * float(np.hypot(*(axis[1] - axis[0])))
    extended[0] = axis[0] * (_EXTRAPOLATION + 1) - axis[1] * _EXTRAPOLATION
    extended[-1] = axis[-1] * (_EXTRAPOLATION + 1) - axis[-2] * _EXTRAPOLATION
    arc = np.concatenate([[-delta], np.cumsum(step)])

    p1 = extended[:-1]
    p2 = extended[1:]
    segment = p2 - p1
    length_squared = np.sum(segment**2, axis=1)
    length_squared[length_squared == 0] = 1.0

    points = np.column_stack([x, y])
    # t = projection parameter of each point onto each segment, clamped to it.
    diff = points[:, None, :] - p1[None, :, :]
    t = np.clip(
        np.sum(diff * segment[None, :, :], axis=2) / length_squared[None, :], 0.0, 1.0
    )
    foot = p1[None, :, :] + t[:, :, None] * segment[None, :, :]
    distance_squared = np.sum((foot - points[:, None, :]) ** 2, axis=2)

    nearest = np.argmin(distance_squared, axis=1)
    rows = np.arange(len(points))

    along = np.hypot(
        foot[rows, nearest, 0] - extended[nearest, 0],
        foot[rows, nearest, 1] - extended[nearest, 1],
    )
    L = arc[nearest] + along

    # Sign from which side of the segment the point lies on.
    side = (points[:, 0] - extended[nearest, 0]) * (
        extended[nearest + 1, 1] - extended[nearest, 1]
    ) - (points[:, 1] - extended[nearest, 1]) * (
        extended[nearest + 1, 0] - extended[nearest, 0]
    )
    orientation = np.where(side < 0, 1.0, -1.0)
    D = np.sqrt(distance_squared[rows, nearest]) * orientation
    return L, D


def project_tracks(
    canonical: pd.DataFrame,
    meshes: dict[str, list[CellMesh]] | list[CellMesh],
    *,
    require_centerline: bool = True,
) -> pd.DataFrame:
    """Add cell coordinates to every localization of a canonical table.

    Each localization is projected onto **its own cell's** centerline (the
    ``cell`` column, assigned by ``smtracker.cellmesh.assign_cells``).

    Returns
    -------
    pandas.DataFrame
        The canonical table plus ``L_um``, ``D_um``, ``l_norm``, ``d_norm``.
        ``.attrs["geometry"]`` holds a ``CellGeometry`` per cell.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise CellCoordError(f"not a canonical table (missing {missing})")
    params = canonical.attrs.get("params")
    if params is None:
        raise CellCoordError(
            "canonical table carries no AcquisitionParams in .attrs['params']; "
            "the mesh is in pixels and the table in µm, so the scale is required."
        )
    if canonical["cell"].isna().any():
        raise CellCoordError(
            "some localizations have no 'cell'; run cellmesh.assign_cells first. "
            "A position cannot be expressed in a cell's coordinates without "
            "knowing which cell it is in."
        )

    lookup = _mesh_lookup(meshes)
    um_per_pixel = float(params.um_per_pixel)

    out = canonical.copy()
    for column in CELL_COORD_COLUMNS:
        out[column] = np.nan

    geometry: dict[str, CellGeometry] = {}
    for cell_key, frame in out.groupby("cell", observed=True):
        cell = lookup.get(str(cell_key))
        if cell is None:
            raise CellCoordError(
                f"no outline for cell {cell_key!r}; the meshes and the assigned "
                "cells disagree."
            )
        if not cell.has_centerline:
            if require_centerline:
                raise CellCoordError(
                    f"cell {cell_key} has no mesh ribs, so no centerline. Cell "
                    "coordinates need Oufti's 'mesh' field."
                )
            continue

        mesh_um = np.asarray(cell.mesh, dtype=float) * um_per_pixel
        L, D = project_to_mesh(
            frame["x_um"].to_numpy(), frame["y_um"].to_numpy(), mesh_um
        )
        info = cell_geometry(cell, um_per_pixel=um_per_pixel)
        geometry[str(cell_key)] = info

        half_width = _half_width_at(mesh_um, L)
        with np.errstate(divide="ignore", invalid="ignore"):
            l_norm = L / info.length_um if info.length_um > 0 else np.full_like(L, np.nan)
            d_norm = np.where(half_width > 0, D / half_width, np.nan)

        out.loc[frame.index, "L_um"] = L
        out.loc[frame.index, "D_um"] = D
        out.loc[frame.index, "l_norm"] = np.clip(l_norm, -0.5, 1.5)
        out.loc[frame.index, "d_norm"] = d_norm

    out.attrs.update(canonical.attrs)
    out.attrs["geometry"] = geometry
    out.attrs["n_cells_projected"] = len(geometry)
    return out


def _mesh_lookup(meshes) -> dict[str, CellMesh]:
    """Index outlines by the ``"<movie>:<cell_id>"`` key ``assign_cells`` writes."""
    if isinstance(meshes, dict):
        flat = [mesh for group in meshes.values() for mesh in group]
    else:
        flat = list(meshes)
    if not flat:
        raise CellCoordError("no cell outlines supplied")
    return {f"{mesh.movie}:{mesh.cell_id}": mesh for mesh in flat}


def _half_width_at(mesh_um: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Local half-width at each arc-length position, by interpolation.

    Cells taper toward the poles, so ``d`` must be normalised by the width
    *where the molecule is*. A global width would push polar molecules toward
    the membrane and invent polar-membrane localisation.
    """
    axis = centerline(mesh_um)
    segment_lengths = np.hypot(*np.diff(axis, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    half_widths = _rib_half_widths(mesh_um)
    return np.interp(np.clip(L, arc[0], arc[-1]), arc, half_widths)
