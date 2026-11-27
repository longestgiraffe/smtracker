"""Canonical table + cell meshes → the two tables ``spatial.io.build_tables`` consumes.

Why this module exists. These two functions were private helpers
(``_spherocylinder_cells``, ``_loc_table``) inside ``scripts/spatial_demo_map.py``, and
**28 scripts imported them from there**, the entire spatial arc's data adapter lived in
a demo script, under underscore-private names. A GUI button cannot call
``scripts/spatial_demo_map._loc_table``, which is the concrete form of "the analyses are
not packaged". Packaged here unchanged in behaviour (ON2).

A defect the move exposed, recorded because it is a live landmine rather than history:
``_loc_table`` hard-coded ``"enzyme": "protein D"``. Every caller (including the ten-group
MSD batch over five different proteins) therefore labelled its localisations ``protein D``.
It has not corrupted any published number, because each group is built and analysed in
its own call and ``dataset`` distinguishes them, but ``enzyme`` IS one of
``spatial.maps._GROUP_KEYS``: concatenating two enzymes' tables would silently merge
them under one label. :func:`localisation_table` therefore makes ``enzyme`` a REQUIRED
keyword with no default, so the mistake cannot be made again by omission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..cellcoords import CellCoordError, cell_geometry
from . import geometry as sg

__all__ = ["spherocylinder_cells", "localisation_table"]

#: Columns ``build_tables`` requires of the localisation table.
LOCALISATION_COLUMNS = ("track_id", "frame", "x_um", "y_um", "cell_id",
                        "enzyme", "condition", "replicate", "session")
#: Columns ``build_tables`` requires of the cell table.
CELL_COLUMNS = ("cell_id", "centroid_x", "centroid_y", "angle", "length_um",
                "width_um")


def spherocylinder_cells(meshes: dict, um_per_pixel: float) -> pd.DataFrame:
    """Oufti meshes → parametric spherocylinder parameters, one row per usable cell.

    ``meshes`` is ``{movie: [mesh, ...]}`` as carried by a loaded project. ``cell_id``
    is ``'<movie>:<cell_id>'``, matching the canonical table's ``cell`` column, so the
    two tables join without a rename.

    The long-axis ``angle`` comes from a PCA of the outline polygon, NOT from the mesh
    ribs: this is the parametric frame, deliberately independent of the rib
    parameterisation (see ``spatial`` vs the legacy ``cellcoords`` path).

    Cells outside :data:`spatial.geometry.LENGTH_RANGE_UM` /
    :data:`WIDTH_RANGE_UM`, or with ``length < width``, are DROPPED: a spherocylinder
    is not a model of them. Meshes that ``cell_geometry`` rejects are skipped. The
    caller sees the drop as a shorter table; count it if the loss matters.
    """
    rows = []
    for movie, mesh_list in meshes.items():
        for m in mesh_list:
            try:
                geo = cell_geometry(m, um_per_pixel=um_per_pixel)
            except CellCoordError:
                continue
            poly = np.asarray(m.polygon, dtype=float) * um_per_pixel      # µm
            centroid = poly.mean(axis=0)
            cov = np.cov((poly - centroid).T)
            w, vec = np.linalg.eigh(cov)
            major = vec[:, int(np.argmax(w))]
            angle = float(np.arctan2(major[1], major[0]))
            L, W = float(geo.length_um), float(geo.max_width_um)
            if not (sg.LENGTH_RANGE_UM[0] <= L <= sg.LENGTH_RANGE_UM[1]):
                continue
            if not (sg.WIDTH_RANGE_UM[0] <= W <= sg.WIDTH_RANGE_UM[1]) or L < W:
                continue
            rows.append({"cell_id": f"{movie}:{m.cell_id}",
                         "centroid_x": centroid[0], "centroid_y": centroid[1],
                         "angle": angle, "length_um": L, "width_um": W})
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


def localisation_table(canonical: pd.DataFrame, *, enzyme: str,
                       condition: str) -> pd.DataFrame:
    """Canonical table → the localisation table ``build_tables`` consumes.

    ``enzyme`` is REQUIRED and has no default: the predecessor of this function
    hard-coded ``"protein D"`` and every one of its 28 callers inherited that label
    regardless of which protein it was analysing. ``enzyme`` is a grouping key
    downstream, so a wrong constant there merges distinct proteins silently.

    ``track_id`` comes from ``uid``, which is unique ACROSS movies: ``track_id`` alone
    is not, and joining on it would collide tracks from different movies.
    """
    if not enzyme or not str(enzyme).strip():
        raise ValueError("enzyme must be a non-empty label: it is a grouping key, "
                         "and a constant there silently merges different proteins")
    missing = [c for c in ("uid", "frame", "x_um", "y_um", "cell", "movie", "day")
               if c not in canonical.columns]
    if missing:
        raise ValueError(f"not a canonical table (missing {missing})")
    return pd.DataFrame({
        "track_id": canonical["uid"].to_numpy(),
        "frame": canonical["frame"].to_numpy(),
        "x_um": canonical["x_um"].to_numpy(),
        "y_um": canonical["y_um"].to_numpy(),
        "cell_id": canonical["cell"].to_numpy(),
        "enzyme": str(enzyme),
        "condition": str(condition),
        "replicate": canonical["movie"].to_numpy(),
        "session": canonical["day"].to_numpy(),
    })
