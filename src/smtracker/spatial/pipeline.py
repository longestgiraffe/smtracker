"""One call: a u-track project condition → the spatial localisation table.

Why this exists. `load_group` lived in `scripts/msd_aj3_shell_sweep.py` and was imported
from there by six other scripts. It is the whole pipeline: discover movies, map
replicate days, load the project, build the spherocylinder cell frame, erode, filter,
build the tables, and it is exactly what a GUI button needs to call. A button cannot
call a script (ON2).

The decision this packaging required, taken rather than left parked. `load_group`
hard-coded five constants from `msd_batch` (`ROOT`, `SIGMA_NM`, `EROSION_PX`,
`D_THRESHOLD`, `MIN_LOC`). Which are parameters and which are project config?

    All five are parameters, and none of them has a default here.

Every one changes the answer, and three of them are choices this project has explicitly
refused to bury: sigma is unmeasured with a 30-76 nm range, the D threshold sits on
APPARENT D so what it selects depends on sigma, and the erosion is a fitted quantity
whose sweep is a documented deliverable. A default would make those choices invisible at
the call site, which is the failure mode the whole robustness inventory exists to
prevent. The caller states them; `SpatialConfig` carries them together so they travel as
a set and can be recorded in provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..canonical import to_canonical
from ..cellmesh import assign_cells, read_project_cell_meshes
from ..grouping import load_utrack_project
from ..utrack_meta import acquisition_params_from_project
from .adapters import localisation_table, spherocylinder_cells
from .io import build_tables

__all__ = ["SpatialConfig", "spatial_localisations"]


@dataclass(frozen=True)
class SpatialConfig:
    """The five choices that decide what the spatial tables contain.

    Frozen and passed as a unit so they can be recorded in provenance together: a
    number quoted without them is not reproducible. None has a default:

    ``sigma_nm``
        Localisation precision. **Unmeasured for this project** (30-76 nm standing
        range, provenance ``theoretical``). Enters the table's noise columns.
    ``erosion_px``
        Per-side shrink applied to each cell before the frame is built. A FITTED
        quantity, swept in the robustness inventory, never a silent default.
    ``d_threshold``
        The low-D class boundary, on **apparent** D. What true D it corresponds to
        depends on ``sigma_nm``: 0.15 is a true D of 0.138 under the pixelation floor
        and 0.028 under the PSF floor, a fivefold difference in class contents.
    ``min_localisations``
        Track-length restriction. u-track already pre-filters at 5, so this is usually
        a no-op on u-track data and load-bearing on TrackMate data.
    ``require_all_inside``
        The cell filter. ``True`` is the legacy "every localisation inside the outline"
        rule; it discards WHOLE tracks and is neither radially neutral nor D-neutral
        (ON3a). ``False`` is majority-inside.
    """

    sigma_nm: float
    erosion_px: float
    d_threshold: float
    min_localisations: int
    require_all_inside: bool


def spatial_localisations(condition_root, condition: str, *, enzyme: str,
                          config: SpatialConfig, dataset: str,
                          day_map=None, with_params: bool = False):
    """Load one condition of a u-track project and return its localisation table.

    ``condition_root`` is the directory holding the movie folders for ONE condition.
    ``enzyme`` is required and has no default: the predecessor of this function
    inherited a hard-coded ``"protein D"`` for every protein (AP2).

    ``day_map`` is an optional ``DayMapping``; when omitted the project's own resolution
    is used. Returns the ``localisations`` table with a ``day`` column joined on.

    Cells outside the spherocylinder size range, and cells whose eroded width or length
    goes non-positive, are dropped, the caller sees a shorter table, so count it if the
    loss matters.
    """
    from ..project import load_project, with_localization_sigma_nm

    root = Path(condition_root)
    conditions = (condition,)
    kwargs = {"day_maps": {condition: day_map}} if day_map is not None else {}
    # config.require_all_inside REACHES THE FILTER HERE. It was declared, documented,
    # CLI-exposed and recorded in provenance while being read by nothing, so a user who
    # set it got the default silently and a provenance record that said otherwise.
    project = load_project(str(root.parent), conditions=conditions,
                           require_all_inside=config.require_all_inside, **kwargs)
    project = with_localization_sigma_nm(project, config.sigma_nm, "configured")

    params = project.params
    um_px = params.um_per_pixel
    cells = spherocylinder_cells(project.meshes, um_px).copy()
    cells["width_um"] -= 2 * config.erosion_px * um_px
    cells["length_um"] -= 2 * config.erosion_px * um_px
    cells = cells[(cells["width_um"] > 0.3) & (cells["length_um"] > cells["width_um"])]

    canon = project.condition(condition)
    canon = canon[canon["cell"].notna()]
    locs = localisation_table(canon, enzyme=enzyme, condition=condition)
    locs = locs[locs["cell_id"].isin(set(cells["cell_id"]))]

    table = build_tables(
        locs, cells, dataset=dataset,
        sigma_um=config.sigma_nm / 1000.0,
        tau_s=float(canon["tau_s"].iloc[0]), angle_units="rad",
        min_localisations=config.min_localisations,
        edges=(0.0, config.d_threshold, np.inf))["localisations"]

    day_of = canon.groupby("uid")["day"].first()
    out = table.assign(day=table["track_id"].map(day_of))
    # with_params=True hands back the AcquisitionParams this function already read, so
    # a caller needing the sigma stamp (deposit) does not re-read the project.
    return (out, params) if with_params else out
