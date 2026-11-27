"""``smtracker.spatial``: cell-frame export + full-cell map.

Two jobs and nothing else (all statistics happen downstream in R):

1. Export tidy tables (``localisations`` / ``steps`` / ``tracks`` / ``cells``) of
   every localisation, step, track and cell in a normalised spherocylinder cell
   frame, plus matched NULL tables with the identical schema so a downstream
   script can ``rbind`` and get its baseline by grouping.
2. Draw a mirrored full-cell map of tracks or localisation density.

No hypothesis tests, p-values, HMM state inference or nucleoid analysis live here.
The schema in ``SCHEMA.md`` is the contract; column names, units and semantics are
fixed. Submodules: ``geometry`` (spherocylinder), ``coords`` (cell frame +
``proj_weight``), ``populations`` (per-track D → labels), ``nulls`` (matched
baselines), ``maps`` (figures + ``map_bins``), ``io`` (read/validate/export).
"""
from __future__ import annotations

from . import coords, geometry, io, maps, nulls, populations

__all__ = ["coords", "geometry", "io", "maps", "nulls", "populations"]
