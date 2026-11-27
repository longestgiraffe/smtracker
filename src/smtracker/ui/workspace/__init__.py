"""The analysis workspace: the sidebar + toolbar chrome around a method view.

Ported from `prototype/shadcn/{workspace,toolbar,compute}.py` (git history,
commit 09da35d — the folder itself was a standalone probe and was never part
of `smtracker`). Only the SIDEBAR and TOOLBAR are ported here; the figure grid
and the actual fit are NOT — every control in this package changes local UI
state only, and nothing here imports `smtracker.comparison` or runs a fit.
Wiring a real Compute press to `comparison_controller` is future work.

`sidebar` began with ONE section, "SQD"; it now holds ten, in three groups
(`sidebar.GROUPS`, 4 Nov 2026: Data tiles, Analyses rows, the Composer at the
foot). Adding a section is a line in `sidebar.SECTIONS` and its key in a group
of `sidebar.GROUPS`; a section in no group is refused when the module loads.

`toolbar` carries the split this project settled on after tracing the old
comparison browser's controls: a `ComputeMark` opens `compute.ComputePanel`,
holding only the parameters that would change the FIT (weighting, shared-D
scope, state count, uncertainty tier) behind a "Run" button; everything that
only changes how an already-computed result is DRAWN (the mobility threshold,
the Berglund D-correction, which conditions/figures are shown) lives in the
toolbar's three other tools — Visualization properties, Visibility,
Conditions — and never needs a refit.
"""
from __future__ import annotations

__all__ = ["compute", "sidebar", "toolbar", "window"]
