"""The application's shadcn/ui shell, in three layers.

Brought over from the `prototype/` folder, which is where every value in here was
argued out against the registry and measured on a real screen. That folder stays
the reference; THIS is the copy the application imports, and the two are expected
to diverge the moment a real backend is wired to a mock-up's control.

    theme/       colour, type, glyphs — what a widget CONSUMES
    widgets/     surfaces, controls, tooltip — what a page ASSEMBLES
    onboarding/  a flow, with its own screens and its own state

The dependencies run one way down that list. A widget reads `theme` and holds no
colour or size of its own; a screen reads both; nothing in `theme` knows that
widgets exist. The split is by what a module IS rather than by which screen uses
it, which is why `controls` sits beside `primitives` instead of beside the page
that presses its buttons.

**Nothing in the analysis spine may import this package.** The spine's defining
property is that it runs headless — 36 000 of the 52 000 lines in `smtracker`
never import a UI toolkit, which is why the analyses can be exercised without a
display. A `from .ui import ...` in an analysis module is the one change that
quietly ends that, so the dependency runs one way only: pages read the analysis
registry (`smtracker.analyses`), never the reverse.

**Qt arrives through `qtpy`**, as everywhere else in this project — the binding
is PySide6 for the licensing reason CLAUDE.md records, and naming it directly in
30 modules is how a project ends up unable to change its mind. The prototype
imports PySide6 by name because it answers to nobody; these copies were rewritten
on the way in.

This module imports NOTHING, deliberately. `theme.tokens` and `theme.hcl` are
Qt-free and can be reached on a machine with no display; everything else pulls in
a toolkit, and an import here would make `smtracker.ui` cost Qt for anyone who
only wanted a colour.
"""
from __future__ import annotations

__all__ = ["onboarding", "theme", "widgets"]
