"""The light / dark switch's words and its one rule, with no Qt in them (thin Qt; `ui.theme` holds the setting).

THE SETTING IS ONE: `ui.theme.saved_scheme()` / `save_scheme()` (key `color_scheme` in `ui.json`). Every window builds
its colours from it at construction (`theme.current()`), so a switch is "save, then open the window again"; the
analysis workspace (`ui.workspace.window._toggle_theme`) and the start page (`ui.onboarding.stages.StagesDialog`)
both ask here which scheme a press switches to and what the button shows and says, so they cannot disagree.

* `other_scheme` -- the scheme a press switches to.
* `switch_glyph` -- the icon: it names what a click SWITCHES TO (a moon while light, a sun while dark).
* `switch_tip` -- the hover text, in the same convention ("Dark mode" while light).
"""
from __future__ import annotations

__all__ = ["other_scheme", "switch_glyph", "switch_tip"]


def _is_dark(scheme) -> bool:
    return scheme == "dark"


def other_scheme(scheme: str) -> str:
    """The scheme a press switches to: dark from light, light from anything else."""
    return "light" if _is_dark(scheme) else "dark"


def switch_glyph(scheme: str) -> str:
    """The icon name (`ui.theme.icons`): a sun while dark, a moon while light."""
    return "sun" if _is_dark(scheme) else "moon"


def switch_tip(scheme: str) -> str:
    """The hover text: "Light mode" while dark, "Dark mode" while light."""
    return "Light mode" if _is_dark(scheme) else "Dark mode"
