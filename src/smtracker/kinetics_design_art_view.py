"""The storyboards of the Run dialog's design animations: what is drawn at each moment of a loop, with no Qt in it.

`ui/workspace/kinetics_design_art.py` paints what `scene` returns (the thin-Qt rule, `tests/test_ui_thin_qt.py`); this is
where the drawing is decided, so a test can ask what each design shows at any moment.

MECHANISM, NOT DATA (as the Spatial Map dialog's `_SettingArt`): ink and greys, ONE accent for the addition. Each loop is
about five seconds (`LOOP_S`); the last stretch holds the finished picture, and `STILL` is the moment a reduced-motion
setting freezes on. Each design shows what is measured from where and which box is the independent unit (spec
`docs/kinetics_designs_spec.md`, "The animations"):

1. Added under the microscope: a chamber of cells under the objective and a timeline of frames. The addition arrives as a
   flow at t0; frames before it are grey (the baseline), after it dark. The same three cells stay outlined throughout.
   Unit badge: chamber.
2. Added to the culture: a flask, a pipette adds at t0 and a clock starts; a hatched stretch "incubation and mounting,
   not observed"; the slide goes under the objective; movie strips appear at t1, t2, t3 on the timeline, each labelled
   with its time since the addition, a baseline strip before t0 from the same flask. Unit badge: culture.
3. Added to each sample: three tubes, each with its own pipette at a different time; each goes under the objective;
   their timelines slide onto ONE axis of time since each sample's own addition. Unit badge: culture (samples within it).

THE SCENE is in a box 1 wide and `HEIGHT` tall; items are plain dicts, ``kind`` one of "rect", "ellipse", "line",
"poly", "text"; ``style`` one of "ink", "grey", "faint", "accent", "hatch"; ``alpha`` 0-1.
"""
from __future__ import annotations

from .kinetics_designs import DESIGN_UNITS, DESIGN_WORDS, DESIGNS

__all__ = ["LOOP_S", "HOLD_FROM", "STILL", "HEIGHT", "STYLES", "scene", "texts", "design_heading"]

#: One loop, in seconds ("about 5 s").
LOOP_S = 5.0
#: From here to the loop's end the finished picture is held.
HOLD_FROM = 0.88
#: The moment a reduced-motion setting freezes on: everything shown.
STILL = 0.95
#: The scene's height, its width being 1.
HEIGHT = 0.62
STYLES = ("ink", "grey", "faint", "accent", "hatch")


def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _ramp(p: float, a: float, b: float) -> float:
    """0 before ``a``, 1 after ``b``, smooth between."""
    x = _clamp((p - a) / (b - a))
    return x * x * (3.0 - 2.0 * x)


def _rect(x, y, w, h, style="ink", *, fill=True, alpha=1.0):
    return {"kind": "rect", "x": x, "y": y, "w": w, "h": h, "style": style, "fill": fill, "alpha": alpha}


def _ellipse(x, y, rx, ry, style="ink", *, fill=True, alpha=1.0):
    return {"kind": "ellipse", "x": x, "y": y, "rx": rx, "ry": ry, "style": style, "fill": fill, "alpha": alpha}


def _line(x1, y1, x2, y2, style="ink", *, width=1.0, alpha=1.0):
    return {"kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "style": style, "width": width, "alpha": alpha}


def _poly(points, style="ink", *, fill=False, alpha=1.0):
    return {"kind": "poly", "points": list(points), "style": style, "fill": fill, "alpha": alpha}


def _text(x, y, text, style="ink", *, size=0.034, align="left", alpha=1.0):
    return {"kind": "text", "x": x, "y": y, "text": text, "style": style, "size": size, "align": align,
            "alpha": alpha}


def _badge(design: str):
    return _text(0.99, 0.0, "unit: " + DESIGN_UNITS[design], "ink", size=0.034, align="right")


# --------------------------------------------------------------------------- #
# 1. Added under the microscope
# --------------------------------------------------------------------------- #

_T0_MICRO = 0.46          # where the addition falls on the timeline
_CELLS = ((0.30, 0.20), (0.50, 0.23), (0.69, 0.18))


def _microscope(p: float) -> list:
    items = [_badge("microscope")]
    items.append(_rect(0.30, 0.0, 0.14, 0.04, "ink"))                       # the objective, above the chamber
    items.append(_text(0.37, 0.05, "objective", "grey", size=0.028, align="center"))
    items.append(_rect(0.08, 0.09, 0.84, 0.21, "grey", fill=False))        # the chamber
    shade = _ramp(p, 0.42, 0.56)
    for x, y in _CELLS:                                                     # the same three cells throughout
        items.append(_ellipse(x, y, 0.045, 0.022, "ink", fill=True, alpha=0.10 + 0.55 * shade))
        items.append(_ellipse(x, y, 0.045, 0.022, "ink", fill=False))
    flow = _ramp(p, 0.30, 0.56)                                             # the addition arrives as a flow
    if 0.0 < flow < 1.0:
        for k in range(5):
            x = 0.06 + (0.88 * flow) - 0.05 * k
            if 0.07 < x < 0.93:
                items.append(_ellipse(x, 0.12 + 0.035 * ((k * 3) % 5), 0.009, 0.009, "accent"))
        items.append(_line(0.0, 0.195, 0.07, 0.195, "accent", width=2.0, alpha=1.0 - flow))
    # the timeline: frames tick as the playhead passes; before the addition grey, after it dark
    y = 0.47
    items.append(_line(0.08, y, 0.92, y, "faint", width=1.0))
    head = 0.10 + 0.80 * _ramp(p, 0.04, 0.86)
    n = 14
    for i in range(n):
        x = 0.10 + 0.80 * i / (n - 1)
        if x <= head + 1e-9:
            items.append(_rect(x - 0.006, y - 0.045, 0.012, 0.09, "grey" if x < _T0_MICRO else "ink"))
    items.append(_line(_T0_MICRO, y - 0.08, _T0_MICRO, y + 0.08, "accent", width=2.0))
    items.append(_text(_T0_MICRO, y + 0.095, "addition", "accent", align="center"))
    items.append(_text(0.26, y + 0.095, "baseline", "grey", align="center"))
    items.append(_text(0.70, y + 0.095, "after", "ink", align="center"))
    return items


# --------------------------------------------------------------------------- #
# 2. Added to the culture
# --------------------------------------------------------------------------- #

_T0_CULTURE = 0.24


def _culture(p: float) -> list:
    items = [_badge("culture")]
    # the flask, a pipette adds at t0 and the liquid takes the accent
    items.append(_rect(0.09, 0.05, 0.04, 0.07, "ink", fill=False))
    items.append(_poly([(0.09, 0.12), (0.13, 0.12), (0.19, 0.28), (0.03, 0.28)], "ink"))
    items.append(_poly([(0.055, 0.21), (0.165, 0.21), (0.19, 0.28), (0.03, 0.28)], "grey", fill=True, alpha=0.35))
    tint = _ramp(p, 0.20, 0.30)
    if tint > 0:
        items.append(_poly([(0.055, 0.21), (0.165, 0.21), (0.19, 0.28), (0.03, 0.28)], "accent", fill=True,
                           alpha=0.45 * tint))
    drop = _ramp(p, 0.06, 0.20)
    items.append(_line(0.11, 0.0, 0.11, 0.045, "ink", width=2.0))             # the pipette
    if 0.0 < drop < 1.0:
        items.append(_ellipse(0.11, 0.05 + 0.15 * drop, 0.009, 0.012, "accent"))
    if p >= 0.20:
        items.append(_text(0.215, 0.20, "t = 0: the clock starts", "accent", size=0.03))
    # the slide goes under the objective
    move = _ramp(p, 0.26, 0.50)
    items.append(_rect(0.54, 0.0, 0.12, 0.04, "ink"))                       # left of the unit badge
    items.append(_text(0.60, 0.05, "objective", "grey", size=0.028, align="center"))
    items.append(_rect(0.22 + 0.33 * move, 0.10 + 0.0 * move, 0.10, 0.025, "grey", fill=True, alpha=0.55))
    # the timeline
    y = 0.47
    items.append(_line(0.04, y, 0.96, y, "faint"))
    items.append(_rect(0.04, y - 0.045, 0.19, 0.09, "grey", alpha=0.8))     # baseline strip, the same flask
    items.append(_text(0.11, y - 0.095, "baseline", "grey", align="center"))
    items.append(_line(_T0_CULTURE, y - 0.08, _T0_CULTURE, y + 0.08, "accent", width=2.0))
    items.append(_text(_T0_CULTURE + 0.015, y - 0.095, "addition", "accent"))
    show = _ramp(p, 0.24, 0.36)
    if show > 0:
        items.append(_rect(_T0_CULTURE + 0.01, y - 0.045, 0.23, 0.09, "hatch", alpha=show))     # ends before +6 min
        items.append(_text(_T0_CULTURE + 0.125, y + 0.07, "incubation and mounting", "grey", size=0.028,
                           align="center", alpha=show))                                         # below the stretch
        items.append(_text(_T0_CULTURE + 0.125, y + 0.108, "not observed", "grey", size=0.028, align="center",
                           alpha=show))
    # the movies' labels stand ABOVE their strips, clear of the caption under the hatched stretch
    for start, when, label in ((0.50, 0.50, "+6 min"), (0.65, 0.62, "+30 min"), (0.80, 0.74, "+60 min")):
        a = _ramp(p, when, when + 0.07)
        if a > 0:
            items.append(_rect(start, y - 0.045, 0.11, 0.09, "ink", alpha=a))
            items.append(_text(start + 0.055, y - 0.095, label, "ink", size=0.03, align="center", alpha=a))
    return items


# --------------------------------------------------------------------------- #
# 3. Added to each sample
# --------------------------------------------------------------------------- #

#: (row's y, the moment its pipette adds, where its addition falls before the rows are aligned)
_SAMPLES = ((0.11, 0.08, 0.40), (0.235, 0.22, 0.50), (0.36, 0.36, 0.46))
_COMMON = 0.28


def _samples(p: float) -> list:
    items = [_badge("sample")]
    slide = _ramp(p, 0.62, 0.88)                                             # the timelines slide onto one axis
    for row, (y, when, t0) in enumerate(_SAMPLES):
        items.append(_text(0.08, y - 0.045, f"sample {row + 1}", "grey", size=0.028))      # beside its tube
        items.append(_rect(0.02, y - 0.03, 0.045, 0.065, "ink", fill=False))              # its tube
        drop = _ramp(p, when, when + 0.08)
        if 0.0 < drop < 1.0:
            items.append(_ellipse(0.0425, y - 0.07 + 0.05 * drop, 0.008, 0.011, "accent"))
        items.append(_line(0.0425, y - 0.085, 0.0425, y - 0.05, "ink", width=1.5, alpha=0.7))   # its own pipette
        added = _ramp(p, when + 0.06, when + 0.12)
        if added > 0:
            items.append(_rect(0.022, y + 0.005, 0.041, 0.03, "accent", alpha=0.5 * added))
        # its timeline: begins at its own addition, then slides to the shared one
        x0 = t0 - (t0 - _COMMON) * slide
        show = _ramp(p, when + 0.10, when + 0.20)
        if show > 0:
            items.append(_line(x0 - 0.12, y + 0.06, 0.96, y + 0.06, "faint", alpha=show))
            items.append(_line(x0, y - 0.035, x0, y + 0.085, "accent", width=2.0, alpha=show))
            items.append(_rect(x0 - 0.12, y + 0.02, 0.10, 0.04, "grey", alpha=show * 0.8))     # baseline movie
            for k, off in enumerate((0.06, 0.18, 0.30)):
                items.append(_rect(x0 + off, y + 0.02, 0.10, 0.04, "ink", alpha=show))
    done = _ramp(p, 0.84, 0.94)
    if done > 0:
        items.append(_line(_COMMON, 0.52, 0.96, 0.52, "ink", width=1.5, alpha=done))
        items.append(_line(_COMMON, 0.49, _COMMON, 0.55, "accent", width=2.0, alpha=done))
        items.append(_text(_COMMON, 0.575, "0", "accent", align="center", alpha=done))
        items.append(_text(0.31, 0.575, "time since each sample's own addition", "ink", size=0.026, alpha=done))
    return items


_SCENES = {"microscope": _microscope, "culture": _culture, "sample": _samples}


def scene(design: str, p: float) -> list:
    """The items to draw for ``design`` at loop progress ``p`` (0 to 1; `HOLD_FROM` on, the finished picture)."""
    if design not in _SCENES:
        raise ValueError(f"unknown design {design!r}: choose one of {', '.join(DESIGNS)}")
    return _SCENES[design](_clamp(float(p)))


def texts(design: str, p: float = STILL) -> list:
    """The words in the picture at progress ``p``, in drawing order."""
    return [i["text"] for i in scene(design, p) if i["kind"] == "text"]


def design_heading(design: str) -> tuple:
    """``(name, explanation line 1, explanation line 2)`` under the animation, from the headless words."""
    from .kinetics_designs import DESIGN_EXPLANATIONS

    return (DESIGN_WORDS[design],) + tuple(DESIGN_EXPLANATIONS[design])
