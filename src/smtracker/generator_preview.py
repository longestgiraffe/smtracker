"""The generator window's PAGES and PICTURES, headless (25 Nov 2026, the pages of `docs/generator_plan.md`).

Asked: "can we make it a bit more beautfil? maybe multiple pages clickable next iwth animations explaning stuff and
some helpers?". The window (`ui.generator.window`) is a rail of pages, each with its settings beside a picture. Which
page holds which setting, which page a problem belongs to, and every number a picture draws are here, as plain
lists, dicts and strings (the thin-Qt rule): the window only lays out and paints.

  PAGES, fields_on(state, page)  the pages, and the settings each shows for the form as it stands
  page_problems(problems)        each problem with the page it belongs to (`problem_page`)
  EXAMPLES, example_state(key)   the sets to start from
  bubbles(state)                 each condition's populations as the SQD chart draws them (D up, area = share)
  cell_animation(state, i)       ONE cell of condition i with its places drawn, and frames of molecules moving in
                                 them, made by the generator's own engine (`generator._simulate`), so what moves on
                                 the page is what a build simulates (a preview: no pilot, no bleaching)
  state_diagram(state, i)        the populations as circles sized by share, the rates between them as arrows,
                                 each population's mean dwell
  kinetic_curve(state, i)        the response h(t) with its delay and half-time, each movie on the clock, the
                                 shares and D before and after
  field_layout(state)            the camera field with cells placed at its scale, or why they do not fit
  frame_preview(state, i)        one movie frame by the build's own renderer (`generator.render_frames`), 8-bit grey
  expected_precision_nm(state)   the localisation precision the photons, background and PSF allow (Mortensen)
  ceiling_warnings(state)        a D faster than a 5 px link reads at this frame interval (`max_measurable_D`)

No Qt.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np

from . import generator as gen
from . import generator_view as gv

__all__ = ["POP_COLOURS", "PAGES", "PAGE_TITLES", "PAGE_NOTES", "CONDITION_PAGES", "RESHAPE_KEYS", "TYPICAL_D",
           "LINK_RADIUS_PX", "EXAMPLES", "example_state", "fields_on", "problem_page", "page_problems", "bubbles",
           "cell_animation", "state_diagram", "kinetic_curve", "field_layout", "frame_preview",
           "expected_precision_nm", "ceiling_warnings", "ANIMATION_FPS"]

#: One colour per population, by its place in the condition: Okabe-Ito, as the SynR figure's bound blue,
#: scanning green and free orange.
POP_COLOURS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00")
#: The window's pages: (key, title, rail glyph).
PAGES = (("start", "Start", "sparkles"), ("populations", "Populations", "layers"),
         ("where", "Where in the cell", "crosshair"), ("switching", "Switching", "shuffle"),
         ("kinetics", "Kinetics", "clock"), ("cells", "Cells and field", "grid-2x2"),
         ("camera", "Camera and movie", "aperture"), ("build", "Build", "play"))
PAGE_TITLES = {key: title for key, title, _g in PAGES}
#: The pages that edit one condition at a time (the window shows the condition picker above them).
CONDITION_PAGES = ("populations", "where", "switching", "kinetics")
#: One line under each page's title.
PAGE_NOTES = {
    "start": "Begin from an example or from the set as it stands, and choose what to build. Every page can be "
             "changed later, in any order.",
    "populations": "Each condition's diffusive populations: how fast each moves and what share of the molecules "
                   "it holds.",
    "where": "Where in the cell each population lives. The picture moves molecules with the generator's own "
             "engine.",
    "switching": "Whether molecules change population, and how fast.",
    "kinetics": "A change after an addition: each movie is recorded at its own time, and the populations move from "
                "before to after.",
    "cells": "The cells' size, how many sit in a movie's field, and how many molecules each shows.",
    "camera": "The frame interval, the pixel, the movie's length and, for movies, the camera and everything that "
              "makes a real movie imperfect.",
    "build": "What will be built, where it is written, and the build itself.",
}
#: What a D usually is, the hints on the Populations page (µm²/s).
TYPICAL_D = (("Bound or immobile", "0 to 0.02"), ("Scanning DNA", "about 0.1 to 0.3"),
             ("A free enzyme in the cytoplasm", "about 1 to 2"), ("Free mNeonGreen", "about 5 to 8"))
#: The linking radius a set is usually tracked at (px; TrackMate's standard, `docs/claude/linking_and_limits.md`).
LINK_RADIUS_PX = 5.0
#: The cell animation's frame rate on the page.
ANIMATION_FPS = 20
#: A choice whose change shows or hides other settings: the page is drawn again.
RESHAPE_KEYS = ("output", "camera", "foci_layout", "foci_depth")

#: Which page shows which field (`generator_view.FIELDS`); the rest of a page is drawn by the window itself.
_PAGE_OF = {
    "cells": ("cell_length_mean_um", "cell_length_sd_um", "cell_length_range_low", "cell_length_range_high",
              "cell_width_um", "dims", "cells_per_movie", "molecules_per_cell", "molecules_sd_log", "field_px_x",
              "field_px_y"),
    "where": ("membrane_depth_um", "pole_depth_um", "midcell_width_um", "foci_layout", "foci_depth", "foci_per_cell",
              "foci_per_cell_min", "foci_positions", "focus_radius_um"),
    "camera": ("pixel_size_nm", "frame_interval_ms", "exposure_ms", "n_frames", "n_subframes", "bleach_mean_frames",
               "localisation_error_nm", "min_localizations"),
    "imperfections": ("camera", "camera_offset_counts", "camera_em_gain", "camera_e_per_count", "camera_read_noise_e",
                      "emission_nm", "numerical_aperture", "photons_per_spot", "background_photons_per_px",
                      "background_change", "illumination_sigma_px", "autofluorescence_photons_per_px",
                      "autofluorescence_half_life_frames", "autofluorescence_cell_cv", "haze_photons_per_px",
                      "defocus_depth_um", "blink_off_per_frame", "blink_on_per_frame", "emission_cv",
                      "emission_corr_frames"),
    "build": ("name", "replicates", "movies_per_replicate", "seed", "write_outlines"),
}
_OWN_CAMERA = ("camera_offset_counts", "camera_em_gain", "camera_e_per_count", "camera_read_noise_e")


def fields_on(state: dict, page: str) -> list:
    """The `generator_view.Field` s a page (or the camera page's "imperfections" card) shows for the form as it
    stands: the localisation error and the shortest track only for tracks, the true outlines only for movies, the
    own camera's values only for an own camera, the foci positions only where they are given, and the count of foci
    only where the layout uses one."""
    f = state["fields"]
    movies = f.get("output") == "movies"
    hide = set()
    if movies:
        hide |= {"localisation_error_nm", "min_localizations"}
    else:
        hide |= {"write_outlines"}
    if f.get("camera") != "own":
        hide |= set(_OWN_CAMERA)
    layout = f.get("foci_layout")
    if layout != "positions":
        hide.add("foci_positions")
    if layout in ("nucleoid", "poles", "positions") or (layout == "midcell" and f.get("foci_depth") != "membrane"):
        hide |= {"foci_per_cell", "foci_per_cell_min"}
    keys = _PAGE_OF.get(page, ())
    by_key = {field.key: field for field in gv.FIELDS}
    return [by_key[k] for k in keys if k in by_key and k not in hide]


# --------------------------------------------------------------------------- #
# Problems, each with its page
# --------------------------------------------------------------------------- #

#: Which page a problem names (`problem_page`): the first rule whose words appear wins (case ignored, the
#: condition's own name left out, since a condition may be called anything).
_PAGE_RULES = (("build", ("choose where the set is written", "choose a new folder")),
               ("kinetics", ("kinetics", "one time per movie", "the half-time", "onset", "movies overlap",
                             "after the addition")),
               ("where", ("'where'", "foci", "membrane", "pole", "mid-cell", "focus", "regions do not overlap")),
               ("switching", (" rate", "switch", "spread between replicates")),
               ("populations", ("condition", "population", "share", "same name")),
               ("cells", ("cell", "field", "dims")),
               ("camera", ("frame", "sub-step", "pixel", "bleaching", "localisation", "exposure", "camera", "photons",
                           "background", "autofluorescence", "haze", "beam", "focal", "brightness", "blinking",
                           "track needs")))


def problem_page(text: str) -> str:
    """The page (`PAGES` key) a problem's words belong to; "build" where none says."""
    words = re.sub(r"^Condition (['\"]).*?\1", "Condition", str(text)).lower()
    for page, keys in _PAGE_RULES:
        if any(k in words for k in keys):
            return page
    return "build"


def page_problems(problems: list) -> list:
    """``[(page, text)]``: each problem with the page where it is put right."""
    return [(problem_page(p), str(p)) for p in problems]


# --------------------------------------------------------------------------- #
# Examples to start from
# --------------------------------------------------------------------------- #


def _example_recipes() -> dict:
    P, C = gen.Population, gen.Condition
    small = dict(replicates=3, movies_per_replicate=10)
    free = gen.Recipe(conditions=(C("free protein", (P(2.0, 1.0, "cell", "free"),)),), name="free protein",
                      note="one free state, nothing bound", **small)
    binder = gen.Recipe(conditions=(
        C("wild type", (P(0.0, 0.3, "nucleoid", "bound"), P(1.0, 0.7, "cell", "free")), exchange_per_s=2.0,
          replicate_sd=0.1),
        C("mutant", (P(0.0, 0.1, "nucleoid", "bound"), P(1.0, 0.9, "cell", "free")), exchange_per_s=2.0,
          replicate_sd=0.1)), name="two-state binder", note="bound in the nucleoid or free; less bound in the mutant",
        **small)
    three = gen.Recipe(conditions=(
        C("condition 1", (P(0.0, 0.30, "nucleoid", "bound"), P(0.25, 0.35, "nucleoid", "scanning"),
                   P(1.8, 0.35, "cell", "free")), exchange_per_s=3.0, replicate_sd=0.12),
        C("condition 2", (P(0.0, 0.12, "nucleoid", "bound"), P(0.25, 0.38, "nucleoid", "scanning"),
                  P(1.8, 0.50, "cell", "free")), exchange_per_s=3.0, replicate_sd=0.12)),
        name="three states", note="bound, scanning the nucleoid, free; as SynR at a high and a low co-activator level",
        **small)
    minutes = (-6, -4, -2, 0.5, 1.5, 3, 5, 7, 10, 14)
    before = (P(0.0, 0.35, "cell", "bound"), P(1.0, 0.65, "cell", "free"))
    kinetics = gen.Recipe(conditions=(
        C("binding rise", before, exchange_per_s=1.0, kinetics=gen.Kinetics(
            movie_minutes=minutes, after_populations=(P(0.0, 0.74, "cell", "bound"), P(1.0, 0.26, "cell", "free")))),
        C("mock", before, exchange_per_s=1.0, kinetics=gen.Kinetics(movie_minutes=minutes))),
        name="kinetics", note="binding rises after a compound is added; a mock beside it", **small)
    return {"free": free, "binder": binder, "three": three, "kinetics": kinetics}


EXAMPLES = (("free", "A free protein", "One diffusing state, nothing bound: the control for how many populations "
                                       "a fit finds."),
            ("binder", "A two-state binder", "Bound in the nucleoid or free, and less bound in a mutant: the "
                                             "simplest condition difference."),
            ("three", "Three states", "Bound, scanning the nucleoid and free, at two levels of binding, as the "
                                      "paper's SynR."),
            ("kinetics", "A kinetics experiment", "Binding rises after a compound is added, a mock beside it; each "
                                                  "movie at its own time."))


def example_state(key: str) -> dict:
    """The form of one example (`EXAMPLES`), the other settings at the generator's defaults."""
    recipes = _example_recipes()
    if key not in recipes:
        raise gen.GeneratorError(f"No example {key!r}.")
    return gv.recipe_to_state(recipes[key])


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #


def _recipe(state):
    try:
        return gv.state_to_recipe(state)
    except (gen.GeneratorError, TypeError, ValueError, KeyError):
        return None


def _cond_index(state: dict, condition: int) -> int:
    return condition if 0 <= condition < len(state["conditions"]) else 0


def _shares_of(cond: dict) -> list:
    """The shares a condition holds: typed, or a rate matrix's stationary ones."""
    shares = [float(p["share"]) for p in cond["populations"]]
    if cond["switching"] == "matrix" and len(shares) > 1:
        try:
            rates = np.array(cond["matrix"], dtype=float)
            np.fill_diagonal(rates, 0.0)
            return [float(x) for x in gen._stationary(rates)]
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return [0.0] * len(shares)
    return shares


def bubbles(state: dict) -> dict:
    """The declared populations as the SQD chart draws them::

        {"columns": [names], "points": [{"column", "D", "share", "y", "r", "colour", "name"}],
         "ticks": [[words, y]], "zero_y"}

    ``y`` runs from 0 (bottom) to 1 on a log axis of D, D 0 drawn on its own "0" tick under the rest (as
    `chart_text.zero_d_stand_in`); ``r`` is the radius as a fraction of the largest, area = share."""
    columns, points = [], []
    for ci, c in enumerate(state["conditions"]):
        columns.append(c["name"] or f"condition {ci + 1}")
        shares = _shares_of(c)
        for pi, p in enumerate(c["populations"]):
            points.append({"column": ci, "D": float(p["D"]), "share": float(shares[pi]),
                           "colour": POP_COLOURS[pi % len(POP_COLOURS)], "name": p["name"] or f"population {pi + 1}"})
    ds = [pt["D"] for pt in points if pt["D"] > 0]
    lo = math.floor(math.log10(min([0.1] + ds) / 1.5))
    hi = math.ceil(math.log10(max([1.0] + ds) * 1.5))
    zero_y = 0.04
    span = max(hi - lo, 1)

    def y_of(d):
        return zero_y if d <= 0 else 0.14 + 0.82 * (math.log10(d) - lo) / span

    for pt in points:
        pt["y"] = y_of(pt["D"])
        pt["r"] = math.sqrt(max(pt["share"], 0.0))
    ticks = [[f"{10.0 ** e:g}", y_of(10.0 ** e)] for e in range(lo, hi + 1)]
    if any(pt["D"] <= 0 for pt in points):
        ticks.insert(0, ["0", zero_y])
    return {"columns": columns, "points": points, "ticks": ticks, "zero_y": zero_y}


def _capsule_outline(length: float, width: float, n_cap: int = 24) -> list:
    from .synthetic import CapsuleCell

    cell = CapsuleCell(1, 0.0, 0.0, 0.0, length, width, ())
    return [[float(x), float(y)] for x, y in cell.outline(n_cap=n_cap)]


def cell_animation(state: dict, condition: int = 0, *, molecules: int = 30, frames: int = 80, seed: int = 0) -> dict:
    """One cell of condition ``condition`` (the mean length, the width), its places, and ``frames`` frames of
    ``molecules`` molecules moving in them, by the generator's own engine, seen from above as the camera sees them.
    Plain lists in µm, the cell centred on (0, 0) with its axis along x::

        {"length", "width", "outline": [[x, y]], "inner": [[x, y]] (the membrane shell's inner edge),
         "lobes": [[centre, half_length, half_width]], "pole_x" (the poles begin beyond +-pole_x),
         "midcell_half", "foci": [[x, y]], "focus_radius", "used": [places a population lives in],
         "colours", "names", "wheres" (each population's place), "frames": [[[x, y, population], ...] per frame],
         "frame_ms", "slower" (how many times slower than the movie the page plays it), "caption" (the words
         under the picture), "error": "" or why there is no animation}
    """
    from .synthetic import CapsuleCell, nucleoid_lobes

    f = state["fields"]
    length = float(f["cell_length_mean_um"])
    width = float(f["cell_width_um"])
    depth = float(f["membrane_depth_um"])
    slower = 1000.0 / ANIMATION_FPS / max(float(f["frame_interval_ms"]), 1e-9)
    speed = (f"{slower:.2g} times slower than the movie" if slower >= 1.05 else
             f"{1.0 / slower:.2g} times faster than the movie" if slower <= 0.95 else "at the movie's speed")
    out = {"length": length, "width": width, "outline": [], "inner": [], "lobes": [], "foci": [],
           "pole_x": length / 2.0 - float(f["pole_depth_um"]), "midcell_half": float(f["midcell_width_um"]) / 2.0,
           "focus_radius": float(f["focus_radius_um"]), "frames": [], "frame_ms": float(f["frame_interval_ms"]),
           "slower": slower, "error": "", "colours": [], "names": [], "used": [], "wheres": [],
           "caption": (f"Seen from above, as the camera sees it, played {speed}: {molecules} molecules moved by the "
                       f"generator's own engine. Under it, the six places, each coloured where a population "
                       f"lives.")}
    if not (length > width > 0):
        out["error"] = "The cell must be longer than it is wide."
        return out
    cell = CapsuleCell(1, 0.0, 0.0, 0.0, length, width, nucleoid_lobes(length))
    out["outline"] = [[float(x), float(y)] for x, y in cell.outline(n_cap=24)]
    if 0 < depth < width / 2.0:
        out["inner"] = _capsule_outline(length - 2 * depth, width - 2 * depth)
    out["lobes"] = [[lb.centre_um, lb.half_length_um, lb.half_width_um] for lb in cell.lobes]
    recipe = _recipe(state)
    if recipe is None:
        out["error"] = "The form cannot be read yet: see Build."
        return out
    rng = np.random.default_rng(seed)
    cond = recipe.conditions[_cond_index(state, condition)]
    out["colours"] = [POP_COLOURS[i % len(POP_COLOURS)] for i in range(len(cond.populations))]
    out["names"] = [p.name or f"population {i + 1}" for i, p in enumerate(cond.populations)]
    out["used"] = sorted({p.where for p in cond.populations})
    out["wheres"] = [p.where for p in cond.populations]
    try:
        sites = gen.foci_sites(cell, recipe, rng)
        out["foci"] = [[float(u), float(v)] for u, v, _z in sites]
        pre = replace(recipe, n_subframes=2, localisation_error_nm=0.0)
        g = gen._geometry([cell], np.zeros(molecules, dtype=int), pre, {id(cell): sites})
        rates = gen.rates_matrix(cond)
        _obs, true, states = gen._simulate(rng, {"length": np.full(molecules, frames)}, g, pre, cond, rates,
                                           [p.share for p in cond.populations],
                                           warmup=min(20, gen.warmup_frames(rates, pre.tau_s)))
    except (gen.GeneratorError, ValueError, IndexError) as exc:
        out["error"] = str(exc)
        return out
    out["frames"] = [[[round(float(true[m, k, 0]), 4), round(float(true[m, k, 1]), 4), int(states[m, k])]
                      for m in range(molecules)] for k in range(frames)]
    return out


def state_diagram(state: dict, condition: int = 0) -> dict:
    """``{"nodes": [{"name", "share", "D", "colour", "x", "y", "dwell_ms"}], "edges": [{"from", "to", "rate"}],
    "mode", "note"}``: the populations around a circle (``x``, ``y`` from 0 to 1), each sized by its share, the rates
    between them as declared. With shares and an exchange rate the rates are the ones that hold the shares
    anywhere in the cell; a build sets the rates into a place by a short pilot, so a place makes them larger."""
    c = state["conditions"][_cond_index(state, condition)]
    shares = _shares_of(c)
    k = len(c["populations"])
    nodes = []
    for i, p in enumerate(c["populations"]):
        if k == 1:
            x, y = 0.5, 0.5
        elif k == 2:
            x, y = (0.2, 0.5) if i == 0 else (0.8, 0.5)
        else:
            a = math.pi / 2 + 2 * math.pi * i / k
            x, y = 0.5 + 0.36 * math.cos(a), 0.52 - 0.36 * math.sin(a)
        nodes.append({"name": p["name"] or f"population {i + 1}", "share": float(shares[i]), "D": float(p["D"]),
                      "colour": POP_COLOURS[i % len(POP_COLOURS)], "x": x, "y": y, "dwell_ms": None})
    edges, note = [], ""
    if c["switching"] != "none" and k > 1:
        recipe = _recipe(state)
        rates = None if recipe is None else gen.rates_matrix(recipe.conditions[_cond_index(state, condition)])
        if rates is not None:
            edges = [{"from": i, "to": j, "rate": float(rates[i, j])} for i in range(k) for j in range(k)
                     if i != j and rates[i, j] > 0]
            for i, n in enumerate(nodes):
                out_rate = float(sum(rates[i, j] for j in range(k) if j != i))
                n["dwell_ms"] = 1000.0 / out_rate if out_rate > 0 else None
            if c["switching"] == "exchange" and any(p["where"] != "cell" for p in c["populations"]):
                note = ("A population lives in a place: the build sets the rates into it by a short pilot, so that "
                        "the shares hold.")
    elif k > 1:
        note = "No switching: each molecule keeps the population it starts in."
    return {"nodes": nodes, "edges": edges, "mode": c["switching"], "note": note}


def kinetic_curve(state: dict, condition: int = 0, *, points: int = 200) -> dict:
    """``{"on", "t_min": [...], "h": [...], "x_ticks", "delay_min", "half_min" (the time h reaches one half), "movies":
    [[start_min, end_min]], "before": [shares], "after": [shares], "D_before", "D_after", "colours", "names", "mock",
    "error"}``: the response h(t) = 1 - 2^-((t - delay) / half-time) the build moves every value along."""
    c = state["conditions"][_cond_index(state, condition)]
    kin = c.get("kinetics")
    out = {"on": kin is not None, "names": [p["name"] or f"population {i + 1}" for i, p in enumerate(c["populations"])],
           "colours": [POP_COLOURS[i % len(POP_COLOURS)] for i in range(len(c["populations"]))], "error": "",
           "t_min": [], "h": [], "movies": []}
    if kin is None:
        return out
    try:
        minutes = gv.parse_minutes(kin["minutes"])
        delay, half = float(kin["delay_s"]), float(kin["half_time_s"])
    except (gen.GeneratorError, TypeError, ValueError) as exc:
        return {**out, "error": str(exc)}
    movie_min = float(state["fields"]["n_frames"]) * float(state["fields"]["frame_interval_ms"]) / 60000.0
    lo = min(min(minutes), 0.0) - 1.0
    hi = max(max(minutes) + movie_min, (delay + 4 * half) / 60.0) + 1.0
    t = np.linspace(lo, hi, points)
    h = gen.response(t * 60.0, delay, half) if half > 0 else (t * 60.0 >= delay).astype(float)
    mock = bool(kin.get("mock"))
    after = c["populations"] if mock else kin["after"]
    shares_after = _shares_of(c) if mock or c["switching"] == "matrix" else [float(a["share"]) for a in after]
    step = next(st for st in (1, 2, 5, 10, 15, 20, 30, 60, 120, 240, 600) if (hi - lo) / st <= 8)
    ticks = [float(x) for x in range(int(math.ceil(lo / step)) * step, int(hi) + 1, step)]
    return {**out, "t_min": [round(float(x), 4) for x in t], "h": [round(float(x), 5) for x in h], "x_ticks": ticks,
            "delay_min": delay / 60.0, "half_min": (delay + half) / 60.0,
            "movies": [[float(m), float(m) + movie_min] for m in minutes], "mock": mock,
            "before": _shares_of(c), "after": shares_after,
            "D_before": [float(p["D"]) for p in c["populations"]], "D_after": [float(a["D"]) for a in after],
            "matrix": c["switching"] == "matrix"}


def field_layout(state: dict, *, seed: int = 0) -> dict:
    """``{"width_um", "height_um", "width_px", "height_px", "pixel_um", "cells": [outline [[x, y]] in µm], "error"}``:
    one field as a build places its cells (`synthetic.place_cells`), or why they do not fit."""
    from .synthetic import SyntheticError, place_cells

    f = state["fields"]
    px = float(f["pixel_size_nm"]) / 1000.0
    w, h = int(f["field_px_x"]), int(f["field_px_y"])
    out = {"width_um": w * px, "height_um": h * px, "width_px": w, "height_px": h, "pixel_um": px, "cells": [],
           "error": ""}
    lo, hi = float(f["cell_length_range_low"]), float(f["cell_length_range_high"])
    rng = np.random.default_rng(seed)
    try:
        lengths = np.clip(rng.normal(float(f["cell_length_mean_um"]), float(f["cell_length_sd_um"]),
                                     int(f["cells_per_movie"])), lo, hi)
        cells = place_cells(rng, lengths, width_um=float(f["cell_width_um"]), field_um=(w * px, h * px))
    except SyntheticError:
        out["error"] = (f"{int(f['cells_per_movie'])} cells of up to {hi:g} µm do not fit this field: fewer or "
                        f"shorter cells, or a larger field.")
        return out
    except (ValueError, TypeError) as exc:
        out["error"] = str(exc)
        return out
    for c in cells:
        out["cells"].append([[round(float(a), 4), round(float(b), 4)] for a, b in c.outline(n_cap=16)])
    return out


def frame_preview(state: dict, condition: int = 0, *, seed: int = 0, size_px: int = 96) -> dict:
    """One movie frame (a ``size_px`` square of the field, as many cells as the field holds per area) by the
    build's own renderer, `generator.render_frames`, with the current camera, photons and imperfections::

        {"width", "height", "grey": [0..255, row by row], "low", "high" (the counts grey 0 and 255 stand for),
         "spots", "error"}

    The molecules sit as a frame mid-movie holds them: each cell shows about as many at once as the bleaching and
    the movie's length leave (at least one), each in its population's place, blurred over the exposure."""
    from .synthetic import SyntheticError, place_cells

    out = {"width": 0, "height": 0, "grey": [], "low": 0.0, "high": 0.0, "spots": 0, "error": ""}
    recipe = _recipe(state)
    if recipe is None:
        out["error"] = "The form cannot be read yet: see Build."
        return out
    W, H = recipe.field_px
    w, h = min(size_px, W), min(size_px, H)
    pre = replace(recipe, field_px=(w, h), n_frames=1)
    rng = np.random.default_rng(seed)
    lo, hi = pre.cell_length_range_um
    n_cells = max(1, int(round(pre.cells_per_movie * (w * h) / float(W * H))))
    try:
        lengths = np.clip(rng.normal(pre.cell_length_mean_um, pre.cell_length_sd_um, n_cells), lo, hi)
        cells = place_cells(rng, lengths, width_um=pre.cell_width_um, field_um=pre.field_um)
    except SyntheticError:
        out["error"] = "The cells do not fit the preview's field."
        return out
    cond = pre.conditions[_cond_index(state, condition)]
    at_once = pre.molecules_per_cell * min(1.0, pre.bleach_mean_frames / max(recipe.n_frames, 1))
    counts = np.maximum(rng.poisson(at_once, len(cells)), 1)
    cell_of = np.repeat(np.arange(len(cells)), counts)
    foci = gen.cell_foci(np.random.default_rng([seed, 31]), cells, pre)
    try:
        g = gen._geometry(cells, cell_of, pre, foci)
        rates = gen.rates_matrix(cond)
        result = gen._simulate(rng, {"length": np.ones(len(cell_of), dtype=int)}, g, pre, cond, rates,
                               [p.share for p in cond.populations],
                               warmup=min(20, gen.warmup_frames(rates, pre.tau_s)), keep_sub=True)
    except (gen.GeneratorError, ValueError, IndexError) as exc:
        out["error"] = str(exc)
        return out
    sub = result[3]
    import pandas as pd

    um = pre.pixel_size_nm / 1000.0
    psf = gen.psf_px_of(pre)
    illum = gen._illumination(pre)
    exposed = pre.exposed_subframes
    rows = []
    for i, ci in enumerate(cell_of):
        c = cells[ci]
        for s in range(exposed):
            u, v, z = (float(x) for x in sub[i, 0, s])
            fx, fy = c.to_field(u, v)
            x, y = float(fx) / um, float(fy) / um
            if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
                continue
            beam = float(illum[int(round(y)), int(round(x))])
            sig = psf * math.sqrt(1.0 + (z / pre.defocus_depth_um) ** 2) if pre.defocus_depth_um > 0 else psf
            rows.append({"track_id": i + 1, "frame": 1, "x": x, "y": y,
                         "photons": pre.photons_per_spot / exposed * beam, "sigma_px": sig})
    spots = pd.DataFrame(rows, columns=["track_id", "frame", "x", "y", "photons", "sigma_px"])
    stack = gen.render_frames(pre, cells, spots, rng, 1, illum=illum)
    frame = np.asarray(stack[0], dtype=float)
    low, high = float(np.percentile(frame, 1.0)), float(np.percentile(frame, 99.8))
    grey = np.clip((frame - low) / max(high - low, 1.0) * 255.0, 0, 255).astype(int)
    return {"width": int(w), "height": int(h), "grey": [int(x) for x in grey.ravel()], "low": low, "high": high,
            "spots": int(len(cell_of)), "error": ""}


# --------------------------------------------------------------------------- #
# Helpers' numbers
# --------------------------------------------------------------------------- #


def expected_precision_nm(state: dict) -> float | None:
    """The localisation precision a still molecule's photons, the background and the PSF allow, by Mortensen et
    al.'s formula (2010), with the EM register's excess noise (a factor 2 in variance) where the camera has an EM
    gain: var = F (s_a^2 / N) (16/9 + 8 pi s_a^2 b / (N a^2)), s_a^2 = s^2 + a^2 / 12. None where the form does not
    read. A moving molecule is blurred over the exposure, so its error is larger."""
    recipe = _recipe(state)
    if recipe is None:
        return None
    s = gen.psf_px_of(recipe) * recipe.pixel_size_nm
    a = recipe.pixel_size_nm
    n = recipe.photons_per_spot
    b = recipe.background_photons_per_px + recipe.haze_photons_per_px + recipe.autofluorescence_photons_per_px
    if not (n > 0 and a > 0 and s > 0):
        return None
    excess = 2.0 if float(gen.camera_calibration(recipe)["em_gain"]) > 1.0 else 1.0
    sa2 = s * s + a * a / 12.0
    return float(math.sqrt(excess * sa2 / n * (16.0 / 9.0 + 8.0 * math.pi * sa2 * b / (n * a * a))))


def ceiling_warnings(state: dict) -> list:
    """A population faster than a `LINK_RADIUS_PX` link still reads within 10 % at this frame interval and pixel
    (`trackmate_run.max_measurable_D`, the limit the preparation window shows): its long steps are lost, so its D
    reads low. Said for movies, which are tracked; tracks are written linked."""
    from .trackmate_run import max_measurable_D

    f = state["fields"]
    if f.get("output") != "movies":
        return []
    try:
        tau_ms = float(f["frame_interval_ms"])
        limit = max_measurable_D(radius_px=LINK_RADIUS_PX, tau_s=tau_ms / 1000.0,
                                 pixel_size_nm=float(f["pixel_size_nm"]))
    except (ValueError, TypeError, ZeroDivisionError):
        return []
    past = []
    for c in state["conditions"]:
        ds = [float(p["D"]) for p in c["populations"]]
        if c.get("kinetics") and not c["kinetics"].get("mock"):
            ds += [float(a["D"]) for a in c["kinetics"]["after"]]
        if max(ds) > limit:
            past.append((c["name"], max(ds)))
    if not past:
        return []
    names = ", ".join(n for n, _d in past)
    fastest = max(d for _n, d in past)
    return [f"{names}: D up to {fastest:g} µm²/s, past what a {LINK_RADIUS_PX:g} px link reads truly at {tau_ms:g} ms "
            f"(about {limit:.2f} µm²/s, within a tenth). Its long steps are lost, so the analysis reads it slower; a "
            f"shorter frame interval or a wider link helps."]
