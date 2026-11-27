"""The generator window's headless side (25 Nov 2026, step 3 of `docs/generator_plan.md`): the form as plain values,
the fields with their words and ranges, the recipe made from it and back, what the window says before a build, and
the rules for adding and removing populations and conditions. The window (`ui.generator.window`) draws this and reads
its controls back; it holds no rule of its own (the thin-Qt rule).

THE STATE is a dict the window edits in place::

    {"fields": {recipe field: value},                       every scalar setting of `generator.Recipe`
     "preset": "clean" | "nikon2" | "nikon1",               the movies' imperfections last applied
     "conditions": [ {"name", "switching": "none" | "exchange" | "matrix", "exchange_per_s", "replicate_sd",
                      "populations": [{"name", "D", "share", "where"}],
                      "matrix": [[rate, ...], ...],
                      "kinetics": None | {"minutes": "-6, -4, ...", "delay_s", "half_time_s",
                                          "onset_sd_replicate_s", "onset_sd_cell_s", "mock",
                                          "after": [{"D", "share"}], "after_matrix": [[...]]}} ]}

No Qt, no numpy beyond what `generator` needs.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, replace

from . import generator as gen

__all__ = ["Field", "FIELDS", "SECTIONS", "SWITCHING", "SWITCHING_WORDS", "WHERE_WORDS", "default_state",
           "warnings", "parse_fractions", "PRESET_FIELDS", "OWN_PRESET", "preset_choices", "add_preset",
           "remove_preset", "preset_from_state",
           "recipe_to_state", "state_to_recipe", "problems", "summary_lines", "parse_minutes", "add_population",
           "remove_population", "add_condition", "remove_condition", "set_kinetics", "apply_preset",
           "visible_sections", "folder_name", "next_stage"]


@dataclass(frozen=True)
class Field:
    """One scalar setting: the `generator.Recipe` field it sets, its words, unit, kind and range."""

    key: str
    section: str
    label: str
    unit: str = ""
    kind: str = "float"            # "float", "int", "choice", "switch", "text"
    low: float = 0.0
    high: float = 1e6
    decimals: int = 2
    choices: tuple = ()            # (value, words) for a choice
    help: str = ""


SECTIONS = ("Cells", "Places in the cell", "Camera and movie", "Replicates", "Output", "Movie imperfections")
FIELDS = (
    Field("cell_length_mean_um", "Cells", "Cell length, mean", "µm", low=0.5, high=20, decimals=2),
    Field("cell_length_sd_um", "Cells", "Cell length, spread (SD)", "µm", low=0, high=5, decimals=2),
    Field("cell_length_range_low", "Cells", "Shortest cell", "µm", low=0.5, high=20, decimals=2),
    Field("cell_length_range_high", "Cells", "Longest cell", "µm", low=0.5, high=20, decimals=2),
    Field("cell_width_um", "Cells", "Cell width", "µm", low=0.2, high=5, decimals=2),
    Field("cells_per_movie", "Cells", "Cells per movie", kind="int", low=1, high=60),
    Field("molecules_per_cell", "Cells", "Molecules per cell", "mean", low=0.5, high=500, decimals=1,
          help="How many molecules each cell shows over the movie."),
    Field("molecules_sd_log", "Cells", "Molecules per cell, spread", "log SD", low=0, high=3, decimals=2),
    Field("dims", "Cells", "Molecules move in", kind="choice", choices=((3, "3D, the cell's volume"),
                                                                       (2, "2D, its projection"))),
    Field("membrane_depth_um", "Places in the cell", "Membrane shell", "µm", low=0.005, high=1, decimals=3),
    Field("pole_depth_um", "Places in the cell", "Poles, depth from each end", "µm", low=0.05, high=5, decimals=2),
    Field("midcell_width_um", "Places in the cell", "Mid-cell band, width", "µm", low=0.05, high=5, decimals=2),
    Field("foci_layout", "Places in the cell", "Foci: where they sit", kind="choice",
          choices=tuple((k, gen.FOCI_LAYOUT_WORDS[k]) for k in gen.FOCI_LAYOUTS),
          help="Foci are binding sites: put a bound or slow population there."),
    Field("foci_depth", "Places in the cell", "Foci: how deep", kind="choice",
          choices=tuple((k, gen.FOCI_DEPTH_WORDS[k]) for k in gen.FOCI_DEPTHS),
          help="On the axis (in focus) or at the membrane. At mid-cell, at the membrane, the foci make a ring."),
    Field("foci_per_cell", "Places in the cell", "Foci per cell (the most)", kind="int", low=1, high=20,
          help="Evenly, at random places and in a ring at mid-cell: how many foci a cell has."),
    Field("foci_per_cell_min", "Places in the cell", "Foci per cell, the fewest (0: always the most)", kind="int",
          low=0, high=20, help="Above 0, each cell draws its own number of foci between this and the most."),
    Field("foci_positions", "Places in the cell", "Foci positions (fractions of the length)", kind="text",
          help="For \"At positions I give\": 0 is one pole, 1 the other; 0.25, 0.75 puts one a quarter in from "
               "each end."),
    Field("focus_radius_um", "Places in the cell", "Focus radius", "µm", low=0.005, high=1, decimals=3),
    Field("pixel_size_nm", "Camera and movie", "Pixel size", "nm", low=10, high=1000, decimals=1),
    Field("frame_interval_ms", "Camera and movie", "Frame interval", "ms", low=0.1, high=10000, decimals=2),
    Field("exposure_ms", "Camera and movie", "Exposure (0: the whole frame)", "ms", low=0, high=10000, decimals=2,
          help="Shorter than the frame interval: the rest of the frame is dead time."),
    Field("n_frames", "Camera and movie", "Frames per movie", kind="int", low=10, high=100000),
    Field("field_px_x", "Camera and movie", "Field width", "px", kind="int", low=16, high=4096),
    Field("field_px_y", "Camera and movie", "Field height", "px", kind="int", low=16, high=4096),
    Field("localisation_error_nm", "Camera and movie", "Localisation error (tracks only)", "nm", low=0, high=500,
          decimals=1, help="Tracks only: in a movie the error follows from the photons, background and PSF."),
    Field("n_subframes", "Camera and movie", "Sub-steps per frame (motion blur)", kind="int", low=1, high=50),
    Field("bleach_mean_frames", "Camera and movie", "Mean frames before bleaching", "frames", low=1, high=10000,
          decimals=1),
    Field("min_localizations", "Camera and movie", "Shortest track kept (tracks only)", "localisations",
          kind="int", low=2, high=1000),
    Field("replicates", "Replicates", "Replicates per condition", kind="int", low=1, high=50),
    Field("movies_per_replicate", "Replicates", "Movies per replicate", kind="int", low=1, high=200),
    Field("seed", "Replicates", "Seed (same seed, same set)", kind="int", low=0, high=2 ** 31 - 1),
    Field("output", "Output", "Build", kind="choice",
          choices=(("tracks", "Tracks (tests the analysis)"),
                   ("movies", "Movies (also tests detection, linking and outlines)"))),
    Field("write_outlines", "Output", "Movies: write the true outlines beside them", kind="switch"),
    Field("name", "Output", "Name of the set", kind="text"),
    Field("camera", "Movie imperfections", "Camera", kind="choice",
          choices=tuple((k, gen.CAMERA_WORDS[k]) for k in gen.CAMERAS)),
    Field("camera_offset_counts", "Movie imperfections", "Own camera: offset", "counts", low=0, high=65535,
          decimals=1, help="Read only when the camera is your own."),
    Field("camera_em_gain", "Movie imperfections", "Own camera: EM gain (1: sCMOS)", low=1, high=10000, decimals=1),
    Field("camera_e_per_count", "Movie imperfections", "Own camera: electrons per count", low=0.001, high=1000,
          decimals=3),
    Field("camera_read_noise_e", "Movie imperfections", "Own camera: read noise", "e-", low=0, high=1000,
          decimals=2),
    Field("emission_nm", "Movie imperfections", "Emission wavelength", "nm", low=300, high=1000, decimals=0),
    Field("numerical_aperture", "Movie imperfections", "Numerical aperture", low=0.1, high=1.7, decimals=2),
    Field("photons_per_spot", "Movie imperfections", "Photons per molecule per frame", low=1, high=1e5, decimals=0),
    Field("background_photons_per_px", "Movie imperfections", "Background", "photons/px", low=0, high=1e4,
          decimals=2),
    Field("background_change", "Movie imperfections", "Background change over the movie", "fraction", low=-0.99,
          high=10, decimals=2, help="-0.3: the background falls by 30 % from the first frame to the last."),
    Field("illumination_sigma_px", "Movie imperfections", "Beam width (0: evenly lit)", "px", low=0, high=1e4,
          decimals=1),
    Field("autofluorescence_photons_per_px", "Movie imperfections", "Autofluorescence in cells", "photons/px",
          low=0, high=1e4, decimals=2),
    Field("autofluorescence_half_life_frames", "Movie imperfections", "Autofluorescence half-life (0: no fading)",
          "frames", low=0, high=1e6, decimals=0),
    Field("autofluorescence_cell_cv", "Movie imperfections", "Autofluorescence, spread between cells", "CV",
          low=0, high=5, decimals=2),
    Field("haze_photons_per_px", "Movie imperfections", "Haze in cells", "photons/px", low=0, high=1e4, decimals=2),
    Field("defocus_depth_um", "Movie imperfections", "Focal depth (0: all in focus)", "µm", low=0, high=10,
          decimals=2),
    Field("blink_off_per_frame", "Movie imperfections", "Blinking: chance to go dark", "per frame", low=0,
          high=0.99, decimals=3),
    Field("blink_on_per_frame", "Movie imperfections", "Blinking: chance to come back", "per frame", low=0.001,
          high=1, decimals=3),
    Field("emission_cv", "Movie imperfections", "Brightness spread", "CV", low=0, high=5, decimals=2),
    Field("emission_corr_frames", "Movie imperfections", "Brightness changes over", "frames", low=0, high=1e4,
          decimals=1),
)
SWITCHING = ("none", "exchange", "matrix")
SWITCHING_WORDS = {"none": "No switching", "exchange": "Shares and an exchange rate",
                   "matrix": "A rate between every pair"}
SWITCHING_NOTES = {
    "none": "Each molecule keeps its population; the shares say how many start in each.",
    "exchange": "Molecules switch so that the shares hold, leaving their population at this rate. Where a "
                "population lives in a place, a short pilot sets the rates so the shares are met.",
    "matrix": "Rate from the population of the row into that of the column, per second. Into a population with "
              "a place, it is the rate while the molecule is inside that place.",
}
WHERE_WORDS = {key: words[0].upper() + words[1:] for key, words in gen._REGION_WORDS.items()}
_SPLIT = {"cell_length_range_low": ("cell_length_range_um", 0), "cell_length_range_high": ("cell_length_range_um", 1),
          "field_px_x": ("field_px", 0), "field_px_y": ("field_px", 1)}


# --------------------------------------------------------------------------- #
# State <-> recipe
# --------------------------------------------------------------------------- #


def _matrix_text(rates) -> list:
    return [[float(x) for x in row] for row in rates]


def recipe_to_state(recipe: gen.Recipe) -> dict:
    fields = {}
    for f in FIELDS:
        if f.key in _SPLIT:
            name, i = _SPLIT[f.key]
            fields[f.key] = getattr(recipe, name)[i]
        elif f.key == "exposure_ms":
            fields[f.key] = 0.0 if recipe.exposure_ms is None else float(recipe.exposure_ms)
        elif f.key == "foci_per_cell_min":
            fields[f.key] = 0 if recipe.foci_per_cell_min is None else int(recipe.foci_per_cell_min)
        elif f.key == "foci_positions":
            fields[f.key] = ", ".join(f"{x:g}" for x in recipe.foci_positions)
        else:
            fields[f.key] = getattr(recipe, f.key)
    conditions = []
    for c in recipe.conditions:
        k = len(c.populations)
        switching = "matrix" if c.rates_per_s is not None else "exchange" if c.exchange_per_s is not None else "none"
        cond = {"name": c.name, "switching": switching,
                "exchange_per_s": float(c.exchange_per_s or 2.0), "replicate_sd": float(c.replicate_sd),
                "populations": [{"name": p.name, "D": float(p.D_um2_s), "share": float(p.share), "where": p.where}
                                for p in c.populations],
                "matrix": (_matrix_text(c.rates_per_s) if c.rates_per_s is not None
                           else [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]),
                "kinetics": None}
        kin = c.kinetics
        if kin is not None:
            after = kin.after_populations or c.populations
            cond["kinetics"] = {
                "minutes": ", ".join(f"{m:g}" for m in kin.movie_minutes), "delay_s": float(kin.delay_s),
                "half_time_s": float(kin.half_time_s), "onset_sd_replicate_s": float(kin.onset_sd_replicate_s),
                "onset_sd_cell_s": float(kin.onset_sd_cell_s), "mock": kin.after_populations is None,
                "after": [{"D": float(p.D_um2_s), "share": float(p.share)} for p in after],
                "after_matrix": (_matrix_text(kin.after_rates_per_s) if kin.after_rates_per_s is not None
                                 else copy.deepcopy(cond["matrix"]))}
        conditions.append(cond)
    #: the recipe's values the form has no field for (the note, a beam's centre from a recipe file), kept as they are
    covered = {_SPLIT.get(f.key, (f.key, 0))[0] for f in FIELDS} | {"conditions"}
    extra = {name: _plain(getattr(recipe, name)) for name in gen.Recipe.__dataclass_fields__ if name not in covered}
    return {"fields": fields, "preset": "nikon2", "conditions": conditions, "extra": extra}


def _plain(value):
    return list(value) if isinstance(value, tuple) else value


def default_state() -> dict:
    """The example recipe, so the window opens on a set that builds as it is."""
    return recipe_to_state(gen.example_recipe())


def parse_minutes(text: str) -> tuple:
    """``"-6, -4, -2, 0.5"`` -> (-6.0, -4.0, -2.0, 0.5). Commas, semicolons or spaces between; a decimal point."""
    parts = [p for p in re.split(r"[,;\s]+", str(text).strip()) if p]
    if not parts:
        raise gen.GeneratorError("Kinetics: give each movie's time after the addition, in minutes.")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise gen.GeneratorError(f"Kinetics: {text!r} is not a list of minutes (write -6, -2, 0.5, 3).") from exc


def parse_fractions(text) -> tuple:
    """``"0.25, 0.75"`` -> (0.25, 0.75); empty -> (). Raises `generator.GeneratorError` on what does not read."""
    if isinstance(text, (list, tuple)):
        return tuple(float(x) for x in text)
    parts = [p for p in re.split(r"[,;\s]+", str(text or "").strip()) if p]
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise gen.GeneratorError(f"Foci positions: {text!r} is not a list of fractions (write 0.25, 0.75).") from exc


def state_to_recipe(state: dict) -> gen.Recipe:
    """The recipe the form describes. Raises `generator.GeneratorError` where the form cannot be read (a list of
    minutes); everything else is `generator.check`'s to judge."""
    f = state["fields"]
    conditions = []
    for c in state["conditions"]:
        pops = tuple(gen.Population(float(p["D"]), float(p["share"]), p["where"], str(p.get("name", "")).strip())
                     for p in c["populations"])
        kin = None
        if c.get("kinetics"):
            k = c["kinetics"]
            after = None if k.get("mock") else tuple(
                gen.Population(float(a["D"]), float(a["share"]), pops[i].where, pops[i].name)
                for i, a in enumerate(k["after"]))
            after_rates = (tuple(tuple(float(x) for x in row) for row in k["after_matrix"])
                           if c["switching"] == "matrix" and not k.get("mock") else None)
            kin = gen.Kinetics(movie_minutes=parse_minutes(k["minutes"]), after_populations=after,
                               after_rates_per_s=after_rates, delay_s=float(k["delay_s"]),
                               half_time_s=float(k["half_time_s"]),
                               onset_sd_replicate_s=float(k["onset_sd_replicate_s"]),
                               onset_sd_cell_s=float(k["onset_sd_cell_s"]))
        conditions.append(gen.Condition(
            name=str(c["name"]).strip(), populations=pops,
            rates_per_s=(tuple(tuple(float(x) for x in row) for row in c["matrix"])
                         if c["switching"] == "matrix" else None),
            exchange_per_s=float(c["exchange_per_s"]) if c["switching"] == "exchange" else None,
            replicate_sd=float(c.get("replicate_sd", 0.0)), kinetics=kin))
    values = {}
    for field in FIELDS:
        if field.key in _SPLIT:
            continue
        value = f[field.key]
        if field.kind == "int":
            value = int(value)
        elif field.kind == "float":
            value = float(value)
        values[field.key] = value
    values["exposure_ms"] = None if not values["exposure_ms"] else values["exposure_ms"]
    values["foci_per_cell_min"] = None if not values["foci_per_cell_min"] else int(values["foci_per_cell_min"])
    values["foci_positions"] = parse_fractions(values["foci_positions"])
    for name, value in (state.get("extra") or {}).items():
        values[name] = tuple(value) if isinstance(value, list) else value
    values["cell_length_range_um"] = (float(f["cell_length_range_low"]), float(f["cell_length_range_high"]))
    values["field_px"] = (int(f["field_px_x"]), int(f["field_px_y"]))
    return gen.Recipe(conditions=tuple(conditions), **values)


def problems(state: dict) -> list:
    """Every reason the form cannot be built, in words (empty when it can)."""
    try:
        recipe = state_to_recipe(state)
    except gen.GeneratorError as exc:
        return [str(exc)]
    except (TypeError, ValueError, KeyError) as exc:
        return [f"The form cannot be read: {exc}"]
    return gen.check(recipe)


def _size_words(n_bytes: int) -> str:
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.1f} GB"
    return f"{n_bytes / 1e6:.0f} MB"


def summary_lines(state: dict) -> list:
    """What the window says beside Build: what will be made, how many movies, and for movies how much is written."""
    try:
        recipe = state_to_recipe(state)
    except gen.GeneratorError:
        return []
    n_movies = recipe.replicates * recipe.movies_per_replicate * len(recipe.conditions)
    lines = [f"{len(recipe.conditions)} condition{'s' if len(recipe.conditions) != 1 else ''}, "
             f"{recipe.replicates} replicate{'s' if recipe.replicates != 1 else ''} of "
             f"{recipe.movies_per_replicate} movies each: {n_movies} movies of {recipe.n_frames} frames at "
             f"{recipe.frame_interval_ms:g} ms."]
    for c in recipe.conditions:
        parts = []
        for i, p in enumerate(c.populations):
            share = "" if c.rates_per_s is not None else f", {100 * p.share:.0f} %"
            parts.append(f"{p.name or f'population {i + 1}'} (D {p.D_um2_s:g}{share}, "
                         f"{gen._REGION_WORDS[p.where]})")
        tail = [] if gen.rates_matrix(c) is None else ["switching"]
        if c.kinetics is not None:
            tail.append("a mock addition" if c.kinetics.after_populations is None else "a change after an addition")
        lines.append(f"{c.name}: " + "; ".join(parts) + (f"; {', '.join(tail)}" if tail else "") + ".")
    if recipe.output == "movies":
        lines.append(f"Movies write about {_size_words(gen.movie_bytes(recipe))}. Track them in Image preparation "
                     f"+ tracking.")
    else:
        lines.append("Tracks are written perfectly linked with the true outlines, ready for the analysis.")
    return lines


def warnings(state: dict) -> list:
    """What the window says beside the problems: choices that build but may not be what was meant (a fast
    population at foci, `generator.foci_warnings`)."""
    try:
        recipe = state_to_recipe(state)
    except gen.GeneratorError:
        return []
    return gen.foci_warnings(recipe)


#: The fields a movie preset sets: the camera (own values included), the photons and every imperfection.
PRESET_FIELDS = tuple(sorted(set(gen.realism_preset("clean")) | {
    "camera_offset_counts", "camera_em_gain", "camera_e_per_count", "camera_read_noise_e", "emission_nm",
    "numerical_aperture"}))
#: A preset of the user's own is named with this mark in front, so it cannot take a built-in's name.
OWN_PRESET = "mine:"


def preset_from_state(state: dict) -> dict:
    return {key: state["fields"][key] for key in PRESET_FIELDS if key in state["fields"]}


def preset_choices(library: dict | None) -> list:
    """``[(key, words)]``: the built-in presets, then the user's own (``OWN_PRESET + name``), alphabetically."""
    out = [(k, gen.REALISM_WORDS[k]) for k in gen.REALISM_PRESETS]
    out += [(OWN_PRESET + name, name) for name in sorted(library or {}, key=str.lower)]
    return out


def add_preset(library: dict | None, name: str, state: dict) -> dict:
    """The library with the form's movie values saved under ``name`` (a name already there is replaced)."""
    name = str(name).strip()
    if not name:
        raise gen.GeneratorError("Give the preset a name.")
    if name.lower() in {w.lower() for w in gen.REALISM_WORDS.values()} | set(gen.REALISM_PRESETS):
        raise gen.GeneratorError(f"{name!r} is a built-in preset's name; choose another.")
    out = dict(library or {})
    out[name] = preset_from_state(state)
    return out


def remove_preset(library: dict | None, key: str) -> dict:
    out = dict(library or {})
    out.pop(key[len(OWN_PRESET):] if key.startswith(OWN_PRESET) else key, None)
    return out


def visible_sections(state: dict) -> tuple:
    """The sections the form shows: the movie imperfections only when movies are built."""
    movies = state["fields"].get("output") == "movies"
    return tuple(s for s in SECTIONS if movies or s != "Movie imperfections")


def folder_name(state: dict) -> str:
    """A folder name from the set's name: letters, digits, spaces, hyphens and underscores kept."""
    name = re.sub(r"[^A-Za-z0-9 _.-]+", " ", str(state["fields"].get("name", ""))).strip(" .")
    return re.sub(r"\s+", " ", name)[:80] or "simulated set"


def next_stage(state: dict) -> str:
    """Where a built set opens: the analysis for tracks, image preparation + tracking for movies."""
    return "track" if state["fields"].get("output") == "movies" else "analyse"


# --------------------------------------------------------------------------- #
# Editing: populations, conditions, kinetics, presets
# --------------------------------------------------------------------------- #


def _resize(matrix: list, k: int) -> list:
    """The k x k matrix: the old entries kept, a new row or column 0 on the diagonal and 1 per s elsewhere."""
    out = []
    for i in range(k):
        row = []
        for j in range(k):
            if i < len(matrix) and j < len(matrix[i]):
                row.append(float(matrix[i][j]))
            else:
                row.append(0.0 if i == j else 1.0)
        out.append(row)
    return out


def _even_shares(pops: list) -> None:
    share = round(1.0 / len(pops), 4)
    for p in pops:
        p["share"] = share
    pops[-1]["share"] = round(1.0 - share * (len(pops) - 1), 4)


def add_population(cond: dict) -> None:
    """One more population (a free one, D 1, anywhere in the cell); the shares made even, the matrices grown."""
    if len(cond["populations"]) >= gen.MAX_POPULATIONS:
        raise gen.GeneratorError(f"At most {gen.MAX_POPULATIONS} populations.")
    cond["populations"].append({"name": "", "D": 1.0, "share": 0.0, "where": "cell"})
    _even_shares(cond["populations"])
    k = len(cond["populations"])
    cond["matrix"] = _resize(cond["matrix"], k)
    if cond.get("kinetics"):
        cond["kinetics"]["after"].append({"D": 1.0, "share": 0.0})
        _even_shares(cond["kinetics"]["after"])
        cond["kinetics"]["after_matrix"] = _resize(cond["kinetics"]["after_matrix"], k)


def remove_population(cond: dict, index: int) -> None:
    """Remove one population (never the last); the shares made even, its row and column taken out."""
    if len(cond["populations"]) <= 1:
        raise gen.GeneratorError("A condition needs at least one population.")
    del cond["populations"][index]
    _even_shares(cond["populations"])
    cond["matrix"] = [[x for j, x in enumerate(row) if j != index] for i, row in enumerate(cond["matrix"])
                      if i != index]
    if len(cond["populations"]) == 1:
        cond["switching"] = "none"
    if cond.get("kinetics"):
        del cond["kinetics"]["after"][index]
        _even_shares(cond["kinetics"]["after"])
        cond["kinetics"]["after_matrix"] = [[x for j, x in enumerate(row) if j != index]
                                            for i, row in enumerate(cond["kinetics"]["after_matrix"]) if i != index]


def add_condition(state: dict, copy_of: int | None = None) -> int:
    """A new condition, a copy of ``copy_of`` (default the last) under a new name; returns its index."""
    source = state["conditions"][-1 if copy_of is None else copy_of]
    new = copy.deepcopy(source)
    names = {c["name"] for c in state["conditions"]}
    stem, n = re.sub(r"\s+\d+$", "", source["name"]) or "condition", 2
    while f"{stem} {n}" in names:
        n += 1
    new["name"] = f"{stem} {n}"
    state["conditions"].append(new)
    return len(state["conditions"]) - 1


def remove_condition(state: dict, index: int) -> None:
    if len(state["conditions"]) <= 1:
        raise gen.GeneratorError("A set needs at least one condition.")
    del state["conditions"][index]


def set_kinetics(cond: dict, on: bool, movies_per_replicate: int) -> None:
    """Kinetics on: a schedule of ``movies_per_replicate`` movies (three before the addition, the rest after, as
    SynK), the populations after equal to before (change them to make an effect); off: none."""
    if not on:
        cond["kinetics"] = None
        return
    n = int(movies_per_replicate)
    before = max(1, min(3, n // 3))
    after_times = [0.5, 1.5, 3, 5, 7, 10, 14, 18, 22, 26, 30]
    minutes = [-2.0 * (before - i) for i in range(before)] + [float(t) for t in after_times[:n - before]]
    while len(minutes) < n:
        minutes.append(minutes[-1] + 4.0)
    cond["kinetics"] = {"minutes": ", ".join(f"{m:g}" for m in minutes), "delay_s": 20.0, "half_time_s": 120.0,
                        "onset_sd_replicate_s": 10.0, "onset_sd_cell_s": 15.0, "mock": False,
                        "after": [{"D": float(p["D"]), "share": float(p["share"])} for p in cond["populations"]],
                        "after_matrix": copy.deepcopy(cond["matrix"])}


def apply_preset(state: dict, name: str, library: dict | None = None) -> None:
    """A preset's movie values written into the fields: a built-in one (`generator.realism_preset`) or one of the
    user's own (``OWN_PRESET + name`` in ``library``)."""
    if name.startswith(OWN_PRESET):
        values = (library or {}).get(name[len(OWN_PRESET):])
        if values is None:
            raise gen.GeneratorError(f"No preset named {name[len(OWN_PRESET):]!r}.")
    else:
        values = gen.realism_preset(name)
    for key, value in values.items():
        if key in state["fields"]:
            state["fields"][key] = value
    state["preset"] = name


def with_folder_check(state: dict, folder) -> list:
    """`problems` plus the output folder's: it must be chosen and must not exist."""
    from pathlib import Path

    out = problems(state)
    if not folder:
        out.append("Choose where the set is written.")
    elif Path(folder).exists():
        out.append(f"{folder} exists: choose a new folder (a set is never written over one).")
    return out
