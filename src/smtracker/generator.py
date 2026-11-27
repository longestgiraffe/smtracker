"""Simulated data sets from a RECIPE: any number of diffusive populations, their D, shares, places in the cell and
switching rates, in capsule cells of a set size, imaged at a set pixel size and frame interval (25 Nov 2026).

Asked: "a generator would be cool. where u can set how many diffusive populations, which D, etc, which cell size
etc.", then switching rates, the camera's resolution, pixel size and field, cell width, a kinetic option, and
"where in the cells which population occurs". The plan, the rules and the build order are `docs/generator_plan.md`;
this module is step 1, the engine: the recipe, its checks, the simulation, and the project written as the analysis
reads it, with its truth. The paper's own sets (SynR, SynG, SynK, free mNG) stay with their scripts
(`scripts/synthetic_builds/`); the generator is a new door to the same machinery (`synthetic.place_cells`, the
capsule cells and their nucleoid lobes, `synthetic.write_trackmate_xml`, the outline writer).

THE MODEL
---------
- Cells: capsules (`synthetic.CapsuleCell`), length drawn from a clipped normal, one width, placed without overlap
  in the camera field (`synthetic.place_cells`, which refuses a field too crowded), with the nucleoid lobes of
  `synthetic.nucleoid_lobes` (one lobe below 3.8 um, two above).
- Populations: each has a D (0 = immobile), a share, and WHERE it lives (`REGIONS`): anywhere in the cell, the
  nucleoid, a shell under the membrane, the poles, a band at mid-cell, or foci (sites spread along the axis). A
  molecule of a population is confined to its region: a micro-step (`synthetic.MICRO_STEP_UM`, 20 nm, as the
  paper's sets) that would leave it is not taken, which keeps the density even over any region's shape.
- Switching: none (a molecule keeps its population; the shares say how many start in each), or a rate (per s) for
  every pair, propagated with the exact sub-step matrix ``expm(Q dt)`` (`simulate.ConditionSpec`). A molecule can
  switch INTO a population that lives in a region only while it is inside that region (SynR's rule: a free molecule
  starts scanning only inside the nucleoid); the switch is otherwise not made. The declared rate is the rate inside
  the region; the realised rates and shares are COUNTED and written as the truth.
- The camera: each frame is the average of the exposed sub-steps' positions (motion blur) plus a Gaussian
  localisation error; a molecule appears at a random frame and bleaches after a geometric number of frames;
  tracks shorter than ``min_localizations`` are not written; the truth keeps each frame's instantaneous position and
  state (ruled 30 Oct 2026: the truth has no blur and no localisation error).
- Replicates are independent (Rule 0): each draws its own factor, lognormal with ``replicate_sd``, on the rates
  INTO the first population (with switching) or on the odds of the first population's share (without).
- 3D in the cell's volume by default (``dims`` 3), as the paper's sets; ``dims`` 2 is the projection.

WHAT IS WRITTEN (a new folder; refused if it exists)
----------------------------------------------------
``<root>/recipe.json`` (what was asked: rebuilding from it gives the same set, draw for draw);
``<root>/<condition>/trackmate/seriesNNN.xml`` and ``cell_meshes/seriesNNN.mat`` (the true capsules, drawn
`synthetic.OUTLINE_MARGIN_SIGMA` localisation errors wider, as the paper's sets); ``day_map.json`` (the replicates);
``bleaching/crops.json`` (pixel size, frame interval, every frame used); ``<root>/truth/truth.json`` (declared and
counted, per condition and replicate) and ``truth/<condition>_localisations.csv.gz`` (every localisation's observed
and true position, true population and D).
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

__all__ = [
    "GeneratorError",
    "REGIONS",
    "Population",
    "Kinetics",
    "Condition",
    "Recipe",
    "check",
    "rates_matrix",
    "build",
    "load_recipe",
    "save_recipe",
    "example_recipe",
    "response",
    "SIMULATED_STARTS_FILE",
    "CAMERAS",
    "REALISM_PRESETS",
    "realism_preset",
    "movie_bytes",
    "FOCI_LAYOUTS",
    "FOCI_DEPTHS",
    "foci_sites",
    "foci_warnings",
    "camera_calibration",
]

#: Where a population may live. "cell": anywhere inside the capsule; "nucleoid": inside the nucleoid lobes;
#: "membrane": a shell of ``membrane_depth_um`` under the membrane; "poles": within ``pole_depth_um`` of either end;
#: "midcell": a band ``midcell_width_um`` wide across the middle; "foci": within ``focus_radius_um`` of the cell's
#: foci, BINDING SITES for a bound or slow population (a fast one only jitters inside its sphere and reads D near 0):
#: where they sit is ``foci_layout`` (`FOCI_LAYOUTS`), how many ``foci_per_cell`` (between ``foci_per_cell_min``
#: and it, drawn per cell, where the layout counts), at what depth ``foci_depth`` (`FOCI_DEPTHS`).
REGIONS = ("cell", "nucleoid", "membrane", "poles", "midcell", "foci")
_REGION_WORDS = {"cell": "anywhere in the cell", "nucleoid": "in the nucleoid", "membrane": "at the membrane",
                 "poles": "at the poles", "midcell": "at mid-cell", "foci": "at foci"}
MAX_POPULATIONS = 5
#: Where the foci sit (25 Nov 2026, asked: "can we decide where the foci is?"):
#:   even      spread evenly along the axis (the first build's only layout: an old recipe builds as before)
#:   nucleoid  at the nucleoid lobes, two per lobe at its centre +- `synthetic.FORK_HALF_SPACING_UM` (SynG's
#:             replication forks, `synthetic.CapsuleCell.fork_sites`)
#:   poles     one at each pole, inside its cap (at the membrane: the pole's tip)
#:   midcell   at mid-cell (at the membrane: a ring of ``foci_per_cell``, like a division ring)
#:   random    anywhere along the axis (at the membrane: at random angles), drawn per cell
#:   positions at the fractions of the length in ``foci_positions`` (0 one pole, 1 the other)
FOCI_LAYOUTS = ("even", "nucleoid", "poles", "midcell", "random", "positions")
FOCI_LAYOUT_WORDS = {"even": "Evenly along the axis", "nucleoid": "At the nucleoid (replication forks)",
                     "poles": "At the poles", "midcell": "At mid-cell", "random": "At random places",
                     "positions": "At positions I give"}
#: How deep: on the cell's axis (mid-depth, in the focal plane) or at the membrane.
FOCI_DEPTHS = ("axis", "membrane")
FOCI_DEPTH_WORDS = {"axis": "On the axis", "membrane": "At the membrane"}
#: Above this D a population at foci is warned about (`foci_warnings`): it only jitters inside its sphere.
FOCI_FAST_D = 0.1


class GeneratorError(ValueError):
    """A recipe that cannot be built, named before anything is written."""


@dataclass(frozen=True)
class Population:
    """One diffusive population: its D (um2/s, 0 = immobile), its share, where it lives, and a name."""

    D_um2_s: float
    share: float
    where: str = "cell"
    name: str = ""


@dataclass(frozen=True)
class Kinetics:
    """A change after an addition (step 2 of `docs/generator_plan.md`; the paper's SynK, `synthetic.Induction`).

    Each movie of a replicate is recorded at its own time on the addition's clock: ``movie_minutes``, one per movie,
    negative before the addition (the baseline). After ``delay_s`` every parameter moves from its value before to its
    value after along ``h(t) = 1 - 2^-((t - delay) / half_time)`` (`response`): each population's D, its share
    (without switching: the share of molecules appearing at that time), and the rates (with switching). The onset
    varies between replicates and between cells (normal SDs, s), so the population's response is smoother than any
    one cell's, as in a real culture. ``after_populations``: the populations after (the same number, each in the same
    place; D and shares may change); None is a MOCK, an addition that changes nothing, imaged on the same schedule.
    ``after_rates_per_s``: the rate matrix after, for a condition given as a matrix."""

    movie_minutes: tuple
    after_populations: tuple | None = None
    after_rates_per_s: tuple | None = None
    delay_s: float = 20.0
    half_time_s: float = 120.0
    onset_sd_replicate_s: float = 10.0
    onset_sd_cell_s: float = 15.0


def response(t, delay_s: float, half_time_s: float) -> np.ndarray:
    """h(t): 0 until the delay, then 1 - 2^-((t - delay) / half-time), so half the change is made one half-time after
    the delay. ``t`` in seconds since the addition."""
    t = np.asarray(t, dtype=float)
    return np.where(t < delay_s, 0.0, 1.0 - np.power(0.5, np.clip(t - delay_s, 0.0, None) / half_time_s))


#: Each movie's simulated recording start, written beside a kinetic condition's tracks: the Kinetics section reads
#: it where a movie has no camera file (`induction_view.camera_starts`), so the clock since the addition is the
#: simulation's own and never guessed.
SIMULATED_STARTS_FILE = "simulated_movie_starts.json"
#: The levels of h the switching matrices are computed at (h rounded to the nearest), and the clock's first day.
KINETIC_LEVELS = 21


@dataclass(frozen=True)
class Condition:
    """One condition: its populations and, optionally, the switching between them.

    ``rates_per_s``: a k x k matrix, entry [i][j] the rate from population i to population j (per s; the diagonal
    is ignored). Without it each molecule keeps its population and the shares say how many start in each. With it
    the shares are ignored: the rates set the occupancy (`rates_matrix`). ``exchange_per_s``: instead of a matrix,
    the total rate at which molecules leave their population, from which the rates that hold the declared shares
    follow (`simulate.rates_for_occupancy`)."""

    name: str
    populations: tuple
    rates_per_s: tuple | None = None
    exchange_per_s: float | None = None
    replicate_sd: float = 0.0
    kinetics: Kinetics | None = None


@dataclass(frozen=True)
class Recipe:
    """Everything a simulated set is made from. Units in the names; a recipe is saved as JSON (`save_recipe`)."""

    conditions: tuple
    # the design
    replicates: int = 3
    movies_per_replicate: int = 10
    cells_per_movie: int = 6
    molecules_per_cell: float = 16.0
    molecules_sd_log: float = 0.0
    # the cells
    cell_length_mean_um: float = 3.6
    cell_length_sd_um: float = 0.5
    cell_length_range_um: tuple = (2.6, 5.0)
    cell_width_um: float = 0.9
    membrane_depth_um: float = 0.05
    pole_depth_um: float = 0.4
    midcell_width_um: float = 0.6
    foci_per_cell: int = 2
    focus_radius_um: float = 0.05
    foci_layout: str = "even"
    #: None: every cell has ``foci_per_cell``; a number: each cell draws its count between it and ``foci_per_cell``
    foci_per_cell_min: int | None = None
    foci_depth: str = "axis"
    #: layout "positions": the fractions of the cell's length the foci sit at (0 one pole, 1 the other)
    foci_positions: tuple = ()
    dims: int = 3
    # the camera
    pixel_size_nm: float = 106.0
    frame_interval_ms: float = 20.0
    exposure_ms: float | None = None
    n_frames: int = 1000
    field_px: tuple = (128, 128)
    localisation_error_nm: float = 25.0
    n_subframes: int = 5
    bleach_mean_frames: float = 15.0
    min_localizations: int = 5
    seed: int = 1
    name: str = "simulated set"
    note: str = ""
    # THE OUTPUT (step 4): "tracks" (perfectly linked, with the true outlines: tests the analysis) or "movies"
    # (camera movies and phase images for the preparation stage: tests detection, linking and segmentation too)
    output: str = "tracks"
    #: movies: write the true outlines beside them, so segmentation may be skipped (the exact capsules are always in
    #: truth/outlines/ to score a segmentation against)
    write_outlines: bool = True
    # THE MOVIES' CAMERA, OPTICS AND IMPERFECTIONS (`REALISM_PRESETS` fills them; read only for output "movies").
    # In a movie the localisation error is not set: it follows from the photons, the background and the PSF.
    camera: str = "nikon2"
    emission_nm: float = 517.0
    numerical_aperture: float = 1.49
    photons_per_spot: float = 127.0
    background_photons_per_px: float = 4.81
    illumination_sigma_px: float = 0.0          # 0: evenly lit; else a Gaussian beam of this width (px)
    illumination_centre_px: tuple | None = None  # None: the field's centre
    autofluorescence_photons_per_px: float = 0.0
    autofluorescence_half_life_frames: float = 0.0   # 0: it does not fade
    autofluorescence_cell_cv: float = 0.0
    background_change: float = 0.0              # the background's change from the first to the last frame (-0.3: falls 30 %)
    #: camera "own": the camera's offset (counts), EM gain (1: an sCMOS, no EM register), electrons per count and
    #: read noise (electrons)
    camera_offset_counts: float = 100.0
    camera_em_gain: float = 1.0
    camera_e_per_count: float = 0.5
    camera_read_noise_e: float = 1.5
    haze_photons_per_px: float = 0.0
    defocus_depth_um: float = 0.0               # 0: every molecule in focus
    blink_off_per_frame: float = 0.0
    blink_on_per_frame: float = 0.4
    emission_cv: float = 0.0
    emission_corr_frames: float = 2.0

    @property
    def tau_s(self) -> float:
        return self.frame_interval_ms / 1000.0

    @property
    def field_um(self) -> tuple:
        return tuple(n * self.pixel_size_nm / 1000.0 for n in self.field_px)

    @property
    def exposed_subframes(self) -> int:
        if self.exposure_ms is None:
            return int(self.n_subframes)
        return int(round(self.n_subframes * self.exposure_ms / self.frame_interval_ms))


# --------------------------------------------------------------------------- #
# The recipe: checks, rates, JSON
# --------------------------------------------------------------------------- #


#: The measured cameras a movie is rendered with (`synthetic.MOVIE_CALIBRATION`, `MOVIE_CALIBRATION_N1`): offset,
#: EM gain, electrons per count, read noise. Their photon budgets fill the presets below.
def _cameras() -> dict:
    from .synthetic import MOVIE_CALIBRATION, MOVIE_CALIBRATION_N1

    return {"nikon2": MOVIE_CALIBRATION, "nikon1": MOVIE_CALIBRATION_N1}


CAMERAS = ("nikon2", "nikon1", "own")
CAMERA_WORDS = {"nikon2": "Nikon 2 (EMCCD, measured)", "nikon1": "Nikon 1 (EMCCD, measured)",
                "own": "My own camera (the values below)"}


def camera_calibration(recipe: "Recipe") -> dict:
    """The camera a movie is rendered with: a measured one (`_cameras`), or the recipe's own values."""
    if recipe.camera == "own":
        return {"offset_counts": float(recipe.camera_offset_counts), "em_gain": float(recipe.camera_em_gain),
                "sensitivity_e_per_count": float(recipe.camera_e_per_count),
                "read_noise_e": float(recipe.camera_read_noise_e), "source": "the recipe's own camera"}
    return _cameras()[recipe.camera]


def _preset_fields() -> dict:
    from .synthetic import MOVIE_CALIBRATION, MOVIE_CALIBRATION_N1, SYNG_N1_REALISM, SYNR_V3_REALISM

    def from_realism(r, cal, camera):
        return {"camera": camera, "photons_per_spot": cal["photons_per_frame"],
                "background_photons_per_px": cal["background_photons_per_px"],
                "illumination_sigma_px": 0.0 if r.illumination_sigma_px >= 1000 else r.illumination_sigma_px,
                "autofluorescence_photons_per_px": r.autofluorescence_photons_per_px,
                "autofluorescence_half_life_frames": float(r.autofluorescence_half_life_frames or 0.0),
                "autofluorescence_cell_cv": r.autofluorescence_cell_cv, "haze_photons_per_px": r.haze_photons_per_px,
                "defocus_depth_um": r.defocus_depth_um, "blink_off_per_frame": r.blink_off_per_frame,
                "blink_on_per_frame": r.blink_on_per_frame, "emission_cv": r.emission_cv,
                "emission_corr_frames": r.emission_corr_frames, "background_change": 0.0}

    clean = {**from_realism(SYNR_V3_REALISM, MOVIE_CALIBRATION, "nikon2"),
             "illumination_sigma_px": 0.0, "autofluorescence_photons_per_px": 0.0,
             "autofluorescence_half_life_frames": 0.0, "autofluorescence_cell_cv": 0.0, "haze_photons_per_px": 0.0,
             "defocus_depth_um": 0.0, "blink_off_per_frame": 0.0, "emission_cv": 0.0}
    return {"clean": clean, "nikon2": from_realism(SYNR_V3_REALISM, MOVIE_CALIBRATION, "nikon2"),
            "nikon1": from_realism(SYNG_N1_REALISM, MOVIE_CALIBRATION_N1, "nikon1")}


#: The presets a movie's imperfections start from: none, or as measured on the lab's two microscopes
#: (`synthetic.SYNR_V3_REALISM`, `SYNG_N1_REALISM`; their sources are written there). Every value stays editable.
REALISM_PRESETS = ("clean", "nikon2", "nikon1")
REALISM_WORDS = {"clean": "Clean: a camera's noise only", "nikon2": "Nikon 2, as measured",
                 "nikon1": "Nikon 1, as measured"}


def realism_preset(name: str) -> dict:
    """The recipe fields a preset sets (camera, photons, background and every imperfection)."""
    presets = _preset_fields()
    if name not in presets:
        raise GeneratorError(f"no preset {name!r}; have {', '.join(REALISM_PRESETS)}")
    return dict(presets[name])


def movie_bytes(recipe: Recipe) -> int:
    """How much a set of movies writes (the camera files, uint16), to say before it is built."""
    one = recipe.field_px[0] * recipe.field_px[1] * recipe.n_frames * 2
    return one * recipe.replicates * recipe.movies_per_replicate * len(recipe.conditions)


def rates_matrix(condition: Condition) -> np.ndarray | None:
    """The condition's rate matrix (per s), or None without switching."""
    k = len(condition.populations)
    if condition.rates_per_s is not None:
        rates = np.array(condition.rates_per_s, dtype=float)
        np.fill_diagonal(rates, 0.0)
        return rates
    if condition.exchange_per_s is not None and k > 1:
        from .simulate import rates_for_occupancy

        return rates_for_occupancy([p.share for p in condition.populations], float(condition.exchange_per_s))
    return None


def check(recipe: Recipe) -> list:
    """Every reason the recipe cannot be built, in words; empty when it can."""
    problems = []
    if not recipe.conditions:
        problems.append("The recipe has no condition.")
    names = [c.name for c in recipe.conditions]
    if len(set(names)) != len(names):
        problems.append("Two conditions have the same name.")
    for c in recipe.conditions:
        label = f"Condition {c.name!r}"
        if not c.name or any(ch in c.name for ch in '\\/:*?"<>|'):
            problems.append(f"{label}: a name must be set and usable as a folder name.")
        k = len(c.populations)
        if not 1 <= k <= MAX_POPULATIONS:
            problems.append(f"{label}: between 1 and {MAX_POPULATIONS} populations, not {k}.")
            continue
        for i, p in enumerate(c.populations, 1):
            if not (np.isfinite(p.D_um2_s) and p.D_um2_s >= 0):
                problems.append(f"{label}, population {i}: D must be 0 or more.")
            if p.where not in REGIONS:
                problems.append(f"{label}, population {i}: 'where' must be one of {', '.join(REGIONS)}.")
            if not 0 <= p.share <= 1:
                problems.append(f"{label}, population {i}: its share must lie between 0 and 1.")
        if c.rates_per_s is None and abs(sum(p.share for p in c.populations) - 1.0) > 1e-6:
            problems.append(f"{label}: the shares add up to {sum(p.share for p in c.populations):.3f}, not 1.")
        if c.rates_per_s is not None and c.exchange_per_s is not None:
            problems.append(f"{label}: give a rate matrix or an exchange rate, not both.")
        if c.rates_per_s is not None:
            rates = np.asarray(c.rates_per_s, dtype=float)
            if rates.shape != (k, k):
                problems.append(f"{label}: the rate matrix must be {k} x {k}, one row and column per population.")
            elif np.any(rates < 0) or not np.all(np.isfinite(rates)):
                problems.append(f"{label}: rates must be 0 or more.")
            elif k > 1 and np.allclose(rates - np.diag(np.diag(rates)), 0):
                problems.append(f"{label}: every rate is 0; for no switching leave the rates out.")
        if c.exchange_per_s is not None and not c.exchange_per_s > 0:
            problems.append(f"{label}: the exchange rate must be above 0.")
        if (c.rates_per_s is not None or c.exchange_per_s is not None) and k == 1:
            problems.append(f"{label}: one population cannot switch; leave the rates out.")
        if c.replicate_sd < 0:
            problems.append(f"{label}: the spread between replicates must be 0 or more.")
        kin = c.kinetics
        if kin is not None:
            if len(kin.movie_minutes) != recipe.movies_per_replicate:
                problems.append(f"{label}: give one time per movie ({recipe.movies_per_replicate}), not "
                                f"{len(kin.movie_minutes)}.")
            if not kin.half_time_s > 0 or kin.delay_s < 0:
                problems.append(f"{label}: the half-time must be above 0 and the delay 0 or more.")
            if kin.onset_sd_replicate_s < 0 or kin.onset_sd_cell_s < 0:
                problems.append(f"{label}: the spreads of the onset must be 0 or more.")
            movie_s = recipe.n_frames * recipe.frame_interval_ms / 1000.0
            minutes = sorted(kin.movie_minutes)
            if any(b - a < movie_s / 60.0 for a, b in zip(minutes, minutes[1:])):
                problems.append(f"{label}: two movies overlap in time (each lasts {movie_s:g} s).")
            if kin.after_populations is not None:
                after = kin.after_populations
                if len(after) != k:
                    problems.append(f"{label}: after the addition there must be the same {k} populations.")
                else:
                    if any(a.where != b.where for a, b in zip(after, c.populations)):
                        problems.append(f"{label}: a population keeps its place in the cell after the addition.")
                    if any(not (np.isfinite(a.D_um2_s) and a.D_um2_s >= 0) for a in after):
                        problems.append(f"{label}: every D after the addition must be 0 or more.")
                    if c.rates_per_s is None and abs(sum(a.share for a in after) - 1.0) > 1e-6:
                        problems.append(f"{label}: the shares after the addition add up to "
                                        f"{sum(a.share for a in after):.3f}, not 1.")
            if c.rates_per_s is not None and kin.after_populations is not None:
                after_rates = None if kin.after_rates_per_s is None else np.asarray(kin.after_rates_per_s, float)
                if after_rates is None or after_rates.shape != (k, k) or np.any(after_rates < 0):
                    problems.append(f"{label}: a condition given as a rate matrix needs the {k} x {k} rate matrix "
                                    f"after the addition.")
    lo, hi = recipe.cell_length_range_um
    for value, words in ((recipe.replicates, "replicates"), (recipe.movies_per_replicate, "movies per replicate"),
                         (recipe.cells_per_movie, "cells per movie"), (recipe.n_frames, "frames"),
                         (recipe.n_subframes, "sub-steps per frame"), (recipe.foci_per_cell, "foci per cell")):
        if int(value) < 1 or int(value) != value:
            problems.append(f"The number of {words} must be a whole number of at least 1.")
    if not recipe.molecules_per_cell > 0:
        problems.append("Molecules per cell must be above 0.")
    if not recipe.cell_width_um > 0 or not 0 < lo <= recipe.cell_length_mean_um <= hi:
        problems.append("The cell width must be above 0, and the mean length must lie inside the length range.")
    if lo < recipe.cell_width_um:
        problems.append("The shortest cell must be at least as long as it is wide.")
    for value, words in ((recipe.pixel_size_nm, "pixel size"), (recipe.frame_interval_ms, "frame interval"),
                         (recipe.bleach_mean_frames, "mean bleaching time")):
        if not value > 0:
            problems.append(f"The {words} must be above 0.")
    if recipe.localisation_error_nm < 0:
        problems.append("The localisation error must be 0 or more.")
    if recipe.exposure_ms is not None:
        exact = recipe.n_subframes * recipe.exposure_ms / recipe.frame_interval_ms
        if abs(exact - round(exact)) > 1e-6 or not 1 <= round(exact) <= recipe.n_subframes:
            problems.append(f"An exposure of {recipe.exposure_ms:g} ms in a {recipe.frame_interval_ms:g} ms frame "
                            f"must cover a whole number of the {recipe.n_subframes} sub-steps.")
    if recipe.dims not in (2, 3):
        problems.append("dims must be 2 or 3.")
    if recipe.min_localizations < 2:
        problems.append("A track needs at least 2 localisations.")
    if len(recipe.field_px) != 2 or min(recipe.field_px) < 16:
        problems.append("The field must be at least 16 x 16 pixels.")
    elif min(recipe.field_um) < hi + 1.0:
        problems.append(f"The field ({recipe.field_um[0]:.1f} x {recipe.field_um[1]:.1f} um) is too small for "
                        f"cells up to {hi:g} um long.")
    for value, words in ((recipe.membrane_depth_um, "membrane depth"), (recipe.pole_depth_um, "pole depth"),
                         (recipe.midcell_width_um, "mid-cell band"), (recipe.focus_radius_um, "focus radius")):
        if not value > 0:
            problems.append(f"The {words} must be above 0.")
    if recipe.membrane_depth_um >= recipe.cell_width_um / 2:
        problems.append("The membrane shell must be thinner than half the cell's width.")
    if recipe.foci_layout not in FOCI_LAYOUTS:
        problems.append(f"Foci: the layout must be one of {', '.join(FOCI_LAYOUTS)}.")
    if recipe.foci_depth not in FOCI_DEPTHS:
        problems.append("Foci: the depth must be the axis or the membrane.")
    if recipe.foci_per_cell_min is not None and not 1 <= int(recipe.foci_per_cell_min) <= int(recipe.foci_per_cell):
        problems.append("Foci: the fewest per cell must be at least 1 and at most the most per cell.")
    if recipe.foci_layout == "positions":
        if not recipe.foci_positions:
            problems.append("Foci: give the positions along the cell (fractions of its length, 0 to 1).")
        elif any(not 0.0 <= float(f) <= 1.0 for f in recipe.foci_positions):
            problems.append("Foci: every position is a fraction of the cell's length, from 0 to 1.")
    if 2 * recipe.focus_radius_um >= recipe.cell_width_um:
        problems.append("Foci: a focus must be narrower than the cell.")
    if recipe.output not in ("tracks", "movies"):
        problems.append("The output must be tracks or movies.")
    if recipe.output == "movies":
        if recipe.camera not in CAMERAS:
            problems.append(f"The camera must be one of {', '.join(CAMERAS)}.")
        if recipe.camera == "own" and not (recipe.camera_offset_counts >= 0 and recipe.camera_em_gain >= 1
                                           and recipe.camera_e_per_count > 0 and recipe.camera_read_noise_e >= 0):
            problems.append("Own camera: the offset and read noise must be 0 or more, the EM gain at least 1 and the "
                            "electrons per count above 0.")
        if not recipe.photons_per_spot > 0 or not recipe.numerical_aperture > 0 or not recipe.emission_nm > 0:
            problems.append("The photons per spot, the numerical aperture and the wavelength must be above 0.")
        for value, words in ((recipe.background_photons_per_px, "background"),
                             (recipe.autofluorescence_photons_per_px, "autofluorescence"),
                             (recipe.haze_photons_per_px, "haze"), (recipe.illumination_sigma_px, "beam width"),
                             (recipe.autofluorescence_half_life_frames, "fading of the autofluorescence"),
                             (recipe.defocus_depth_um, "focal depth"), (recipe.emission_cv, "brightness spread"),
                             (recipe.autofluorescence_cell_cv, "autofluorescence spread")):
            if value < 0:
                problems.append(f"The {words} must be 0 or more.")
        if not 0 <= recipe.blink_off_per_frame < 1 or not 0 < recipe.blink_on_per_frame <= 1:
            problems.append("Blinking: the chance to go dark must lie in [0, 1), to come back in (0, 1].")
        if recipe.background_change <= -1:
            problems.append("The background cannot fall by 100 % or more.")
    if 2 * recipe.pole_depth_um >= lo:
        problems.append("The two poles overlap in the shortest cell: make the pole depth smaller.")
    return problems


def _to_plain(obj):
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def recipe_to_dict(recipe: Recipe) -> dict:
    return {"format": "smtracker generator recipe 1", **_to_plain(asdict(recipe))}


def recipe_from_dict(data: dict) -> Recipe:
    data = dict(data)
    data.pop("format", None)
    conditions = []
    for c in data.pop("conditions"):
        c = dict(c)
        pops = tuple(Population(**p) for p in c.pop("populations"))
        rates = c.pop("rates_per_s", None)
        kin = c.pop("kinetics", None)
        if kin is not None:
            kin = dict(kin)
            after = kin.pop("after_populations", None)
            after_rates = kin.pop("after_rates_per_s", None)
            kin = Kinetics(movie_minutes=tuple(float(x) for x in kin.pop("movie_minutes")),
                           after_populations=None if after is None else tuple(Population(**a) for a in after),
                           after_rates_per_s=None if after_rates is None else
                           tuple(tuple(float(x) for x in row) for row in after_rates), **kin)
        conditions.append(Condition(populations=pops, rates_per_s=None if rates is None else
                                    tuple(tuple(float(x) for x in row) for row in rates), kinetics=kin, **c))
    for key in ("cell_length_range_um", "field_px", "illumination_centre_px", "foci_positions"):
        if data.get(key) is not None:
            data[key] = tuple(data[key])
    known = {f for f in Recipe.__dataclass_fields__}
    unknown = sorted(set(data) - known)
    if unknown:
        raise GeneratorError(f"The recipe has settings this version does not know: {', '.join(unknown)}.")
    return Recipe(conditions=tuple(conditions), **data)


def foci_warnings(recipe: Recipe) -> list:
    """A population at foci faster than `FOCI_FAST_D`: said, not refused (a confined jitter is a choice one may
    make on purpose)."""
    out = []
    for c in recipe.conditions:
        pops = list(c.populations)
        if c.kinetics is not None and c.kinetics.after_populations is not None:
            pops += list(c.kinetics.after_populations)
        for i, p in enumerate(pops):
            if p.where == "foci" and p.D_um2_s > FOCI_FAST_D:
                name = p.name or f"population {i % len(c.populations) + 1}"
                out.append(f"{c.name}, {name}: foci are binding sites for a bound or slow population. At D "
                           f"{p.D_um2_s:g} it only jitters inside its {1000 * recipe.focus_radius_um:g} nm focus and "
                           f"reads a D near 0.")
                break
    return out


def save_recipe(recipe: Recipe, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(recipe_to_dict(recipe), indent=1), encoding="utf-8")
    return path


def load_recipe(path) -> Recipe:
    return recipe_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def example_recipe() -> Recipe:
    """A small two-condition set: a slow population in the nucleoid and a free one, switching, the slow share
    higher in the first condition. What the command line writes as a starting point."""
    pops = (Population(0.05, 0.3, "nucleoid", "slow"), Population(1.0, 0.7, "cell", "free"))
    return Recipe(conditions=(
        Condition("condition 1", pops, exchange_per_s=3.0, replicate_sd=0.1),
        Condition("condition 2", (replace(pops[0], share=0.15), replace(pops[1], share=0.85)), exchange_per_s=3.0,
                  replicate_sd=0.1)),
        name="example set", note="a slow population in the nucleoid, two conditions")


# --------------------------------------------------------------------------- #
# Geometry, vectorised over molecules (each molecule carries its own cell)
# --------------------------------------------------------------------------- #


@dataclass
class _Geometry:
    """The cell of every molecule, in the cell's own frame (u along the axis, v and z across)."""

    half_axis: np.ndarray        # half the straight segment
    radius: np.ndarray
    half_length: np.ndarray
    lobe_c: np.ndarray           # (M, 2) lobe centres along u
    lobe_a: np.ndarray           # (M, 2) half-lengths
    lobe_b: np.ndarray           # (M, 2) half-widths
    lobe_on: np.ndarray          # (M, 2) bool
    foci: np.ndarray             # (M, F, 3) the foci's (u, v, z) in the cell's frame
    foci_on: np.ndarray          # (M, F) bool: a cell with fewer foci than F leaves the rest off

    def take(self, index) -> "_Geometry":
        return _Geometry(*(getattr(self, f)[index] for f in self.__dataclass_fields__))


def _membrane_point(c, u0: float, angle: float, depth: float, dims: int) -> tuple:
    """A point ``depth`` under the membrane at ``u0`` along the axis and ``angle`` around it (in 2D: the side
    ``angle`` points to)."""
    r = c.radius - depth
    if dims == 2:
        return (u0, r if math.cos(angle) >= 0 else -r, 0.0)
    return (u0, r * math.cos(angle), r * math.sin(angle))


def foci_sites(cell, recipe: Recipe, rng=None) -> list:
    """The foci of one cell, ``[(u, v, z), ...]`` in its frame (`FOCI_LAYOUTS`). Layouts that draw (random,
    random angles at the membrane, a count between a min and a max) need ``rng``; the others draw nothing, so an
    old recipe (evenly, on the axis, a fixed count) builds as it did."""
    layout, depth = recipe.foci_layout, recipe.foci_depth
    rho = recipe.focus_radius_um
    membrane = depth == "membrane"
    n = int(recipe.foci_per_cell)
    if recipe.foci_per_cell_min is not None and layout in ("even", "random", "midcell"):
        n = int(rng.integers(int(recipe.foci_per_cell_min), n + 1))
    half = cell.length_um / 2.0

    def angle():
        return float(rng.uniform(0.0, 2.0 * math.pi)) if rng is not None else 0.0

    def at(u0):
        return _membrane_point(cell, u0, angle(), rho, recipe.dims) if membrane else (u0, 0.0, 0.0)

    if layout == "even":
        span = cell.length_um - cell.width_um / 2.0
        return [at(-span / 2.0 + (i + 0.5) * span / n) for i in range(n)]
    if layout == "nucleoid":
        return [at(u) for u, _v in cell.fork_sites()]
    if layout == "poles":
        if membrane:
            return [(-(half - rho), 0.0, 0.0), (half - rho, 0.0, 0.0)]
        inset = min(cell.radius, recipe.pole_depth_um) / 2.0
        return [(-(half - inset), 0.0, 0.0), (half - inset, 0.0, 0.0)]
    if layout == "midcell":
        if not membrane:
            return [(0.0, 0.0, 0.0)]
        start = angle()
        return [_membrane_point(cell, 0.0, start + 2.0 * math.pi * k / n, rho, recipe.dims) for k in range(n)]
    if layout == "random":
        span = 2.0 * max(cell.half_axis, 0.0)
        return [at(float(rng.uniform(-span / 2.0, span / 2.0))) for _ in range(n)]
    if layout == "positions":
        lo, hi = -half + rho, half - rho
        return [at(float(np.clip(-half + float(f) * cell.length_um, lo, hi))) for f in recipe.foci_positions]
    raise GeneratorError(f"unknown foci layout {layout!r}")


def cell_foci(rng, cells, recipe: Recipe) -> dict:
    """``{id(cell): its foci}`` for cells just placed; drawn from ``rng`` only where the layout draws."""
    return {id(c): foci_sites(c, recipe, rng) for c in cells}


def _geometry(cells, cell_of, recipe: Recipe, foci: dict | None = None) -> _Geometry:
    sites = [foci[id(c)] if foci is not None and id(c) in foci else foci_sites(c, recipe) for c in cells]
    n_f = max(1, max(len(s_) for s_ in sites))
    rows = []
    for c, s_ in zip(cells, sites):
        lobes = list(c.lobes)[:2] + [None] * (2 - min(2, len(c.lobes)))
        pad = list(s_) + [(np.nan, np.nan, np.nan)] * (n_f - len(s_))
        rows.append((c.half_axis, c.radius, c.length_um / 2.0,
                     [lb.centre_um if lb else 0.0 for lb in lobes], [lb.half_length_um if lb else 1.0 for lb in lobes],
                     [lb.half_width_um if lb else 1.0 for lb in lobes], [lb is not None for lb in lobes], pad,
                     [k < len(s_) for k in range(n_f)]))
    cols = list(zip(*rows))
    arr = [np.asarray(col, dtype=bool if i in (6, 8) else float) for i, col in enumerate(cols)]
    return _Geometry(*(a[cell_of] for a in arr))


def _in_region(code: int, u, v, z, g: _Geometry, recipe: Recipe) -> np.ndarray:
    """Is each point inside the cell AND inside region ``REGIONS[code]``."""
    q = np.clip(u, -g.half_axis, g.half_axis)
    d2 = (u - q) ** 2 + v * v + z * z
    inside = d2 <= g.radius ** 2
    region = REGIONS[code]
    if region == "cell":
        return inside
    if region == "nucleoid":
        rho = ((u[:, None] - g.lobe_c) / g.lobe_a) ** 2 + ((v * v + z * z)[:, None]) / g.lobe_b ** 2
        return inside & np.any((rho <= 1.0) & g.lobe_on, axis=1)
    if region == "membrane":
        return inside & (d2 >= (g.radius - recipe.membrane_depth_um) ** 2)
    if region == "poles":
        return inside & (np.abs(u) >= g.half_length - recipe.pole_depth_um)
    if region == "midcell":
        return inside & (np.abs(u) <= recipe.midcell_width_um / 2.0)
    if region == "foci":
        d = ((u[:, None] - g.foci[:, :, 0]) ** 2 + (v[:, None] - g.foci[:, :, 1]) ** 2
             + (z[:, None] - g.foci[:, :, 2]) ** 2)
        return inside & np.any(g.foci_on & (np.nan_to_num(d, nan=np.inf) <= recipe.focus_radius_um ** 2), axis=1)
    raise GeneratorError(f"unknown region {region!r}")


def _sample_in_region(rng, codes, g: _Geometry, recipe: Recipe, tries: int = 400):
    """Uniform starting positions, each inside its own population's region (rejection from the cell's box; foci
    drawn around their sites, a small target)."""
    m = len(codes)
    u, v, z = np.zeros(m), np.zeros(m), np.zeros(m)
    todo = np.ones(m, dtype=bool)
    foci = codes == REGIONS.index("foci")
    if foci.any():
        idx = np.flatnonzero(foci)
        # one of the cell's own foci (the first ``count`` columns are on), drawn evenly
        count = g.foci_on[idx].sum(axis=1)
        if len(count) and (count == count[0]).all():
            site = rng.integers(0, int(count[0]), len(idx))       # every cell alike: the first build's own draw
        else:
            site = np.minimum((rng.random(len(idx)) * count).astype(int), np.maximum(count - 1, 0))
        for _ in range(tries):
            r = recipe.focus_radius_um
            du, dv, dz = (rng.uniform(-r, r, len(idx)) for _ in range(3))
            if recipe.dims == 2:
                dz = np.zeros(len(idx))
            centre = g.foci[idx, site]
            nu, nv, nz = centre[:, 0] + du, centre[:, 1] + dv, centre[:, 2] + dz
            ok = (du * du + dv * dv + dz * dz <= r * r) & _in_region(REGIONS.index("cell"), nu, nv, nz,
                                                                    g.take(idx), recipe)
            u[idx[ok]], v[idx[ok]], z[idx[ok]] = nu[ok], nv[ok], nz[ok]
            todo[idx[ok]] = False
            idx, site = idx[~ok], site[~ok]
            if not len(idx):
                break
    for _ in range(tries):
        idx = np.flatnonzero(todo)
        if not len(idx):
            break
        sub = g.take(idx)
        uu = rng.uniform(-sub.half_length, sub.half_length)
        vv = rng.uniform(-sub.radius, sub.radius)
        zz = rng.uniform(-sub.radius, sub.radius) if recipe.dims == 3 else np.zeros(len(idx))
        ok = np.zeros(len(idx), dtype=bool)
        for code in np.unique(codes[idx]):
            mask = codes[idx] == code
            ok[mask] = _in_region(int(code), uu[mask], vv[mask], zz[mask], sub.take(np.flatnonzero(mask)), recipe)
        u[idx[ok]], v[idx[ok]], z[idx[ok]] = uu[ok], vv[ok], zz[ok]
        todo[idx[ok]] = False
    if todo.any():
        raise GeneratorError("A population's region is too small to place molecules in (try a larger region).")
    return u, v, z


# --------------------------------------------------------------------------- #
# The simulation: one replicate of one condition, all its movies at once
# --------------------------------------------------------------------------- #


def region_overlap(rng, cells, recipe: Recipe, condition: Condition, n_points: int = 4000,
                   foci: dict | None = None) -> np.ndarray:
    """P[i][j]: the share of population i's region (over these cells, by volume) that lies inside population j's.

    A switch into a population with a region is made only inside it, so a declared rate k_ij is realised as
    k_ij P[i][j]. With shares and an exchange rate the generator divides by P to hold the shares asked for (as
    SynR raises its F -> S rate inside the nucleoid, `synthetic.rates_for`); a matrix typed by hand is taken as the
    rate inside the region, and P goes into the truth so the realised rate can be read."""
    k = len(condition.populations)
    codes = [REGIONS.index(p.where) for p in condition.populations]
    g = _geometry(cells, np.arange(len(cells)), recipe, foci)
    cell_of = rng.integers(0, len(cells), n_points)
    sub = g.take(cell_of)
    overlap = np.ones((k, k))
    for i in range(k):
        u, v, z = _sample_in_region(rng, np.full(n_points, codes[i]), sub, recipe)
        for j in range(k):
            if i != j and REGIONS[codes[j]] != "cell":
                overlap[i, j] = float(_in_region(codes[j], u, v, z, sub, recipe).mean())
    return overlap


def _sub_matrix(rates: np.ndarray, dt: float) -> np.ndarray:
    from scipy.linalg import expm

    q = rates.copy()
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    a = np.clip(expm(q * dt), 0.0, None)
    return a / a.sum(axis=1, keepdims=True)


def _stationary(rates: np.ndarray) -> np.ndarray:
    from .simulate import ConditionSpec, StateSpec

    k = rates.shape[0]
    return ConditionSpec("x", tuple(StateSpec(1.0) for _ in range(k)), rates_per_s=rates).stationary()


#: The warm-up before a molecule is recorded, in multiples of the slowest relaxation (1 / the smallest rate out of a
#: population), bounded: long enough that state and position are drawn from the chain's own stationary state,
#: spatial correlations included (a molecule that just unbound is still in its region).
WARMUP_RELAXATIONS = 4.0
WARMUP_FRAMES_MAX = 300
#: The pilot that holds the declared shares when a population lives in a region (shares + exchange rate):
#: molecules, recorded frames and rounds.
PILOT_MOLECULES, PILOT_FRAMES, PILOT_ROUNDS = 600, 100, 3


def warmup_frames(rates, tau: float) -> int:
    if rates is None:
        return 0
    out = rates.sum(axis=1)
    slowest = float(out[out > 0].min()) if np.any(out > 0) else 0.0
    if slowest <= 0:
        return 0
    return int(min(WARMUP_FRAMES_MAX, math.ceil(WARMUP_RELAXATIONS / slowest / tau)))


def _simulate(rng, molecules: dict, g: _Geometry, recipe: Recipe, condition: Condition, rates, shares,
              warmup: int = 0, kin: dict | None = None, keep_sub: bool = False):
    """Every molecule's frames: observed (blurred, with error) and true positions in the cell's frame, and its true
    population per frame. ``molecules``: per molecule its number of frames. ``warmup`` frames are run first and not
    recorded (`warmup_frames`). ``kin`` (a kinetic condition, `_kinetic_levels`): per molecule ``t0``, its first
    frame's time since ITS onset reference (s); the response's delay and half-time; D after; and per level of h the
    cumulative sub-step matrices and the starting probabilities. The warm-up runs at the first frame's h."""
    k = len(condition.populations)
    D = np.array([p.D_um2_s for p in condition.populations], dtype=float)
    D_after = D if kin is None else np.asarray(kin["D_after"], dtype=float)
    codes_of = np.array([REGIONS.index(p.where) for p in condition.populations])
    m = len(molecules["length"])
    tau = recipe.tau_s
    dt = tau / recipe.n_subframes

    def h_at(frame):
        if kin is None:
            return np.zeros(m)
        return response(kin["t0"] + max(frame, 0) * tau, kin["delay_s"], kin["half_time_s"])

    if kin is None:
        start_p = _stationary(rates) if rates is not None else np.asarray(shares, dtype=float)
        start_p = start_p / start_p.sum()
        state = rng.choice(k, size=m, p=start_p)
        cum = np.cumsum(_sub_matrix(rates, dt), axis=1) if rates is not None else None
    else:
        levels = kin["start"].shape[0]
        lev = np.rint(h_at(0) * (levels - 1)).astype(int)
        state = np.minimum((rng.random(m)[:, None] > np.cumsum(kin["start"][lev], axis=1)).sum(axis=1), k - 1)
        cum = kin["cum"]
    u, v, z = _sample_in_region(rng, codes_of[state], g, recipe)
    from .synthetic import MICRO_STEP_UM

    d_max = float(max(D.max(), D_after.max()))
    n_micro = max(1, math.ceil(2.0 * d_max * dt / MICRO_STEP_UM ** 2)) if d_max > 0 else 0
    length = molecules["length"]
    t_max = int(length.max())
    obs = np.full((m, t_max, 2), np.nan)
    true = np.full((m, t_max, 2), np.nan)
    true_state = np.full((m, t_max), -1, dtype=int)
    exposed = recipe.exposed_subframes
    subs = np.full((m, t_max, exposed, 3), np.nan) if keep_sub else None
    for f in range(-int(warmup), t_max):
        alive = length > f if f >= 0 else np.ones(m, dtype=bool)
        h = h_at(f)
        if kin is not None:
            lev = np.rint(h * (kin["start"].shape[0] - 1)).astype(int)
        if f >= 0:
            true[alive, f, 0], true[alive, f, 1] = u[alive], v[alive]
            true_state[alive, f] = state[alive]
        acc_u, acc_v = np.zeros(m), np.zeros(m)
        for s in range(recipe.n_subframes):
            if s < exposed:
                acc_u += u
                acc_v += v
                if keep_sub and f >= 0:
                    subs[alive, f, s, 0], subs[alive, f, s, 1], subs[alive, f, s, 2] = u[alive], v[alive], z[alive]
            if cum is not None:
                draw = rng.random(m)
                rows = cum[state] if kin is None else cum[lev, state]
                proposed = np.minimum((draw[:, None] > rows).sum(axis=1), k - 1)
                change = alive & (proposed != state)
                if change.any():
                    idx = np.flatnonzero(change)
                    allowed = np.zeros(len(idx), dtype=bool)
                    for code in np.unique(codes_of[proposed[idx]]):
                        sel = codes_of[proposed[idx]] == code
                        j = idx[sel]
                        allowed[sel] = _in_region(int(code), u[j], v[j], z[j], g.take(j), recipe)
                    state[idx[allowed]] = proposed[idx[allowed]]
            if n_micro:
                D_now = D[state] + (D_after[state] - D[state]) * h
                sd = np.sqrt(2.0 * np.clip(D_now, 0.0, None) * dt / n_micro)
                moving = alive & (sd > 0)
                if moving.any():
                    idx = np.flatnonzero(moving)
                    sub = g.take(idx)
                    codes = codes_of[state[idx]]
                    sdi = sd[idx]
                    for _ in range(n_micro):
                        nu = u[idx] + rng.normal(0.0, 1.0, len(idx)) * sdi
                        nv = v[idx] + rng.normal(0.0, 1.0, len(idx)) * sdi
                        nz = (z[idx] + rng.normal(0.0, 1.0, len(idx)) * sdi) if recipe.dims == 3 else z[idx]
                        ok = np.zeros(len(idx), dtype=bool)
                        for code in np.unique(codes):
                            sel = codes == code
                            ok[sel] = _in_region(int(code), nu[sel], nv[sel], nz[sel],
                                                 sub.take(np.flatnonzero(sel)), recipe)
                        keep = idx[ok]
                        u[keep], v[keep], z[keep] = nu[ok], nv[ok], nz[ok]
        if f >= 0:
            obs[alive, f, 0] = acc_u[alive] / exposed
            obs[alive, f, 1] = acc_v[alive] / exposed
    if keep_sub:
        return obs, true, true_state, subs
    return obs, true, true_state


def _kinetic_levels(recipe: Recipe, rates_before, rates_after, shares_before, shares_after) -> dict:
    """Per level of h (`KINETIC_LEVELS`): the cumulative sub-step matrices of ``(1 - h) before + h after`` and the
    starting probabilities (their stationary occupancy; without switching the shares the same way)."""
    hs = np.linspace(0.0, 1.0, KINETIC_LEVELS)
    dt = recipe.tau_s / recipe.n_subframes
    if rates_before is None:
        start = np.array([(1 - h) * np.asarray(shares_before) + h * np.asarray(shares_after) for h in hs])
        return {"cum": None, "start": start / start.sum(axis=1, keepdims=True)}
    mats = [(1 - h) * rates_before + h * rates_after for h in hs]
    return {"cum": np.array([np.cumsum(_sub_matrix(r, dt), axis=1) for r in mats]),
            "start": np.array([_stationary(r) for r in mats])}


def hold_shares(rng, cells, recipe: Recipe, condition: Condition, rates: np.ndarray, target,
                foci: dict | None = None) -> tuple:
    """Rates that hold the declared shares where populations live in regions: a pilot on these cells, the rates
    INTO each population with a region scaled by the ratio of asked to reached odds, `PILOT_ROUNDS` times.

    Why a pilot and not a formula: a molecule that leaves a region-bound population is still inside the region, so
    it re-enters more often than a molecule spread evenly over the cell; how much more depends on D, the rates and
    the region's shape. Returns the rates and the pilot's last occupancy (localisations)."""
    k = len(condition.populations)
    target = np.asarray(target, dtype=float)
    regional = [j for j, p in enumerate(condition.populations) if p.where != "cell"]
    if not regional:
        return rates, None
    rates = rates.copy()
    reached = None
    for _ in range(PILOT_ROUNDS):
        cell_of = rng.integers(0, len(cells), PILOT_MOLECULES)
        g = _geometry(cells, cell_of, recipe, foci)
        mol = {"length": np.full(PILOT_MOLECULES, PILOT_FRAMES)}
        _o, _t, states = _simulate(rng, mol, g, recipe, condition, rates, None,
                                   warmup=warmup_frames(rates, recipe.tau_s))
        reached = np.bincount(states[states >= 0].ravel(), minlength=k) / max(1, (states >= 0).sum())
        for j in regional:
            p, q = float(np.clip(reached[j], 1e-4, 1 - 1e-4)), float(np.clip(target[j], 1e-4, 1 - 1e-4))
            rates[np.arange(k) != j, j] *= (q / (1 - q)) / (p / (1 - p))
    return rates, reached


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #


def _design_for_writers(recipe: Recipe):
    """The paper's writers take a `synthetic.SynRDesign`: the fields they read, from the recipe."""
    from .synthetic import SynRDesign

    return SynRDesign(pixel_size_nm=recipe.pixel_size_nm, frame_interval_s=recipe.tau_s, n_frames=recipe.n_frames,
                      field_px=tuple(recipe.field_px), sigma_um=recipe.localisation_error_nm / 1000.0)


#: The bins of the kinetic truth table (s on the addition's clock).
KINETIC_BIN_S = 10.0
#: The simulated clock: replicate r's addition at 10:00 on day r of this month (any date would do; it is written so
#: the Kinetics section's addition clock reads it as it reads a real experiment's).
CLOCK_ORIGIN = (2026, 1, 1, 10, 0)


def _tilt(shares, factor: float) -> np.ndarray:
    """The shares with the first population's odds multiplied by ``factor`` (a replicate's draw)."""
    shares = np.asarray(shares, dtype=float)
    if len(shares) < 2 or factor == 1.0:
        return shares
    odds = shares[0] / max(1e-12, 1 - shares[0]) * factor
    first = odds / (1 + odds)
    rest = shares[1:] / max(1e-12, shares[1:].sum()) * (1 - first)
    return np.concatenate([[first], rest])


def _counted(states_per_track, k: int, tau: float) -> dict:
    """Occupancy over localisations and steps, transition counts frame to frame, and the realised rates."""
    loc = np.zeros(k)
    step = np.zeros(k)
    trans = np.zeros((k, k))
    for seq in states_per_track:
        seq = np.asarray(seq)
        np.add.at(loc, seq, 1)
        if len(seq) > 1:
            np.add.at(step, seq[:-1], 1)
            np.add.at(trans, (seq[:-1], seq[1:]), 1)
    out = {"occupancy_localisations": (loc / max(loc.sum(), 1)).round(6).tolist(),
           "occupancy_steps": (step / max(step.sum(), 1)).round(6).tolist(),
           "transitions_frame_to_frame": trans.astype(int).tolist(),
           "n_localisations": int(loc.sum()), "n_steps": int(step.sum())}
    rates = np.zeros((k, k))
    for i in range(k):
        if step[i] > 0:
            rates[i] = trans[i] / step[i] / tau
    np.fill_diagonal(rates, 0.0)
    out["rates_per_s_counted"] = rates.round(4).tolist()
    return out


def build(recipe: Recipe, root, *, on_line=None, on_progress=None) -> dict:
    """Build the set at ``root`` (a new folder) and return its truth, also written to ``truth/truth.json``.
    ``on_line(text)`` hears each replicate's line, ``on_progress(done, total)`` counts the replicates built (every
    condition's), for a progress bar.

    WRITTEN WHOLE OR NOT AT ALL: the set is built in ``<root>.partial`` and renamed at the end, so a build that is
    refused half-way (cells that do not fit a field, a region too small) leaves nothing behind."""
    import shutil

    problems = check(recipe)
    if problems:
        raise GeneratorError(" ".join(problems))
    final = Path(root)
    if final.exists():
        raise GeneratorError(f"{final} exists: a simulated set is built into a new folder, never over one.")
    work = final.with_name(final.name + ".partial")
    if work.exists():
        raise GeneratorError(f"{work} is left from a build that stopped: remove it first.")
    try:
        truth = _build(recipe, work, on_line=on_line, on_progress=on_progress)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    work.rename(final)
    return truth


def _build(recipe: Recipe, root, *, on_line=None, on_progress=None) -> dict:
    import pandas as pd

    from .synthetic import OUTLINE_MARGIN_SIGMA, _write_meshes, place_cells, write_trackmate_xml

    root = Path(root)
    say = on_line or (lambda _line: None)
    total = len(recipe.conditions) * recipe.replicates
    done = 0

    def progress():
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    if on_progress is not None:
        on_progress(0, total)
    design = _design_for_writers(recipe)
    rng = np.random.default_rng(recipe.seed)
    (root / "truth").mkdir(parents=True)
    save_recipe(recipe, root / "recipe.json")
    truth = {"name": recipe.name, "synthetic": True, "note": recipe.note,
             "written_by": "smtracker.generator", "outline_margin_sigma": OUTLINE_MARGIN_SIGMA,
             "rules": ["the truth is counted from each frame's instantaneous position and population",
                       "a switch into a population with a region is made only inside that region"],
             "conditions": {}}
    lo, hi = recipe.cell_length_range_um
    tau = recipe.tau_s
    sigma = recipe.localisation_error_nm / 1000.0
    movies_out = recipe.output == "movies"
    for cond_index, condition in enumerate(recipe.conditions):
        k = len(condition.populations)
        base_rates = rates_matrix(condition)
        shares = np.array([p.share for p in condition.populations], dtype=float)
        cond_dir = root / condition.name
        crops, day_map, starts, additions = {}, {}, {}, {}
        rows, cell_rows = [], []
        cond_truth = {"populations": [asdict(p) for p in condition.populations],
                      "rates_per_s_declared": None if base_rates is None else base_rates.round(6).tolist(),
                      "stationary_declared": None if base_rates is None else _stationary(base_rates).round(6).tolist(),
                      "replicates": {}}
        all_states = []
        for rep in range(1, recipe.replicates + 1):
            factor = float(math.exp(rng.normal(0.0, condition.replicate_sd))) if condition.replicate_sd else 1.0
            rates = None
            rep_shares = shares
            if base_rates is not None:
                rates = base_rates.copy()
                rates[1:, 0] *= factor
            elif k > 1 and condition.replicate_sd:
                rep_shares = _tilt(shares, factor)
            movies = []
            foci_map: dict = {}
            for mi in range(recipe.movies_per_replicate):
                movie = f"series{(rep - 1) * recipe.movies_per_replicate + mi + 1:03d}"
                lengths = np.clip(rng.normal(recipe.cell_length_mean_um, recipe.cell_length_sd_um,
                                             recipe.cells_per_movie), lo, hi)
                try:
                    cells = place_cells(rng, lengths, width_um=recipe.cell_width_um, field_um=recipe.field_um)
                except Exception as exc:  # the placer's own refusal, said in the recipe's words
                    raise GeneratorError(f"{recipe.cells_per_movie} cells of up to {hi:g} um do not fit a field of "
                                         f"{recipe.field_px[0]} x {recipe.field_px[1]} pixels: {exc}") from exc
                movies.append((movie, cells))
                # the foci from their OWN stream: a recipe whose layout draws nothing builds as before, draw for draw
                foci_map.update(cell_foci(np.random.default_rng([int(recipe.seed), 31, cond_index, rep, mi]), cells,
                                          recipe) if recipe.foci_layout in ("random",) or recipe.foci_depth ==
                                "membrane" or recipe.foci_per_cell_min is not None else cell_foci(None, cells, recipe))
            # the molecules of every movie of the replicate, simulated together
            cell_index, movie_index, firsts, lens, cell_of_global = [], [], [], [], []
            all_cells = []
            for mv, (movie, cells) in enumerate(movies):
                for c in cells:
                    count = recipe.molecules_per_cell
                    if recipe.molecules_sd_log:
                        count = count * math.exp(rng.normal(0.0, recipe.molecules_sd_log))
                    n = int(rng.poisson(count))
                    first = rng.integers(1, recipe.n_frames + 1, n)
                    life = rng.geometric(1.0 / recipe.bleach_mean_frames, n) if recipe.bleach_mean_frames > 1 \
                        else np.ones(n, dtype=int)
                    life = np.minimum(life, recipe.n_frames - first + 1)
                    # a movie renders every molecule; the tracker's minimum drops the short ones
                    keep = life >= (1 if movies_out else recipe.min_localizations)
                    cell_index += [len(all_cells)] * int(keep.sum())
                    movie_index += [mv] * int(keep.sum())
                    firsts += list(first[keep])
                    lens += list(life[keep])
                    all_cells.append(c)
            if not lens:
                raise GeneratorError(f"Condition {condition.name!r}, replicate {rep}: no molecule stays visible for "
                                     f"{recipe.min_localizations} frames; raise the molecules per cell or the "
                                     f"bleaching time.")
            cell_of = np.asarray(cell_index)
            g = _geometry(all_cells, cell_of, recipe, foci_map)
            molecules = {"length": np.asarray(lens), "first": np.asarray(firsts)}
            overlap = None
            if rates is not None:
                overlap = region_overlap(rng, all_cells, recipe, condition, foci=foci_map)
                needed = (rates > 0) & (overlap <= 0)
                if needed.any():
                    i, j = map(int, np.argwhere(needed)[0])
                    raise GeneratorError(
                        f"Condition {condition.name!r}: population {i + 1} ({_REGION_WORDS[condition.populations[i].where]}) "
                        f"can never switch to population {j + 1} ({_REGION_WORDS[condition.populations[j].where]}): "
                        f"their regions do not overlap.")
            pilot = None
            if rates is not None and condition.exchange_per_s is not None:
                # shares asked for: the rates into a region set by a pilot so that the shares are met (`hold_shares`)
                rates, pilot = hold_shares(rng, all_cells, recipe, condition, rates, _stationary(rates), foci_map)
            warmup = warmup_frames(rates, tau)
            kin_args, clock0, rates_after = None, None, None
            kin = condition.kinetics
            if kin is not None:
                after_pops = condition.populations if kin.after_populations is None else kin.after_populations
                D_after = np.array([a.D_um2_s for a in after_pops], dtype=float)
                shares_after = _tilt([a.share for a in after_pops], factor)
                if rates is not None:
                    if kin.after_populations is None:
                        rates_after = rates
                    elif condition.rates_per_s is not None:
                        rates_after = np.array(kin.after_rates_per_s, dtype=float)
                        np.fill_diagonal(rates_after, 0.0)
                        rates_after[1:, 0] *= factor
                    else:
                        from .simulate import rates_for_occupancy

                        rates_after = rates_for_occupancy([a.share for a in after_pops], float(condition.exchange_per_s))
                        rates_after[1:, 0] *= factor
                        rates_after, _ = hold_shares(rng, all_cells, recipe, replace(condition, populations=after_pops),
                                                     rates_after, _stationary(rates_after), foci_map)
                    warmup = max(warmup, warmup_frames(rates_after, tau))
                levels = _kinetic_levels(recipe, rates, rates_after, rep_shares, shares_after)
                off_rep = float(rng.normal(0.0, kin.onset_sd_replicate_s)) if kin.onset_sd_replicate_s else 0.0
                off_cell = (rng.normal(0.0, kin.onset_sd_cell_s, len(all_cells)) if kin.onset_sd_cell_s
                            else np.zeros(len(all_cells)))
                movie_start = np.array([60.0 * kin.movie_minutes[mv] for mv in movie_index])
                clock0 = movie_start + (np.asarray(firsts) - 1) * tau
                kin_args = {"t0": clock0 - off_rep - off_cell[cell_of], "delay_s": kin.delay_s,
                            "half_time_s": kin.half_time_s, "D_after": D_after, **levels}
            result = _simulate(rng, molecules, g, recipe, condition, rates, rep_shares, warmup=warmup, kin=kin_args,
                               keep_sub=movies_out)
            obs, true_pos, true_state = result[:3]
            if movies_out:
                _render_replicate(recipe, condition, rep, movies, all_cells, cell_of, movie_index, firsts, lens,
                                  true_pos, true_state, result[3], kin, kin_args, clock0, cond_dir, root, rng,
                                  rows, crops, day_map, additions, design)
                counted = _counted([list(true_state[i, :lens[i]]) for i in range(len(lens))], k, tau)
                cond_truth["replicates"][str(rep)] = {"factor": factor, "molecules": len(lens), **counted}
                all_states += [list(true_state[i, :lens[i]]) for i in range(len(lens))]
                for mv, (movie, cells) in enumerate(movies):
                    for c in cells:
                        cell_rows.append({"replicate": rep, "movie": movie, "cell": c.cell_id, "cx_um": c.cx_um,
                                          "cy_um": c.cy_um, "angle_rad": c.angle, "length_um": c.length_um,
                                          "width_um": c.width_um,
                                          "foci_uvz_um": json.dumps([[round(float(x), 5) for x in site]
                                                                     for site in foci_map.get(id(c), [])])})
                say(f"{condition.name}, replicate {rep}: {len(lens):,} molecules rendered into "
                    f"{len(movies)} movies")
                progress()
                continue
            per_movie = {mv: [] for mv in range(len(movies))}
            rep_states = []
            for i in range(len(lens)):
                n = lens[i]
                c = all_cells[cell_of[i]]
                ou, ov = obs[i, :n, 0], obs[i, :n, 1]
                x, y = c.to_field(ou, ov)
                x = np.asarray(x) + rng.normal(0.0, sigma, n)
                y = np.asarray(y) + rng.normal(0.0, sigma, n)
                tx, ty = c.to_field(true_pos[i, :n, 0], true_pos[i, :n, 1])
                frames = list(range(int(firsts[i]), int(firsts[i]) + n))
                states = [int(s) for s in true_state[i, :n]]
                rep_states.append(states)
                if kin_args is not None:
                    clock = clock0[i] + np.arange(n) * tau
                    h = response(kin_args["t0"][i] + np.arange(n) * tau, kin.delay_s, kin.half_time_s)
                else:
                    clock = h = np.full(n, np.nan)
                per_movie[movie_index[i]].append({"frames": frames, "x_um": list(map(float, x)),
                                                  "y_um": list(map(float, y)), "cell": c.cell_id,
                                                  "tx": list(map(float, np.asarray(tx))),
                                                  "ty": list(map(float, np.asarray(ty))), "states": states,
                                                  "clock": list(map(float, clock)), "h": list(map(float, h))})
            for mv, (movie, cells) in enumerate(movies):
                tracks = per_movie[mv]
                for t_index, t in enumerate(tracks):
                    t["track"] = t_index + 1
                if not tracks:
                    raise GeneratorError(f"Condition {condition.name!r}, {movie}: no track long enough.")
                write_trackmate_xml(cond_dir / "trackmate" / f"{movie}.xml", tracks, design=design,
                                    image_name=f"{movie}.tif")
                _write_meshes(cond_dir / "cell_meshes" / f"{movie}.mat", cells, movie, design)
                crops[movie] = [1, recipe.n_frames]
                day_map[movie] = rep
                if kin is not None:
                    import datetime as _dt

                    addition = _dt.datetime(*CLOCK_ORIGIN) + _dt.timedelta(days=rep - 1)
                    starts[movie] = (addition + _dt.timedelta(seconds=60.0 * kin.movie_minutes[mv])).isoformat()
                    additions[rep] = addition.strftime("%Y-%m-%d %H:%M")
                for c in cells:
                    cell_rows.append({"replicate": rep, "movie": movie, "cell": c.cell_id, "cx_um": c.cx_um,
                                      "cy_um": c.cy_um, "angle_rad": c.angle, "length_um": c.length_um,
                                      "width_um": c.width_um,
                                      "foci_uvz_um": json.dumps([[round(float(x), 5) for x in site]
                                                                 for site in foci_map.get(id(c), [])])})
                for t in tracks:
                    for j, frame in enumerate(t["frames"]):
                        s = t["states"][j]
                        d_true = condition.populations[s].D_um2_s
                        if kin is not None:
                            d_true = d_true + (kin_args["D_after"][s] - d_true) * t["h"][j]
                        rows.append((rep, movie, t["cell"], t["track"], frame, t["x_um"][j], t["y_um"][j],
                                     t["tx"][j], t["ty"][j], s + 1, d_true, t["clock"][j], t["h"][j]))
            counted = _counted(rep_states, k, tau)
            cond_truth["replicates"][str(rep)] = {
                "factor": factor, "tracks": len(rep_states),
                **({} if rates_after is None else {"rates_per_s_after": rates_after.round(6).tolist()}),
                **({} if rates is None else {"rates_per_s_inside_regions": rates.round(6).tolist(),
                                              "region_overlap": overlap.round(4).tolist(),
                                              "warmup_frames": warmup,
                                              **({} if pilot is None else
                                                 {"pilot_occupancy": np.round(pilot, 4).tolist()})}),
                **counted}
            all_states += rep_states
            say(f"{condition.name}, replicate {rep}: {len(rep_states):,} tracks, "
                + ", ".join(f"{p.name or i + 1} {100 * s:.1f} %" for i, (p, s) in
                            enumerate(zip(condition.populations, counted["occupancy_steps"]))) + " of steps")
            progress()
        cond_truth.update(_counted(all_states, k, tau))
        truth["conditions"][condition.name] = cond_truth
        if movies_out:
            _finish_movie_condition(recipe, condition, cond_dir, root, rows, cell_rows, crops, day_map, additions,
                                    cond_truth)
            continue
        (cond_dir / "day_map.json").write_text(json.dumps({
            "movie_to_day": day_map, "source": "synthetic", "series_per_day": recipe.movies_per_replicate,
            "warnings": [], "written_by": "smtracker.generator.build"}, indent=2), encoding="utf-8")
        (cond_dir / "bleaching").mkdir(parents=True, exist_ok=True)
        (cond_dir / "bleaching" / "crops.json").write_text(json.dumps({
            "crops": crops, "pixel_size_nm": recipe.pixel_size_nm, "frame_interval_s": tau,
            "exposure_s": None if recipe.exposure_ms is None else recipe.exposure_ms / 1000.0,
            "note": [f"SIMULATED ({recipe.name}): every frame of every movie is used"]}, indent=2), encoding="utf-8")
        table = pd.DataFrame(rows, columns=["replicate", "movie", "cell", "track", "frame", "x_um", "y_um",
                                            "x_true_um", "y_true_um", "true_population", "true_D_um2_s",
                                            "time_since_addition_s", "response"])
        if condition.kinetics is not None:
            _write_kinetic_truth(root, condition, table)
            (cond_dir / SIMULATED_STARTS_FILE).write_text(json.dumps({
                "movies": starts, "written_by": "smtracker.generator",
                "note": "each movie's simulated recording start; the Kinetics section reads it where a movie has "
                        "no camera file"}, indent=2), encoding="utf-8")
            from .kinetics_designs import save_culture_times

            save_culture_times(cond_dir, additions)
            cond_truth["kinetics"] = {**_to_plain(asdict(condition.kinetics)),
                                      "additions": {str(k): v for k, v in additions.items()},
                                      "rule": "h(t) = 1 - 2^-((t - onset - delay) / half-time); D, shares and rates "
                                              "move from before to after along h; onset = the replicate's and "
                                              "the cell's offsets"}
        else:
            table = table.drop(columns=["time_since_addition_s", "response"])
        with gzip.open(root / "truth" / f"{condition.name}_localisations.csv.gz", "wt", encoding="utf-8") as fh:
            table.to_csv(fh, index=False)
        pd.DataFrame(cell_rows).to_csv(root / "truth" / f"{condition.name}_cells.csv", index=False)
    (root / "truth" / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


#: Movies without kinetics: replicate r recorded on day r from `CLOCK_ORIGIN`, the movies this far apart beyond
#: their own length (s).
MOVIE_GAP_S = 30.0
MOVIE_COLUMNS = ["replicate", "movie", "cell", "track", "frame", "x_true_um", "y_true_um", "x_px", "y_px",
                 "true_population", "true_D_um2_s", "z_um", "dark", "time_since_addition_s", "response"]


def _illumination(recipe: Recipe) -> np.ndarray:
    cols, rows = recipe.field_px
    if not recipe.illumination_sigma_px:
        return np.ones((rows, cols))
    cx, cy = recipe.illumination_centre_px or ((cols - 1) / 2.0, (rows - 1) / 2.0)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    beam = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * float(recipe.illumination_sigma_px) ** 2))
    return beam / beam.mean()


def _movie_start(recipe: Recipe, kin, rep: int, mv: int):
    import datetime as _dt

    day = _dt.datetime(*CLOCK_ORIGIN) + _dt.timedelta(days=rep - 1)
    if kin is not None:
        return day + _dt.timedelta(seconds=60.0 * kin.movie_minutes[mv])
    return day + _dt.timedelta(seconds=mv * (recipe.n_frames * recipe.tau_s + MOVIE_GAP_S))


def psf_px_of(recipe: Recipe) -> float:
    """The PSF's sigma in pixels, from the wavelength and the aperture (`optics.psf_sigma_nm`)."""
    from .optics import psf_sigma_nm

    return psf_sigma_nm(emission_nm=recipe.emission_nm, numerical_aperture=recipe.numerical_aperture) / \
        recipe.pixel_size_nm


def render_frames(recipe: Recipe, cells, spots, rng, n: int, *, design=None, illum=None) -> np.ndarray:
    """``n`` camera frames of one field: the spots (`moviesim.render_movie`'s columns, one row per exposed
    sub-step) on the background, the beam, the haze, the cells' autofluorescence and its fading, a background that
    changes over the movie, and the camera's noise (`camera_calibration`). The ONE renderer of a movie's frames: a
    build's (`_render_replicate`) and the window's preview frame (`generator_preview.frame_preview`)."""
    from .moviesim import EMCCD_EXCESS_NOISE, CameraSpec, OpticsSpec, PhotonSpec, render_movie
    from .synthetic import _cell_mask

    design = _design_for_writers(recipe) if design is None else design
    illum = _illumination(recipe) if illum is None else illum
    cal = camera_calibration(recipe)
    cols, rows_px = recipe.field_px
    optics = OpticsSpec(psf_sigma_px=psf_px_of(recipe), pixel_size_nm=recipe.pixel_size_nm)
    camera = CameraSpec(offset_counts=max(float(cal["offset_counts"]), 1e-6), em_gain=float(cal["em_gain"]),
                        sensitivity_e_per_count=cal["sensitivity_e_per_count"],
                        read_noise_e=float(cal.get("read_noise_e", 0.0)),
                        # no EM register (gain 1, an sCMOS): no excess noise
                        excess_noise=EMCCD_EXCESS_NOISE if float(cal["em_gain"]) > 1.0 else 1.0)
    photons = PhotonSpec(photons_per_frame=recipe.photons_per_spot,
                         background_photons_per_px=recipe.background_photons_per_px)
    mask = _cell_mask(cells, design)
    extras = []
    if recipe.autofluorescence_photons_per_px > 0:
        cv = recipe.autofluorescence_cell_cv
        sd_log = math.sqrt(math.log(1.0 + cv * cv)) if cv > 0 else 0.0
        weights = {c.cell_id: float(math.exp(rng.normal(-0.5 * sd_log ** 2, sd_log))) if sd_log else 1.0
                   for c in cells}
        half = recipe.autofluorescence_half_life_frames
        fade = np.power(0.5, np.arange(n) / half) if half > 0 else np.ones(n)
        extras.append((recipe.autofluorescence_photons_per_px * illum * _cell_mask(cells, design, weights), fade))
    if recipe.background_change:
        ramp = recipe.background_change * np.arange(n) / max(1, n - 1)
        extras.append((recipe.background_photons_per_px * illum, ramp))
    stack, _rendered = render_movie(
        spots, optics=optics, camera=camera, photons=photons, field_shape=(rows_px, cols), margin_px=0.0,
        seed=int(rng.integers(2 ** 31)), centre=False, n_frames=n,
        background_map=illum * (recipe.background_photons_per_px + recipe.haze_photons_per_px * mask),
        extra_background=extras or None)
    return stack


def _render_replicate(recipe, condition, rep, movies, all_cells, cell_of, movie_index, firsts, lens, true_pos,
                      true_state, sub, kin, kin_args, clock0, cond_dir, root, rng, rows, crops, day_map, additions,
                      design) -> None:
    """One replicate's movies: each molecule's exposed sub-steps rendered as spots (motion blur), dimmer and wider
    out of focus, dark while it blinks, its brightness varying; the background, the beam, the haze, the cells'
    autofluorescence and its fading, and a background that changes over the movie; the camera's noise. Writes the
    camera files (`synthetic.write_stk`, stamped with the recording clock), a phase image and, if asked, the true
    outlines; appends the truth rows."""
    import pandas as pd
    import tifffile

    from .cellmesh import CellMesh, write_cell_meshes
    from .synthetic import _write_meshes, render_phase, write_stk

    cal = camera_calibration(recipe)
    um = recipe.pixel_size_nm / 1000.0
    psf_px = psf_px_of(recipe)
    exposed = recipe.exposed_subframes
    illum = _illumination(recipe)
    n = recipe.n_frames
    cols, rows_px = recipe.field_px
    # each molecule's blinking (an on/off chain per frame, starting on) and brightness (lognormal, AR(1) in log)
    m, t_max = true_state.shape
    dark = np.zeros((m, t_max), dtype=bool)
    if recipe.blink_off_per_frame > 0:
        on = np.ones(m, dtype=bool)
        for f in range(t_max):
            dark[:, f] = ~on
            draw = rng.random(m)
            on = np.where(on, draw >= recipe.blink_off_per_frame, draw < recipe.blink_on_per_frame)
    bright = np.ones((m, t_max))
    if recipe.emission_cv > 0:
        sd_log = math.sqrt(math.log(1.0 + recipe.emission_cv ** 2))
        corr = math.exp(-1.0 / recipe.emission_corr_frames) if recipe.emission_corr_frames > 0 else 0.0
        x = rng.normal(0.0, 1.0, m)
        for f in range(t_max):
            if f:
                x = corr * x + math.sqrt(1 - corr * corr) * rng.normal(0.0, 1.0, m)
            bright[:, f] = np.exp(sd_log * x - 0.5 * sd_log ** 2)
    first = np.asarray(firsts)
    for mv, (movie, cells) in enumerate(movies):
        mols = [i for i in range(m) if movie_index[i] == mv]
        parts = []
        for i in mols:
            c = all_cells[cell_of[i]]
            L = lens[i]
            su, sv, sz = sub[i, :L, :, 0], sub[i, :L, :, 1], sub[i, :L, :, 2]
            fx, fy = c.to_field(su, sv)
            px, py = np.asarray(fx) / um, np.asarray(fy) / um
            frames = np.repeat(first[i] + np.arange(L), exposed).reshape(L, exposed)
            inside = (px >= 0) & (px <= cols - 1) & (py >= 0) & (py <= rows_px - 1)
            beam = illum[np.clip(np.rint(py).astype(int), 0, rows_px - 1), np.clip(np.rint(px).astype(int), 0,
                                                                                     cols - 1)]
            ph = recipe.photons_per_spot / exposed * beam * bright[i, :L, None]
            sig = (psf_px * np.sqrt(1.0 + (sz / recipe.defocus_depth_um) ** 2) if recipe.defocus_depth_um > 0
                   else np.full(sz.shape, psf_px))
            keep = inside & ~dark[i, :L, None]
            parts.append(pd.DataFrame({"track_id": i + 1, "frame": frames[keep], "x": px[keep], "y": py[keep],
                                       "photons": ph[keep], "sigma_px": sig[keep]}))
        spots = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["track_id", "frame", "x", "y", "photons", "sigma_px"])
        stack = render_frames(recipe, cells, spots, rng, n, design=design, illum=illum)
        start = _movie_start(recipe, kin, rep, mv)
        write_stk(cond_dir / "raw" / f"{movie}.stk", stack, start=start, tau_s=recipe.tau_s,
                  exposure_s=None if recipe.exposure_ms is None else recipe.exposure_ms / 1000.0,
                  em_gain=cal.get("em_gain"))
        (cond_dir / "phase").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(cond_dir / "phase" / f"{movie}.tif", render_phase(cells, design, rng),
                         photometric="minisblack")
        exact = root / "truth" / "outlines" / condition.name
        exact.mkdir(parents=True, exist_ok=True)
        write_cell_meshes(exact / f"{movie}.mat", [CellMesh(movie=movie, cell_id=str(c.cell_id),
                                                            polygon=c.outline() / um, mesh=c.ribs() / um)
                                                   for c in cells],
                          image_shape=(rows_px, cols), pixel_origin=0)
        if recipe.write_outlines:
            _write_meshes(cond_dir / "cell_meshes" / f"{movie}.mat", cells, movie, design)
        crops[movie] = [1, n]
        day_map[movie] = rep
        if kin is not None:
            import datetime as _dt

            additions[rep] = (_dt.datetime(*CLOCK_ORIGIN) + _dt.timedelta(days=rep - 1)).strftime("%Y-%m-%d %H:%M")
        for i in mols:
            c = all_cells[cell_of[i]]
            L = lens[i]
            tx, ty = c.to_field(true_pos[i, :L, 0], true_pos[i, :L, 1])
            tz = sub[i, :L, 0, 2]
            if kin_args is not None:
                clock = clock0[i] + np.arange(L) * recipe.tau_s
                h = response(kin_args["t0"][i] + np.arange(L) * recipe.tau_s, kin.delay_s, kin.half_time_s)
            else:
                clock = h = np.full(L, np.nan)
            for j in range(L):
                s_ = int(true_state[i, j])
                d_true = condition.populations[s_].D_um2_s
                if kin_args is not None:
                    d_true = d_true + (kin_args["D_after"][s_] - d_true) * h[j]
                rows.append((rep, movie, c.cell_id, i + 1, int(first[i] + j), float(tx[j]), float(ty[j]),
                             float(tx[j]) / um, float(ty[j]) / um, s_ + 1, d_true, float(tz[j]), bool(dark[i, j]),
                             float(clock[j]), float(h[j])))


def _finish_movie_condition(recipe, condition, cond_dir, root, rows, cell_rows, crops, day_map, additions,
                            cond_truth) -> None:
    """A movie condition's records: the crop (every frame), the optics the preparation window reads, the
    replicates, the addition times of a kinetic condition, and the truth tables."""
    import pandas as pd

    (cond_dir / "bleaching").mkdir(parents=True, exist_ok=True)
    (cond_dir / "bleaching" / "crops.json").write_text(json.dumps({
        "crops": crops, "pixel_size_nm": recipe.pixel_size_nm, "frame_interval_s": recipe.tau_s,
        "exposure_s": None if recipe.exposure_ms is None else recipe.exposure_ms / 1000.0,
        "emission_nm": recipe.emission_nm, "numerical_aperture": recipe.numerical_aperture,
        "note": [f"SIMULATED ({recipe.name}): the whole movie is the crop"]}, indent=2), encoding="utf-8")
    (cond_dir / "day_map.json").write_text(json.dumps({
        "movie_to_day": day_map, "source": "synthetic", "series_per_day": recipe.movies_per_replicate,
        "warnings": [], "written_by": "smtracker.generator.build"}, indent=2), encoding="utf-8")
    table = pd.DataFrame(rows, columns=MOVIE_COLUMNS)
    if condition.kinetics is not None:
        from .kinetics_designs import save_culture_times

        save_culture_times(cond_dir, additions)
        _write_kinetic_truth(root, condition, table)
        cond_truth["kinetics"] = {**_to_plain(asdict(condition.kinetics)),
                                  "additions": {str(k): v for k, v in additions.items()}}
    else:
        table = table.drop(columns=["time_since_addition_s", "response"])
    with gzip.open(root / "truth" / f"{condition.name}_localisations.csv.gz", "wt", encoding="utf-8") as fh:
        table.to_csv(fh, index=False)
    pd.DataFrame(cell_rows).to_csv(root / "truth" / f"{condition.name}_cells.csv", index=False)


def _write_kinetic_truth(root: Path, condition: Condition, table) -> None:
    """``truth/<condition>_kinetics.csv``: per `KINETIC_BIN_S` bin of the addition's clock, the localisations,
    the counted share of every population, the mean true D of each, and the mean response h."""
    import pandas as pd

    k = len(condition.populations)
    t = table["time_since_addition_s"]
    edges = np.floor(t / KINETIC_BIN_S) * KINETIC_BIN_S
    out = []
    for start, part in table.groupby(edges, sort=True):
        row = {"bin_start_s": float(start), "bin_end_s": float(start) + KINETIC_BIN_S, "localisations": len(part),
               "response_mean": float(part["response"].mean())}
        counts = np.bincount(part["true_population"].to_numpy() - 1, minlength=k)
        for i, p in enumerate(condition.populations):
            name = p.name or f"population {i + 1}"
            row[f"share {name}"] = float(counts[i] / max(1, len(part)))
            sub = part[part["true_population"] == i + 1]
            row[f"D {name}"] = float(sub["true_D_um2_s"].mean()) if len(sub) else float("nan")
        out.append(row)
    pd.DataFrame(out).to_csv(root / "truth" / f"{condition.name}_kinetics.csv", index=False)


def describe(recipe: Recipe) -> str:
    """One paragraph a person reads: what the recipe will build."""
    lines = []
    for c in recipe.conditions:
        pops = "; ".join(f"{p.name or f'population {i + 1}'}: D {p.D_um2_s:g} um2/s, {_REGION_WORDS[p.where]}"
                         + ("" if c.rates_per_s is not None else f", {100 * p.share:.0f} %")
                         for i, p in enumerate(c.populations))
        switching = ("no switching" if rates_matrix(c) is None else "switching")
        lines.append(f"{c.name}: {pops} ({switching})")
    return (f"{len(recipe.conditions)} condition(s), {recipe.replicates} replicates of {recipe.movies_per_replicate} "
            f"movies, {recipe.cells_per_movie} cells per movie, {recipe.n_frames} frames of "
            f"{recipe.frame_interval_ms:g} ms, {recipe.pixel_size_nm:g} nm pixels. " + " | ".join(lines))
