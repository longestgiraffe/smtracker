"""The preparation window's setting guides: what each setting DOES, as a picture.

Asked for: "it would be good to have helpers and animation like in analysis as well,
especially for tracking and mb some image prep". The analysis does this in the Spatial
Map's Run dialog (`ui.workspace.spatial_map_chart._SettingArt`, words in
`spatial_map_view.setting_help`): a column of settings that keeps every value on screen,
beside a live MECHANISM diagram that moves when the value moves, and an info line whose
longer words are a tooltip. This module is the headless half of the same thing for
preparation and tracking; `ui.prep.guide` paints it.

THE HOUSE RULE FOR THESE PICTURES, carried over: an invented input is labelled invented,
and every count on a picture is COMPUTED from what the picture draws, never typed. Where
the open condition supplies a real number -- its measured noise, its optics, its frame
interval, its movie's curve and floors -- the picture uses it and says so, because the
point of a guide is to show what the setting does to THIS data.

Pure: numpy and the project's own functions, no Qt (`tests/test_headless_spine.py`).
Every payload is plain lists and floats, for a widget that imports no numpy.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

__all__ = ["TRACKING_KEYS", "CROP_KEYS", "NAMES", "guide_text", "value_text",
           "threshold_picture", "radius_picture", "linking_picture", "gap_picture",
           "tolerance_picture", "start_rule_picture", "end_picture", "picture"]

#: The Detect & link guide, in the order the grid shows its columns.
TRACKING_KEYS = ("threshold", "radius_px", "linking_max_distance_px", "max_frame_gap")
#: The Movies & crop guide.
CROP_KEYS = ("start_rule", "flat_tolerance", "end")

NAMES = {
    "threshold": "Detection threshold",
    "radius_px": "Spot radius",
    "linking_max_distance_px": "Linking radius",
    "max_frame_gap": "Frame gap",
    "start_rule": "Start rule",
    "flat_tolerance": "Flat within",
    "end": "Every movie ends at",
}

#: (one sentence under the title, the info line, its tooltip). The sentence says what
#: the CHOICE is; the tooltip carries the mechanism and the measurement behind it.
_TEXT = {
    "threshold": (
        "How bright a spot must be, against this movie's own noise, to count as a "
        "molecule. Each movie is detected at its own threshold.",
        "Why it is measured",
        "The threshold is k times each movie's LoG noise, on TrackMate's spot-quality scale. "
        "The default k is 7.2, the strict end of the usual range (3-5). A noise 1.5 times the "
        "median or more suggests two sessions in one condition. A value you type applies to "
        "every movie."),
    "radius_px": (
        "How big a molecule looks: the detector's kernel, matched to the microscope's "
        "spot size from the emission wavelength, the NA and the pixel size.",
        "Why it is not the linking radius",
        "Spot radius comes from the optics (3 PSF sigmas: 2.80 px for mNeonGreen at NA "
        "1.49 and 106 nm pixels); linking radius comes from how far molecules move in one "
        "frame. The two can differ fivefold between proteins."),
    "linking_max_distance_px": (
        "How far a molecule may move between two frames and still be joined into the "
        "same track. A longer jump starts a new track.",
        "Why 5 px, and what it costs",
        "5 px reproduced a reference u-track analysis on six proteins (slow fraction "
        "within 1.4 pp); 12 px inflated it by 12.8 pp. Wider radii add wrong links, which "
        "look like slow steps. The fast-state D is a bound at any radius."),
    "max_frame_gap": (
        "How many frames a molecule may vanish for and still be joined back into its "
        "track.",
        "Why 3, and why it is a time",
        "TrackMate's detector flickers near threshold: with no gap closing the median "
        "track has 2 localisations (6 for u-track); at a gap of 3 the counts agree. A gap of 0 "
        "shifts the slow fraction by several points. The same gap is a longer time at a slower "
        "frame rate, so it is also shown in ms."),
    "start_rule": (
        "Where each movie's crop begins: after the dense, bleaching start, once the "
        "field has settled to single molecules.",
        "What each rule reads",
        "Strictest uses the longest flat stretch of the curve, or the latest floor if "
        "none is flat. Density: spots per frame fall to a resolvable level. Amplitude: "
        "single-molecule brightness is reached. Field mean: the frame mean reaches its "
        "tail. You can edit the proposed range; 'Cut all starts' sets one frame for every "
        "movie."),
    "flat_tolerance": (
        "How flat 'flat' is: the band, as a share of the curve's own level, inside which "
        "the frame mean must stay for the field to count as settled.",
        "Why a camera offset moves it",
        "The band is a share of the raw counts, camera offset included. 2 % suits a small "
        "offset; with a large offset use 0.25-0.5 %, or the floor lands too early. The "
        "floor line warns when the band exceeds a quarter of the fluorescence."),
    "end": (
        "Where every movie's crop ends. Late frames add little: the molecules have "
        "bleached away while the background has not.",
        "Why the tail is worth cutting, and how much",
        "Real detections fall along a movie as fluorophores bleach; background "
        "detections (camera noise, autofluorescence) do not, so late frames carry more "
        "background. 'Measure from the tracks here' reads this condition's tracks and "
        "outlines: per block of frames, the outside rate as a share of the in-cell rate, "
        "and how much in-cell data the block holds. The outside rate is a lower bound on "
        "the false rate inside cells, so the share is a floor."),
}


def guide_text(key: str) -> dict:
    """``{title, desc, info, tip}`` for one setting's guide."""
    desc, info, tip = _TEXT[key]
    return {"title": NAMES[key], "desc": desc, "info": info, "tip": tip}


def value_text(key: str, state, *, rule: str | None = None,
               tolerance: float | None = None, end: int | None = None) -> str:
    """What the nav row shows under a setting's name, for the condition in ``state``."""
    if key in TRACKING_KEYS:
        value = state.tracking.get(key) if state is not None else None
        if value is None:
            return "not set"
        if key == "max_frame_gap":
            return f"{int(value)} frames"
        if key == "threshold":
            how = ("typed" if "threshold" in state.typed else
                   "per movie" if state.noise else "")
            return f"{float(value):.0f}" + (f" · {how}" if how else "")
        return f"{float(value):.2f} px"
    if key == "start_rule":
        return (rule or "strictest").replace("_", " ")
    if key == "flat_tolerance":
        return f"{100 * float(tolerance if tolerance is not None else 0.02):g} %"
    if key == "end":
        return f"frame {int(end)}" if end else "not set"
    raise KeyError(key)


# ---------------------------------------------------------------- detection threshold
#: The invented LoG trace: five spots at these heights (in units of the noise SD).
_SPOT_X = (0.12, 0.30, 0.48, 0.67, 0.86)
#: Invented heights in noise SDs, chosen so the rule's 7.2 (TrackMate's scale) finds three
#: of five with no false crossing, about the measured recall of ~0.7 at matched thresholds.
_SPOT_H = (4.5, 6.0, 8.5, 12.0, 16.0)
_SPOT_W = 0.010


@lru_cache(maxsize=1)
def _invented_trace(n: int = 360):
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(11)
    noise = gaussian_filter1d(rng.normal(0.0, 1.0, n), 1.6)
    noise = (noise - noise.mean()) / noise.std()
    x = np.linspace(0.0, 1.0, n)
    trace = noise.copy()
    for c, h in zip(_SPOT_X, _SPOT_H):
        trace += h * np.exp(-0.5 * ((x - c) / _SPOT_W) ** 2)
    return x, trace


def _threshold_counts(x, trace, k: float) -> dict:
    """What a cut at ``k`` does to the drawn trace: spots found, spots missed, and noise
    excursions crossing it away from every spot (false detections)."""
    near = np.zeros(x.size, dtype=bool)
    found = 0
    for c in _SPOT_X:
        window = np.abs(x - c) <= 2.5 * _SPOT_W
        near |= np.abs(x - c) <= 4.0 * _SPOT_W
        found += bool(trace[window].max() >= k)
    above = (trace >= k) & ~near
    false = int(np.sum(above[1:] & ~above[:-1]) + int(above[0]))
    return {"found": found, "missed": len(_SPOT_X) - found, "false": false}


def threshold_picture(state) -> dict:
    """The cut through an invented LoG trace, at the k this condition's threshold means.

    k is the threshold over the condition's MEDIAN measured noise when there is one, so a
    typed 60 against a noise of 10.8 draws a line at 5.6 noise SDs and misses spots it
    would have found at the rule's 7.2. Before a measurement it is the rule's own k."""
    from .trackmate_run import NOISE_FLAG_MIN_RATIO, UTRACK_MATCHED_K

    x, trace = _invented_trace()
    noise = getattr(state, "noise", None) if state is not None else None
    threshold = state.tracking.get("threshold") if state is not None else None
    if noise and threshold:
        k = float(threshold) / float(noise["sigma_median"])
        source = (f"{float(threshold):.1f} / median noise {noise['sigma_median']:.1f} "
                  f"= {k:.1f} noise SDs")
    else:
        k = float(UTRACK_MATCHED_K)
        source = (f"the rule: {UTRACK_MATCHED_K:g} x each movie's own noise "
                  + ("(measure to see a typed value against it)" if threshold else
                     "(not measured yet)"))
    movies = []
    if noise:
        median = float(noise["sigma_median"])
        for row in noise.get("per_movie", ()):
            if row.get("ok"):
                ratio = float(row["sigma_log"]) / median
                movies.append({"movie": row["movie"], "sigma": float(row["sigma_log"]),
                               "ratio": ratio,
                               "odd": bool(ratio >= NOISE_FLAG_MIN_RATIO
                                           or ratio <= 1 / NOISE_FLAG_MIN_RATIO)})
    counts = _threshold_counts(x, trace, k)
    # WHAT THE PAINTER COMPARES AGAINST A MOVING LINE, worked out here so it only compares:
    # each spot's peak, and every local maximum of the noise away from the spots.
    near = np.zeros(x.size, dtype=bool)
    peaks = []
    for c in _SPOT_X:
        window = np.abs(x - c) <= 2.5 * _SPOT_W
        near |= np.abs(x - c) <= 4.0 * _SPOT_W
        peaks.append(float(trace[window].max()))
    inner = np.arange(1, x.size - 1)
    local = inner[(trace[inner] >= trace[inner - 1]) & (trace[inner] > trace[inner + 1])
                  & ~near[inner] & (trace[inner] > 1.5)]
    noise_peaks = [{"x": float(x[i]), "y": float(trace[i])} for i in local]
    lo = min(-3.0, float(trace.min()))
    # THE TALLEST SPOT INSIDE THE PLOT, with room above it for the "spots found" line (tester B24, 3 Nov 2026): the
    # top was k + 1, so the two tallest invented peaks and their filled markers were drawn past the plot's top, and
    # "3 of 5 spots found (filled)" showed one filled
    top = max(float(trace.max()), max(peaks) if peaks else float(trace.max()))
    hi = max(10.0, k + 1.0, top + 0.2 * (top - lo))
    return {"key": "threshold", "x": x.tolist(), "trace": trace.tolist(),
            "spots": [{"x": c, "h": h, "peak": pk} for c, h, pk in zip(_SPOT_X, _SPOT_H, peaks)],
            "noise_peaks": noise_peaks,
            "k": k, "range": (lo, hi), "counts": counts, "movies": movies,
            "readout": (f"Line at {source}. {counts['found']} of the {len(_SPOT_X)} "
                        f"invented spots reach it and {counts['missed']} fall short; noise "
                        f"reaches it {counts['false']} time(s) on this strip.")}


# ---------------------------------------------------------------- spot radius
def radius_picture(state) -> dict:
    """The microscope's spot on a pixel grid, and the kernel radius over it."""
    from .optics import OpticsError, psf_sigma_nm

    cal = state.calibration if state is not None else {}
    radius = state.tracking.get("radius_px") if state is not None else None
    emission, aperture, pixel = (cal.get("emission_nm"), cal.get("numerical_aperture"),
                                 cal.get("pixel_size_nm"))
    psf_px = None
    note = ""
    if emission and aperture and pixel:
        try:
            psf_nm = psf_sigma_nm(emission_nm=float(emission),
                                  numerical_aperture=float(aperture))
            psf_px = psf_nm / float(pixel)
            note = (f"PSF sigma {psf_nm:.1f} nm = {psf_px:.2f} px (emission "
                    f"{float(emission):.0f} nm, NA {float(aperture):.2f}, "
                    f"{float(pixel):.0f} nm pixels)")
        except OpticsError as exc:
            note = f"the optics do not give a spot: {exc}"
    else:
        missing = [n for n, v in (("emission", emission), ("NA", aperture),
                                  ("pixel size", pixel)) if not v]
        note = (f"set {', '.join(missing)} on Calibration & save to derive it; the spot "
                f"drawn is an invented one")
    drawn_psf = psf_px if psf_px else (float(radius) / 3.0 if radius else 0.93)
    drawn_radius = float(radius) if radius else 3.0 * drawn_psf
    sigmas = drawn_radius / drawn_psf
    readout = (f"{note}. Radius {drawn_radius:.2f} px = {sigmas:.1f} spot sigmas"
               + ("" if radius else " (no radius set; 3 sigmas drawn)") + ".")
    half = int(math.ceil(drawn_radius + 1.5))
    return {"key": "radius_px", "psf_px": drawn_psf, "radius": drawn_radius,
            "half": half, "derived": psf_px is not None, "readout": readout}


# ---------------------------------------------------------------- linking radius
@lru_cache(maxsize=1)
def _invented_walk(n_steps: int = 26):
    """An invented molecule, steps Rayleigh-distributed so a few exceed 5 px."""
    # SEED 52 IS CHOSEN, and for the picture: at 5 px this walk breaks twice (two usable
    # tracks and a two-localisation fragment the analysis drops), at 8 px never, at 3 px
    # seven times -- so the default radius shows the mechanism instead of hiding it.
    rng = np.random.default_rng(52)
    steps = rng.rayleigh(2.2, n_steps)
    angles = rng.uniform(0.0, 2 * math.pi, n_steps)
    xy = np.vstack([[0.0, 0.0], np.cumsum(np.c_[steps * np.cos(angles),
                                                 steps * np.sin(angles)], axis=0)])
    xy -= xy.mean(axis=0)
    # LAID ALONG ITS LONG AXIS, to fill a wide plate. A rotation changes no step length,
    # so no break moves.
    _u, _s, vt = np.linalg.svd(xy, full_matrices=False)
    xy = xy @ vt.T
    return xy, steps


def _segments(n_points: int, breaks) -> list:
    """Localisations per track when the walk is cut before every step in ``breaks``."""
    cuts = [0] + [i + 1 for i in breaks] + [n_points]
    return [b - a for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def linking_picture(state) -> dict:
    """An invented walk linked at this radius: every step longer than the radius breaks
    the track, and only tracks of at least five localisations reach the analysis."""
    from .trackmate_run import MIN_USABLE_LOCALISATIONS, apparent_D_ceiling, max_measurable_D

    radius = (state.tracking.get("linking_max_distance_px") if state is not None
              else None) or 5.0
    xy, steps = _invented_walk()
    breaks = [i for i, s in enumerate(steps) if s > float(radius)]
    segments = _segments(len(xy), breaks)
    usable = sum(1 for n in segments if n >= MIN_USABLE_LOCALISATIONS)
    readout = (f"This invented walk ({len(xy)} localisations) breaks {len(breaks)} "
               f"time(s) at {float(radius):g} px: {len(segments)} track(s), {usable} "
               f"long enough to use (>= {MIN_USABLE_LOCALISATIONS}).")
    cal = state.calibration if state is not None else {}
    tau, pixel = cal.get("frame_interval_s"), cal.get("pixel_size_nm")
    if tau and pixel:
        reach = max_measurable_D(radius_px=float(radius), tau_s=float(tau),
                                 pixel_size_nm=float(pixel))
        ceiling = apparent_D_ceiling(radius_px=float(radius), tau_s=float(tau),
                                     pixel_size_nm=float(pixel))
        readout += (f" At {float(tau) * 1000:.4g} ms this radius measures D up to "
                    f"{reach:.2f} µm²/s within 10 %; a faster population reads low, "
                    f"never above {ceiling:.2f}.")
    else:
        readout += " Set the frame interval and pixel size to see what it can measure."
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return {"key": "linking_max_distance_px", "points": xy.tolist(),
            "steps": steps.tolist(), "radius": float(radius), "breaks": breaks,
            "segments": segments, "usable": usable,
            "bounds": [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])],
            "readout": readout}


# ---------------------------------------------------------------- frame gap
#: The invented molecule's 30 frames: it vanishes for 1, 2, 3 and 4 frames.
_ABSENT = {5, 10, 11, 17, 18, 19, 24, 25, 26, 27}
_GAP_FRAMES = 30


def gap_picture(state) -> dict:
    """An invented molecule that vanishes for 1, 2, 3 and 4 frames, linked at this gap."""
    from .trackmate_run import MIN_USABLE_LOCALISATIONS, gap_window

    gap = state.tracking.get("max_frame_gap") if state is not None else None
    gap = int(gap) if gap is not None else 3
    present = [f not in _ABSENT for f in range(_GAP_FRAMES)]
    tracks, start, last, missing = [], None, None, 0
    for f, here in enumerate(present):
        if here:
            if start is None:
                start = f
            elif missing > gap:
                tracks.append((start, last))
                start = f
            last, missing = f, 0
        elif start is not None:
            missing += 1
    if start is not None:
        tracks.append((start, last))
    locs = [sum(present[a:b + 1]) for a, b in tracks]
    usable = sum(1 for n in locs if n >= MIN_USABLE_LOCALISATIONS)
    readout = (f"At a gap of {gap}, this invented molecule ({sum(present)} "
               f"localisations) is {len(tracks)} track(s) of "
               f"{', '.join(str(n) for n in locs)}; {usable} reach "
               f"{MIN_USABLE_LOCALISATIONS} localisations.")
    tau = state.calibration.get("frame_interval_s") if state is not None else None
    if tau:
        window = gap_window(gap_frames=gap, tau_s=float(tau))
        readout += (f" {gap} frame(s) at {float(tau) * 1000:.4g} ms is "
                    f"{window['unobserved_ms']:.0f} ms unobserved.")
    return {"key": "max_frame_gap", "present": present, "gap": gap,
            "tracks": [list(t) for t in tracks], "locs": locs, "usable": usable,
            "readout": readout}


# ---------------------------------------------------------------- the crop
def _movie_curve(curves):
    """``(frames, means, floor)`` of a movie the window has read, or an invented one."""
    if curves is not None and getattr(curves, "bleaching", None) is not None:
        from .prep_view import reduce_series

        frames, means = reduce_series(curves.bleaching["frame"], curves.bleaching["mean_counts"],
                                      max_points=400)
        return frames, means, curves.floor, curves.movie
    n = 3000
    frames = np.arange(1, n + 1, dtype=float)
    means = 507.0 + 1500.0 * np.exp(-frames / 260.0) + 160.0 * np.exp(-frames / 2500.0)
    return frames.tolist(), means.tolist(), None, None


def tolerance_picture(curves, tolerance: float) -> dict:
    """This movie's frame mean, the flat band at ``tolerance`` and where it starts, with
    the camera's dark level under it -- the reason a band of raw counts can swallow a
    decay. An invented Nikon-like curve before a movie is read."""
    from .prep_model import refloor

    tolerance = float(tolerance)
    frames, means, floor, movie = _movie_curve(curves)
    if curves is not None and floor is not None:
        floor = refloor(curves, tolerance).floor
        start, dark, share = floor.by_flat, floor.dark_level, floor.flat_band_share
    else:
        from .movieprep import FLAT_MIN_FRAMES, leading_dark_level, longest_flat_stretch

        arr = np.asarray(means, dtype=float)
        flat = longest_flat_stretch(arr, tolerance=tolerance, min_length=FLAT_MIN_FRAMES)
        start = None if flat is None else int(frames[flat[0]])
        dark = 507.0 if leading_dark_level(arr) is None else leading_dark_level(arr)
        level = float(np.median(arr[flat[0]:flat[1]])) if flat else float(arr[-1])
        share = tolerance * level / (level - dark) if level > dark else None
    arr = np.asarray(means, dtype=float)
    after = arr[np.asarray(frames, dtype=float) >= start] if start else arr[-max(10, arr.size // 10):]
    level = float(np.median(after))
    who = f"{movie}" if movie else "an invented movie"
    readout = (f"{who}: flat within {100 * tolerance:g} % "
               + (f"starts at frame {start}" if start else "finds no flat stretch")
               + (f"; the band is {100 * share:.0f} % of the fluorescence above the "
                  f"{dark:.0f}-count dark level" if share is not None and dark else "")
               + ("; most of the decay counts as flat" if share and share > 0.25 else "")
               + ".")
    return {"key": "flat_tolerance", "frames": list(frames), "means": list(means),
            "level": level, "band": tolerance * level, "start": start,
            "dark": dark, "share": share, "tolerance": tolerance, "readout": readout}


def start_rule_picture(curves, rule: str) -> dict:
    """This movie's floors on its own frame axis, and which one the rule takes."""
    frames, means, floor, movie = _movie_curve(curves)
    if floor is not None:
        floors = {"strictest": floor.earliest_defensible, "density": floor.by_density,
                  "amplitude": floor.by_amplitude, "field_mean": floor.by_field_mean}
        n = int(floor.n_frames)
    else:
        n = int(frames[-1])
        floors = {"strictest": 1180, "density": 640, "amplitude": 410,
                  "field_mean": 1180}
    chosen = floors.get(rule)
    who = movie or "an invented movie"
    readout = (f"{who}: " + "; ".join(
        f"{k.replace('_', ' ')} {v}" if v else f"{k.replace('_', ' ')} none"
        for k, v in floors.items())
        + f". '{rule.replace('_', ' ')}' starts at "
        + (f"frame {chosen}, keeping {100 * (n - chosen + 1) / n:.0f} % of the movie."
           if chosen else "no frame: this movie gives that rule nothing."))
    return {"key": "start_rule", "frames": list(frames), "means": list(means),
            "floors": floors, "rule": rule, "chosen": chosen, "n_frames": n,
            "readout": readout}


def end_picture(curves, end, purity=None) -> dict:
    """Where the crop ends, against what the late frames hold.

    With ``purity`` (`prep_model.tail_purity`, measured from this condition's tracks and
    outlines): per block of frames, the background as a share of the in-cell rate, and
    what the in-cell data kept would be if every movie ended at ``end``. Without it, the
    open movie's spots per frame along its tail -- the density curve read at opening."""
    end = int(end) if end else None
    if purity and purity.get("rows"):
        rows = purity["rows"]
        kept = sum(r["of_all"] for r in rows if end is None or r["last"] <= end)
        partial = [r for r in rows if end is not None and r["first"] <= end < r["last"]]
        if partial:
            r = partial[0]
            kept += r["of_all"] * (end - r["first"] + 1) / (r["last"] - r["first"] + 1)
        late = [r for r in rows if end is not None and r["first"] > end]
        worst = max((r["share"] or 0.0) for r in rows)
        readout = (f"Measured on {len(purity['movies'])} movie(s): background is "
                   f"{100 * (rows[0]['share'] or 0):.1f} % of the in-cell rate in the first "
                   f"{purity['block']} frames and {100 * (rows[-1]['share'] or 0):.1f} % in "
                   f"the last block (a floor: cells autofluoresce).")
        if end is not None:
            readout += (f" Ending at frame {end} keeps {100 * min(kept, 1.0):.0f} % of the "
                        f"in-cell localisations"
                        + (f" and drops {len(late)} block(s) where background reaches "
                           f"{100 * max((r['share'] or 0) for r in late):.1f} %"
                           if late else "") + ".")
        return {"key": "end", "mode": "purity", "rows": rows, "end": end,
                "worst": worst, "kept": kept, "readout": readout}
    if curves is not None and getattr(curves, "density", None) is not None:
        frames = [float(f) for f in curves.density["frame"]]
        spots = [float(v) for v in curves.density["n_spots"]]
        who = curves.movie
    else:
        frames = [float(f) for f in range(1, 4001, 25)]
        spots = [float(6.0 * math.exp(-f / 220.0) + 0.3) for f in frames]
        who = "an invented movie"
    readout = (f"{who}: spots per frame along the movie (the density curve read at opening). "
               f"Press 'Measure from the tracks here' once this condition is tracked and "
               f"outlined to see the background's share block by block.")
    return {"key": "end", "mode": "density", "frames": frames, "spots": spots, "end": end,
            "readout": readout}


def picture(key: str, state=None, *, curves=None, rule: str = "strictest",
            tolerance: float = 0.02, end=None, purity=None) -> dict:
    """The payload for one setting's picture."""
    if key == "threshold":
        return threshold_picture(state)
    if key == "radius_px":
        return radius_picture(state)
    if key == "linking_max_distance_px":
        return linking_picture(state)
    if key == "max_frame_gap":
        return gap_picture(state)
    if key == "flat_tolerance":
        return tolerance_picture(curves, tolerance)
    if key == "start_rule":
        return start_rule_picture(curves, rule)
    if key == "end":
        return end_picture(curves, end, purity)
    raise KeyError(key)
