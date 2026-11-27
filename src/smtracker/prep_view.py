"""What the preparation stage SAYS, built from `prep_model`'s data: plain strings and lists.

The window draws these and decides nothing; the same text reaches the old
`tracking_gui`/`croppanel` window and the workspace-style one being built beside it, so
the two cannot word a result differently. Imports no Qt.
"""
from __future__ import annotations

from .prep_model import CutReport

__all__ = ["common_cut_note", "common_cut_log", "outline_note", "outlines_log",
           "outlines_prompt", "threshold_setting", "calibration_note", "condition_note",
           "mismatch_log", "CHART_POINTS", "reduce_series", "crop_chart", "movie_rows",
           "condition_status", "sidebar_row", "floor_note", "propose_note", "START_RULES",
           "CALIBRATION_COLUMNS", "calibration_cells", "save_state", "fluorophore_names",
           "interval_note", "field_names", "TRACKING_COLUMNS", "tracking_cells",
           "tracking_status", "noise_lines", "reach_lines", "crowding_line",
           "layout_note", "fiji_lines", "track_rows", "progress_line", "run_header",
           "export_rows",
           "export_line", "radius_check_lines", "pilot_lines", "pixel_size_line",
           "TRACKERS", "UTRACK_FIELDS", "UTRACK_GROUPS", "UTRACK_CHOICES",
           "UTRACK_STARTING", "utrack_cells", "utrack_differences", "utrack_engine_lines",
           "utrack_condition_rows", "utrack_track_rows", "utrack_progress_line",
           "utrack_run_header"]


def _proposed_words(report: CutReport) -> str:
    """What happened first to the read movies that had no range yet (`apply_common_cut`)."""
    words = ""
    if report.proposed:
        words += f" {len(report.proposed)} had no range yet and start where the rule puts them."
    if report.unfloored:
        words += (f" {len(report.unfloored)} start at frame 1 because the rule found no flat "
                  f"stretch in them: " + ", ".join(report.unfloored[:3]) + ".")
    return words


def common_cut_note(report: CutReport) -> str:
    """The crop panel's one line after "Cut all starts", "Cut all tails" or "Track N frames"."""
    if report.length is not None:
        example = next(((stem, a, b) for stem, (a, b) in report.crops.items() if stem in report.cut), None)
        message = f"{len(report.cut)} movie(s) now track {report.length} frames from their own start"
        if example:
            message += f" ({example[0]}: {example[1]}-{example[2]})"
        if report.shorter:
            message += (f"; {len(report.shorter)} are shorter and end at their own last frame: "
                        + ", ".join(report.shorter[:3]))
        if report.skipped:
            message += f"; {len(report.skipped)} left alone"
        return message + "." + _proposed_words(report) + " Movies read later get the same."
    if report.end is not None:
        message = f"{len(report.cut)} movie(s) now end at frame {report.end}"
        if report.shorter:
            message += (f"; {len(report.shorter)} end earlier, at their own last frame: "
                        + ", ".join(report.shorter[:3]))
        if report.skipped:
            message += (f"; {len(report.skipped)} left alone, their start is past it: "
                        + ", ".join(f"{stem} (starts at {first})"
                                    for stem, first, _last in report.skipped[:3]))
        return message + "." + _proposed_words(report) + " Movies read later get the same end."
    message = f"{len(report.cut)} movie(s) now start at frame {report.start}"
    if report.skipped:
        message += (f"; {len(report.skipped)} left alone, they end before it: "
                    + ", ".join(f"{stem} (ends at {last})"
                                for stem, _first, last in report.skipped[:3]))
    return message + "." + _proposed_words(report) + " Movies read later get the same start."


def common_cut_log(reports: dict) -> str | None:
    """The log line for the conditions NOT on screen, ``{condition: CutReport}``.

    ``None`` when no other condition had a range to cut, so nothing is logged.
    """
    reports = {name: r for name, r in reports.items() if r.crops}
    if not reports:
        return None
    first = next(iter(reports.values()))
    what = (f"every movie cut at frame {first.end}" if first.end is not None
            else f"every movie tracks {first.length} frames from its own start" if first.length is not None
            else f"every movie starts at frame {first.start}")
    n_movies = sum(len(r.cut) for r in reports.values())
    line = (f"{what} in {len(reports)} more condition(s) ({n_movies} movie(s)): "
            f"{', '.join(reports)}")
    skipped = [f"{name}/{stem}" for name, r in reports.items() for stem, *_ in r.skipped]
    if skipped:
        line += f"; left alone, the range would be empty: {', '.join(skipped[:4])}"
    return line + ". Press Save to record them."


# ---------------------------------------------------------------- the condition overview
def outline_note(state) -> str:
    """The row's outline count: a COUNT, because the helpers outlines unlock degrade per
    movie and "18/30" is a real state a yes/no would hide. ``""`` with no state."""
    if state is None:
        return ""
    if not state.has_folder or not state.n_movies:
        return "   no outlines"
    if state.complete:
        return "   outlines ok"
    return f"   outlines {state.n_with_outlines}/{state.n_movies}"


def outlines_log(missing) -> str:
    """Said ONCE per set, and it names what outlines buy: an offer that only says "you
    could segment now" reads as a chore and gets skipped."""
    from .cellmesh import OUTLINES_UNLOCK

    return ("no cell outlines yet for: " + ", ".join(sorted(missing))
            + ". They are optional for tracking; with them you would get:\n  - "
            + "\n  - ".join(OUTLINES_UNLOCK))


def outlines_prompt(missing) -> str:
    """The prompt on the tracking page where outlines are missing."""
    return (f"No cell outlines for {', '.join(sorted(missing))}. Segmenting first is "
            f"optional; with outlines the threshold below is measured against the "
            f"false-positive rate outside cells, and the linking ambiguity is "
            f"counted inside cells only.")


def threshold_setting(measured, *, per_movie=None, typed=False, n_crops=0,
                      n_movies=0) -> str:
    """The detection threshold a row shows, and WHY it is absent when it is.

    A threshold measures the noise INSIDE each crop, so it cannot exist before the ranges
    do; a bare dash would read as "this failed" rather than "not its turn yet". Where the
    movies differ the SPREAD is shown: one lab project spans 2.1x in noise over its three days,
    and a single number would hide exactly that.
    """
    if measured is None:
        return ("LoG detection threshold: needs ranges first" if n_crops < n_movies
                else "LoG detection threshold: not measured")
    if typed:
        return f"LoG detection threshold {measured:.0f} (typed, all movies)"
    spread = dict(per_movie or {})
    if len(spread) > 1:
        lo, hi = min(spread.values()), max(spread.values())
        if hi > lo * 1.05:
            return (f"LoG detection threshold per movie {lo:.0f}-{hi:.0f}"
                    f" (median {measured:.0f})")
    return f"LoG detection threshold {measured:.0f}"


def calibration_note(row, differs) -> str:
    """The calibration, but only where the conditions DISAGREE about it: printing 106 nm
    on every row when they all say 106 trains the eye to skip the line."""
    odd = []
    if "pixel_size_nm" in differs and row.pixel_size_nm:
        odd.append(f"{row.pixel_size_nm:g} nm")
    if "frame_interval_s" in differs and row.frame_interval_s:
        odd.append(f"{row.frame_interval_s * 1000:g} ms")
    if "emission_nm" in differs and row.emission_nm:
        odd.append(f"em {row.emission_nm:g}")
    return ("   [" + ", ".join(odd) + "]") if odd else ""


def condition_note(counts, *, setting: str, calibration: str = "", outlines: str = "",
                   active: bool = False) -> str:
    """One overview row's text. "RANGES", not "cropped": most ranges are PROPOSED, and a
    proposal is a floor, not a decision nobody has looked at yet."""
    no_floor = f", {counts.n_bad} no crop floor" if counts.n_bad else ""
    return (f"{counts.n_curves}/{counts.n_movies} read{no_floor}, "
            f"{counts.n_crops} ranges   {setting}{calibration}{outlines}"
            + (" ←" if active else ""))


def mismatch_log(differs) -> str:
    """Said once per set: conditions not imaged alike can be TRACKED together, and their
    D values cannot be COMPARED afterwards."""
    what = {"pixel_size_nm": "pixel size (D cannot be compared at all)",
            "frame_interval_s": "frame interval (D values are not comparable)",
            "emission_nm": "fluorophore (a different PSF and spot radius)"}
    return ("these conditions were not imaged alike: "
            + "; ".join(what[k] for k in sorted(differs))
            + ". You can track them together; each uses its own numbers. "
              "Their D values should not be compared afterwards.")


# ---------------------------------------------------------------- the crop chart
#: Points a drawn curve is thinned to. Enough for a ~1200 px wide chart to show every
#: spike, few enough that repainting during a drag stays far below a frame.
CHART_POINTS = 1200


def reduce_series(frames, values, max_points: int = CHART_POINTS) -> tuple:
    """``(frames, values)`` thinned to about ``max_points``, KEEPING THE EXTREMES.

    Each bucket contributes its minimum and its maximum, in frame order, so a spike in a
    4500-frame curve survives the thinning -- a stride would step over it, and the spike
    is often exactly what the analyst is looking at (the laser switching on, a flicker).
    Plain lists of floats, for a widget that imports no numpy.
    """
    import numpy as np

    f = np.asarray(frames, dtype=float)
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(f) & np.isfinite(v)
    f, v = f[keep], v[keep]
    if f.size <= max_points:
        return f.tolist(), v.tolist()
    n_buckets = max(1, max_points // 2)
    edges = np.linspace(0, f.size, n_buckets + 1).astype(int)
    out_f, out_v = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        lo, hi = a + int(np.argmin(v[a:b])), a + int(np.argmax(v[a:b]))
        for i in sorted({lo, hi}):
            out_f.append(float(f[i]))
            out_v.append(float(v[i]))
    return out_f, out_v


def crop_chart(curves, *, crop=None, proposal=None,
               max_points: int = CHART_POINTS) -> dict:
    """Everything the crop chart draws for one movie, as plain lists and strings.

    The same three panels `plots.draw_crop_curves` has always drawn: the frame mean and
    the spots per frame over the whole acquisition (read TOGETHER -- either alone can be
    misread), and the KEPT RANGE ON ITS OWN SCALE, because beside the bleach-down spike a
    kept range that is still falling looks flat. With no crop set the third panel
    previews what the FLOOR would keep, which is the question at that moment. The floor
    is a line labelled as a floor, never a shaded "suggested region".
    """
    n = int(curves.n_frames)
    whole = crop is None or (int(crop[0]) <= 1 and int(crop[1]) >= n)
    frames = curves.bleaching["frame"].to_numpy(dtype=float)
    means = curves.bleaching["mean_counts"].to_numpy(dtype=float)
    mean_f, mean_v = reduce_series(frames, means, max_points)
    dens_f, dens_v = reduce_series(curves.density["frame"].to_numpy(dtype=float),
                                   curves.density["n_spots"].to_numpy(dtype=float),
                                   max_points)
    floor = getattr(curves, "floor", None)
    floor_start = (int(floor.earliest_defensible)
                   if floor is not None and floor.earliest_defensible is not None else None)
    if not whole:
        lo, hi, prefix = int(crop[0]), int(crop[1]), ""
    elif floor_start is not None:
        lo, hi, prefix = floor_start, n, f"preview at the floor {floor_start}: "
    else:
        lo, hi, prefix = 1, n, ""
    inside = (frames >= lo) & (frames <= hi)
    zoom_f, zoom_v = reduce_series(frames[inside], means[inside], max_points)
    verdict = ""
    try:
        from .movieprep import crop_quality

        quality = crop_quality(curves, lo, hi)
        verdict = (f"{prefix}{quality['verdict']}; "
                   f"{quality['mean_spots_per_frame']:.1f} spots/frame (mean)")
    except Exception:                                  # noqa: BLE001 - cosmetic
        pass
    return {
        "movie": str(curves.movie),
        "n_frames": n,
        "title": (f"{curves.movie}    "
                  + ("no crop set: all frames" if whole
                     else f"keeping {int(crop[0])}–{int(crop[1])}")),
        "whole": bool(whole),
        "crop": None if whole else (int(crop[0]), int(crop[1])),
        "mean": {"frames": mean_f, "values": mean_v},
        "density": {"frames": dens_f, "values": dens_v},
        "zoom": {"frames": zoom_f, "values": zoom_v, "range": (lo, hi),
                 "preview": bool(whole), "label": "preview" if whole else "kept range"},
        "floor": floor_start,
        "floor_label": (f"no earlier than {floor_start}" if floor_start is not None
                        else ""),
        "offset_warning": getattr(floor, "offset_warning", "") if floor is not None else "",
        "proposal": (None if proposal is None
                     else (int(proposal["start"]), int(proposal["end"]))),
        "verdict": verdict,
    }


# ---------------------------------------------------------------- the new window's lists
#: The start rules a proposal can use, ``(key, label)``, in `movieprep.CROP_START_RULES`'
#: order. Here so the window imports no numpy-bearing module to read a list of names.
START_RULES = (("strictest", "strictest"), ("density", "density"),
               ("amplitude", "amplitude"), ("field_mean", "field mean"))


def movie_rows(state) -> list:
    """One row per movie of a condition, for the Movies & crop list.

    ``status`` is "read", "reading" or "error"; an error is a movie that could not be
    read or proposed for, named with its reason rather than left looking pending.
    """
    rows = []
    for path, stem in zip(state.movies, state.names):
        curves = state.curves.get(stem)
        crop = state.crops.get(stem)
        if stem in state.errors:
            status, detail = "error", state.errors[stem]
        elif curves is None:
            status, detail = "reading", "reading…"
        else:
            status, detail = "read", f"{int(curves.n_frames)} frames"
        floor = getattr(getattr(curves, "floor", None), "earliest_defensible", None)
        rows.append({
            "movie": stem, "file": path.name, "status": status, "detail": detail,
            "range": f"{crop[0]}–{crop[1]}" if crop else "no range",
            "has_range": crop is not None,
            "unsaved": crop is not None and tuple(crop) != tuple(state.saved.get(stem, ())),
            "floor": None if floor is None else int(floor),
            "n_frames": None if curves is None else int(curves.n_frames),
        })
    return rows


def condition_status(state) -> str:
    """One line for a condition: how far preparation has got, and its folder's layout
    (`organise.folder_layout`'s word: organised, loose files, half-organised, nothing
    found)."""
    c = state.counts()
    text = f"{c.n_curves}/{c.n_movies} read, {c.n_crops}/{c.n_movies} ranges"
    if c.n_bad:
        text += f", {c.n_bad} not readable"
    if state.unsaved:
        text += " · unsaved"
    elif state.saved:
        text += " · saved"
    kind = getattr(getattr(state, "layout", None), "kind", "")
    if kind:
        text += f" · {kind}"
    return text


def layout_note(state) -> str:
    """The calibration page's note for a condition whose folder is NOT organised: its
    layout in full (what filing would move where), or ``""``."""
    layout = getattr(state, "layout", None)
    if layout is None or layout.kind == "organised":
        return ""
    return f"{state.name}: {layout.words}"


def sidebar_row(state) -> dict:
    """What the sidebar row of a condition shows: the bar and the line beside it."""
    c = state.counts()
    return {"name": state.name, "done": c.done, "total": max(c.n_movies, 1),
            "line": condition_status(state), "unsaved": bool(state.unsaved)}


def floor_note(curves) -> str:
    """The floor line under the chart: every criterion, the floor they give, and the
    camera-offset warning when the band is mostly offset (`CropFloor.offset_warning`)."""
    floor = getattr(curves, "floor", None)
    if floor is None:
        return ""
    line = (f"floors: density {floor.by_density} · amplitude {floor.by_amplitude} "
            f"· field mean {floor.by_field_mean} · flat {floor.by_flat}  →  no earlier "
            f"than {floor.earliest_defensible} of {floor.n_frames}")
    warning = getattr(floor, "offset_warning", "")
    return line + (f"\n{warning}" if warning else "")


#: The calibration grid's columns: ``(key, heading, unit)``. The frame interval is shown
#: and typed in ms; `crops.json` keeps seconds.
CALIBRATION_COLUMNS = (("pixel_size_nm", "Pixel size", "nm"),
                       ("frame_interval_s", "Frame interval", "ms"),
                       ("label", "Fluorophore", ""),
                       ("emission_nm", "Emission", "nm"),
                       ("numerical_aperture", "NA", ""))


def calibration_cells(state) -> dict:
    """``{key: text}`` for one condition's row; ``""`` where nothing is recorded."""
    cal = state.calibration
    out = {}
    for key, _heading, _unit in CALIBRATION_COLUMNS:
        value = cal.get(key)
        if value is None:
            out[key] = ""
        elif key == "frame_interval_s":
            out[key] = f"{float(value) * 1000:g}"
        elif key == "label":
            out[key] = str(value)
        else:
            out[key] = f"{float(value):g}"
    return out


def save_state(state) -> tuple:
    """``(text, ready)`` for a condition's Save: what is missing, or what it would do."""
    from .prep_model import REQUIRED_CALIBRATION, words_list

    # IN WORDS (tester B12, 3 Nov 2026): stripping "_s" from "pixel_size" read "needs pixelize"
    gaps = [k for k in REQUIRED_CALIBRATION if state.calibration.get(k) is None]
    if gaps:
        return f"needs {words_list(gaps)}", False
    counts = state.counts()
    if not counts.n_crops:
        return "no ranges yet", False
    if counts.n_crops < counts.n_movies:
        missing = counts.n_movies - counts.n_crops
        return f"{missing} movie(s) without a range", True
    return ("unsaved changes" if state.unsaved
            else "saved" if state.saved else "ready to save"), True


def field_names(keys) -> str:
    """Calibration keys as words: ``frame_interval_s`` is "frame interval"."""
    words = {key: heading.lower() if key != "numerical_aperture" else "NA"
             for key, heading, _unit in CALIBRATION_COLUMNS}
    return ", ".join(words.get(k, k) for k in keys)


def interval_note(name: str, check) -> str:
    """Said when a RECORDED frame interval disagrees with what the acquisition measured,
    with the factor it would put on every D; ``""`` when it agrees or cannot be checked."""
    if not check or check.get("agrees") is not False:
        return ""
    from .chart_text import fold_words

    rec, meas, ratio = check["recorded_s"], check["measured_s"], check["ratio"]
    return (f"{name}: the saved frame interval is {rec * 1000:g} ms but the acquisition "
            f"measures {meas * 1000:.4f} ms. Every D from this condition would be off "
            f"by {fold_words(ratio, 2)}. Correct it before tracking.")


def fluorophore_names() -> tuple:
    """The label presets, by name (`optics.LABEL_PRESETS`)."""
    from .optics import preset_names

    return tuple(preset_names())


def propose_note(done, failed, rule: str, pending: int = 0) -> str:
    """After "Propose for all": how many, by which rule, and which had no floor."""
    note = f"proposed {rule} ranges for {len(done)} movie(s); you can edit every value"
    if pending:
        note += f". {pending} still being read get the same rule as they arrive"
    if failed:
        note += f". No proposal for {len(failed)}: " + "; ".join(
            f"{stem} ({why})" for stem, why in failed[:2])
    return note


# ---------------------------------------------------------------- detection and linking
#: The tracking grid's columns: ``(key, heading, unit)``, in `TRACKING_FIELDS`' order.
TRACKING_COLUMNS = (("threshold", "LoG threshold (median)", ""),
                    ("radius_px", "Spot radius", "px"),
                    ("linking_max_distance_px", "Linking radius", "px"),
                    ("max_frame_gap", "Frame gap", "frames"))


def tracking_cells(state) -> dict:
    """``{key: text}`` for one condition's tracking row; ``""`` where nothing is set."""
    out = {}
    for key, _heading, _unit in TRACKING_COLUMNS:
        value = state.tracking.get(key)
        if value is None:
            out[key] = ""
        elif key == "max_frame_gap":
            out[key] = f"{int(value)}"
        elif key == "threshold":
            out[key] = f"{float(value):.0f}"
        else:
            out[key] = f"{float(value):.2f}"
    return out


def tracking_status(state) -> str:
    """What the row's threshold IS: per movie (with the spread), typed, or why absent."""
    from .prep_model import measurable, tracking_gaps

    if "threshold" in state.typed:
        return "typed: one threshold for every movie"
    noise = state.noise
    if noise:
        rows = [r for r in noise.get("per_movie", ()) if r.get("ok")]
        if len(rows) > 1:
            lo = min(r["threshold"] for r in rows)
            hi = max(r["threshold"] for r in rows)
            text = f"per movie {lo:.0f}-{hi:.0f}"
        else:
            text = "measured"
        if noise.get("mixed_sessions"):
            from .chart_text import fold_words

            text += f" · noise spans {fold_words(noise['sigma_ratio'], 1)}"
        return text
    if not measurable(state):
        return "needs a range on every movie first"
    from .prep_model import words_list

    gaps = [g for g in tracking_gaps(state) if g != "threshold"]
    return "not measured" + (f" · set the {words_list(gaps)}" if gaps else "")


def noise_lines(state) -> list:
    """The measurement, movie by movie, and the warning when the batch is mixed --
    naming the movies unlike their neighbours (`trackmate_run.noise_outliers`), never
    excluding any."""
    noise = state.noise
    if not noise:
        return []
    lines = [f"{state.name}: threshold {noise['threshold']:.1f} = {noise['k']:g} x the "
             f"median LoG noise ({noise['sigma_median']:.1f}) over {noise['n_movies']} "
             f"movie(s) at spot radius {noise['radius_px']:.2f} px"]
    failed = [r for r in noise["per_movie"] if not r["ok"]]
    if failed:
        lines.append(f"  {len(failed)} could not be measured: "
                     + "; ".join(f"{r['movie']} ({r['error']})" for r in failed[:3]))
    if noise.get("mixed_sessions"):
        from .chart_text import fold_words

        lines.append(f"  the noise spans {noise['sigma_min']:.1f} to {noise['sigma_max']:.1f}"
                     f" ({fold_words(noise['sigma_ratio'], 2)}), more than one session's spread; "
                     f"each movie is detected at its own threshold")
        from .trackmate_run import TrackMateRunError, noise_outliers

        try:
            odd = noise_outliers(noise["per_movie"])
            if odd.attrs.get("n_flagged"):
                for row in odd[odd["noise_flagged"]].itertuples():
                    lines.append(f"    {row.movie}: noise {row.sigma_log:.1f} "
                                 f"({fold_words(row.sigma_ratio_to_median, 2, of=True)} the median); check "
                                 f"this one. Nothing is excluded")
        except TrackMateRunError as exc:
            lines.append(f"    the movies could not be compared: {exc}")
    return lines


def reach_lines(state) -> list:
    """What the linking radius and frame gap MEAN for this acquisition: the fastest D
    the radius measures within 10 %, the ceiling a fit can never exceed, and the gap as
    a TIME. Nothing is printed when the frame interval or pixel size is missing -- a
    number describing a default is the error this project keeps paying for."""
    from .trackmate_run import apparent_D_ceiling, gap_window, max_measurable_D

    tau = state.calibration.get("frame_interval_s")
    pixel = state.calibration.get("pixel_size_nm")
    radius = state.tracking.get("linking_max_distance_px")
    gap = state.tracking.get("max_frame_gap")
    if not tau or not pixel:
        return [f"{state.name}: set the frame interval and pixel size to see what the "
                f"linking radius can measure"]
    lines = []
    if radius:
        reach = max_measurable_D(radius_px=float(radius), tau_s=float(tau),
                                 pixel_size_nm=float(pixel))
        ceiling = apparent_D_ceiling(radius_px=float(radius), tau_s=float(tau),
                                     pixel_size_nm=float(pixel))
        lines.append(f"{state.name}: at {float(tau) * 1000:.4g} ms a {float(radius):g} px "
                     f"radius measures D up to {reach:.2f} µm²/s within 10 %; a faster "
                     f"population reads low, never above {ceiling:.2f}")
    if gap is not None:
        window = gap_window(gap_frames=int(gap), tau_s=float(tau))
        lines.append(f"{state.name}: a gap of {int(gap)} frame(s) is "
                     f"{window['unobserved_ms']:.0f} ms unobserved")
    return lines


def crowding_line(state) -> str:
    """How busy the field is over the TRACKED range, at the linking radius the run will
    use (snapped to the recorded ladder), and which movies are unlike the rest. From the
    curves already held; ``""`` when there is nothing to say yet."""
    from .movieprep import CROWDING_RADII_PX, MoviePrepError, crowding_summary

    if not state.curves:
        return ""
    wanted = float(state.tracking.get("linking_max_distance_px") or 5.0)
    radius = min(CROWDING_RADII_PX, key=lambda r: abs(r - wanted))
    try:
        got = crowding_summary(dict(state.curves), radius_px=radius,
                               ranges=dict(state.crops))
    except MoviePrepError:
        return ""
    if got.empty:
        return ""
    bad = got[got["flagged"]]
    line = (f"{state.name}: {got.attrs['median_loc_per_frame']:.1f} detections/frame over "
            f"the tracked range, {100 * got.attrs['median_crowded_share']:.0f} % with "
            f"another within {radius:g} px")
    if len(bad):
        line += (f"; {len(bad)} movie(s) unlike the rest: "
                 + ", ".join(f"{r.movie} ({r.reason})" for r in bad.itertuples()))
    return line


# ---------------------------------------------------------------- the Track page
def fiji_lines(install, *, absence: str = "") -> list:
    """What the Fiji card says: the install in use, whether it can track, and any other
    install seen -- or, with none, where it looked and the way out (`absence`)."""
    if install is None:
        return [line.strip() for line in absence.splitlines() if line.strip()] or [
            "No Fiji installation was found."]
    lines = [f"Fiji at {install.root}"]
    if install.has_trackmate:
        lines.append(f"TrackMate present, found by {install.origin}")
    else:
        lines.append("TrackMate was not found in this install, so it cannot track. Choose "
                     "another, or update this one with Fiji's updater.")
    if install.other_candidates:
        lines.append("also found: " + ", ".join(str(r) for r in install.other_candidates))
    return lines


#: What "Movies at once" does, per tracker: the Track page's and the Batch page's tooltip.
AT_ONCE_TIPS = {
    "utrack": ("How many movies u-track tracks side by side, each in its own MATLAB session (about 2 "
               "threads). It does not change the tracks. If the data is on a hard disk, put the "
               "u-track work folder on an SSD (Options)."),
    "trackmate": ("How many movies TrackMate tracks side by side, each in its own Fiji. "
                  "The tracks do not depend on it."),
}


def at_once_tip(tracker: str, standard: int) -> str:
    """The "Movies at once" tooltip for ``tracker``, with this computer's standard."""
    return f"{AT_ONCE_TIPS[tracker]} This computer's standard: {standard}."


def track_rows(states: dict, *, running: str | None = None, progress=None,
               queued=(), overwrite: bool = False, finished: dict | None = None,
               at_once: int = 1) -> list:
    """One row per condition for the Track page: ``{name, movies, status, ready,
    progress}``. ``ready`` means the condition can START now; the status says why not,
    or what starting would cost and overwrite. ``progress`` is ``(done, total)`` for the
    condition running. ``finished`` maps a condition tracked in this session to its
    `run_entry`, so its row says what the run produced rather than only that an earlier
    run is in the way."""
    from .prep_model import PrepError, minutes_at_once, run_inputs

    rows = []
    for name, state in states.items():
        row = {"name": name, "movies": f"{len(state.movies)} movie(s)", "ready": False,
               "progress": None}
        if name == running:
            done, total = progress or (0, len(state.movies))
            row.update(status=f"tracking {done}/{total}", progress=(done, total))
        elif name in queued:
            row.update(status="queued")
        elif finished and name in finished:
            status = finished_status(finished[name])
            if state.earlier_run and not overwrite:
                status += "; tick Replace to track again"
            row.update(status=status, ready=overwrite or not state.earlier_run)
        else:
            try:
                inputs = run_inputs(state)
            except PrepError as exc:
                row.update(status=str(exc).split(": ", 1)[-1])
            else:
                status = (f"ready, about "
                          f"{minutes_at_once(inputs.minutes, len(state.movies), at_once):.0f} min")
                if state.earlier_run:
                    status += (f"; replaces an earlier run ({state.earlier_run} files)"
                               if overwrite else
                               f"; an earlier run is in trackmate/ ({state.earlier_run} "
                               f"files); tick Replace to track again")
                row.update(status=status, ready=overwrite or not state.earlier_run)
        rows.append(row)
    return rows


def run_entry(name: str, result, error: str) -> dict:
    """One finished condition, as the window keeps it until the run's banner is shown:
    ``{name, error, outcome}`` -- the error text when the run itself failed."""
    outcome = dict((result or {}).get("outcome") or {})
    return {"name": name, "error": error or ("" if result is not None else "no result"),
            "outcome": outcome}


def finished_status(entry: dict) -> str:
    """A tracked condition's row status: what the run produced, in one line."""
    if entry["error"]:
        return f"FAILED: {entry['error']}"
    out = entry["outcome"]
    parts = []
    if out.get("usable_tracks") is not None:
        parts.append(f"{out['usable_tracks']:,} usable tracks")
    if out.get("movies") is not None:
        done = out["movies"] - out.get("failed", 0) - out.get("stopped", 0)
        parts.append(f"from {done} of {out['movies']} movie(s)")
    head = "done" if not (out.get("failed") or out.get("stopped")) else "done, incomplete"
    tail = []
    if out.get("failed"):
        tail.append(f"{out['failed']} failed")
    if out.get("stopped"):
        tail.append(f"{out['stopped']} not run (stopped)")
    text = head + (": " + " ".join(parts) if parts else "")
    return text + (f"; {', '.join(tail)}" if tail else "")


def finished_banner(entries: list, *, tracker: str = "trackmate") -> dict:
    """What the Track page says when a run -- every queued condition -- has ended:
    ``{title, lines, tone}``, tone ``ok`` or ``warn``. A failed or stopped condition is
    named first, never folded into a total."""
    failed = [e for e in entries if e["error"]]
    stopped = [e for e in entries if not e["error"] and e["outcome"].get("stopped")]
    partial = [e for e in entries if not e["error"] and e["outcome"].get("failed")]
    if stopped:
        title = "Tracking stopped"
    elif failed or partial:
        title = "Tracking finished, with failures"
    else:
        title = "Tracking finished"
    ok = [e for e in entries if not e["error"]]
    movies = sum(e["outcome"].get("movies", 0) - e["outcome"].get("failed", 0)
                 - e["outcome"].get("stopped", 0) for e in ok)
    usable = [e["outcome"].get("usable_tracks") for e in ok]
    head = f"{len(entries)} condition(s) run"
    if any("movies" in e["outcome"] for e in ok):
        head += f", {movies} movie(s) tracked"
    if usable and all(u is not None for u in usable):
        head += f", {sum(usable):,} usable tracks"
    lines = [head + "."]
    lines += [f"{e['name']}: {finished_status(e)}" for e in failed + stopped + partial]
    lines += [f"{e['name']}: {finished_status(e)}" for e in entries
              if e not in failed + stopped + partial]
    where = ("tif/<movie>/TrackingPackage/ folders"
             if tracker == "utrack" else "trackmate/ folder")
    lines.append(f"The tracks are in each condition's {where}, ready for the analysis. "
                 f"Done closes this window and returns to the menu.")
    return {"title": title, "lines": lines,
            "tone": "warn" if failed or stopped or partial else "ok"}


def progress_line(index: int, total: int, row: dict) -> str:
    """One movie's line in the run log, from `track_condition`'s plain progress row."""
    from .trackmate_run import STOPPED_ERROR

    head = f"[{index}/{total}] {row['movie']}"
    if row.get("ok"):
        return (f"{head}: {row['n_spots']} spots, {row['n_tracks']} tracks, threshold "
                f"{row['threshold']:.0f}, {row['seconds']:.0f} s")
    if row.get("error") == STOPPED_ERROR:
        return f"{head}: not run (stopped)"
    return f"{head}: FAILED: {row.get('error')}"


def run_header(inputs) -> list:
    """What a run is about to do, logged BEFORE it starts: the cost, the settings, where
    the tracks go, and whether each movie gets its own threshold."""
    lines = [f"{inputs.condition}: tracking {len(inputs.movies)} movie(s), about "
             f"{inputs.minutes:.0f} min -> {inputs.output_dir}",
             f"    {inputs.settings.stamp()}"]
    if inputs.per_movie:
        values = sorted(inputs.per_movie.values())
        lines.append(f"    each movie at its own threshold: {values[0]:.0f} to "
                     f"{values[-1]:.0f} over {len(values)} movie(s)")
    else:
        lines.append(f"    one threshold for every movie: {inputs.settings.threshold:g}")
    return lines


# ---------------------------------------------------------------- stacks for u-track
def export_rows(states: dict, *, overwrite: bool = False, running: str | None = None,
                progress=None, queued=()) -> list:
    """One line per condition for the export card: ``{name, status, ready}``."""
    from .prep_model import PrepError, export_inputs

    rows = []
    for name, state in states.items():
        row = {"name": name, "ready": False}
        if name == running:
            done, total = progress or (0, len(state.movies))
            row["status"] = f"writing {done}/{total}"
        elif name in queued:
            row["status"] = "queued"
        else:
            try:
                got = export_inputs(state)
            except PrepError as exc:
                row["status"] = str(exc).split(": ", 1)[-1]
            else:
                n = len(got["movies"])
                if state.stacks and not overwrite:
                    row["status"] = (f"tif/ already holds {state.stacks} stack(s); tick "
                                     f"Replace to write them again")
                else:
                    row["status"] = (f"ready: {n} stack(s) to tif/"
                                     + (f", replacing {state.stacks}" if state.stacks else ""))
                    row["ready"] = True
        rows.append(row)
    return rows


def export_line(index: int, total: int, row: dict) -> str:
    """One movie's line while stacks are written."""
    head = f"[{index}/{total}] {row['movie']}"
    if row.get("ok"):
        return (f"{head}: frames {row['crop_start']}-{row['crop_end']}, "
                f"{row['n_frames_kept']} kept")
    return f"{head}: FAILED: {row.get('error')}"


# ---------------------------------------------------------------- the linking radius
def radius_check_lines(got: dict) -> list:
    """What the wider-radius check found, and the one way it must not be read."""
    lo, hi = got["implied_D_interval"]
    return [
        f"{got['condition']}: {100 * got['beyond_narrow']:.1f} % of steps lie beyond "
        f"{got['narrow_radius_px']:g} px ({got['n_steps_narrow']} steps at "
        f"{got['narrow_radius_px']:g} px, {got['n_steps_wide']} at "
        f"{got['wide_radius_px']:g}) over {', '.join(got['checked_movies'])}.",
        f"That implies D of at least {got['implied_D_floor']:.2f} [{lo:.2f}, {hi:.2f}] "
        f"µm²/s, against {got['max_measurable_D']:.2f} this radius measures within 10 % "
        f"and a {got['apparent_D_ceiling']:.2f} ceiling: {got['band']}. "
        f"{got['verdict']}.",
        "This does not call for a wider radius: fractions were validated at 5 px, and wider radii "
        "can inflate them.",
    ]


def pilot_lines(got: dict, radius_now: float) -> list:
    """Measure: the step lengths of tracks already here (the floor) and the share of
    detections with a neighbour within the radius in use (the ceiling)."""
    steps, crowd = got["steps"], got["crowd"]
    lines = [f"From {got['source']}: {steps['n_steps']} consecutive-frame steps, median "
             f"{steps['median']:.2f}, p95 {steps['p95']:.2f}, p99 {steps['p99']:.2f}, max "
             f"{steps['max']:.2f} px."]
    if got.get("radius_used_px"):
        lines.append(f"Those tracks were linked at {got['radius_used_px']:g} px, so no step "
                     f"above it can appear; {100 * steps['censored_fraction']:.1f} % sit in "
                     f"its top tenth. Re-measure from a generous run if that is large.")
    lines.append(f"At {crowd['radius_px']:g} px, {100 * crowd['fraction_with_neighbour']:.1f} % "
                 f"of detections have another within one radius; links for those are guesses "
                 f"(this sets the ceiling).")
    lines.append(f"A radius of p99 = {steps['p99']:.2f} px would be a reasonable setting; this "
                 f"condition holds {float(radius_now):.2f}.")
    return lines


def pixel_size_line(camera_um, magnification, tube, value) -> str:
    """The arithmetic, shown before anything is applied -- and the .nd2's own trap named
    when the tube lens is left out."""
    if value is None:
        return "Set the camera pixel and the objective to compute it."
    text = (f"{float(camera_um):g} µm / ({float(magnification):g} x {float(tube):g}) = "
            f"{value:.1f} nm per pixel")
    if abs(float(tube) - 1.0) < 1e-9:
        text += (". With no tube lens this is what some .nd2 files claim (160 nm for 16 µm at "
                 "100x). A tube lens changes it (1.5x gives 106.7 nm), and every D scales with "
                 "the square of the pixel size.")
    return text


# ---------------------------------------------------------------- u-track
#: The two trackers the Track page offers: key, words.
TRACKERS = (("trackmate", "TrackMate (Fiji)"), ("utrack", "u-track (MATLAB)"))

#: The u-track settings card, top to bottom: ``(key, label, unit, group, kind, advanced,
#: help)``. ``kind`` is "number", "switch" (off/on) or "choice" (`UTRACK_CHOICES`). The
#: help is the one line under the setting; every claim in it is u-track's own source or
#: the 2063 lab runs on the lab's data drive (`utrack_run.LAB_STANDARD`).
UTRACK_FIELDS = (
    ("alpha_loc_max", "Spot significance (alphaLocMax)", "", "Detection", "number", False,
     "How unlikely a local maximum must be against its local background (31 x 31 px) to "
     "count as a spot. Smaller is stricter. Default 0.025; u-track's own default is 0.05."),
    ("psf_sigma_px", "PSF sigma", "px", "Detection", "number", False,
     "A spot's Gaussian width. Empty: u-track derives it from the NA, emission wavelength "
     "and pixel size (for example 0.93 px for mNeonGreen at NA 1.49 and 106 nm)."),
    ("background_alpha", "Cell-free background test", "alpha", "Detection", "number", False,
     "Empty: off. When set, each frame with the cells blanked out gives one mean and SD, "
     "and a spot must also clear that at this alpha (u-track suggests 0.001). It is a "
     "threshold; nothing is subtracted. Needs outlines."),
    ("integ_window", "Frames averaged", "each side", "Detection", "number", True,
     "Frames averaged on each side before spots are looked for (window 2w + 1). Use 0 "
     "for moving molecules."),
    ("do_mmf", "Mixture-model fitting", "", "Detection", "switch", True,
     "Split one blob into several spots when an extra Gaussian improves the fit (F-test). "
     "Off by default; slower."),
    ("alpha_r", "Test: one more spot", "alpha", "Detection", "number", True,
     "F-test for adding a Gaussian; used only with mixture-model fitting."),
    ("alpha_a", "Test: amplitude above zero", "alpha", "Detection", "number", True,
     "A fitted spot whose amplitude is not significantly above zero is removed."),
    ("alpha_d", "Test: distance between spots", "alpha", "Detection", "number", True,
     "Two fitted spots closer than this test allows are merged."),
    ("alpha_f", "Test: residuals against noise", "alpha", "Detection", "number", True,
     "Final test of the fit's residuals against the background noise. 0 (the default) "
     "turns it off."),
    ("num_sigma_iter", "PSF re-estimation", "iterations", "Detection", "number", True,
     "Re-estimate the PSF width from isolated spots, up to this many times. 0 (the "
     "default) keeps the given width."),
    ("bit_depth", "Camera bit depth", "bits", "Detection", "number", True,
     "Images are divided by 2^bits - 1, so amplitudes come out normalised. Use 16 for a 16-bit camera."),
    ("max_search_radius_px", "Search radius, largest", "px", "Linking", "number", False,
     "u-track's radius is adaptive: the multiplier below times each track's own step SD, "
     "clipped to this range. The default is 5 px; it also sets the fastest D that can "
     "be linked."),
    ("min_search_radius_px", "Search radius, smallest", "px", "Linking", "number", False,
     "The floor of the adaptive radius. The default is 0."),
    ("brown_std_mult", "Radius multiplier", "x step SD", "Linking", "number", False,
     "How many of a track's own step SDs a link may reach. 3 = about 2.1 RMS steps."),
    ("use_local_density", "Widen where isolated", "", "Linking", "switch", True,
     "Lets an isolated molecule use a wider radius, up to half its nearest-neighbour "
     "distance. On by default."),
    ("nn_window", "Neighbour window", "frames", "Linking", "number", True,
     "Frames looked back for nearest-neighbour distances."),
    ("frames_bridged", "Frames bridged", "frames", "Gaps and length", "number", False,
     "Frames a molecule may vanish for and still be joined (u-track's maximum gap). "
     "0 = no gap closing, the default (timeWindow 1)."),
    ("min_track_len", "Shortest segment kept", "localisations", "Gaps and length",
     "number", False,
     "Segments shorter than this are dropped before gap closing and never reach the "
     "tracks. The default is 5, the analysis's own minimum."),
    ("merge_split", "Merges and splits", "", "Gaps and length", "choice", False,
     "Whether tracks may merge or split. The default is None; the analysis turns each "
     "piece of a merged or split track into its own track."),
    ("gap_penalty", "Gap penalty", "", "Gaps and length", "number", True,
     "Cost multiplier per extra frame of a gap (cost x penalty^(gap - 2))."),
    ("brown_scaling", "Radius growth over a gap", "two powers", "Gaps and length", "number",
     True, "How the search radius grows with gap length, before and after the confidence "
     "time. Typed as two numbers: 0.5, 0.01."),
    ("amp_ratio_limit", "Merge/split brightness ratio", "min, max", "Gaps and length",
     "number", True, "Bounds on the brightness ratio a merge or split may show."),
    ("res_limit_px", "Resolution limit", "px", "Gaps and length", "number", True,
     "Widens the merge/split search radius. Empty: none."),
    ("gap_exclude_ms", "Gaps before merges", "", "Gaps and length", "switch", True,
     "Gap closing takes priority over merging and splitting."),
    ("strategy_bd", "Birth/death cost", "", "Gaps and length", "number", True,
     "How the cost of starting or ending a track is set: 0 automatic, -1 derived, above 0 "
     "a percentile."),
)
UTRACK_GROUPS = ("Detection", "Linking", "Gaps and length")
#: The words of each "choice" setting, by value.
UTRACK_CHOICES = {"merge_split": ("None", "Both", "Merge only", "Split only")}


def _num(value) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ", ".join(f"{v:g}" for v in value)
    return f"{value:g}"


def utrack_cells(settings) -> dict:
    """``{key: text}`` for the card: a number as typed, a switch ``"1"``/``"0"``, a choice
    its index; an optional setting that is off is empty (its placeholder says what)."""
    out = {}
    for key, _label, _unit, _group, kind, _adv, _help in UTRACK_FIELDS:
        value = getattr(settings, key)
        if kind == "switch":
            out[key] = "1" if value else "0"
        elif kind == "choice":
            out[key] = str(int(value))
        else:
            out[key] = _num(value)
    return out


def utrack_differences(settings) -> list:
    """Every setting that differs from the standard settings, as ``label: value (standard: value)``,
    so a run that departs from them says so before it starts."""
    from .utrack_run import LAB_STANDARD

    out = []
    for key, label, _unit, _group, kind, _adv, _help in UTRACK_FIELDS:
        now, lab = getattr(settings, key), getattr(LAB_STANDARD, key)
        if now == lab:
            continue
        def word(v, kind=kind, key=key):
            if kind == "switch":
                return "on" if v else "off"
            if kind == "choice":
                return UTRACK_CHOICES[key][int(v)]
            return "off" if v is None else _num(v)
        out.append(f"{label}: {word(now)} (standard: {word(lab)})")
    return out


#: What the window says while MATLAB starts for its check (14 Oct 2026: "only load matlab
#: ... before the tracking starts, and tell the user matlab is starting ... dont close").
UTRACK_STARTING = ("Starting MATLAB to check u-track (about 30 s). Please don't close the "
                   "window; the run starts when MATLAB answers")


def utrack_engine_lines(matlab, utrack, *, health=None, error: str = "",
                        absence: str = "", checking: bool = False) -> list:
    """What the MATLAB card says: the MATLAB and u-track in use, what the health check
    found, or where it looked and the way out. MATLAB is STARTED only when a u-track run or
    batch begins (``checking`` while it starts), never on opening the window."""
    lines = []
    if matlab is not None:
        lines.append(f"MATLAB {matlab.release} at {matlab.executable}")
    if utrack is not None:
        lines.append(f"u-track at {utrack.software} (found by {utrack.origin})")
        if utrack.other_candidates:
            lines.append("also found: " + ", ".join(utrack.other_candidates))
    if matlab is None or utrack is None:
        lines += [ln for ln in absence.splitlines() if ln.strip()]
        return lines
    if error:
        lines.append(f"MATLAB did not answer: {error}")
    elif health is None and checking:
        lines.append(UTRACK_STARTING)
    elif health is None:
        lines.append("found; MATLAB is started and checked when tracking begins")
    else:
        missing = [name for key, name in (("statistics", "Statistics"),
                                          ("optimization", "Optimization"),
                                          ("image", "Image Processing")) if not health[key]]
        if missing:
            lines.append(f"MATLAB answered, but the {', '.join(missing)} toolbox(es) are "
                         f"missing. u-track needs them.")
        else:
            lines.append(f"ready: MATLAB {health['version']} answered with u-track and the "
                         f"Statistics, Optimization and Image Processing toolboxes.")
    return lines


def utrack_condition_rows(states: dict, settings) -> list:
    """One line per condition for the u-track card: ``{name, psf, background, status}``.
    The PSF is the one u-track will derive (or the one typed); the status is the reason
    the condition cannot be tracked yet, or "ready"."""
    from .prep_model import PrepError, utrack_inputs, utrack_psf_preview

    rows = []
    for name, state in states.items():
        if settings.psf_sigma_px is not None:
            psf = f"{settings.psf_sigma_px:g} px (typed)"
        else:
            preview = utrack_psf_preview(state)
            if preview is None:
                cal = state.calibration
                missing = [word for key, word in (("numerical_aperture", "NA"),
                                                  ("emission_nm", "emission"),
                                                  ("pixel_size_nm", "pixel size"))
                           if not cal.get(key)]
                psf = "needs " + " and ".join(missing or ["the optics"])
            else:
                psf = f"about {preview:.2f} px (from the optics)"
        outlines = state.outlines
        cover = ("no outlines" if outlines is None or not outlines.has_folder else
                 f"outlines {outlines.n_with_outlines}/{outlines.n_movies}")
        try:
            utrack_inputs(state, settings)
            status = "ready"
        except PrepError as exc:
            status = str(exc).split(": ", 1)[-1]
        rows.append({"name": name, "psf": psf, "background": cover, "status": status})
    return rows


def utrack_track_rows(states: dict, settings, *, running: str | None = None, progress=None,
                      queued=(), overwrite: bool = False, finished: dict | None = None,
                      at_once: int = 1) -> list:
    """`track_rows` for a u-track run: the same row shape and the same ``ready`` rule,
    with u-track's refusals and its earlier runs (``tif/<movie>/movieData.mat``)."""
    from .prep_model import PrepError, minutes_at_once, utrack_earlier, utrack_inputs

    rows = []
    for name, state in states.items():
        row = {"name": name, "movies": f"{len(state.movies)} movie(s)", "ready": False,
               "progress": None}
        earlier = utrack_earlier(state) if state.movies else 0
        if name == running:
            done, total = progress or (0, len(state.movies))
            row.update(status=f"tracking {done}/{total}", progress=(done, total))
        elif name in queued:
            row.update(status="queued")
        elif finished and name in finished:
            status = finished_status(finished[name])
            if earlier and not overwrite:
                status += "; tick Replace to track again"
            row.update(status=status, ready=overwrite or not earlier)
        else:
            try:
                inputs = utrack_inputs(state, settings)
            except PrepError as exc:
                row.update(status=str(exc).split(": ", 1)[-1])
            else:
                status = (f"ready, about "
                          f"{minutes_at_once(inputs.minutes, len(state.movies), at_once):.0f} min")
                if earlier:
                    status += (f"; replaces the u-track run of {earlier} movie(s)"
                               if overwrite else
                               f"; {earlier} movie(s) already hold a u-track run; tick "
                               f"Replace to track again")
                row.update(status=status, ready=overwrite or not earlier)
        rows.append(row)
    return rows


def utrack_progress_line(index: int, total: int, row: dict) -> str:
    """One movie's line in the run log, from `track_condition_utrack`'s progress row."""
    from .utrack_run import STOPPED_ERROR

    head = f"[{index}/{total}] {row['movie']}"
    if row.get("ok"):
        return (f"{head}: {row['n_tracks']} tracks from {row['n_frames']} frames, "
                f"{row['seconds']:.0f} s")
    if row.get("error") == STOPPED_ERROR:
        return f"{head}: not run (stopped)"
    return f"{head}: FAILED: {row.get('error')}"


def utrack_run_header(inputs) -> list:
    """What a u-track run is about to do, logged BEFORE it starts."""
    lines = [f"{inputs.condition}: u-track on {len(inputs.movies)} movie(s), "
             f"{inputs.frames:,} frames, about {inputs.minutes:.0f} min -> "
             f"{inputs.folder / 'tif'}",
             f"    {inputs.settings.stamp()}"]
    differs = utrack_differences(inputs.settings)
    if differs:
        lines.append("    differs from the standard settings: " + "; ".join(differs))
    if inputs.earlier:
        lines.append(f"    replaces the u-track run of {inputs.earlier} movie(s)")
    return lines
