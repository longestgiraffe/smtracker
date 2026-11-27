"""What the onboarding Confirm screens print, decided from a loaded project, with no Qt anywhere in it.

`ui/onboarding/adapt.py` and `scan.py` are the Qt-layer wiring (the model the screens read, the worker threads); this is
what they read from the canonical table. They are split because of the Thin Qt rule (docs/gui_tool_views.md;
`tests/test_ui_thin_qt.py` walks the whole GUI layer): a wrong number on the screen a lab member checks before trusting
anything must fail in a headless test, and no table method belongs in `ui/`.

Everything returns plain lists, dicts, strings and numbers:

  * `series_rows` — one dict per movie (name, day, found, kept, cells, localizations);
  * `frames_by_day` — the highest frame index seen, plus one, per replicate day;
  * `imaging_days` / `group_blocks` — the imaging days (by acquisition time, never by import time, else by series
    naming), and the
    confirmed grouping the loaded ``day`` column holds; two different things once a grouping has been saved;
  * `odd_day_notes` — the "N frames, not M" note for a day shorter or longer than the rest, ONE rule for both the model
    the Confirm page builds and the day table (they used to carry a copy each);
  * `day_table` — the Confirm screen's table, one dict per replicate day;
  * `acquisition_cards` — the five acquisition cards, `(value, caption)` each; `condition_cards` the same for ONE
    condition from its own record (`condition_acquisition`), and `acquisition_differences` the Confirm page's line
    when the conditions were not recorded alike.

The rules kept here:
  * a movie's name loses its condition prefix by stripping that known prefix, never by splitting: a condition name may
    contain an underscore or a slash ("3 LB_series001");
  * a movie the tracker produced nothing for is absent from ``tracks_found``, so ``found`` falls back to ``kept`` (a ratio
    of 100 %, never a division by zero);
  * a parameter the tracker did not record is a dash with a caption saying so, never a plausible default: this is the
    screen a user checks before trusting a number;
  * ``frames`` is the highest frame index seen, not the acquisition length, and is labelled as what the files recorded.
"""
from __future__ import annotations

import re

__all__ = [
    "PENDING",
    "TRACKER_LABELS",
    "acquisition_cards",
    "acquisition_differences",
    "condition_acquisition",
    "condition_cards",
    "composition_words",
    "day_table",
    "found_by_movie",
    "frames_by_day",
    "group_blocks",
    "imaging_days",
    "leave_out_question",
    "holds_acquisitions",
    "load_failure",
    "movie_count",
    "odd_day_notes",
    "range_text",
    "series_rows",
    "short_movie_name",
    "needs_tracker_choice",
    "tracker_questions",
    "tracker_choice_ready",
    "tracker_note",
    "tracker_subtitle",
    "tab_rows",
]

#: What a cell shows when the data cannot answer. Kept identical to `confirm.PENDING`.
PENDING = "–"


def short_movie_name(qualified: str, condition: str) -> str:
    """`project.qualified_movie` joins with an UNDERSCORE, not a slash, and a condition name may contain either. Stripping
    the known prefix is the only split that cannot cut a name in half."""
    qualified = str(qualified)
    prefix = f"{condition}_"
    return qualified.removeprefix(prefix)


def found_by_movie(canonical) -> dict:
    """``{qualified movie: tracks the tracker found}``: the table's ``tracks_found`` record, which `cellmesh.assign_cells`
    writes where the filter runs (the only place the dropped tracks still exist to be counted). Empty where the table
    carries none."""
    return dict(canonical.attrs.get("tracks_found", {}))


def series_rows(table, condition: str, found_by_movie: dict) -> list[dict]:
    """One dict per movie of ``table``, in movie order."""
    rows = []
    for movie, part in table.groupby("movie", sort=True):
        day = part["day"].iloc[0] if "day" in part and len(part) else None
        kept = int(part["uid"].nunique())
        rows.append({
            "name": short_movie_name(movie, condition),
            "day": None if day is None else int(day),
            # A movie the tracker produced nothing for is absent from `tracks_found`.
            "found": int(found_by_movie.get(str(movie), kept)),
            "kept": kept,
            "cells": int(part["cell"].nunique()) if "cell" in part else 0,
            "localizations": len(part),
        })
    return rows


def frames_by_day(table, indices: list, day_of_series: dict | None = None, condition: str = "") -> dict:
    """``{day: highest frame index seen + 1}`` for each day in ``indices`` (0 where the day has no rows or the table
    no frame column). The day is the table's ``day`` column, or, when ``day_of_series`` is given, that mapping of the
    SHORT series name (the movie less ``condition``'s prefix) -- the IMAGING day, which differs from the column once a
    confirmed grouping has been saved (see `imaging_days`)."""
    if day_of_series is None:
        days = table["day"]
    else:
        days = table["movie"].map(lambda movie: day_of_series.get(short_movie_name(movie, condition)))
    out = {}
    for index in indices:
        part = table[days == index]
        out[index] = int(part["frame"].max()) + 1 if "frame" in part and len(part) else 0
    return out


def imaging_days(loaded_days: dict, stamps: dict | None = None, *, series_per_day: int = 10) -> dict:
    """``{series: imaging day}``: by ACQUISITION TIME first, by the series names where the times cannot say.

    THE LOADED ``day`` COLUMN IS NOT THE IMAGING DAY ONCE A GROUPING HAS BEEN CONFIRMED. "Open workspace" writes the
    group board's arrangement into ``day_map.json``, and the next load reads it back as the ``day`` column -- so a
    Confirm screen that took its days from that column showed the GROUPS as days, and "Reset to days" rebuilt exactly
    the groups already on screen. The imaging days are therefore derived here from the movies themselves, so a saved
    grouping can never overwrite them.

    BY DATE FIRST, whatever the number of series a day holds. ``stamps`` is ``{series: (timestamp, source)}``
    (`daymap.read_acquisition_stamps`: the raw ``.nd2``/``.stk`` first, then ``movieData.mat``); when EVERY dated series
    has a true ACQUISITION time, the movies are clustered into sessions by `daymap.propose_from_timestamps` (the spine's
    one rule: a gap of more than an hour starts a new day).

    AN IMPORT TIME DOES NOT DECIDE A DAY. ``createTime_`` is when u-track imported a movie, and one import session can
    hold movies imaged months apart. Measured on protein A/Malate: all 29 movies were imported on 9 March 2026, series041-050
    at 07:36 and series001-009 at 08:18, so the gap rule made ONE day of two sessions eight months apart (25.02.2026
    and 25.06.2025 by the ``.nd2`` files). So a condition with any import-time stamp goes to the series naming.
    The days are then NUMBERED by their lowest series number, not by date -- ruled: the day holding 001-010 is D1
    whenever it was imaged (`_numbered_by_series`). Two sessions on one calendar date stay two days, as that rule
    decides. Where any
    series has no readable time (a TrackMate project, an unreadable file), the series naming decides instead
    (`grouping.day_map_from_series_blocks`, blocks of ``series_per_day``); where the names carry no series number
    either, the loaded days are returned unchanged, never a guess. A series the loader gave no day keeps none.
    """
    from .daymap import ACQUISITION_SOURCES, MovieStamp, propose_from_timestamps
    from .grouping import GroupingError, day_map_from_series_blocks

    dated = [name for name, day in loaded_days.items() if day is not None]
    if not dated:
        return dict(loaded_days)
    derived = None
    times = stamps or {}
    if all(times.get(name, (None, None))[0] is not None and times[name][1] in ACQUISITION_SOURCES
           for name in dated):
        proposal = propose_from_timestamps([MovieStamp(name, *times[name]) for name in dated])
        if proposal.ok:
            derived = _numbered_by_series(proposal.movie_to_day)
    if derived is None:
        try:
            derived, _spans = day_map_from_series_blocks(dated, series_per_day=series_per_day)
        except GroupingError:
            return dict(loaded_days)
    return {name: (None if day is None else derived[name]) for name, day in loaded_days.items()}


def _numbered_by_series(movie_to_day: dict) -> dict:
    """Renumber days so D1 is the day holding the lowest series number (ruled), not the oldest session.

    Which movies share a day is unchanged; only the labels move. Where a name carries no series number there is
    nothing to order by, and the proposal's time order is kept."""
    from .grouping import GroupingError, series_number

    try:
        lowest: dict = {}
        for movie, day in movie_to_day.items():
            n = series_number(movie)
            lowest[day] = min(lowest.get(day, n), n)
    except GroupingError:
        return dict(movie_to_day)
    order = {day: i + 1 for i, day in enumerate(sorted(lowest, key=lowest.get))}
    return {movie: order[day] for movie, day in movie_to_day.items()}


def group_blocks(loaded_days: dict) -> list:
    """The confirmed grouping as the board's blocks: ``[[series, ...] per loaded day, in day order]``.

    The loaded ``day`` column IS the grouping the workspace fits over (the confirmed one, once saved), so the board
    opens on it rather than on a factory reset. A series with no loaded day is in no block."""
    days = sorted({day for day in loaded_days.values() if day is not None})
    return [sorted(name for name, day in loaded_days.items() if day == d) for d in days]


def odd_day_notes(frames: list) -> list:
    """A note per day: "N frames, not M" for a day shorter or longer than the usual one, else None. Derived, not typed:
    nothing is said unless the files say it (at least two different lengths)."""
    lengths = [f for f in frames if f]
    notes: list = [None] * len(frames)
    if len(set(lengths)) < 2:
        return notes
    usual = max(set(lengths), key=lengths.count)
    for i, f in enumerate(frames):
        if f and f != usual:
            notes[i] = f"{f:,} frames, not {usual:,}"
    return notes


def composition_words(n_days: int) -> str:
    """A group lane's subtitle: how many IMAGING DAYS (sessions) its series come from. A group is one replicate; the
    days are when its movies were taken, so this never says "replicates" (tester B9, 3 Nov 2026: "2 replicates
    merged" on one replicate read as two independent cultures pooled)."""
    if n_days <= 0:
        return "No dated series"
    return "From one imaging day" if n_days == 1 else f"From {n_days} imaging days"


def range_text(movies: list) -> str:
    """``001 – 004`` from the movie names, or the one name when there is one."""
    numbers = [m for m in (re.search(r"(\d+)\s*$", n) for n in movies) if m]
    if len(numbers) >= 2:
        return f"{numbers[0].group(1)} – {numbers[-1].group(1)}"
    if len(movies) == 1:
        return movies[0]
    return f"{len(movies)} series"


def day_table(table) -> list[dict]:
    """One dict per replicate day, for the Confirm screen's table.

    Keys: ``day``, ``series`` (the range, as the folders are numbered), ``date``, ``frames``, ``cells``, ``kept``,
    ``found``, ``note``. TWO ARE ALWAYS ``None``, and that is the data rather than an omission: ``date`` (the canonical
    table carries no acquisition date, and a file's modification time is not one) and ``found`` (the table is
    POST-filter, so a track the outline or length filter dropped has no row here at all)."""
    if not len(table):
        return []
    rows = []
    for day, part in table.groupby("day", sort=True):
        movies = sorted({str(m).split("/", 1)[-1] for m in part["movie"]})
        rows.append({
            "day": int(day),
            "series": range_text(movies),
            "date": None,
            "frames": int(part["frame"].max()) + 1 if "frame" in part else None,
            "cells": int(part["cell"].nunique()) if "cell" in part else None,
            "kept": int(part["uid"].nunique()),
            "found": None,
            "note": None,
        })
    for row, note in zip(rows, odd_day_notes([r["frames"] for r in rows]), strict=True):
        row["note"] = note
    return rows


def movie_count(table) -> int:
    return int(table["movie"].nunique()) if len(table) else 0


def acquisition_cards(params, movies: int) -> list[tuple[str, str]]:
    """The five acquisition cards as ``(value, caption)``: pixel size, frame interval, PSF width, linking ceiling and
    localisation precision. Every one comes off `AcquisitionParams`, which is READ from the tracker's own files; one
    the files did not record is a dash with a caption saying so."""
    def num(value, fmt="{:g}"):
        return PENDING if value is None else fmt.format(value)

    sigma = getattr(params, "localization_precision_nm", None)
    source = getattr(params, "localization_sigma_source", "")
    if sigma is None and hasattr(params, "localization_sigma_nm"):
        sigma = params.localization_sigma_nm            # the theoretical fallback
    psf = getattr(params, "psf_sigma_px", None)
    radius = getattr(params, "search_radius_px", None)
    return [
        (num(getattr(params, "pixel_size_nm", None), "{:.1f}"),
         f"Identical in all {movies} series" if movies else ""),
        (num(getattr(params, "frame_interval_ms", None)),
         "the time between frames, used for every D"),
        (num(psf, "{:.2f}"),
         "u-track's fitted PSF width" if psf is not None
         else "not recorded by the tracker"),
        (num(radius),
         "Caps the largest D that can be measured" if radius is not None
         else "not recorded by the tracker"),
        (num(sigma, "{:.0f}"),
         f"source: {source}".replace("_", " ") if source else ""),
    ]


# ------------------------------------------------ each condition's OWN acquisition
# Found 3 Nov 2026 (tester B1): every condition's cards showed the PROJECT's `params`, which the loader merges from the
# first condition, so a condition linked at 8 px read "Search radius 5 px", one recorded at 30 ms read 25 ms and one at
# 106.5 nm read "106.0, identical in all 30 series". DISPLAY ONLY: what the analysis computes is unchanged.

def _rates_text(rates) -> str:
    """``[ms, ...]`` as one value or a range: ``"25"``, ``"25-25.9"``."""
    lo, hi = min(rates), max(rates)
    return f"{lo:.4g}" if _same(lo, hi) else f"{lo:.4g}-{hi:.4g}"


def _same(a, b, rel: float = 1e-6) -> bool:
    """One setting (`canonical.same_setting`'s tolerance): 106.000003 is 106."""
    return abs(float(a) - float(b)) <= rel * max(abs(float(a)), abs(float(b)), 1e-12)


def _distinct(values) -> list:
    out: list = []
    for v in sorted(float(x) for x in values if x is not None):
        if not out or not _same(out[-1], v):
            out.append(v)
    return out


def _grouped(said: dict) -> str:
    """``{condition: "25 ms"}`` as one phrase: each condition named where there are few ("A 25 ms, B 30 ms"), else
    each value with how many conditions have it, a value only one or two have naming them ("25 ms for 7
    conditions; 20 ms for one, measured")."""
    if len(said) <= 4:
        return ", ".join(f"{name} {value}" for name, value in said.items())
    by_value: dict = {}
    for name, value in said.items():
        by_value.setdefault(value, []).append(name)
    parts = []
    for value, names in sorted(by_value.items(), key=lambda kv: -len(kv[1])):
        parts.append(f"{value} for {' and '.join(names)}" if len(names) <= 2
                     else f"{value} for {len(names)} conditions")
    return "; ".join(parts)


def condition_acquisition(loaded, condition: str) -> dict:
    """``condition``'s acquisition AS RECORDED for its own movies, never the project's merged copy:
    ``{"pixel_size_nm", "frame_intervals_ms" (sorted distinct, per movie), "rescaled_to_ms" (the one interval the
    movies are read at, or None), "search_radius_px", "psf_sigma_px"}``, None where nothing was recorded.

    Read from the loader's per-condition record (``canonical.attrs["acquisition_by_condition"]``, cache v21), the
    pixel sizes it loaded within 1 % (``attrs["pixel_size_mixed"]``), and the table's ``tau_s`` per movie with the
    interval each rescaled movie was RECORDED at (``attrs["tau_rescaled"]``); the project's ``params`` only where
    none of these says."""
    canonical = getattr(loaded, "canonical", None)
    attrs = getattr(canonical, "attrs", None) or {}
    params = getattr(loaded, "params", None)
    record = (attrs.get("acquisition_by_condition") or {}).get(condition) or {}
    mixed = attrs.get("pixel_size_mixed") or {}

    def pick(key, fallback):
        if key not in record:                          # no record (a table loaded before v21): the project's
            return fallback
        value = record[key]                            # recorded, or recorded as missing (TrackMate's PSF)
        return None if value is None else float(value)

    pixel = pick("pixel_size_nm", mixed.get(condition, getattr(params, "pixel_size_nm", None)))
    rescaled = attrs.get("tau_rescaled") or {}
    rates, rescaled_to = [], None
    try:
        table = loaded.condition(condition)
    except Exception:                                  # noqa: BLE001 - a screen drawn without a table reads params
        table = None
    if table is not None and len(table) and "tau_s" in getattr(table, "columns", ()):
        per_movie = table.groupby("movie", sort=False)["tau_s"].first()
        for movie, tau in per_movie.items():
            moved = rescaled.get(str(movie))
            if moved:
                rates.append(float(moved["frame_interval_ms"]))
                rescaled_to = float(moved["rescaled_to_ms"])
            else:
                rates.append(1000.0 * float(tau))
    if not rates:
        stated = pick("frame_interval_ms", getattr(params, "frame_interval_ms", None))
        rates = [stated] if stated is not None else []
    return {"pixel_size_nm": None if pixel is None else float(pixel),
            "frame_intervals_ms": _distinct(rates),
            "rescaled_to_ms": rescaled_to,
            "search_radius_px": pick("search_radius_px", _own_radius(params, condition)),
            "psf_sigma_px": pick("psf_sigma_px", getattr(params, "psf_sigma_px", None))}


def _own_radius(params, condition: str):
    """``condition``'s search radius off the project's ``params`` (its own where the conditions were linked at different
    radii, `AcquisitionParams.search_radius_px_for`), for a screen with no per-condition record."""
    if hasattr(params, "search_radius_px_for"):
        from .canonical import CanonicalError

        try:
            return params.search_radius_px_for(condition)
        except CanonicalError:
            return None
    return getattr(params, "search_radius_px", None)


def condition_cards(loaded, condition: str) -> list[tuple[str, str]]:
    """The five acquisition cards for ONE condition, as `acquisition_cards`, with that condition's own recorded pixel
    size, frame interval(s), PSF width and search radius, and a caption saying where it differs from the others."""
    params = getattr(loaded, "params", None)
    names = list(getattr(loaded, "conditions", None) or [condition])
    try:
        movies = movie_count(loaded.condition(condition))
    except Exception:                                  # noqa: BLE001 - a screen drawn before a load
        movies = 0
    cards = acquisition_cards(params, movies)
    own = condition_acquisition(loaded, condition)
    others = {n: condition_acquisition(loaded, n) for n in names if n != condition}

    def other_values(key, fmt):
        """The OTHER conditions' values that differ from this one's."""
        mine = own.get(key)
        seen = [v for v in _distinct(o[key] for o in others.values() if o.get(key) is not None)
                if mine is None or not _same(v, mine)]
        return ", ".join(fmt.format(v) for v in seen)

    pixel = own["pixel_size_nm"]
    if pixel is not None:
        caption = cards[0][1]
        used = getattr(params, "pixel_size_nm", None)
        if used is not None and not _same(pixel, used):
            off = 100.0 * ((float(used) / pixel) ** 2 - 1.0)
            caption = f"as recorded; read at {float(used):.1f} nm, so D {off:+.1f} %"
        elif any(o["pixel_size_nm"] is not None and not _same(o["pixel_size_nm"], pixel) for o in others.values()):
            caption = f"as recorded; others {other_values('pixel_size_nm', '{:.1f}')} nm"
        cards[0] = (f"{pixel:.1f}", caption)
    rates = own["frame_intervals_ms"]
    if rates:
        caption = cards[1][1]
        if own["rescaled_to_ms"] is not None:
            caption = f"as recorded; every D read at {own['rescaled_to_ms']:.4g} ms, one interval for all"
        elif len(rates) > 1:
            caption = "as recorded, per movie; each D at its own"
        else:
            theirs = [r for r in _distinct(r for o in others.values() for r in o["frame_intervals_ms"])
                      if not _same(r, rates[0])]
            onto = next((o["rescaled_to_ms"] for o in others.values() if o["rescaled_to_ms"] is not None), None)
            if theirs:
                said = _rates_text(theirs).replace("-", " to ")
                caption = (f"the time between frames, used for every D; others recorded at {said} ms are read at it"
                           if onto is not None and _same(onto, rates[0])
                           else f"as recorded; others {said} ms, each D at its own")
        cards[1] = (_rates_text(rates), caption)
    psf = own["psf_sigma_px"]
    cards[2] = ((f"{psf:.2f}", "u-track's fitted PSF width") if psf is not None
                else (PENDING, "not recorded by the tracker"))
    radius = own["search_radius_px"]
    if radius is None:
        cards[3] = (PENDING, "not recorded by the tracker")
    else:
        caption = "Caps the largest D that can be measured"
        if any(o["search_radius_px"] is not None and not _same(o["search_radius_px"], radius)
               for o in others.values()):
            caption = f"Caps the largest D that can be measured; others {other_values('search_radius_px', '{:g}')} px"
        cards[3] = (f"{radius:g}", caption)
    return cards


def acquisition_differences(loaded) -> str:
    """The Confirm page's line when the conditions were NOT recorded alike, or "" when they were (asked by the
    tester, 3 Nov 2026: up to 1 % in pixel size and 10 % in frame interval the loader opens a project with a warning
    only QC repeated): the pixel sizes (QC's own words, `project.pixel_mix_text`), the frame intervals and whether
    they were put on one, and the search radii, each with what it means for a comparison."""
    names = list(getattr(loaded, "conditions", None) or [])
    if not names:
        return ""
    attrs = getattr(getattr(loaded, "canonical", None), "attrs", None) or {}
    params = getattr(loaded, "params", None)
    facts = {n: condition_acquisition(loaded, n) for n in names}
    out = []
    mixed = attrs.get("pixel_size_mixed")
    if mixed and getattr(params, "pixel_size_nm", None):
        from .project import pixel_mix_text

        out.append("Not one pixel size: " + pixel_mix_text(mixed, params.pixel_size_nm) + ".")
    rates = _distinct(r for f in facts.values() for r in f["frame_intervals_ms"])
    if len(rates) > 1:
        spread = 100.0 * (rates[-1] / rates[0] - 1.0)
        listed = _grouped({n: f"{_rates_text(f['frame_intervals_ms'])} ms" for n, f in facts.items()
                           if f["frame_intervals_ms"]})
        to = next((f["rescaled_to_ms"] for f in facts.values() if f["rescaled_to_ms"] is not None), None)
        if to is not None:
            out.append(f"Not one frame interval: they differ by up to {spread:.1f} % ({listed}). Every movie is "
                       f"read at {to:.4g} ms, with its steps rescaled so D is comparable and the fractions do "
                       "not change. The detection ceiling and motion blur stay at each movie's own interval.")
        else:
            out.append(f"Not one frame interval: they differ by up to {spread:.1f} % ({listed}). Each D uses its "
                       "own interval, but the detection ceiling and motion blur differ, so a comparison across "
                       "them also compares acquisitions.")
    # THE LOAD'S OWN WORDS for a project linked at several radii (`project.radius_mix_text`, tester D10): the warning,
    # QC's caption and this line are one rule
    from .project import radius_mix_text

    radii = radius_mix_text({n: f["search_radius_px"] for n, f in facts.items()})
    if radii:
        out.append(radii + ".")
    return " ".join(out)


# ------------------------------------------------- a condition tracked by both tools
#: How each tracker is named on screen, in the order the choice offers them.
TRACKER_LABELS = {"utrack": "u-track", "trackmate": "TrackMate", "table": "Track table"}


def tracker_questions(options: dict) -> list[dict]:
    """One question per condition holding output from BOTH trackers.

    ``options`` is `project.tracker_options`: ``{condition: {"utrack": n, ...}}``.
    Each question is ``{"name", "options": [(key, words)]}``, the words naming the
    tracker and how many movies it tracked there. Conditions with one tracker ask
    nothing (asked for: "only if two or more options")."""
    out = []
    for name, found in options.items():
        if len(found) < 2:
            continue
        out.append({"name": name,
                    "options": [(key, f"{TRACKER_LABELS[key]} \u00b7 {found[key]} movie"
                                      f"{'s' if found[key] != 1 else ''}")
                                for key in TRACKER_LABELS if key in found]})
    return out


def _effective(options: dict, chosen: dict) -> dict:
    """``{condition: tracker or None}``: the choice made, else the only tracker there."""
    return {name: chosen.get(name) or (next(iter(found)) if len(found) == 1 else None)
            for name, found in options.items()}


def needs_tracker_choice(options: dict) -> bool:
    """Ask before loading when a condition was tracked twice, or when the conditions were
    tracked by different tools (the loader would otherwise refuse the mix unannounced)."""
    if tracker_questions(options):
        return True
    return len({v for v in _effective(options, {}).values() if v}) > 1


def tracker_choice_ready(options: dict, chosen: dict) -> bool:
    """Every condition tracked twice has a choice. There is NO default: the two trackers
    disagree, so picking one silently would move reported numbers."""
    return all(chosen.get(q["name"]) for q in tracker_questions(options))


#: The trackers as a sentence names their output ("tracks from both u-track and TrackMate").
_TRACKER_WORDS = {"utrack": "u-track", "trackmate": "TrackMate", "table": "track tables"}


def tracker_subtitle(options: dict) -> str:
    """The line under the page's title: what it asks, or, where no condition was tracked twice and the page is
    shown only because the conditions come from different tools (30 Oct 2026: two u-track conditions beside a
    track table of the truth), that there is nothing to choose. It said "pick which to read for each" over an
    empty list."""
    asked = tracker_questions(options)
    if asked:
        tools = [key for key in TRACKER_LABELS if any(key == k for q in asked for k, _w in q["options"])]
        whom = "This condition holds" if len(asked) == 1 else "These conditions hold"
        return (f"{whom} tracks from both {' and '.join(_TRACKER_WORDS[k] for k in tools)}. The two do not give "
                "identical tracks, so choose which to read for each.")
    used = [key for key in TRACKER_LABELS if any(key in found for found in options.values())]
    return (f"These conditions were tracked by different tools ({', '.join(_TRACKER_WORDS[k] for k in used)}). "
            "There is nothing to choose; each condition is read from its own tool.")


def tracker_note(options: dict, chosen: dict) -> str:
    """Said under the choice when the conditions end up read from different tools: which conditions come from which,
    and what that means FOR THE PAIR AT HAND (asked by the tester, 3 Nov 2026: the u-track vs TrackMate figures were
    quoted for a simulation's truth table beside u-track, where they do not apply). u-track and TrackMate were
    measured on the same movies; a track table is whatever made it (another tool, a simulation's truth), and how far
    it reads from either was never measured here."""
    effective = {k: v for k, v in _effective(options, chosen).items() if v}
    used = {v for v in effective.values()}
    if len(used) < 2:
        return ""
    groups = []
    for key, words in TRACKER_LABELS.items():
        names = [n for n, v in effective.items() if v == key]
        if not names:
            continue
        source = "track tables" if key == "table" else words
        groups.append(f"{', '.join(names)} from {source}" if len(names) <= 4
                      else f"{len(names)} conditions from {source}")
    kind = "sources" if "table" in used else "trackers"
    said = [f"These conditions will be read from different {kind}: " + "; ".join(groups) + "."]
    if {"utrack", "trackmate"} <= used:
        said.append("On the same movies u-track and TrackMate differ by about 1 pp in the slow fraction and 10 to "
                    "18 % in D, so comparing them also compares the trackers.")
    if "table" in used:
        others = " and ".join(_TRACKER_WORDS[k] for k in ("utrack", "trackmate") if k in used)
        said.append(f"A track table comes from outside SMTracker (another tool, or a simulation's ground truth), so "
                    f"comparing it with {others} tracks also compares how the tracks were made. How far they "
                    "differ has not been measured.")
    return " ".join(said)


#: What each missing u-track field is, in words (`utrack_meta.MissingAcquisitionError`).
_MISSING_WORDS = {
    "pixelSize_": "a pixel size",
    "timeInterval_": "a frame interval (not in u-track's record or in a camera file beside the movie)",
}


def leave_out_question(by_condition: dict) -> dict:
    """The loading page's question when movies were tracked without a value every D scales with (asked 20 Oct
    2026: "if they print no frame interval set -> ask if movie is ignored"). ``by_condition`` is
    ``MissingAcquisitionError.by_condition``, ``{condition: [(movie, field, path)]}``. Returns ``{"title", "body",
    "leave_out"}``; ``leave_out`` is what `load_project(leave_out=...)` takes, each movie with its reason, which the
    table keeps (``attrs["left_out_movies"]``)."""
    rows = [(c, m, f) for c, found in by_condition.items() for m, f, *_rest in found]
    n = len(rows)
    lines = [f"{m} in {c} was tracked without {_MISSING_WORDS.get(f, f.rstrip('_'))}." for c, m, f in rows]
    title = (f"{rows[0][1]} cannot be analysed" if n == 1 else f"{n} movies cannot be analysed")
    # THE FIX FOR THE VALUE MISSING (tester B11, 3 Nov 2026): every question offered "state the pixel size in an
    # acquisition.json", which cannot state a frame interval (`utrack_meta._STATABLE` holds the pixel size only)
    fields = {f for _c, _m, f in rows}
    them = "it" if n == 1 else "them"
    fixes = []
    if "pixelSize_" in fields:
        fixes.append("for a missing pixel size, set it in u-track or give it in an acquisition.json beside the "
                     "movies (see the manual)")
    if "timeInterval_" in fields:
        fixes.append("for a missing frame interval, set it in u-track, or put the movie's camera file (.nd2 or "
                     ".stk) in raw/ beside it")
    if fields - {"pixelSize_", "timeInterval_"}:
        fixes.append("set the missing value in u-track")
    body = (" ".join(lines) + " Every diffusion coefficient depends on it, so SMTracker does not guess it. "
            + ("Leave it out and open the rest?" if n == 1 else "Leave them out and open the rest?")
            + f" To keep {them}: " + "; ".join(fixes) + "; then open the folder again.")
    leave_out: dict = {}
    for c, m, f in rows:
        leave_out.setdefault(c, {})[m] = (f"left out when opened: tracked without "
                                          f"{_MISSING_WORDS.get(f, f.rstrip('_')).split(' (')[0]}")
    return {"title": title, "body": body, "leave_out": leave_out}


# ------------------------------------------------------ a folder the door cannot open
#: The first part of a message the loader or discovery builds as ``"<ExceptionType>: <reason>"``.
_EXCEPTION_PREFIX = re.compile(r"^\s*[A-Z][A-Za-z]*(?:Error|Exception)\s*:\s*")

#: ``(pattern in the message, title, advice)``, the first that matches wins: what went wrong
#: in plain words, and what to DO about it. Found 3 Nov 2026 (tester B2): every refusal read
#: "Nothing was loaded. Choose a different folder.", wrong for a table without units, two
#: pixel sizes, or movies not yet tracked, none of which another folder fixes.
_FAILURES = (
    ("does not say whether its positions are pixels",
     "The track tables do not say their units",
     "State the units of the positions (pixels, µm or nm) in a format.json in each condition's "
     "track_tables folder, for example {\"units\": \"um\"}, then open this folder again. "
     "A wrong unit changes every D by the square of the factor."),
    ("could not be read",
     "Some tracks could not be read",
     "Fix or remove the files named above, then open this folder again."),
    ("the pixel size (nm) differs between conditions",
     "The conditions were recorded at different pixel sizes",
     "Open conditions with the same pixel size together; sizes up to 1 % apart load with a "
     "warning. If a pixel size is wrong, correct it where it is recorded (u-track's movie "
     "settings, or Calibration & save in Preparation & tracking) and open this folder again."),
    ("do not record their acquisition parameters",
     "The pixel size or frame interval is not recorded",
     "State them on Calibration & save in Preparation & tracking (Back to menu), which writes them "
     "beside the tracks, then open this folder again."),
    ("were not all tracked by the same tool",
     "The conditions were tracked by different tools",
     "Open the conditions of each tracker separately, or open the folder again and choose one "
     "tracker for every condition."),
    ("replicate mapping for",
     "A condition's replicates could not be read",
     "Correct or delete that condition's day_map.json (the replicates are then numbered from the "
     "series again), then open this folder again."),
    ("every movie lacks cell outlines",
     "No movie has cell outlines",
     "Draw the outlines in Cell outlines (Back to menu), then open this folder again."),
    ("project folder not found",
     "This folder cannot be found",
     "Choose a different folder."),
)

#: A refusal's code words, in the words the screens use.
_CODE_WORDS = (("pixel_size_nm", "the pixel size (nm)"), ("frame_interval_s", "the frame interval (s)"),
               ("frame_interval_ms", "the frame interval (ms)"))

#: Folders `holds_acquisitions` never descends into: tracker output, never a camera file.
_NOT_RAW = {"trackingpackage", "trackmate", "track_tables", "cell_meshes", "backups", ".smtracker_warm"}


def holds_acquisitions(root, *, depth: int = 3) -> bool:
    """Whether ``root`` or a folder up to ``depth`` levels under it holds a raw acquisition (``.nd2``, ``.stk``):
    the sign that a folder with no tracks is one not TRACKED yet. Stops at the first one found."""
    import os

    from .rawmovie import ACQUISITION_SUFFIXES

    root = os.path.normpath(str(root))
    try:
        for here, dirs, files in os.walk(root):
            if any(os.path.splitext(f)[1].lower() in ACQUISITION_SUFFIXES for f in files):
                return True
            if len(os.path.relpath(here, root).split(os.sep)) >= depth:
                dirs[:] = []
            else:
                dirs[:] = [d for d in dirs if d.lower() not in _NOT_RAW]
    except OSError:
        return False
    return False


def load_failure(message: str, root=None, *, has_acquisitions: bool = False) -> dict:
    """The loading page's words for a folder that could not be opened: ``{"title", "detail", "advice", "summary"}``.

    ``message`` is the whole refusal (``"ProjectError: ..."``), never cut: the reason is often its last clause, and a
    cut at 200 characters lost it ("so these are not one pro"). The exception's name is dropped (a code word), and a
    path under ``root`` is shown from ``root`` on (the folder is named above), so a line has room to wrap.
    ``has_acquisitions`` (`holds_acquisitions`) turns "nothing tracked here" into "track these first"."""
    text = str(message or "").strip()
    text = _EXCEPTION_PREFIX.sub("", text, count=1)
    if root:
        root_text = str(root).rstrip("/\\")
        for sep in ("\\", "/"):
            text = text.replace(root_text + sep, "")
    for code, words in _CODE_WORDS:
        text = text.replace(code, words)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    nothing_here = ("nor any folder under it contains" in text) or text.startswith("No conditions found")
    if nothing_here:
        # discovery's hint ("This should be the folder containing ...") is the advice now, said once
        detail = ("Nothing in this folder or under it holds u-track results (TrackingPackage/tracks), TrackMate "
                  "files (trackmate/*.xml) or track tables (track_tables/).")
        if has_acquisitions:
            return {"title": "These movies are not tracked yet",
                    "detail": detail + " It holds raw acquisitions (.nd2 or .stk).",
                    "advice": "Track them first in Preparation & tracking (Back to menu), then open this folder "
                              "here again.",
                    "summary": "not tracked yet"}
        return {"title": "No conditions found here",
                "detail": detail,
                "advice": "Choose the folder that contains your condition folders (or one condition folder), "
                          "with movies tracked by u-track, TrackMate or a track table.",
                "summary": "nothing found"}
    for pattern, title, advice in _FAILURES:
        if pattern in text:
            return {"title": title, "detail": " ".join(lines), "advice": advice,
                    "summary": "could not be opened"}
    return {"title": lines[0] if lines else "This folder could not be opened",
            "detail": " ".join(lines[1:]),
            "advice": "Nothing was loaded. Fix what is named above and open this folder again, or choose a "
                      "different folder.",
            "summary": "could not be opened"}


def unsegmented_notice(dropped, conditions) -> dict | None:
    """The window shown after a load that LEFT OUT movies without cell outlines (asked 22 Oct 2026, a screenshot of
    "CellMeshError: no cell outlines for movies: ['Malate_series011', 'Malate_series021']": "just load it and give a
    window telling us"). ``dropped`` is ``canonical.attrs["dropped_unsegmented"]`` (qualified names,
    ``<condition>_<movie>``); ``conditions`` the project's. Returns ``{"title", "body", "movies"}``, or None when
    nothing was left out."""
    dropped = [str(m) for m in (dropped or [])]
    if not dropped:
        return None
    named = []
    for qualified in dropped:
        hit = max((c for c in conditions if qualified.startswith(f"{c}_")), key=len, default=None)
        named.append((qualified[len(hit) + 1:], hit) if hit else (qualified, None))
    n = len(named)
    listed = [f"{m} in {c}" if c else m for m, c in named]
    title = f"{n} movie{'' if n == 1 else 's'} without cell outlines left out"
    body = ((listed[0] if n == 1 else ", ".join(listed[:-1]) + " and " + listed[-1])
            + (" was" if n == 1 else " were") + " tracked but never segmented, so " + ("its" if n == 1 else "their")
            + " tracks cannot be placed in cells. Everything else is open. To include "
            + ("it" if n == 1 else "them") + ", draw the outlines in the Cell outlines window and open the folder "
            "again. QC lists " + ("it" if n == 1 else "them") + " per condition.")
    return {"title": title, "body": body, "movies": listed}


def tab_rows(widths, available: float, *, gap: float = 4.0, pad: float = 4.0) -> list:
    """The Confirm screen's condition tabs packed into rows that fit ``available`` px:
    ``[(start, stop), ...]``, each a slice of the tabs in order, as many to a row as fit.

    Asked 15 Oct 2026: with many conditions "u only see a few of it and the rest are not
    visible cause the text is too long" -- one fixed row of 14 of one user's names was 3 017 px
    wide. A tray is ``pad`` on each side, its tabs ``gap`` apart (`Segmented`'s own). A tab
    wider than a row on its own still gets a row of its own, never none."""
    widths = [float(w) for w in widths]
    rows, start, used = [], 0, 2.0 * pad
    for i, w in enumerate(widths):
        need = w if i == start else gap + w
        if i > start and used + need > available:
            rows.append((start, i))
            start, used, need = i, 2.0 * pad, w
        used += need
    if widths:
        rows.append((start, len(widths)))
    return rows
