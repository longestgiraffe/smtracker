"""The Spatial Map's replicate page: what it shows and how its player steps. Headless.

The redesign of the replicate-day page (mockup 4 of the UI designer's spot-card notes, approved
13 Oct 2026), tried behind the rail's Test button until 23 Nov 2026 and since then THE
replicate page (asked: "I like the redes map page can we finish the idea and include"; the
page it replaced, `SpatialDayDetail`, is gone). Its left card is the days' map and the
clicked spot; its right card is the spot's largest tracks as JUMP STRIPS and a player that
plays the picked track's own frames, zoomed to its cell.

Invariants:
- Nothing is computed here that the spot does not already carry. Every number comes from
  `spatial_map_view.spot_contributors` (its `tracks`, `cells`, `by_track`, `by_cell`,
  `floor_nm`); this module only arranges and words them.
- Colour says RANK, never identity: one neutral ramp, darkest = largest, the same in the
  Cells and Tracks columns and the strips. Bright hues are the conditions' everywhere
  else in the workspace, and they would compete with the map's own colormap.
- A track hidden from the map by its eye stays LISTED under the spot it was hidden from
  (`listed`), after the spot's own tracks, so one more click brings it back; it is never
  counted in the spot's shares, which are the redrawn map's.
- A track's frames are the tracker's own numbering: frame f is page f - 1 of
  `<condition>/tif/<movie>.tif` (measured, `docs/claude/spatial_maps.md`). The clip is
  the track's frames plus `images_view.LEAD_FRAMES` either side, clamped to the movie.

Interfaces: `listed`, `strip_rows`, `axis_max_nm`, `axis_ticks`, `contributor_rows`,
`meter`, `spot_notes`, `ink_ramp`, `clip_window`, `advance`, `travelled`, `caption`,
`frame_text`, the speeds, and the page's words.
"""
from __future__ import annotations

from math import ceil

from .images_view import LEAD_FRAMES

__all__ = ["listed", "spot_notes", "no_spots_text", "HIDDEN_TEXT", "EYE_TIPS", "strip_rows", "axis_max_nm", "axis_ticks", "contributor_rows", "meter",
           "ink_ramp", "clip_window", "advance", "travelled", "caption", "frame_text",
           "spot_heading", "spot_position", "track_view", "step_bin", "TIP_SELECT", "TIP_MOVE", "SPEEDS", "SPEED_LABELS", "SPEED_TIPS",
           "DEFAULT_SPEED", "BACKGROUNDS", "BACKGROUND_TIPS", "TOP", "clip_result"]

#: How many cells and tracks each column names (the spot record's own `top`).
TOP = 5

#: One neutral ramp per scheme, darkest = largest in light (lightest in dark), then the
#: "other" swatch. Zinc, the workspace's own neutrals.
_RAMP = {
    "light": (("#18181B", "#3F3F46", "#71717A", "#A1A1AA", "#C4C4CC"), "#E4E4E7"),
    "dark": (("#FAFAFA", "#D4D4D8", "#A1A1AA", "#71717A", "#52525B"), "#3F3F46"),
}

#: Playback speeds against the acquisition's own frame rate. Slower than the Images page
#: offers (½×): a 14-frame track lasts under 0.3 s at 1×.
SPEEDS = (0.25, 0.5, 1.0, 4.0)
#: The bundled Inter carries ¼ and ½ (U+00BC, U+00BD) and × (U+00D7), checked against its cmap.
SPEED_LABELS = ("¼×", "½×", "1×", "4×")
SPEED_TIPS = ("A quarter of the acquisition's frame rate",
              "Half the acquisition's frame rate",
              "The acquisition's own frame rate",
              "Four times the acquisition's frame rate")
DEFAULT_SPEED = 0.5

#: What the player draws behind the track: the recorded frames, or one still picture.
BACKGROUNDS = ("Movie", "Still")
BACKGROUND_TIPS = ("The recorded frames, as the tracker saw them",
                   "One still picture of the cell (the phase image)")

STRIPS_TITLE = "The spot's largest tracks: pick one to look at"
SPOT_HINT = "Click a spot on the map to see which cells and tracks make it."
PICK_HINT = "Click a spot on the map, then pick one of its tracks to play it."
READING = "Reading frames…"
#: What a track hidden from the map shows where its share was.
HIDDEN_TEXT = "hidden from the map"
#: The eye's two tips (asked 16 Oct 2026: "an eye symbol, if u click on it it gets a / on the eye and the track is
#: excluded from the map").
EYE_TIPS = ("Hide this track from the map", "Hidden from the map: click to show it again")


def ink_ramp(scheme: str = "light") -> tuple[tuple[str, ...], str]:
    """The five rank shades and the "other" shade for a colour scheme."""
    return _RAMP["dark" if scheme == "dark" else "light"]


def _short(name, condition) -> str:
    from .spatial_map_view import _short as short

    return short(name, condition)


def _cell_number(cell, condition) -> str | None:
    """"Glucose_series050:3" -> "3"; None when the name carries no cell."""
    short = _short(cell, condition)
    head, sep, tail = short.rpartition(":")
    return tail if sep and head else None


def _track_title(track: dict) -> str:
    tid = track.get("track_id")
    if tid is None:
        return f"Track {str(track['uid']).rpartition(':')[2]}"
    if isinstance(tid, float) and tid.is_integer():
        tid = int(tid)
    return f"Track {tid}"


def _where(movie, cell, condition) -> str:
    number = _cell_number(cell, condition)
    movie_name = _short(movie, condition)
    return f"{movie_name} · cell {number}" if number is not None else movie_name


def listed(spot: dict, hidden_here: list[dict], hidden_uids) -> dict:
    """The spot as the page lists it: its own tracks that are not hidden from the map, then
    the tracks hidden from THIS spot (each the track record it had when it was hidden, marked
    ``hidden``), so they can be brought back. The redrawn map no longer has them, so nothing
    else of the spot (its shares, n_eff, the other count) counts them."""
    hidden_uids = set(hidden_uids)
    shown = [tr for tr in spot["tracks"] if tr["uid"] not in hidden_uids]
    seen = {tr["uid"] for tr in shown}
    back = [{**tr, "hidden": True} for tr in hidden_here if tr["uid"] not in seen]
    return {**spot, "tracks": shown + back}


def strip_rows(spot: dict) -> list[dict]:
    """One row per listed track, in the spot's order (largest first, the hidden ones last)."""
    cond = spot.get("condition")
    floor = spot.get("floor_nm")
    rows = []
    for rank, tr in enumerate(spot["tracks"]):
        jumps = [float(j) for j in tr.get("jumps_nm") or []]
        inside = (tr.get("within_floor") if tr.get("within_floor") is not None
                  else (None if floor is None else sum(j <= floor for j in jumps)))
        hidden = bool(tr.get("hidden"))
        rows.append({"hidden": hidden,
            "uid": tr["uid"], "rank": rank, "title": _track_title(tr),
            "movie": _short(tr["movie"], cond), "cell_number": _cell_number(tr["cell"], cond),
            "where": _where(tr["movie"], tr["cell"], cond),
            "jumps_nm": jumps, "median_nm": tr.get("median_jump_nm"),
            "inside": inside, "n_jumps": len(jumps), "share": float(tr["share"]),
            "first_frame": tr.get("first_frame"), "last_frame": tr.get("last_frame"),
            "inside_text": (HIDDEN_TEXT if hidden else
                            "" if inside is None else f"{inside} of {len(jumps)} inside"),
        })
    return rows


_CEILINGS = (100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000)


def axis_max_nm(rows: list[dict], floor_nm=None) -> float:
    """The strips' shared axis end: the smallest round ceiling over every jump and the
    band, so no tick is ever clipped."""
    top = max([0.0] + [j for r in rows for j in r["jumps_nm"]] + [float(floor_nm or 0)])
    for c in _CEILINGS:
        if top <= c:
            return float(c)
    return float(ceil(top / 5000.0) * 5000)


def axis_ticks(axis_max: float) -> list[float]:
    """Three to six labelled ticks from 0 to `axis_max`."""
    for step in (25, 50, 100, 150, 200, 250, 500, 1000, 2000, 2500, 5000):
        if axis_max / step <= 6:
            return [float(step * k) for k in range(int(axis_max // step) + 1)]
    return [0.0, float(axis_max)]


def meter(spot: dict, unit: str) -> dict:
    """The worth line of one column: n_eff of N, and the meter's fill (0..1)."""
    part = spot["by_cell" if unit == "cells" else "by_track"]
    n = int(part["n"])
    return {"label": "Cells" if unit == "cells" else "Tracks",
            "value": f"{part['n_eff']:.1f}", "what": f"independent {unit}",
            "of": f"of {n:,}", "fill": (part["n_eff"] / n) if n else 0.0}


def _pct(share: float) -> str:
    from .spatial_map_view import _pct as pct

    text = pct(share)
    return text if text.startswith("<") else text.replace("%", " %")


def contributor_rows(spot: dict, unit: str) -> dict:
    """The named contributors of one column and the rest as one "other" row.

    Cells: (name "Cell 3", movie, share); tracks: (name "Track 112", movie, "cell 1", share).
    `shares` are the bar's segments (0..1), largest first; `other` the remainder."""
    cond = spot.get("condition")
    part = spot["by_cell" if unit == "cells" else "by_track"]
    rows = []
    if unit == "cells":
        for c in spot.get("cells") or []:
            number = _cell_number(c["cell"], cond)
            rows.append({"name": f"Cell {number}" if number is not None else
                         _short(c["cell"], cond), "movie": _short(c["movie"], cond),
                         "cell": None, "share": c["share"], "pct": _pct(c["share"])})
    else:
        for tr in spot["tracks"]:
            number = _cell_number(tr["cell"], cond)
            hidden = bool(tr.get("hidden"))
            rows.append({"name": _track_title(tr), "movie": _short(tr["movie"], cond),
                         "cell": None if number is None else f"cell {number}",
                         "share": 0.0 if hidden else tr["share"],
                         "pct": HIDDEN_TEXT if hidden else _pct(tr["share"]), "uid": tr["uid"],
                         "hidden": hidden})
    counted = [r for r in rows if not r.get("hidden")]
    named = sum(r["share"] for r in counted)
    rest = max(0, int(part["n"]) - len(counted))
    other = max(0.0, 1.0 - named)
    return {"rows": rows, "other_n": rest,
            "other": (f"{rest:,} other" if rest else ""),
            "other_share": other, "other_pct": _pct(other) if rest else ""}


def spot_heading(spot: dict) -> str:
    """"Replicate 3 · Short jumps" -- a split by hand named by its own range."""
    from .spatial_map_view import day_label, kinds_for, panel_titles_for

    method = spot.get("method") or "sqd"
    n_pop = spot.get("n_pop") or 2
    ranges = spot.get("ranges")
    kinds = kinds_for(method, n_pop, ranges)
    title = (panel_titles_for(method, n_pop, ranges)[kinds.index(spot["kind"])]
             if spot["kind"] in kinds else str(spot["kind"]))
    return f"{day_label(spot['day'])} · {title}"


def no_spots_text(payload: dict) -> str:
    """What the page says where a condition's replicate maps keep no spot record: a turning-back map
    (`spatialangles`) keeps pairs of steps, never which tracks made a spot; any other map was made
    before the record existed."""
    if payload.get("show") in ("angles", "turning"):
        return "The spots of a turning-back map cannot be opened: it keeps pairs of steps, not their tracks."
    return "Run the map again to inspect its spots."


def spot_notes(spot: dict) -> list[str]:
    """The quiet lines under the spot's position, only where they say something of THIS spot:
    on a value map, that the listed tracks are the WHOLE value bin's; for a grey bin, why it is
    grey -- the spot readout's own rows (`spatial_map_view.spot_rows`), so the words exist once.
    An ordinary bin's size and kernel are the map's settings (its caption), the same at every
    spot, and a line for them pushed the maps off a 768 px laptop screen (23 Nov 2026)."""
    from .spatial_map_view import spot_rows

    rows = spot_rows(spot)
    whole = bool(spot.get("value_bin")) and int(spot.get("bin_factor", 1)) > 1
    notes = [value for name, value in rows if name == "Grey" or (name == "Bin" and whole)]
    made = dict(rows).get("Made of", "")
    if made.startswith("nothing"):
        notes.append("Made of " + made)
    return notes


def spot_position(spot: dict) -> str:
    """The quiet line under the heading: where the spot is, and whether it is folded."""
    text = f"{spot['x_um']:.2f} µm along · {spot['y_um']:+.2f} µm across".replace("-", "−")
    if spot.get("mirrored"):
        text += " · folded"
    if spot.get("masked"):
        text += " · a masked bin"
    return text


#: The replicate page's hover lens (asked 14 Oct 2026): its diameter and how far from the pointer
#: it sits (never over what is pointed at), in LOGICAL pixels (Qt's, so 150 % Windows scaling
#: and Retina are the same apparent size).
LOUPE_SIZE = 150
LOUPE_OFFSET = 22
#: A bin needs no lens from this many logical pixels (its mean side, `bin_side`). Measured on
#: HiWi Test (Malate, 14 Oct 2026): 100 nm bins are 9-11 px on screen and want it; 345 nm bins
#: are 27 x 23 px and plainly do not. WCAG's 24 px click-target minimum was tried first and
#: was too strict: it put the lens on 23 px bins anyone reads and clicks easily.
LOUPE_TARGET_PX = 16.0
#: The lens enlarges a bin to this size, so it always shows about four bins across.
LOUPE_SHOWN_PX = 36.0
LOUPE_MAX_ZOOM = 4.0
#: How far the lens's picture travels towards the pointer per frame (~60 a second): a 1 px
#: pointer step is 4 px of picture at 4x, so it GLIDES there instead of jumping (asked 14 Oct
#: 2026: "the zoomed image kinda moves disruptedly").
LOUPE_GLIDE = 0.45
LOUPE_FRAME_MS = 16


def bin_side(width: float, height: float) -> float:
    """A bin's size on screen as one number: the mean side (geometric), so a long, thin bin
    is judged by its area rather than by its thinner side."""
    return (max(0.0, float(width)) * max(0.0, float(height))) ** 0.5


def loupe_wanted(bin_px: float) -> bool:
    """Whether a bin drawn `bin_px` (its `bin_side`, logical pixels) needs the lens."""
    return 0 < float(bin_px) < LOUPE_TARGET_PX


def loupe_zoom(bin_px: float) -> float:
    """How much the lens magnifies a bin drawn `bin_px` wide: to about `LOUPE_SHOWN_PX`, never
    below 1 nor above `LOUPE_MAX_ZOOM`."""
    if not bin_px > 0:
        return 1.0
    return min(LOUPE_MAX_ZOOM, max(1.0, LOUPE_SHOWN_PX / float(bin_px)))


#: The tips beside the map's title (asked 14 Oct 2026): how to select a spot, then how to move
#: it. No modes: the mouse always selects, the arrow keys always move the selection.
TIP_SELECT = "Tip: click any spot on a map to see what makes it"
TIP_MOVE = "Tip: \u2190 \u2192 \u2191 \u2193 move the spot \u00b7 Esc clears it"


def step_bin(day_payload: dict, ix: int, iy: int, dx: int, dy: int):
    """The bin one step from (ix, iy) in direction (dx, dy) -- dy = +1 is UP, towards row 0's
    far side, as row 0 is the bottom wall -- or None at the cell's edge. Masked bins are
    stepped onto like any other: they can be clicked too, and the card says they are masked."""
    inside = (day_payload.get("spots") or {}).get("inside")
    if not inside:
        return None
    n_short, n_long = len(inside), len(inside[0])
    jx, jy = int(ix) + int(dx), int(iy) + int(dy)
    if not (0 <= jx < n_long and 0 <= jy < n_short) or not inside[jy][jx]:
        return None
    return jx, jy


def clip_window(first: int, last: int, movie_first: int, movie_last: int,
                lead: int = LEAD_FRAMES) -> tuple[int, int]:
    """The frames the player plays: the track's, `lead` either side, inside the movie."""
    lo = max(int(movie_first), int(first) - lead)
    hi = min(int(movie_last), int(last) + lead)
    if hi < lo:
        raise ValueError(f"track frames {first}-{last} lie outside the movie's "
                         f"{movie_first}-{movie_last}")
    return lo, hi


def advance(frame: float, step: float, lo: int, hi: int, loop: bool) -> tuple[float, bool]:
    """One tick of the clock: the next (fractional) frame and whether it still plays.
    A loop wraps from past `hi` back to `lo`; otherwise it stops on `hi`."""
    nxt = float(frame) + float(step)
    if nxt <= hi:
        return nxt, True
    if not loop:
        return float(hi), False
    span = hi - lo + 1
    return lo + (nxt - lo) % span, True


def travelled(frames: list[int], frame: int) -> tuple[int, bool]:
    """How much of a track has been travelled by `frame`: the index of its last point at
    or before it (-1 before the track starts), and whether its molecule is ON this frame."""
    from bisect import bisect_right

    k = bisect_right(frames, int(frame)) - 1
    return k, (k >= 0 and frames[k] == int(frame))


#: The player's view around a track, in pixels: never narrower than this (about 1.7 µm
#: at 106 nm), and the track's own extent times `TRACK_PAD` when it is larger. A whole
#: cell leaves a stuck molecule a few pixels wide.
TRACK_VIEW_MIN_PX = 16.0
TRACK_PAD = 1.6


def track_view(points, image_size=None, *, min_px: float = TRACK_VIEW_MIN_PX,
               pad: float = TRACK_PAD) -> list[float]:
    """The player's view of one track: a square around its points, `pad` times its extent
    and at least `min_px` wide, kept inside the image when the image is larger."""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    side = max(min_px, pad * max(max(xs) - min(xs), max(ys) - min(ys)))
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    if image_size is not None:
        w, h = float(image_size[0]), float(image_size[1])
        if side <= w:
            x0 = min(max(x0, -0.5), w - 0.5 - side)
        if side <= h:
            y0 = min(max(y0, -0.5), h - 0.5 - side)
    return [x0, y0, x0 + side, y0 + side]


def caption(row: dict) -> tuple[str, str]:
    """The player's caption: the track's name, then where and when it is."""
    rest = row["where"]
    if row.get("first_frame") is not None:
        rest += f" · frames {row['first_frame']}–{row['last_frame']}"
    return row["title"], rest


def frame_text(frame, lo: int, hi: int) -> str:
    """The counter beside the bar."""
    return f"frame {int(frame):,}" if frame is not None else f"frames {lo:,}–{hi:,}"


def clip_result(project, condition: str, uid: str) -> dict:
    """The player's input for one PROJECT uid (the spot's track names): which movie and
    cell it is, the movie's payload (outlines, tracks, still picture), the clip's frame
    window, the cell's box, and the recorded frames -- or why they could not be read, in
    which case the player shows the still. Run on a worker thread."""
    from pathlib import Path

    from .images_view import _spots, movie_payload, movie_root
    from .trackclip import TrackClipError, clip_frames
    from .trackviewer import track_target

    target = track_target(project, condition, uid, projected=False)
    movie, short_uid = target["short_movie"], target["short_uid"]
    payload = movie_payload(project, condition, movie, background="phase")
    track = next((tr for tr in payload["tracks"] if tr["uid"] == short_uid), None)
    if track is None:
        raise LookupError(f"track {short_uid} is not in {movie}'s tracks")
    axis = payload["time"]
    lo, hi = clip_window(track["frames"][0], track["frames"][-1], axis["first"], axis["last"])
    cell = track["cell"]
    image = payload.get("image")
    view = track_view(track["points"],
                      None if image is None else (image["width"], image["height"]))
    recorded = payload.get("frames")
    expected = int(recorded[0]) if recorded else None
    clip, clip_error = None, ""
    try:
        # The movie's OWN folder, and its localisations so the frames are checked to line
        # up (`trackclip`: the stack, u-track's frames, or the raw file with its range).
        clip = clip_frames(Path(movie_root(project, condition, movie)), movie, lo, hi,
                           expected_frames=expected, spots=_spots(payload["tracks"]))
    except TrackClipError as exc:
        clip_error = str(exc)
    return {"payload": payload, "track": track, "cell": cell, "view": list(view),
            "lo": lo, "hi": hi, "clip": clip, "clip_error": clip_error, "movie": movie}
