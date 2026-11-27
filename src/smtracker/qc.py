"""Quality control: "should I trust this dataset?", answered before comparing.

Everything here is pure numpy/pandas and headless-testable; `smtracker.plots`
draws it and `smtracker.widgets` docks it, neither computing anything.

What this is for
----------------
Run before any condition comparison. The panels answer, in order:

1. **What did the length filter cost?** (``length_distribution``) Track length is
   not neutral: long tracks are preferentially slow (CLAUDE.md), so the length
   floor (fewer than 5 localisations: 3 steps or fewer) removes a *biased* slice,
   and how big that slice is belongs on screen. The removed tracks have no rows in
   the table QC is given; `canonical.to_canonical` records them and they are put
   back here, drawn as removed.
2. **Was this really single-molecule?** (``localizations_per_frame``) Linking is
   only trustworthy while molecules are sparse enough that the nearest detection
   in the next frame is the same molecule. Frames denser than that are *flagged
   with the criterion printed*, never auto-cropped: this is a display, not the
   auto-window tool.
3. **What is n, actually?** (``group_counts``) Tracks, localizations and cells
   per condition/day/movie. The pseudoreplication rule says the unit is the cell
   or the day, so those counts (not 13 500 displacements) are the sample size
   every claim rests on.
4. **Is one day or one cell carrying the result?** (``unit_summary`` +
   ``flag_outliers``) With n = 3 days a single bad day swings a fraction, so the
   per-unit spread must be visible *before* pooling, not discovered after.

Outliers are FLAGGED, never dropped
-----------------------------------
``flag_outliers`` uses **median ± k·MAD**, not mean ± SD: with a handful of units
the mean and SD are themselves dragged by the very outlier being looked for, so a
mean-based rule hides what it is meant to find. The threshold used is returned
and printed. Nothing here removes data: dropping a day is a scientific
decision, and the tool's job is to make it visible and deliberate.

Invariants
----------
- Every function reports its own N.
- Fails loud on a non-canonical table or an empty group.
- No Qt, no napari, no matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS, DEFAULT_MIN_LOCALIZATIONS, AcquisitionParams, longest_removed_steps

__all__ = [
    "control_contamination",
    "AUTOFLUORESCENT_MEDIA",
    "QCError",
    "DEFAULT_MAD_K",
    "TRACK_SUMMARY_COLUMNS",
    "track_summary",
    "length_distribution",
    "localizations_per_frame",
    "group_counts",
    "detection_density",
    "CELL_INVENTORY_COLUMNS",
    "sample_size",
    "in_cell_share",
    "IN_CELL_FLAG_DROP",
    "unit_summary",
    "flag_outliers",
    "OutlierFlags",
]

#: Robust-outlier width. 3.5 MAD ≈ 2.4 σ for normal data, i.e. deliberately
#: conservative: this flags for inspection, so a false alarm costs a glance
#: while a miss costs a wrong conclusion.
DEFAULT_MAD_K = 3.5

#: Scale factor making the MAD a consistent estimator of σ for normal data.
_MAD_TO_SIGMA = 1.4826
#: Same, for the mean absolute deviation: the fallback when the MAD is 0.
_MEANAD_TO_SIGMA = 1.2533

TRACK_SUMMARY_COLUMNS = [
    "uid", "condition", "day", "movie", "cell",
    "n_localizations", "n_steps", "n_gaps", "first_frame", "duration_s",
    "median_step_um", "appd_um2_per_s", "net_displacement_um", "radius_gyration_um",
]


class QCError(RuntimeError):
    """Raised when a QC view cannot be computed honestly."""


def _require_canonical(canonical: pd.DataFrame) -> AcquisitionParams:
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise QCError(f"not a canonical table (missing {missing})")
    if canonical.empty:
        raise QCError("no localizations")
    params = canonical.attrs.get("params")
    if not isinstance(params, AcquisitionParams):
        raise QCError("canonical table carries no AcquisitionParams")
    return params


# --------------------------------------------------------------------------- #
# The per-track table, drives both the track list and the QC panels
# --------------------------------------------------------------------------- #


def track_summary(canonical: pd.DataFrame, *, hmm_fit=None) -> pd.DataFrame:
    """One row per track, with the facts needed to find it in a list.

    This is the single source of truth for "what is in this dataset, per track":
    the interactive track list, the N-per-group panel and the outlier panel all
    read it, so they cannot drift apart.

    ``median_step_um`` is deliberately the median, not the mean: a single
    tracking error produces one enormous step, and a mean would let it define
    the track. With ``hmm_fit`` given, ``slow_fraction`` and ``n_switches`` are
    added per track.

    VECTORIZED, not a per-track Python loop: a real project's per-track
    for-loop here (the original shape, still in git history) holds the GIL
    for the whole pass -- measured at ~1.2 s for 12 000 tracks -- which is
    long enough for a QThreadPool worker calling this (as `comparison.
    appd_points` does, once per condition) to visibly stall the GUI thread
    even though it is technically "off" it, because Python's GIL is one
    resource shared by both. Every quantity below is instead one pandas
    groupby aggregation over the WHOLE table, so the heavy lifting runs in
    C/Cython and actually releases the GIL. Every edge case the per-track
    loop handled (a 1-localization track: no steps, `median_step_um`/
    `appd_um2_per_s` NaN, `radius_gyration_um`/`net_displacement_um` 0) is
    preserved -- see `tests/test_qc.py::TestTrackSummary` and the
    differential check against the old loop in `scripts/bench_track_summary.py`.
    """
    params = _require_canonical(canonical)
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    by_uid = ordered.groupby("uid", sort=False, observed=True)

    # One row per track, literally its first/last localization -- `.nth`,
    # never `.first()`/`.last()`, because those SKIP a NaN value in the
    # column being reduced where `.iloc[0]`/`.iloc[-1]` (the original) never
    # would have.
    first = by_uid.nth(0).set_index("uid")
    last = by_uid.nth(-1).set_index("uid")
    sizes = by_uid.size()
    uids = sizes.index

    # A step belongs to the track it moved WITHIN -- `.diff()` on a grouped
    # Series resets (to NaN) at every group's own first row, exactly where
    # `np.diff(x)` has nothing to subtract from either.
    dx = by_uid["x_um"].diff()
    dy = by_uid["y_um"].diff()
    step_um = np.hypot(dx, dy)
    frame_gap = by_uid["frame"].diff()
    is_gap = frame_gap.notna() & (frame_gap != 1)

    by_step_uid = ordered["uid"]
    median_step = step_um.groupby(by_step_uid, observed=True).median()
    mean_sq_step = (step_um**2).groupby(by_step_uid, observed=True).mean()
    n_gaps = is_gap.groupby(by_step_uid, observed=True).sum()

    centred_sq = ((ordered["x_um"] - by_uid["x_um"].transform("mean"))**2
                 + (ordered["y_um"] - by_uid["y_um"].transform("mean"))**2)
    mean_centred_sq = centred_sq.groupby(by_step_uid, observed=True).mean()

    table = pd.DataFrame({
        "uid": uids,
        "condition": first.loc[uids, "condition"].to_numpy(),
        "day": first.loc[uids, "day"].to_numpy(),
        "movie": first.loc[uids, "movie"].to_numpy(),
        "cell": first.loc[uids, "cell"].to_numpy(),
        "n_localizations": sizes.to_numpy(),
        "n_steps": (sizes - 1).to_numpy(),
        "n_gaps": n_gaps.reindex(uids, fill_value=0).astype(int).to_numpy(),
        "first_frame": first.loc[uids, "frame"].astype(int).to_numpy(),
        "duration_s": ((last.loc[uids, "frame"] - first.loc[uids, "frame"])
                       * params.tau_s).astype(float).to_numpy(),
        "median_step_um": median_step.reindex(uids).to_numpy(dtype=float),
        "appd_um2_per_s": (mean_sq_step.reindex(uids) / (4.0 * params.tau_s)
                          ).to_numpy(dtype=float),
        "net_displacement_um": np.hypot(
            (last.loc[uids, "x_um"] - first.loc[uids, "x_um"]).to_numpy(dtype=float),
            (last.loc[uids, "y_um"] - first.loc[uids, "y_um"]).to_numpy(dtype=float),
        ),
        "radius_gyration_um": np.sqrt(
            mean_centred_sq.reindex(uids).to_numpy(dtype=float)),
    }, columns=TRACK_SUMMARY_COLUMNS)
    if hmm_fit is not None:
        table = _add_hmm_columns(table, canonical, hmm_fit)
    table.attrs["n_tracks"] = len(table)
    table.attrs["n_localizations"] = int(table["n_localizations"].sum())
    table.attrs["tau_s"] = params.tau_s
    return table


def _add_hmm_columns(table: pd.DataFrame, canonical: pd.DataFrame, fit) -> pd.DataFrame:
    """Per-track slow fraction and switch count, from posterior decoding."""
    from .hmm import decode_states

    decoded = decode_states(canonical, fit)
    grouped = decoded.sort_values(["uid", "frame"]).groupby("uid", observed=True)
    stats = grouped["state"].agg(
        slow_fraction=lambda s: float((s == 1).mean()),
        n_switches=lambda s: int((np.diff(s.to_numpy()) != 0).sum()),
        n_decoded_steps="size",
    )
    return table.merge(stats, left_on="uid", right_index=True, how="left")


# --------------------------------------------------------------------------- #
# Panel 1, what the length filter costs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LengthDistribution:
    """Steps-per-track, and the price of the length floor: tracks of ``min_steps`` steps or fewer are removed (3, at
    the default 5 localisations)."""

    counts: np.ndarray
    edges: np.ndarray
    min_steps: int
    n_tracks: int
    n_tracks_removed: int
    n_localizations: int
    n_localizations_removed: int
    #: A user's cap on track length (`canonical.cap_track_length`), or None.
    max_steps: int | None = None
    #: Whether the table recorded the tracks the length floor removed (``attrs["length_floor_removed"]``, written by
    #: `canonical.to_canonical`). False for a table built some other way: its removed count then misses them.
    floor_recorded: bool = True

    @property
    def fraction_tracks_removed(self) -> float:
        return self.n_tracks_removed / self.n_tracks if self.n_tracks else 0.0

    @property
    def fraction_localizations_removed(self) -> float:
        return (
            self.n_localizations_removed / self.n_localizations
            if self.n_localizations else 0.0
        )

    def caption(self) -> str:
        kept = (f"keep {self.min_steps + 1} or more steps" if self.max_steps is None
                else f"keep {self.min_steps + 1} to {self.max_steps} steps")
        unrecorded = ("" if self.floor_recorded else
                      " | tracks the length floor removed before this table are not recorded")
        return (
            f"N = {self.n_tracks} tracks / {self.n_localizations} localizations | "
            f"filter: {kept} | "
            f"removed {self.n_tracks_removed} tracks "
            f"({100 * self.fraction_tracks_removed:.1f} %) and "
            f"{self.n_localizations_removed} localizations "
            f"({100 * self.fraction_localizations_removed:.1f} %)"
            f"{unrecorded}"
        )


def length_distribution(
    canonical: pd.DataFrame, *, min_steps: int | None = None, summary: pd.DataFrame | None = None
) -> LengthDistribution:
    """Histogram of steps per track, with the cost of the length cut.

    ``min_steps`` is the filter boundary: tracks with **more** than this many
    steps are kept. By default it is the floor the table was filtered with
    (`canonical.longest_removed_steps` of its ``min_localizations``: 3 steps at 5
    localisations, so a 4-step track counts as kept, as it is in every analysis).
    The cost is reported for both tracks and localizations, because those differ a
    lot when the discarded tracks are the short ones.
    """
    table = track_summary(canonical) if summary is None else summary
    if table.empty:
        raise QCError("no tracks to summarise")
    if min_steps is None:
        min_steps = longest_removed_steps(canonical.attrs.get("min_localizations", DEFAULT_MIN_LOCALIZATIONS))
    steps = table["n_steps"].to_numpy()
    removed = steps <= min_steps
    # THE TRACKS A USER'S CAP LEFT OUT are not in the table; they are put back here, as
    # removed, from the record the cap kept -- this condition's only.
    cap = canonical.attrs.get("track_cap") or {}
    conditions = set(table["condition"].astype(str)) if "condition" in table else set()
    capped = [r for r in cap.get("removed", ())
              if not conditions or r["condition"] in conditions]
    cut_steps = np.array([r["steps"] for r in capped], dtype=int)
    # AND THE TRACKS THE LENGTH FLOOR REMOVED, the same way, from `to_canonical`'s record ({condition: {steps: n}}).
    recorded = canonical.attrs.get("length_floor_removed")
    floor = [int(s) for c, hist in (recorded or {}).items() if not conditions or c in conditions
             for s, k in hist.items() for _ in range(int(k))]
    floor_steps = np.array(floor, dtype=int)
    everything = np.concatenate([steps, cut_steps, floor_steps])
    edges = np.arange(0, everything.max() + 2) - 0.5
    counts, _ = np.histogram(everything, bins=edges)
    return LengthDistribution(
        counts=counts,
        edges=edges,
        min_steps=int(min_steps),
        n_tracks=len(table) + len(capped) + len(floor_steps),
        n_tracks_removed=int(removed.sum()) + len(capped) + len(floor_steps),
        n_localizations=int(table["n_localizations"].sum())
        + sum(r["localizations"] for r in capped) + int((floor_steps + 1).sum()),
        n_localizations_removed=int(table.loc[removed, "n_localizations"].sum())
        + sum(r["localizations"] for r in capped) + int((floor_steps + 1).sum()),
        max_steps=cap.get("max_steps"),
        floor_recorded=recorded is not None,
    )


# --------------------------------------------------------------------------- #
# Panel 2: was this really single-molecule?
# --------------------------------------------------------------------------- #


def localizations_per_frame(
    canonical: pd.DataFrame,
    *,
    max_per_frame: int | None = None,
    search_radius_px: float | None = None,
    field_shape: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Localizations per frame per movie, with dense frames flagged.

    The criterion, printed rather than assumed: linking is ambiguous once a
    molecule's search disc is likely to contain a *second* molecule. With
    ``n`` localizations over a field of ``A`` pixels and a search radius ``r``,
    the expected number of others inside the disc is ``n·πr²/A``; the frame is
    flagged when that exceeds 1. Fall back to an explicit ``max_per_frame`` when
    the field size is unknown.

    Nothing is cropped: that is the auto-window tool's job, not a QC display's.

    The radius is each movie's CONDITION's own (tester D10, 4 Nov 2026: a project
    linked at 5 and 8 px flagged both at 5): on a table whose conditions were
    linked at different radii every movie gets its own threshold (the
    ``threshold`` column, ``attrs["threshold_by_condition"]``), ``attrs["threshold"]``
    is NaN and the criterion names each radius. An explicit ``search_radius_px``
    applies to every movie.
    """
    params = _require_canonical(canonical)

    table = (
        canonical.groupby(["movie", "frame"], observed=True)
        .size()
        .rename("n_localizations")
        .reset_index()
    )

    radius = search_radius_px
    if radius is None and max_per_frame is None and field_shape is not None:
        if params.has_mixed_search_radius:
            radii = {str(c): params.search_radius_px_for(c)
                     for c in pd.unique(canonical["condition"].astype(str))}
            from .canonical import distinct_radii

            if len(distinct_radii(radii.values())) > 1:
                return _localizations_per_frame_by_condition(canonical, table, radii, field_shape)
            radius = next(iter(radii.values()), None)
        else:
            radius = params.search_radius_px

    criterion = None
    if max_per_frame is not None:
        threshold = float(max_per_frame)
        criterion = f"explicit limit of {max_per_frame} localizations per frame"
    elif radius is not None and field_shape is not None:
        area = float(field_shape[0] * field_shape[1])
        threshold = area / (np.pi * radius**2)
        criterion = (
            f"expected neighbours inside a {radius:g} px search disc exceeds 1 "
            f"on a {field_shape[0]}x{field_shape[1]} px field "
            f"(> {threshold:.0f} localizations/frame)"
        )
    else:
        threshold = float("inf")
        criterion = (
            "no criterion available: needs a search radius and a field size, "
            "or an explicit max_per_frame"
        )

    table["ambiguous"] = table["n_localizations"] > threshold
    table.attrs["threshold"] = threshold
    table.attrs["criterion"] = criterion
    table.attrs["n_frames"] = int(len(table))
    table.attrs["n_ambiguous"] = int(table["ambiguous"].sum())
    return table


def _localizations_per_frame_by_condition(canonical, table, radii: dict, field_shape) -> pd.DataFrame:
    """`localizations_per_frame` on a table whose conditions were linked at different search radii: each movie's
    frames flagged at its OWN condition's disc (tester D10). ``radii`` is ``{condition: radius_px or None}``; a
    condition without a radius flags nothing and says so."""
    area = float(field_shape[0] * field_shape[1])
    by_condition = {c: (area / (np.pi * r**2) if r else float("inf")) for c, r in radii.items()}
    movie_condition = (canonical[["movie", "condition"]].drop_duplicates("movie")
                       .set_index("movie")["condition"].astype(str))
    table["threshold"] = table["movie"].map(movie_condition).map(by_condition).astype(float)
    table["ambiguous"] = table["n_localizations"] > table["threshold"]
    said = ", ".join(f"{c} {r:g} px (> {by_condition[c]:.0f})" if r else f"{c} no radius recorded"
                     for c, r in radii.items())
    table.attrs["threshold"] = float("nan")
    table.attrs["threshold_by_condition"] = by_condition
    table.attrs["criterion"] = (
        f"expected neighbours inside each condition's own search disc exceeds 1 on a "
        f"{field_shape[0]}x{field_shape[1]} px field (linked at different radii: {said} localizations/frame)")
    table.attrs["n_frames"] = int(len(table))
    table.attrs["n_ambiguous"] = int(table["ambiguous"].sum())
    return table


# --------------------------------------------------------------------------- #
# Panel 3: what n actually is
# --------------------------------------------------------------------------- #


def group_counts(
    canonical: pd.DataFrame,
    *,
    by=("condition", "day", "movie"),
    summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tracks / localizations / cells per group, the sample size, made visible.

    The independent unit here is the **cell** or the **day**, never the
    displacement (see the pseudoreplication rule), so the cell count is the
    column that matters for any interval or test.
    """
    _require_canonical(canonical)
    table = track_summary(canonical) if summary is None else summary
    keys = [k for k in by if k in table.columns]
    if not keys:
        raise QCError(f"none of {by} is a column of the summary")

    counts = table.groupby(keys, observed=True).agg(
        n_tracks=("uid", "size"),
        n_localizations=("n_localizations", "sum"),
        n_cells=("cell", "nunique"),
        median_steps=("n_steps", "median"),
    ).reset_index()
    counts.attrs["by"] = tuple(keys)
    counts.attrs["n_tracks"] = int(counts["n_tracks"].sum())
    return counts


# --------------------------------------------------------------------------- #
# Panel 4: is one unit carrying the result?
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OutlierFlags:
    """Robust outlier flags, with the threshold that produced them."""

    values: np.ndarray
    flagged: np.ndarray
    median: float
    scale: float
    k: float
    scale_estimator: str

    @property
    def low(self) -> float:
        return self.median - self.k * self.scale

    @property
    def high(self) -> float:
        return self.median + self.k * self.scale

    def caption(self) -> str:
        if self.scale == 0:
            return (
                f"n = {len(self.values)} units | every value identical "
                f"({self.median:.4g}); nothing to flag"
            )
        return (
            f"n = {len(self.values)} units | median {self.median:.4g} | "
            f"threshold median ± {self.k:g}·{self.scale_estimator} = "
            f"[{self.low:.4g}, {self.high:.4g}] | "
            f"{int(self.flagged.sum())} flagged"
        )


def flag_outliers(values, *, k: float = DEFAULT_MAD_K) -> OutlierFlags:
    """Flag by **median ± k·MAD**, never mean ± SD.

    With a handful of units the mean and SD are dragged by the very outlier
    being searched for, so a mean-based rule hides what it exists to find. The
    MAD does not move. Values are flagged for inspection and never removed.

    The MAD = 0 trap, which bites exactly where it matters
    ------------------------------------------------------
    With few units, a *single* deviant one is the whole point, n = 3 days with
    one bad day is the motivating case. But if the other units happen to agree
    closely, the median absolute deviation is **zero**, and ``median ± k·0``
    flags nothing: the rule goes blind precisely on the scenario it exists for.
    Seven identical cells plus one wild cell gives MAD = 0 and, naively, "all
    clear".

    So when the MAD vanishes this falls back to the (still robust, still
    median-centred) **mean absolute deviation about the median**, scaled to be
    comparable: the Iglewicz-Hoaglin remedy. Only when *every* value is
    identical is nothing flagged, which is then correct rather than blind. The
    estimator actually used is reported in ``scale_estimator`` and printed.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise QCError("no values to flag")
    if k <= 0:
        raise QCError(f"k must be > 0, got {k}")

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise QCError("no finite value to flag")
    median = float(np.median(finite))
    deviations = np.abs(finite - median)

    scale = float(np.median(deviations) * _MAD_TO_SIGMA)
    estimator = "MAD"
    if scale == 0:
        scale = float(deviations.mean() * _MEANAD_TO_SIGMA)
        estimator = "meanAD (MAD was 0)"

    if scale == 0:
        flagged = np.zeros(values.shape, dtype=bool)
    else:
        flagged = np.abs(values - median) > k * scale
    flagged &= np.isfinite(values)
    return OutlierFlags(
        values=values, flagged=flagged, median=median, scale=scale, k=k,
        scale_estimator=estimator,
    )


def unit_summary(
    canonical: pd.DataFrame,
    *,
    unit: str = "cell",
    hmm_fit=None,
    k: float = DEFAULT_MAD_K,
    summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-unit key readouts, with outliers flagged (never dropped).

    ``unit`` is ``"cell"``, ``"day"`` or ``"movie"``, the bootstrap units. Each
    row carries N, median track length and, when an HMM fit is supplied, the
    unit's slow fraction, which is the readout a bad day actually moves. Flags
    come from ``flag_outliers`` on the slow fraction where available, otherwise
    on the median track length.
    """
    _require_canonical(canonical)
    if unit not in ("cell", "day", "movie"):
        raise QCError(f"unit must be cell/replicate/movie, got {unit!r}")
    table = track_summary(canonical, hmm_fit=hmm_fit) if summary is None else summary

    aggregations = {
        "n_tracks": ("uid", "size"),
        "n_localizations": ("n_localizations", "sum"),
        "median_steps": ("n_steps", "median"),
        "median_appd": ("appd_um2_per_s", "median"),
    }
    if "slow_fraction" in table.columns:
        aggregations["slow_fraction"] = ("slow_fraction", "mean")

    out = table.groupby(unit, observed=True).agg(**aggregations).reset_index()
    if out.empty:
        raise QCError(f"no {unit} to summarise")

    # Flag on the most informative readout available. Track length is the last
    # resort: a unit can be perfectly ordinary in length while its mobility is
    # wildly off, which is exactly the bad-day case.
    candidates = [c for c in ("slow_fraction", "median_appd", "median_steps")
                  if c in out.columns]
    target = candidates[0]
    flags = flag_outliers(out[target].to_numpy(), k=k)
    for candidate in candidates[1:]:
        if flags.scale > 0:
            break
        # The chosen readout is degenerate (every unit identical); try the next
        # rather than reporting a silent "all clear".
        target = candidate
        flags = flag_outliers(out[candidate].to_numpy(), k=k)
    out["flagged"] = flags.flagged
    out.attrs.update(
        {
            "unit": unit,
            "flagged_on": target,
            "n_units": int(len(out)),
            "median": flags.median,
            "scale": flags.scale,
            "scale_estimator": flags.scale_estimator,
            "k": k,
            "low": flags.low,
            "high": flags.high,
            "caption": f"{unit}s flagged on {target}: {flags.caption()}",
        }
    )
    return out


#: Media whose autofluorescence differs enough that a control in one does not bound a
#: sample in the other. MEASURED here as in-cell minus local out-of-cell background,
#: matched protein, scope and label: LB runs **1.4-2.2x** minimal in both proteins tested.
#: A WT control in LB beside samples in minimal OVERSTATES contamination, the direction that makes a
#: sample look dirtier than it is.
AUTOFLUORESCENT_MEDIA = {"lb": 1.0, "rich": 1.0, "minimal": 0.5, "mm": 0.5,
                         "s750": 0.5, "none": 0.0}


def control_contamination(canonical, *, control_conditions, by=("condition",),
                          media=None):
    """How much of each sample's track rate the CONTROL could account for.

    Two controls answer two different questions and this handles both the same way,
    because the arithmetic is the same and only the interpretation differs:

    * a **no-cell / agar-pad** control bounds what the DETECTOR invents on an empty
      field: measured here at 0.014 spots per frame, **0.6 %** of a sample, i.e. an
      empty field is empty at the shipped threshold;
    * an **unlabelled WT** control bounds what the CELLS contribute by
      autofluorescence, which is the larger term and the one that moves with medium.

    The unit is tracks per cell per minute, and three units disagree here. Measured
    on the WT against a matched-medium labelled sample, the three readings differ widely:

        spots per FRAME              per FIELD, so it moves with cell density
        tracks per cell-minute       (by mean) / (by median)
        localisations per cell-min   length-independent

    The per-frame figure understates it because the WT fields hold fewer cells than the
    samples' (about half): scaling by the cell ratio lands on the per-cell number. Track
    LENGTH explains only about 1.1x of it, so the unit
    that matters is per CELL-time, not per field and not per track.

    The rate is per cell per minute, never per cell. A control recorded over 2001 frames
    against samples recorded over 1001 is watched twice as long, and dividing by cells
    alone reported the WT at about twice its honest share of the labelled sample, see
    `detection_density`, which also records that pairing the right time with the wrong
    number of cells once REVERSED the ordering entirely.

    ``media`` maps condition name to medium (free text; matched case-insensitively
    against `AUTOFLUORESCENT_MEDIA`). Where a sample and its control differ, the row is
    flagged and the ratio is labelled an OVER-estimate rather than silently compared:
    LB is 1.4-2.2x minimal here, and the WT this project owns is in LB while the samples
    are not. A caveat that is computed is one the reader cannot skip.

    Returns one row per sample condition with ``tracks_per_cell_per_minute``, the
    control's rate, ``contamination`` as their ratio, and ``caveat``.
    """
    control_conditions = [str(c) for c in control_conditions]
    if not control_conditions:
        raise QCError("control_conditions is empty: name at least one control")
    density = detection_density(canonical, by=by)
    if "condition" not in density.columns:
        raise QCError("detection_density returned no 'condition' column; "
                      "pass by=('condition', ...)")
    # THE MEDIAN, NOT THE MEAN, and a 7-cell control is why.
    #
    # MEASURED on one project's own WT: 7 cells, one of which holds 44 per cent of all 158
    # tracks, so the mean sits far above the MEDIAN and the contamination share reads about
    # twice as high by the mean as by the median. A control is exactly the
    # place with few units, and this project already uses median +- MAD for cell
    # outliers for the same reason.
    grouped = density.groupby("condition")["tracks_per_cell_per_minute"]
    rates = grouped.median().to_dict()
    means = grouped.mean().to_dict()
    spans = {k: (float(v.min()), float(v.max())) for k, v in grouped}
    counts = grouped.size().to_dict()
    missing = [c for c in control_conditions if c not in rates]
    if missing:
        raise QCError(
            f"control condition(s) not in this project: {', '.join(missing)}. "
            f"Loaded: {', '.join(sorted(rates))}. A control has to be LOADED to be "
            f"compared: it is a folder like any other condition.")

    control_rate = sum(rates[c] for c in control_conditions) / len(control_conditions)
    control_mean = sum(means[c] for c in control_conditions) / len(control_conditions)
    control_units = sum(counts[c] for c in control_conditions)
    media = {str(k).strip().lower(): str(v).strip().lower()
             for k, v in (media or {}).items()}

    def medium_of(name):
        return media.get(str(name).strip().lower())

    control_medium = {medium_of(c) for c in control_conditions} - {None}
    rows = []
    for name, rate in sorted(rates.items()):
        if name in control_conditions:
            continue
        caveat = ""
        sample_medium = medium_of(name)
        if control_medium and sample_medium and sample_medium not in control_medium:
            other = sorted(control_medium)[0]
            if (AUTOFLUORESCENT_MEDIA.get(other, 0.5)
                    > AUTOFLUORESCENT_MEDIA.get(sample_medium, 0.5)):
                caveat = (f"control is in {other} and this sample in {sample_medium}; "
                          f"LB runs 1.4 to 2.2 times minimal here, so this OVERSTATES the "
                          f"contamination")
            else:
                caveat = (f"control is in {other} and this sample in {sample_medium}; "
                          f"an unmatched medium does not bound this sample")
        elif not control_medium:
            caveat = ("media not given, so a medium mismatch cannot be excluded: the "
                      "WT this project owns is in LB while the MSD samples are not")
        # A CONTROL ON FEW UNITS CANNOT PIN A SHARE, and saying so beats a number.
        # Where the median and the mean disagree by more than half, the spread IS the
        # answer: on the WT above they give 38 and 80 per cent of the same sample.
        spread_caveat = ""
        if control_units < 10:
            spread_caveat = (f"the control rests on {control_units} unit(s); "
                             f"median {control_rate:.1f} against mean "
                             f"{control_mean:.1f} per cell per minute")
        elif control_mean > 1.5 * max(control_rate, 1e-9):
            spread_caveat = (f"the control's mean ({control_mean:.1f}) far exceeds its "
                             f"median ({control_rate:.1f}): a few units dominate it")
        if spread_caveat:
            caveat = (caveat + "; " + spread_caveat) if caveat else spread_caveat
        rows.append({
            "condition": name,
            "tracks_per_cell_per_minute": rate,
            "control_per_cell_per_minute": control_rate,
            "control_per_cell_per_minute_MEAN": control_mean,
            "control_units": control_units,
            "contamination": (control_rate / rate) if rate else float("nan"),
            "contamination_by_mean": (control_mean / rate) if rate else float("nan"),
            "control": ", ".join(control_conditions),
            "caveat": caveat,
        })
    if not rows:
        raise QCError(
            "every loaded condition was named as a control, so there is no sample to "
            "compare against")
    out = pd.DataFrame(rows)
    out.attrs["control_conditions"] = list(control_conditions)
    return out


def detection_density(canonical: pd.DataFrame, *,
                      by=("condition", "day", "movie")) -> pd.DataFrame:
    """Tracks per cell per MINUTE: the sample size on a common footing.

    Why not tracks per cell. That divides by the cells and not by how long each was
    watched, so it compares a movie recorded for 2001 frames with one recorded for 1001
    as though they were the same measurement. Measured on one project's own controls:
    the unlabelled WT's tracks per cell read as about half the labelled sample's, but the
    WT movies are 2001 frames and the sample's 1001, and per unit time the share is
    about half of that. The unnormalised figure overstated the contamination twofold.

    Why minutes and not frames. The frame interval is not a constant across this
    project: 20, 20.16 and 20.472 ms appear in the u-track data and 50 ms in the protein I
    acquisitions, so a per-frame rate is not comparable between datasets and a
    per-second one is. Each row's own ``tau_s`` is used, so a project with mixed rates
    stays correct rather than being given one movie's interval.

    The denominator is CELL-TIME, and getting that wrong reversed the answer. A first
    version divided by ``n_cells x (total time of every movie in the group)``. But a
    cell is watched during ITS OWN movie only, so that inflates the exposure by the
    number of movies and the error is not a constant: it is 1/n_movies, and n_movies
    differs per group. Measured: the unlabelled WT (3 movies) came out at several times the labelled
    sample's rate (30 movies), i.e. the control looked dirtier than the labelled sample.
    Per cell-time the same tables show the control at a THIRD of the sample, the
    opposite ordering, and the one that agrees with the per-1000-frame figures already
    recorded in ``docs/adaptive_threshold.md``. Same shape as
    the pooled-span defect above, one level deeper: that one summed the time correctly
    and then paired it with the wrong number of cells.

    Both units are returned. Per second reads naturally for the labelled samples here
    (0.5 to 1.6) and per minute for the sparse ones (0.4 upward); neither is a
    different measurement.

    The span is the observed one, so a movie is credited from its first detected frame
    to its last rather than from its true first to last frame. Measured: 1980 of 2001
    frames on the WT and 960 of 1001 on a labelled sample, i.e. 96-99 per cent, and it runs
    in the direction that slightly INFLATES a dense group. The canonical table does not
    carry the movie length, so this is stated rather than corrected.

    ON SPARSE DATA BOTH SHORTCUTS FAIL, and `sample_size` is the one to use there: it
    counts the OUTLINED cells (a cell that yielded nothing included) over each movie's
    TRACKED frames, both supplied by the caller. Measured on one user's d2.1: 9 of 52 cells
    yielded no track, and the observed spans covered 74 per cent of the tracked frames
    (one movie 11 of 1001).

    This is a density, not a concentration. It counts tracks the detector produced and
    the filters kept, so it moves with the threshold, the labelling efficiency and the
    autofluorescence: all of which is why it is worth looking at beside a fraction that
    is meant to be immune to them.
    """
    _require_canonical(canonical)
    if "tau_s" not in canonical.columns:
        raise QCError(
            "no 'tau_s' column, so the observation time cannot be computed; a density "
            "per frame would not be comparable between datasets acquired at different "
            "frame intervals")

    keys = [k for k in by if k in canonical.columns]
    if not keys:
        raise QCError(f"none of {by} is a column of the canonical table")

    rows = []
    for key, frame in canonical.groupby(keys, observed=True):
        tau = float(frame["tau_s"].median())
        # PER MOVIE, THEN SUMMED. The frames spanned are a property of a movie, and
        # pooling thirty of them and taking max - min gives the span of the UNION rather
        # than the time actually watched: on a 30-movie condition of 1001-frame movies
        # that read 1500 frames instead of 30 030, and the density came out about a third
        # of the truth. Each movie also carries its own tau, so the sum is over
        # frames x tau rather than over frames alone.
        if "movie" in frame.columns:
            per_movie = frame.groupby("movie", observed=True).agg(
                lo=("frame", "min"), hi=("frame", "max"), tau=("tau_s", "median"),
                cells=("cell", "nunique"))
            watched = (per_movie["hi"] - per_movie["lo"] + 1) * per_movie["tau"]
            # CELL-TIME, and this is what the denominator has to be. A cell is watched
            # for ITS OWN movie and for nothing else, so a condition's exposure is the
            # sum over movies of (that movie's duration x its cells) - NOT the total
            # duration of every movie multiplied by the total number of cells.
            cell_seconds = float((watched * per_movie["cells"]).sum())
            seconds = float(watched.sum())
            span = float((per_movie["hi"] - per_movie["lo"] + 1).sum())
            cells = int(per_movie["cells"].sum())
        else:
            span = float(frame["frame"].max() - frame["frame"].min() + 1)
            seconds = span * tau
            cells = int(frame["cell"].nunique())
            cell_seconds = seconds * cells
        minutes = seconds / 60.0
        tracks = int(frame["uid"].nunique())
        rows.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
            "n_tracks": tracks, "n_cells": cells,
            "frames_spanned": int(span), "tau_s": tau,
            "minutes_observed": minutes,
            "cell_minutes_observed": cell_seconds / 60.0,
            "tracks_per_cell": tracks / cells if cells else float("nan"),
            "tracks_per_cell_per_second": (tracks / cell_seconds
                                           if cell_seconds else float("nan")),
            "tracks_per_cell_per_minute": (tracks / (cell_seconds / 60.0)
                                           if cell_seconds else float("nan")),
        })
    out = pd.DataFrame(rows)
    out.attrs["by"] = tuple(keys)
    out.attrs["note"] = ("tracks per cell per minute; this density depends on the "
                         "detection threshold and the labelling and is not a concentration")
    return out


#: What one row of `sample_size`'s input carries: ONE OUTLINED CELL.
CELL_INVENTORY_COLUMNS = ("condition", "day", "movie", "cell", "n_tracks",
                          "n_localizations", "frames", "tau_s")


def sample_size(cells: pd.DataFrame, *, by=("condition", "day")) -> pd.DataFrame:
    """Cells, tracks and tracks per cell per unit TIME, from ONE ROW PER OUTLINED CELL.

    Asked for: "a list of how many cells, replicates, tracks per cell (per 1000 frames
    or something)". The inventory of a dataset, per condition and per replicate day.

    WHY THE INPUT IS THE OUTLINES AND NOT THE TRACKS. `detection_density` counts the
    cells the canonical table holds, i.e. cells with at least one surviving track, and
    the movie length as the span of those tracks. Both are right on dense data and both
    flatter sparse data, in the same direction: a cell that yielded nothing is not in
    the table, so it leaves the denominator, and a movie with one track is credited the
    few frames that track spans. MEASURED on one user's d2 (MetaMorph, 1001 frames tracked
    per movie): d2.1 has **52 outlined cells of which 9 yielded no track**, and its
    movies' track spans sum to 7 391 of the 10 010 frames tracked (series001: 11 of
    1001). Read the canonical way that is 3.5 tracks per cell over 74 per cent of the
    time; counted here it is 2.9 per cell, and per unit time lower again. So the cells
    are the OUTLINED cells (zero where a cell yielded nothing) and each is watched for
    its own movie's TRACKED frames (`project.movie_frames`, read from the tracker's own
    record) -- both supplied by the caller, since neither survives into the canonical
    table.

    Per 1000 frames AND per minute: frames compare movies of one acquisition and read
    the way a movie is cropped; minutes compare acquisitions at different frame
    intervals (20 against 50 ms here), which a per-frame rate cannot. The rate is
    POOLED -- all tracks over all cell-time -- so a cell counts by how long it was
    watched; the median per cell sits beside it because a few busy cells can carry a
    pooled rate (the WT control's top cell held 44 per cent of its tracks), and
    ``largest_cell_share`` names that directly.

    A density of what the detector found and the filters kept, not a concentration:
    it moves with the threshold, the labelling and the autofluorescence.
    """
    missing = [c for c in CELL_INVENTORY_COLUMNS if c not in cells.columns]
    if missing:
        raise QCError(f"a cell inventory needs the columns {missing}")
    if cells.empty:
        raise QCError("no outlined cells to count")
    keys = [k for k in by if k in cells.columns]
    if not keys:
        raise QCError(f"none of {by} is a column of the cell inventory")
    rows = []
    for key, frame in cells.groupby(keys, observed=True, dropna=False, sort=True):
        tracks = frame["n_tracks"].to_numpy(dtype=float)
        frames = frame["frames"].to_numpy(dtype=float)
        seconds = frames * frame["tau_s"].to_numpy(dtype=float)
        total = float(tracks.sum())
        cell_frames = float(frames.sum())
        cell_seconds = float(seconds.sum())
        per_movie = frame.groupby("movie", observed=True)["frames"].first()
        rows.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
            "n_movies": int(frame["movie"].nunique()),
            "n_cells": int(len(frame)),
            "n_empty_cells": int((tracks == 0).sum()),
            "n_tracks": int(total),
            "n_localizations": int(frame["n_localizations"].sum()),
            "tracks_per_cell_median": float(np.median(tracks)),
            "tracks_per_cell_mean": total / len(frame),
            "largest_cell_share": (float(tracks.max()) / total) if total else float("nan"),
            "frames_min": int(per_movie.min()),
            "frames_max": int(per_movie.max()),
            "cell_frames": cell_frames,
            "cell_minutes": cell_seconds / 60.0,
            "tracks_per_cell_per_1000_frames": (1000.0 * total / cell_frames
                                                if cell_frames else float("nan")),
            "tracks_per_cell_per_minute": (total / (cell_seconds / 60.0)
                                           if cell_seconds else float("nan")),
        })
    out = pd.DataFrame(rows)
    out.attrs["by"] = tuple(keys)
    out.attrs["note"] = ("outlined cells, each counted over its own movie's tracked "
                         "frames; a density, not a concentration")
    return out


#: The yield `flag_low_yield` judges: tracks per outlined cell per 1000 tracked frames.
YIELD_COLUMN = "tracks_per_cell_per_1000_frames"
#: Fewer replicate days than this and a day's yield is not judged against the others.
YIELD_MIN_DAYS = 3


def flag_low_yield(days: pd.DataFrame, *, k: float = DEFAULT_MAD_K) -> pd.DataFrame:
    """One condition's replicate days (`sample_size(cells, by=("condition", "day"))`) with a
    ``low_yield`` column: a day whose yield -- tracks per outlined cell per 1000 tracked
    frames (`YIELD_COLUMN`) -- lies BELOW its condition's median day by more than k x the
    robust scale. The SAME rule as every other QC flag (`flag_outliers`, median +- k.MAD,
    the meanAD fallback when the MAD is 0), on the LOW side only: a day with more tracks per
    cell is not a quality problem.

    Ruled by the experimenter (13 Oct 2026: "Same MAD rule as the other QC flags"), with
    its limit said out loud: with 3-4 days a rule centred on the median cannot flag HALF
    the days. a lab condition has two days at ~5 tracks per cell and two at
    ~14 -- two of four are the data, not outliers, and nothing is flagged; one bad day
    among good ones is what this catches. Under `YIELD_MIN_DAYS` days nothing is judged.

    Flags, never drops. ``attrs``: ``median``, ``low`` (the threshold), ``k``,
    ``scale_estimator``, ``judged`` (False under the minimum) and ``note``."""
    if YIELD_COLUMN not in days.columns:
        raise QCError(f"a replicate table needs {YIELD_COLUMN!r} (qc.sample_size by replicate)")
    out = days.copy()
    values = out[YIELD_COLUMN].to_numpy(dtype=float)
    out["low_yield"] = False
    finite = int(np.isfinite(values).sum())
    if finite < YIELD_MIN_DAYS:
        out.attrs.update({"judged": False, "k": k, "note": (
            f"{finite} replicate{'s' if finite != 1 else ''} with a yield: under "
            f"{YIELD_MIN_DAYS}, so no replicate is judged against the others")})
        return out
    flags = flag_outliers(values, k=k)
    low = flags.flagged & (values < flags.median)
    out["low_yield"] = low
    out.attrs.update({
        "judged": True, "median": flags.median, "low": flags.low, "k": k,
        "scale_estimator": flags.scale_estimator,
        "note": (f"a replicate is low when its tracks per cell per 1000 frames is below the "
                 f"condition's median replicate ({flags.median:.1f}) by more than {k:g} x "
                 f"{flags.scale_estimator} (under {flags.low:.1f})")})
    return out


#: A movie is NAMED when its in-cell share falls this far below its condition's median,
#: in percentage points.
#:
#: MEASURED ON THE REFERENCE GROUP rather than chosen: its 29 movies keep a
#: median **87.5 per cent** of their tracks, range 0 to 96.6, and two stand out,
#: 33 and 22 points below. 15 points sits
#: below both and above the ordinary scatter (the third-worst is 12 points
#: down). It is a POINT DIFFERENCE and not a ratio because the quantity is already a
#: share: 90 against 80 per cent and 9 against 8 are the same ratio and not the same
#: event.
IN_CELL_FLAG_DROP = 15.0


def in_cell_share(canonical: pd.DataFrame) -> pd.DataFrame:
    """Per MOVIE: tracks found, tracks inside a cell outline, and the share.

    THE ONE QC LINE THAT SHOWS A MOVIE CONTRIBUTING NOTHING. The cell filter is the
    largest silent loss in this pipeline -- it keeps only tracks lying entirely inside an
    outline -- and the canonical table is POST-filter, so a movie it emptied has no rows
    at all and is invisible in every panel drawn from that table. Measured on the
    reference group the moment this existed: **`series009` had 376 tracks found and 0
    kept**, because its outlines were deleted, and nothing anywhere said so. That case is
    already recorded in this project's notes as a silent nothing-contribution; this is
    what makes it visible without reading the notes.

    THE COUNTS COME FROM WHERE THE FILTER RAN, never from arithmetic on this table.
    `cellmesh.assign_cells` records them and `canonical.to_canonical` hands them over
    explicitly; the dropped tracks have no rows here, so nothing downstream could
    reconstruct them. RAISES where they are absent rather than reporting a clean
    pipeline -- a table loaded through a path that does not carry them is UNMEASURED, and
    "100 per cent kept" is the most flattering possible way to say "I do not know".

    IT IS A LOOK-AT-THIS, NOT A VERDICT. A low share has three quite different causes
    that want three different responses, and this cannot tell them apart: the movie was
    under-segmented (fix the segmentation), the label genuinely sits outside cells (a
    result, not a defect), or the outlines and the tracks disagree about coordinates (a
    bug). `cellmesh.assignment_funnel(meshes=...)` classifies the discards by distance to
    the nearest outline, which is what separates them.
    """
    _require_canonical(canonical)
    by_movie = canonical.attrs.get("cell_filter_by_movie")
    if not by_movie:
        raise QCError(
            "this table carries no per-movie cell-filter counts, so the in-cell share "
            "cannot be read from it (an older cache entry or a hand-built frame lacks them). "
            "Re-load the project with use_canonical_cache=False.")

    condition_of = {}
    if "condition" in canonical.columns:
        condition_of = (canonical.groupby("movie", observed=True)["condition"]
                        .first().to_dict())

    near_edge = canonical.attrs.get("cell_filter_near_edge_by_movie") or {}
    rows = []
    for movie, counts in by_movie.items():
        found = int(counts.get("found", 0))
        kept = int(counts.get("kept", 0))
        rows.append({
            "condition": condition_of.get(movie, ""),
            "movie": movie,
            "tracks_found": found,
            "tracks_in_cells": kept,
            "tracks_outside": found - kept,
            "in_cell_share": (kept / found) if found else float("nan"),
            # kept only because the outline was widened (`cellmesh.assign_cells`,
            # tolerance_px); 0 under the legacy filter, which records no such count
            "by_tolerance": int(counts.get("by_tolerance", 0)),
            # dropped, but every point within NEAR_EDGE_PX of an outline (`assign_cells`)
            "near_edge": int(near_edge.get(str(movie), 0)),
        })
    out = pd.DataFrame(rows).sort_values(["condition", "movie"]).reset_index(drop=True)

    # THE REFERENCE IS THE CONDITION, NOT THE PROJECT. Two conditions can lose very
    # different fractions -- that is the whole reason this loss is worth printing -- so
    # comparing a movie against a project-wide median would flag the whole of the
    # harder-hit condition and none of its actual outliers.
    med = out.groupby("condition")["in_cell_share"].transform("median")
    out["condition_median"] = med
    out["drop_pp"] = 100.0 * (med - out["in_cell_share"])
    out["flagged"] = (out["drop_pp"] >= IN_CELL_FLAG_DROP) | (out["tracks_in_cells"] == 0)
    out["reason"] = [
        ("no track lies inside an outline, so this movie contributes nothing to any analysis, and "
         "its other panels cannot show that because it has no tracks left in them"
         if row.tracks_in_cells == 0 else
         (f"{row.drop_pp:.0f} points below this condition's median "
          f"({100 * row.in_cell_share:.0f} % against "
          f"{100 * row.condition_median:.0f} %)" if row.flagged else ""))
        for row in out.itertuples()
    ]

    kept_total = int(out["tracks_in_cells"].sum())
    found_total = int(out["tracks_found"].sum())
    out.attrs["n_flagged"] = int(out["flagged"].sum())
    out.attrs["tracks_found"] = found_total
    out.attrs["tracks_in_cells"] = kept_total
    out.attrs["share"] = (kept_total / found_total) if found_total else float("nan")
    out.attrs["note"] = (
        f"{kept_total} of {found_total} tracks lie entirely inside a cell outline "
        f"({100 * out.attrs['share']:.0f} %). A low share is a prompt to look, not a "
        f"verdict: outlines drawn too small, a label outside the cells and a coordinate "
        f"mismatch all look the same here. The distance of the dropped tracks from the "
        f"nearest outline tells them apart.")
    # THE COST OF A TIGHT OUTLINE, stated as a count with its definition, never as a verdict:
    # SynR's segmented outlines (11 % small) left 28 % of found tracks just outside, the reference
    # project about a tenth as many, and only the first moved the fractions by points.
    out.attrs["tracks_near_edge"] = int(out["near_edge"].sum())
    if "cell_filter_near_edge_by_movie" in canonical.attrs and found_total:
        px = float(canonical.attrs.get("cell_filter_near_edge_px", 1.0))
        out.attrs["note"] += (
            f" {out.attrs['tracks_near_edge']} dropped tracks "
            f"({100 * out.attrs['tracks_near_edge'] / found_total:.0f} % of those found) lie "
            f"within {px:g} px of an outline: outlines drawn tight drop free molecules near the "
            f"membrane and raise the slow fraction. The Outline tolerance option (Analysis "
            f"options) keeps them; compare the fractions with and without it.")
    # A WIDENED FILTER SAYS SO, with what it added: the share above is not the legacy one.
    tolerance_nm = float(canonical.attrs.get("outline_tolerance_nm", 0.0) or 0.0)
    out.attrs["outline_tolerance_nm"] = tolerance_nm
    out.attrs["tracks_by_tolerance"] = int(out["by_tolerance"].sum())
    if tolerance_nm > 0:
        out.attrs["note"] = (
            f"Outlines widened by {tolerance_nm:g} nm (the Outline tolerance option): "
            f"{out.attrs['tracks_by_tolerance']} of these tracks are kept only because of "
            f"it. " + out.attrs["note"].replace("lie entirely inside a cell outline",
                                                 "lie inside a widened cell outline"))
    return out
