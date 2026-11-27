"""Run a catalogue analysis on EVERY selected condition, side by side.

Why this is separate from ``comparison.py``. That module holds the six method views the
browser already ships and they work; this adds the other seventeen tools without editing
them. It deliberately returns the SAME shape those views return --
``{"per_condition": {name: {...}}}`` -- so the tabs plug into the existing
``_ComparisonTab``, caching and drawing rather than becoming a parallel system. A parallel
system beside the one in use is the defect this whole file exists to avoid repeating.

Side by side is the default, not an option. Opening four conditions and computing one
answers a question nobody asked. A single condition is the case where a single condition
was loaded, which the folder layout already distinguishes: a folder holding movies is one
condition, a folder holding condition folders is a comparison.

What this does not do. It draws conditions beside each other with each condition's own
value and its own spread. It computes NO between-condition difference, p-value or delta
endpoint -- that is the line Amendment 4 drew, and which statistic to use for "are these
two different" is still open. The layer report below is what a reader needs to judge the
comparison by eye, and it is not a test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analyses import ProjectScope, AnalysisError, build_input, by_key

__all__ = [
    "CatalogueViewError",
    "catalogue_view",
    "layer_summary",
    "density_summary",
    "DEFAULT_UNIT",
]


class CatalogueViewError(RuntimeError):
    """Raised when a catalogue view cannot be assembled honestly."""


#: The independent unit, and why it is the day. The experimenter chooses the replicate
#: day at load time precisely so this is available; cells within a day share a culture.
DEFAULT_UNIT = "day"

#: Multistart count for a DIAGNOSTIC fit -- one that is never quoted. See the measurement
#: cited at its use site; the reported-number default stays `sqd._DEFAULT_STARTS`.
DIAGNOSTIC_STARTS = 4


def catalogue_view(project, conditions, *, analysis_key: str, params=None):
    """One catalogue analysis over every condition.

    Returns ``{"per_condition": {name: {...}}, "analysis": key, "failed": {...}}``.

    A condition that raises is kept and named, not dropped. Showing three columns where
    four were asked for reads as "the fourth has no data", which is a different statement
    from "the fourth failed to fit".
    """
    analysis = by_key(analysis_key)
    values = dict(params or {})
    missing = [p.name for p in analysis.required_parameters if values.get(p.name) is None]
    if missing:
        raise CatalogueViewError(
            f"{analysis_key}: these have no default and must be chosen: "
            f"{', '.join(missing)}")

    per_condition, failed = {}, {}
    for condition in conditions:
        try:
            # An analysis wanting cell coordinates gets the PROJECTED table. Without
            # this the entry resolves, passes every signature check, and then raises for
            # every condition - a button that raises, which is the exact failure
            # INPUT_FIRST_ARG exists to catch one level up.
            table = (project.projected(condition)
                     if analysis.input_kind == "cell_frame"
                     else project.condition(condition))
            # THE PROJECT TRAVELS TOO, for the one kind that is not derived from the
            # table. Every other analysis is a function of the canonical table alone;
            # `spot_intensity` and `detection` read the raw acquisition and the tracker
            # output, which the table cannot carry -- by the time it exists, the movies
            # are a path nobody kept.
            # `getattr`, NOT `project.roots`. This runner takes anything that quacks
            # like a project -- the tests pass a stand-in with only `condition()` --
            # and reaching for an attribute that only the real class has broke EVERY
            # analysis, not just the two that need a root. A scope with no root is a
            # legitimate state; `project_scope` analyses refuse it by name.
            scope = ProjectScope(
                project=project, condition=condition,
                root=(getattr(project, "roots", None) or {}).get(condition))
            data = build_input(analysis.input_kind, table, scope=scope)
            per_condition[condition] = {
                "result": analysis.resolve()(data, **values),
                "input": data,
                "n_tracks": int(table["uid"].nunique()) if "uid" in table else None,
            }
        except Exception as exc:                       # noqa: BLE001 - kept and named
            failed[condition] = f"{type(exc).__name__}: {exc}"

    if not per_condition:
        raise CatalogueViewError(
            f"{analysis_key} failed for every condition: "
            + "; ".join(f"{k}: {v}" for k, v in failed.items()))
    return {"per_condition": per_condition, "analysis": analysis_key,
            "failed": failed, "parameters": values}


def layer_summary(project, conditions, *, n_pop: int = 2, quantity: str = "fraction_1",
                  min_observations: int = 50):
    """``{condition: layer_report}`` -- the day-to-day and cell-to-cell spread.

    Computed once per condition and shown on every tab, because it is a property of the
    EXPERIMENT rather than of whichever analysis is open. On a lab project the
    day-to-day spread was several times smaller than the cell-to-cell spread, so the
    replicate days agree much better than cells within one day do.

    That contrast is what a reader needs to judge a side-by-side comparison without a
    test: "n = 3 days" reads as a weakness until the three days are visibly in agreement.

    A condition whose spread cannot be formed -- one day, too few usable cells -- is
    NAMED with the reason rather than omitted, since a missing line reads as "no
    variation" instead of "not measurable here".
    """
    from .aggregate import fit_by_unit, layer_report
    from .observables import squared_displacements

    out = {}
    for condition in conditions:
        try:
            table = project.condition(condition)
            per_cell = fit_by_unit(
                squared_displacements(table), n_pop=n_pop, unit="cell",
                shared_D=True, min_observations=min_observations,
                # A SPREAD DIAGNOSTIC DOES NOT NEED THE INSURANCE STARTS. This line is
                # one fit per cell -- 84 of them on a real condition -- and it runs on
                # every panel, so at the 50-start default it dominated the wait before
                # any analysis was chosen. MEASURED over all 18 groups: at n_pop = 2 the
                # answer is identical to the 50-start one in 18 of 18, at every count
                # down to a single start (`legacy_out/tables/sqd_starts_sweep.csv`), so
                # 4 carries margin over what the data required. This is NOT licence to
                # lower it for a REPORTED number, and it is n_pop-specific: the same
                # sweep shows n_pop = 3 still disagreeing at 8 starts in 4 of 18.
                n_starts=DIAGNOSTIC_STARTS,
                # MOVIE MUST BE CARRIED, or the layer report folds the largest layer
                # into "cell": movie-to-movie is a large share of the variance here,
                # more than the day in every condition.
                group_columns=("condition", "day", "movie"))
            out[condition] = layer_report(per_cell, quantity=quantity)
        except Exception as exc:                       # noqa: BLE001
            out[condition] = {
                "error": f"{type(exc).__name__}: {exc}",
                "caption": (f"spread not measurable for {condition}: "
                            f"{type(exc).__name__}"),
            }
    return out


def density_summary(project, conditions):
    """``{condition: report}`` -- how many tracks each cell yielded per unit TIME.

    Shown beside `layer_summary` and for the same reason: it is a property of the
    EXPERIMENT, not of whichever analysis is open, so it is computed once per condition
    and printed above every group rather than becoming a panel. It is also the number
    that says whether two conditions drawn side by side are even comparable in sample
    size, which no fraction can show.

    Why per unit time and not per cell. Tracks per cell divides by the cells and not by
    how long each was watched, so a 2001-frame movie and a 1001-frame one are compared
    as though they were the same measurement. And the denominator is CELL-TIME -- the
    sum over movies of (that movie's duration x its cells) -- not the total duration of
    every movie times the total cell count, which overstates the exposure by the movie
    count and, because the movie count differs per group, REORDERS them. See
    `qc.detection_density`, which carries the measurement.

    This is a density, not a concentration. It moves with the detection threshold, the
    labelling efficiency and the autofluorescence, so it belongs beside a fraction that
    is meant to be immune to all three -- as a check on the fraction, never as a claim.

    A condition whose density cannot be formed is NAMED with the reason, exactly as
    above: a missing line reads as "nothing detected" rather than "not measurable".
    """
    from .qc import detection_density

    out = {}
    for condition in conditions:
        try:
            table = project.condition(condition)
            row = detection_density(table, by=("condition",)).iloc[0]
        except Exception as exc:                       # noqa: BLE001
            out[condition] = {
                "error": f"{type(exc).__name__}: {exc}",
                "caption": (f"density not measurable for {condition}: "
                            f"{type(exc).__name__}"),
            }
            continue
        out[condition] = {
            "n_tracks": int(row["n_tracks"]),
            "n_cells": int(row["n_cells"]),
            "cell_minutes_observed": float(row["cell_minutes_observed"]),
            "tracks_per_cell": float(row["tracks_per_cell"]),
            "tracks_per_cell_per_minute": float(row["tracks_per_cell_per_minute"]),
            "tracks_per_cell_per_second": float(row["tracks_per_cell_per_second"]),
            "caption": (
                f"{int(row['n_tracks'])} tracks from {int(row['n_cells'])} cells, "
                f"{row['tracks_per_cell_per_minute']:.1f} per cell per minute "
                f"({row['tracks_per_cell_per_second']:.2f} per second) over "
                f"{row['cell_minutes_observed']:.1f} cell-minutes"),
        }
    return out
