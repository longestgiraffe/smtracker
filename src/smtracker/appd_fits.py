"""THE APPARENT DIFFUSION SECTION'S FITS, as the Run's settings make them -- with no Qt anywhere in it (3 Nov 2026).

Asked 3 Nov 2026: "for appd we need more pop fits next to each other as well? also simultanous fit etc, replicate etc
options. right now appd looks not done", then "start building everything". Until then the section's Run had ONE
Populations setting for every condition, and its Uncertainty did two jobs at once: "Replicates" also meant "D shared
across the replicates" and "None" also meant "pooled". Now the Run is SQD's, in SQD's own words (`workspace_view`):

* MODELS, per condition (`SECTION_MODELS`: Best (BIC), 1, 2, 3, 4), several at once drawn SIDE BY SIDE in the
  Populations bubbles -- `bubble_view.side_by_side` itself, so each condition is a group of columns, a shade of its
  colour per model, Best drawn once where it chose a count also picked, the columns in order of count. Best stops at 3
  unless 4 is picked (`best_max_pop`, SQD's rule; a ticked 4 is fitted whatever Best's range).
* FORCE SAME D ACROSS (`workspace_view.SHARED_D_NAME`): **Replicates** -- one D over the condition's own replicates,
  fractions per replicate (`appd_gamma.fit_appd_gamma_simultaneous`, the section's fit since 16 Oct 2026); **Don't
  force anywhere** -- each replicate fitted alone at the condition's count, its own D and fractions, so D has a spread
  as well; **Conditions** (a folder design's factor) and **Whole data set** -- one D for every condition of the group
  or of the set: the joint fit over conditions, each replicate's fractions at that D (`appd_gamma.fractions_at`),
  SQD's design (`comparison.condition_replicate_fit`). A row may set Replicates or Don't force anywhere for itself;
  Conditions and Whole data set are the whole Run's. Sharing D across conditions is the STRONGER assumption: on six
  enzymes it grew the condition difference in all six while the fit got worse in all six (`appd_gamma`'s note), which
  the Run dialog's help says.
* UNCERTAINTY (`SECTION_UNCERTAINTY`), SQD's five: Replicates (mean +- SD over replicates), Movies and Cells (a 95 %
  interval from resampling them around the pooled fit, one session only: with replicates the replicate is the unit
  and the interval is withheld, said), Chunks (each condition's movies -- its cells when there are too few movies --
  dealt at random into k pseudo-replicates by SQD's own deal, `comparison.pseudo_replicate_deal`, drawn as
  replicates, with every other of `comparison.PSEUDO_SPLITS` deals kept for how far a p moves; withheld, as Movies and
  Cells are, from a condition with two or more replicates: `comparison.chunks_withheld`, 4 Nov 2026), None (pooled).

A condition fitted on its own is designed inside its pool job (`condition_designs`, from `appd_view.appd_condition`);
the set-wide designs need every condition's tracks, so they are made once the last condition has landed, from the
results' own per-track values (`joint_designs`, one more pool job). `bubble_plot` turns either into the Populations
figure (a `BubblePlotData`, so the chart, brackets, table and export need nothing new) and `day_bubbles` one design's
replicates. Every design is plain data (picklable).

RULE 0 holds throughout: a condition's replicates are its own; a joint fit over conditions shares D, never a pairing.

EXACT-ZERO TRACKS (ruled 4 Nov 2026, `appd_gamma`'s module head): a condition's zero decision (`appd_gamma.zero_tracks`,
made once in `appd_view.appd_condition`) is imposed on every fit of its designs -- its replicates together or alone,
its chunks, cells and resamples -- so each holds the immobile population (D 0, its share counted) where the condition
does. A joint fit decides once on its group's pooled tracks, each condition at its own share. Every design carries the
decision (``zero_jumps``: its note beside the fit warnings), and a joint fit that joins conditions with exact zeros to
conditions without is said (`bubble_view.immobile_mix_warnings`): the latter's slowest population is empty by
construction.
"""
from __future__ import annotations

import numpy as np

from .workspace_view import SHARED_D_ORDER, SHARED_D_PILLS as _PILLS, setting_words as _setting_words

__all__ = [
    "BEST_MAX_POP", "DEFAULTS", "HELP", "MODEL_LABELS", "SECTION_MIN_STEPS", "SECTION_MODELS", "SECTION_UNCERTAINTY",
    "SECTION_UNCERTAINTY_LABELS", "SET_SCOPES", "bubble_plot", "condition_designs", "condition_settings",
    "count_warnings",
    "day_bubbles", "design_for", "fit_counts", "is_set_wide", "joint_designs", "joint_inputs", "models_of",
    "primary_count", "resolve_count", "run_config", "unit_values_of", "PRECISION_MODES", "PRECISION_LABELS",
    "with_precision", "shared_error_warnings",
]

#: The models a condition can be fitted with, in the order their pills stand: Best (BIC) and the counts.
SECTION_MODELS = ("best", "1", "2", "3", "4")
MODEL_LABELS = {"best": "Best (BIC)", "1": "1", "2": "2", "3": "3", "4": "4"}
#: Uncertainty, SQD's five in SQD's order of words.
SECTION_UNCERTAINTY = ("day", "movie", "cell", "chunk", "none")
SECTION_UNCERTAINTY_LABELS = ("Replicates", "Movies", "Cells", "Chunks", "None")
#: The shortest track used, in single-frame steps (u-track's own minimum is 4 steps, 5 localisations).
SECTION_MIN_STEPS = (4, 9, 19)
#: How far Best looks: 3 populations, or 4 when picked (SQD's "Best (BIC) stops at three populations, four is opt-in").
BEST_MAX_POP = (3, 4)
#: The Force same D across values that fit conditions TOGETHER (one D across them): the whole Run's, never a row's.
SET_SCOPES = ("global", "group")
#: What a Run starts from: every condition at Best, D forced across its replicates, mean +- SD over replicates.
DEFAULTS = {"models": ["best"], "shared_d": "condition", "uncertainty": "day", "min_steps": 4, "best_max_pop": 3,
            "chunks": 5, "rows": {}, "precision": "off"}
#: Localisation error (6 Nov 2026), SQD's words: Not modelled (the fit as before), or Per spot -- each track its own
#: noise offset from the tracker's per-spot precision through a calibration measured on immobile spots
#: (`with_precision`, `appd_gamma`'s module head).
PRECISION_MODES = ("off", "per_spot")
PRECISION_LABELS = ("Not modelled", "Per spot")
#: What every Populations caption begins with: what was fitted.
APPD_HEAD = "gamma mixture on apparent D, one value per track over all its steps, each track its own shape"
#: Resamples for the Movies and Cells intervals; the fewest tracks a cell needs to enter the per-cell fractions (one
#: track is a 0 or 100 % guess, `appd_view.cells_payload`'s rule).
BOOT = 200
CELL_MIN_TRACKS = 3


#: The Run dialog's words: per setting (the line under the control, the hover text). The Force same D across and
#: Uncertainty pills' own words are SQD's (`workspace_view`); these say what they mean for the apparent D.
HELP = {
    "models": ("Which fits each condition shows. Tick several to see them side by side.",
               "Best (BIC) gives each condition the count with the lowest BIC, up to 3 populations (4 if Best is "
               "set to look that far). Conditions can end up with different counts, so tick a number to compare "
               "like with like. Two or more ticks draw each condition as a group of columns, one per fit. "
               "n counts tracks."),
    "shared_d": ("Where the populations' D are the same.",
                 f"{_PILLS['condition']} (the default): one D over a condition's replicates, with fractions fitted "
                 f"per replicate. {_PILLS['replicate']}: each replicate is fitted alone, so D has a spread too. "
                 f"Conditions and {_PILLS['global']}: one D for every condition of a group or of the set. This is "
                 "a stronger assumption: it can enlarge the difference between conditions even though the fit gets worse. "
                 "Replicates are not paired across conditions."),
    "uncertainty": ("What the ± is over.",
                    "Replicates: mean ± SD over replicates (a spread, not a confidence interval). Movies and "
                    "Cells: a 95 % interval from resampling them, for a single session only. Chunks: movies (or "
                    "cells) are dealt at random into chunks and treated as replicates; for data imaged without "
                    "replicates, it measures variation within one culture. None: one pooled fit, no interval."),
    "min_steps": ("The shortest track used.",
                  "A track's apparent D is the mean of all its steps. A longer minimum gives each value more "
                  "precision but keeps fewer tracks, and those are biased towards slow molecules (the Checks page "
                  "shows how much). The default, 5 points, is the u-track minimum."),
    "best_max_pop": ("How many populations Best (BIC) may choose.",
                     "Up to 3 (default), as in SQD's Best; 4 lets BIC choose four. A ticked 4 is always fitted."),
    "precision": ("How the fit treats the error in placing each spot.",
                  "Not modelled: a bound molecule placed less precisely reads faster, so part of the bound "
                  "population is fitted as a middle one, and Best (BIC) can read it as two populations of similar "
                  "D. Per spot: each track gets its own localisation error, from the tracker's per-spot precision "
                  "and a calibration measured on immobile spots. A bound population's D then reads near 0."),
}


class AppDRunError(ValueError):
    """The Run's settings cannot make a Run, and the message says why (the dialog shows it, nothing runs)."""


# =====================================================================================================================
# The Run's settings
# =====================================================================================================================

def _models(raw, where: str) -> list:
    models = [str(m) for m in (raw or ())]
    bad = [m for m in models if m not in SECTION_MODELS]
    if bad:
        raise AppDRunError(f"{where}: {bad[0]!r} is not a model; the models are "
                           + ", ".join(MODEL_LABELS[m] for m in SECTION_MODELS))
    return [m for m in SECTION_MODELS if m in set(models)]


def is_set_wide(shared_d) -> bool:
    """Whether a Force same D across value fits conditions together (Conditions, Whole data set)."""
    from .workspace_view import is_set_wide as _set_wide

    return _set_wide(shared_d)


def run_config(values: dict | None = None, *, conditions=(), factors: dict | None = None) -> dict:
    """The Run dialog's values, checked and in one form: ``{"models", "shared_d", "scope", "groups", "factor",
    "uncertainty", "min_steps", "best_max_pop", "chunks", "rows"}``. ``rows`` maps a condition to its own ``models``
    and, under Replicates / Don't force anywhere, its own ``shared_d``. RAISES `AppDRunError` naming what is wrong:
    a model or value that is not one, a row's own Force same D under a set-wide value, rows naming different models
    under a set-wide value (the conditions share one fit, so one set of models), a factor a condition lacks.

    A config kept from before 3 Nov 2026 (``{"states", "uncertainty", "min_steps"}``) reads as what it meant: its
    "states" the one model, its Uncertainty "day" (then also "D across replicates") as Replicates here."""
    from .workspace_view import (CHUNKS_RANGE, ROW_SHARED_D, SHARED_D_NAME, resolve_shared_d, setting_words,
                                 shared_d_pill)

    v = {**DEFAULTS, **dict(values or {})}
    if "states" in (values or {}) and "models" not in (values or {}):
        v["models"] = [str(values["states"])]
    models = _models(v["models"], "Every condition")
    if not models:
        raise AppDRunError("pick at least one model (Best or a count of populations)")
    shared_d = str(v["shared_d"])
    if not (shared_d in ("condition", "replicate", "global") or shared_d.startswith("group:")):
        raise AppDRunError(f"{_setting_words(*SHARED_D_ORDER)}, not {shared_d!r}")
    uncertainty = str(v["uncertainty"])
    if uncertainty not in SECTION_UNCERTAINTY:
        raise AppDRunError(f"uncertainty must be one of {SECTION_UNCERTAINTY}, got {uncertainty!r}")
    if int(v["min_steps"]) not in SECTION_MIN_STEPS:
        raise AppDRunError(f"min_steps must be one of {SECTION_MIN_STEPS}, got {v['min_steps']!r}")
    if int(v["best_max_pop"]) not in BEST_MAX_POP:
        raise AppDRunError(f"Best looks up to {BEST_MAX_POP[0]} or {BEST_MAX_POP[1]} populations, "
                           f"not {v['best_max_pop']!r}")
    lo, hi = CHUNKS_RANGE
    if not lo <= int(v["chunks"]) <= hi:
        raise AppDRunError(f"Chunks deals each condition into {lo} to {hi} chunks, not {v['chunks']!r}")
    if str(v["precision"]) not in PRECISION_MODES:
        raise AppDRunError(f"Localisation error is {' or '.join(PRECISION_LABELS)}, not {v['precision']!r}")
    conditions = [str(c) for c in conditions]
    try:
        scope, groups, factor = resolve_shared_d(shared_d, conditions, factors)
    except ValueError as exc:
        raise AppDRunError(str(exc)) from exc
    rows = {}
    for condition, row in dict(v.get("rows") or {}).items():
        row = dict(row or {})
        own = {}
        if row.get("models"):
            own["models"] = _models(row["models"], str(condition))
        if row.get("shared_d") not in (None, "", shared_d):
            if str(row["shared_d"]) not in ROW_SHARED_D:
                raise AppDRunError(f"{condition}'s own {SHARED_D_NAME} is {shared_d_pill(ROW_SHARED_D[0])} or "
                                   f"{shared_d_pill(ROW_SHARED_D[1])}, not {row['shared_d']!r}")
            if scope in SET_SCOPES:
                raise AppDRunError(f"{setting_words(shared_d)} fits the conditions together, so no condition has its "
                                   f"own; {condition} asked for {shared_d_pill(row['shared_d'])}. Pick "
                                   f"{shared_d_pill(ROW_SHARED_D[0])} or {shared_d_pill(ROW_SHARED_D[1])} to set it "
                                   f"per condition.")
            own["shared_d"] = str(row["shared_d"])
        if own:
            rows[str(condition)] = own
    if scope in SET_SCOPES:
        differ = sorted(c for c, row in rows.items() if row.get("models") and row["models"] != models)
        if differ:
            raise AppDRunError(f"{setting_words(shared_d)} fits the conditions together, with one set of models; "
                               f"{', '.join(differ)} asked for others. Pick the models on the Every condition line.")
        rows = {c: {k: val for k, val in row.items() if k != "models"} for c, row in rows.items()}
        rows = {c: row for c, row in rows.items() if row}
    return {"models": models, "shared_d": shared_d, "scope": scope, "groups": groups, "factor": factor,
            "uncertainty": uncertainty, "min_steps": int(v["min_steps"]), "best_max_pop": int(v["best_max_pop"]),
            "chunks": int(v["chunks"]), "rows": rows, "precision": str(v["precision"])}


def with_precision(config: dict, project, conditions=None) -> dict:
    """``config`` with the Run's per-spot calibration chosen ONCE for the conditions (``"calibration"``: ``(c_nm, p,
    source)``, `workspace_view.run_precision`, as the SQD and HMM Runs choose theirs), so every condition models the
    noise alike and the calibration is in each condition's cache key; ``config`` as it is under Not modelled. RAISES
    `AppDRunError` saying why there is none (a track table, two files, two trackers)."""
    if config.get("precision", "off") != "per_spot":
        return config
    from .workspace_view import RunSettingsError, run_precision

    try:
        precision = run_precision(project, list(conditions if conditions is not None else project.conditions))
    except RunSettingsError as exc:
        raise AppDRunError(str(exc)) from exc
    return {**config, "calibration": (float(precision[0]), float(precision[1]), str(precision[2]))}


def models_of(config: dict, condition: str) -> list:
    """The models ``condition`` is fitted with and shows: its row's, else the Every condition line's."""
    row = (config.get("rows") or {}).get(str(condition)) or {}
    return list(row.get("models") or config["models"])


def condition_settings(config: dict, condition: str) -> dict:
    """What ``condition``'s pool job is given (`appd_view.appd_condition`'s keywords, and the job's cache key): its
    models, its own Force same D (a row's, else the Run's; a set-wide value as it is), Uncertainty, the shortest
    track, Best's range and the chunks. Only what moves THIS condition's numbers, so another row's change does not
    make it again."""
    row = (config.get("rows") or {}).get(str(condition)) or {}
    scope = config["scope"] if config["scope"] in SET_SCOPES else str(row.get("shared_d") or config["scope"])
    out = {"models": tuple(models_of(config, condition)), "scope": scope, "uncertainty": config["uncertainty"],
           "min_steps": int(config["min_steps"]), "best_max_pop": int(config["best_max_pop"]),
           "chunks": int(config["chunks"])}
    if config.get("precision") == "per_spot":
        # PER SPOT: the Run's one calibration (`with_precision`), in the job's keywords and so in its cache key
        if config.get("calibration") is None:
            raise AppDRunError("Localisation error Per spot needs the Run's calibration")
        out["precision"] = tuple(config["calibration"])
    return out


def fit_counts(models, best_max_pop: int) -> tuple:
    """The population counts a condition is fitted at (pooled; the Distributions page compares them): 1 to Best's
    range, and to 4 when a 4 is ticked."""
    top = max([int(best_max_pop)] + [int(m) for m in models if str(m).isdigit()])
    return tuple(range(1, top + 1))


def resolve_count(model, selection: dict) -> int:
    """A model's count: Best's is the count BIC recommends (``selection``), a number is itself."""
    return int(selection["recommended"]) if str(model) == "best" else int(model)


def primary_count(models, selection: dict) -> int:
    """The ONE count the condition's other pages read (the histogram's cut, each track's posterior: the Per cell and
    Checks pages, the gallery): Best's when Best is ticked, else the smallest ticked count of two or more (a slow
    share needs two populations), else 1. The model is `workspace_view.primary_model`, the one rule."""
    from .workspace_view import primary_model

    model = primary_model(models)
    if model is None or not str(model).isdigit():
        return int(selection["recommended"])
    return int(model)


# =====================================================================================================================
# A condition fitted on its own: Force same D across Replicates or Don't force anywhere
# =====================================================================================================================

def _label(value):
    """A replicate or chunk as a plain label: an integer ordinal when it is one (a day is an ordinal WITHIN its
    condition, RULE 0), else its text."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(as_float) if as_float.is_integer() else str(value)


def _head(model, k, scope, uncertainty, pooled_D, pooled_f, n_tracks) -> dict:
    return {"model": str(model), "n_pop": int(k), "scope": scope, "uncertainty": uncertainty,
            "pooled": {"D": [float(d) for d in pooled_D], "fraction": [float(f) for f in pooled_f]},
            "D": [float(d) for d in pooled_D], "fraction": [float(f) for f in pooled_f],
            "sd": [0.0] * int(k), "D_sd": [0.0] * int(k), "spread": "none", "units": None, "ci": None,
            "withheld": False, "cells": [], "cells_left_out": 0, "splits": [], "n_tracks": int(n_tracks),
            "notes": []}


def _need(k: int, zeros) -> int:
    """The fewest tracks THAT MOVE a unit needs for a ``k``-population fit: `appd_gamma.MIN_VALUES_PER_POPULATION` per
    fitted population -- one fewer where the immobile population of exact-zero tracks is counted, not fitted."""
    from .appd_gamma import MIN_VALUES_PER_POPULATION

    return MIN_VALUES_PER_POPULATION * (int(k) - 1 if _immobile(zeros) else int(k))


def _immobile(zeros) -> bool:
    """Whether a zero decision (`appd_gamma.zero_tracks`' result, its dict, a rule string, or None) holds the
    immobile population."""
    if zeros is None:
        return False
    if isinstance(zeros, str):
        return zeros == "immobile"
    if isinstance(zeros, dict):
        return bool(zeros.get("immobile"))
    return bool(getattr(zeros, "immobile", False))


def _rule(zeros) -> str:
    """A zero decision as the rule a fit imposes (`appd_gamma.zero_rule_of`): "auto" without one."""
    if zeros is None:
        return "auto"
    if isinstance(zeros, str):
        return zeros
    return "immobile" if _immobile(zeros) else "kept"


def _zero_dict(zeros):
    return None if zeros is None or isinstance(zeros, str) else (zeros if isinstance(zeros, dict)
                                                                    else zeros.to_dict())


def _units_with_enough(units, need: int, word: str, usable=None):
    """``(labels kept, notes)``: the replicates (or chunks) holding at least ``need`` tracks, in order; chunk 0 (a
    track outside every dealt cell) is never one. ``usable`` (each track's mask: the ones a gamma fit holds, those that
    move) counts only those, where some track is exactly zero."""
    if usable is not None:
        unit_text = np.asarray(units, dtype=object).astype(str)
        labels = np.unique(unit_text)
        counts = np.array([int(np.sum(np.asarray(usable)[unit_text == u])) for u in labels])
    else:
        labels, counts = np.unique(np.asarray(units, dtype=object).astype(str), return_counts=True)
    order = sorted(range(len(labels)), key=lambda i: (_label(labels[i]) if isinstance(_label(labels[i]), int)
                                                      else float("inf"), str(labels[i])))
    kept, notes = [], []
    for i in order:
        if labels[i] in ("0", "None", "nan"):
            continue
        if counts[i] >= need:
            kept.append(labels[i])
        else:
            notes.append(f"{word} {_label(labels[i])} left out: {int(counts[i])} tracks"
                         + (" that move" if usable is not None else "") + f", fewer than {need}")
    return kept, notes


def _usable(x):
    """Each track's mask of what a gamma fit holds (the tracks that move), or None where every track moves."""
    moving = np.asarray(x, dtype=float) > 0
    return None if moving.all() else moving


def _at(o, rows):
    """Per-spot precision's noise offsets of ``rows`` (a mask or indices), None without per-spot precision."""
    return None if o is None else o[rows]


def _cell_rows(tracks_cells, tracks_movies, x, n, units, unit_D, k, o=None) -> tuple:
    """Each cell's fractions at its unit's D (``unit_D``: ``{unit label: D}``), the nested mixed model's data
    (`significance`'s cell tests): ``([{"cell", "movie", "day", "n_steps", "fractions"}], left out)``. ``n_steps`` is
    the cell's TRACKS here (one apparent D each), the weight the meta-analysis and the bootstrap give a cell."""
    from .appd_gamma import fractions_at

    rows, left = [], 0
    if tracks_cells is None:
        return rows, left
    cells = np.asarray(tracks_cells, dtype=object)
    movies = np.asarray(tracks_movies, dtype=object) if tracks_movies is not None else np.full(len(cells), "")
    units = np.asarray(units, dtype=object).astype(str)
    for cell in sorted({str(c) for c in cells if c is not None and c == c}):
        mine = np.array([str(c) == cell for c in cells])
        unit = units[mine][0]
        if unit not in unit_D:
            continue
        if int(mine.sum()) < CELL_MIN_TRACKS:
            left += 1
            continue
        f = fractions_at(x[mine], n[mine], unit_D[unit], offset=_at(o, mine))
        if f is None:
            left += 1
            continue
        day = _label(unit)
        rows.append({"cell": cell, "movie": str(movies[mine][0]), "day": int(day) if isinstance(day, int) else 0,
                     "n_steps": int(mine.sum()), "fractions": [float(v) for v in f][:k]})
    return rows, left


def _resampled(x, n, groups, refit, n_boot: int = BOOT, seed: int = 0):
    """``refit(rows) -> (D, fractions) | None`` over ``n_boot`` resamples of ``groups`` (each track's movie or cell)
    with replacement: the 2.5 / 97.5 percentiles of what converged, None under 80 % converged."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups, dtype=object).astype(str)
    labels = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in labels}
    fractions, Ds = [], []
    for _ in range(int(n_boot)):
        picked = np.concatenate([rows[g] for g in rng.choice(labels, size=len(labels), replace=True)])
        got = refit(picked)
        if got is not None:
            Ds.append(got[0])
            fractions.append(got[1])
    if len(fractions) < 0.8 * n_boot:
        return None
    f = np.asarray(fractions, dtype=float)
    lo_f, hi_f = np.percentile(f, [2.5, 97.5], axis=0)
    out = {"fraction": [(float(a), float(b)) for a, b in zip(lo_f, hi_f)], "D": None, "n_units": int(len(labels)),
           "n_boot": int(len(fractions))}
    if Ds[0] is not None:
        d = np.asarray(Ds, dtype=float)
        lo_d, hi_d = np.percentile(d, [2.5, 97.5], axis=0)
        out["D"] = [(float(a), float(b)) for a, b in zip(lo_d, hi_d)]
    return out


def _held(x, n, D, f0, o=None):
    """A resample's fractions at a D HELD (no D of its own): ``(None, fractions)``, or None when they cannot be read."""
    from .appd_gamma import fractions_at

    got = fractions_at(x, n, D, f0, offset=o)
    return None if got is None else (None, got)


def _deals(table, tracks, k: int) -> tuple:
    """Chunks (Uncertainty "Chunks"): ``(labels, splits, pseudo)`` -- each track's chunk under SQD's deal (seed
    `comparison.PSEUDO_SEED`), each track's chunk under every other deal, and what the figure says of them. The deal
    is made on the condition's whole table, so it is SQD's deal of the same movies (or cells)."""
    from .comparison import PSEUDO_SEED, PSEUDO_SPLITS, pseudo_replicate_deal, pseudo_replicate_labels

    labels, splits = None, []
    by = "movie"
    for seed in range(PSEUDO_SEED, PSEUDO_SEED + PSEUDO_SPLITS):
        mapping, by_now = pseudo_replicate_deal(table, k, seed=seed)
        dealt = pseudo_replicate_labels(tracks, mapping, by_now)
        if labels is None:
            labels, by = dealt, by_now
        else:
            splits.append(dealt)
    # the chunks this condition fills (`comparison.pseudo_chunk_count`): the count set at most
    pseudo = {"k": int(labels.max()) if labels is not None and len(labels) else int(k), "by": by,
              "seed": PSEUDO_SEED, "real_replicates": int(tracks["day"].nunique())}
    return labels, splits, pseudo


def condition_designs(table, tracks, x, n, fits: dict, selection: dict, settings: dict, zeros=None, o=None) -> dict:
    """``{model: design}`` for every model of a condition fitted ON ITS OWN (Force same D across Replicates or Don't
    force anywhere), from its pooled fits (``fits``: count -> `GammaFit`) and BIC ``selection``. A set-wide scope
    returns ``{}`` (`joint_designs` makes those). A model whose count did not converge is ``{"model", "n_pop",
    "failed"}``. ``table`` is the condition's whole table (the Chunks deal is made on it), ``tracks`` its
    `appd.track_appd` rows. ``zeros`` is the condition's exact-zero decision (`appd_gamma.zero_tracks`), imposed on
    every fit here and carried by every design (``zero_jumps``). ``o``: each track's noise offset under per-spot
    precision (`appd_gamma`'s module head), in every fit here; None without it."""
    if settings["scope"] in SET_SCOPES:
        return {}
    uncertainty = settings["uncertainty"]
    units = pseudo = None
    splits = []
    held = chunks_held(tracks) if uncertainty == "chunk" else None
    if uncertainty == "day":
        units = np.array([str(_label(d)) for d in tracks["day"]], dtype=object)
    elif uncertainty == "chunk" and not held:
        dealt, split_labels, pseudo = _deals(table, tracks, int(settings["chunks"]))
        units = np.array([str(v) for v in dealt], dtype=object)
        splits = [np.array([str(v) for v in s], dtype=object) for s in split_labels]
    out = {}
    for model in settings["models"]:
        k = resolve_count(model, selection)
        if k not in fits:
            # why, in the pooled fit's own words where it said them (one population beside exact-zero tracks)
            why = next((r.get("failed") for r in selection.get("rows") or () if int(r["n_pop"]) == k
                        and r.get("failed")), "")
            out[model] = {"model": str(model), "n_pop": k,
                          "failed": why if _immobile(zeros) and why else f"the {k}-population fit did not converge"}
            continue
        out[model] = _own_design(model, k, fits[k], tracks, x, n, settings["scope"], uncertainty, units, splits,
                                 pseudo, held, zeros=zeros, o=o)
    return out


def chunks_held(tracks) -> str | None:
    """Why Chunks are withheld from a condition (its tracks' replicate labels): SQD's rule, `comparison.chunks_withheld`
    (ruled 4 Nov 2026) -- None where it holds one replicate and the chunks may be dealt."""
    from .comparison import chunks_withheld

    days = tracks["day"] if hasattr(tracks, "columns") else (tracks or {}).get("day") or ()
    return chunks_withheld(len(set(str(d) for d in days)))


def _withhold_chunks(design: dict) -> dict:
    """A condition's design under Chunks where it has replicates: the pooled fit, no spread, and the reason."""
    from .comparison import CHUNKS_WITHHELD

    design["withheld"] = True
    design["notes"].append(CHUNKS_WITHHELD)
    return design


def _own_design(model, k, pooled, tracks, x, n, scope, uncertainty, units, splits, pseudo, held=None,
                zeros=None, o=None) -> dict:
    from . import appd_gamma as ag

    design = _head(model, k, scope, uncertainty, pooled.D, pooled.fraction, len(x))
    rule = _rule(zeros)
    if zeros is not None:
        design["zero_jumps"] = _zero_dict(zeros)
    if held:
        return _withhold_chunks(design)
    word = "chunk" if pseudo else "replicate"
    real_days = int(tracks["day"].nunique())
    cells = tracks["cell"].tolist() if "cell" in tracks.columns else None
    movies = tracks["movie"].astype(str).tolist() if "movie" in tracks.columns else None
    if units is not None:
        kept, notes = _units_with_enough(units, _need(k, zeros), word, _usable(x))
        design["notes"] += notes
        per_f, per_D, labels, counts = [], [], [], []
        if len(kept) >= 2 and scope == "condition":
            masks = [units == u for u in kept]
            if k >= 2:
                sim = ag.fit_appd_gamma_simultaneous({u: x[m] for u, m in zip(kept, masks)},
                                                     n_steps={u: n[m] for u, m in zip(kept, masks)}, n_pop=k,
                                                     unit_kind="day", zeros=rule,
                                                     offset=None if o is None else {u: o[m]
                                                                                    for u, m in zip(kept, masks)})
                D = [float(d) for d in sim.D]
                per_f = [[float(v) for v in f] for f in sim.fractions_by_unit]
            else:
                D, per_f = [float(d) for d in pooled.D], [[1.0] for _ in kept]
            labels, counts = list(kept), [int(m.sum()) for m in masks]
            unit_D = {u: D for u in kept}
        elif len(kept) >= 2:
            # DON'T FORCE ANYWHERE: each replicate (or chunk) fitted alone at the condition's count -- its own D
            unit_D = {}
            for u in kept:
                m = units == u
                try:
                    fit = ag.fit_appd_gamma(x[m], n_steps=n[m], n_pop=k, zeros=rule, offset=_at(o, m))
                except ag.AppDGammaError as exc:
                    design["notes"].append(f"{word} {_label(u)} left out: its own {k}-population fit failed ({exc})")
                    continue
                order = np.argsort(fit.D)
                per_D.append([float(fit.D[i]) for i in order])
                per_f.append([float(fit.fraction[i]) for i in order])
                labels.append(u)
                counts.append(int(m.sum()))
                unit_D[u] = per_D[-1]
            D = np.mean(per_D, axis=0).tolist() if per_D else list(pooled.D)
        if len(labels) >= 2:
            f = np.asarray(per_f, dtype=float)
            design.update(D=[float(d) for d in D], fraction=f.mean(axis=0).tolist(),
                          sd=f.std(axis=0, ddof=1).tolist(), spread="sd",
                          D_sd=(np.asarray(per_D).std(axis=0, ddof=1).tolist() if per_D else [0.0] * k),
                          units={"kind": "chunk" if pseudo else "day", "labels": [_label(u) for u in labels],
                                 "n_tracks": counts, "fractions": per_f, "D": per_D or None})
            if scope == "condition":
                # EACH CELL AT THE D ITS CONDITION'S REPLICATES SHARE: the cell tests' data (cells in movies in
                # replicates, as SQD's). None under Don't force anywhere, SQD's rule (`comparison`): each replicate
                # has its own D there, so no D is shared by the cells and their fractions measure different states.
                design["cells"], design["cells_left_out"] = _cell_rows(cells, movies, x, n, units, unit_D, k, o)
            if pseudo:
                design["pseudo"] = dict(pseudo)
                if scope == "condition":
                    # every other deal's per-chunk fractions at the same D: how far a p moves with the deal
                    for split in splits:
                        chunks = sorted({s for s in split if s not in ("0",)}, key=lambda s: int(s))
                        row = []
                        for c in chunks:
                            got = ag.fractions_at(x[split == c], n[split == c], design["D"],
                                                  offset=_at(o, split == c))
                            if got is not None:
                                row.append([float(v) for v in got])
                        if len(row) >= 2:
                            design["splits"].append(row)
        else:
            design["notes"].append(f"fewer than two {word}s with enough tracks: the bubble is the pooled fit")
    elif uncertainty in ("movie", "cell"):
        if real_days >= 2:
            design["withheld"] = True
            design["notes"].append(f"{uncertainty} interval withheld: two or more replicates make the replicate the "
                                   f"unit, and {uncertainty}s within a replicate are not independent (choose "
                                   f"Replicates)")
        else:
            groups = (tracks["cell"].astype(str) if uncertainty == "cell" else tracks["movie"].astype(str)).to_numpy()
            if len(set(groups)) >= 2:
                ci = _resampled(x, n, groups, lambda rows: ag.refit(x[rows], n[rows], pooled.D, pooled.fraction,
                                                                    _at(o, rows)))
                if ci is not None:
                    design.update(ci=dict(ci, unit=uncertainty), spread="ci")
                else:
                    design["notes"].append(f"the {uncertainty} interval: fewer than 80 % of the resamples converged")
    return design


# =====================================================================================================================
# Conditions fitted together: Force same D across Conditions (a factor) or Whole data set
# =====================================================================================================================

def joint_inputs(results: dict, conditions) -> dict:
    """What `joint_designs` needs of each landed condition, from its result alone (no project): every track's apparent
    D, step count, replicate, movie, cell and chunk labels, and its own pooled fits (``fits`` summaries, for the
    group check). Conditions that failed are left out."""
    out = {}
    for c in conditions:
        res = results.get(c)
        if not res or "error" in res:
            continue
        tr = res["tracks"]
        out[c] = {"x": list(tr["appd"]), "n": list(tr["n_steps"]), "day": list(tr["day"]),
                  # per-spot precision's noise offset per track (None without it), every joint fit's too
                  "o": list(tr["offset"]) if tr.get("offset") is not None else None,
                  "movie": list(tr.get("movie") or []), "cell": list(tr.get("cell") or []),
                  "chunk": list(tr.get("chunk") or []), "chunk_splits": [list(s) for s in tr.get("chunk_splits") or []],
                  "pseudo": res.get("pseudo"), "fits": res["fits"],
                  # the frame interval, for the joint exact-zero decision's lattice test (`appd_gamma.zero_tracks`)
                  "tau_s": res.get("tau_s")}
    return out


def _groups_of(config: dict, conditions) -> dict:
    """``{group name: [conditions]}``: one group of every condition under Whole data set, a group per value of the
    factor under Conditions."""
    if config["scope"] == "global":
        return {"": list(conditions)}
    groups: dict = {}
    for c in conditions:
        groups.setdefault(str((config.get("groups") or {}).get(c, "")), []).append(c)
    return groups


def joint_designs(inputs: dict, config: dict) -> dict:
    """``{"models": {model: {condition: design}}, "selections": {group: BIC rows}, "errors": {condition: why}}`` under
    a set-wide Force same D across: per group (every condition under Whole data set; each value of the factor under
    Conditions) and model, ONE D for the group -- the joint gamma fit over its conditions (each condition a unit,
    `fit_appd_gamma_simultaneous` with unit_kind "condition") -- at the model's count (Best: BIC on the group's pooled
    tracks, as SQD chooses one count for a joint fit), then each condition's replicates, chunks or cells at that D
    (`appd_gamma.fractions_at`). A condition alone in its group keeps its own D (SQD's rule)."""
    from . import appd_gamma as ag

    models = list(config["models"])
    out = {"models": {m: {} for m in models}, "selections": {}, "errors": {}}
    for name, members in _groups_of(config, [c for c in inputs]).items():
        xs = {c: np.asarray(inputs[c]["x"], dtype=float) for c in members}
        ns = {c: np.asarray(inputs[c]["n"], dtype=float) for c in members}
        per_spot = [inputs[c].get("o") is not None for c in members]
        if any(per_spot) and not all(per_spot):
            off = [c for c, on in zip(members, per_spot) if not on]
            raise AppDRunError(f"one D forced across conditions with and without per-spot precision ({', '.join(off)}: "
                               f"their tracks carry none) cannot model the error in some of them only: force D "
                               f"across Replicates, or leave {', '.join(off)} out")
        os_ = {c: np.asarray(inputs[c]["o"], dtype=float) for c in members} if all(per_spot) else None
        pooled_x = np.concatenate([xs[c] for c in members])
        pooled_n = np.concatenate([ns[c] for c in members])
        pooled_o = None if os_ is None else np.concatenate([os_[c] for c in members])
        # EXACT-ZERO TRACKS, decided once on the group's pooled tracks (each condition at its own frame interval),
        # imposed on every fit of the group: each condition then holds the immobile population at its own share
        taus = [inputs[c].get("tau_s") for c in members]
        zeros = ag.zero_tracks(pooled_x, pooled_n, tau_s=None if any(t is None for t in taus) else np.concatenate(
            [np.full(len(xs[c]), float(t)) for c, t in zip(members, taus)]))
        rule = _rule(zeros)
        counts = fit_counts(models, config["best_max_pop"])
        pooled_fits, rows = {}, []
        for k in counts:
            try:
                pooled_fits[k] = ag.fit_appd_gamma(pooled_x, n_steps=pooled_n, n_pop=k, zeros=rule, offset=pooled_o)
            except ag.AppDGammaError as exc:
                rows.append({"n_pop": k, "bic": None, "delta": None, "failed": str(exc)})
        best_pool = {k: f for k, f in pooled_fits.items() if k <= int(config["best_max_pop"])}
        recommended = min(best_pool, key=lambda k: best_pool[k].bic) if best_pool else None
        low = min((f.bic for f in best_pool.values()), default=None)
        rows += [{"n_pop": k, "bic": float(f.bic), "delta": float(f.bic - low) if low is not None else None,
                  "failed": ""} for k, f in pooled_fits.items()]
        selection = {"rows": sorted(rows, key=lambda r: r["n_pop"]), "recommended": recommended,
                     "members": list(members)}
        if zeros is not None:
            selection["zero_tracks"] = zeros.to_dict()
            if zeros.immobile and 1 in counts:
                selection["notes"] = [ag.one_population_note(zeros, int(config["best_max_pop"]))]
        out["selections"][name] = selection
        for model in models:
            if model == "best" and recommended is None:
                for c in members:
                    out["models"][model][c] = {"model": model, "n_pop": None,
                                               "failed": "no count could be fitted to the group's pooled tracks"}
                continue
            k = resolve_count(model, selection)
            try:
                designs = _joint_group(model, k, members, xs, ns, inputs, config, name, pooled_fits.get(k),
                                       zeros=zeros, os_=os_)
            except ag.AppDGammaError as exc:
                designs = {c: {"model": model, "n_pop": k, "failed": f"the joint {k}-population fit failed ({exc})"}
                           for c in members}
            out["models"][model].update(designs)
    return out


def _joint_group(model, k, members, xs, ns, inputs, config, group, pooled_fit, zeros=None, os_=None) -> dict:
    from . import appd_gamma as ag

    scope, uncertainty = config["scope"], config["uncertainty"]
    rule = _rule(zeros)
    if _immobile(zeros) and k < 2:
        # one population cannot hold the immobile population of exact-zero tracks beside moving ones
        raise ag.AppDGammaError(ag._refused_one(zeros))
    if len(members) >= 2 and k >= 2:
        joint = ag.fit_appd_gamma_simultaneous({c: xs[c] for c in members}, n_steps={c: ns[c] for c in members},
                                               n_pop=k, unit_kind="condition", zeros=rule, offset=os_)
        D = [float(d) for d in joint.D]
        own_f = {c: [float(v) for v in f] for c, f in zip(members, joint.fractions_by_unit)}
    elif len(members) >= 2:
        D = [float(d) for d in pooled_fit.D] if pooled_fit is not None else [float(np.mean(np.concatenate(
            [xs[c] for c in members])))]
        own_f = {c: [1.0] for c in members}
    else:                                                      # alone in its group: its own D
        (c,) = members
        own = ag.fit_appd_gamma(xs[c], n_steps=ns[c], n_pop=k, zeros=rule, offset=None if os_ is None else os_[c])
        D, own_f = [float(d) for d in own.D], {c: [float(v) for v in own.fraction]}
    out = {}
    for c in members:
        x, n = xs[c], ns[c]
        o = None if os_ is None else os_[c]
        own_fit = (inputs[c]["fits"].get(k) or inputs[c]["fits"].get(str(k)) or {})
        design = _head(model, k, scope, uncertainty, D, own_f[c], len(x))
        design["shared_D"] = list(D)
        if own_fit and "D" in own_fit and len(members) >= 2:
            design["own_D"] = [float(d) for d in own_fit["D"]]
        design["group"] = {"factor": config.get("factor") or "", "name": group, "members": list(members)}
        if zeros is not None:
            design["zero_jumps"] = _zero_dict(zeros)
        days = np.array([str(_label(d)) for d in inputs[c]["day"]], dtype=object)
        real_days = len(set(days))
        cells = inputs[c]["cell"] or None
        movies = inputs[c]["movie"] or None
        if uncertainty == "chunk" and chunks_held({"day": list(days)}):
            out[c] = _withhold_chunks(design)
            continue
        if uncertainty in ("day", "chunk"):
            pseudo = inputs[c].get("pseudo") if uncertainty == "chunk" else None
            units = (np.array([str(v) for v in inputs[c]["chunk"]], dtype=object) if uncertainty == "chunk"
                     else days)
            word = "chunk" if uncertainty == "chunk" else "replicate"
            kept, notes = _units_with_enough(units, _need(k, zeros), word, _usable(x))
            design["notes"] += notes
            per_f, labels, counts = [], [], []
            for u in kept:
                m = units == u
                got = ag.fractions_at(x[m], n[m], D, own_f[c], offset=_at(o, m))
                if got is not None:
                    per_f.append([float(v) for v in got])
                    labels.append(u)
                    counts.append(int(m.sum()))
            if len(per_f) >= 2:
                f = np.asarray(per_f, dtype=float)
                design.update(fraction=f.mean(axis=0).tolist(), sd=f.std(axis=0, ddof=1).tolist(), spread="sd",
                              units={"kind": word if word == "chunk" else "day",
                                     "labels": [_label(u) for u in labels], "n_tracks": counts, "fractions": per_f,
                                     "D": None})
                design["cells"], design["cells_left_out"] = _cell_rows(cells, movies, x, n, units,
                                                                       {u: D for u in labels}, k, o)
                if pseudo:
                    design["pseudo"] = dict(pseudo)
                    for split in inputs[c].get("chunk_splits") or []:
                        split = np.array([str(v) for v in split], dtype=object)
                        row = [ag.fractions_at(x[split == s], n[split == s], D, own_f[c], offset=_at(o, split == s))
                               for s in sorted({s for s in split if s != "0"}, key=int)]
                        row = [[float(v) for v in r] for r in row if r is not None]
                        if len(row) >= 2:
                            design["splits"].append(row)
            else:
                design["notes"].append(f"fewer than two {word}s with enough tracks: the bubble is the condition's "
                                       f"fractions at the shared D")
        elif uncertainty in ("movie", "cell"):
            if real_days >= 2:
                design["withheld"] = True
                design["notes"].append(f"{uncertainty} interval withheld: two or more replicates make the replicate "
                                       f"the unit (choose Replicates)")
            else:
                groups = np.asarray(cells if uncertainty == "cell" else movies, dtype=object).astype(str)
                if groups.size == len(x) and len(set(groups)) >= 2:
                    ci = _resampled(x, n, groups, lambda rows, x=x, n=n, c=c, o=o: _held(x[rows], n[rows], D,
                                                                                       own_f[c], _at(o, rows)))
                    if ci is not None:
                        design.update(ci=dict(ci, unit=uncertainty), spread="ci")
        out[c] = design
    return out


# =====================================================================================================================
# The Populations figure
# =====================================================================================================================

def design_for(results: dict, joint: dict | None, condition: str, model: str) -> dict | None:
    """The design behind ``condition``'s column at ``model``: the set-wide joint fit's, else the condition's own."""
    if joint is not None:
        return ((joint.get("models") or {}).get(str(model)) or {}).get(condition)
    res = results.get(condition) or {}
    if "designs" not in res and res and "error" not in res:
        return _legacy_design(res)
    designs = res.get("designs") or {}
    if str(model) in designs:
        return designs[str(model)]
    # BEST IS ITS COUNT'S FIT (`bubble_view.fold_best`'s reasoning): a Best asked of a result fitted at the count Best
    # chose, or that count asked of a result fitted at Best, is the same fit
    chosen = (res.get("selection") or {}).get("recommended")
    if str(model) == "best" and chosen is not None and str(chosen) in designs:
        return designs[str(chosen)]
    best = designs.get("best") or {}
    if str(model).isdigit() and best.get("n_pop") == int(model):
        return best
    return None


def _legacy_design(res: dict) -> dict:
    """A result made before 3 Nov 2026 (a session kept then: one count, no designs) as the one design it drew -- its
    replicates' simultaneous fit, its cell interval, or its pooled fit -- so the session opens as it was."""
    settings = res.get("settings") or {}
    uncertainty = settings.get("uncertainty", "day")
    k = int(res["n_pop"])
    pooled = res["pooled"]
    design = _head(settings.get("states", str(k)), k, "condition", uncertainty, pooled["D"], pooled["fraction"],
                   res.get("n_tracks", 0))
    days = res.get("days")
    if uncertainty == "day" and days:
        design.update(D=list(days["D"]), fraction=list(days["mean"]), sd=list(days["sd"]), spread="sd",
                      units={"kind": "day", "labels": list(days["labels"]), "n_tracks": list(days["n_tracks"]),
                             "fractions": [list(f) for f in days["fractions"]], "D": None})
    elif uncertainty == "cell" and res.get("cell_ci"):
        design.update(ci=dict(res["cell_ci"], unit="cell", n_units=res["cell_ci"].get("n_cells")), spread="ci")
    design["withheld"] = bool(res.get("cell_withheld"))
    design["notes"] = list(res.get("notes") or [])
    return design


def unit_values_of(design: dict) -> dict | None:
    """What a significance bracket tests for one drawn column (`BubblePlotData.unit_values`): its per-replicate (or
    per-chunk) fractions, whose MEAN is the drawn bubble, the other deals of the chunks, and each cell's fractions at
    the same D (the cell tests) -- None where the bubble is no mean over replicates."""
    units = design.get("units")
    if design.get("spread") != "sd" or not units or len(units["fractions"]) < 2:
        return None
    return {"unit": "chunk" if units["kind"] == "chunk" else "day", "replicates": list(units["labels"]),
            "fractions": [list(f) for f in units["fractions"]], "D": list(design["D"]),
            **({"splits": [list(s) for s in design["splits"]]} if design.get("splits") else {}),
            "cells": list(design.get("cells") or []), "cells_left_out": int(design.get("cells_left_out") or 0)}


def _sample(design: dict, replicates: int = 1) -> dict:
    """A column's sample (`bubble_view.sample_text`). With no spread drawn the bubble is every replicate pooled:
    ``replicates`` of them, so the line reads "3 replicates pooled · 600 tracks" as SQD's does (found by the overnight
    tester, 4 Nov 2026: Movies or Cells withheld over 3 replicates read "600 tracks" only)."""
    units = design.get("units")
    if design.get("spread") == "sd" and units:
        return {"unit": units["kind"], "n": len(units["labels"]), "tracks": design.get("n_tracks")}
    ci = design.get("ci")
    if ci:
        return {"unit": ci.get("unit", "cell"), "n": ci.get("n_units"), "tracks": design.get("n_tracks")}
    return {"unit": "pooled", "n": max(1, int(replicates)), "tracks": design.get("n_tracks")}


def _replicates_in(result: dict) -> int:
    """How many replicates a condition's result holds (its tracks' replicate labels); 1 where it keeps none."""
    days = ((result or {}).get("tracks") or {}).get("day") or ()
    return max(1, len(set(days)))


def _one_model(results, joint, conditions, model, config) -> tuple:
    """One model's figure over ``conditions``: ``(BubblePlotData, caption, payloads)`` in `bubble_view`'s shape."""
    from .appd_gamma import d_at_floor
    from .bubble_view import (BubblePlotData, BubblePoint, fit_warnings, group_warnings, immobile_mix_warnings,
                              scopes_caption, spread_caption, warnings_caption, zero_jump_notes)
    from .workspace_view import scope_words

    uncertainty = config["uncertainty"]
    points, errors, unit_values, samples, payloads, scopes, withheld = [], {}, {}, {}, {}, {}, []
    pseudo_by, pseudo_real, pseudo_held, pseudo_ks = {}, {}, {}, {}
    for column, c in enumerate(conditions):
        res = results.get(c)
        if res is None or "error" in res:
            errors[column] = (res or {}).get("error", "not computed")
            continue
        design = design_for(results, joint, c, model)
        if design is None:
            errors[column] = "not computed"
            continue
        if "failed" in design:
            errors[column] = design["failed"]
            continue
        payloads[c] = design
        scopes[c] = design["scope"]
        if design.get("withheld"):
            withheld.append(c)
        ci = design.get("ci")
        units = design.get("units")
        for k in range(int(design["n_pop"])):
            points.append(BubblePoint(
                # a D of 0 is the immobile population of exact-zero tracks, drawn at its "0" tick (`chart_text`)
                column=column, x_offset=0.0,
                d=(0.0 if float(design["D"][k]) == 0 else max(float(design["D"][k]), 1e-6)),
                fraction=float(design["fraction"][k]),
                fraction_sd=float(design["sd"][k]), n_replicates=len(units["labels"]) if units else 1,
                population=k + 1, style_index=0, spread=design["spread"],
                interval=tuple(ci["fraction"][k]) if ci else None,
                d_sd=float(design["D_sd"][k]) if design["spread"] == "sd" else 0.0,
                d_interval=tuple(ci["D"][k]) if ci and ci.get("D") else None))
            # A D ON THE GAMMA MIXTURE'S FLOOR IS SAID (10 Nov 2026): the table and the hover read "not resolved"
            # (`bubble_view.d_text`), as SQD's bound
            points[-1].d_bound = "lower" if d_at_floor(design["D"][k]) else ""
        values = unit_values_of(design)
        if values is not None:
            unit_values[c] = values
        samples[c] = _sample(design, _replicates_in(res))
        if design.get("pseudo"):
            pseudo_by[c] = design["pseudo"]["by"]
            pseudo_ks[c] = int(design["pseudo"].get("k") or config["chunks"])
            pseudo_real[c] = int(design["pseudo"]["real_replicates"])
        elif uncertainty == "chunk" and design.get("withheld"):
            # CHUNKS WITHHELD (`comparison.chunks_withheld`): drawn pooled, the reason for the caption and brackets
            from .comparison import chunks_withheld

            pseudo_held[c] = chunks_withheld(_replicates_in(res))
    plot_uncertainty = "day" if uncertainty == "chunk" else uncertainty
    plot = BubblePlotData(conditions=list(conditions), points=points, errors=errors, series_labels=[],
                          scope=config["scope"], fraction_weighting="track", uncertainty=plot_uncertainty,
                          unit_values=unit_values, samples=samples, d_scopes=scopes)
    if uncertainty == "chunk":
        from .comparison import PSEUDO_SEED, PSEUDO_SPLITS

        plot.pseudo = {"k": int(config["chunks"]), "seed": PSEUDO_SEED, "splits": PSEUDO_SPLITS, "by": pseudo_by,
                       "ks": pseudo_ks,
                       "real_replicates": pseudo_real, "withheld": pseudo_held}
    if config["scope"] == "group":
        plot.sharing_factor = str(config.get("factor") or "")
        plot.sharing_groups = dict(config.get("groups") or {})
    plot.count_warnings = count_warnings(results, [c for c in conditions if c in payloads], model, config)
    plot.warnings = fit_warnings(plot.points, plot.conditions) + group_warnings(payloads) + plot.count_warnings
    # NOT MODELLED: two slow populations of similar D may be one bound population (`shared_error_warnings`)
    plot.warnings += shared_error_warnings(plot.points, plot.conditions, config)
    # exact-zero tracks (`appd_gamma.zero_tracks`): each fit's note, as SQD's bubbles say theirs, and a joint fit
    # joining conditions with them to conditions without (ruled 4 Nov 2026: a warning, not a refusal)
    plot.warnings += zero_jump_notes(payloads) + immobile_mix_warnings(payloads, noun="tracks")
    counts = ", ".join(f"{c}: {payloads[c]['n_pop']}" for c in conditions if c in payloads)
    # PER SPOT (6 Nov 2026): the calibration said, as SQD's caption and legend say theirs (`bubble_view.precision_of`)
    if config.get("precision") == "per_spot" and config.get("calibration"):
        off = [c for c in conditions if ((results.get(c) or {}).get("settings") or {}).get("precision_off")]
        plot.precision = str(config["calibration"][2]) + (
            f"; not modelled for {', '.join(off)}: their tracks carry no per-spot precision (a track table; a truth "
            f"condition has no localisation error)" if off else "")
    caption = (f"{APPD_HEAD} -- populations {counts}  --  {spread_caption(plot_uncertainty, withheld, config['scope'], plot.pseudo, plot.points, BOOT)}"
               f"; {scope_words(config['scope'], config.get('factor') or '')}"
               + warnings_caption(plot.warnings) + scopes_caption(plot) + appd_precision_caption(plot.precision))
    return plot, caption, payloads


def appd_precision_caption(precision: str) -> str:
    """What per-spot precision means in the apparent D, for a caption ("" without it, the default)."""
    if not precision:
        return ""
    from .bubble_view import PER_SPOT_D_IS

    return ("   [localisation error per spot: each track its own, from its spots' precision through a calibration "
            f"measured on immobile spots ({precision}); D {PER_SPOT_D_IS}]")


def count_warnings(results: dict, conditions, model, config: dict) -> list:
    """A COUNT SET BY HAND WHERE BEST (BIC) CHOOSES FEWER (ruled 4 Nov 2026, `bubble_view.forced_count_warning`):
    "<condition>: Best (BIC) chooses 1 population here, ...; kept at the 2 you set", per condition fitted at a number
    -- read off the BIC every condition's own pooled fits already carry (`appd_view.appd_condition`'s ``selection``,
    Best's range and the count set), so nothing is fitted for it. Not under a set-wide Force same D across, whose
    count is the joint fit's."""
    from .bubble_view import forced_count_warning

    if not str(model).isdigit() or config["scope"] in SET_SCOPES:
        return []
    k, out = int(model), []
    for c in conditions:
        selection = (results.get(c) or {}).get("selection") or {}
        top = int(selection.get("max_pop") or config.get("best_max_pop") or BEST_MAX_POP[0])
        rows = [[int(r["n_pop"]), float(r["bic"])] for r in selection.get("rows") or ()
                if r.get("bic") is not None and (int(r["n_pop"]) <= top or int(r["n_pop"]) == k)]
        words = forced_count_warning(k, rows)
        if words:
            out.append(f"{c}: {words}")
    return out


def shared_error_warnings(points, conditions, config: dict) -> list:
    """WHY THE TWO SLOWEST POPULATIONS CAN READ ALIKE UNDER "NOT MODELLED" (10 Nov 2026; docs/claude/
    sqd_and_statistics.md, "THE APPARENT D'S GAMMA SHAPE UNDER LOCALISATION ERROR"). Consecutive steps share a
    localisation, so the apparent D of a molecule that does not move scatters more than a gamma of the track's step
    count allows (its variance 1.375-fold at four steps), and the fit can read ONE bound population as two of similar
    D: on declared truth Best (BIC) chose two for 3000 bound tracks, at 0.035 and 0.053 µm²/s. Localisation error Per
    spot carries the cause in its shape (`appd_gamma.shape_and_scale`). Not modelled has no localisation error to
    compute it from, so the over-count stays and is SAID: one line naming each condition whose two slowest MOVING
    populations `bubble_view.fit_checks` finds similar in D, with the cause and the two remedies. Beside
    `fit_warnings`' own line, whose words are every section's and are not changed. Nothing under Per spot."""
    from .bubble_view import fit_checks

    if config.get("precision") == "per_spot":
        return []
    slowest: dict = {}
    for p in points:
        get = p.get if isinstance(p, dict) else (lambda k, p=p: getattr(p, k))
        d, key = float(get("d")), (int(get("column")), int(get("style_index") or 0))
        if d > 0 and (key not in slowest or d < slowest[key][0]):
            slowest[key] = (d, int(get("population")))
    names = []
    for check in fit_checks(points, conditions):
        key = (check["column"], check["style"])
        if (len(check["populations"]) == 2 and key in slowest and check["populations"][0] == slowest[key][1]
                and check["name"] not in names):
            names.append(check["name"])
    if not names:
        return []
    return [f"{', '.join(names)}: the two slowest populations sit near the localisation floor, so they may be "
            "one bound population read as two. Consecutive steps share a localisation, so a bound molecule "
            "scatters wider than this fit allows while Localisation error is Not modelled. Run with "
            "Localisation error Per spot, or fit one population fewer."]


def bubble_plot(results: dict, conditions, config: dict, joint: dict | None = None) -> tuple:
    """The Populations figure: ``(BubblePlotData, caption, payloads)``. Every model any condition shows; one model is
    the plain figure (a column per condition), two or more are fitted SIDE BY SIDE (`bubble_view.side_by_side`: each
    condition a group of its models' columns, a shade per model, Best once where it chose a count also picked, in
    order of count). ``payloads`` maps each column's name to its design (the replicate page a click opens).
    ``joint`` is `joint_designs`' result under a set-wide Force same D across."""
    from .bubble_view import side_by_side

    conditions = [str(c) for c in conditions]
    shown = {c: (list(config["models"]) if config["scope"] in SET_SCOPES else models_of(config, c))
             for c in conditions}
    ticked = [m for m in SECTION_MODELS if any(m in ms for ms in shown.values())]
    if len(ticked) <= 1:
        plot, caption, payloads = _one_model(results, joint, conditions, ticked[0] if ticked else "best", config)
        return plot, caption, payloads
    runs = []
    for model in ticked:
        mine = [c for c in conditions if model in shown[c]]
        runs.append((model, *_one_model(results, joint, mine, model, config)))
    plot, caption, payloads = side_by_side(runs, order=conditions, shown=shown, head=APPD_HEAD)
    return plot, caption, payloads


def day_bubbles(design: dict | None) -> dict | None:
    """One drawn column's replicates (or chunks) as bubbles -- each its own fractions at the column's D, or at its
    own D under Don't force anywhere -- and the table under them, in the SQD bubble payload's shape. None where the
    column is no mean over replicates."""
    if not design or "failed" in design or not design.get("units"):
        return None
    units = design["units"]
    word = "Chunk" if units["kind"] == "chunk" else "Replicate"
    labels = [f"{word} {u}" for u in units["labels"]]
    own_D = units.get("D")
    points, rows = [], []
    for column, (fractions, n) in enumerate(zip(units["fractions"], units["n_tracks"])):
        Ds = own_D[column] if own_D else design["D"]
        for k, f in enumerate(fractions):
            points.append({"column": column, "x_offset": 0.0, "d": Ds[k], "fraction": f, "fraction_sd": 0.0,
                           "n_replicates": 1, "population": k + 1, "style_index": 0, "spread": "none",
                           "interval": None})
        cells = [f"{100 * v:.1f} %" + (f" (D {Ds[k]:.3g})" if own_D else "") for k, v in enumerate(fractions)]
        rows.append([labels[column], f"{n:,}", *cells])
    if own_D:
        columns = [word, "Tracks", *[f"Population {k + 1}" for k in range(len(design["D"]))]]
    else:
        columns = [word, "Tracks", *[f"Population {k + 1} (D {d:.3g})" for k, d in enumerate(design["D"])]]
    rows.append(["Mean ± SD", f"{sum(units['n_tracks']):,}",
                 *[f"{100 * m:.1f} ± {100 * s:.1f} %" for m, s in zip(design["fraction"], design["sd"])]])
    return {"conditions": labels, "points": points, "errors": {}, "scope": design["scope"],
            "fraction_weighting": "track", "uncertainty": "none", "unit_values": {}, "columns": columns,
            "rows": rows}
