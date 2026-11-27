"""The HMM section's numbers and sentences, with no Qt anywhere in it.

`ui/workspace/hmm_chart.py` is the Qt shell; this is what it draws from (the Thin Qt rule,
`tests/test_ui_thin_qt.py`), and a process-pool worker imports it without Qt.

WHAT THE SECTION IS. The diffusive-state HMM (`smtracker.hmm`) -- the one estimator here
that MEASURES switching rather than averaging over it -- as a sidebar section of its own
(asked 13 Oct 2026: "then the hmm tab"). v2 had none: vbSPT ran outside it, in MATLAB.
`compute_hmm_view` gives one condition's states (apparent D, occupancy beside the initial
and stationary distributions), the switching rates with their interval, the dwell times,
and with "Best" the BIC table. It returns plain lists, floats and strings.

THE RULES IT KEEPS, each from CLAUDE.md:

- RATES LEAD. The per-frame transition probability is the measured quantity; a dwell time
  is the model's extrapolation (exponential, Markovian) and usually exceeds the median track,
  so it is printed AFTER the rates and labelled model-derived.
- A RATE NEEDS TRANSITIONS. A bootstrap resample with no transition out of a state is
  counted (`n_degenerate`, kept, never dropped); when any occur the rates are called not
  identifiable and their interval is printed as such. A rarely switching condition's off-rate is the example:
  its interval spans a dwell from well under a second to minutes.
- OCCUPANCY IS A TIME-AVERAGE, NOT THE STATIONARY DISTRIBUTION. A 5-6 step track never
  mixes, so the time-average sits between the initial distribution pi and the stationary
  one; all three are given, so nobody checks occupancy against the rates and finds a
  "discrepancy" that is not one.
- SIGMA = 0 BY DEFAULT: apparent D, comparable with SQD. The correction at the tracker's
  sigma is a SENSITIVITY check (at 76 nm the slow state becomes localisation noise).
- BIC RECOMMENDS, THE ANALYST DECIDES; a difference under 2 is inconclusive. Three states
  at most here (the third state's D sits at the localisation floor).
- INTERVALS OVER INDEPENDENT UNITS: per-day fits (mean +- SD over replicate days, each day
  independent, Rule 0), or a movie or cell bootstrap on a single session only (with two or
  more days movies and cells are not independent -- Report A, the gate the SQD and MSD
  sections keep; Movies, 19 Oct 2026: cells in one movie share its focus and field). Rates
  take PERCENTILE intervals: bounded at zero and right-skewed.

NO STATISTIC BETWEEN CONDITIONS: each condition is computed on its own and drawn beside the
others. A condition that fails is a card that says why, never an absence.

THE PAGE, REDRAWN (13 Oct 2026: "too much text ... check also the scripts python for the
manuscript, we had some good ideas there"). The manuscript's HMM figure
(`scripts/figure_hmm_four_panel.py`) compares the conditions on FOUR quantities of the slow
state -- occupancy, the rate into it, the rate out of it, its apparent D -- with each
replicate day an open circle (jittered, no order implied), their mean filled and a +-1 SD
bar that is day-to-day SPREAD, not a confidence interval. `comparison_payload` gives that
strip for the top of the page. Each card then carries the diagram and two compact tables
(`card_tables`); the explanations every card repeated are said once (`READING_NOTES`).
With more than two states the rates into and out of the slow state are summed over the
other states (`slow_numbers`), which for two states are exactly the two fitted rates.
The strip's Export draws this same payload (`figure_export`'s "hmm_comparison"), so it also
carries what the source data and the legend need: each day's name and each condition's
spread, sigma and counts.

THE DWELL AND THE RATES WITH THEIR SPREAD (asked 12 Nov 2026: "can we get for hmm a dwell time and rate +- error.
also for export option to export dwell time as well"). Under Replicates a card's every rate and every model dwell
carries the replicates' mean +- SD beside the pooled fit (each replicate fitted on its own, `_interval`), as the
occupancy did; a replicate whose state was never left has no dwell and is counted, never averaged in. Under Movies
or Cells the dwell carries a 95 % interval: the rate out's (the sum of the state's exits, per resample) INVERTED,
since 1 / x falls as x rises, so a rate-out band reaching zero leaves the dwell with no upper bound, said in words
(`_exit_bands`). The strip may draw a FIFTH panel, the dwell in ms (`DWELL_PANEL`), after the rate out it comes
from: off unless asked (`comparison_payload(dwell=True)`, the strip's "Dwell time" box), and the payload always
carries it (``"dwell_panel"``) so the export can draw it on its own box (`figure_export`'s `FigureSpec.hmm_dwell`).
The dwell stays MODEL-DERIVED and printed after the rates (RATES LEAD, above).

THE TEST PAGE (the UI designer's redesign: a deck of two pages, cards with each state's exits written out, a table of one state)
was RETIRED on 23 Nov 2026 ("retire hmm redesign. keep old"); its table lives on as the section's Table page
(`condition_table`, `CONDITION_TABLE_COLUMNS`, `ui.workspace.hmm_table`).
"""
from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

import math

__all__ = ["STATE_CHOICES", "STATE_LABELS", "UNCERTAINTY_MODES", "UNCERTAINTY_LABELS",
           "SIGMA_MODES", "SIGMA_LABELS", "DEFAULTS", "per_spot", "per_spot_words", "precision_of", "N_BOOT", "compute_hmm_view",
           "card_caption", "card_lines", "diagram_payload", "hmm_config", "config_problem",
           "SETTING_HELP", "setting_help", "rate_text", "slow_numbers", "COMPARE_PANELS", "comparison_tip",
           "comparison_payload", "spread_line", "card_tables", "READING_NOTES", "DWELL_PANEL",
           "STATES", "state_index", "state_numbers", "merged_note", "compare_panels", "pair_comparison", "group_comparison", "replicate_number", "pair_lines",
           "PAIR_LINES_WORDS",
           "missing_note", "PAIR_ROWS", "RULE0_NOTE", "state_strips", "STATE_TABLE_COLUMNS", "condition_table", "CONDITION_TABLE_COLUMNS",
           "CHUNK_NOTE", "chunks_of", "corrected", "d_word", "fit_flags", "sigma_note", "unit_words"]

#: The state count: fixed, or chosen by BIC over 1-3 (or 1-4, `hmm_run.BEST_MAX`) ("best"). 4 since 3 Nov 2026.
STATE_CHOICES = ("2", "3", "4", "best")
STATE_LABELS = ("2 states", "3 states", "4 states", "Best (BIC)")
#: Chunks since 3 Nov 2026 (`hmm_run`): SQD's random pseudo-replicates, fitted as replicates.
UNCERTAINTY_MODES = ("none", "day", "movie", "cell", "chunk")
UNCERTAINTY_LABELS = ("None", "Replicates", "Movies", "Cells", "Chunks")
#: Force same D across, as `hmm_run` passes it to one condition's job: its replicates at the condition's D
#: ("condition"), each its own ("replicate", the behaviour before 3 Nov 2026), or the group's / set's D held
#: ("group", "global", with ``fixed_D``).
SHARED_D_SCOPES = ("condition", "replicate", "group", "global")
#: The single-session choices: a percentile interval from resampling this unit.
RESAMPLED_UNITS = ("movie", "cell")
#: A resampled unit, as a sentence counts it.
UNIT_WORDS = {"movie": "movies", "cell": "cells"}
#: Apparent D (the default), corrected for the tracker's one localisation error, or PER SPOT (6 Nov 2026,
#: `smtracker.precision`): each step's own noise, from the tracker's per-spot precision through a calibration measured
#: on immobile spots (the project's `precision.PROJECT_FILE`, else the tracker's built-in one), never learned.
SIGMA_MODES = ("off", "tracker", "per_spot")
SIGMA_LABELS = ("Off (apparent D)", "Tracker's sigma", "Per spot")
DEFAULTS = {"states": "2", "uncertainty": "day", "sigma": "off"}
#: Movie or cell resamples. 40 took 19 s on the reference project (2005 tracks, one core).
N_BOOT = 100
#: The largest state count "Best" tries.
MAX_STATES = 3


def _resampled(interval: dict) -> bool:
    """True for a single session's movie or cell interval (`RESAMPLED_UNITS`)."""
    return interval.get("mode") in RESAMPLED_UNITS


#: Where a sigma came from when the TRACKER recorded none: computed from the optics (`AcquisitionParams`).
THEORETICAL = "theoretical"


def _sigma_words(nm: float, provenance) -> str:
    """The σ correction in words: "17 nm (tracker_reported)", or for a THEORETICAL sigma -- the tracker recorded none
    -- "none recorded; theoretical 31 nm used" (the overnight tester's question 2, 4 Nov 2026: a theoretical sigma was
    applied with only "(theoretical)" after it, and moved a slow D from 0.051 to 0.004)."""
    if str(provenance or "") == THEORETICAL:
        return f"none recorded; theoretical {nm:.0f} nm used"
    return f"{nm:.0f} nm ({provenance or 'tracker'})"


def _sigma_um(project, mode: str) -> tuple[float, str]:
    if mode == "off":
        return 0.0, "off (apparent D, comparable with SQD)"
    from .spatial_map_view import sigma_stamp

    stamp = sigma_stamp(project)
    nm = stamp.get("sigma_nm")
    if not nm:
        raise ValueError("this project carries no localisation sigma to correct with")
    return float(nm) / 1000.0, _sigma_words(float(nm), stamp.get("sigma_provenance"))


def precision_of(project, conditions, mode: str, given=None):
    """The calibration a per-spot fit of ``conditions`` (one name, or several fitted together) uses, None for every
    other sigma mode: ``given`` -- the Run's own ``(c_nm, p, source)``, chosen once for all its conditions
    (`hmm_run.with_precision`) -- else `precision.calibration_for` on their folders and trackers. RAISES ValueError
    saying why there is none (a track table, two files, two trackers)."""
    if mode != "per_spot":
        return None
    from .precision import Calibration, PrecisionError, calibration_for

    try:
        if given is not None:
            c_nm, p, source = given
            return Calibration(c_nm=float(c_nm), p=float(p), source=str(source))
        names = [conditions] if isinstance(conditions, str) else list(conditions)
        canonical = getattr(project, "canonical", None)
        trackers = (canonical.attrs.get("trackers") if canonical is not None else None) or {}
        roots = getattr(project, "roots", {}) or {}
        return calibration_for([roots[c] for c in names if c in roots],
                               {trackers.get(c, "utrack") for c in names})
    except PrecisionError as exc:
        raise ValueError(f"Per spot: {exc}") from exc


def per_spot_words(calibration) -> str:
    """The σ correction in words under Per spot: "per spot, σ = 3.26 × reported^0.763 nm (the built-in u-track
    calibration ...)"."""
    return (f"per spot, σ = {calibration.c_nm:.3g} × reported^{calibration.p:.3g} nm "
            f"({calibration.source})")


def _inputs(rows, calibration) -> tuple:
    """``(sequences, fit keywords)`` for one table of tracks: the step sequences, and under per-spot precision each
    step's own noise and the calibration it was made with (`hmm.step_sequences`), so every fit of the section -- the
    pooled one, Best's, each replicate's -- models the noise the same way."""
    from .hmm import step_sequences

    if calibration is None:
        return step_sequences(rows), {}
    sequences, noises = step_sequences(rows, calibration=calibration)
    return sequences, {"noise_sequences": noises, "calibration": calibration}


def sigma_note(project) -> str:
    """What the Run dialog says under "Tracker's sigma" when the tracker recorded no sigma: the theoretical one that
    would be used, by name (never applied silently); "" when one is recorded or none can be stated."""
    from .spatial_map_view import sigma_stamp

    stamp = sigma_stamp(project)
    nm = stamp.get("sigma_nm")
    if nm and str(stamp.get("sigma_provenance") or "") == THEORETICAL:
        return (f"This project records no localisation sigma. The tracker's sigma option would use a theoretical "
                f"{nm:.0f} nm computed from the optics, not measured.")
    if not nm:
        return "This project records no localisation sigma to correct with."
    return ""


def corrected(result: dict) -> bool:
    """Whether ``result``'s D are corrected for localisation error (Tracker's sigma, or Per spot), not apparent."""
    return (result.get("settings") or {}).get("sigma") == "tracker" or per_spot(result)


def per_spot(result: dict) -> bool:
    """Whether ``result`` was fitted with each step's own localisation error (Per spot) -- not a condition of a Per
    spot Run whose tracks carry no precision (``settings["precision_off"]``), fitted without the model."""
    settings = result.get("settings") or {}
    return settings.get("sigma") == "per_spot" and not settings.get("precision_off")


def d_word(result: dict) -> str:
    """How a result's D is labelled: "apparent", "σ-corrected" under the σ correction (found by the overnight
    tester, 4 Nov 2026: a corrected D of 0.004 was labelled "D apparent"), "per-spot" under Per spot."""
    return "per-spot" if per_spot(result) else "σ-corrected" if corrected(result) else "apparent"


def _fit_dict(fit) -> dict:
    import numpy as np

    rates = fit.rates_per_s
    dwell = fit.dwell_times_s
    initial = fit.initial_distribution
    return {
        "n_states": int(fit.n_states),
        "D": [float(d) for d in fit.D],
        "occupancy": [float(o) for o in fit.occupancy],
        "initial": None if initial is None else [float(v) for v in initial],
        "stationary": [float(v) for v in fit.stationary],
        "dwell_s": [None if not np.isfinite(d) else float(d) for d in dwell],
        "rates": [[float(r) for r in row] for row in rates],
        "transient": [bool(v) for v in fit.is_transient()],
        "degenerate": bool(fit.has_degenerate_chain),
        "absorbing": [bool(v) for v in fit.is_absorbing],
        "bic": float(fit.bic),
        "n_steps": int(fit.n_steps),
        "n_tracks": int(fit.n_sequences),
        "starts": (int(fit.n_converged_starts), int(fit.n_starts)),
    }


def compute_hmm_view(project, condition: str, *, states: str, uncertainty: str,
                     sigma: str, shared_d: str = "replicate", fixed_D=None, best_max: int = MAX_STATES,
                     chunks: int = 5, precision=None) -> dict:
    """One condition's HMM: the fit, its interval, the diagram and the words.

    ``shared_d`` and ``fixed_D`` are Force same D across (`hmm_run`, 3 Nov 2026): "condition" fits every replicate
    at the condition's pooled D, "replicate" each replicate at its own (the behaviour before), "group" / "global"
    the pooled fit AND every replicate at ``fixed_D``, the group's or set's D (`hmm_run.joint_D`). ``best_max`` is how
    far Best looks; ``chunks`` the pseudo-replicates of Uncertainty "chunk"; ``precision`` the Run's per-spot
    calibration ``(c_nm, p, source)`` under sigma "per_spot" (`hmm_run.with_precision`; without it the condition's
    own, `precision_of`).

    Picklable and run on the process pool. An expected refusal comes back as
    ``{"condition": ..., "error": "<Type>: <message>"}`` naming what refused; anything
    else raises."""
    from .hmm import HMMError, fit_hmm, select_n_states
    from .sqd import EVIDENCE_THRESHOLDS

    if states not in STATE_CHOICES:
        raise ValueError(f"states must be one of {STATE_CHOICES}, got {states!r}")
    if uncertainty not in UNCERTAINTY_MODES:
        raise ValueError(f"uncertainty must be one of {UNCERTAINTY_MODES}, got {uncertainty!r}")
    if sigma not in SIGMA_MODES:
        raise ValueError(f"sigma must be one of {SIGMA_MODES}, got {sigma!r}")
    if shared_d not in SHARED_D_SCOPES:
        raise ValueError(f"shared_d must be one of {SHARED_D_SCOPES}, got {shared_d!r}")
    if shared_d in ("group", "global") and fixed_D is None:
        raise ValueError(f"shared_d {shared_d!r} fits at the group's D, so fixed_D is required")
    settings = {"states": states, "uncertainty": uncertainty, "sigma": sigma, "shared_d": shared_d,
                "best_max": int(best_max), "chunks": int(chunks),
                "fixed_D": None if fixed_D is None else [float(d) for d in fixed_D],
                **({"precision": [float(precision[0]), float(precision[1]), str(precision[2])]}
                   if precision is not None else {})}
    try:
        part = project.condition(condition)
        tau = float(project.params.tau_s)
        from .precision import carries_precision

        if sigma == "per_spot" and not carries_precision(part):
            # NO PRECISION IN THESE TRACKS (a truth condition, a track table): fitted without the model, said
            calibration = None
            settings["precision_off"] = True
            sigma_um, sigma_words = 0.0, ("per spot: none recorded for this condition (its tracks carry no per-spot "
                                          "precision), so the error is not modelled")
        else:
            calibration = precision_of(project, condition, sigma, given=precision)
        if calibration is not None:
            # PER SPOT: no one sigma; each step's own noise travels with the sequences (`_inputs`)
            sigma_um, sigma_words = 0.0, per_spot_words(calibration)
        elif not settings.get("precision_off"):
            sigma_um, sigma_words = _sigma_um(project, sigma)
        sequences, noise = _inputs(part, calibration)
        noise_kw = {"localization_sigma_um": sigma_um, **noise}
        selection = forced_bic = None
        if fixed_D is not None:
            # THE GROUP'S (OR THE SET'S) D, HELD: this condition's transitions and occupancy at the shared states
            fit = fit_hmm(sequences, n_states=len(fixed_D), tau_s=tau, fixed_D=fixed_D, **noise_kw)
        elif states == "best":
            chosen = select_n_states(sequences, min_states=1, max_states=int(best_max), tau_s=tau, **noise_kw)
            best = min(f.bic for f in chosen.fits)
            rows = [(int(f.n_states), float(f.bic), float(f.bic - best)) for f in chosen.fits]
            fit = chosen.recommended
            runner_up = sorted(r[2] for r in rows)[1] if len(rows) > 1 else math.inf
            selection = {"rows": rows, "recommended": int(fit.n_states),
                         "margin": float(runner_up),
                         "inconclusive": bool(runner_up < EVIDENCE_THRESHOLDS[0][0])}
        elif int(states) >= 2:
            # A COUNT SET BY HAND, CHECKED AGAINST BEST (ruled 4 Nov 2026): Best's scores on the same steps, so the
            # card can say where it chooses fewer (`bubble_view.forced_count_warning`); the fit STAYS at the count
            # set. The fit at it IS Best's there (the same `fit_hmm` call), so its BIC is Best's score, and only the
            # counts below are fitted beside it -- the counts above only where one below wins
            # (`bubble_view.forced_count_scores`)
            from .bubble_view import forced_count_scores

            fit = fit_hmm(sequences, n_states=int(states), tau_s=tau, **noise_kw)
            forced_bic = forced_count_scores(
                int(states), range(1, max(int(best_max), int(states)) + 1),
                lambda lo, hi: [[int(f.n_states), float(f.bic)] for f in select_n_states(
                    sequences, min_states=lo, max_states=hi, tau_s=tau, **noise_kw).fits],
                own_bic=fit.bic)
        else:
            fit = fit_hmm(sequences, n_states=int(states), tau_s=tau, **noise_kw)
    except (HMMError, ValueError) as exc:
        return {"condition": condition, "settings": settings,
                "error": f"{type(exc).__name__}: {exc}"}

    result = {
        "condition": condition,
        "settings": settings,
        "tau_s": tau,
        "sigma_words": sigma_words,
        "selection": selection,
        **({"forced_bic": forced_bic} if forced_bic else {}),
        "fit": _fit_dict(fit),
        # each replicate (or chunk) at the condition's own D, the group's, or its own (Don't force anywhere)
        "interval": _interval(part, fit, uncertainty, tau, sigma_um,
                              hold=None if shared_d == "replicate" else list(fit.D),
                              boot_fixed=fixed_D, chunks=int(chunks), calibration=calibration),
    }
    result["diagram"] = diagram_payload(result)
    result["lines"] = card_lines(result)
    return result


def _day_name(day, word: str = "replicate") -> str:
    try:
        return f"{word} {int(day)}"
    except (TypeError, ValueError):
        return f"{word} {day}"


def _interval(part, fit, mode: str, tau: float, sigma_um: float, *, hold=None, boot_fixed=None,
              chunks: int = 5, calibration=None) -> dict | None:
    """The spread over independent units, per the Uncertainty setting. ``hold``: the D every replicate (or chunk) is
    fitted at (Force same D across Replicates, Conditions, Whole data set), None for each its own; ``boot_fixed``:
    the D a movie or cell resample is refitted at (a set-wide D), None to refit it; ``calibration``: per-spot
    precision, every replicate and resample fitted with its steps' own noise as the pooled fit was."""
    import numpy as np

    from .hmm import HMMError, bootstrap_hmm, fit_hmm

    if mode == "none":
        return None
    n = int(fit.n_states)
    word, pseudo = "replicate", None
    if mode == "chunk":
        # SQD'S OWN DEAL of this condition's movies (its cells when too few) into pseudo-replicates
        from .comparison import (ComparisonError, PSEUDO_SEED, chunks_withheld, pseudo_replicate_deal,
                                 pseudo_replicate_labels)

        # WITHHELD WHERE THE CONDITION HAS REPLICATES (SQD's rule, ruled 4 Nov 2026): the card says why, no spread
        held = chunks_withheld(part["day"].nunique()) if "day" in part.columns else None
        if held:
            return {"mode": "day", "unit": "chunk", "note": f"{held}.", "chunks_withheld": True}
        try:
            mapping, by = pseudo_replicate_deal(part, int(chunks))
        except ComparisonError as exc:
            return {"mode": "day", "unit": "chunk", "note": f"No chunks: {exc}"}
        part = part.copy()
        part["day"] = pseudo_replicate_labels(part, mapping, by)
        part = part[part["day"] > 0]
        word, pseudo = "chunk", {"k": int(max(mapping.values())), "by": by, "seed": PSEUDO_SEED}
        mode = "day"
    days = sorted(part["day"].dropna().unique()) if "day" in part.columns else []
    if mode == "day":
        if len(days) < 2:
            return {"mode": "day", "note": (
                f"{len(days)} replicate: there is no spread between replicates to show. Choose Movies "
                f"or Cells for an interval over this session's movies or cells.")}
        values, failed = [], []
        for day in days:
            rows = part[part["day"] == day]
            try:
                sequences, noise = _inputs(rows, calibration)
                one = fit_hmm(sequences, n_states=n, tau_s=tau,
                              localization_sigma_um=sigma_um, fixed_D=hold, **noise)
            except HMMError as exc:
                failed.append((_day_name(day, word), str(exc)))
                continue
            values.append({"day": _day_name(day, word), **_fit_dict(one)})
        out = {"mode": "day", "values": values, "failed": failed, "held_D": hold is not None}
        if pseudo is not None:
            out.update(unit="chunk", chunks=pseudo)
        if len(values) >= 2:
            occ = np.array([v["occupancy"] for v in values])
            out["occupancy_mean"] = [float(x) for x in occ.mean(axis=0)]
            out["occupancy_sd"] = [float(x) for x in occ.std(axis=0, ddof=1)]
            out["degenerate_days"] = [v["day"] for v in values if v["degenerate"]]
        return out
    # movies or cells, one session
    unit, words = mode, UNIT_WORDS[mode]
    if len(days) >= 2:
        return {"mode": unit, "withheld": (
            f"No {unit_word(unit)} interval: this condition holds {len(days)} replicates, {words} "
            f"within a replicate are not independent, so a {unit_word(unit)} interval would be too "
            f"narrow. Choose Replicates.")}
    try:
        boot = bootstrap_hmm(part, n_states=n, tau_s=tau, unit=unit, n_boot=N_BOOT,
                             localization_sigma_um=sigma_um, n_jobs=1, fixed_D=boot_fixed,
                             calibration=calibration)
    except HMMError as exc:
        return {"mode": unit, "withheld": f"No {unit_word(unit)} interval: {exc}"}

    def band(column):
        if column not in boot.columns:
            return None
        values = boot[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return None
        lo, hi = (float(v) for v in values.quantile([0.025, 0.975]))
        return (lo, hi)

    rates = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                rates[f"{i}>{j}"] = band(f"rate_{i + 1}_to_{j + 1}")
    return {"mode": unit, "n_units": int(boot.attrs["n_units"]),
            "n_boot": int(boot.attrs["n_ok"]), "n_degenerate": int(boot.attrs["n_degenerate"]),
            "occupancy": [band(f"occupancy_{i + 1}") for i in range(n)],
            "D": [band(f"D_{i + 1}") for i in range(n)],
            "rates": rates, **_exit_bands(boot, n), "caveat": boot.attrs.get("caveat", "")}


def _exit_bands(boot, n: int) -> dict:
    """Each state's RATE OUT (the sum of its exits, per resample) and MODEL DWELL (1 / that) as 95 % percentile
    bands over the resamples: ``{"rate_out": [(lo, hi) | None], "dwell": [(lo, hi) | None]}``, in /s and s.

    The dwell's band is the rate out's INVERTED, never a percentile of the resamples' dwells: 1 / x falls as x rises,
    so the 2.5th percentile of the dwell is 1 / the 97.5th of the rate out, and a resample with no exit (rate out 0,
    an infinite dwell) stays in the count where a percentile of dwells would have to drop it. A rate-out band reaching
    zero leaves the dwell with NO UPPER BOUND, ``(lo, None)``; one that is zero throughout, no band at all."""
    import numpy as np

    rate_out, dwell = [], []
    for i in range(n):
        columns = [f"rate_{i + 1}_to_{j + 1}" for j in range(n) if j != i]
        if not columns or not all(c in boot.columns for c in columns):
            rate_out.append(None)
            dwell.append(None)
            continue
        values = boot[columns].sum(axis=1, min_count=len(columns)).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            rate_out.append(None)
            dwell.append(None)
            continue
        lo, hi = (float(v) for v in values.quantile([0.025, 0.975]))
        rate_out.append((lo, hi))
        dwell.append(None if hi <= 0 else (1.0 / hi, 1.0 / lo if lo > 0 else None))
    return {"rate_out": rate_out, "dwell": dwell}


# --------------------------------------------------------------------------- #
# The words
# --------------------------------------------------------------------------- #


def _names(n: int) -> list[str]:
    return ["slow", "fast"] if n == 2 else ["slow", "middle", "fast"] if n == 3 else ["state 1"]


#: What a D on the per-spot fit's floor reads (`hmm.d_at_floor`, 10 Nov 2026): never the floor as a value.
NOT_RESOLVED = "not resolved"


def _d(value: float) -> str:
    return NOT_RESOLVED if _at_floor(value) else f"{value:.3g} µm²/s"


def _d3(value) -> str:
    """A D to 3 significant figures, or "not resolved" on the floor."""
    return NOT_RESOLVED if _at_floor(value) else f"{value:.3g}"


def _at_floor(value) -> bool:
    from .hmm import d_at_floor

    return d_at_floor(value)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def chunks_of(interval) -> dict | None:
    """The random chunks a spread is over (Uncertainty Chunks), in `bubble_view.chunk_words`' shape -- or None when
    the spread is over replicates. (Found by the overnight tester, 4 Nov 2026: chunks were called replicates on every
    card and caption of this section; SQD and the apparent-D bubbles say chunks.)"""
    interval = interval or {}
    if interval.get("unit") != "chunk":
        return None
    pseudo = dict(interval.get("chunks") or {})
    if not pseudo.get("k"):
        return None
    if isinstance(pseudo.get("by"), str):
        pseudo["by"] = {"": pseudo["by"]}                     # `pseudo_replicate_deal` names one unit per condition
    return pseudo


def unit_words(interval) -> tuple:
    """``(one, many)``: what a Replicates-mode spread is over, "replicate(s)" or "chunk(s)"."""
    chunked = (interval or {}).get("unit") == "chunk"
    return ("chunk", "chunks") if chunked else ("replicate", "replicates")


def fit_flags(result: dict) -> list:
    """SQD's and the apparent-D bubbles' fit warning on this HMM fit, by THEIR rule (`bubble_view.fit_warnings`,
    reused, not copied): a state holding under 5 % of the steps, or two neighbouring states whose D are within twofold
    -- each ending "fit one state fewer". Asked by the overnight tester (round 2, 4 Nov 2026): three states on
    two-state data (0.040 and 0.052 µm²/s, slow 6 %) got the "left within two frames" note but not this one. States
    are numbered from the slowest, as the populations are."""
    from .bubble_view import fit_warnings, forced_count_warning

    fit = result["fit"]
    points = [{"column": 0, "style_index": 0, "population": k + 1, "d": float(d), "fraction": float(occ)}
              for k, (d, occ) in enumerate(zip(fit["D"], fit["occupancy"]))]
    name = str(result.get("condition") or "This fit")
    flags = [w.replace("populations", "states").replace("population", "state")
             for w in fit_warnings(points, [name])]
    # a count set by hand where Best (BIC) chooses fewer (ruled 4 Nov 2026): said, the count kept
    forced = forced_count_warning(fit["n_states"], result.get("forced_bic"), "state")
    return flags + ([f"{name}: {forced}"] if forced else [])


def rate_text(rate: float, band=None, degenerate: int = 0) -> str:
    """``"7.7 /s [5.1, 11.2]"``; a band from resamples with no transition says so."""
    text = f"{rate:.3g} /s"
    if band is not None:
        text += f" [{band[0]:.3g}, {band[1]:.3g}]"
        if degenerate:
            text += " (not identifiable)"
    return text


def card_caption(result: dict) -> str:
    fit = result["fit"]
    return (f"{fit['n_states']} states · {fit['n_tracks']:,} tracks · {fit['n_steps']:,} "
            f"steps · τ {1000 * result['tau_s']:.1f} ms · σ correction {result['sigma_words']}"
            + sharing_words(result))


def sharing_words(result: dict) -> str:
    """The card caption's words for Force same D across (3 Nov 2026), "" for a result from before (each replicate its
    own D, never said then): " · same D across replicates", " · same D across the whole data set" ..."""
    from .workspace_view import scope_words

    shared = (result.get("settings") or {}).get("shared_d")
    if not shared:
        return ""
    words = scope_words(shared)
    if (result.get("interval") or {}).get("unit") == "chunk":
        words = words.replace("replicate", "chunk")         # "same D across chunks", "each chunk its own"
    return f" · {words}"


def card_lines(result: dict) -> list:
    """The card's sentences, ``[(text, kind)]`` with kind "head", "line" or "flag"."""
    fit = result["fit"]
    n = fit["n_states"]
    names = _names(n)
    interval = result.get("interval") or {}
    degenerate = interval.get("n_degenerate", 0) if _resampled(interval) else 0
    lines = []
    if n >= 2:
        lines.append(("Switching rates (per second)", "head"))
        bands = interval.get("rates", {}) if _resampled(interval) else {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                lines.append((f"{names[i]} → {names[j]}: "
                              + rate_text(fit["rates"][i][j], bands.get(f"{i}>{j}"),
                                          degenerate), "line"))
        if fit["degenerate"]:
            lines.append(("The fitted chain never leaves a state: no transition out of it "
                          "was observed, so its rates and dwell are not measured.", "flag"))
        if degenerate:
            lines.append((f"{degenerate} of {interval['n_boot']} {interval['mode']} resamples "
                          f"held no transition out of a state, so the rates are not identifiable. "
                          f"Quote the occupancy instead of a rate or a dwell.", "flag"))
    lines.append(("States", "head"))
    for i in range(n):
        occ = f"occupancy {_pct(fit['occupancy'][i])}"
        if _resampled(interval) and interval.get("occupancy"):
            band = interval["occupancy"][i]
            if band:
                occ += f" [{_pct(band[0])}, {_pct(band[1])}]"
        if interval.get("mode") == "day" and interval.get("occupancy_mean"):
            occ += (f" ({unit_words(interval)[1]} {_pct(interval['occupancy_mean'][i])} ± "
                    f"{_pct(interval['occupancy_sd'][i])})")
        lines.append((f"{names[i]}: D {_d(fit['D'][i])} ({d_word(result)}) · {occ}", "line"))
    if fit["initial"] is not None and n >= 2:
        lines.append(("Occupancy is the time average over the observed steps. A track of 5-6 "
                      "steps does not mix, so it lies between the share at the start of a track (π "
                      + " / ".join(_pct(v) for v in fit["initial"]) + ") and the stationary "
                      "share (" + " / ".join(_pct(v) for v in fit["stationary"]) + ").", "line"))
    if n >= 2:
        dwell = " · ".join(f"{names[i]} {'never left' if d is None else f'{1000 * d:.0f} ms'}"
                           for i, d in enumerate(fit["dwell_s"]))
        lines.append((f"Dwell times from the model (exponential, Markovian; usually longer "
                      f"than the tracks): {dwell}", "line"))
        transient = [names[i] for i, v in enumerate(fit["transient"]) if v]
        if transient:
            lines.append((f"{', '.join(transient)}: left within two frames, so these are molecules "
                          f"caught mid-switch and may not be a population.", "flag"))
    for warning in fit_flags(result):
        lines.append((f"Check the fit: {warning}.", "flag"))
    selection = result.get("selection")
    if selection:
        lines.append(("State count (BIC recommends a count; the choice is yours)", "head"))
        lines.append((" · ".join(f"{k}: ΔBIC {d:.1f}" for k, _b, d in selection["rows"]),
                      "line"))
        if selection["inconclusive"]:
            lines.append((f"Inconclusive: the runner-up is within {selection['margin']:.1f} "
                          f"of the best (under 2).", "flag"))
        if selection["recommended"] >= 3:
            lines.append(("A third state's D often sits at the localisation floor, so it may reflect "
                          "variation in localisation error and not a separate population.", "line"))
    if interval.get("mode") == "day":
        if interval.get("note"):
            lines.append((interval["note"], "line"))
        for v in interval.get("values", []):
            rates = (" · ".join(f"{names[i]}→{names[j]} {v['rates'][i][j]:.3g} /s"
                                for i in range(n) for j in range(n) if i != j)
                     if n >= 2 else "")
            lines.append((f"{v['day']}: " + " / ".join(_pct(o) for o in v["occupancy"])
                          + (f" · {rates}" if rates else ""), "line"))
        for day, why in interval.get("failed", []):
            lines.append((f"{day}: no fit ({why})", "flag"))
        if interval.get("degenerate_days"):
            lines.append((f"{', '.join(interval['degenerate_days'])}: no transition out of a "
                          f"state, so that {unit_words(interval)[0]}'s rates are not measured.", "flag"))
    if _resampled(interval):
        if interval.get("withheld"):
            lines.append((interval["withheld"], "line"))
        else:
            lines.append((f"Intervals: 95 % percentile over {interval['n_boot']} resamples of "
                          f"{interval['n_units']} {UNIT_WORDS[interval['mode']]}."
                          + (f" {interval['caveat']}" if interval.get("caveat") else ""),
                          "line"))
    return lines


def diagram_payload(result: dict) -> dict:
    """What the card's state diagram paints: one node per state (its occupancy, D and
    name) and one arrow per pair of states (both rates). Plain values only."""
    fit = result["fit"]
    n = fit["n_states"]
    names = _names(n)
    interval = result.get("interval") or {}
    bands = interval.get("rates", {}) if _resampled(interval) else {}
    nodes = [{"name": names[i], "D": fit["D"][i], "occupancy": fit["occupancy"][i],
              "label": f"{names[i]}\nD {_d3(fit['D'][i])}\n{_pct(fit['occupancy'][i])}"}
             for i in range(n)]
    arrows = [{"from": i, "to": j, "rate": fit["rates"][i][j],
               "text": rate_text(fit["rates"][i][j], bands.get(f"{i}>{j}"))}
              for i in range(n) for j in range(n) if i != j]
    return {"nodes": nodes, "arrows": arrows,
            "not_identifiable": bool(interval.get("n_degenerate")) or fit["degenerate"]}


# --------------------------------------------------------------------------- #
# The page: the comparison across conditions, the compact card, the notes said once
# --------------------------------------------------------------------------- #

#: The comparison strip's panels: ``(key, title, unit)``, the manuscript's four.
COMPARE_PANELS = (
    ("occupancy", "Slow-state occupancy", "%"),
    ("rate_in", "Rate into the slow state", "/s"),
    ("rate_out", "Rate out of the slow state", "/s"),
    ("D", "Slow-state D (apparent)", "µm²/s"),
)


#: The OPTIONAL FIFTH PANEL (asked 12 Nov 2026: "can we get for hmm a dwell time and rate +- error. also for export
#: option to export dwell time as well"): the state's model dwell, 1 / its rate out, in ms, drawn after the rate out
#: it comes from. Off unless asked: the strip's "Dwell time" box (`comparison_payload(dwell=True)`), the export's own
#: (`figure_export.FigureSpec.hmm_dwell`).
DWELL_PANEL = ("dwell", "Dwell in the slow state (model)", "ms")

#: What a panel's numbers are multiplied by to be drawn in its unit (the fit holds shares and seconds).
_SCALE = {"%": 100.0, "ms": 1000.0}

#: The states a page can be switched to, by D order (slow = lowest D).
STATES = ("slow", "middle", "fast")


def state_index(n_states: int, state: str) -> int | None:
    """Where ``state`` sits in a fit of ``n_states`` (states are ordered by D), or None when
    that fit has no such state: two states are slow and fast, with no middle; one state is
    only "slow". The one place this matching is decided."""
    n = int(n_states)
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    if n >= 3:
        return STATES.index(state) if n == 3 else {"slow": 0, "middle": 1, "fast": n - 1}[state]
    if n == 2:
        return {"slow": 0, "fast": 1}.get(state)
    return 0 if state == "slow" else None


def merged_note(n_states: int, state: str) -> str | None:
    """Why a state of a 2-state fit is not the same population as a 3-state fit's: it merges
    two of them. None where nothing is merged."""
    if int(n_states) != 2 or state == "middle":
        return None
    return ("2 states: merges a 3-state fit's slow + middle" if state == "slow"
            else "2 states: merges a 3-state fit's middle + fast")


def missing_note(n_states: int, state: str) -> str:
    return f"no {state} state ({int(n_states)} state{'' if int(n_states) == 1 else 's'})"


def state_numbers(fit: dict, s: int) -> dict:
    """State ``s``'s four numbers from one fit dict (`_fit_dict`), and its model dwell.

    Out of it: the sum of its rates to every other state, the inverse of its model dwell.
    Into it: the rate at which a molecule NOT in it enters it, ``sum_i st_i k(i->s) /
    sum_i st_i`` over the other states i, weighted by the fitted chain's stationary
    distribution ``st``. For two states both are exactly the fitted rates; one state has
    none."""
    n = int(fit["n_states"])
    s = int(s)
    out = {"occupancy": float(fit["occupancy"][s]), "D": float(fit["D"][s]),
           "rate_in": None, "rate_out": None, "dwell": None}
    if n < 2:
        return out
    k, st = fit["rates"], fit["stationary"]
    others = [i for i in range(n) if i != s]
    out["rate_out"] = float(sum(k[s][j] for j in others))
    if out["rate_out"] > 0:
        out["dwell"] = 1.0 / out["rate_out"]
    weight = sum(st[i] for i in others)
    if weight > 0:
        out["rate_in"] = float(sum(st[i] * k[i][s] for i in others) / weight)
    return out


def slow_numbers(fit: dict) -> dict:
    """The slow state's four numbers (`state_numbers` at the first, lowest-D state)."""
    out = state_numbers(fit, 0)
    out.pop("dwell")
    return out


def _band_of(interval: dict, key: str, n: int, s: int = 0):
    """A movie or cell interval for one of state ``s``'s four numbers, where the bootstrap holds
    it directly: occupancy and D always, the rates only for two states (where they ARE fitted
    rates)."""
    if not _resampled(interval) or interval.get("withheld"):
        return None
    if key == "dwell":
        # the rate out's band inverted (`_exit_bands`); none where it has no upper bound, said on the card instead
        band = ((interval.get("dwell") or []) + [None] * (s + 1))[s]
        return None if band is None or band[0] is None or band[1] is None else (band[0], band[1])
    if key == "rate_out" and n >= 3:
        return ((interval.get("rate_out") or []) + [None] * (s + 1))[s]
    if key == "occupancy":
        return ((interval.get("occupancy") or []) + [None] * (s + 1))[s]
    if key == "D":
        return ((interval.get("D") or []) + [None] * (s + 1))[s]
    if n == 2 and key in ("rate_in", "rate_out"):
        other = 1 - s
        return (interval.get("rates") or {}).get(
            f"{other}>{s}" if key == "rate_in" else f"{s}>{other}")
    return None


def spread_line(interval: dict | None) -> str | None:
    """What a result's spread is over, in one sentence (the card's spread line, and the
    comparison export's per-condition note), or None when none was computed."""
    interval = interval or {}
    if interval.get("mode") == "day":
        pseudo = chunks_of(interval)
        if pseudo and not interval.get("note"):
            from .bubble_view import chunk_words

            return f"Spread over {chunk_words(pseudo)}."
        return interval.get("note") or (
            f"Spread over {len(interval.get('values', ()))} replicates.")
    if _resampled(interval):
        return interval.get("withheld") or (
            f"Intervals: 95 % percentile over {interval['n_boot']} resamples of "
            f"{interval['n_units']} {UNIT_WORDS[interval['mode']]}.")
    return None


def compare_panels(state: str = "slow", sigma_corrected: bool = False, dwell: bool = False) -> tuple:
    """The strip's four panels for ``state``; "slow" gives `COMPARE_PANELS` exactly. ``sigma_corrected``: every
    condition's D is corrected for localisation error, and the D panel says so rather than "apparent". ``dwell``:
    the fifth, `DWELL_PANEL`, after the rate out."""
    cap = state[0].upper() + state[1:]
    panels = COMPARE_PANELS if state == "slow" else (
        ("occupancy", f"{cap}-state occupancy", "%"),
        ("rate_in", f"Rate into the {state} state", "/s"),
        ("rate_out", f"Rate out of the {state} state", "/s"),
        ("D", f"{cap}-state D (apparent)", "µm²/s"))
    if sigma_corrected:
        word = "per-spot" if sigma_corrected == "per_spot" else "σ-corrected"
        panels = tuple((k, title.replace("(apparent)", f"({word})"), unit) for k, title, unit in panels)
    if dwell:
        title = DWELL_PANEL[1] if state == "slow" else f"Dwell in the {state} state (model)"
        at = [k for k, _t, _u in panels].index("rate_out") + 1
        panels = panels[:at] + (("dwell", title, DWELL_PANEL[2]),) + panels[at:]
    return panels


def replicate_number(label):
    """The replicate number in a per-replicate fit's name (`_day_name`: "replicate 2" -> 2), or None for a chunk or a
    name without one: what a pair tag from the Groups board is keyed by."""
    word, _, number = str(label).rpartition(" ")
    try:
        return int(number) if word == "replicate" else None
    except ValueError:
        return None


def comparison_payload(entries: list, state: str = "slow", grouping=None, pairs=None, dwell: bool = False) -> dict:
    """The strip across conditions: ``entries`` is ``[(key, name, colour, result)]`` in the
    page's order, results with an error left out. Returns ``{"panels": [{"key", "title",
    "unit", "rows": [{"key", "name", "colour", "n_states", "value", "days", "day_labels",
    "mean", "sd", "band", "band_unit"}]}], "conditions": [...], "caption"}``. ``days`` are the per-day
    values (open circles) with their mean and SD, ``day_labels`` each one's replicate day
    as the condition names it (source data only: drawn, the days carry no order); ``band``
    a 95 % movie or cell interval; ``value`` the pooled fit. A percentage is in percent.
    ``conditions`` says per condition what the numbers came from -- ``{"key", "name",
    "colour", "n_states", "spread", "failed", "sigma", "n_tracks", "n_steps", "tau_s"}`` --
    for the exported figure's source data and legend (`figure_export`'s
    "hmm_comparison", which draws from THIS payload and recomputes nothing).

    ``grouping`` ``(factors, nesting)`` (folder designs, `factor_layout`): the conditions in the nesting's order, and
    ``"layout"`` their grouped columns (`factor_layout.columns`), which the chart and its export draw; without it the
    layout is empty and every column stands as before.

    ``pairs`` ``{condition key: {replicate number: tag}}``: the replicates declared as pairs on the Groups board
    (`project.Project.replicate_pairs`, 5 Nov 2026). Each row then carries ``"day_numbers"`` (each value's replicate
    number, None for a chunk) and ``"day_tags"`` (each value's tag, "" without), and ``"paired"`` says whether any tag
    is shared by two conditions drawn -- what the pair lines join and the paired test reads; nothing is drawn or
    tested from them unless asked. ``"held_D"`` on a row: its replicates' D were held (Force same D across), so its D
    panel has no spread between replicates.

    ``dwell`` (12 Nov 2026): draw the fifth panel, the state's model dwell in ms (`DWELL_PANEL`). It is computed
    either way and kept as ``"dwell_panel"``, so an export can draw it on its own box (`FigureSpec.hmm_dwell`); its
    rows carry ``"never_left"``, the replicates whose state was never left (no dwell: left out of the mean, named)."""
    factors, nesting = grouping if grouping else ({}, ())
    if factors and len(tuple(nesting or ())) >= 2:
        from .factor_layout import nested

        order = nested([e[0] for e in entries], factors, tuple(nesting))
        entries = sorted(entries, key=lambda e: order.index(e[0]))
    fitted = [e[3] for e in entries if e[3] and "fit" in e[3]]
    panels = compare_panels(state, sigma_corrected=(
        "per_spot" if fitted and all(per_spot(r) for r in fitted)
        else bool(fitted) and all(corrected(r) for r in fitted)), dwell=True)
    rows_by_panel = {key: [] for key, _t, _u in panels}
    modes, counts = set(), {}
    conditions = []
    missing = []
    for key, name, colour, result in entries:
        if not result or "fit" not in result:
            continue
        fit = result["fit"]
        n = int(fit["n_states"])
        counts.setdefault(n, []).append(name)
        interval = result.get("interval") or {}
        conditions.append({
            "key": key, "name": name, "colour": colour, "n_states": n,
            "spread": spread_line(interval),
            "failed": [list(f) for f in interval.get("failed", ())]
            if interval.get("mode") == "day" else [],
            "sigma": result.get("sigma_words"), "n_tracks": fit.get("n_tracks"),
            "n_steps": fit.get("n_steps"), "tau_s": result.get("tau_s")})
        s = state_index(n, state)
        if s is None:
            # A fit without this state: a row that says so, never a neighbour's number.
            missing.append(name)
            for panel, _title, _unit in panels:
                rows_by_panel[panel].append({
                    "key": key, "name": name, "colour": colour, "n_states": n, "value": None,
                    "days": [], "day_labels": [], "mean": None, "sd": None, "band": None,
                    "missing": missing_note(n, state)})
            continue
        pooled = state_numbers(fit, s)
        per_day = [(str(v.get("day", f"replicate {i + 1}")), state_numbers(v, s))
                   for i, v in enumerate(interval.get("values", ()))
                   if int(v.get("n_states", n)) > s] \
            if interval.get("mode") == "day" else []
        chunked = bool(chunks_of(interval))
        tags = {} if chunked else {int(d): str(t) for d, t in ((pairs or {}).get(key) or {}).items()}
        if len(per_day) >= 2:
            modes.add("chunk" if chunked else "day")
        elif _band_of(interval, "occupancy", n, s) is not None:
            modes.add(interval["mode"])
        else:
            modes.add("none")
        for panel, _title, unit in panels:
            scale = _SCALE.get(unit, 1.0)
            value = pooled[panel]
            kept = [(label, d[panel] * scale) for label, d in per_day if d[panel] is not None]
            days = [v for _label, v in kept]
            numbers = [None if chunked else replicate_number(label) for label, _v in kept]
            row = {"key": key, "name": name, "colour": colour, "n_states": n,
                   "value": None if value is None else value * scale,
                   "unit": unit_words(interval),
                   "days": days if len(days) >= 2 else [],
                   "day_labels": [label for label, _v in kept] if len(days) >= 2 else [],
                   "day_numbers": numbers if len(days) >= 2 else [],
                   "day_tags": [tags.get(k, "") if k is not None else "" for k in numbers] if len(days) >= 2 else [],
                   "held_D": bool(interval.get("held_D")),
                   "mean": None, "sd": None, "band": None}
            if panel == "dwell":
                # a replicate whose state was never left has no dwell: left out of the mean, and named
                row["never_left"] = [label for label, d in per_day if d["dwell"] is None and n >= 2]
            if len(days) >= 2:
                mean = sum(days) / len(days)
                row["mean"] = mean
                row["sd"] = math.sqrt(sum((d - mean) ** 2 for d in days) / (len(days) - 1))
            band = _band_of(interval, panel, n, s)
            if band is not None and not row["days"]:
                row["band"] = (band[0] * scale, band[1] * scale)
                row["band_unit"] = interval["mode"]                  # "movie" or "cell"
            rows_by_panel[panel].append(row)
    words = []
    if "day" in modes:
        words.append("Open circles: replicates, in no particular order; filled: their mean; "
                     "bar: ±1 SD between replicates (a spread, not a confidence interval)")
    if "chunk" in modes:
        from .bubble_view import chunk_words

        pseudo = next(chunks_of(e[3].get("interval")) for e in entries
                      if e[3] and "fit" in e[3] and chunks_of(e[3].get("interval")))
        words.append(f"Open circles: chunks, {chunk_words(pseudo)}; filled: their mean; bar: ±1 SD over the chunks "
                     "(a spread, not a confidence interval)")
    for unit in RESAMPLED_UNITS:
        if unit in modes:
            words.append(f"bar: 95 % percentile interval over {unit_word(unit)} resamples (a single session)")
    if "none" in modes:
        words.append("a lone point: the pooled fit, with no spread")
    if len(counts) > 1:
        words.append("state counts differ (" + "; ".join(
            f"{n} state{'' if n == 1 else 's'}: {', '.join(names)}"
            for n, names in sorted(counts.items()))
            + f"): each condition's {state} state is taken from its own fit")
    elif any(n >= 3 for n in counts):
        words.append("with three states, the rates to or from the other two states are summed")
    if missing:
        words.append(f"no {state} state in {', '.join(missing)} (fitted with fewer states)")
    if dwell:
        words.append(f"dwell: the model's estimate, 1 / the rate out of the {state} state; usually longer than the "
                     "tracks")
        for row in rows_by_panel["dwell"]:
            if row.get("never_left"):
                words.append(f"{row['name']}: {', '.join(row['never_left'])} never left the {state} state, so no "
                             "dwell; left out of its mean")
    held = [name for _key, name, _colour, result in entries
            if result and "fit" in result and (result.get("interval") or {}).get("chunks_withheld")]
    if held:
        from .comparison import CHUNKS_WITHHELD

        words.append(f"{', '.join(held)}: {CHUNKS_WITHHELD}, drawn as the pooled fit")
    for key, name, _colour, result in entries:
        if result and "fit" in result:
            words += [f"check the fit: {w}" for w in fit_flags(dict(result, condition=name))]
    words.append("no statistic between conditions is drawn")
    layout = {}
    if factors and len(tuple(nesting or ())) >= 2:
        from .factor_layout import columns

        drawn = [c["key"] for c in conditions]
        layout = columns(drawn, factors, tuple(nesting))
        if not layout.get("groups"):
            layout = {}
    # A TAG TWO DRAWN CONDITIONS SHARE: a pair the lines can join and the paired test can read
    users: dict = {}
    for row in rows_by_panel[panels[0][0]] if panels else ():
        for tag in set(t for t in row.get("day_tags") or () if t):
            users.setdefault(tag, set()).add(row["key"])
    dwell_panel = next(({"key": key, "title": title, "unit": unit, "rows": rows_by_panel[key]}
                        for key, title, unit in panels if key == "dwell"), None)
    return {"panels": [{"key": key, "title": title, "unit": unit, "rows": rows_by_panel[key]}
                       for key, title, unit in panels if dwell or key != "dwell"],
            "dwell_panel": dwell_panel, "dwell": bool(dwell),
            "conditions": conditions, "layout": layout,
            "paired": any(len(keys) >= 2 for keys in users.values()),
            "caption": ". ".join(w[0].upper() + w[1:] for w in words) + "."}


#: What the pair lines are, in the caption on screen and the exported key (5 Nov 2026).
PAIR_LINES_WORDS = "lines join replicates declared as pairs on the Groups board (one culture or session)"


def pair_lines(rows: list) -> list:
    """The PAIR LINES of one panel (asked 5 Nov 2026: "lines joining paired replicates ... off by default"): for each
    tag two or more of ``rows`` (the panel's conditions, in drawn order) share, ``(tag, [(column, value index), ...])``
    through every column holding it, left to right. Display only: nothing is tested from it. A row without
    per-replicate values (a pooled fit, a chunk) holds no tag."""
    found: dict = {}
    for k, row in enumerate(rows):
        for j, tag in enumerate(row.get("day_tags") or ()):
            if tag and j < len(row.get("days") or ()):
                found.setdefault(tag, []).append((k, j))
    return [(tag, points) for tag, points in sorted(found.items()) if len({k for k, _j in points}) >= 2]


# --------------------------------------------------------------------------- #
# The redesigned page (the Test page): the deck of cards and the comparison area
# --------------------------------------------------------------------------- #

#: The rows of the comparison table, and how each is shown.
PAIR_ROWS = (("occupancy", "Occupancy", "%", 100.0),
             ("rate_in", "Rate in", "/s", 1.0),
             ("rate_out", "Rate out", "/s", 1.0),
             ("dwell", "Dwell (model)", "ms", 1000.0),
             ("D", "D (apparent)", "µm²/s", 1.0))

#: Beside every interval between conditions (Rule 0, CLAUDE.md).
RULE0_NOTE = ("Replicates are independent, so the interval is Welch's for two independent samples; replicate n "
              "of one condition is not paired with replicate n of another. It is narrower than a paired interval "
              "because it has more degrees of freedom, which favours the effect. Shown: mean ± SD over the "
              "replicates.")


#: The pair comparison's note when its sides are random chunks (Uncertainty Chunks), not replicates.
CHUNK_NOTE = ("The values are random chunks of one culture (pseudo-replicates), not replicates. The interval shows how "
              "precisely these cells were measured, and says nothing about another culture. It is Welch's interval "
              "for two independent samples; shown: mean ± SD over the chunks.")


def _fmt(value, unit: str) -> str:
    if value is None:
        return "–"
    if unit == "%":
        return f"{value:.0f} %" if value >= 10 else f"{value:.1f} %"
    if unit == "ms":
        return f"{value:.0f} ms"
    if unit == "µm²/s":
        if _at_floor(value):
            return NOT_RESOLVED
        return f"{value:.3f}" if value < 0.1 else f"{value:.2f}"
    return f"{value:.2f} /s" if value < 1 else f"{value:.1f} /s"


def comparison_tip(panel: dict, row: dict) -> dict:
    """What the strip's tooltip says for one condition in one panel (the UI designer, 21 Oct 2026):
    ``{"heading", "rows": [(name, value)], "colour"}``. Exactly what that column draws -- the
    replicate days' mean, +-1 SD (the day-to-day spread, as the caption says, not a confidence
    interval) and the days themselves, low to high since they carry no order; or the pooled
    fit and its 95 % interval over movies or cells; or the pooled fit alone; or why a fit has
    no such state. Numbers to 3 significant figures, a percentage to one decimal."""
    unit = panel["unit"]

    def num(v) -> str:
        if v is None:
            return "–"
        if unit == "µm²/s" and _at_floor(v):
            return NOT_RESOLVED
        return f"{v:.1f}" if unit == "%" else f"{v:.3g}"

    def with_unit(text: str) -> str:
        return f"{text} {unit}" if text != "–" else text

    rows = []
    if row.get("missing"):
        rows.append(("No value", row["missing"]))
    elif row["days"]:
        days = sorted(row["days"])
        many = (row.get("unit") or ("replicate", "replicates"))[1]
        rows.append((f"Mean of {len(days)} {many}", with_unit(num(row["mean"]))))
        rows.append(("±1 SD", with_unit(num(row["sd"]))))
        rows.append((f"{many[0].upper()}{many[1:]}, low to high", ", ".join(num(d) for d in days)))
        if row["value"] is not None:
            rows.append(("Pooled fit", with_unit(num(row["value"]))))
    else:
        rows.append(("Pooled fit", with_unit(num(row["value"]))))
        if row.get("band") is not None:
            lo, hi = row["band"]
            rows.append((f"95 % interval ({row.get('band_unit', 'resampled')}s)",
                         with_unit(f"{num(lo)}–{num(hi)}")))
    return {"heading": row["name"], "rows": rows, "colour": row.get("colour")}


#: The deck card's states table (mockups 21.2 + 22.2): ``(key, header, group)``. Two groups by
#: MEANING: what a state is (D, its model dwell) and three shares of the molecules (at track
#: start, on average, in the long run), each share column summing to 100 %.
STATE_TABLE_COLUMNS = (("D", "D µm²/s", "per state"), ("dwell", "dwell ms", "per state"),
                       ("start", "start", "share of molecules, %"),
                       ("average", "average", "share of molecules, %"),
                       ("longrun", "long run", "share of molecules, %"))


#: The Test page's Table view (mockups 24-27): one state's numbers, a row per condition.
CONDITION_TABLE_COLUMNS = (("D", "D µm²/s", "state"), ("dwell", "dwell ms", "state"),
                           ("start", "start %", "state"), ("average", "average %", "state"),
                           ("longrun", "long run %", "state"),
                           ("rate_in", "rate in /s", "rates"), ("rate_out", "rate out /s", "rates"))

#: Under the HMM section's Table page (22 Nov 2026): what its columns are.
CONDITION_TABLE_NOTE = ("One state in every condition. Dwell is the model's (1 / rate out); start, average and long "
                        "run are the state's share at a track's first step, over all steps, and at equilibrium.")


def condition_table(entries, state: str) -> dict:
    """``state`` in every condition, a row each (``entries`` = ``(key, label, colour, result)``):
    D, the model dwell, the three shares and the rates into and out of the state, formatted as
    the card's table formats them. A fit without the state has "—" and says why. ``range``:
    each column's lowest and highest value over the rows that have it (the bars under the
    numbers)."""
    rows = []
    for key, name, colour, result in entries:
        fit = result["fit"]
        n = int(fit["n_states"])
        s = state_index(n, state)
        values = {}
        if s is not None:
            nums = state_numbers(fit, s)
            dwell = nums["dwell"]
            raw = {"D": float(fit["D"][s]), "dwell": None if dwell is None else 1000 * dwell,
                   "start": 100 * float(fit["initial"][s]),
                   "average": 100 * float(fit["occupancy"][s]),
                   "longrun": 100 * float(fit["stationary"][s]),
                   "rate_in": nums["rate_in"], "rate_out": nums["rate_out"]}
            for k, v in raw.items():
                if v is None:
                    values[k] = ("–", None)
                elif k == "D":
                    values[k] = (_fmt(v, "µm²/s"), v)
                elif k == "dwell":
                    values[k] = (f"{v:.0f}", v)
                elif k.startswith("rate"):
                    values[k] = (_fmt(v, "/s").replace(" /s", ""), float(v))
                else:
                    values[k] = (f"{v:.1f}", v)
        rows.append({"key": key, "name": name, "colour": colour, "n_states": n,
                     "missing": None if s is not None else missing_note(n, state),
                     "values": values})
    ranges = {}
    for k, _h, _g in CONDITION_TABLE_COLUMNS:
        vals = [r["values"][k][1] for r in rows if r["values"] and r["values"][k][1] is not None]
        ranges[k] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return {"columns": [(k, h, g) for k, h, g in CONDITION_TABLE_COLUMNS], "rows": rows,
            "range": ranges}


def _day_values(result: dict, s: int, key: str, scale: float) -> list[float]:
    """State ``s``'s ``key`` on each fitted replicate day (Days mode), scaled."""
    interval = result.get("interval") or {}
    if interval.get("mode") != "day":
        return []
    out = []
    for v in interval.get("values", ()):
        if int(v.get("n_states", 0)) <= s:
            continue
        x = state_numbers(v, s)[key]
        if x is not None:
            out.append(float(x) * scale)
    return out


def _mean_sd(values: list[float]):
    if len(values) < 2:
        return None, None
    m = sum(values) / len(values)
    return m, math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def pair_comparison(a: dict, b: dict, state: str) -> dict:
    """Two conditions, ``b`` against ``a`` (as "b − a"), for ``state``: both rate matrices,
    their difference (only when the two fits have the same states), and per quantity both
    sides' mean ± SD over their days, the difference of the means and, in Days mode with two
    or more days on each side, the Welch 95 % interval (`uncertainty.two_sample_interval`,
    UNPAIRED -- Rule 0). Elsewhere the difference alone, and why."""
    from .uncertainty import two_sample_interval

    fa, fb = a["fit"], b["fit"]
    na, nb = int(fa["n_states"]), int(fb["n_states"])
    sa, sb = state_index(na, state), state_index(nb, state)
    out = {"a": {"rates": fa["rates"], "names": _names(na), "highlight": sa},
           "b": {"rates": fb["rates"], "names": _names(nb), "highlight": sb},
           "diff": None, "rows": [], "note": RULE0_NOTE, "reason": None}
    if chunks_of(a.get("interval")) or chunks_of(b.get("interval")):
        out["note"] = CHUNK_NOTE
    out["warnings"] = []
    if na == nb:
        out["diff"] = [[None if i == j else fb["rates"][i][j] - fa["rates"][i][j]
                        for j in range(na)] for i in range(na)]
    else:
        out["reason"] = (f"No difference matrix: the fits have different states ({na} and {nb}).")
        # the same word, not the same population: say which side's state is a merger
        for res, n in ((a, na), (b, nb)):
            note = merged_note(n, state)
            if note:
                out["warnings"].append(
                    f"{res.get('condition', 'One condition')}: {note}, so its {state} state is "
                    f"not the same population as in a {max(na, nb)}-state fit.")
    if sa is None or sb is None:
        out["reason"] = (out["reason"] or "") + f" No {state} state in " + \
            (a.get("condition", "one") if sa is None else b.get("condition", "one")) + "."
        out["warnings"].append(out["reason"].strip())
        return out
    pa, pb = state_numbers(fa, sa), state_numbers(fb, sb)
    for key, name, unit, scale in PAIR_ROWS:
        da, db = _day_values(a, sa, key, scale), _day_values(b, sb, key, scale)
        ma, sda = _mean_sd(da)
        mb, sdb = _mean_sd(db)
        va = ma if ma is not None else (None if pa[key] is None else pa[key] * scale)
        vb = mb if mb is not None else (None if pb[key] is None else pb[key] * scale)
        if key == "D" and corrected(a) and corrected(b):
            name = name.replace("(apparent)", "(per-spot)" if per_spot(a) and per_spot(b) else "(σ-corrected)")
        row = {"key": key, "name": name, "unit": unit,
               "a": _fmt(va, unit) + ("" if sda is None else f" ± {_fmt(sda, unit).split(' ')[0]}"),
               "b": _fmt(vb, unit) + ("" if sdb is None else f" ± {_fmt(sdb, unit).split(' ')[0]}"),
               "diff": None, "interval": None, "why": None}
        if va is not None and vb is not None:
            d = vb - va
            sign = "+" if d >= 0 else "−"
            row["diff"] = sign + _fmt(abs(d), unit).replace(" %", " pp")
        if len(da) >= 2 and len(db) >= 2:
            diff, lo, hi, df, _na, _nb = two_sample_interval(da, db)
            if lo == lo and hi == hi:                    # not nan
                f = (lambda x: f"{x:+.1f}") if unit in ("%", "ms") else \
                    (lambda x: f"{x:+.3f}") if unit == "µm²/s" else (lambda x: f"{x:+.2f}")
                row["interval"] = f"[{f(lo)}, {f(hi)}] · df {df:.1f}".replace("-", "−")
        else:
            many = unit_words(a.get("interval") if (a.get("interval") or {}).get("mode") == "day"
                              else b.get("interval"))[1]
            row["why"] = (f"no interval: fewer than two {many} on a side"
                          if (a.get("interval") or {}).get("mode") == "day"
                          or (b.get("interval") or {}).get("mode") == "day"
                          else "no interval: computed without per-replicate fits")
        out["rows"].append(row)
    return out


def state_strips(result: dict) -> list[dict]:
    """One condition's per-state strips, in percent: track start (pi), occupancy (the pooled
    fit, each replicate day's, their mean and SD) and the stationary mix, per state. Each of
    the three adds up to 100 % over the states; occupancy usually lies between the other two
    because tracks are short."""
    fit = result["fit"]
    n = int(fit["n_states"])
    names = _names(n)
    interval = result.get("interval") or {}
    days_by_state = [[] for _ in range(n)]
    if interval.get("mode") == "day":
        for v in interval.get("values", ()):
            if int(v.get("n_states", 0)) == n:
                for i in range(n):
                    days_by_state[i].append(100.0 * float(v["occupancy"][i]))
    rows = []
    for i in range(n):
        m, sd = _mean_sd(days_by_state[i])
        rows.append({"name": names[i], "index": i,
                     "pi": None if fit.get("initial") is None else 100.0 * float(fit["initial"][i]),
                     "occupancy": 100.0 * float(fit["occupancy"][i]),
                     "days": days_by_state[i] if len(days_by_state[i]) >= 2 else [],
                     "mean": m, "sd": sd,
                     "stationary": 100.0 * float(fit["stationary"][i])})
    return rows


def group_comparison(results: list, state: str) -> dict:
    """Three or more conditions side by side for ``state``: each one's matrix and, per
    quantity, its mean ± SD over its days (or its pooled value). No difference: a
    difference is between two."""
    cols = []
    for r in results:
        fit = r["fit"]
        n = int(fit["n_states"])
        s = state_index(n, state)
        cells = []
        for key, _name, unit, scale in PAIR_ROWS:
            if s is None:
                cells.append("–")
                continue
            m, sd = _mean_sd(_day_values(r, s, key, scale))
            pooled = state_numbers(fit, s)[key]
            v = m if m is not None else (None if pooled is None else pooled * scale)
            cells.append(_fmt(v, unit) + ("" if sd is None else
                                          f" ± {_fmt(sd, unit).split(' ')[0]}"))
        cols.append({"rates": fit["rates"], "names": _names(n), "highlight": s,
                     "missing": None if s is not None else missing_note(n, state),
                     "cells": cells})
    return {"columns": cols, "rows": [name for _k, name, _u, _s in PAIR_ROWS]}


def _dwell_cell(fit: dict, i: int, interval: dict, cell: bool, values: list, many: str) -> str:
    """State ``i``'s model dwell on a card: the pooled fit's, then its 95 % interval over movie or cell resamples
    (the rate out's inverted, `_exit_bands`; "no upper bound" where some resamples never left), or the replicates'
    mean +- SD with a count of those that never left (no dwell, so never averaged in)."""
    n = int(fit["n_states"])
    if n < 2:
        return "–"
    dwell = fit["dwell_s"][i]
    text = "never left" if dwell is None else f"{1000 * dwell:.0f} ms"
    if cell:
        band = ((interval.get("dwell") or []) + [None] * (i + 1))[i]
        if band is not None and band[0] is not None:
            top = "no upper bound" if band[1] is None else f"{1000 * band[1]:.0f}"
            text += f" [{1000 * band[0]:.0f}, {top}]"
    if values:
        found = [state_numbers(v, i)["dwell"] for v in values]
        finite = [1000.0 * d for d in found if d is not None]
        m, sd = _mean_sd(finite)
        parts = [] if m is None else [f"{many} {m:.0f} ± {sd:.0f} ms"]
        never = len(found) - len(finite)
        if never:
            parts.append(f"{never} never left")
        if parts:
            text += f" ({'; '.join(parts)})"
    return text


def card_tables(result: dict) -> dict:
    """A card's compact content: ``{"rates": [[from -> to, rate]], "states": {"columns",
    "rows"}, "bic": str | None, "flags": [str]}``. The sentences each card used to repeat
    are `READING_NOTES`, said once on the page."""
    fit = result["fit"]
    n = fit["n_states"]
    names = _names(n)
    interval = result.get("interval") or {}
    cell = _resampled(interval) and not interval.get("withheld")      # movies or cells
    degenerate = interval.get("n_degenerate", 0) if cell else 0
    bands = interval.get("rates", {}) if cell else {}
    # THE REPLICATES' MEAN +- SD beside every rate and dwell (12 Nov 2026), as beside the occupancy
    values = [v for v in interval.get("values") or () if int(v.get("n_states", n)) == n] \
        if interval.get("mode") == "day" else []
    many = unit_words(interval)[1]

    def rate_cell(i: int, j: int) -> str:
        text = rate_text(fit["rates"][i][j], bands.get(f"{i}>{j}"))
        m, sd = _mean_sd([float(v["rates"][i][j]) for v in values])
        return text if m is None else text + f" ({many} {m:.3g} ± {sd:.2g} /s)"

    rates = [[f"{names[i]} → {names[j]}", rate_cell(i, j)]
             for i in range(n) for j in range(n) if i != j]
    columns = ["State", "D (µm²/s)", "Occupancy", "Track start π", "Stationary",
               "Dwell (model)"]
    rows = []
    for i in range(n):
        occ = _pct(fit["occupancy"][i])
        if cell and interval.get("occupancy") and interval["occupancy"][i]:
            lo, hi = interval["occupancy"][i]
            occ += f" [{_pct(lo)}, {_pct(hi)}]"
        if interval.get("mode") == "day" and interval.get("occupancy_mean"):
            occ += (f" ({unit_words(interval)[1]} {_pct(interval['occupancy_mean'][i])} ± "
                    f"{_pct(interval['occupancy_sd'][i])})")
        rows.append([names[i], _d3(fit['D'][i]), occ,
                     "–" if fit["initial"] is None else _pct(fit["initial"][i]),
                     _pct(fit["stationary"][i]), _dwell_cell(fit, i, interval, cell, values, many)])
    flags = []
    if fit["degenerate"]:
        flags.append("No transition out of a state was observed: its rates and dwell are "
                      "not measured.")
    if degenerate:
        flags.append(f"Rates not identifiable: {degenerate} of {interval['n_boot']} "
                     f"{interval['mode']} resamples held no transition out of a state. Quote the "
                     f"occupancy.")
    transient = [names[i] for i, v in enumerate(fit["transient"]) if v]
    if transient and n >= 2:
        flags.append(f"{', '.join(transient)}: left within two frames, so these are molecules caught "
                     f"mid-switch and may not be a population.")
    flags += [f"Check the fit: {warning}." for warning in fit_flags(result)]
    for day, why in interval.get("failed", ()):
        flags.append(f"{day}: no fit ({why}).")
    if interval.get("degenerate_days"):
        flags.append(f"{', '.join(interval['degenerate_days'])}: no transition out of a "
                     f"state, so that {unit_words(interval)[0]}'s rates are not measured.")
    bic = None
    selection = result.get("selection")
    if selection:
        bic = "ΔBIC " + " · ".join(f"{k} state{'' if int(k) == 1 else 's'} {'best' if d == 0 else f'+{d:.1f}'}"
                                   for k, _b, d in selection["rows"])
        if selection["inconclusive"]:
            flags.append(f"State count inconclusive: the runner-up is within "
                         f"{selection['margin']:.1f} of the best.")
    # the rates' header names an interval only where one is drawn beside them (a movie or cell resampling; found by
    # the overnight tester, 4 Nov 2026: "Rate [interval]" over rates with none, under Replicates)
    rate_columns = ["Switch", "Rate [95 % interval]" if bands else "Rate"]
    return {"rates": rates, "rate_columns": rate_columns, "states": {"columns": columns, "rows": rows}, "bic": bic,
            "spread": spread_line(interval), "flags": flags}


#: What every card used to repeat, said once on the page (the "How to read" hover card).
READING_NOTES = (
    "Switching rates (per second) are what the model measures. A dwell time is calculated from them and is "
    "usually longer than the tracks, so read it as an estimate.\n\n"
    "Occupancy is the time average over the observed steps. Short tracks do not mix, so it lies between the share at "
    "the start of a track (π) and the long-run share the rates imply (stationary).\n\n"
    "D is apparent unless you turn the σ correction on, so it compares with SQD. The slow state's D sits near the "
    "localisation floor.\n\n"
    "BIC recommends a state count; a difference under 2 is inconclusive.\n\n"
    "Spread is over independent units: replicates, or movies or cells within a single session. No statistic between "
    "conditions is computed here.")


# --------------------------------------------------------------------------- #
# The Run dialog's rules and words
# --------------------------------------------------------------------------- #


SETTING_HELP = {
    "states": ("How many diffusive states the model has.",
               "Two states is the standard model. Three lets BIC test whether "
               "a third state is supported; its slowest D often sits at the localisation "
               "floor. Best fits one to three states and picks by BIC; a difference "
               "under 2 is inconclusive."),
    "uncertainty": ("Over replicates, or this session's cells.",
                    "Replicates: one fit per replicate, shown as mean ± SD of the occupancy. Cells: percentile "
                    "intervals from 100 refitted resamples of the cells, for a single replicate only."),
    "sigma": ("Correct D for localisation error, or not.",
              "Off (the default) reports apparent D, comparable with the SQD fit. The "
              "tracker's sigma subtracts the localisation offset; use it as a check: if "
              "the slow state disappears with it, it was localisation noise."),
}


def setting_help(key: str) -> tuple[str, str]:
    return SETTING_HELP[key]


def config_problem(included: dict) -> str:
    """Why Run is off, or "" when it may run."""
    if not any(included.values()):
        return "Pick at least one condition."
    return ""


def hmm_config(included: dict, states: str, uncertainty: str, sigma: str) -> dict:
    """The committed settings. RAISES while the draft cannot run."""
    problem = config_problem(included)
    if problem:
        raise ValueError(problem)
    if states not in STATE_CHOICES or uncertainty not in UNCERTAINTY_MODES \
            or sigma not in SIGMA_MODES:
        raise ValueError("unknown HMM setting")
    return {"conditions": [k for k, on in included.items() if on], "states": states,
            "uncertainty": uncertainty, "sigma": sigma}
