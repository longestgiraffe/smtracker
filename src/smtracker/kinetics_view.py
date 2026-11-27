"""The Kinetics view's numbers and sentences, with no Qt anywhere in it.

`ui/workspace/kinetics_chart.py` is the Qt shell; this is what it draws from. They are split
because of the Thin Qt rule (docs/gui_tool_views.md; `tests/test_ui_thin_qt.py` walks the
whole GUI layer): a wrong number must fail in a headless test, not look subtly wrong on
screen, and a process-pool worker should import this module without importing Qt.

The words and rules of the Qt side live here too: what the configuration dialog's Run gate says and what it hands
back (`config_problem`, `kinetics_config`), the bin note, the card's caption, the model table's cells and the chart's
tooltip, so none of them is decided inside a Qt class.

`compute_kinetics` is picklable and runs on the process pool, one condition at a time. It
returns the full spine result (the curve, the fits, the alignment) AND two ready-to-draw
entries, ``"chart"`` and ``"text"``, made of plain lists, floats and strings, which are the
only things the Qt layer reads.

The rules the view keeps are stated in `kinetics_chart`'s docstring; the ones that live in
this file: the slow fraction is read from a two-state HMM fitted ONCE per condition (only its
per-step posterior is binned), the spine is always handed either a bin width in seconds or
``binned=False`` (its silent ten-bin default is unreachable from here), and a fit whose
parameters average outside 0-100 % is reported, never drawn or turned into a number.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "UNITS",
    "UNITS_BY_READOUT",
    "PERMUTATION_WORDS",
    "REFERENCE_WORDS",
    "REFERENCE_TIP",
    "reference_note",
    "alignment_lines",
    "BIN_CHOICES",
    "bin_guess",
    "clock_starts",
    "mobility_preview",
    "PREVIEW_WORDS",
    "PREVIEW_NOTE",
    "PREVIEW_MAX_CONDITIONS",
    "trigger_bin_guess",
    "auto_bin",
    "bin_note",
    "card_caption",
    "change_line",
    "compute_kinetics",
    "config_problem",
    "fit_lines",
    "flag_lines",
    "kinetics_config",
    "READOUT_NOTES",
    "SPREAD_WORDS",
    "MODEL_HEADINGS",
    "model_cells",
    "response_hover",
]

#: What each unit option means to the spine's `unit=` (the independent unit, never the bin). "replicate" (7 Nov 2026)
#: holds whole replicates out; only the Populations over time readout offers it (`UNITS_BY_READOUT`).
UNITS = ("movie", "cell", "replicate")
UNITS_BY_READOUT = {"mobility": UNITS, "populations": UNITS, "slow_fraction": ("movie", "cell")}


# --------------------------------------------------------------------------- #
# Compute: one condition, on the process pool
# --------------------------------------------------------------------------- #


def _model_value(model: str, p: dict, t):
    """The recommended model at times ``t``, from the mean of the per-unit parameters;
    ``None`` when a parameter is missing (a model fitted on fewer than two units)."""
    if model == "linear":
        a, b = p.get("linear.intercept"), p.get("linear.slope")
        if a is None or b is None or not (np.isfinite(a["mean"]) and np.isfinite(b["mean"])):
            return None
        return a["mean"] + b["mean"] * t
    a, b, k = (p.get("exponential.plateau"), p.get("exponential.amplitude"),
               p.get("exponential.rate"))
    if a is None or b is None or k is None:
        return None
    if not all(np.isfinite(x["mean"]) for x in (a, b, k)):
        return None
    return a["mean"] + b["mean"] * np.exp(-k["mean"] * t)


def _slow_state_weight(decoded: pd.DataFrame, D) -> np.ndarray:
    """Each decoded step's probability of being in the SLOW state, at exactly two states.

    `hmm.decode_states` returns the winning state and its posterior; at two states the
    other state's probability is the remainder, so the soft weight is ``posterior`` where
    the step is slow and ``1 - posterior`` where it is not. The slow state is found from
    ``D`` rather than assumed to be state 1. Raises at any other state count.

    THE SAME FORMULA IS INLINE IN `spatialhmm.maps_from_projected_hmm`. It is repeated
    here on purpose: this section does not edit the analysis modules the original
    application also runs. If the two are ever merged, that is where.
    """
    from .hmm import HMMError

    D = np.asarray(D, dtype=float)
    if D.size != 2:
        raise HMMError(
            f"a slow-state weight needs two states, got {D.size}. With more, the "
            f"posterior of the winning state does not determine the slow state's share.")
    slow_state = int(np.argsort(D)[0]) + 1
    state = decoded["state"].to_numpy()
    posterior = decoded["posterior"].to_numpy(float)
    return np.where(state == slow_state, posterior, 1.0 - posterior)


def compute_kinetics(project, condition: str, *, trigger_frame: int, unit: str,
                     bin_seconds: float | None, align: bool) -> dict:
    """One condition's triggered response: the curve, the fits, the honest flags.

    ``bin_seconds=None`` means UNBINNED (one point per track); a number is the bin width
    in SECONDS. There is no third case: the spine's default of ten bins is never reached
    from here. Picklable, and the HMM fit makes it slow enough to belong on the process
    pool. Every failure comes back as ``{"error": "<Type>: <message>"}`` naming what
    refused (a condition whose tracks never switch state, a mixed frame interval, a trigger
    past the end), so the card says why instead of vanishing. The first is
    `kinetics.require_switching` (ruled 4 Nov 2026: Kinetics keeps refusing it, while the
    HMM maps draw it).
    """
    from .hmm import decode_states, fit_hmm, step_sequences
    from .kinetics import (
        KineticsError,
        alignment_gain,
        detect_onsets,
        fit_response,
        onset_alignment_null,
        require_switching,
        response_curve,
    )

    try:
        canonical = project.condition(condition)
        tau_s = float(project.params.tau_s)
        # Fitted ONCE on the whole condition; only the per-step posterior is binned.
        hmm_fit = fit_hmm(step_sequences(canonical), n_states=2, tau_s=tau_s, n_starts=8)
        # NO SWITCHING WITHIN TRACKS, NO KINETICS (ruled 4 Nov 2026): refused by name, as before 7575e0f7, which let
        # `decode_states` decode such a chain for the HMM maps and, by the way, this curve
        require_switching(hmm_fit)
        steps = decode_states(canonical, hmm_fit)
        steps["w_slow"] = _slow_state_weight(steps, hmm_fit.D)
        curve = response_curve(
            steps, trigger_frame=int(trigger_frame), observable="slow_fraction",
            unit=unit, tau_s=tau_s, binned=bin_seconds is not None,
            bin_seconds=bin_seconds)
        window_frames = int(steps["frame"].max() - int(trigger_frame) + 1)
        fit = fit_response(curve, n_frames=window_frames)
    except Exception as exc:                          # noqa: BLE001 - named on the card
        return {"error": f"{type(exc).__name__}: {exc}"}

    t_max = float(curve["t_s"].max())
    grid = np.linspace(0.0, t_max, 120)
    fitted = _model_value(fit["recommended"], fit["parameters"], grid)
    # The model is fitted per unit and its parameters are AVERAGED, which is only meaningful
    # while the per-unit fits are well constrained. A fraction cannot leave 0-100 %, so an
    # averaged curve that does is reported as that, not drawn and not turned into a number.
    out_of_range = bool(fitted is not None and (fitted.min() < -0.02 or fitted.max() > 1.02))
    if out_of_range:
        fitted = None
    change = None
    if fitted is not None:
        change = float((fitted[-1] - fitted[0]) * 100.0)

    mean_curve = None
    if bin_seconds is not None:
        good = curve[curve["ok"] & np.isfinite(curve["value"])]
        mean_curve = (good.groupby("bin")
                      .agg(t_s=("t_s", "mean"), mean=("value", "mean"),
                           sd=("value", "std"), n_units=(unit, "nunique"))
                      .reset_index())

    alignment = None
    if align:
        if bin_seconds is None:
            alignment = {"error": "alignment needs a binned curve"}
        else:
            alignment = {}
            try:
                onsets = detect_onsets(curve)
                alignment["n_located"] = int(onsets.attrs.get("n_located", 0))
                alignment["n_unlocated"] = int(onsets.attrs.get("n_unlocated", 0))
                located = onsets["onset_s"].to_numpy(float)
                located = located[np.isfinite(located)]
                alignment["onset_sd_s"] = float(located.std(ddof=1)) if located.size > 1 else None
            except KineticsError as exc:
                alignment["onsets_error"] = str(exc)
            try:
                alignment["null"] = onset_alignment_null(curve)
            except KineticsError as exc:
                alignment["null_error"] = str(exc)
            try:
                alignment["gain"] = alignment_gain(curve)
            except KineticsError as exc:
                alignment["gain_error"] = str(exc)

    result = {
        "condition": condition,
        "settings": {"trigger_frame": int(trigger_frame), "unit": unit,
                     "bin_seconds": bin_seconds, "align": bool(align), "tau_s": tau_s,
                     "window_frames": window_frames},
        "D": [float(d) for d in hmm_fit.D],
        "curve": curve,
        "mean_curve": mean_curve,
        "fit": fit,
        "fit_curve": None if fitted is None else (grid, fitted),
        "fit_out_of_range": out_of_range,
        "change_pp": change,
        "t_max_s": t_max,
        "alignment": alignment,
    }
    # What the Qt layer draws and prints, as plain lists, floats and strings: the chart and the
    # card never touch a DataFrame or numpy (the Thin Qt rule, tests/test_ui_thin_qt.py).
    result["chart"] = _chart_payload(curve, mean_curve, grid, fitted, unit, bin_seconds is not None)
    result["text"] = _text_payload(result)
    return result


def _chart_payload(curve, mean_curve, grid, fitted, unit: str, binned: bool) -> dict:
    """Everything `ResponseChart` draws, in percent, as plain Python.

    ``units``: one ``[(t, value), ...]`` list per unit (binned); ``points``: one
    ``(t, value)`` per track (unbinned); ``mean``: ``(t, mean, sd_or_None, n_units)`` per
    bin; ``fit``: the recommended model sampled on a grid, or None; ``y_range``: the data and
    fit extremes, so the axis follows the data rather than a fixed 0-100.
    """
    good = curve[curve["ok"] & np.isfinite(curve["value"])]
    if good.empty:
        return {"binned": binned, "unit": unit, "t_max": 0.0, "units": [], "mean": [],
                "points": [], "fit": None, "y_range": None}
    fit_pts = (None if fitted is None
               else [(float(t), float(v) * 100.0) for t, v in zip(grid, fitted)])
    lo, hi = float(good["value"].min()) * 100.0, float(good["value"].max()) * 100.0
    if fit_pts:
        lo, hi = min(lo, min(v for _t, v in fit_pts)), max(hi, max(v for _t, v in fit_pts))
    units, points, mean = [], [], []
    if binned:
        for _key, part in good.groupby(unit, observed=True):
            part = part.sort_values("t_s")
            units.append([(float(t), float(v) * 100.0)
                          for t, v in zip(part["t_s"], part["value"])])
        for row in mean_curve.itertuples():
            sd = float(row.sd) * 100.0 if np.isfinite(row.sd) else None
            mean.append((float(row.t_s), float(row.mean) * 100.0, sd, int(row.n_units)))
    else:
        points = [(float(t), float(v) * 100.0) for t, v in zip(good["t_s"], good["value"])]
    return {"binned": binned, "unit": unit, "t_max": float(good["t_s"].max()), "units": units,
            "mean": mean, "points": points, "fit": fit_pts, "y_range": (lo, hi)}


def _text_payload(result: dict) -> dict:
    """The card's sentences and table rows, ready to print."""
    fit = fit_lines(result)
    align = result.get("alignment")
    return {"models": fit["models"], "parameters": fit["parameters"],
            "change": change_line(result), "flags": flag_lines(result),
            "alignment": None if align is None else alignment_lines(align)}


# --------------------------------------------------------------------------- #
# The words, as pure functions of the result (checked without Qt)
# --------------------------------------------------------------------------- #


def _ci(entry: dict, scale: float = 1.0, digits: int = 2) -> str:
    lo, hi = entry["ci95"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "interval not available"
    return f"95 % {lo * scale:+.{digits}f} to {hi * scale:+.{digits}f}"


def fit_lines(result: dict) -> dict:
    """``{"models": [(name, held-out error, against the best, held-out units won,
    recommended)], "parameters": [text]}``: the model table ranks by how well each model,
    fitted without a unit, predicts it (`kinetics.fit_response`)."""
    fit = result["fit"]
    errors = sorted(fit["heldout_error"].items(), key=lambda kv: kv[1])
    best = errors[0][1]
    won = fit.get("folds_won") or {}
    models = [(name, err, err - best, int(won.get(name, 0)), name == fit["recommended"])
              for name, err in errors]
    params = []
    p = fit["parameters"]
    n, unit = fit["n_units"], fit["unit"]
    if "linear.slope" in p:
        e = p["linear.slope"]
        params.append(f"linear: slope {e['mean'] * 100:+.2f} pp/s ({_ci(e, 100)} over "
                      f"{n} {unit}s resampled)")
    # THE EXPONENTIAL'S NUMBERS ONLY WHERE IT IS THE CHOICE: fitted to a line, its rate runs to
    # its floor and its "half-life" to thousands of seconds, a number that means nothing.
    if "exponential.rate" in p and fit["recommended"] == "exponential":
        e = p["exponential.rate"]
        params.append(f"exponential: rate {e['mean']:.3f} /s ({_ci(e, 1, 3)})")
    if "halflife_s" in fit and fit["recommended"] == "exponential":
        lo_hi = fit.get("halflife_ci95")
        params.append(f"half-life {fit['halflife_s']:.2f} s"
                      + (f" (95 % {lo_hi[0]:.2f} to {lo_hi[1]:.2f} s)" if lo_hi else ""))
    return {"models": models, "parameters": params}


def change_line(result: dict) -> str:
    """The fitted change over the window, beside the drift an UNTRIGGERED population
    shows. The null is not zero, so both are always printed together.

    The drift is a LAB REFERENCE (`kinetics.UNTRIGGERED_DRIFT_SOURCE`, ruled 4 Nov 2026), so the
    sentence names the data set it was measured on and says it is not this project's: its
    conditions (glucose, malate) are that data set's, whatever this project's are called."""
    drift = result["fit"]["untriggered_drift_pp"]
    source = result["fit"]["untriggered_drift_source"]
    parts = [f"{v[0]:+.2f} pp ({v[1]:+.2f} to {v[2]:+.2f}) in {name}" for name, v in drift.items()]
    ref = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    window = f"{result['t_max_s']:.1f} s"
    change = result["change_pp"]
    if change is not None:
        own = f"fitted change over the {window} window: {change:+.1f} pp"
    elif result.get("fit_out_of_range"):
        own = (f"no change over the {window} window is reported: the fitted curve leaves "
               f"0-100 %")
    else:
        own = f"the fitted change over the {window} window could not be formed"
    return (f"{own}. Reference from other data, not this project: in {source} the slow fraction drifted "
            f"{ref}. A response has to beat a band of that size, not zero.")


def flag_lines(result: dict) -> list:
    """Every caveat the spine raised, plus the resolvable range, as sentences."""
    fit, curve = result["fit"], result["curve"]
    out = []
    if fit.get("inconclusive"):
        out.append(fit["caveat"])
    if "resolvable" in fit:
        r = fit["resolvable"]
        out.append(f"resolvable half-life range for this acquisition: {r['min_halflife_s']:.2f} to "
                   f"{r['max_halflife_s']:.2f} s")
    if result.get("fit_out_of_range"):
        out.append("the fitted curve leaves 0-100 %, so it is not drawn")
    if fit.get("halflife_caveat"):
        out.append(fit["halflife_caveat"])
    if fit.get("track_span_caveat"):
        out.append(fit["track_span_caveat"])
    if curve.attrs.get("selection_caveat"):
        out.append(curve.attrs["selection_caveat"])
    if fit.get("n_failed_fits"):
        out.append(f"{fit['n_failed_fits']} of the {fit['n_boot']} {fit['unit']} resamples "
                   f"could not be fitted and are not in the intervals")
    before = curve.attrs.get("n_steps_before_trigger")
    if before:
        out.append(f"{before:,} steps before the trigger were not used")
    return out


def alignment_lines(alignment: dict) -> list:
    """The adaptive alignment, with the null it requires printed beside it every time."""
    if "error" in alignment:
        return [alignment["error"]]
    lines = []
    if "onsets_error" in alignment:
        lines.append(alignment["onsets_error"])
    elif "n_located" in alignment:
        total = alignment["n_located"] + alignment["n_unlocated"]
        sd = alignment.get("onset_sd_s")
        lines.append(f"onsets located in {alignment['n_located']} of {total} units"
                     + (f"; spread {sd:.2f} s" if sd is not None else ""))
    null = alignment.get("null")
    if null is not None:
        lines.append(
            f"null: observed step {null['observed_step'] * 100:.1f} pp, chance "
            f"{null['chance_step'] * 100:.1f} ± {null['chance_sd'] * 100:.1f} pp, excess "
            f"{null['excess'] * 100:+.1f} pp. Beats the null: "
            f"{'yes' if null['beats_null'] else 'no'}")
        lines.append("Only the excess over chance is evidence: aligning each unit on a feature "
                     "found in its own noise manufactures that feature.")
    elif "null_error" in alignment:
        lines.append("null: " + alignment["null_error"])
    gain = alignment.get("gain")
    if gain is not None:
        lines.append(f"alignment reads as: {gain['verdict']}")
    elif "gain_error" in alignment:
        lines.append(alignment["gain_error"])
    return lines


# --------------------------------------------------------------------------- #
# The configuration dialog's rules, the card's caption and the chart's tooltip
# --------------------------------------------------------------------------- #


def _bin_width(text: str) -> float | None:
    """The typed bin width in seconds, or None where it is not a positive finite number."""
    try:
        width = float(text)
    except ValueError:
        return None
    return width if math.isfinite(width) and width > 0 else None


def config_problem(rows: dict, binned: bool, width_text: str, *, clock: str = "trigger",
                   additions: dict | None = None, unit: str | None = None, readout: str = "populations",
                   reference: str | None = None, samples: dict | None = None, movies: dict | None = None,
                   mock: str | None = None, mock_after_text: str | None = None) -> str:
    """Why Run must stay disabled, or "" when the draft is complete.

    ``rows`` maps each condition to ``(included, trigger_frame_text)``. Neither the trigger frame nor the bin width
    has a default: a condition is included only with a typed frame, and a binned curve needs a width in seconds
    (or the curve must be left unbinned), so the spine's silent ten-bin fallback is unreachable from here.
    Under the ``"addition"`` clock (`induction_view.CLOCKS`) the frames are not asked: ``additions``
    ``{condition: {day: typed time}}`` must hold a readable time for every day of every included condition. Under the
    ``"sample"`` clock (the design "Added to each sample") ``samples`` ``{condition: {sample: {"movies", "time"}}}`` and
    ``movies`` ``{condition: [movie, ...]}`` must give EVERY movie of every included condition a sample with a time
    that reads (`kinetics_designs.sample_problem`). ``mock`` / ``mock_after_text``: the mock addition and the typed
    start of the after window (`mock_problem`), asked of the readouts that offer it only."""
    included = [key for key, (on, _text) in rows.items() if on]
    if not included:
        return "Include at least one condition."
    if mock and readout in MOCK_READOUTS:
        problem = mock_problem(mock, included, mock_after_text)
        if problem:
            return problem
    if clock == "sample":
        from .induction_view import parse_addition
        from .kinetics_designs import sample_problem

        problem = sample_problem({k: (samples or {}).get(k, {}) for k in included}, movies or {},
                                 parse=parse_addition)
        if problem:
            return problem
    elif clock == "addition":
        from .induction_view import addition_problem

        problem = addition_problem({key: (True, (additions or {}).get(key, {})) for key in included})
        if problem:
            return problem
    else:
        missing = [key for key in included if not rows[key][1]]
        if missing:
            return f"Type a trigger frame for {missing[0]}."
    if binned and _bin_width(width_text) is None:
        return "Type a bin width in seconds, or choose Unbinned."
    if unit is not None and unit not in UNITS_BY_READOUT.get(readout, UNITS):
        return f"The slow-fraction readout holds out movies or cells; {unit.capitalize()} is for Populations over time."
    if reference:
        from pathlib import Path

        if not Path(reference).is_file():
            return f"The reference file {reference} cannot be found."
    return ""


def kinetics_config(rows: dict, binned: bool, width_text: str, unit: str, align: bool, *,
                    readout: str = "slow_fraction", n_pop: int = 2, spread: bool = False,
                    clock: str = "trigger", additions: dict | None = None, permutation: bool = False,
                    reference: str | None = None, precision: bool = False, design: str | None = None,
                    samples: dict | None = None, movies: dict | None = None,
                    bin_auto: bool = False, mock: str | None = None, mock_after_text: str | None = None) -> dict:
    """The committed settings. RAISES while the draft is incomplete: a missing width must never read back as
    "unbinned". ``readout`` is the section's question (`induction_view.READOUTS`): the D's and shares over time
    (``n_pop`` populations, ``spread`` the onset test between movies) or the HMM slow fraction (``align``). The
    ``bin_auto``: ``width_text`` is Auto's guess (`auto_bin`), kept in the config as ``bin_auto`` so the
    next dialog opens on Auto. The ``"addition"`` clock belongs to the populations readout only (the HMM readout's spine takes a trigger frame):
    its conditions carry no frame, and ``additions`` their days' times. ``precision``: the populations readout with
    each step's own localisation error (Per spot, 6 Nov 2026; the window resolves the calibration,
    `workspace_view.run_precision`).

    ``design`` (`kinetics_designs.DESIGNS`) belongs to the populations readout and is written into the config only
    when given, so a config made without it (every one before 7 Nov 2026) reads as `design_of` reads it. It fixes the
    clock: "culture" the addition clock, "sample" the per-sample clock (``samples`` ``{condition: {sample: {"movies",
    "time"}}}``, ``movies`` ``{condition: [movies]}`` for the check that every movie has a time), "microscope" the
    ``clock`` given (the trigger frame, or one addition time per replicate).

    ``mock`` (8 Nov 2026, `kinetics_mock`): the condition whose addition is a MOCK, every other ticked condition read
    against it; ``mock_after_text`` the typed start of the after window in seconds ("" or None: Auto). Both are
    written only for the readouts that offer them (`MOCK_READOUTS`) and only when a mock is chosen, so a config made
    without them reads as "None"."""
    from .induction_view import CLOCKS, N_POPS, POPULATION_READOUTS, READOUTS
    from .kinetics_designs import DESIGN_CLOCKS, DESIGNS

    if readout not in READOUTS:
        raise ValueError(f"unknown readout {readout!r}")
    populations = readout in POPULATION_READOUTS
    if readout == "mobility":
        n_pop = 1                                    # the mean mobility is the model with one population
    if design is not None:
        if design not in DESIGNS:
            raise ValueError(f"unknown design {design!r}")
        if not populations:
            raise ValueError(f"the experiment design belongs to Populations over time, not {readout!r}")
        allowed = DESIGN_CLOCKS[design]
        clock = allowed[0] if len(allowed) == 1 else clock
        if clock not in allowed:
            raise ValueError(f"the {clock!r} clock does not belong to the design {design!r}")
    if clock not in CLOCKS or (clock in ("addition", "sample") and not populations):
        raise ValueError(f"the {clock!r} clock is not offered for {readout!r}")
    problem = config_problem(rows, binned, width_text, clock=clock, additions=additions, unit=unit, readout=readout,
                             reference=reference if populations else None, samples=samples, movies=movies,
                             mock=mock, mock_after_text=mock_after_text)
    if problem:
        raise ValueError(problem)
    if readout == "populations" and int(n_pop) not in N_POPS:
        raise ValueError(f"{n_pop} populations: the model holds {' or '.join(map(str, N_POPS))}")
    sample_table = None
    if clock == "sample":
        included = [key for key, (on, _text) in rows.items() if on]
        conditions = {key: None for key in included}
        times = None
        sample_table = {key: {name: {"movies": [str(m) for m in e["movies"]], "time": str(e["time"]).strip()}
                              for name, e in (samples or {}).get(key, {}).items()} for key in included}
    elif clock == "addition":
        included = [key for key, (on, _text) in rows.items() if on]
        conditions = {key: None for key in included}
        times = {key: {int(day): str(text).strip() for day, text in (additions or {}).get(key, {}).items()}
                 for key in included}
    else:
        conditions = {key: int(text) for key, (on, text) in rows.items() if on and text}
        times = None
    return {"conditions": conditions, "clock": clock, "additions": times,
            "unit": unit,
            "bin_seconds": _bin_width(width_text) if binned else None,
            **({"bin_auto": True} if bin_auto and binned else {}),
            "readout": readout,
            "n_pop": int(n_pop) if populations else None,
            "spread": bool(spread and populations),
            "permutation": bool(permutation and populations),
            "reference": str(reference) if reference and populations else None,
            "align": bool(align and binned and not populations),
            # per spot only when on, so every config (and session) made before reads as it was
            **({"precision": True} if precision and populations else {}),
            # the design only when given, so a config without it reads as before (`kinetics_designs.design_of`)
            **({"design": design} if design is not None else {}),
            **({"samples": sample_table} if sample_table is not None else {}),
            **(mock_entries(mock, mock_after_text) if mock and readout in MOCK_READOUTS else {})}


#: What each readout answers, under its switch in the Run dialog.
READOUT_NOTES = {
    "mobility": ("One D at each time, the mean of every jump (model-free in practice): a change in D and a change in "
                 "the bound share both move it; Populations over time tells which. The least noisy readout."),
    "populations": ("Each population's D and share over time, from one model fitted to every step at its own time; "
                    "which of the two moved is chosen on movies held out."),
    "slow_fraction": ("The slow state's share over time, from a two-state HMM fitted once: a D that changes reads "
                      "here as a share that changes."),
}

#: The populations readout's per-spot switch (6 Nov 2026).
PRECISION_WORDS = ("Localisation error per spot: each step's own, from the tracker's precision and a "
                   "calibration measured on immobile spots (conditions whose tracks carry none are fitted without it)")

#: The onset test's switch, and why it is off unless asked.
SPREAD_WORDS = ("Also test whether movies started responding at different times (about 20 extra fits a condition, "
                "against shuffled movies)")
#: The time-label permutation test's switch (7 Nov 2026).
PERMUTATION_WORDS = ("Also run the permutation test: does the order of the clock matter at all? (199 shuffles of the "
                     "movies' clocks within each replicate, every changing model refitted on each; shown beside the "
                     "held-out verdict)")
#: The same switch under the design "Added to each sample": what is dealt is the samples' addition times.
PERMUTATION_WORDS_SAMPLE = ("Also run the permutation test: does the order of the clock matter at all? (199 shuffles of "
                            "the samples' addition times among the samples of each replicate, every changing model "
                            "refitted on each; shown beside the held-out verdict)")
# --------------------------------------------------------------------------- #
# A mock addition to read the change against (8 Nov 2026, `kinetics_mock`)
# --------------------------------------------------------------------------- #

#: The Run dialog's readouts that offer a mock, and the readout `kinetics_mock.compare` reads for each.
MOCK_READOUTS = {"mobility": "mean_D", "populations": "bound_share"}
MOCK_WORDS = "Compare with a mock addition (optional)"
MOCK_NONE = "None"
MOCK_TIP = ("A condition imaged on the same schedule with nothing added (the same handling). Every other ticked "
            "condition is read against it: real sessions drift with no addition at all, so a change in one "
            "condition alone is confounded with time in the session.")
MOCK_AFTER_WORDS = "After window"
MOCK_AFTER_CHOICES = ("Auto", "Start in seconds")
MOCK_AFTER_NOTE = ("Auto starts where the treated condition's fitted response is 90 % complete, else at the middle "
                   "of the record after the addition. The baseline is every step before the addition.")


def mock_after_seconds(text) -> float | None:
    """The typed start of the after window in seconds since the addition, or None for Auto ("" or None).
    RAISES ValueError for text that is not a number of 0 or more."""
    if text is None or not str(text).strip():
        return None
    try:
        value = float(str(text).strip())
    except ValueError:
        raise ValueError(f"The after window's start {str(text).strip()!r} is not a number of seconds.") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError("The after window starts at 0 s or later (seconds since the addition).")
    return value


def mock_problem(mock: str, included: list, after_text) -> str:
    """Why a chosen mock cannot be run: it must be ticked (it needs its own addition times), at least one OTHER
    condition must be ticked to compare with it, and a typed after start must read. "" when it can."""
    if mock not in included:
        return f"Tick {mock} too: the mock addition needs its own addition times."
    if len(included) < 2:
        return "Tick a condition other than the mock to compare with it."
    try:
        mock_after_seconds(after_text)
    except ValueError as exc:
        return str(exc)
    return ""


def mock_entries(mock: str, after_text) -> dict:
    """The config's two keys: ``mock`` the condition's key, ``mock_after_s`` the typed start (None: Auto)."""
    return {"mock": str(mock), "mock_after_s": mock_after_seconds(after_text)}


#: The reference curves' field (7 Nov 2026).
REFERENCE_WORDS = "Compare with a known curve (optional)"
#: Said beside the row's Browse button: what the file is (CSV columns t_s, population, D_um2_s, share).
REFERENCE_TIP = "A CSV with the columns t_s, population, D_um2_s and share; drawn dashed in each population's ink."


def reference_note(path, found: list) -> str:
    """The reference row's sentence: "" with no file; "Found reference_SynK.csv" where the dialog picked up the file
    in the project root on its own (``found`` the files `induction_view.find_reference` listed; ``path`` the one in
    use), the file's name where it was chosen. Several matches are said, never silently the first."""
    from pathlib import Path

    if not path:
        return ""
    name = Path(path).name
    if found and Path(path) == Path(found[0]):
        more = f" ({len(found)} files match reference*.csv; using the first)" if len(found) > 1 else ""
        return f"Found {name}{more}"
    return name




# --------------------------------------------------------------------------- #
# The bin width guessed from the movies (7 Nov 2026, "Auto" in the Run dialog)
# --------------------------------------------------------------------------- #

#: The round widths Auto chooses among, seconds. Bin edges sit on multiples of the width counted from the addition
#: (`induction.binned_fits`: ``lo = floor(min / bin_s) * bin_s``), which is what the rule below reads.
BIN_CHOICES = (10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0)
#: Under a trigger inside long movies the width is about this many bins over the time observed after the trigger.
TRIGGER_BINS = 8
_TRIGGER_CHOICES = (0.5, 1.0, 2.0, 5.0) + BIN_CHOICES + (600.0, 900.0, 1800.0, 3600.0)


#: Auto keeps every movie at least this far (seconds) inside its bin's edges, or two frame intervals, whichever is
#: larger: a camera stamp a few frames off must not put a frame of a movie into the bin before (8 Nov 2026).
EDGE_MARGIN_S = 2.0


def edge_margin_s(tau_s: float | None) -> float:
    """The room Auto leaves between a movie and its bin's edges: `EDGE_MARGIN_S`, or two frame intervals."""
    return max(EDGE_MARGIN_S, 2.0 * float(tau_s or 0.0))


def _same_bin(a: float, b: float, width: float, margin: float = 0.0) -> bool:
    """Whether the span ``a`` to ``b`` lies in one bin ``margin`` seconds clear of the bin's edges (edges at multiples
    of ``width``): the span widened by the margin at both ends still lies in one bin."""
    return math.floor((a - margin) / width) == math.floor((b + margin) / width)


def bin_guess(spans: dict, replicate_of: dict | None = None, tau_s: float | None = None) -> tuple:
    """``(width in s or None, why)`` for movies that each last a stretch of the clock: ``spans`` ``{movie: (first step,
    last step), seconds since the addition}``, ``replicate_of`` ``{movie: replicate}`` (the unit, Rule 0: movies are
    time points of a replicate's response).

    The smallest round width (`BIN_CHOICES`) that is at least the longest movie and puts EVERY movie wholly inside one
    bin, edges on multiples of the width from the addition; among those, one that also keeps two movies of the SAME
    replicate in different bins (a bin then holds one movie of each replicate, never two of one) when there is one.
    EVERY movie lies at least `edge_margin_s` (2 s, or two frame intervals ``tau_s``) inside its bin's edges: a movie
    that starts exactly on an edge is valid with exact synthetic stamps, but a real stamp 20 ms earlier would put one
    frame in the bin before. Where no round width keeps that margin for every movie, the margin is dropped and the
    smallest width that holds every movie is taken, with the movies that sit within it said. Where no round width
    works at all, the longest movie rounded up to a whole second, said."""
    if not spans:
        return None, "no movies"
    longest = max(b - a for a, b in spans.values())
    margin = edge_margin_s(tau_s)
    held = [w for w in BIN_CHOICES if w >= longest and all(_same_bin(a, b, w) for a, b in spans.values())]
    fits = [w for w in held if all(_same_bin(a, b, w, margin) for a, b in spans.values())]
    margin_dropped = False
    if not fits and held:
        # no width keeps the margin: every width that holds the movies, the replicates' separation still preferred
        # below (review, 8 Nov 2026: taking only the smallest here skipped that preference)
        fits, margin_dropped = held, True
    if not fits:
        width = float(math.ceil(max(longest, 1e-9)))
        return width, (f"Auto: {width:g} s, the longest movie ({longest:g} s) rounded up; no round width keeps "
                       f"every movie inside one bin")
    replicate_of = replicate_of or {}

    def shared(width: float) -> bool:
        seen = set()
        for movie, (a, _b) in spans.items():
            key = (replicate_of.get(movie), math.floor(a / width))
            if key in seen:
                return True
            seen.add(key)
        return False

    def close(width: float) -> str:
        if not margin_dropped:
            return ""
        near = sum(1 for a, b in spans.values() if not _same_bin(a, b, width, margin))
        return (f"; {near} of {len(spans)} movies lie within {margin:g} s of a bin edge (a camera stamp a little off "
                f"moves a frame into the next bin) and no round width keeps them clear")

    for width in fits:
        if not shared(width):
            return width, f"Auto: {width:g} s, one bin per movie" + close(width)
    return fits[0], (f"Auto: {fits[0]:g} s, every movie inside one bin (two movies of one replicate share a bin, "
                     f"no round width separates them)" + close(fits[0]))


def trigger_bin_guess(span_after_s: float) -> tuple:
    """``(width, why)`` for a trigger inside long movies: the time observed after the trigger divided into about
    `TRIGGER_BINS` bins, rounded to the nearest round width (on a log scale). None where nothing follows the trigger."""
    if not span_after_s > 0:
        return None, "Auto: nothing is observed after the trigger"
    target = span_after_s / TRIGGER_BINS
    width = min(_TRIGGER_CHOICES, key=lambda w: abs(math.log(w / target)))
    return width, f"Auto: {width:g} s, about {TRIGGER_BINS} bins over the {span_after_s:g} s after the trigger"


def clock_starts(ex: dict, *, clock: str, trigger_text: str = "", additions: dict | None = None,
                 samples: dict | None = None) -> tuple:
    """``({movie: seconds from the induction to its first frame} or None, why not)`` for ONE condition's extents
    (`induction_view.movie_extents`) under the draft's clock: the trigger frame (every movie starts at minus its frames
    before the trigger), the replicates' addition times, or the samples' own. An unfinished draft (a time not typed, a
    camera start unreadable) says why and gives no starts: a clock is never guessed. The one definition under Auto's bin
    width and the raw-data preview."""
    from . import induction_view as iv
    from .induction import InductionError

    tau = ex["tau_s"]
    if clock == "trigger":
        if not str(trigger_text).strip():
            return None, "type the trigger frame first"
        return {m: -(int(trigger_text) - 1) * tau for m in ex["movies"]}, ""
    if ex["started"] is None:
        return None, ex["problem"]
    day_of = {m: d for m, (d, _lo, _hi) in ex["movies"].items()}
    try:
        if clock == "addition":
            return iv._addition_starts(ex["started"], day_of, tau, additions or {}), ""
        table = samples or {}
        starts, _record = iv._sample_starts(ex["started"], tau, {n: e["time"] for n, e in table.items()},
                                            {m: n for n, e in table.items() for m in e["movies"]})
        return starts, ""
    except InductionError as exc:                               # an unfinished draft: said, never swallowed
        return None, f"waits for the addition times ({exc})"


def auto_bin(extents: dict, *, clock: str, rows: dict, additions: dict | None = None,
             samples: dict | None = None) -> tuple:
    """The Run dialog's Auto width for the draft: ``(width or None, text)``. ``extents`` ``{condition:
    `induction_view.movie_extents`}`` for every INCLUDED condition (``rows`` ``{condition: (included, trigger text)}``);
    ``additions`` ``{condition: {replicate: typed time}}`` under the addition clock, ``samples`` ``{condition: {sample:
    {"movies", "time"}}}`` under the per-sample clock. A draft that cannot give a width yet (a time not typed, a
    camera start unreadable) says why, never a number."""
    included = [key for key, (on, _t) in rows.items() if on]
    if not included:
        return None, "Auto: include a condition first"
    spans: dict = {}
    reps: dict = {}
    if clock == "trigger":
        after = 0.0
        for key in included:
            text = rows[key][1]
            if not str(text).strip():
                return None, "Auto: type the trigger frame first"
            ex = extents[key]
            for _m, (_d, _lo, hi) in ex["movies"].items():
                after = max(after, (hi - int(text)) * ex["tau_s"])
        return trigger_bin_guess(after)
    for key in included:
        ex = extents[key]
        starts, why = clock_starts(ex, clock=clock, additions=(additions or {}).get(key, {}),
                                   samples=(samples or {}).get(key, {}))
        if starts is None:
            return None, (f"Auto: {why}" if ex["started"] is None else f"Auto {why}")
        tau = ex["tau_s"]
        for m, (d, lo, hi) in ex["movies"].items():
            spans[(key, m)] = (starts[m] + (lo - 1) * tau, starts[m] + (hi - 1) * tau)
            reps[(key, m)] = (key, d)
    return bin_guess(spans, reps, tau_s=max(extents[key]["tau_s"] for key in included))


# --------------------------------------------------------------------------- #
# The raw-data preview in the Run dialog (8 Nov 2026)
# --------------------------------------------------------------------------- #

#: The preview's title and what it is, said under it.
PREVIEW_WORDS = "What the data look like, before any fit"
PREVIEW_NOTE = ("One dot per movie: its mean squared jump over 4 tau (a model-free apparent D) against its time since "
                "the addition, coloured by replicate. No fit; it updates as the times and the design change.")
#: Conditions drawn at most (a dialog row each); the rest are said.
PREVIEW_MAX_CONDITIONS = 4


def mobility_preview(mobility: dict, extent: dict, *, clock: str, trigger_text: str = "",
                     additions: dict | None = None, samples: dict | None = None) -> dict:
    """``{"points": [{"movie", "replicate", "rank", "colour", "t_s", "D"}], "problem": ""}`` for ONE condition: each movie's
    mean squared jump over 4 tau (``mobility`` from `induction_view.movie_mobility`) at its mean time on the draft's clock
    (`clock_starts`; the mean first frame of its jump pairs). ``rank`` is the replicate's order among the condition's own
    (colour only: Rule 0, replicate 1 of one condition shares nothing with replicate 1 of another). A missing time gives
    NO points and says what is missing (a line of text in the dialog), never a guess. No fit is made."""
    starts, why = clock_starts(extent, clock=clock, trigger_text=trigger_text, additions=additions, samples=samples)
    if starts is None:
        return {"points": [], "problem": why[:1].upper() + why[1:]}
    tau = float(mobility["tau_s"])
    lag = float(mobility.get("lag_time_s", tau))
    from .figstyle import OKABE_ITO

    replicates = sorted({m["replicate"] for m in mobility["movies"].values()})
    points = []
    for movie, m in mobility["movies"].items():
        if movie not in starts or m["n"] < 1:
            continue
        rank = replicates.index(m["replicate"])
        points.append({"movie": movie, "replicate": m["replicate"], "rank": rank,
                       "colour": OKABE_ITO[rank % len(OKABE_ITO)],
                       "t_s": float(starts[movie]) + (m["mean_frame"] - 1.0) * tau,
                       "D": m["mean_sq_um2"] / (4.0 * lag)})
    return {"points": sorted(points, key=lambda p: p["t_s"]), "problem": "" if points else "No movies to draw."}


def bin_note(binned: bool, readout: str = "slow_fraction") -> str:
    if readout in ("populations", "mobility"):
        return ("Bins only draw the free fits beside the model, to show what the data look like; the model uses "
                "every step at its own time." if binned else "No free per-bin fits drawn beside the model.")
    if binned:
        return ("Bins start at the trigger; the width is in seconds so it means the same on a movie "
                "of any length.")
    return ("One point per track, placed at its own moment: the tightest fit, but a long track "
            "smears a fast decay.")


def card_caption(result: dict) -> str:
    s = result["settings"]
    binning = ("unbinned, one point per track" if s["bin_seconds"] is None
               else f"{s['bin_seconds']:g} s bins")
    return (f"trigger at frame {s['trigger_frame']} \u00b7 {binning} \u00b7 per {s['unit']} \u00b7 "
            f"slow state D {result['D'][0]:.3f} \u00b5m\u00b2/s")


#: The model table's headings (`model_cells`): ranked on units held out, never on bins.
MODEL_HEADINGS = ("model", "held-out error", "against the best", "units won", "")


def model_cells(models: list) -> list:
    """The model table's rows as text: ``(name, held-out error (pp²), against the best,
    units won, "recommended" or "", recommended)``."""
    return [(name, f"{err * 1e4:.2f}", f"{delta * 1e4:+.2f}", str(won),
             "recommended" if best else "", bool(best))
            for name, err, delta, won, best in models]


def response_hover(chart: dict, t: float) -> str | None:
    """The chart's tooltip at ``t`` seconds after the trigger: the nearest binned mean, or the nearest track when
    unbinned; None where there is nothing to name."""
    if chart["binned"] and chart["mean"]:
        mean = chart["mean"]
        i = min(range(len(mean)), key=lambda k: abs(mean[k][0] - t))
        at, value, sd, n = mean[i]
        spread = f" \u00b7 spread {sd:.1f} pp" if sd is not None else ""
        return (f"{at:.1f} s after the trigger\nmean {value:.1f} %\n"
                f"{n} {chart['unit']}s{spread}")
    points = chart["points"]
    if not points:
        return None
    i = min(range(len(points)), key=lambda k: abs(points[k][0] - t))
    return f"{points[i][0]:.1f} s after the trigger\ntrack {points[i][1]:.1f} % slow"
