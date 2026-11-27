"""The Apparent Diffusion section's numbers, with no Qt anywhere in it.

`ui/workspace/appd_section.py` is the Qt shell; this is what it draws from. They are split because of the Thin Qt rule
(`tests/test_ui_thin_qt.py` walks the whole GUI layer): a wrong number must fail in a headless test, not look subtly
wrong on screen, and a process-pool worker imports this module without importing Qt.

ONE per-track apparent D everywhere (`appd.track_appd`: all single-frame steps, one value per track, fitted with each
track's own gamma shape) and ONE computation per condition (`appd_condition`, picklable, plain data), from which the
page helpers below make what each page paints -- and what each page's Export draws (`figure_export`'s "appd_*" kinds).

The section's first version -- a violin "Distribution" over whole-track means with gaps counted as steps, and a
"Bubble Plot" over 4-step sub-tracks that counted a long track several times -- was replaced on 16 Oct 2026 and its
code removed the same night (asked: "remove old appd code"); the record is in `docs/claude/sqd_and_statistics.md`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "SECTION_COUNTS", "SECTION_DEFAULTS", "SECTION_HELP", "SECTION_MIN_STEPS", "SECTION_UNCERTAINTY",
    "SECTION_UNCERTAINTY_LABELS",
    "appd_condition", "bic_text", "cell_length_payload", "cells_payload", "fit_text", "gallery_lines",
    "histogram_payload", "length_payload", "one_population", "switching_payload", "ONE_POPULATION",
]

#: A runner-up within this dBIC of the best is "inconclusive", the rule everywhere here
#: (`appd_gamma.select_n_pop_gamma`'s "inconclusive against the best").
_INCONCLUSIVE_DBIC = 2.0


#: What a condition fitted with ONE population has in place of a slow share (4 Nov 2026, found by the overnight tester:
#: a free protein's every cell read "slow share 100 %"): the Per cell chart, the cell-length card and the Length check
#: say this instead of drawing a share. The Checks page's switching check says the same of itself.
ONE_POPULATION = "one population, no slow share"


def one_population(result) -> bool:
    """Whether ``result`` (`appd_condition`) was fitted with one population, so its "slowest population" is all of it."""
    return int((result or {}).get("n_pop") or 0) == 1


def _day_label(day):
    """A replicate day as a plain label: an integer ordinal when it is one, else its text.
    A label only -- a day number is an ordinal WITHIN its condition (RULE 0)."""
    try:
        as_float = float(day)
    except (TypeError, ValueError):
        return str(day)
    return int(as_float) if as_float.is_integer() else str(day)


# =====================================================================================================================
# THE APPARENT DIFFUSION SECTION, REBUILT (16 Oct 2026)
# =====================================================================================================================
#
# Asked: "take a look at matlab version, think about what we learned ... and then lets build it". ONE per-track
# apparent D everywhere in the section (`appd.track_appd`: all single-frame steps, one value per track, fitted with
# each track's own gamma shape), and ONE computation per condition (`appd_condition`) from which every page draws:
#
# * Populations: the bubbles (`appd_bubbles`, the SQD bubble payload's shape so the chart, brackets, table and export
#   need nothing new): D shared over the condition's days, fractions per day, mean +- SD over days;
# * Distributions: the histogram on a log-D axis with each fit's expected counts per bin (exact, from every track's
#   own gamma), the cut(s) where the chosen fit's populations cross, what an immobile molecule reads (sigma^2/tau) and
#   the detection ceiling, and BIC for 1 to 3 populations;
# * Per cell: each cell's slow share (the mean posterior of its tracks), by day;
# * Switching check: the histogram SQD's populations predict if no molecule switched within its track, beside the
#   observed one -- a track that switched reads in between, which is why appD's differences are smaller than SQD's;
# * Length check: apparent D and slow share against track length (survivorship: long tracks are the slow ones);
# * Track gallery: every track's value, steps, cell and posterior, for picking examples from a range.
#
# What v2 did and why it is not repeated (the MATLAB study of 16 Oct 2026): an MSD-slope D over lags 1-7 of tracks of
# 9 or more points, filtered by R^2 (circular, 28 % of tracks lost); its gamma shape (8) did not match the 7 lags; its
# BIC multiplied densities and underflowed in every fit; the best start was chosen by R^2 from unseeded random starts;
# its "+- sd" was a 95 % half-width, the third population's invented; a failed fit returned zeros; the bubble plot drew
# three populations whatever was best.

#: The population counts every condition is fitted with by default (the Distributions page's BIC table compares them;
#: 4 too when Best looks that far or a 4 is ticked, `appd_fits.fit_counts`).
SECTION_COUNTS = (1, 2, 3)
#: THE RUN'S SETTINGS ARE `appd_fits`' since 3 Nov 2026 (models per condition side by side, Force same D across,
#: SQD's five Uncertainty choices), named here too for the pages that read them.
from .appd_fits import DEFAULTS as SECTION_DEFAULTS  # noqa: E402
from .appd_fits import HELP as SECTION_HELP  # noqa: E402
from .appd_fits import SECTION_MIN_STEPS, SECTION_UNCERTAINTY, SECTION_UNCERTAINTY_LABELS  # noqa: E402,F401

#: The cell bootstrap: resamples, and the interval it reports.
CELL_BOOT = 200
#: The histogram: log-spaced bins over the middle 99.8 % of the values.
HIST_BINS = 48
#: Track-length bins for the Length check, in steps: (lowest, highest) inclusive; None is open-ended.
LENGTH_BINS = ((4, 5), (6, 9), (10, 19), (20, 49), (50, None))
#: The Switching check's "between the states" band: clearly away from both SQD states, by this factor.
BETWEEN_FACTOR = 2.0


def _fit_expected(edges, x, n, D, fractions, o=None):
    """A fit's expected tracks per bin over the tracks it was fitted to: the moving ones (every track, without an
    exact zero). Where the fit holds the immobile population of exact-zero tracks (D 0, `appd_gamma`'s module head)
    its moving populations are the mixture of the moving tracks, so each is summed over their shapes at its share of
    them; the immobile one has no track in any bin (`_expected_counts`). ``o``: each track's noise offset (per-spot
    precision)."""
    D = np.asarray(D, dtype=float)
    f = np.asarray(fractions, dtype=float)
    moving = np.asarray(x, dtype=float) > 0
    if np.any(D == 0):
        rest = 1.0 - float(f[D == 0].sum())
        f = np.where(D == 0, f, f / rest if rest > 0 else 0.0)
    return _expected_counts(edges, np.asarray(n, dtype=float)[moving], D, f,
                            None if o is None else np.asarray(o, dtype=float)[moving])


def _expected_counts(edges, n, D, fractions, o=None):
    """Each population's EXPECTED count in every bin: ``f_k sum_i [G(hi; n_i, D_k/n_i) - G(lo; n_i, D_k/n_i)]`` over
    the tracks' own step counts -- the exact bin integral of the fitted mixture, as SQD's jump histogram uses. ``o``:
    each track's noise offset (per-spot precision), its population's mean then ``D_k + o_i`` and its shape the fit's
    effective one (`appd_gamma.shape_and_scale`)."""
    from scipy import stats

    from .appd_gamma import shape_and_scale

    edges = np.asarray(edges, dtype=float)
    n = np.asarray(n, dtype=float)
    o = np.zeros(len(n)) if o is None else np.asarray(o, dtype=float)
    out = []
    for d, f in zip(D, fractions):
        if float(d) == 0:
            # the immobile population of exact-zero jumps (`sqd`'s module head): its tracks' apparent D is 0, in no bin
            out.append([0.0] * (len(edges) - 1))
            continue
        shape, scale = shape_and_scale(float(d), n, o)
        cdf = stats.gamma.cdf(edges[None, :], a=shape[:, None], scale=scale[:, None])
        out.append((float(f) * np.diff(cdf, axis=1).sum(axis=0)).tolist())
    return out


def _posteriors(x, n, D, fractions, o=None):
    """Each track's posterior for each population (rows: tracks), with its own shape: `appd_gamma.posteriors`, the
    one formula (an exact zero is the immobile population's where the fit holds one); ``o`` each track's noise offset
    (per-spot precision)."""
    from .appd_gamma import posteriors

    return posteriors(x, n, D, fractions, offset=o)


def _calibration(precision):
    """The Run's per-spot calibration ``(c_nm, p, source)`` as a `precision.Calibration`, None without one."""
    if precision is None:
        return None
    from .precision import Calibration

    c_nm, p, source = precision
    return Calibration(c_nm=float(c_nm), p=float(p), source=str(source))


def _fit_summary(fit) -> dict:
    out = {"n_pop": int(fit.n_pop), "D": [float(d) for d in fit.D], "fraction": [float(f) for f in fit.fraction],
           "bic": float(fit.bic), "loglik": float(fit.loglik),
           "notes": [m for m in (fit.collapsed_note, fit.slow_convergence_note) if m]}
    if getattr(fit, "immobile", False):
        # the first population is the immobile one of exact-zero tracks (D 0, counted), only where the fit holds it
        out["immobile"] = True
    return out


def appd_condition(project, condition: str, *, models=None, scope: str = "condition", uncertainty: str = "day",
                   min_steps: int = 4, best_max_pop: int = 3, chunks: int = 5, states: str | None = None,
                   precision=None) -> dict:
    """Everything the Apparent Diffusion section draws for ONE condition, as plain data (picklable; one process-pool
    job per condition). RAISES `AppDError` / `AppDGammaError` when the condition cannot be fitted at all.

    The keywords are `appd_fits.condition_settings`' (the condition's models, its Force same D across, Uncertainty,
    shortest track, Best's range, chunks); ``states`` is the one model of a call from before 3 Nov 2026;
    ``precision`` the Run's per-spot calibration ``(c_nm, p, source)`` (Localisation error Per spot, 6 Nov 2026,
    `appd_fits.with_precision`): every fit then gives each track its own noise offset (`appd_gamma`'s module head),
    and the Checks page's SQD comparisons are the per-spot SQD fit.

    Keys: ``condition, settings, n_tracks, n_too_short, tau_s, fits {count: summary or {"failed"}}, selection,
    n_pop, chosen_by, pooled, designs {model: design}, pseudo, days, notes, cell_ci, cell_withheld, histogram,
    cuts_um2_s, never_most_likely, cut_n_steps, immobile_um2_s, ceiling_um2_s, cells, length, switching, tracks``.
    ``designs`` are the condition's own bubbles per model (`appd_fits.condition_designs`; empty under a set-wide
    Force same D across, made by `appd_fits.joint_designs` once every condition has landed). ``n_pop`` is the ONE
    count the other pages read (`appd_fits.primary_count`); ``days`` and ``cell_ci`` are that count's design's
    replicates and interval, as before."""
    from . import appd_fits as af
    from . import appd_gamma as ag
    from .appd import NOISE_OFFSET_COLUMN, track_appd
    from .spatialappd import appd_crossings

    if models is None:
        models = (states,) if states is not None else tuple(af.DEFAULTS["models"])
    if scope not in ("condition", "replicate", "global", "group"):
        raise ValueError(f"scope must be condition, replicate, global or group, got {scope!r}")
    checked = af.run_config({"models": list(models), "uncertainty": uncertainty, "min_steps": min_steps,
                             "best_max_pop": best_max_pop, "chunks": chunks})
    settings = dict(af.condition_settings(checked, condition), scope=scope)
    calibration = _calibration(precision)
    if calibration is not None:
        settings["precision"] = [float(calibration.c_nm), float(calibration.p), str(calibration.source)]
    table = project.condition(condition)
    if calibration is not None:
        from .precision import carries_precision

        if not carries_precision(table):
            # NO PRECISION IN THESE TRACKS (a truth condition, a track table): fitted without the model, said
            calibration = None
            settings["precision_off"] = True
    tracks = track_appd(table, min_steps=settings["min_steps"], calibration=calibration)
    x = tracks["appd_um2_per_s"].to_numpy(dtype=float)
    n = tracks["n_steps"].to_numpy(dtype=float)
    # PER SPOT: each track's own noise offset, in every fit, density and posterior below (None: the model as before)
    o = None if calibration is None else tracks[NOISE_OFFSET_COLUMN].to_numpy(dtype=float)
    # EXACT-ZERO TRACKS (ruled 4 Nov 2026, `appd_gamma`'s module head): decided ONCE for the condition, every fit below
    # (each count, each replicate, chunk, fold, resample) under that decision, so all hold the same populations
    zeros = ag.zero_tracks(x, n, tau_s=float(tracks.attrs["tau_s"]))
    rule = ag.zero_rule_of(zeros)

    # EVERY COUNT, ONCE: the Distributions page compares them, Best picks among them (up to its range), the rest
    # reuse them
    counts = af.fit_counts(settings["models"], settings["best_max_pop"])
    fits, summaries = {}, {}
    for k in counts:
        try:
            fits[k] = ag.fit_appd_gamma(x, n_steps=n, n_pop=k, zeros=rule, offset=o)
            summaries[k] = _fit_summary(fits[k])
        except ag.AppDGammaError as exc:
            summaries[k] = {"n_pop": k, "failed": str(exc)}
    in_range = {k: f for k, f in fits.items() if k <= settings["best_max_pop"]}
    if not in_range:
        raise ag.AppDGammaError(f"{condition}: no population count from 1 to {settings['best_max_pop']} could be "
                                f"fitted (" + "; ".join(v["failed"] for v in summaries.values() if "failed" in v)
                                + ")")
    best_bic = min(f.bic for f in in_range.values())
    selection = {"rows": [{"n_pop": k, "bic": summaries[k].get("bic"),
                           "delta": (summaries[k]["bic"] - best_bic) if "bic" in summaries[k] else None,
                           "failed": summaries[k].get("failed", "")} for k in counts],
                 "recommended": int(min(in_range, key=lambda k: in_range[k].bic)),
                 "max_pop": int(settings["best_max_pop"])}
    deltas = sorted(r["delta"] for r in selection["rows"] if r["delta"] is not None and r["n_pop"] in in_range)
    selection["margin"] = deltas[1] if len(deltas) > 1 else None
    selection["inconclusive"] = bool(selection["margin"] is not None and selection["margin"] < _INCONCLUSIVE_DBIC)
    if zeros is not None and zeros.immobile and 1 in counts:
        # one population is no candidate beside the immobile one (`appd_gamma.select_n_pop_gamma`'s rule)
        selection["notes"] = [ag.one_population_note(zeros, settings["best_max_pop"])]
    n_pop = af.primary_count(settings["models"], selection)
    if n_pop not in fits:
        raise ag.AppDGammaError(f"{condition}: the {n_pop}-population fit did not converge "
                                f"({summaries[n_pop]['failed']})")
    chosen = fits[n_pop]

    # THE BUBBLES, one design per model: D forced across this condition's own replicates or free per replicate, with
    # the Uncertainty's spread (RULE 0: no other condition's replicate enters). Set-wide designs come later, joint.
    designs = af.condition_designs(table, tracks, x, n, fits, selection, settings, zeros=zeros, o=o)
    primary = (designs.get("best") if (designs.get("best") or {}).get("n_pop") == n_pop else None) \
        or designs.get(str(n_pop))
    notes = ([zeros.note] if zeros is not None else []) + list(summaries[n_pop]["notes"]) \
        + list((primary or {}).get("notes") or [])
    days = None
    if primary and primary.get("units") and "failed" not in primary:
        u = primary["units"]
        days = {"labels": list(u["labels"]), "n_tracks": list(u["n_tracks"]), "D": list(primary["D"]),
                "fractions": [list(f) for f in u["fractions"]], "mean": list(primary["fraction"]),
                "sd": list(primary["sd"])}
    cell_ci = (primary or {}).get("ci")
    cell_withheld = bool((primary or {}).get("withheld"))
    pseudo, chunk_labels, chunk_splits = None, None, None
    if settings["scope"] in af.SET_SCOPES and settings["uncertainty"] == "chunk" and not af.chunks_held(tracks):
        # the joint fit deals no chunks of its own: SQD's deal of this condition, kept for it (`joint_inputs`)
        dealt, splits, pseudo = af._deals(table, tracks, settings["chunks"])
        chunk_labels, chunk_splits = [int(v) for v in dealt], [[int(v) for v in sp] for sp in splits]

    # THE HISTOGRAM, on a log axis, with every fit's exact expected counts -- of the tracks that move: an apparent D
    # of exactly 0 has no place on a log axis (the immobile population, or zeros left out of the fit, said in notes)
    moving = x[x > 0]
    lo, hi = np.quantile(moving, [0.001, 0.999])
    edges = np.geomspace(max(lo, 1e-5), max(hi, lo * 10), HIST_BINS + 1)
    observed = np.histogram(moving, bins=edges)[0]
    expected = {k: _fit_expected(edges, x, n, fits[k].D, fits[k].fraction, o) for k in fits}

    # THE CUTS where the chosen populations cross, at the median track's step count (they move with the count) and,
    # per spot, its noise offset: a population's tracks then read D + offset on this axis of apparent D
    cut_n = int(np.median(n))
    cut_o = 0.0 if o is None else float(np.median(o))
    cuts, never = [], []
    if n_pop >= 2:
        crossing = appd_crossings(np.asarray(chosen.D, dtype=float) + cut_o, np.maximum(chosen.fraction, 1e-12),
                                  cut_n)
        # the immobile population's cut at 0 (exact-zero tracks) is no apparent D on the histogram's log axis
        cuts = [float(c) for c in crossing["cuts"] if float(c) > 0]
        never = [int(i) for i in crossing["never_most_likely"]]
    params = project.params
    immobile = float(params.D_static_error_offset)
    # THE CONDITION'S OWN CEILING, at the radius it was linked at (tester D10: free mNG at 5 px and 8 px drew both at 5)
    ceiling = params.detection_ceiling_D_for(condition)

    post = _posteriors(x, n, chosen.D, chosen.fraction, o)
    slow = post[:, 0]
    # PER CELL: each cell's slow share is the mean posterior of its tracks
    cells = []
    if "cell" in tracks.columns:
        frame = tracks.assign(_slow=slow)
        frame = frame[frame["cell"].notna()]
        for (day, movie, cell), g in frame.groupby(["day", "movie", "cell"], sort=True, observed=True):
            cells.append({"day": _day_label(day), "movie": str(movie), "cell": str(cell),
                          "n_tracks": int(len(g)), "slow_share": float(g["_slow"].mean()),
                          "median_appd": float(g["appd_um2_per_s"].median())})
        # each cell's LENGTH, from its outline (asked 16 Oct 2026: "fraction vs cell length"); None where the
        # project has no outline with a centreline for it
        from .qc_view import cell_length_by_key

        lengths = cell_length_by_key(project, [c["cell"] for c in cells])
        # and its slow share from the SQD fit (17 Oct 2026), the readout the manuscript rests on
        sqd_share = _sqd_cell_shares(table, n_pop, calibration)
        for c in cells:
            c["length_um"] = lengths.get(c["cell"])
            c["sqd_slow_share"] = sqd_share.get(c["cell"])
    # LENGTH: apparent D and slow share by how many steps a track has
    length = []
    for low, high in LENGTH_BINS:
        mine = (n >= low) & ((n <= high) if high is not None else True)
        if mine.any():
            q25, q50, q75 = np.quantile(x[mine], [0.25, 0.5, 0.75])
            length.append({"low": low, "high": high, "n_tracks": int(mine.sum()), "median": float(q50),
                           "q25": float(q25), "q75": float(q75), "slow_share": float(slow[mine].mean())})

    # FIT QUALITY (3 Nov 2026): every converged fit's ECDF residual and gap, and held-out replicates (or cells)
    quality = fit_quality(x, n, summaries, day=tracks["day"].to_numpy(),
                          cell=tracks["cell"].to_numpy() if "cell" in tracks.columns else None, zeros=rule, o=o)
    return {"condition": condition, "settings": settings, "n_tracks": int(len(x)),
            # exact-zero tracks met, and what was done (`appd_gamma.zero_tracks`): only where some were met
            **({"zero_tracks": zeros.to_dict()} if zeros is not None else {}),
            "n_too_short": int(tracks.attrs.get("n_tracks_too_short", 0)), "tau_s": float(tracks.attrs["tau_s"]),
            "fits": summaries, "selection": selection, "n_pop": n_pop,
            "chosen_by": "best" if "best" in settings["models"] else "fixed", "pooled": _fit_summary(chosen),
            "designs": designs, "pseudo": pseudo,
            "days": days, "notes": notes, "cell_ci": cell_ci, "cell_withheld": cell_withheld,
            "histogram": {"edges": edges.tolist(), "observed": observed.tolist(),
                          "expected": {str(k): v for k, v in expected.items()}},
            "cuts_um2_s": cuts, "never_most_likely": never, "cut_n_steps": cut_n,
            "immobile_um2_s": immobile, "ceiling_um2_s": None if ceiling is None else float(ceiling),
            "cells": cells, "length": length,
            "switching": _switching(table, edges, n, x, n_pop, calibration, o),
            **({"precision_left_out": int(tracks.attrs.get("precision_left_out", 0))} if o is not None else {}),
            "quality": quality,
            "tracks": {"uid": tracks["uid"].astype(str).tolist(), "appd": x.tolist(),
                       "n_steps": tracks["n_steps"].astype(int).tolist(),
                       "day": [_day_label(d) for d in tracks["day"]],
                       "movie": tracks["movie"].astype(str).tolist() if "movie" in tracks else [],
                       "cell": ([None if c is None or c != c else str(c) for c in tracks["cell"]]
                                if "cell" in tracks else []),
                       "slow": slow.tolist(),
                       **({"offset": o.tolist()} if o is not None else {}),
                       **({"chunk": chunk_labels, "chunk_splits": chunk_splits} if chunk_labels is not None
                          else {})}}


def _sqd_cell_shares(table, n_pop: int, calibration=None) -> dict:
    """Each cell's slow share FROM THE SQD FIT: every jump's probability of the slowest population under the
    condition's pooled SQD fit (step-weighted, the SQD section's default; the count the Run chose), averaged over
    the cell's jumps -- the time each cell's molecules spend in the slow state, as the SQD fractions count it.
    ``{cell: share}``; empty under one population or when the fit fails (the card then says so). ``calibration``:
    per-spot precision, the SQD fit then the per-spot one (each jump its own noise; steps lacking a spot's precision
    left out, `sqd.fit_sqd`'s rule)."""
    import pandas as pd

    from .observables import squared_displacements
    from .sqd import SQDError, fit_sqd

    if n_pop < 2:
        return {}
    sd = squared_displacements(table)
    sd = sd[sd["cell"].notna()]
    if not len(sd):
        return {}
    noise = None
    try:
        if calibration is not None:
            from .precision import PrecisionError, step_noise, usable_steps

            try:
                sd = sd.loc[usable_steps(sd)]
                noise = step_noise(sd, calibration)
            except PrecisionError:
                return {}
            fit = fit_sqd(sd, n_pop=n_pop, weighting="step", gradient="analytic", method="mle",
                          step_noise_um2=noise)
        else:
            fit = fit_sqd(sd, n_pop=n_pop, weighting="step", gradient="analytic", ecdf_max_points=2000)
    except SQDError:
        return {}
    D = np.asarray(fit.D, dtype=float)
    f = np.asarray(fit.fractions, dtype=float)
    r2 = sd["sq_disp_um2"].to_numpy(dtype=float)
    if noise is not None:
        from .sqd import posteriors

        post = posteriors(r2, D, f, float(fit.tau_lag_s), step_noise=noise)[:, int(np.argmin(D))]
    elif fit.immobile:
        # the slowest population is the immobile one of exact-zero jumps (`sqd`'s module head): `sqd.posteriors`
        from .sqd import posteriors

        post = posteriors(r2, D, f, float(fit.tau_lag_s))[:, int(np.argmin(D))]
    else:
        means = 4.0 * D * float(fit.tau_lag_s)
        log_dens = np.log(np.maximum(f, 1e-300))[None, :] - np.log(means)[None, :] - r2[:, None] / means[None, :]
        log_dens -= log_dens.max(axis=1, keepdims=True)
        dens = np.exp(log_dens)
        post = dens[:, int(np.argmin(D))] / dens.sum(axis=1)
    frame = pd.DataFrame({"cell": sd["cell"].astype(str).to_numpy(), "p": post})
    return {str(k): float(v) for k, v in frame.groupby("cell", sort=False)["p"].mean().items()}


def _switching(table, edges, n, x, n_pop: int, calibration=None, o=None) -> dict:
    """What SQD's populations predict for this histogram if no molecule switched within its track: each track in ONE
    state for all its steps, so its apparent D is Gamma(n_i, D_k / n_i) with SQD's D_k and SQD's TRACK-weighted
    fractions (each molecule once), at the tracks' own step counts. The observed excess between the states is what
    switching does. ``between`` is the band clearly away from both states (a factor `BETWEEN_FACTOR` above the
    slowest D and below the fastest). ``calibration`` / ``o``: per-spot precision -- SQD's per-spot fit, each track
    its own noise offset, and the band on the axis of apparent D at the median track's offset."""
    from .observables import squared_displacements
    from .sqd import SQDError, fit_sqd

    if n_pop < 2:
        return {"available": False, "why": "one population: nothing to switch between"}
    try:
        if calibration is not None:
            fit = fit_sqd(squared_displacements(table), n_pop=n_pop, weighting="track", gradient="analytic",
                          method="mle", precision=calibration)
        else:
            fit = fit_sqd(squared_displacements(table), n_pop=n_pop, weighting="track", gradient="analytic")
    except SQDError as exc:
        return {"available": False, "why": f"the SQD fit failed: {exc}"}
    D, f = [float(d) for d in fit.D], [float(v) for v in fit.fractions]
    predicted = np.sum(_expected_counts(edges, n, D, f, o), axis=0)
    offsets = np.zeros(len(n)) if o is None else np.asarray(o, dtype=float)
    shift = float(np.median(offsets)) if o is not None else 0.0
    # the slowest MOVING D: an immobile population of exact-zero jumps (D 0) predicts no track inside the band
    low, high = (min(d for d in D if d > 0) + shift) * BETWEEN_FACTOR, (D[-1] + shift) / BETWEEN_FACTOR
    between_obs = between_pred = None
    if high > low:
        from scipy import stats

        from .appd_gamma import shape_and_scale

        def inside_band(dk):
            shape, scale = shape_and_scale(dk, n, offsets)
            return np.mean(stats.gamma.cdf(high, a=shape, scale=scale) - stats.gamma.cdf(low, a=shape, scale=scale))

        inside = (x > low) & (x < high)
        between_obs = float(inside.mean())
        between_pred = float(sum(fk * inside_band(dk) for dk, fk in zip(D, f) if dk > 0))
    return {"available": True, "sqd_D": D, "sqd_fractions": f, "predicted": predicted.tolist(),
            "band": [low, high] if high > low else None, "between_observed": between_obs,
            "between_predicted": between_pred}


# -- what each page paints, from `appd_condition`'s result (plain data; the pages compute nothing) ----------------

def _pct(v) -> str:
    return f"{100.0 * float(v):.1f} %"


def _d3(d) -> str:
    """A population's D to 3 significant figures, or "not resolved" on the gamma mixture's floor
    (`appd_gamma.d_at_floor`, 10 Nov 2026)."""
    from .appd_gamma import d_at_floor

    return "not resolved" if d_at_floor(d) else f"{d:.3g}"


def fit_text(result: dict, n_pop: int) -> str:
    """One fit in words: its D and fractions, slowest first."""
    fit = result["fits"].get(n_pop) or result["fits"].get(str(n_pop))
    if not fit or "failed" in fit:
        return f"{n_pop} population{'' if n_pop == 1 else 's'}: did not converge"
    D = " / ".join(_d3(d) for d in fit["D"])
    f = " / ".join(f"{100 * v:.1f}" for v in fit["fraction"])
    return f"D {D} µm²/s · {f} %"


def bic_text(result: dict) -> str:
    """BIC over 1-3 populations: each count's difference to the best, and when the call is inconclusive."""
    parts = []
    immobile = bool((result.get("zero_tracks") or {}).get("immobile"))
    for row in result["selection"]["rows"]:
        k = row["n_pop"]
        name = f"{k} population{'' if k == 1 else 's'}"
        if row["delta"] is None and immobile and int(k) == 1:
            # beside the immobile population of exact-zero tracks one population is no candidate (`appd_gamma`)
            parts.append(f"{name} not compared (exact-zero tracks)")
        elif row["delta"] is None:
            parts.append(f"{name} did not converge")
        elif row["delta"] == 0:
            parts.append(f"{name} best")
        else:
            parts.append(f"{name} +{row['delta']:.0f}")
    text = "BIC: " + " · ".join(parts)
    if result["selection"]["inconclusive"]:
        text += " (inconclusive: the runner-up is within 2)"
    return text


#: A population's expected count below this share of the fit's peak is not drawn (`histogram_payload`).
DRAWN_FLOOR = 0.005


def histogram_payload(result: dict, n_pop: int) -> dict:
    """The Distributions chart for ONE fit, in the shape it had before 3 Nov 2026: bin edges and centres (geometric),
    the observed counts, each population's expected counts and their sum, and the marks -- the cut(s) where the
    Run's populations cross (only for its own fit), what an immobile molecule reads, the detection ceiling. Since the
    page draws a set of fits it is `distribution_payload` at that one count (the one computation); a count that did
    not converge has no curve."""
    d = distribution_payload(result, (int(n_pop),))
    fit = d["fits"].get(int(n_pop))
    if fit is None:                                  # did not converge: the bars alone
        return {"edges": d["edges"], "centres": d["centres"], "observed": d["observed"], "components": [],
                "components_drawn": [], "total": [], "y_max": float(max(d["observed"] or [1])) * 1.08, "cuts": [],
                "immobile": d["immobile"], "ceiling": d["ceiling"]}
    return {"edges": d["edges"], "centres": d["centres"], "observed": d["observed"],
            "components": fit["components"], "components_drawn": fit["components_drawn"], "total": fit["expected"],
            "y_max": d["y_max"], "cuts": d["cuts"], "immobile": d["immobile"], "ceiling": d["ceiling"]}


def gallery_lines(result: dict, lo: float, hi: float, limit: int = 12) -> dict:
    """The tracks whose apparent D lies in [lo, hi], longest first: ``{"n": how many, "lines": [{"uid", "title",
    "detail"}]}`` -- what the Track gallery lists for a range picked on the histogram."""
    tr = result["tracks"]
    picked = [i for i, v in enumerate(tr["appd"]) if lo <= v <= hi]
    picked.sort(key=lambda i: -tr["n_steps"][i])
    lines = []
    for i in picked[:limit]:
        movie = tr["movie"][i] if tr["movie"] else ""
        cell = tr["cell"][i] if tr["cell"] else None
        short = str(movie).split("_", 1)[-1] if movie else ""
        lines.append({"uid": tr["uid"][i],
                      "title": f"{tr['appd'][i]:.3g} µm²/s · {tr['n_steps'][i]} steps",
                      "detail": " · ".join(b for b in (short, f"cell {str(cell).rsplit(':', 1)[-1]}" if cell else "",
                                                     f"replicate {tr['day'][i]}", f"slow {_pct(tr['slow'][i])}") if b)})
    return {"n": len(picked), "lines": lines}


def grouped_cells(cells: dict, factors: dict, nesting) -> dict:
    """``cells`` (`cells_payload`) with its columns in the nesting's order and ``"layout"`` their grouped positions
    (`factor_layout.columns`, folder designs); unchanged without two factors."""
    from .factor_layout import columns, nested

    nesting = tuple(nesting or ())
    if not factors or len(nesting) < 2 or not cells.get("columns"):
        return cells
    by_key = {c["condition"]: c for c in cells["columns"]}
    order = nested(list(by_key), factors, nesting)
    layout = columns(order, factors, nesting)
    return dict(cells, columns=[by_key[k] for k in order], layout=layout if layout.get("groups") else {})


def cells_payload(results: dict, conditions: list, min_tracks: int = 3) -> dict:
    """The Per cell SuperPlot: per condition each cell's slow share (%) as a dot -- jittered deterministically,
    sized by its tracks, shaded by its day -- each day's mean, and the condition's mean +- SD over the day means.
    Cells with fewer than ``min_tracks`` tracks are left out and counted (one track is a 0 or 100 % guess)."""
    import numpy as np

    rng = np.random.default_rng(0)
    columns = []
    for c in conditions:
        res = results.get(c)
        if res is None or "error" in res:
            continue
        cells = [cell for cell in res["cells"] if cell["n_tracks"] >= min_tracks]
        day_order = sorted({cell["day"] for cell in res["cells"]}, key=str)
        if one_population(res):
            # no slow share: the one population is every track, so each cell would read 100 %
            columns.append({"condition": c, "points": [], "day_means": [], "mean": None, "sd": None,
                            "n_days": len(day_order), "hidden": 0, "n_cells": len(cells), "why": ONE_POPULATION})
            continue
        points = [{"x": float(rng.uniform(-0.28, 0.28)), "y": 100.0 * cell["slow_share"],
                   "n": cell["n_tracks"], "day_index": day_order.index(cell["day"]), "day": cell["day"],
                   "movie": str(cell["movie"]), "cell": str(cell["cell"]),
                   "label": f"{str(cell['movie']).split('_', 1)[-1]} · cell {str(cell['cell']).rsplit(':', 1)[-1]}"
                            f" · replicate {cell['day']} · {cell['n_tracks']} tracks · slow {_pct(cell['slow_share'])}"}
                  for cell in cells]
        means = []
        for i, day in enumerate(day_order):
            mine = [p["y"] for p in points if p["day_index"] == i]
            if mine:
                means.append({"day": day, "day_index": i, "y": float(np.mean(mine)), "n_cells": len(mine)})
        ys = [m["y"] for m in means]
        columns.append({"condition": c, "points": points, "day_means": means,
                        "mean": float(np.mean(ys)) if ys else None,
                        "sd": float(np.std(ys, ddof=1)) if len(ys) > 1 else None,
                        "n_days": len(day_order), "hidden": len(res["cells"]) - len(cells),
                        "n_cells": len(points)})
    hidden = sum(col["hidden"] for col in columns)
    single = [col["condition"] for col in columns if col.get("why")]
    caption = ("each dot one cell's slow share (the mean posterior of its tracks), sized by its tracks and shaded by "
               "its replicate; open circles are each replicate's mean over its cells, the line their mean ± SD"
               + (f"; {hidden} cell{'' if hidden == 1 else 's'} under {min_tracks} tracks left out" if hidden else "")
               + (f"; {', '.join(single)}: {ONE_POPULATION}" if single else ""))
    return {"columns": columns, "caption": caption, "min_tracks": min_tracks}


def switching_payload(result: dict) -> dict | None:
    """The Switching check: the observed histogram against what SQD's populations predict with no switching inside
    a track, the band between the states, and the sentence that reads it. None when it is not available."""
    s = result.get("switching") or {}
    if not s.get("available"):
        return {"available": False, "why": s.get("why", "not computed")}
    hist = result["histogram"]
    edges = [float(e) for e in hist["edges"]]
    centres = [(a * b) ** 0.5 for a, b in zip(edges, edges[1:])]
    peak = max([*hist["observed"], *s["predicted"], 1])
    text = (f"SQD's populations (track-weighted): D {' / '.join(f'{d:.3g}' for d in s['sqd_D'])} µm²/s at "
            f"{' / '.join(f'{100 * f:.1f}' for f in s['sqd_fractions'])} %. ")
    if s["band"] is not None:
        lo, hi = s["band"]
        text += (f"Between the states ({lo:.3g}–{hi:.3g} µm²/s): {_pct(s['between_observed'])} of the tracks, "
                 f"{_pct(s['between_predicted'])} if no molecule switched within its track.")
    return {"available": True, "edges": edges, "centres": centres, "observed": [int(v) for v in hist["observed"]],
            "predicted": [float(v) for v in s["predicted"]], "band": s["band"], "y_max": float(peak) * 1.08,
            "text": text}


#: The cell-length comparison's classes: quartiles of every condition's measured cells together, and the fewest cells
#: a condition needs in a class for that class to enter its length-matched value.
LENGTH_CLASSES = 4
LENGTH_CLASS_MIN_CELLS = 2


#: Where the cell-length card's slow share comes from: the apparent-D fit (each track's posterior) or the SQD fit
#: (each jump's posterior), averaged per cell.
CELL_SHARE_SOURCES = ("appd", "sqd")
CELL_SHARE_LABELS = ("Apparent D", "SQD")


def cell_length_payload(results: dict, conditions: list, min_tracks: int = 3, source: str = "appd") -> dict:
    """FRACTION AGAINST CELL LENGTH (asked 16 Oct 2026): every cell's slow share (the mean posterior of its tracks,
    the Per cell chart's) against its length, and a LENGTH-MATCHED comparison of the conditions.

    Classes are the quartiles of all conditions' measured cells together. A condition's raw value is the mean over
    its cells; its length-matched value is the mean of its per-class means weighted by how many cells of ALL
    conditions fall in each class -- what it would read with everyone's length distribution -- over the classes
    where every condition has at least `LENGTH_CLASS_MIN_CELLS` cells. Cells under ``min_tracks`` tracks are left
    out, as on the Per cell chart. Descriptive, no test: cells within a day are not independent, and days are the
    unit of any comparison (RULE 0: each condition's own cells, nothing paired).

    ``{"available", "why", "points": {condition: [{"x", "y", "n", "day"}]}, "binned": {condition: [(centre, mean)]},
    "edges", "columns", "rows", "text", "caption", "x_range"}``."""
    import numpy as np

    if source not in CELL_SHARE_SOURCES:
        raise ValueError(f"source must be one of {CELL_SHARE_SOURCES}, got {source!r}")
    share_key = "slow_share" if source == "appd" else "sqd_slow_share"
    measured, single = {}, []
    for c in conditions:
        res = results.get(c)
        if not res or "error" in res:
            continue
        if one_population(res):
            single.append(c)                    # no slow share: every cell would read 100 % (appD) or none (SQD)
            continue
        cells = [dict(cell, slow_share=cell[share_key]) for cell in res.get("cells") or []
                 if cell["n_tracks"] >= min_tracks and cell.get("length_um") is not None
                 and cell.get(share_key) is not None]
        if cells:
            measured[c] = cells
    if not measured:
        fitted = [c for c in conditions if results.get(c) and "error" not in results[c]]
        if single and len(single) == len(fitted):
            return {"available": False, "why": f"every condition was fitted with {ONE_POPULATION}"}
        why = ("no cell has a measured length (the outlines need a centreline)" if source == "appd"
               else "no cell has both a measured length and an SQD slow share (one population, or the fit failed)")
        return {"available": False, "why": why}
    pooled = np.array([cell["length_um"] for cells in measured.values() for cell in cells], dtype=float)
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, LENGTH_CLASSES + 1))
    edges[-1] += 1e-9
    weights = np.histogram(pooled, bins=edges)[0].astype(float)

    def klass(length):
        return int(min(LENGTH_CLASSES - 1, np.searchsorted(edges, length, side="right") - 1))

    points, binned, per_class = {}, {}, {}
    for c, cells in measured.items():
        points[c] = [{"x": cell["length_um"], "y": 100.0 * cell["slow_share"], "n": cell["n_tracks"],
                      "day": cell["day"]} for cell in cells]
        groups = [[] for _ in range(LENGTH_CLASSES)]
        for cell in cells:
            groups[klass(cell["length_um"])].append(100.0 * cell["slow_share"])
        per_class[c] = groups
        binned[c] = [(float((edges[b] + edges[b + 1]) / 2.0), float(np.mean(g))) for b, g in enumerate(groups) if g]
    shared = [b for b in range(LENGTH_CLASSES)
              if all(len(per_class[c][b]) >= LENGTH_CLASS_MIN_CELLS for c in measured)]
    rows, raws, matched = [], {}, {}
    for c, cells in measured.items():
        ys = np.array([100.0 * cell["slow_share"] for cell in cells], dtype=float)
        xs = np.array([cell["length_um"] for cell in cells], dtype=float)
        raws[c] = float(ys.mean())
        if shared:
            w = weights[shared]
            matched[c] = float(np.sum(w * np.array([np.mean(per_class[c][b]) for b in shared])) / w.sum())
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(cells) >= 5 and np.ptp(xs) > 0 else None
        rows.append([c, f"{len(cells):,}", f"{np.median(xs):.2f}", f"{raws[c]:.1f}",
                     "–" if c not in matched else f"{matched[c]:.1f}",
                     "–" if slope is None else f"{slope:+.1f}"])
    text = ""
    if len(measured) >= 2:
        spread_raw = max(raws.values()) - min(raws.values())
        text = f"spread between conditions {spread_raw:.1f} points"
        if len(matched) == len(measured):
            spread_m = max(matched.values()) - min(matched.values())
            text += f"; at matched cell length {spread_m:.1f} points ({len(shared)} of {LENGTH_CLASSES} length classes)"
        else:
            text += "; no length class holds cells of every condition, so nothing is matched"
    return {"available": True, "points": points, "binned": binned, "edges": [float(e) for e in edges],
            "columns": ["Condition", "Cells", "Median length (µm)", "Slow share (%)", "Length-matched (%)",
                        "Slope (points per µm)"],
            "rows": rows, "text": text,
            "source": source,
            "caption": (("each dot one cell's slow share (from the SQD fit: the mean probability of its jumps being "
                         "in the slowest population)" if source == "sqd" else "each dot one cell's slow share")
                        + " against its length (cells under "
                        f"{min_tracks} tracks left out); lines, each condition's mean per length class (quartiles of "
                        "all cells); length-matched: the class means weighted by every condition's cells together. "
                        "Descriptive: cells within a replicate are not independent"
                        + (f". Not drawn ({ONE_POPULATION}): {', '.join(single)}" if single else "")),
            "x_range": (float(pooled.min()), float(pooled.max()))}


#: A track-length bin with fewer tracks than this is drawn faded and never quoted: one track is a 0 or 100 % guess.
LENGTH_MIN_TRACKS = 10


def length_payload(result: dict) -> dict:
    """The Length check: per track-length bin its tracks, slow share (%) and apparent D (median, quartiles); ``few``
    marks a bin under `LENGTH_MIN_TRACKS`, which the sentence leaves out."""
    single = one_population(result)
    bins = []
    for b in result["length"]:
        name = f"{b['low']}–{b['high']}" if b["high"] is not None else f"{b['low']}+"
        bins.append({"label": f"{name} steps", "n": b["n_tracks"],
                     "slow": None if single else 100.0 * b["slow_share"],
                     "median": b["median"], "q25": b["q25"], "q75": b["q75"],
                     "few": b["n_tracks"] < LENGTH_MIN_TRACKS})
    if single:
        # every track is the one population: a share by length would read 100 % in every bin
        return {"bins": bins, "available": False, "why": ONE_POPULATION,
                "text": f"not available: {ONE_POPULATION} (each bin's median apparent D is under it)"}
    quoted = [b for b in bins if not b["few"]]
    if len(quoted) >= 2:
        first, last = quoted[0], quoted[-1]
        text = (f"slow share {first['slow']:.0f} % in {first['label']}, {last['slow']:.0f} % in {last['label']} "
                f"({last['n']:,} tracks)"
                + (": long tracks are the slow ones, so a track-length cut moves the fractions"
                   if last["slow"] > first["slow"] + 5 else ""))
    else:
        text = "too few long tracks to compare lengths"
    few = [b for b in bins if b["few"]]
    if few:
        text += f"; faded: under {LENGTH_MIN_TRACKS} tracks"
    return {"bins": bins, "text": text, "available": True}


# =====================================================================================================================
# THE DISTRIBUTIONS PAGE AT THE JUMP DISTANCES PAGE'S LEVEL (3 Nov 2026)
# =====================================================================================================================
#
# Asked on the audit of the section ("start building everything"): the Distributions page drew ONE fit at a time
# ("Fit drawn" 1 | 2 | 3) and judged the counts by BIC alone. It now reads as the SQD section's Jump distances page
# does, whose rules it REUSES (`jumps_view`) rather than copies:
#
# * SHOW IS A SET OF FITS (`jumps_view.toggle_show`, `drawn_fits`, `in_colour`, `components_drawn`): each count that
#   converged is a toggle and the last one on stays on. One fit alone is drawn in the ink with its populations in
#   shades of the condition's colour (and the cut, when it is the Run's count); two or more each in its own colour and
#   v2's line style (`jumps_view.FIT_COLOURS`, `FIT_STYLE_WORDS`), their populations optional (Components). A card
#   starts on the Run's count (`default_show`). The counts are whatever the result holds, never a fixed 1-3.
# * RESIDUALS under the histogram: the PEARSON residual per bin, (observed - expected) / sqrt(expected), the expected
#   count being the exact per-track gamma integral the histogram already draws; a bin expecting under
#   `PEARSON_MIN_EXPECTED` track carries none (one stray track there would read as a residual of tens). And the ECDF
#   residual: the mixture's CDF (every track its own gamma shape) minus the empirical one, unbinned, at
#   `jumps_view.ECDF_POINTS` points.
# * A FIT-QUALITY TABLE (`quality_table`): per count D and fractions, ΔBIC (the Run's maximum-likelihood fits, BIC's n
#   the tracks), the largest ECDF gap (descriptive, no p-value, as on the jump page) and CROSS-VALIDATION over
#   independent units -- each replicate held out in turn when there are two or more, else folds of cells -- fitted
#   without it and scored by the exact log-likelihood of its tracks (`fit_quality`). A population that only fits noise
#   buys nothing on tracks it never saw.
# * THE CURVES ARE THE POOLED FIT of every track (`pooled_note`): under Uncertainty Replicates the Populations bubbles
#   show the D shared across the condition's replicates, a slightly different fit, and the card says so.
# * ALL CONDITIONS: every condition's histogram as its card draws it, on one page and in one figure
#   (`figure_export`'s "appd_overview").

__all__ += ["CV_STARTS", "PEARSON_MIN_EXPECTED", "converged_fits", "default_show", "distribution_payload",
            "fit_curves", "fit_quality", "pooled_note", "quality_table", "quality_verdict", "show_choices"]

#: A bin expecting fewer tracks than this carries no Pearson residual: its value would be one stray track over a
#: near-zero square root.
PEARSON_MIN_EXPECTED = 1.0
#: The largest ECDF gap is measured on at most this many of the sorted values (evenly spaced in rank).
KS_MAX_POINTS = 5000
#: Cross-validation: the multistart of each held-out fit (the Run's own fits use 24), and the fewest cells to fold.
CV_STARTS = 8
CV_MIN_CELLS = 4


def converged_fits(result: dict) -> dict:
    """``{count: fit summary}``: the result's fits that converged, keyed by their count as an int (however it
    travelled). The one reading of "which fits exist" for the page and the figures."""
    return {int(k): v for k, v in (result.get("fits") or {}).items() if v and "failed" not in v}


def show_choices(result: dict) -> tuple:
    """The Show pills, ``((1, "1"), ...)``: every count that converged (`jumps_view.show_choices`)."""
    from .jumps_view import show_choices as _choices

    return _choices(converged_fits(result))


def default_show(result: dict) -> tuple:
    """Where a card's Show starts: the Run's own count, else the jump page's rule (BIC's count, else two, else the
    fewest made)."""
    from .jumps_view import default_show as _default

    fits = converged_fits(result)
    k = result.get("n_pop")
    if k is not None and int(k) in fits:
        return (int(k),)
    return _default({"fits": fits, "bic_best": (result.get("selection") or {}).get("recommended")})


def _drawn_floor(components: list, total: list) -> list:
    """Each population's expected counts with every bin under `DRAWN_FLOOR` of the fit's peak as None: a curve is
    drawn only where it carries something (it ran along the floor under the bars, 16 Oct 2026)."""
    floor = DRAWN_FLOOR * max(total or [1.0])
    return [[v if v >= floor else None for v in comp] for comp in components]


def fit_curves(result: dict) -> dict:
    """Every converged fit as the page draws it: ``{count: {"D", "fraction", "components" (expected tracks per bin,
    per population), "components_drawn" (the same, None where it carries nothing), "expected" (their sum),
    "pearson" (per bin, None under `PEARSON_MIN_EXPECTED`), "ecdf_residual" ([(apparent D, model - ECDF)]; empty for
    a result made before 3 Nov 2026), "ks" (the largest ECDF gap, or None)}}``."""
    import math

    hist = result["histogram"]
    observed = hist["observed"]
    quality = (result.get("quality") or {}).get("fits") or {}
    out = {}
    for k, fit in converged_fits(result).items():
        components = hist["expected"].get(str(k)) or hist["expected"].get(k) or []
        total = [float(sum(col)) for col in zip(*components)] if components else []
        pearson = [None if e < PEARSON_MIN_EXPECTED else (float(o) - e) / math.sqrt(e)
                   for o, e in zip(observed, total)]
        mine = quality.get(k) or quality.get(str(k)) or {}
        out[k] = {"D": [float(d) for d in fit["D"]], "fraction": [float(f) for f in fit["fraction"]],
                  "components": components, "components_drawn": _drawn_floor(components, total), "expected": total,
                  "pearson": pearson,
                  "ecdf_residual": [(float(a), float(b)) for a, b in mine.get("ecdf_residual") or ()],
                  "ks": mine.get("ks")}
    return out


def distribution_payload(result: dict, show=None, *, components: bool = True) -> dict:
    """What a Distributions card (or an All conditions tile) draws for the counts ``show`` (any Show value; the card's
    start, `default_show`, when None): the bins, the observed tracks, the fits drawn (`jumps_view.drawn_fits`) with
    whose populations are drawn (`jumps_view.components_drawn`: none with Components off; a one-population fit has
    none of its own), the y axis's top, and the marks -- the cut where the Run's populations cross, only when that
    fit is drawn alone; what an immobile molecule reads; the detection ceiling. ``"pooled"``: the curves are the
    pooled fit of every track (`pooled_note`)."""
    from .jumps_view import components_drawn, drawn_fits, histogram_top, in_colour

    curves = fit_curves(result)
    hist = result["histogram"]
    edges = [float(e) for e in hist["edges"]]
    centres = [(a * b) ** 0.5 for a, b in zip(edges, edges[1:])]
    observed = [int(v) for v in hist["observed"]]
    drawn = drawn_fits(curves, default_show(result) if show is None else show)
    parts = components_drawn(curves, drawn, components)
    top = histogram_top(observed, curves, drawn) if drawn else float(max(observed or [1]))
    n_pop = result.get("n_pop")
    return {"edges": edges, "centres": centres, "observed": observed, "show": tuple(drawn), "every": in_colour(drawn),
            "fits": {k: curves[k] for k in drawn}, "parts": list(parts), "y_max": float(top) * 1.08,
            "cuts": list(result.get("cuts_um2_s") or []) if drawn == [n_pop] else [],
            "immobile": result.get("immobile_um2_s"), "ceiling": result.get("ceiling_um2_s"), "pooled": True,
            "ecdf_ok": bool(drawn) and all(curves[k]["ecdf_residual"] for k in drawn)}


def pooled_note(result: dict) -> str:
    """Which fit the Distributions page draws: the POOLED fit of every track. Under Uncertainty Replicates the
    Populations bubbles show the D shared across the condition's replicates (fractions per replicate), a slightly
    different number, which is said with both."""
    text = "Curves: the pooled fit of every track."
    days = result.get("days") or {}
    pooled = (result.get("pooled") or {}).get("D")
    if days.get("D") and pooled:
        shared = " / ".join(f"{d:.3g}" for d in days["D"])
        own = " / ".join(f"{d:.3g}" for d in pooled)
        text += (f" The Populations bubbles show D shared across this condition's replicates, {shared} µm²/s; the "
                 f"pooled fit gives {own} µm²/s.")
    return text


def _mixture_cdf(points, shapes, weights, D, fractions, inclusive: bool = True, offsets=None):
    """The mixture's CDF at ``points`` over a sample whose tracks have step counts ``shapes`` in proportions
    ``weights``: ``sum_k f_k sum_s w_s G(x; s, D_k / s)``, every track its own gamma shape. The immobile population of
    exact-zero tracks (a D of 0) is a step at 0: P(X <= x) holds it at 0 (``inclusive``), P(X < x) only above.
    ``offsets``: per-spot precision, each (shape, offset) group's noise offset beside its shape (`_shape_groups`),
    its gamma then the fit's effective shape (`appd_gamma.shape_and_scale`)."""
    from scipy import stats

    from .appd_gamma import shape_and_scale

    points = np.asarray(points, dtype=float)
    o = np.zeros(len(shapes)) if offsets is None else np.asarray(offsets, dtype=float)
    total = np.zeros(len(points))
    for d, f in zip(D, fractions):
        if float(d) == 0:
            total += float(f) * ((points >= 0) if inclusive else (points > 0))
            continue
        shape, scale = shape_and_scale(float(d), shapes, o)
        g = stats.gamma.cdf(points[:, None], a=shape[None, :], scale=scale[None, :])
        total += float(f) * (g @ weights)
    return total


def _shape_groups(n, o):
    """``(shapes, weights, offsets)`` of the moving tracks for `_mixture_cdf`: each step count's share, and with
    per-spot precision (``o``) each step count split further by noise offset into bins a twentieth of a natural log
    wide, at each bin's mean offset -- a few hundred groups for thousands of tracks, the CDF within about 1e-3 of the
    per-track sum."""
    n = np.asarray(n, dtype=float)
    if o is None:
        shapes, counts = np.unique(n, return_counts=True)
        return shapes, counts / counts.sum(), None
    o = np.asarray(o, dtype=float)
    keys = np.stack([n, np.round(20.0 * np.log(np.maximum(o, 1e-12)))], axis=1)
    groups, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = np.ravel(inverse)
    offsets = np.bincount(inverse, weights=o) / counts
    return groups[:, 0], counts / counts.sum(), offsets


def _labels(values):
    """Unit labels as text, None where there is none (a track outside every cell)."""
    if values is None:
        return None
    return np.array([None if v is None or (isinstance(v, float) and v != v) else str(_day_label(v))
                     for v in values], dtype=object)


def _cross_validate(x, n, counts, day, cell, n_starts: int, seed: int, zeros: str = "auto", o=None) -> dict:
    """Held-out log-likelihood per population count, over independent units: each replicate left out in turn when
    there are two or more, else `jumps_view.N_FOLDS` folds of cells (at least `CV_MIN_CELLS`). Each fold's fits are
    made without its tracks (`appd_gamma.fit_appd_gamma`, ``n_starts`` seeded starts) and scored by the exact
    log-likelihood of them (`appd_gamma.mixture_logpdf`, each its own shape). Every count scored in every fold is
    compared: ``{"unit", "n_folds", "n_units", "n_test", "delta_total", "delta_per_track", "wins", "best"}``, or
    ``{"unit", "note"}`` saying why not. ``zeros`` is the condition's rule for exact-zero tracks (`appd_gamma`), so
    every fold's fit holds its populations and a held-out zero scores the immobile share. ``o``: each track's noise
    offset (per-spot precision), in every fold's fit and score."""
    from .appd_gamma import AppDGammaError, fit_appd_gamma, mixture_logpdf
    from .jumps_view import N_FOLDS

    if not counts:
        return {"unit": None, "note": "no fit converged"}
    days, cells = _labels(day), _labels(cell)
    n_days = len({d for d in days if d is not None}) if days is not None else 0
    n_cells = len({c for c in cells if c is not None}) if cells is not None else 0
    if n_days >= 2:
        unit, keys = "day", days
        folds = [[d] for d in sorted({d for d in days if d is not None})]
    elif n_cells >= CV_MIN_CELLS:
        order = sorted({c for c in cells if c is not None})
        np.random.default_rng(seed).shuffle(order)
        k = min(N_FOLDS, len(order))
        unit, keys, folds = "cell", cells, [order[i::k] for i in range(k)]
    else:
        return {"unit": None, "note": (f"{n_days or 1} replicate and {n_cells} cell{'' if n_cells == 1 else 's'}: "
                                       f"nothing independent to hold out (two replicates, or {CV_MIN_CELLS} cells, "
                                       f"are needed)")}
    per_fold = []
    for fold in folds:
        held = set(fold)
        test = np.array([k in held for k in keys], dtype=bool)
        if not test.any() or test.all():
            continue
        scores = {}
        for k in counts:
            try:
                fit = fit_appd_gamma(x[~test], n_steps=n[~test], n_pop=int(k), n_starts=n_starts, seed=seed,
                                     zeros=zeros, offset=None if o is None else o[~test])
            except AppDGammaError:
                continue
            scores[int(k)] = float(np.sum(mixture_logpdf(x[test], fit.D, fit.fraction, n[test],
                                                         immobile=bool(getattr(fit, "immobile", False)),
                                                         offset=None if o is None else o[test])))
        if scores:
            per_fold.append({"units": list(fold), "n_test": int(test.sum()), "scores": scores})
    if not per_fold:
        return {"unit": unit, "note": "no fold could be fitted"}
    pops = sorted(set.intersection(*(set(f["scores"]) for f in per_fold)))
    if not pops:
        return {"unit": unit, "note": "no population count was fitted in every fold"}
    total = {k: sum(f["scores"][k] for f in per_fold) for k in pops}
    n_test = sum(f["n_test"] for f in per_fold)
    best = max(total.values())
    wins = {k: 0 for k in pops}
    for f in per_fold:
        wins[max(pops, key=lambda k: f["scores"][k])] += 1
    return {"unit": unit, "n_folds": len(per_fold), "n_units": len({u for f in folds for u in f}),
            "n_test": int(n_test), "delta_total": {k: float(total[k] - best) for k in pops},
            "delta_per_track": {k: float((total[k] - best) / n_test) for k in pops},
            "wins": wins, "best": int(max(pops, key=lambda k: total[k]))}


def fit_quality(x, n, fits: dict, *, day=None, cell=None, n_starts: int = CV_STARTS, seed: int = 0,
                zeros: str = "auto", o=None) -> dict:
    """How well each converged fit (``fits``: ``{count: summary}`` as `appd_condition` holds them; failed ones are
    skipped) describes the per-track apparent D ``x`` (step counts ``n``): ``{"fits": {count: {"ecdf_residual":
    [(apparent D, model CDF - ECDF)] at `jumps_view.ECDF_POINTS` quantiles, "ks": the largest ECDF gap}}, "cv":
    `_cross_validate`'s}``. ``day`` / ``cell``: each track's replicate and cell (None where unknown). The ECDF and the
    model both count every track once, each with its own gamma shape. The gap is DESCRIPTIVE: tracks within a cell
    and a replicate are not independent, so it carries no p-value.

    ``zeros`` is the condition's rule for exact-zero tracks (`appd_gamma`'s module head): "kept" (left out of the
    fits) scores the tracks that move, as the fits saw; "immobile" scores every track, the immobile population a step
    at 0 that the ECDF meets at its own step (the gap read tie-aware, as the Jump distances page's). ``o``: each
    track's noise offset (per-spot precision), in the model's CDF and in every held-out fit."""
    from .jumps_view import ECDF_POINTS

    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    o = None if o is None else np.asarray(o, dtype=float)
    if zeros == "kept":
        keep = x > 0
        x, n = x[keep], n[keep]
        o = None if o is None else o[keep]
        day = None if day is None else np.asarray(day, dtype=object)[keep]
        cell = None if cell is None else np.asarray(cell, dtype=object)[keep]
    still = zeros == "immobile"
    good = {int(k): v for k, v in (fits or {}).items() if v and "failed" not in v}
    xs = np.sort(x)
    total = len(xs)
    # the moving populations are the mixture of the tracks that MOVE (every track, without an exact zero)
    shapes, weights, offsets = _shape_groups(n[x > 0], None if o is None else o[x > 0])
    q = np.linspace(0.5 / ECDF_POINTS, 1.0 - 0.5 / ECDF_POINTS, ECDF_POINTS)
    points = np.quantile(xs, q)
    ecdf = np.searchsorted(xs, points, side="right") / total
    ranks = np.unique(np.linspace(0, total - 1, min(total, KS_MAX_POINTS)).astype(int))
    out = {}
    for k, fit in good.items():
        model = _mixture_cdf(points, shapes, weights, fit["D"], fit["fraction"], offsets=offsets)
        at = _mixture_cdf(xs[ranks], shapes, weights, fit["D"], fit["fraction"], offsets=offsets)
        if still:
            # THE IMMOBILE POPULATION IS A STEP AT 0: the ECDF above the model AT a value (ties included), below it
            # just under the value, so the zeros' step is met by the model's own (`jumps_view._ks_distance`)
            below = _mixture_cdf(xs[ranks], shapes, weights, fit["D"], fit["fraction"], inclusive=False,
                                 offsets=offsets)
            last = np.searchsorted(xs, xs[ranks], side="right")
            first = np.searchsorted(xs, xs[ranks], side="left")
            ks = float(max(np.max(last / total - at), np.max(below - first / total)))
        else:
            i = ranks + 1.0
            ks = float(max(np.max(i / total - at), np.max(at - (i - 1.0) / total)))
        out[k] = {"ecdf_residual": [(float(a), float(m - e)) for a, m, e in zip(points, model, ecdf)], "ks": ks}
    return {"fits": out, "cv": _cross_validate(x, n, sorted(good), day, cell, int(n_starts), int(seed),
                                               zeros="immobile" if still else "auto", o=o)}


def _unit_words(cv: dict) -> str:
    return {"day": "replicates", "cell": "cells"}.get(cv.get("unit"), "units")


def quality_table(result: dict) -> dict:
    """The fit-quality table under a Distributions card, as the Jump distances page's (`jumps_view.table_rows`):
    ``{"columns", "rows", "notes"}``. A count that did not converge says so; a result made before 3 Nov 2026 (no
    ``"quality"``) shows BIC and dashes."""
    quality = result.get("quality") or {}
    cv = quality.get("cv") or {}
    unit = _unit_words(cv)
    # SHORT HEADERS AND CELLS: seven columns share a Distributions card half a window wide (the first draw cut "Largest
    # ECDF gap" and the three-population D's); what is held out is said in the notes
    columns = ["Populations", "D (µm²/s)", "Fractions (%)", "ΔBIC", "ECDF gap", "Held-out ΔlogL", "Folds won"]
    selection = {int(r["n_pop"]): r for r in (result.get("selection") or {}).get("rows", [])}
    gaps = quality.get("fits") or {}
    zeros = result.get("zero_tracks") or {}
    rows = []
    for k in sorted(int(k) for k in (result.get("fits") or {})):
        if zeros.get("immobile") and k == 1:
            continue                       # no candidate beside the immobile population: the table starts at two
        fit = result["fits"].get(k) or result["fits"].get(str(k)) or {}
        if not fit or "failed" in fit:
            rows.append([str(k), "did not converge", "–", "–", "–", "–", "–"])
            continue
        delta = selection.get(k, {}).get("delta")
        ks = (gaps.get(k) or gaps.get(str(k)) or {}).get("ks")
        held = (cv.get("delta_total") or {}).get(k)
        # three populations or more: one value a line (on one, the D's of three ran past half a window)
        joint = " / " if len(fit["D"]) <= 2 else "\n"
        rows.append([str(k), joint.join(_d3(d) for d in fit["D"]),
                     joint.join(f"{100 * f:.1f}" for f in fit["fraction"]),
                     "–" if delta is None else f"{delta:.1f}", "–" if ks is None else f"{100 * ks:.2f} pp",
                     "–" if held is None else f"{held:.1f}",
                     "–" if not cv.get("wins") else f"{cv['wins'].get(k, 0)} of {cv['n_folds']}"])
    notes = ["ΔBIC from the maximum-likelihood fits drawn (n = tracks, one per molecule); lowest is preferred, a "
             "difference under 2 is inconclusive."]
    if not quality:
        notes.append("The largest ECDF gap and the held-out fits need a new Run.")
    elif cv.get("note"):
        notes.append(f"Cross-validation: {cv['note']}.")
    elif cv.get("n_folds"):
        best = cv["best"]
        notes.append(f"Cross-validation over {cv['n_units']} {unit} in {cv['n_folds']} folds: fitted without each "
                     f"fold, scored on its tracks; the held-out log-likelihood (Held-out ΔlogL, summed over the held-out "
                     f"{unit}) is best at {best} population{'' if best == 1 else 's'} (0 = best)."
                     + (" Cells share their replicate with the training cells, so this is the weaker test."
                        if cv["unit"] == "cell" else ""))
    notes.append("The largest ECDF gap is descriptive (the Kolmogorov distance): tracks of one cell and one replicate "
                 "are not independent, so no p-value is given.")
    if zeros.get("note"):
        # exact-zero tracks (`appd_gamma.zero_tracks`): what every fit on this card did with them, and why 1 is missing
        notes.append(zeros["note"] + (" One population cannot hold them beside the moving ones, so the table "
                                      "starts at two." if zeros.get("immobile") else ""))
    return {"columns": columns, "rows": rows, "notes": notes}


def quality_verdict(result: dict) -> str:
    """The card's one line under its table: what BIC and the held-out units each prefer (`jumps_view.verdict_line`'s
    wording)."""
    parts = []
    selection = result.get("selection") or {}
    best = selection.get("recommended")
    if best is not None:
        part = f"BIC prefers {best} population{'' if best == 1 else 's'}"
        if selection.get("inconclusive"):
            part += f" (inconclusive: the runner-up is {selection['margin']:.1f} behind)"
        parts.append(part)
    cv = (result.get("quality") or {}).get("cv") or {}
    if cv.get("note"):
        parts.append(f"held out: {cv['note'].split(':')[0]}")
    elif cv.get("n_folds"):
        k = cv["best"]
        parts.append(f"held-out {_unit_words(cv)} prefer {k} population{'' if k == 1 else 's'} "
                     f"({cv['wins'].get(k, 0)} of {cv['n_folds']} folds won)")
    return " · ".join(parts)
