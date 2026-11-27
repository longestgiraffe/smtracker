"""Controllers for the method browser: pure, headless-testable, no Qt.

Each function takes ``(project, condition, *, n_boot)`` and returns
``(draw, status)``: ``draw(figure)`` paints an existing analysis onto a
matplotlib Figure, and ``status`` is a one-line human summary with the real
numbers. All computation lives here (or in the analysis modules these call);
the Qt tab in ``smtracker.browser`` only presses the button and shows the
canvas. So a wrong number fails in a controller test, never subtly on screen.

Everything is computed from the already-tested analysis functions: this module
adds no new science, it only arranges existing results for display.
"""

from __future__ import annotations

import numpy as np

from . import plots

__all__ = [
    "control_intensity",
    "control_line",
    "fractions",
    "superplot_distribution",
    "superplot_fraction",
    "kinetics",
    "state_count",
    "spatial",
    "confinement",
    "clustering",
    "spoton",
    "qc",
]


def control_intensity(project, condition, control, *, laser_power_mw,
                      n_movies=3, on_progress=None) -> str:
    """Per-spot integrated intensity, sample against control. Plain text.

    The second half of the control question. `control_line` bounds contamination by RATE;
    this asks whether it could instead be removed by BRIGHTNESS. Measured once on
    one labelled sample against the unlabelled WT, the answer is no, a best separation over any cut
    of under 20 percentage points, and about a third of WT spots brighter than the MEDIAN real
    molecule, but that was one protein in one medium, and the comparison belongs in the
    tab so it can be asked of any pair.

    **It reads frames, so it is slow and the caller must not run it on the GUI thread.**
    `on_progress(done, total, note)` reports movie by movie.

    ``laser_power_mw`` has NO DEFAULT. Intensity scales with illumination, so comparing
    two groups recorded at different powers compares the lasers; `spot_intensity` refuses
    a mismatched calibration for the same reason and this passes the value straight
    through rather than inventing one.

    TIER 1 (raw integration): a flat field scales both groups together and cannot close a
    gap of this size, and the question here is relative.
    """
    import numpy as np
    import pandas as pd

    from .grouping import discover_utrack_project
    from .ingest import read_utrack_mat
    from .spot_intensity import IntensityCalibration, integrated_intensity

    if not control or control == condition:
        return ("the control and the sample are the same condition: pick another, "
                "or 'none'") if control else ""
    if not laser_power_mw:
        return ("set the laser power first: intensity scales with illumination, so "
                "without it this would compare the lasers rather than the samples")

    calibration = IntensityCalibration(laser_power_mw=float(laser_power_mw),
                                       source="tier1-raw")
    out = {}
    for label in (condition, control):
        try:
            root = project.roots[label]
        except Exception:                             # noqa: BLE001 - named on screen
            return f"no folder recorded for {label!r}, so its frames cannot be read"
        try:
            found = discover_utrack_project(root)[:int(n_movies)]
            if not found:
                return f"{label}: no tracker output under {root}"
            tables = [read_utrack_mat(path, movie=movie) for movie, path in found]
            raw = pd.concat(tables, ignore_index=True)
            if on_progress is not None:
                on_progress(0, len(found), f"{label}: reading {len(found)} movie(s)")
            got = integrated_intensity(raw, found[0][1].parents[2].parent,
                                       calibration=calibration,
                                       laser_power_mw=float(laser_power_mw),
                                       movies=[m for m, _ in found])
        except Exception as exc:                      # noqa: BLE001 - named on screen
            return f"{label}: {type(exc).__name__}: {exc}"
        values = pd.to_numeric(got["integrated"], errors="coerce").dropna().to_numpy()
        if not len(values):
            return f"{label}: no spots could be integrated"
        out[label] = values

    sample, ctrl = out[condition], out[control]
    grid = np.percentile(np.concatenate([sample, ctrl]), np.linspace(0, 100, 400))
    best = max((sample >= g).mean() - (ctrl >= g).mean() for g in grid)
    above_median = float((ctrl >= np.median(sample)).mean())
    text = (f"{condition} median {np.median(sample):,.0f} (n={len(sample)}) against "
            f"{control} {np.median(ctrl):,.0f} (n={len(ctrl)}): the control is "
            f"{np.median(ctrl) / np.median(sample) * 100:.0f} % as bright")
    text += (f"\nbest separation over ANY brightness cut: {best * 100:.1f} percentage "
             f"points; {above_median * 100:.0f} % of control spots are brighter than the "
             f"MEDIAN real one")
    if best < 0.30:
        text += ("\nso brightness CANNOT separate them: bound the contamination by "
                 "rate, do not filter by intensity")
    return text


def control_line(project, condition, control, *, media=None) -> str:
    """One line comparing a sample's track rate against a CONTROL's. Plain text.

    The compute lives here, not in the Qt tab. `browser.py` is a thin shell and an AST
    test forbids numpy/pandas/scipy anywhere in it: including inside a method, which is
    stricter than it first looks and caught this on the first try. Concatenating two
    conditions is pandas, so it is here.

    Returns the sentence to show, or a sentence saying why there is none. It never
    raises: a control that cannot be compared must leave the four QC panels alone, since
    they are independent measurements and neither is a precondition for the other.
    """
    import pandas as pd

    from .qc import control_contamination

    if not control or control == condition:
        return ("the control and the sample are the same condition: pick another, "
                "or 'none'") if control else ""
    try:
        sample = project.condition(condition)
        table = pd.concat([sample, project.condition(control)], ignore_index=True)
        # `.attrs` does not survive concat, and `detection_density` needs tau.
        table.attrs = dict(sample.attrs)
        got = control_contamination(table, control_conditions=[control],
                                    media=media or {})
    except Exception as exc:                          # noqa: BLE001 - named on screen
        return f"control comparison unavailable: {type(exc).__name__}: {exc}"
    row = got.iloc[0]
    text = (f"{row['condition']} {row['tracks_per_cell_per_minute']:.1f} tracks per "
            f"cell per minute against {control} "
            f"{row['control_per_cell_per_minute']:.1f}: the control is "
            f"{row['contamination'] * 100:.1f} % of the sample's rate")
    if row["caveat"]:
        text += f"\n{row['caveat']}"
    return text


def _part(project, condition):
    return project.condition(condition)


# --------------------------------------------------------------------------- #
# Fractions + cross-method
# --------------------------------------------------------------------------- #


def fractions(project, condition, *, n_boot=50):
    from .observables import squared_displacements
    from .sqd import bootstrap_sqd, fit_sqd

    obs = squared_displacements(_part(project, condition))
    fit = fit_sqd(obs, n_pop=2, weighting="track")
    boot = bootstrap_sqd(obs, n_pop=2, unit="cell", n_boot=n_boot, weighting="track")
    slow = 100 * float(fit.fractions[0])
    lo, hi = (100 * boot["fraction_1"].quantile([0.025, 0.975])).tolist()

    def draw(figure):
        ax = figure.add_subplot(111)
        ax.bar([0], [slow], yerr=[[slow - lo], [hi - slow]], capsize=6,
               color="#2b6cb0", width=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([condition])
        ax.set_ylabel("slow / bound fraction (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"SQD bound fraction: track-weighted, cell bootstrap "
                     f"(n_boot={n_boot})", fontsize=9, loc="left")

    return draw, (f"{condition}: slow {slow:.1f}% [{lo:.1f}, {hi:.1f}] "
                  f"(SQD track-weighted, {boot.attrs['n_units']}-cell bootstrap)")


def superplot_distribution(project, condition, *, n_boot=50):
    from .qc import track_summary

    summary = track_summary(_part(project, condition))
    points = summary[["uid", "day", "appd_um2_per_s"]].dropna()

    def draw(figure):
        plots.draw_superplot_distribution(figure, points, value="appd_um2_per_s")

    return draw, (f"{condition}: {len(points)} tracks across "
                  f"{points['day'].nunique()} replicate(s): per-track appD, "
                  "switching-degraded view")


def superplot_fraction(project, condition, *, n_boot=50):
    from .hmm import HMMError, fit_hmm, step_sequences
    from .observables import squared_displacements
    from .qc import track_summary, unit_summary
    from .sqd import bootstrap_sqd, fit_sqd

    part = _part(project, condition)
    tau = project.params.tau_s
    try:
        fit = fit_hmm(step_sequences(part), n_states=2, tau_s=tau, n_starts=6)
        summary = track_summary(part, hmm_fit=fit)
        frac_col = "slow_fraction"
    except HMMError:
        # If the HMM will not decode, fall back to a per-cell appD readout, the
        # SuperPlot still shows the per-cell structure, just of a coarser number.
        summary = track_summary(part)
        frac_col = "appd_um2_per_s"
    # Per-cell frame carrying the replicate DAY, which unit_summary drops.
    per_cell = (
        summary.groupby("cell", observed=True)
        .agg(day=("day", "first"), **{frac_col: (frac_col, "mean")})
        .reset_index()
    )

    obs = squared_displacements(part)
    overall = float(fit_sqd(obs, n_pop=2, weighting="cell").fractions[0])
    boot = bootstrap_sqd(obs, n_pop=2, unit="cell", n_boot=n_boot, weighting="cell")
    lo, hi = boot["fraction_1"].quantile([0.025, 0.975]).tolist()

    def draw(figure):
        plots.draw_superplot_fraction(
            figure, per_cell, frac=frac_col, ci=(lo, hi), overall=overall
        )

    return draw, (f"{condition}: {len(per_cell)} cells, overall slow "
                  f"{overall:.3f} [{lo:.3f}, {hi:.3f}] (cell bootstrap)")


# --------------------------------------------------------------------------- #
# Kinetics
# --------------------------------------------------------------------------- #


def kinetics(project, condition, *, n_boot=50):
    from .hmm import HMMError, bootstrap_hmm, fit_hmm, step_sequences

    part = _part(project, condition)
    tau = project.params.tau_s
    fit = fit_hmm(step_sequences(part), n_states=2, tau_s=tau, n_starts=8)
    rates = fit.rates_per_s
    on, off = float(rates[1, 0]), float(rates[0, 1])

    on_ci = off_ci = None
    identifiable = None
    try:
        boot = bootstrap_hmm(part, n_states=2, tau_s=tau, unit="cell", n_boot=n_boot)
        on_ci = boot["rate_2_to_1"].quantile([0.025, 0.975]).tolist()
        off_ci = boot["rate_1_to_2"].quantile([0.025, 0.975]).tolist()
        identifiable = boot.attrs["n_degenerate"] == 0
    except HMMError:
        pass

    def draw(figure):
        ax = figure.add_subplot(111)
        labels = ["on (fast→slow)", "off (slow→fast)"]
        vals = [on, off]
        errs = None
        if on_ci and off_ci:
            errs = [[on - on_ci[0], off - off_ci[0]], [on_ci[1] - on, off_ci[1] - off]]
        ax.bar([0, 1], vals, yerr=errs, capsize=6, color=["#2b6cb0", "#c05621"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_ylabel("rate (1/s)")
        flag = "" if identifiable is None else (
            "  [rates identifiable]" if identifiable else "  [LOW-SWITCHING: rates unreliable]"
        )
        ax.set_title(f"HMM switching rates: {condition}{flag}", fontsize=9, loc="left")

    slow = 100 * float(fit.occupancy[0])
    degen = " (degenerate chain)" if fit.has_degenerate_chain else ""
    return draw, (f"{condition}: slow {slow:.1f}%, on {on:.2f} / off {off:.2f} 1/s"
                  f"{degen}; D = {np.round(fit.D, 4).tolist()}")


def state_count(project, condition, *, n_boot=50):
    from .saspt_adapter import run_saspt

    result = run_saspt(_part(project, condition))
    modes = result.modes()

    def draw(figure):
        ax = figure.add_subplot(111)
        ax.plot(result.diffusion_grid, result.occupations, color="#2b6cb0")
        ax.set_xscale("log")
        ax.set_xlabel("D (µm²/s, log)")
        ax.set_ylabel("posterior occupancy")
        ax.set_title(f"saSPT spectrum: {condition}: {len(modes)} resolved mode(s)",
                     fontsize=9, loc="left")

    described = ", ".join(f"D={r.D_um2_per_s:.3g} ({100 * r.occupancy:.0f}%)"
                          for r in modes.itertuples())
    return draw, f"{condition}: {len(modes)} modes: {described}"


# --------------------------------------------------------------------------- #
# Spatial / confinement / clustering
# --------------------------------------------------------------------------- #


def spatial(project, condition, *, n_boot=50):
    from .cellcoords import project_tracks
    from .hmm import fit_hmm, step_sequences
    from .maps import fold_map, labels_from_hmm, state_enrichment

    part = _part(project, condition)
    projected = project_tracks(part, project.meshes)
    fit = fit_hmm(step_sequences(projected), n_states=2, tau_s=project.params.tau_s,
                  n_starts=6)
    labelled = fold_map(labels_from_hmm(projected, fit, min_posterior=0.7))
    table = state_enrichment(labelled, axis="d_norm", bins=5)
    wide = table.pivot(index="d_norm", columns="state", values="enrichment")

    def draw(figure):
        ax = figure.add_subplot(111)
        for state, colour, name in ((1, "#2b6cb0", "slow"), (2, "#c05621", "fast")):
            if state in wide.columns:
                ax.plot(wide.index, wide[state], "o-", color=colour, label=name)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("d_norm  0 = axis → 1 = membrane")
        ax.set_ylabel("radial enrichment")
        ax.set_title(f"HMM per-step radial map: {condition}", fontsize=9, loc="left")
        ax.legend(fontsize=8, frameon=False)

    slow = wide[1].to_numpy() if 1 in wide.columns else np.array([np.nan])
    contrast = slow[-1] / slow[0] if slow[0] else float("nan")
    from .chart_text import fold_words

    return draw, f"{condition}: slow-state radial contrast {fold_words(contrast, 2)} (start-point)"


def confinement(project, condition, *, n_boot=50):
    from .confinement import confinement_null
    from .sqd import fit_sqd
    from .observables import squared_displacements

    part = _part(project, condition)
    fast_D = float(max(fit_sqd(squared_displacements(part), n_pop=2).D))
    null = confinement_null(part, radius_um=0.1, D_um2_s=fast_D, min_steps=3,
                            n_replicates=3)
    obs = null.observed["confined_segment_fraction"]
    chance = null.chance["confined_segment_fraction"]
    excess = null.excess["confined_segment_fraction"]

    def draw(figure):
        ax = figure.add_subplot(111)
        ax.bar([0, 1, 2], [obs, chance, excess],
               color=["#2b6cb0", "#a0aec0", "#c05621"])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["observed", "chance", "excess"])
        ax.set_ylabel("confined-segment fraction")
        ax.set_title(f"Confinement excess over free-Brownian null: {condition}",
                     fontsize=9, loc="left")

    return draw, (f"{condition}: confined {100 * obs:.1f}% observed, "
                  f"{100 * chance:.1f}% chance, {100 * excess:.1f}% EXCESS")


def clustering(project, condition, *, n_boot=50):
    from .clustering import select_k

    part = _part(project, condition)
    selection = select_k(part, D_um2_s=0.3, k_values=(2, 3), n_lags=4,
                         feature="log", n_replicates=3)
    table = selection.table

    def draw(figure):
        ax = figure.add_subplot(111)
        x = table["k"].to_numpy()
        ax.plot(x, table["observed"], "o-", color="#2b6cb0", label="observed")
        ax.plot(x, table["chance"], "s--", color="#a0aec0", label="chance (null)")
        ax.set_xlabel("k")
        ax.set_ylabel("Calinski-Harabasz")
        ax.set_title(f"Clustering vs free-Brownian null: {condition}", fontsize=9,
                     loc="left")
        ax.legend(fontsize=8, frameon=False)

    return draw, f"{condition}: {selection.verdict}"


def spoton(project, condition, *, n_boot=50):
    from .spoton import fit_spoton

    part = _part(project, condition)
    # dZ -> infinity is the uncorrected fit (correction is a no-op); labelled.
    fit = fit_spoton(part, n_states=2, dZ_um=1e6, lags=(1, 2, 3))
    slow = 100 * float(fit.fractions[0])

    def draw(figure):
        ax = figure.add_subplot(111)
        ax.bar([0, 1], [100 * f for f in fit.fractions],
               color=["#2b6cb0", "#c05621"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["slow", "fast"])
        ax.set_ylabel("fraction (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Spot-On (uncorrected, dZ not applied): {condition}",
                     fontsize=9, loc="left")

    return draw, (f"{condition}: slow {slow:.1f}% (Spot-On, UNCORRECTED: "
                  f"dZ unmeasured); D = {np.round(fit.D, 4).tolist()}")


def qc(project, condition, *, n_boot=50):
    from .qc import (group_counts, length_distribution, localizations_per_frame,
                     track_summary, unit_summary)

    part = _part(project, condition)
    summary = track_summary(part)
    lengths = length_distribution(part, summary=summary)
    density = localizations_per_frame(part, field_shape=(126, 130))
    counts = group_counts(part, summary=summary)
    units = unit_summary(part, unit="cell", summary=summary)

    def draw(figure):
        plots.draw_qc_dashboard(figure, lengths=lengths, density=density,
                                counts=counts, units=units)

    return draw, (f"{condition}: {summary.attrs['n_tracks']} tracks, "
                  f"{part['cell'].nunique()} cells, "
                  f"{int(units['flagged'].sum())} flagged")
