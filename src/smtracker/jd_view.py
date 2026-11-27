"""The SQD section's Signed jumps page, headless (agreed 17 Oct 2026: "the along/across signed jump-distance fit in
the workspace as a cross-check").

Each condition's SIGNED one-dimensional jumps -- ALONG the cell's axis, ACROSS it, and the camera's x and y stacked --
against the zero-mean normal mixtures at 1, 2 and 3 populations (`jd.fit_jd`, the legacy GMM model with its defects
fixed; D = sigma^2 / (2 tau), the one-dimensional Einstein relation). Plain data for `ui.workspace.jd_chart`.

What it is for, and what it is not
-----------------------------------
- A CROSS-CHECK, never a second estimate to average: a signed jump and a squared displacement are different
  observables (`jd` module head). On the camera's axes the page sets the SQD fit of the same jumps beside it
  (step-weighted, the same lag) so a reader sees whether the two agree.
- ALONG AGAINST ACROSS is the reason for the page: a B. subtilis cell is about 1 um wide, so a fast molecule's jumps
  across it are cut short by the membrane and its across-D reads lower than its along-D. The ratio measures that
  confinement, not an anisotropic diffusion coefficient.
- The cell's axes come from the projection (`Project.projected`, the same coordinates as the MSD section's along /
  across and the Spatial Map); a track without a centreline has no along or across jump (counted).
- THE SPREAD IS OVER REPLICATE DAYS (RULE 0: days are independent), mean +- SD of each day's own fit, never an
  interval on pooled jumps. One session: no spread.
- Jumps are taken only between frames exactly ``lag`` apart (`jd.signed_displacements`); outliers are removed by the
  fit's own rule (`jd.remove_outliers`), and the histogram is drawn from the same cleaned jumps.
"""
from __future__ import annotations

import numpy as np

__all__ = ["JD_AXES", "JD_AXIS_LABELS", "compute_jd_view", "card_payload", "summary_table", "READING_NOTES",
           "N_BINS", "MAX_POP"]

#: The three ways a jump is measured, the page's switch.
JD_AXES = ("along", "across", "xy")
JD_AXIS_LABELS = {"along": "Along the cell", "across": "Across the cell", "xy": "Camera x, y"}
N_BINS = 60
MAX_POP = 3

READING_NOTES = (
    "Each condition's signed jumps along one axis (a jump to the left is negative), fitted with zero-mean normal "
    "mixtures of 1, 2 and 3 populations. Bars are the jumps, the solid line is the fit shown and thin lines are "
    "its populations. D = sigma^2 / (2 tau).\n\n"
    "Along and Across the cell use each jump's direction relative to its own cell's axis. A cell is about 1 um "
    "wide, so the membrane limits jumps across it and the across D reads lower. The ratio in the table measures "
    "this confinement; it is not a direction-dependent diffusion coefficient. Camera x, y combines both camera "
    "axes and is shown beside the SQD fit of the same jumps.\n\n"
    "This is a cross-check of the SQD fractions; do not average it with them. The spread is the SD over "
    "replicates, each fitted alone.")


def _axis_table(project, condition: str, axis: str):
    """The table whose ``x_um`` holds the axis measured (`msd._components`: one mapping of the cell's axes)."""
    from .msd import _components

    if axis == "xy":
        return project.condition(condition), "xy"
    projected = project.projected(condition)
    finite = projected[["L_um", "D_um"]].notna().all(axis=1)
    return _components(projected[finite], axis)[0], "x"


def _ks(values_sorted, fit) -> float:
    from .jd import _mixture_cdf

    n = values_sorted.size
    model = _mixture_cdf(fit["sigma_um"], fit["fraction"], values_sorted, fit.get("noise_groups"))
    upper = np.arange(1, n + 1) / n
    return float(max(np.max(np.abs(model - upper)), np.max(np.abs(model - (upper - 1.0 / n)))))


def _axis(project, condition: str, axis: str, *, lag: int, max_pop: int, calibration=None) -> dict:
    """One axis: the histogram, every fit and each day's fits. ``calibration``: per-spot precision, every fit with
    each jump's own noise (`jd.fit_jd`'s ``noise_um2``), the jumps without one left out first."""
    from .jd import JdError, _mixture_cdf, fit_jd, remove_outliers, signed_displacements

    table, direction = _axis_table(project, condition, axis)
    tau = float(np.median(table["tau_s"].to_numpy(dtype=float))) * int(lag)
    steps = signed_displacements(table, direction=direction, lag_frames=int(lag), calibration=calibration)
    if calibration is not None:
        steps = steps[np.isfinite(steps["noise_um2"].to_numpy(dtype=float))]
    values = steps["displacement_um"].to_numpy(dtype=float)
    noise = steps["noise_um2"].to_numpy(dtype=float) if calibration is not None else None
    cleaned = remove_outliers(values)
    kept = np.sort(cleaned["values"])
    n = kept.size
    top = float(np.quantile(np.abs(kept), 0.995)) * 1000.0
    edges_nm = np.linspace(-top, top, N_BINS + 1)
    observed = np.histogram(kept * 1000.0, bins=edges_nm)[0]
    lo, hi = edges_nm[:-1] / 1000.0, edges_nm[1:] / 1000.0
    fits = {}
    for k in range(1, int(max_pop) + 1):
        try:
            fit = fit_jd(values, tau_s=tau, n_pop=k, noise_um2=noise)
        except JdError as exc:
            fits[k] = {"failed": str(exc)}
            continue
        sigma, frac, groups = fit["sigma_um"], fit["fraction"], fit.get("noise_groups")
        expected = n * (_mixture_cdf(sigma, frac, hi, groups) - _mixture_cdf(sigma, frac, lo, groups))
        components = [[float(v) for v in n * f * (_mixture_cdf([s], [1.0], hi, groups)
                                                   - _mixture_cdf([s], [1.0], lo, groups))]
                      for s, f in zip(sigma, frac)]
        fits[k] = {"D": [float(d) for d in fit["D_um2_s"]], "fractions": [float(f) for f in frac],
                   "sigma_nm": [float(1000.0 * s) for s in sigma],
                   "expected": [float(v) for v in expected], "components": components,
                   "ks": _ks(kept, fit)}
    days = {}
    if "day" in steps.columns and steps["day"].nunique() >= 2:
        for k in range(1, int(max_pop) + 1):
            rows = []
            for day, part in steps.groupby("day", sort=True):
                try:
                    fit = fit_jd(part["displacement_um"].to_numpy(dtype=float), tau_s=tau, n_pop=k,
                                 noise_um2=None if noise is None else part["noise_um2"].to_numpy(dtype=float))
                except JdError:
                    continue
                rows.append({"day": _plain(day), "D": [float(d) for d in fit["D_um2_s"]],
                             "fractions": [float(f) for f in fit["fraction"]],
                             "n": int(fit["n_displacements"])})
            if len(rows) >= 2:
                f = np.array([r["fractions"] for r in rows])
                d = np.array([r["D"] for r in rows])
                days[k] = {"values": rows, "fraction_mean": f.mean(axis=0).tolist(),
                           "fraction_sd": f.std(axis=0, ddof=1).tolist(), "D_mean": d.mean(axis=0).tolist(),
                           "D_sd": d.std(axis=0, ddof=1).tolist()}
    return {"axis": axis, "n_jumps": int(n), "n_tracks": int(steps["uid"].nunique()),
            "n_removed": int(cleaned["n_removed"]),
            "tracks_without_axis": int(project.condition(condition)["uid"].nunique() - table["uid"].nunique()),
            "tau_s": tau, "edges_nm": [float(e) for e in edges_nm], "observed": [int(c) for c in observed],
            "fits": fits, "days": days}


def _plain(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(f) if f.is_integer() else f


def compute_jd_view(project, condition: str, *, lag: int = 1, max_pop: int = MAX_POP, precision=None) -> dict:
    """One condition's signed jumps on every axis (`JD_AXES`), each at 1..``max_pop`` populations, with each day's
    fits and, on the camera's axes, the SQD fit of the same jumps. Picklable; an expected refusal is an
    ``{"error"}`` entry for that axis (or the whole condition), naming what refused. ``precision``: the SQD Run's
    per-spot calibration ``(c_nm, p, source)`` for this condition (Localisation error Per spot; None for a condition
    whose tracks carry none): every fit with each jump's own noise, the SQD beside it the per-spot one."""
    from .jd import JdError
    from .msd import MSDError
    from .observables import ObservableError, squared_displacements
    from .sqd import SQDError, fit_sqd, zero_jumps

    calibration = None
    if precision is not None:
        from .precision import Calibration

        calibration = Calibration(c_nm=float(precision[0]), p=float(precision[1]), source=str(precision[2]))
    out = {"condition": condition, "lag": int(lag), "max_pop": int(max_pop), "axes": {}, "sqd": {},
           **({"precision": [float(precision[0]), float(precision[1]), str(precision[2])]}
              if precision is not None else {})}
    for axis in JD_AXES:
        try:
            out["axes"][axis] = _axis(project, condition, axis, lag=int(lag), max_pop=int(max_pop),
                                      calibration=calibration)
        except (JdError, MSDError, ObservableError, ValueError, KeyError) as exc:
            out["axes"][axis] = {"axis": axis, "error": f"{type(exc).__name__}: {exc}"}
    if all("error" in a for a in out["axes"].values()):
        out["error"] = "; ".join(f"{JD_AXIS_LABELS[k]}: {a['error']}" for k, a in out["axes"].items())
        return out
    try:
        canonical = project.condition(condition)
        sd = squared_displacements(canonical) if int(lag) == 1 else squared_displacements(canonical,
                                                                                         lag_frames=int(lag))
        # exact-zero jumps as an immobile population (`sqd`'s module head): one population cannot hold it beside the
        # moving ones, so the SQD column starts at two (the card shows its empty mark at one)
        zeros = zero_jumps(sd["sq_disp_um2"].to_numpy(dtype=float))
        for k in range(2 if zeros is not None and zeros.immobile else 1, int(max_pop) + 1):
            fit = (fit_sqd(sd, n_pop=k, weighting="step", method="mle", gradient="analytic", precision=calibration)
                   if calibration is not None
                   else fit_sqd(sd, n_pop=k, weighting="step", ecdf_max_points=2000, gradient="analytic"))
            out["sqd"][k] = {"D": [float(d) for d in fit.D], "fractions": [float(f) for f in fit.fractions]}
    except (SQDError, ObservableError) as exc:
        out["sqd"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _pct(values) -> str:
    return " / ".join(f"{100 * v:.1f}" for v in values)


def _d(values) -> str:
    return " / ".join(f"{v:.3g}" for v in values)


def card_payload(result: dict, axis: str, k: int, colour: str) -> dict:
    """One condition's card on ``axis`` at ``k`` populations: the histogram (`JumpHistogram`'s payload, its x axis
    signed) and the table, or ``{"error"}``."""
    entry = (result.get("axes") or {}).get(axis)
    if result.get("error") and not entry:
        return {"error": result["error"]}
    if entry is None or "error" in entry:
        return {"error": (entry or {}).get("error", "not computed")}
    fits = {kk: f for kk, f in entry["fits"].items() if "failed" not in f}
    if k not in fits:
        return {"error": entry["fits"].get(k, {}).get("failed", f"no {k}-population fit")}
    fit = fits[k]
    histogram = {"edges": entry["edges_nm"], "observed": entry["observed"],
                 "fits": {kk: {"expected": f["expected"], "components": f["components"]} for kk, f in fits.items()},
                 "selected": k, "colour": colour, "x_label": "Signed jump (nm)"}
    days = entry["days"].get(k)
    other = {"along": "across", "across": "along"}.get(axis)
    other_fit = None
    if other:
        o = (result.get("axes") or {}).get(other) or {}
        other_fit = (o.get("fits") or {}).get(k)
        if other_fit and "failed" in other_fit:
            other_fit = None
    sqd = (result.get("sqd") or {}).get(k) if axis == "xy" else None
    columns = ["Population", "D (µm²/s)", "Fraction (%)", "Replicates (SD)"]
    columns.append("SQD, same jumps" if axis == "xy" else f"{JD_AXIS_LABELS[other].split()[0]}: D, ratio")
    rows = []
    for i in range(k):
        spread = (f"± {100 * days['fraction_sd'][i]:.1f} ({len(days['values'])} replicates)" if days else "one session")
        if axis == "xy":
            extra = (f"{sqd['D'][i]:.3g} µm²/s, {100 * sqd['fractions'][i]:.1f} %" if sqd and i < len(sqd["D"])
                     else "–")
        elif other_fit:
            ratio = fit["D"][i] / other_fit["D"][i] if other_fit["D"][i] > 0 else float("nan")
            from .chart_text import fold_words

            extra = f"{other_fit['D'][i]:.3g}, {fold_words(ratio, 2)}"
        else:
            extra = "–"
        rows.append([f"population {i + 1}" + (" (slowest)" if i == 0 and k > 1 else ""), f"{fit['D'][i]:.3g}",
                     f"{100 * fit['fractions'][i]:.1f}", spread, extra])
    caption = (f"{entry['n_jumps']:,} jumps from {entry['n_tracks']:,} tracks ({JD_AXIS_LABELS[axis].lower()}, "
               f"{1000 * entry['tau_s']:g} ms apart); {entry['n_removed']:,} beyond the fit's outlier rule left out; "
               f"largest gap between the fit and the ECDF of the jumps {fit['ks']:.3f}.")
    if axis != "xy" and entry.get("tracks_without_axis"):
        caption += (f" {entry['tracks_without_axis']:,} tracks without a centreline have no along or across "
                    "jump.")
    if result.get("precision"):
        caption += (f" Localisation error per spot: each jump's spread is its own, through {result['precision'][2]}, "
                    "so D has the error taken out.")
    return {"histogram": histogram, "columns": columns, "rows": rows, "caption": caption}


def summary_table(results: dict, conditions: list, labels: dict, axis: str, k: int) -> dict:
    """Every condition on ``axis`` at ``k`` populations: jumps, the fractions (slowest first, with the SD over days)
    and the D's -- and along against across for the fastest population."""
    columns = ["Condition", "Jumps", "Fractions (%)", "Replicates (SD)", "D (µm²/s)", "Along / across D (fastest)"]
    rows = []
    for c in conditions:
        res = results.get(c) or {}
        entry = (res.get("axes") or {}).get(axis) or {}
        fit = (entry.get("fits") or {}).get(k)
        name = labels.get(c, c)
        if res.get("error") or "error" in entry or not fit or "failed" in fit:
            why = res.get("error") or entry.get("error") or (fit or {}).get("failed") or "not computed"
            rows.append([name, "–", "–", "–", "–", why[:60]])
            continue
        days = entry["days"].get(k)
        spread = _pct(days["fraction_sd"]) if days else "one session"
        a = ((res["axes"].get("along") or {}).get("fits") or {}).get(k)
        b = ((res["axes"].get("across") or {}).get("fits") or {}).get(k)
        ratio = (f"{a['D'][-1] / b['D'][-1]:.2f}" if a and b and "failed" not in a and "failed" not in b
                 and b["D"][-1] > 0 else "–")
        rows.append([name, f"{entry['n_jumps']:,}", _pct(fit["fractions"]), spread, _d(fit["D"]), ratio])
    return {"columns": columns, "rows": rows,
            "caption": (f"{JD_AXIS_LABELS[axis]}, {k} population{'s' * (k != 1)}: fractions slowest first; Replicates = "
                        "SD over replicates, each fitted alone. A cross-check of the SQD fractions on "
                        "another observable; do not average it with them.")}
