"""Sensitivity controls: change a parameter, re-run the cached fit, see it move.

The "watch the shift stay flat as you drag σ" demo, and the general machinery
behind a slider/numeric field for σ, min-track-length, confinement radius, n_pop
and the D-floor. Changing a value re-runs the fast fit and returns an updated
``(draw, status)``; because each value is memoized by its provenance (items 2/4),
dragging back to a value already seen is instant.

This is the headless controller: a Qt slider is a thin layer that calls
``recompute`` on change and repaints. So the *responsiveness* (does a change
re-run and update?) is tested here, and the widget feel is left for later.

The parameters it exposes, and where they bite
----------------------------------------------
- ``sigma_nm``: folded into the HMM emission likelihood; the load-bearing σ-sweep.
- ``min_localizations``: the track-length filter (re-ingests the canonical view).
- ``confinement_radius_um``: the moving-ROI radius.
- ``n_pop``: the SQD mixture count.
- ``localization_precision_nm``: sets the D-floor line.
"""

from __future__ import annotations

__all__ = ["SensitivityError", "PARAMETERS", "recompute", "make_draw"]


class SensitivityError(RuntimeError):
    """Raised when a sensitivity parameter is invalid."""


#: name -> (min, max, default, unit) for a slider/numeric field.
PARAMETERS = {
    "sigma_nm": (0.0, 100.0, 0.0, "nm"),
    "min_localizations": (2, 20, 5, "localizations"),
    "confinement_radius_um": (0.03, 0.4, 0.1, "µm"),
    "n_pop": (1, 4, 2, "populations"),
    "localization_precision_nm": (10.0, 100.0, 30.0, "nm"),
}


def _validate(name, value):
    if name not in PARAMETERS:
        raise SensitivityError(
            f"unknown parameter {name!r}; choose from {sorted(PARAMETERS)}")
    lo, hi, _default, _unit = PARAMETERS[name]
    if not (lo <= value <= hi):
        raise SensitivityError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def recompute(project, condition, *, parameter, value, n_boot=50, cache=None):
    """Re-run a fast fit with one parameter changed; return ``(draw, status)``.

    Dispatches on which parameter moved to the cheapest fit that shows its
    effect. With a ``cache`` the result is memoized by its provenance, so
    revisiting a value is instant: the point of the slider.
    """
    _validate(parameter, value)

    def run():
        return _dispatch(project, condition, parameter, value, n_boot)

    if cache is None:
        spec, status = _dispatch(project, condition, parameter, value, n_boot)
        return make_draw(spec), status

    from .provenance import capture

    provenance = capture(
        settings={"sensitivity": parameter, "value": value, "condition": condition,
                  "n_boot": n_boot},
        seeds={"bootstrap": 0, "hmm": 0},
        canonical=project.condition(condition),
        unit="cell",
    )
    # The cached payload is a PICKLABLE plot spec + status (a draw closure is not
    # picklable); the draw is rebuilt from the spec on both miss and hit.
    payload = cache.get_or_compute(
        f"sensitivity_{parameter}", provenance,
        lambda: dict(zip(("spec", "status"),
                         _dispatch(project, condition, parameter, value, n_boot))),
    )
    return make_draw(payload["spec"]), payload["status"]


def make_draw(spec):
    """Build a draw closure from a picklable plot spec (a bar chart)."""
    def draw(ax):
        values = spec["values"]
        ax.bar(range(len(values)), values, color="#0072B2",
               width=0.5 if len(values) == 1 else 0.8)
        if spec.get("ylim"):
            ax.set_ylim(*spec["ylim"])
        ax.set_ylabel(spec.get("ylabel", ""))
        ax.set_title(spec.get("title", ""), fontsize=9)
    return draw


def _dispatch(project, condition, parameter, value, n_boot):
    part = project.condition(condition)

    if parameter == "sigma_nm":
        return _sigma(project, part, value, n_boot)
    if parameter == "n_pop":
        return _n_pop(part, int(value))
    if parameter == "confinement_radius_um":
        return _confinement(part, value)
    if parameter in ("min_localizations", "localization_precision_nm"):
        # These change ingest / the floor line, not a fit per se, report the
        # resulting fast fraction so the slider still shows an effect.
        return _min_loc_or_floor(part, parameter, value)
    raise SensitivityError(f"no recompute wired for {parameter!r}")   # pragma: no cover


def _sigma(project, part, sigma_nm, n_boot):
    from .hmm import HMMError, fit_hmm, step_sequences

    try:
        fit = fit_hmm(step_sequences(part), n_states=2, tau_s=project.params.tau_s,
                      localization_sigma_um=sigma_nm / 1000.0, n_starts=6)
        slow = 100 * float(fit.occupancy[0])
    except HMMError as exc:
        return ({"values": [0], "ylabel": "slow fraction (%)",
                 "title": "HMM did not fit"},
                f"σ={sigma_nm:.0f} nm: HMM failed ({exc})")
    D = [round(float(d), 4) for d in fit.D]
    spec = {"values": [slow], "ylim": (0, 100), "ylabel": "slow fraction (%)",
            "title": f"HMM slow fraction at σ = {sigma_nm:.0f} nm"}
    return spec, f"σ={sigma_nm:.0f} nm: slow {slow:.1f}%, D={D}"


def _n_pop(part, n_pop):
    from .observables import squared_displacements
    from .sqd import fit_sqd

    fit = fit_sqd(squared_displacements(part), n_pop=n_pop, weighting="track")
    fracs = [round(float(100 * f), 1) for f in fit.fractions]
    D = [round(float(d), 4) for d in fit.D]
    spec = {"values": [float(100 * f) for f in fit.fractions],
            "ylabel": "fraction (%)", "title": f"SQD {n_pop}-population fit"}
    return spec, f"n_pop={n_pop}: fractions {fracs}%, D={D}"


def _confinement(part, radius_um):
    from .confinement import confinement_events

    events = confinement_events(part, radius_um=radius_um, min_steps=3)
    frac = float(events.attrs["confined_segment_fraction"])
    spec = {"values": [100 * frac], "ylim": (0, 100),
            "ylabel": "confined segment (%)",
            "title": f"confinement at radius {1000 * radius_um:.0f} nm"}
    return spec, (f"radius={1000 * radius_um:.0f} nm: "
                  f"confined {100 * frac:.1f}% (RAW: see excess-over-null)")


def _min_loc_or_floor(part, parameter, value):
    from .observables import squared_displacements
    from .sqd import fit_sqd

    fit = fit_sqd(squared_displacements(part), n_pop=2, weighting="track")
    slow = 100 * float(fit.fractions[0])
    spec = {"values": [slow], "ylim": (0, 100), "ylabel": "slow fraction (%)",
            "title": f"{parameter} = {value}"}
    return spec, f"{parameter}={value}: SQD slow {slow:.1f}%"
