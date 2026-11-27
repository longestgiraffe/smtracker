"""The nulls the Clustering section scores k-means against, headless (7 Nov 2026).

k-means always returns clusters and Calinski-Harabasz never says "none", so the index is read against the same pipeline
on tracks with the effect deliberately absent (`clustering` module head). WHICH effect is absent is the question, and the
review of 7 Nov 2026 found the section asked only the weakest one:

``"free"`` -- ONE free Brownian population, no localisation noise. Beating it means only "the per-track MSD curves are not
those of one diffusion coefficient". Any condition with two SQD populations beats it, so on its own it says nothing about
whether clustering finds anything the SQD fit has not already counted.

``"sqd"`` -- tracks drawn from the condition's own FITTED SQD MIXTURE (`smtracker.sqd`, populations chosen by BIC at lag 1
unless a count is set), with the condition's track lengths and localisation noise. Beating it answers "does clustering
find more than the SQD mixture already explains?": shape that the populations, their lengths and the noise cannot make,
such as switching between states, confinement, a state with a non-Brownian curve, or a survivorship effect the length
resampling does not carry. The draw is the condition's OWN mixture, so a pool of conditions is drawn condition by
condition (each its own populations, lengths and noise) and pooled.

INVARIANTS.
- The mixture is FITTED to the data being tested, so the null is not parameter-free: its D and fractions are estimates.
  The fit's uncertainty is not propagated (one fit, many draws); the statement is "against the fitted mixture", not
  "against every mixture".
- Each simulated track keeps one population for its whole length (no switching), and takes it independently of its
  length. The fractions are SQD's, a share of STEPS; real long tracks are slower on average (survivorship), which this null
  does not reproduce: a data excess driven purely by survivorship will therefore show as beyond the SQD mixture. It is
  said in the verdict's caveat, never hidden.
- Localisation noise, PER SPOT where the tracks carry it (7 Nov 2026, after main's per-spot precision was merged in):
  given the Run's calibration (`precision.Calibration`, measured on immobile spots, never learned), the mixture is
  fitted with each step's own noise (`sqd.fit_sqd(precision=...)`, as the SQD section's Per spot), and every
  simulated track COPIES A REAL TRACK: its length and its spots' calibrated per-axis sigma, localisation by
  localisation, so a dim, noisy track stays noisy in the null and the spread of noise between tracks (which alone
  makes MSD curves differ) is the data's own. Tracks with a spot lacking a precision are left out of the pool and
  counted. Without a calibration, or for a condition whose tracks carry no precision (a track table), the sigma is
  the one stated for the acquisition (`AcquisitionParams.localization_precision_nm`) unless the Run gives one; with
  neither it is 0 and the payload says "no localisation error". That ONE sigma is given to the SQD fit and to the
  simulation alike, so the mixture's D are the noise-free ones the simulation then blurs.
- Every draw is seeded: a result is reproducible, which is what lets it be cached.
- Fails loud: a mixture that cannot be fitted raises `ClusteringError`, never returns a null of zeros.
"""
from __future__ import annotations

import math

__all__ = ["NULLS", "NULL_LABELS", "NULL_QUESTIONS", "SURVIVORSHIP_CAVEAT", "effective_sigma", "sqd_mixture",
           "spot_tracks", "copied_tracks", "null_for", "null_words"]

NULLS = ("free", "sqd")
NULL_LABELS = {"free": "One free population", "sqd": "The SQD mixture"}
#: What beating each null means, said once and read by the card, the verdict and the manual.
NULL_QUESTIONS = {
    "free": "Do the curves hold more than one diffusion coefficient? (Any condition with two SQD populations passes.)",
    "sqd": "Does clustering find more than the SQD mixture already explains?",
}
#: What the SQD-mixture null does not carry, said wherever its verdict is (7 Nov 2026, the review's point 3).
SURVIVORSHIP_CAVEAT = ("A drawn track takes its population whatever its length, while real long tracks are slower on "
                       "average (survivorship): a length effect alone reads as structure beyond the mixture.")
#: Fewest steps an SQD mixture is fitted on.
MIN_STEPS = 50


def effective_sigma(table, sigma_nm=None) -> tuple:
    """``(sigma_um, source)``: the localisation error the SQD fit and the simulation share. ``sigma_nm`` given: that.
    Else the acquisition's configured or measured precision; else 0 (``source`` says which)."""
    if sigma_nm is not None:
        if not math.isfinite(float(sigma_nm)) or float(sigma_nm) < 0:
            raise ValueError(f"sigma_nm must be a number >= 0, got {sigma_nm!r}")
        return float(sigma_nm) / 1000.0, "set in the Run"
    params = table.attrs.get("params")
    precision = getattr(params, "localization_precision_nm", None)
    if precision is not None:
        return float(precision) / 1000.0, "the acquisition's stated precision"
    return 0.0, "none stated: no localisation error"


def sqd_mixture(table, *, sigma_um: float = 0.0, max_pop: int = 3, n_pop=None, calibration=None) -> dict:
    """The condition's SQD mixture at lag 1: ``{"D", "fractions", "n_pop", "n_steps", "sigma_um"}``, populations
    ascending in D (D 0 is the immobile population of exact-zero jumps). ``n_pop`` None: the lowest BIC among
    1..``max_pop``; a number: that many. ``calibration`` (a `precision.Calibration`): each step's own noise through it
    (the SQD section's Per spot), never beside ``sigma_um``."""
    from .clustering import ClusteringError
    from .observables import squared_displacements
    from .precision import PrecisionError
    from .sqd import SQDError, fit_sqd, select_n_pop

    steps = squared_displacements(table, lag_frames=1)
    if len(steps) < MIN_STEPS:
        raise ClusteringError(f"only {len(steps)} steps: too few to fit the SQD mixture the null is drawn from")
    noise = ({"precision": calibration} if calibration is not None
             else {"localization_sigma_um": sigma_um})
    try:
        if n_pop is None:
            fit = select_n_pop(steps, max_pop=max_pop, method="mle", gradient="analytic", **noise).recommended
        else:
            fit = fit_sqd(steps, n_pop=int(n_pop), method="mle", gradient="analytic", **noise)
    except (SQDError, PrecisionError) as exc:
        raise ClusteringError(f"the SQD mixture could not be fitted: {exc}") from exc
    D = [float(d) for d in fit.D]
    fractions = [float(f) for f in fit.fractions]
    if not all(math.isfinite(v) for v in D + fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ClusteringError("the SQD fit returned no usable populations")
    return {"D": D, "fractions": fractions, "n_pop": len(D), "n_steps": int(len(steps)), "sigma_um": float(sigma_um),
            "per_spot": calibration is not None}


def spot_tracks(table, calibration) -> tuple:
    """``(sigmas, left_out)``: each track's calibrated per-axis sigma (um), localisation by localisation in frame order,
    for every track whose spots ALL carry the tracker's precision; ``left_out`` counts the tracks that do not."""
    import numpy as np

    from .canonical import LOC_SIGMA_COLUMN, without_attrs

    plain = without_attrs(table)[["uid", "frame", LOC_SIGMA_COLUMN]].sort_values(["uid", "frame"], kind="stable")
    sigmas, left_out = [], 0
    for _uid, group in plain.groupby("uid", sort=False):
        reported = group[LOC_SIGMA_COLUMN].to_numpy(dtype=float)
        if len(reported) < 2 or not np.all(np.isfinite(reported) & (reported > 0)):
            left_out += 1
            continue
        sigmas.append(calibration.true_sigma_um(reported))
    return sigmas, left_out


def copied_tracks(table, sigmas, *, D, fractions, seed: int):
    """Tracks drawn from the mixture, each COPYING a real track (resampled with replacement): its length and its
    spots' calibrated sigma, localisation by localisation (`spot_tracks`)."""
    import numpy as np

    from .simulate import mixture_tracks

    rng = np.random.default_rng(seed)
    chosen = [sigmas[i] for i in rng.integers(0, len(sigmas), size=len(sigmas))]
    out = mixture_tracks([len(s) for s in chosen], D=D, fractions=fractions, params=table.attrs["params"],
                         sigma_um=0.0, seed=seed)
    # `mixture_tracks` writes the tracks in the order given, localisation by localisation
    noise = np.concatenate(chosen)
    out["x_um"] = out["x_um"].to_numpy(dtype=float) + rng.normal(0.0, noise)
    out["y_um"] = out["y_um"].to_numpy(dtype=float) + rng.normal(0.0, noise)
    return out


def null_for(kind: str, tables: dict, *, sigma_nm=None, sqd_pop=None, precision=None):
    """``(null_tracks, info)`` for `clustering.select_k` over the pool of ``tables`` ({condition: canonical table}).

    ``"free"``: ``(None, info)``, select_k's own free Brownian null. ``"sqd"``: a callable ``seed -> pooled canonical
    table`` drawing every condition from its own fitted mixture (`sqd_mixture`), and ``info`` holding each condition's
    populations. ``info["kind"]`` always names the null.

    ``precision``: {condition: (c_nm, p, source) or None}, the Run's per-spot calibration (`workspace_view.run_precision`)
    for each condition whose tracks carry precision. Such a condition is fitted and drawn PER SPOT (module head); a
    condition with None, or every condition when a sigma is typed (``sigma_nm``), takes one sigma (`effective_sigma`)."""
    from .clustering import ClusteringError, pool_tables
    from .precision import Calibration, carries_precision
    from .simulate import matched_mixture_null

    if kind not in NULLS:
        raise ValueError(f"the null is one of {NULLS}, got {kind!r}")
    if kind == "free":
        return None, {"kind": "free", "question": NULL_QUESTIONS["free"]}
    mixtures, sigmas, copies = {}, {}, {}
    for name, table in tables.items():
        given = (precision or {}).get(name)
        calibration = None
        if given is not None and sigma_nm is None and carries_precision(table):
            calibration = Calibration(c_nm=float(given[0]), p=float(given[1]), source=str(given[2]))
        try:
            if calibration is not None:
                copies[name], left_out = spot_tracks(table, calibration)
                if not copies[name]:
                    raise ClusteringError("no track has a precision on every spot")
                mixtures[name] = sqd_mixture(table, n_pop=sqd_pop, calibration=calibration)
                source = calibration.words() + (f"; {left_out} tracks with a spot lacking it left out" if left_out
                                                else "")
                sigmas[name] = (None, source)
            else:
                sigma_um, source = effective_sigma(table, sigma_nm)
                mixtures[name] = sqd_mixture(table, sigma_um=sigma_um, n_pop=sqd_pop)
                sigmas[name] = (sigma_um, source)
        except ClusteringError as exc:
            raise ClusteringError(f"{name}: {exc}") from exc

    def draw(seed: int):
        simulated = {}
        for i, (name, table) in enumerate(tables.items()):
            mix = mixtures[name]
            if name in copies:
                simulated[name] = copied_tracks(table, copies[name], D=mix["D"], fractions=mix["fractions"],
                                                seed=seed * 1009 + i)
            else:
                simulated[name] = matched_mixture_null(table, D=mix["D"], fractions=mix["fractions"],
                                                       sigma_um=mix["sigma_um"], seed=seed * 1009 + i)
        return pool_tables(simulated)

    info = {"kind": "sqd", "question": NULL_QUESTIONS["sqd"],
            "mixtures": {name: dict(m, sigma_source=sigmas[name][1]) for name, m in mixtures.items()}}
    return draw, info


def null_words(info: dict, n_draws: int, d_invariant=None) -> str:
    """The sentence a card gives for the chance it used."""
    if info["kind"] == "sqd":
        parts = []
        for name, m in info["mixtures"].items():
            pops = ", ".join(f"{100 * f:.0f} % at D {d:.3g}" for d, f in zip(m["D"], m["fractions"]))
            error = ("each track the localisation error of the real track it copies, " + m["sigma_source"]
                     if m.get("per_spot") else f"{m['sigma_um'] * 1000:.0f} nm error, {m['sigma_source']}")
            parts.append(f"{name}: {pops} ({error})")
        return (f"chance from {n_draws} sets of tracks drawn from each condition's fitted SQD mixture, with its "
                f"track lengths and localisation error ({'; '.join(parts)}). {SURVIVORSHIP_CAVEAT}")
    return (f"chance from {n_draws} sets of one free Brownian population of the same track lengths"
            + (" (the verdict does not depend on their D)." if d_invariant else "."))
