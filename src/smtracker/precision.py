"""PER-SPOT PRECISION: each localisation's own error in the step models (SQD now; HMM and apparent D to follow).

WHY (6 Nov 2026; docs/synthetic_synr.md, "Where glucose's bound share goes"): a bound molecule's step is
localisation error alone, and that error differs from spot to spot -- small for a bright spot in focus, larger for a
dim one. One SQD population has one typical step length, so the fit takes the well-placed bound steps as its slow
population and reads the rest as a slower MOBILE one: on SynR it read 22.6 % of a 36.9 % bound share, and the same
steps given one precision read 37.9 %. Given each spot's own precision, a population's expected r^2 for step i is

    2 ndim D tau + ndim (s_a^2 + s_b^2)

with s_a, s_b the TRUE per-axis error of the step's two spots. A bound step is then EXPECTED to be long where its
spots were placed badly, and is no longer read as movement.

WHAT IS NEGLECTED, SAID (10 Nov 2026, the per-spot audit's finding 7): consecutive steps of a track share their
middle spot, so their errors are correlated, and the SQD mixture and the HMM's emission treat a track's steps as
independent given their noise, as the fits without per-spot precision do. The apparent D's gamma did too until the
same day; under Per spot it now carries that correlation in its SHAPE (`appd_gamma.shape_and_scale`), because there
the correlation changes the fitted law itself (a track's value is a mean over correlated steps) and not only how many
independent steps a likelihood counts. A calibration
measured on a stationary measurement's own steps (`measure_from_tracks`) is an EFFECTIVE per-step noise at lag 1,
that correlation included, and holds for lag-1 steps. The refusal of lags 1-4 together rests on another ground: they
reuse each jump in up to four displacements, so a likelihood over them counts it up to four times.

THE TRACKER'S NUMBER IS A LOWER BOUND, so it reaches a fit only through a CALIBRATION, a curve from the reported to
the true per-spot sigma (docs/claude/sigma_and_floor.md, "u-track's per-spot sigma against the TRUE error": on SynR's
bound spots u-track ranks the spots right and reads 1.3-2.2 times small). The curve is a power law,
sigma_true = c * sigma_reported^p (nm), from one of:

  - REPORTED: the tracker's numbers unchanged (c 1, p 1), for comparison: still a lower bound;
  - `measure`: fitted to stationary spots whose true error is known -- a stationary synthetic set (Stationary_N1_3d,
    knows only what the simulator models), SynR's bound spots, or later a stationary measurement on the microscope
    (fixed cells, immobilised protein); kept as a file (`save` / `load`).

NEVER LEARNED BY THE FIT (ruled 6 Nov 2026: "yes DROP B this is important"). Learned together with the populations,
the calibration took the freedom to make short moving steps "bound": on SynR's mutants (6.5 % bound) it read 26-30 %,
an invented bound population (docs/claude/sqd_and_statistics.md, "PER-SPOT PRECISION"). It was built, tested and
removed; it is not to come back. A calibration is MEASURED on spots known to be immobile, or it is not used.

Interfaces:
  Calibration(c_nm, p, source)      the curve; `true_sigma_um(reported_um)`, `words()`
  REPORTED                          the identity, labelled as the tracker's lower bound
  measure(reported_um, error_um)    the power law by maximum likelihood on per-spot errors; with the decile check
  measure_from_tracks(canonical)    the power law from a STATIONARY MEASUREMENT's own steps, no truth needed: the
                                    tool for fixed cells or immobilised protein on the microscope (stationarity
                                    checked, outliers modelled)
  spot_sigmas(sd)                   each step's two spots' reported sigma (um, n x 2), refused where missing
  step_noise(sd, calibration)       each step's per-axis localisation variance (um^2), aligned with ``sd``'s rows
  save(path, calibration) / load(path)

Invariants: never a guessed precision (a table without the tracker's column is refused by name, never filled); NaN
anywhere is refused, never dropped silently; the calibration and its source travel together; the calibration is
always given, never fitted to the data being analysed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .observables import LOC_SIGMA_STEP_COLUMNS

__all__ = ["PrecisionError", "Calibration", "REPORTED", "STATIONARY_LAG_RATIO_MAX", "measure", "measure_from_tracks",
           "MAX_MISSING_SHARE", "PRESETS", "PROJECT_FILE", "calibration_for", "usable_steps", "spot_sigmas",
           "step_noise", "save", "load", "carries_precision", "conditions_with_precision", "noise_groups",
           "NOISE_BINS_PER_LOG", "acquisition_of", "acquisition_mismatch", "ACQUISITION_TOLERANCE"]

#: A model's CDF over jumps of many noises: the noises grouped into bins this many to a natural-log unit, each at its
#: mean (`noise_groups`), within about 1e-3 of the per-jump sum.
NOISE_BINS_PER_LOG = 20


class PrecisionError(RuntimeError):
    """Raised when per-spot precision is missing, malformed, or a calibration cannot be made."""


@dataclass(frozen=True)
class Calibration:
    """sigma_true = c_nm * sigma_reported_nm ** p, per spot and per axis; ``source`` says where it comes from."""

    c_nm: float = 1.0
    p: float = 1.0
    source: str = "the tracker's own numbers, uncalibrated (a lower bound)"
    #: How it was measured, for the record: spots used, the decile check (`measure`).
    record: dict = field(default_factory=dict)

    def __post_init__(self):
        if not (np.isfinite(self.c_nm) and self.c_nm > 0 and np.isfinite(self.p) and self.p > 0):
            raise PrecisionError(f"a calibration needs c > 0 and p > 0, got c {self.c_nm}, p {self.p}")

    def true_sigma_um(self, reported_um) -> np.ndarray:
        reported_nm = 1000.0 * np.asarray(reported_um, dtype=float)
        return self.c_nm * np.power(reported_nm, self.p) / 1000.0

    def words(self) -> str:
        if self.c_nm == 1.0 and self.p == 1.0:
            return f"per-spot precision as the tracker reported it ({self.source})"
        return f"per-spot precision, calibrated: sigma = {self.c_nm:.3g} x reported^{self.p:.3g} (nm), {self.source}"


REPORTED = Calibration()


def measure(reported_um, error_x_um, error_y_um, *, source: str, bins: int = 10) -> Calibration:
    """The power law from STATIONARY spots: each spot's reported sigma and its true error on each axis.

    Maximum likelihood: each axis error ~ N(0, sigma_true^2), sigma_true = c reported^p; minimised over (log c, p).
    The record keeps the decile check (mean reported, true RMS, the curve's value) so a reader sees whether a power
    law describes the spots at all."""
    from scipy.optimize import minimize

    rep_nm = 1000.0 * np.asarray(reported_um, dtype=float)
    ex = 1000.0 * np.asarray(error_x_um, dtype=float)
    ey = 1000.0 * np.asarray(error_y_um, dtype=float)
    if not (len(rep_nm) == len(ex) == len(ey)):
        raise PrecisionError("reported sigma and the errors must have one value per spot")
    keep = np.isfinite(rep_nm) & (rep_nm > 0) & np.isfinite(ex) & np.isfinite(ey)
    if keep.sum() < 50 * bins:
        raise PrecisionError(f"only {int(keep.sum())} usable spots: too few to measure a calibration "
                             f"({50 * bins} at least)")
    rep_nm, ex, ey = rep_nm[keep], ex[keep], ey[keep]
    log_rep = np.log(rep_nm)
    e2 = ex ** 2 + ey ** 2

    def nll(theta):
        # two axes, each N(0, s^2): -log L = sum(log s^2 + (ex^2 + ey^2) / (2 s^2)) + const; a trial far off (the
        # simplex's first steps) overflows to inf, which the simplex simply rejects
        log_c, p = theta
        log_s2 = 2.0 * (log_c + p * log_rep)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = float(np.sum(log_s2 + e2 / (2.0 * np.exp(log_s2))))
        return value if np.isfinite(value) else np.inf

    start = np.array([np.log(np.sqrt(np.mean(e2) / 2.0) / np.median(rep_nm)), 1.0])
    result = minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    if not result.success:
        raise PrecisionError(f"the calibration fit did not converge: {result.message}")
    c_nm, p = float(np.exp(result.x[0])), float(result.x[1])
    edges = np.quantile(rep_nm, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, rep_nm, side="right") - 1, 0, bins - 1)
    deciles = []
    for b in range(bins):
        sel = which == b
        mean_rep = float(rep_nm[sel].mean())
        deciles.append({"spots": int(sel.sum()), "reported_nm": mean_rep,
                        "true_rms_nm": float(np.sqrt(e2[sel].mean() / 2.0)),
                        "curve_nm": float(np.sqrt(np.mean((c_nm * rep_nm[sel] ** p) ** 2)))})
    record = {"spots": int(keep.sum()), "deciles": deciles,
              "true_over_reported": float(np.sqrt(e2.mean() / 2.0) / np.sqrt(np.mean(rep_nm ** 2)))}
    return Calibration(c_nm=c_nm, p=p, source=source, record=record)


#: A stationary measurement whose steps two frames apart are longer than its one-frame steps by more than this ratio
#: (of their MEDIANS over each step's reported noise) is moving -- drift, molecules not fixed -- and its calibration
#: would read the movement as error (`measure_from_tracks`). The stationary synthetic set reads 1.03.
STATIONARY_LAG_RATIO_MAX = 1.10


def measure_from_tracks(canonical: pd.DataFrame, *, source: str, bins: int = 10) -> Calibration:
    """(See below.) The acquisition of ``canonical`` (its ``attrs["params"]``) is kept in the record
    (``record["acquisition"]``, 10 Nov 2026), so a Run elsewhere can say it differs (`acquisition_mismatch`)."""
    cal = _measure_from_tracks(canonical, source=source, bins=bins)
    acquisition = acquisition_of(canonical.attrs.get("params"))
    if not acquisition:
        return cal
    return Calibration(c_nm=cal.c_nm, p=cal.p, source=cal.source, record={**cal.record, "acquisition": acquisition})


def _measure_from_tracks(canonical: pd.DataFrame, *, source: str, bins: int = 10) -> Calibration:
    """The power law from a STATIONARY MEASUREMENT's own tracks, where no truth exists (fixed cells, immobilised
    protein on the microscope; asked 6 Nov 2026, "the software being ready to be calibrated against that").

    Every molecule is immobile, so each step's displacement is the difference of its two spots' errors: per axis a
    variance s_a^2 + s_b^2, r^2 exponential with mean m = 2 (s_a^2 + s_b^2) (2D), s = c reported^p. That is also
    exactly the noise a one-frame step carries in the SQD model, so a calibration measured this way is the one the fit
    needs (it includes whatever correlation consecutive errors have; +0.08 on the stationary set). It is therefore an
    EFFECTIVE per-step noise at lag 1, not the spots' true error alone, and holds for lag-1 steps.

    OUTLIERS ARE PART OF THE MODEL, not trimmed: a mislink or a false detection is a long step (7 % of the stationary
    set's steps exceed 20 times their noise), so the fit is (1 - e) Exp(m_i) + e Exp(M) with e and M free; the record
    keeps e. THE STATIONARITY IS CHECKED, not assumed: an immobile molecule's steps two frames apart are as long as its
    one-frame steps, so the MEDIANS of r^2 over the step's reported noise at the two lags are compared (medians, since
    one mislink makes one long step at lag 1 and two at lag 2); above `STATIONARY_LAG_RATIO_MAX` the calibration is
    REFUSED."""
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp

    from .observables import squared_displacements

    sd = squared_displacements(canonical, lag_frames=1)
    r2 = 1e6 * sd["sq_disp_um2"].to_numpy(dtype=float)
    # the count first (10 Nov 2026, the audit's finding 8): a near-empty table took a median of almost nothing before
    if len(r2) < 50 * bins:
        raise PrecisionError(f"only {len(r2)} steps: too few to measure a calibration ({50 * bins} at least)")
    sd2 = squared_displacements(canonical, lag_frames=2)
    spots = 1000.0 * spot_sigmas(sd)
    spots2 = 1000.0 * spot_sigmas(sd2)
    r2_2 = 1e6 * sd2["sq_disp_um2"].to_numpy(dtype=float)
    ratio = float(np.median(r2_2 / (spots2 ** 2).sum(axis=1)) / np.median(r2 / (spots ** 2).sum(axis=1)))
    if ratio > STATIONARY_LAG_RATIO_MAX:
        raise PrecisionError(
            f"these molecules move: steps two frames apart are {ratio:.2f} times the one-frame steps (median; an "
            f"immobile molecule's are equal, at most {STATIONARY_LAG_RATIO_MAX} accepted). Drift or unfixed molecules "
            f"would be read as localisation error, so no calibration is made from them.")
    log_a, log_b = np.log(spots[:, 0]), np.log(spots[:, 1])

    def nll(theta):
        # (1 - e) Exp(m_i) + e Exp(M), m_i = 2 c^2 (a^2p + b^2p); in log space
        log_c, p, logit_e, log_M = theta
        e = float(expit(logit_e))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            m = 2.0 * np.exp(2.0 * log_c) * (np.exp(2.0 * p * log_a) + np.exp(2.0 * p * log_b))
            M = np.exp(log_M)
            terms = np.stack([np.log1p(-e) - np.log(m) - r2 / m, np.full_like(r2, np.log(e) - log_M) - r2 / M])
            value = -float(np.sum(logsumexp(terms, axis=0)))
        return value if np.isfinite(value) else np.inf

    rms_rep = np.sqrt((spots ** 2).mean(axis=1))
    typical = np.median(r2) / np.log(2.0)                          # the bulk's mean r^2, from its median
    start = np.array([np.log(np.sqrt(typical / 4.0) / np.median(rms_rep)), 1.0, np.log(0.05 / 0.95),
                      np.log(20.0 * typical)])
    result = minimize(nll, start, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000, "maxfev": 20000})
    if not result.success:
        raise PrecisionError(f"the calibration fit did not converge: {result.message}")
    c_nm, p = float(np.exp(result.x[0])), float(result.x[1])
    edges = np.quantile(rms_rep, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, rms_rep, side="right") - 1, 0, bins - 1)
    deciles = []
    for b in range(bins):
        sel = which == b
        curve = c_nm * spots[sel] ** p
        # observed from the MEDIAN (outlier-proof): an exponential's median is its mean x ln 2, and m = 4 s^2
        deciles.append({"steps": int(sel.sum()), "reported_nm": float(rms_rep[sel].mean()),
                        "observed_nm": float(np.sqrt(np.median(r2[sel]) / np.log(2.0) / 4.0)),
                        "curve_nm": float(np.sqrt(np.mean(curve ** 2)))})
    record = {"steps": int(len(r2)), "tracks": int(canonical["uid"].nunique()), "lag2_over_lag1": ratio,
              "outlier_share": float(expit(result.x[2])), "outlier_rms_nm": float(np.sqrt(np.exp(result.x[3]) / 2.0)),
              "deciles": deciles, "from": "the measurement's own one-frame steps (no truth)"}
    return Calibration(c_nm=c_nm, p=p, source=source, record=record)


#: THE BUILT-IN CALIBRATIONS (6 Nov 2026), one per tracker, measured on the stationary synthetic set
#: `a synthetic stationary set (Stationary_N1_3d)` against its truth (`scripts/synthetic_builds/stationary_calibration.py`):
#: NIKON 1 at 106 nm and 20 ms, each tracker at the lab standard. They know only what the simulator models; another
#: microscope needs its own (a stationary measurement, `measure_from_tracks`, saved as `PROJECT_FILE`).
#: THE ACQUISITION THEY WERE MEASURED AT (10 Nov 2026, the per-spot audit's finding 5): the stationary set's own
#: movieData (106 nm, 20 ms, PSF 0.932 px), checked against every Run's (`acquisition_mismatch`).
STATIONARY_SET_ACQUISITION = {"pixel_size_nm": 106.0, "frame_interval_ms": 20.0, "psf_sigma_nm": 0.9316808592733641 * 106.0}
PRESETS = {
    "utrack": Calibration(
        c_nm=3.261813683878758, p=0.7632051683220792,
        source="the built-in u-track calibration: the stationary synthetic set (Nikon 1, u-track at the lab "
               "standard) against its truth; it knows only what the simulator models",
        record={"acquisition": dict(STATIONARY_SET_ACQUISITION)}),
    "trackmate": Calibration(
        c_nm=2.6208475673418032, p=0.8247030154608546,
        source="the built-in TrackMate calibration: the stationary synthetic set (Nikon 1, TrackMate at the lab "
               "standard, each spot fitted by spotfit) against its truth; it knows only what the simulator models",
        record={"acquisition": dict(STATIONARY_SET_ACQUISITION)}),
}

#: How far a Run's acquisition may sit from a calibration's before its words say so: the pixel size by the project's
#: own tolerance for mixed pixel sizes (1 %, ruled 26 Oct 2026), the frame interval by the one for mixed intervals
#: (10 %, ruled 22 Oct 2026), the PSF width (in nm) by 10 %.
ACQUISITION_TOLERANCE = {"pixel_size_nm": 0.01, "frame_interval_ms": 0.10, "psf_sigma_nm": 0.10}
_ACQUISITION_WORDS = {"pixel_size_nm": ("pixels of", "nm"), "frame_interval_ms": ("frames of", "ms"),
                      "psf_sigma_nm": ("a PSF width of", "nm")}


def acquisition_of(params) -> dict:
    """What a calibration's transfer depends on, from an `AcquisitionParams` (or anything with its fields): the pixel
    size, the frame interval and the PSF width in nm; a field the record lacks is left out."""
    out = {}
    px = getattr(params, "pixel_size_nm", None)
    tau = getattr(params, "frame_interval_ms", None)
    psf = getattr(params, "psf_sigma_px", None)
    if px and np.isfinite(px):
        out["pixel_size_nm"] = float(px)
    if tau and np.isfinite(tau):
        out["frame_interval_ms"] = float(tau)
    if psf and px and np.isfinite(psf):
        out["psf_sigma_nm"] = float(psf) * float(px)
    return out


def acquisition_mismatch(calibration: "Calibration", params) -> str:
    """"" where ``params`` (the Run's acquisition) matches what ``calibration`` was measured at, or either is not
    known; else the sentence saying how they differ (10 Nov 2026, the per-spot audit's finding 5). A calibration is a
    curve whose exponent is not 1, so it is not free of scale: one measured on one microscope may not hold on another.
    Said, never refused: the built-in curve is still the best guess a project without its own measurement has."""
    made = dict((calibration.record or {}).get("acquisition") or {})
    here = acquisition_of(params)
    off = [(k, made[k], here[k]) for k, tol in ACQUISITION_TOLERANCE.items()
           if made.get(k) and here.get(k) and abs(here[k] - made[k]) / made[k] > tol]
    if not off:
        return ""
    then = ", ".join(f"{_ACQUISITION_WORDS[k][0]} {a:.3g} {_ACQUISITION_WORDS[k][1]}" for k, a, _b in off)
    now = ", ".join(f"{b:.3g} {_ACQUISITION_WORDS[k][1]}" for k, _a, b in off)
    return (f"measured at {then}; these tracks were taken at {now}, so the curve may not hold here, and a stationary "
            f"measurement on this microscope gives its own")
#: A project's own calibration (a stationary measurement's, `measure_from_tracks` -> `save`), read in place of the
#: built-in one from a condition's folder or the project folder above it (`calibration_for`).
PROJECT_FILE = "precision_calibration.json"


def noise_groups(noise):
    """``(noise, weights)``: each jump's own localisation noise grouped into narrow log bins (`NOISE_BINS_PER_LOG`),
    each at its mean and weighted by its share of the jumps -- a per-spot model's CDF over the jumps (the Jump
    distances and Signed jumps pages' curves) in a few hundred terms instead of one per jump."""
    noise = np.asarray(noise, dtype=float)
    keys = np.round(NOISE_BINS_PER_LOG * np.log(np.maximum(noise, 1e-12))).astype(int)
    _groups, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = np.ravel(inverse)
    return np.bincount(inverse, weights=noise) / counts, counts / counts.sum()


def carries_precision(table: pd.DataFrame) -> bool:
    """Whether ``table``'s tracks carry the tracker's per-spot precision at all (any spot with one): u-track's do,
    TrackMate's when their spots were fitted (`spotfit`); a track table's -- a synthetic TRUTH condition (exact
    positions, no localisation error) or another tool's -- do not. A condition WITHOUT any is fitted in a Per spot Run
    with no localisation-error model (6 Nov 2026, asked: "i cannot do synR with per SPOT ... we need a workaround"),
    exact for a truth condition; one with SOME spots missing is still refused by the fits (`usable_steps`)."""
    from .canonical import LOC_SIGMA_COLUMN

    if LOC_SIGMA_COLUMN not in table.columns:
        return False
    values = table[LOC_SIGMA_COLUMN].to_numpy(dtype=float)
    return bool(np.any(np.isfinite(values) & (values > 0)))


def conditions_with_precision(canonical: pd.DataFrame) -> set:
    """The conditions of ``canonical`` whose tracks carry per-spot precision (`carries_precision`), in one pass."""
    from .canonical import LOC_SIGMA_COLUMN

    if LOC_SIGMA_COLUMN not in canonical.columns or "condition" not in canonical.columns:
        return set()
    values = canonical[LOC_SIGMA_COLUMN].to_numpy(dtype=float)
    ok = np.isfinite(values) & (values > 0)
    return {str(c) for c in np.unique(canonical["condition"].astype(str).to_numpy()[ok])}


def calibration_for(roots, trackers) -> Calibration:
    """The calibration a Run's per-spot precision uses, from its conditions' folders (``roots``) and trackers
    (``trackers``, as `project.load_project` records them in ``canonical.attrs["trackers"]``): ONE
    `PROJECT_FILE` found in a condition's folder or the project folder above it, else the built-in calibration of
    the one tracker the Run's conditions share. Refused, by name: two different files, conditions tracked by
    different trackers without a file, a track table (no tracker's precision)."""
    found = project_file(roots)
    if found is not None:
        return load(found)
    return builtin_for(trackers)


def project_file(roots) -> Path | None:
    """The ONE `PROJECT_FILE` in the conditions' folders (``roots``) or the folder above each, or None; two different
    ones refused by name (`calibration_for`'s first half)."""
    found = set()
    for root in roots:
        for folder in (Path(root), Path(root).parent):
            candidate = folder / PROJECT_FILE
            if candidate.is_file():
                found.add(candidate.resolve())
                break
    if len(found) > 1:
        raise PrecisionError(f"{len(found)} different calibration files for one Run "
                             f"({', '.join(sorted(str(f) for f in found))}): keep one")
    return next(iter(found)) if found else None


def builtin_for(trackers) -> Calibration:
    """The built-in calibration of the one tracker ``trackers`` names (`PRESETS`); refused, by name, for a track
    table or two trackers (`calibration_for`'s second half)."""
    kinds = {str(t) for t in trackers}
    if kinds - set(PRESETS):
        raise PrecisionError(f"no per-spot precision for conditions read from {', '.join(sorted(kinds - set(PRESETS)))}:"
                             f" a track table carries no tracker's precision")
    if len(kinds) != 1:
        raise PrecisionError("the Run's conditions were tracked by u-track and TrackMate, whose calibrations differ: "
                             f"run them apart, or put one {PROJECT_FILE} in the project folder")
    return PRESETS[next(iter(kinds))]


#: At most this share of steps may lack a spot's precision (a spot fit that failed: 1 in 2 051 spots on a SynR movie,
#: 6 Nov 2026); those steps are LEFT OUT, counted and said, as a gap breaks a track. Above it something is wrong with
#: the precision itself (a movie not fitted, a tracker that writes none) and the per-spot fit is refused.
MAX_MISSING_SHARE = 0.01


def usable_steps(sd: pd.DataFrame, *, max_missing: float | None = MAX_MISSING_SHARE) -> np.ndarray:
    """Which steps have both spots' precision (a boolean mask in ``sd``'s order), the rest to be left out by the
    caller and counted. Refused, by name, where the table has no per-spot precision at all or more than
    ``max_missing`` (`MAX_MISSING_SHARE`) of its steps lack it. ``max_missing=None``: no share refused, for a PART of
    a condition whose whole was checked (a replicate's, a cell's or a chunk's fractions at the condition's D: one cell
    with 2 such steps of 97 refused a whole Run, 7 Nov 2026)."""
    missing = [c for c in LOC_SIGMA_STEP_COLUMNS if c not in sd.columns]
    if missing:
        raise PrecisionError(
            "these steps carry no per-spot precision: the tracker wrote none (u-track does; TrackMate does when its "
            "spots were fitted, `spotfit`; track tables do not). Per-spot precision cannot be used on them.")
    spots = sd[list(LOC_SIGMA_STEP_COLUMNS)].to_numpy(dtype=float)
    ok = (np.isfinite(spots) & (spots > 0)).all(axis=1)
    lacking = float(1.0 - ok.mean()) if len(ok) else 0.0
    if max_missing is not None and lacking > max_missing:
        raise PrecisionError(
            f"{int((~ok).sum())} of {len(ok)} steps ({100 * lacking:.1f} %) have a spot without a precision, more "
            f"than the {100 * max_missing:.0f} % a failed fit here and there explains: refused, never filled in")
    return ok


def spot_sigmas(sd: pd.DataFrame) -> np.ndarray:
    """Each step's two spots' REPORTED sigma (um, n x 2), in ``sd``'s row order, before any calibration. Refused, by
    name, where the table has no per-spot precision (a TrackMate or track-table project: the tracker wrote none) or
    where any spot lacks it."""
    missing = [c for c in LOC_SIGMA_STEP_COLUMNS if c not in sd.columns]
    if missing:
        raise PrecisionError(
            "these steps carry no per-spot precision: the tracker wrote none (u-track does; TrackMate does when its "
            "spots were fitted, `spotfit`; track tables do not). Per-spot precision cannot be used on them.")
    spots = sd[list(LOC_SIGMA_STEP_COLUMNS)].to_numpy(dtype=float)
    bad = ~(np.isfinite(spots) & (spots > 0)).all(axis=1)
    if bad.any():
        raise PrecisionError(f"{int(bad.sum())} of {len(spots)} steps have a spot without a precision: refused, "
                             f"never filled in")
    return spots


def step_noise(sd: pd.DataFrame, calibration: Calibration) -> np.ndarray:
    """Each step's per-axis localisation variance (um^2), s_a^2 + s_b^2 through ``calibration``, in ``sd``'s row
    order: what `sqd.fit_sqd(step_noise_um2=...)` takes. Refused as `spot_sigmas` refuses."""
    spots = spot_sigmas(sd)
    return calibration.true_sigma_um(spots[:, 0]) ** 2 + calibration.true_sigma_um(spots[:, 1]) ** 2


def save(path, calibration: Calibration) -> Path:
    path = Path(path)
    path.write_text(json.dumps({"format": "smtracker per-spot precision calibration 1", **asdict(calibration)},
                               indent=2), encoding="utf-8")
    return path


def load(path) -> Calibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.pop("format", None) != "smtracker per-spot precision calibration 1":
        raise PrecisionError(f"{path} is not a per-spot precision calibration file")
    return Calibration(**data)
