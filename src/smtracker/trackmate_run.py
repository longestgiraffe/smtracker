"""RUNNING TrackMate headless: the writer half, beside `trackmate` which reads.

`smtracker.trackmate` parses TrackMate's XML into the shared raw table. This drives Fiji
to PRODUCE that XML, so a user never opens Fiji themselves and never sees TrackMate's own
interface: SMTracker shows the controls, generates the script, and reads the result back.
The two halves are separate modules because they are separate concerns, one needs Fiji
on the machine, the other needs nothing at all and must keep working on files from a
colleague who ran TrackMate by hand.

Everything is in pixels, and that is forced rather than hoped for. TrackMate's RADIUS
and LINKING_MAX_DISTANCE are expressed in whatever the image calibration says, and a TIFF
carrying a µm calibration would silently reinterpret every number. This project's raw
stage is pixels (because cell outlines are), so the script SETS the calibration to
1 pixel = 1 unit before configuring anything. `trackmate.read_trackmate_xml` already
converts back from µm when a file needs it; here the conversion is avoided entirely.

The defaults match u-track, so the two trackers are comparable. u-track's confirmed
settings on this project are `maxSearchRadius` 5 px, `timeWindow` 1 (no gap closing) and
`mergeSplit` 0 (no merging or splitting). The tracker settings below reproduce exactly
that, so a TrackMate run differs from u-track in its DETECTOR and not in its linking
policy. Change them and the comparison stops being one.

But 5 px is a cap, not the radius u-track used. Read across 400 movies, u-track runs
`minSearchRadius` 0, `maxSearchRadius` 5, `brownStdMult` 3 and `useLocalDensity` 1: the
radius on each link is 3 x that track's OWN Kalman-estimated displacement s.d., clipped
into the bracket. TrackMate's LAP tracker has no adaptive radius, so it gets the cap.
See `UTRACK_LINKING_POLICY`.

Radius and threshold have no default. They are the two settings that decide what
counts as a molecule, and a default would be this module inventing the answer the
detection-threshold optimizer exists to measure. `smtracker.detection` can sweep the
threshold against the false-positive rate outside cell outlines: quantitative, where
TrackMate's own slider is visual.

Not a replacement for u-track. The reference data is u-track and stays u-track; anything
produced here is a second tracker's opinion and needs its own validation before the two
are pooled or compared.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .childproc import NO_WINDOW

from .chart_text import fold_words

__all__ = [
    "TrackMateRunError",
    "TrackMateSettings",
    "UTRACK_EQUIVALENT",
    "run_trackmate",
    "build_script",
    "discover_movies",
    "run_batch",
    "batch_qc",
    "noise_outliers",
    "summarise_run",
    "track_counts",
    "MIN_USABLE_LOCALISATIONS",
    "read_batch",
    "sweep_detection_threshold",
    "SWEEP_FLOOR_THRESHOLD",
    "UTRACK_COMPARABLE",
    "PURITY_TARGET",
    "preset_choices",
    "settings_from_preset",
    "linking_distance_px",
    "FRAME_GAP_NOTE",
    "truncation_bias",
    "max_measurable_D",
    "gap_window",
    "ADAPTIVE_BASE",
    "LINKING_RADIUS_PX",
    "LINKING_RADIUS_NOTE",
    "UTRACK_LINKING_POLICY",
    "max_trackable_D",
    "apparent_D_ceiling",
    "step_length_summary",
    "linking_ambiguity",
    "widening_yield",
    "WIDENING_YIELD_CALIBRATION",
    "WIDENING_VERDICTS",
    "implied_D_from_yield",
]


class TrackMateRunError(RuntimeError):
    """Raised when a headless TrackMate run cannot be made or trusted."""


#: The linking policy u-track used on this project, reproduced so the two trackers differ
#: only in detection. From `utrack_meta`, CONFIRMED: `maxSearchRadius` 5 px, gap-closing
#: `timeWindow` 1 (i.e. none), `mergeSplit` 0. The 5 is u-track's CEILING on an adaptive
#: per-track radius (`UTRACK_LINKING_POLICY`); TrackMate applies it to every link,
#: because its tracker has no adaptive option.
UTRACK_EQUIVALENT = {
    "linking_max_distance_px": 5.0,
    "gap_closing_max_distance_px": 5.0,
    "max_frame_gap": 0,            # 0 = no gap closing, matching timeWindow = 1
    "allow_splitting": False,
    "allow_merging": False,
}


#: SETTINGS THAT MAKE TRACKMATE COMPARABLE WITH u-track ON THIS DATA, measured rather
#: than guessed (a lab project, 8 movies per condition, identical pipeline both sides).
#:
#: The threshold DIFFERS BY CONDITION because one condition is the dimmer one. These two values
#: were chosen by MATCHING u-track's YIELD and then confirmed end to end (fractions within
#: 1.5 pp); purity at them is 91.5 % and 92.3 % in the two conditions, against u-track's own 92.4 %.
#:
#: **A PURITY TARGET IS NOT A SUBSTITUTE FOR YIELD MATCHING, AND CLAIMING IT WAS IS A
#: MISTAKE RECORDED HERE.** "Pick the threshold where in-cell purity reaches ~92 %" looked
#: self-contained -- it needs only cell outlines, no reference tracker -- and it reproduced
#: the yield-matched answer on one lab condition, which is the single case it was derived from.
#: Checked on five other proteins it does NOT: it lands at a stable 50-70 (median 60) but
#: yields **2.35 to 5.63x u-track's detections** (median 3.13x), and on another condition it gives
#: ~100 where yield matching gives 130. The two criteria genuinely disagree by two to six
#: fold.
#:
#: PURITY CANNOT ADJUDICATE BETWEEN THEM, which is the whole difficulty: at ~92 % purity
#: BOTH trackers are equally clean while TrackMate finds several times more. Either u-track
#: is conservative and the extras are real molecules, or purity is too coarse a bar to
#: tell. That is the RECALL question, and nothing measurable on real data answers it --
#: only simulated movies with known truth would.
#:
#: GAP CLOSING IS ON, WHICH DEPARTS FROM u-track'S POLICY DELIBERATELY. u-track runs
#: `timeWindow = 1`, i.e. none, and matching that was the original mistake: u-track does
#: not NEED gap closing because its mixture-model fit re-finds a molecule reliably, while
#: TrackMate's LoG flickers near threshold and the track breaks. WHAT SHOULD MATCH IS THE
#: RESULT, NOT THE POLICY. Without gap closing the counts never converge; with gap 3 they
#: do. Frame gaps are safe downstream because `observables.squared_displacements` pairs
#: by FRAME DIFFERENCE by default, so a bridged gap is excluded rather than counted as a
#: single step.
#:
#: Linking stays at 5 px: 3 px was measured much worse (129 against 246 usable tracks).
UTRACK_COMPARABLE = {
    "Glucose": {"radius_px": 3.0, "threshold": 60.0, "max_frame_gap": 3},
    "Malate": {"radius_px": 3.0, "threshold": 130.0, "max_frame_gap": 3},
}

#: The purity these settings happen to sit at. NOT a criterion to select by -- see above:
#: selecting on it gives 2.35-5.63x u-track's yield across five other proteins. Kept as a
#: SANITY FLOOR: a threshold whose purity falls below this is admitting background, which
#: is a different and much cheaper judgement than choosing the threshold.
PURITY_FLOOR = 0.92
PURITY_TARGET = PURITY_FLOOR          # backwards-compatible alias


#: **THE SPOT RADIUS AND THE LINKING RADIUS ARE DIFFERENT QUANTITIES, AND 3 AGAINST 5 IS
#: NOT A DISAGREEMENT.** Asked directly ("is 3 good? in utrack 5?") and the honest
#: answer is that they measure different things and both are right:
#:
#:   SPOT RADIUS (3 px)     how big a MOLECULE looks. Set by the microscope: the LoG
#:                          kernel is matched to the PSF, so it follows from emission
#:                          wavelength, NA and pixel size. DERIVABLE: `optics.
#:                          detection_radius_px` gives 2.80 px on this project's optics,
#:                          3.50 px for a red dye at 646 nm, 4.34 px at NA 1.2.
#:   LINKING RADIUS (5 px)  how far a molecule may MOVE between frames. Set by the
#:                          physics: `sqrt(4 D tau)` for the fastest population, with an
#:                          allowance. u-track's 5 px is this one.
#:
#: Nothing about the microscope makes those equal, and a window that calls both "radius"
#: invites exactly the question that was asked.
_RADIUS_KINDS = ("spot", "linking")


def linking_distance_px(*, d_um2_s: float, tau_s: float, pixel_size_nm: float,
                        n_sigma: float = 2.0) -> dict:
    """How far the FASTEST population travels between frames, as a linking radius.

    A 2-D Brownian step has RMS displacement `sqrt(4 D tau)`; `n_sigma` is how many of
    those a link is allowed to span. **`n_sigma` = 2 is a CHOICE, not a constant**, and
    it is reported rather than buried: the step length is Rayleigh with per-coordinate
    scale `sqrt(2 D tau)`, so a radius of `n` RMS steps is overshot with probability
    **`exp(-n^2)`**, 1.8 % at n = 2, 37 % at n = 1. Each overshoot fails the link and
    breaks the track.

    **CORRECTED: this returned `exp(-n^2/2)` at first, which is the survival for `n`
    PER-COORDINATE standard deviations, not for `n` RMS steps**, the two differ by a
    factor sqrt(2) inside the exponent, so it reported 13.5 % where the truth is 1.8 %.
    Both formulas are right for their own units and the code was using one while the
    docstring quoted the other. Quote a survival WITH the unit its multiplier is in.

    D has no default, and the circularity is real: D comes from the analysis, which
    comes after tracking. So this is a helper for a number you must supply from a pilot
    fit or the literature, never an automatic setting. What it IS good for is checking
    a number you already have: this project's D_fast 0.85 um^2/s at tau 20 ms and
    106 nm pixels gives 4.92 px, which is where u-track's 5 comes from.

    u-track did not solve the circularity by picking a number: it adapts per TRACK,
    and 5 px is a cap. Read from `movieData.mat` across 400 MSD movies (see
    `UTRACK_LINKING_POLICY`): `minSearchRadius` 0, `maxSearchRadius` 5, `brownStdMult`
    3, `useLocalDensity` 1. The radius actually used on any given link is 3 x the
    Kalman-estimated per-coordinate displacement s.d. of THAT track, clipped to the
    bracket and shrunk further where the field is crowded. In this function's units
    3 per-coordinate s.d. is `3/sqrt(2)` = **2.12 RMS steps**, so u-track's allowance
    and the 2x default here are within 6 % of each other. TrackMate's LAP tracker has
    no adaptive radius, so a single number must be chosen; the cap is the right one to
    choose, and it is what the box already holds.
    """
    from math import exp, sqrt

    if not (d_um2_s and d_um2_s > 0):
        raise TrackMateRunError(
            "linking_distance_px needs a diffusion coefficient and has no default: the "
            "linking radius is how far the fastest population moves in one frame, and "
            "a guess would silently decide which molecules can be tracked at all")
    if not (tau_s and tau_s > 0):
        raise TrackMateRunError(f"frame interval must be positive, got {tau_s!r}")
    if not (pixel_size_nm and pixel_size_nm > 0):
        raise TrackMateRunError(f"pixel size must be positive, got {pixel_size_nm!r}")
    rms_um = sqrt(4.0 * float(d_um2_s) * float(tau_s))
    radius_um = float(n_sigma) * rms_um
    return {
        "linking_max_distance_px": radius_um * 1000.0 / float(pixel_size_nm),
        "rms_step_nm": rms_um * 1000.0,
        "n_sigma": float(n_sigma),
        # Rayleigh survival IN RMS UNITS: P(r > n·RMS) = exp(-n²), because the
        # per-coordinate scale is RMS/sqrt(2). The share of genuine steps that overshoot
        # the radius and break their own track, the cost of choosing n_sigma.
        "steps_missed": float(exp(-float(n_sigma) ** 2)),
        # The same allowance expressed the way u-track parametrises it, so the two are
        # comparable without anybody converting by hand and getting the sqrt(2) wrong.
        "brown_std_mult_equivalent": float(n_sigma) * 2.0 ** 0.5,
        "d_um2_s": float(d_um2_s), "tau_s": float(tau_s),
        "pixel_size_nm": float(pixel_size_nm),
    }


#: There is no correlation between the spot radius and the linking radius, and the
#: reason is short: the only variable they share cancels. Asked directly: "is there a
#: correlation between spot radius and linking radius we should make?"
#:
#:     spot radius  = 3 sigma_PSF / pixel,      sigma_PSF from emission wavelength and NA
#:     link radius  = n sqrt(4 D tau) / pixel,  from the fastest population and the frame
#:
#: Both divide by the pixel size, so in their RATIO the pixel size cancels and what is
#: left compares `sqrt(D tau)` against `lambda / NA`, optics against kinetics, with no
#: physics joining them. COMPUTED over plausible regimes, ratio link/spot:
#:
#:     this project (mNG, NA 1.49, D 0.85, 20 ms)   2.80 / 4.92   1.76
#:     JF646 on the same scope                      3.50 / 4.92   1.41
#:     NA 1.20 objective                            3.48 / 4.92   1.42
#:     slow membrane protein, D 0.05                2.80 / 1.19   0.43
#:     free cytoplasmic GFP, D 8                    2.80 / 15.09  5.39
#:     100 ms frames                                2.80 / 11.00  3.93
#:     65 nm pixels                                 4.56 / 8.02   1.76   <- unchanged
#:
#: **0.43 to 5.39, a factor of 12.5**, and the one row that does NOT move the ratio is
#: the one that changes the pixel size, which is exactly the shared term. A rule tying
#: them would be right on this microscope and wrong by fivefold on a faster protein,
#: while looking principled. So they stay two boxes.
_SPOT_LINK_RATIO_IS_NOT_A_CONSTANT = True


def max_trackable_D(*, radius_px: float, tau_s: float, pixel_size_nm: float,
                    n_sigma: float = 2.0) -> float:
    """The fastest D a given linking radius can follow. `linking_distance_px` inverted.

    The question a new project actually asks. "5 px worked on one protein: is it right at
    50 ms?" is answered by turning the radius back into the physics it encodes, and the
    answer is no: the radius buys `sqrt(D·tau)`, so **2.5x the frame interval costs 2.5x
    the trackable D at a fixed radius**. Measured on the two real projects here, both at
    106 nm, both with `n_sigma` 2:

        20 ms project   tau 20.4717 ms   5 px follows D up to 0.857 um^2/s
        protein I  tau 51.7477 ms   5 px follows D up to 0.339 um^2/s

    So the same 5 px that just covers a fast state at 20 ms cuts off at a THIRD of it on
    the slower acquisition. A fast population like that would break its own
    tracks there; ~7.9 px is the matching radius.
    """
    if not (radius_px and radius_px > 0):
        raise TrackMateRunError(f"radius must be positive, got {radius_px!r}")
    if not (tau_s and tau_s > 0):
        raise TrackMateRunError(f"frame interval must be positive, got {tau_s!r}")
    if not (pixel_size_nm and pixel_size_nm > 0):
        raise TrackMateRunError(f"pixel size must be positive, got {pixel_size_nm!r}")
    if not (n_sigma and n_sigma > 0):
        raise TrackMateRunError(f"n_sigma must be positive, got {n_sigma!r}")
    rms_um = float(radius_px) * float(pixel_size_nm) / 1000.0 / float(n_sigma)
    return rms_um ** 2 / (4.0 * float(tau_s))


def truncation_bias(*, d_um2_s: float, radius_px: float, tau_s: float,
                    pixel_size_nm: float) -> float:
    """`fitted D / true D` for a population of diffusion `d_um2_s`. Exact.

    **The third member of a family, and the one that answers "what can I measure".** A
    step longer than the linking radius does not merely go missing: it BREAKS the
    track, so the longest steps are removed from the very distribution the fit reads.
    The fitted D is therefore biased DOWN, by an amount that is a property of the
    radius rather than of the molecule.

    For 2-D Brownian steps the step length is Rayleigh, so the truncated second moment
    is closed form. With ``U = R^2 / (4 D tau)``::

        fitted / true = [1 - (1 + U) exp(-U)] / [1 - exp(-U)]

    VERIFIED against simulation (600 000 steps per point) to **0.002** or better, at
    both frame intervals in this project and D from 0.3 to 2.0.

    It agrees with `apparent_D_ceiling` IN THE LIMIT, which is the cross-check that the
    three functions describe one thing: as D grows, ``U -> 0`` and the ratio tends to
    ``U/2``, so the fitted value approaches ``R^2 / (8 tau)``, exactly what that
    function returns. `max_trackable_D` is neither of those: it is a STEP-LOSS
    convention (the D at which the radius equals ``n_sigma`` RMS steps), which is why a
    D above it is still perfectly measurable: a D of 0.95 sits above the 0.858 that
    convention gives and reads at 93 % of the truth.
    """
    import math

    for name, value in (("d_um2_s", d_um2_s), ("radius_px", radius_px),
                        ("tau_s", tau_s), ("pixel_size_nm", pixel_size_nm)):
        if not (value and value > 0):
            raise TrackMateRunError(f"{name} must be positive, got {value!r}")
    radius_um = float(radius_px) * float(pixel_size_nm) / 1000.0
    u = radius_um ** 2 / (4.0 * float(d_um2_s) * float(tau_s))
    if u > 700:                        # exp(-u) underflows, and the bias is nil there
        return 1.0
    return (1.0 - (1.0 + u) * math.exp(-u)) / (1.0 - math.exp(-u))


def linking_corrected_D(d_read: float, *, radius_px: float, tau_s: float, pixel_size_nm: float):
    """``(true_D, kind)``: the D whose steps, cut at the linking radius, read as ``d_read`` (`truncation_bias`
    inverted, 10 Nov 2026, the per-spot audit's finding 6: the SQD likelihood is not normalised for the cut, so a
    fast population's D reads low). ``kind`` "value", or "bound" where ``d_read`` is at or past what a radius can
    read at all (within 2 % of ``R^2 / (8 tau)``, `apparent_D_ceiling`): there the true D is only "at least
    ``d_read``" and ``true_D`` is None. Exact for one Brownian population and a fixed radius: u-track's adaptive radius
    reads a slower population with less loss, so for it this is an upper estimate of the correction."""
    radius_um = float(radius_px) * float(pixel_size_nm) / 1000.0
    ceiling = radius_um ** 2 / (8.0 * float(tau_s))
    if not (d_read > 0):
        return float(d_read), "value"
    if d_read >= 0.98 * ceiling:
        return None, "bound"
    lo, hi = float(d_read), float(d_read)
    while hi * truncation_bias(d_um2_s=hi, radius_px=radius_px, tau_s=tau_s, pixel_size_nm=pixel_size_nm) < d_read:
        hi *= 2.0
        if hi > 1e6:
            return None, "bound"
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid * truncation_bias(d_um2_s=mid, radius_px=radius_px, tau_s=tau_s, pixel_size_nm=pixel_size_nm) < d_read:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), "value"


def max_measurable_D(*, radius_px: float, tau_s: float, pixel_size_nm: float,
                     tolerance: float = 0.10) -> float:
    """The fastest D that still READS within `tolerance` of the truth.

    Asked directly ("with 50 ms acquisition, whats the max D we can measure") and this
    is the honest form of the answer, because the limit is a bias rather than a wall.
    MEASURED at 106 nm, tolerance 10 %::

        radius     tau 20.47 ms     tau 51.75 ms
          5 px         0.95             0.37
          8 px         2.45             0.96
         12 px         5.50             2.16

    So a 2.5x slower frame costs about 2.5x the measurable D at a fixed radius, and at
    ~50 ms a 5 px radius tops out near **0.37**. Above the limit the number does not
    stop existing, it stops meaning what it says: it converges on `apparent_D_ceiling`
    however fast the molecule really is.

    u-TRACK is the same above its own cap. Its radius is adaptive (`brownStdMult` 3
    times each track's own displacement s.d.) but capped at `maxSearchRadius`, so below
    ``(cap/3)^2 / (2 tau)`` it picks a narrower radius and is essentially unbiased, and
    above it behaves exactly like a fixed radius of the cap. At 5 px that crossover is
    D = 0.76 at 20.47 ms and **0.30** at 51.75 ms.
    """
    if not 0 < tolerance < 1:
        raise TrackMateRunError(
            f"tolerance is a fraction between 0 and 1, got {tolerance!r}")
    lo, hi = 1e-4, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2.0
        ratio = truncation_bias(d_um2_s=mid, radius_px=radius_px, tau_s=tau_s,
                                pixel_size_nm=pixel_size_nm)
        if ratio >= 1.0 - tolerance:
            lo = mid
        else:
            hi = mid
    return lo


def gap_window(*, gap_frames: int, tau_s: float, d_um2_s: float | None = None):
    """What a frame gap costs IN TIME, rather than in frames.

    Asked for after the gap setting was carried from a 20 ms project to a 50 ms one:
    "we should add this as a hint or calculate the gap in our trackmate interface for
    different ms".

    A gap of ``n`` frames lets a molecule vanish for ``n * tau`` and then be linked to
    whatever reappears, so the SAME number of frames allows very different amounts of
    motion. Gap 3 is **61 ms** at 20.47 ms and **155 ms** at 51.75 ms, over which a
    molecule at D = 0.3 covers 271 nm against 432 nm. The frame count is not portable
    between acquisitions; the time is what to hold fixed.

    ``d_um2_s`` has NO DEFAULT and the travel is omitted without one, how far a
    molecule goes while unseen depends on how fast it is, and inventing a D here would
    print a distance that is a property of this function rather than of the experiment.
    """
    import math

    if gap_frames < 0:
        raise TrackMateRunError(f"gap_frames must be >= 0, got {gap_frames!r}")
    if not (tau_s and tau_s > 0):
        raise TrackMateRunError(f"frame interval must be positive, got {tau_s!r}")
    unobserved_s = float(gap_frames) * float(tau_s)
    out = {"gap_frames": int(gap_frames), "tau_s": float(tau_s),
           "unobserved_ms": unobserved_s * 1000.0, "travel_nm": None}
    if d_um2_s is not None:
        if not d_um2_s > 0:
            raise TrackMateRunError(f"d_um2_s must be positive, got {d_um2_s!r}")
        out["travel_nm"] = math.sqrt(4.0 * float(d_um2_s) * unobserved_s) * 1000.0
    return out


def apparent_D_ceiling(*, radius_px: float, tau_s: float,
                       pixel_size_nm: float) -> float:
    """The largest apparent D a linking radius can EVER return: `a^2 / (8*tau)`.

    A different quantity from `max_trackable_D`, and the two must not be confused.
    `max_trackable_D` answers "what D can I follow with an acceptable loss of steps"
    (0.858 at 5 px / 20.4717 ms, using `n_sigma` = 2). This answers "what is the hardest
    limit on the number that comes out", and it is exactly **2x** larger, **1.715**,
    because a radius of `n` RMS steps gives `a^2/(4 n^2 tau)`, and the ceiling is the
    `n = sqrt(2)` case. Quote each WITH its definition; this project has already recorded
    what happens when two counts computed differently are compared.

    Why it saturates rather than merely truncating. A step longer than the radius cannot
    be linked at all, so every retained step is shorter than `a`. As the true D grows the
    retained steps approach uniform-in-the-disc, whose mean square is `a^2/2`, so the
    apparent `D = <r^2>/(4 tau)` climbs to `a^2/(8 tau)` and stops. VERIFIED by simulation
    at 5, 7, 8, 10 and 12 px, agreeing with the closed form to under 1 %.

    MEASURED approach at 5 px, tau 20.4717 ms:

        true D   0.50   0.86   2.00   6.00   100
        lost     0.1 %  1.8 %  18 %   57 %   97 %
        apparent 0.50   0.80   1.25   1.55   1.71

    So a true D of 6 is recorded as ~1.55 and a true D of 100 as ~1.71: **the measurement
    stops distinguishing them.** And the real case is worse, because an overshoot does not
    drop one step: it breaks the track, and the fragments then meet the minimum-length
    filter and are deleted. Fast molecules do not merely read slow; they leave the
    dataset.

    u-track and trackmate share this wall, since u-track's adaptive radius only ever
    moves BELOW its 5 px cap. So the two tools agreeing on D_fast is not confirmation of
    D_fast: it is the shared-bias case this project refuses everywhere else. Settle it by
    sweeping the radius (`scripts/radius_sweep.py`), never by the agreement.
    """
    if not (radius_px and radius_px > 0):
        raise TrackMateRunError(f"radius must be positive, got {radius_px!r}")
    if not (tau_s and tau_s > 0):
        raise TrackMateRunError(f"frame interval must be positive, got {tau_s!r}")
    if not (pixel_size_nm and pixel_size_nm > 0):
        raise TrackMateRunError(f"pixel size must be positive, got {pixel_size_nm!r}")
    a_um = float(radius_px) * float(pixel_size_nm) / 1000.0
    return a_um ** 2 / (8.0 * float(tau_s))


def step_length_summary(table, *, radius_used_px: float | None = None) -> dict:
    """The frame-to-frame step-length distribution, straight off a tracked result.

    The answer to "D is not known yet". It is not needed: D is fitted from these step
    lengths, so the steps are one level less derived than D is. A pilot run gives them,
    and a percentile of them is the linking radius: no diffusion model in between.

    But the distribution is truncated by the radius that produced it, which is the
    trap, because the number being chosen is the number that shaped the data it is being
    chosen from. No step longer than the pilot's radius can appear, at any percentile.
    MEASURED on 20 u-track movies at its 5 px cap: max step **4.994**, with **1.95 %** of
    steps in the top tenth of the range and **0.12 %** in the top hundredth. So the
    truncation removes a modest tail rather than dominating: percentiles up to about p99
    are usable, the extreme tail is not measurable this way, and `censored_fraction` is
    returned so the caller can see which case they are in. Run the pilot DELIBERATELY
    GENEROUS, or this re-measures the previous setting.

    Only steps one frame apart are used: a bridged gap is two frames of travel and would
    inflate the radius.
    """
    import numpy as np

    needed = {"track_id", "frame", "x", "y"}
    missing = needed - set(table.columns)
    if missing:
        raise TrackMateRunError(
            f"step_length_summary needs {sorted(needed)}; missing {sorted(missing)}")
    key = ["movie", "track_id"] if "movie" in table.columns else ["track_id"]
    frame = table.sort_values(key + ["frame"])
    grouped = frame.groupby(key, sort=False)
    dx, dy = grouped["x"].diff(), grouped["y"].diff()
    lag = grouped["frame"].diff()
    keep = (lag == 1) & dx.notna()
    step = np.hypot(dx[keep].to_numpy(), dy[keep].to_numpy())
    if step.size == 0:
        raise TrackMateRunError(
            "no consecutive-frame steps in this table, so no step-length distribution "
            "can be measured: a linking radius cannot be read off single localisations")
    qs = {f"p{q}": float(np.percentile(step, q)) for q in (50, 90, 95, 99, 99.9)}
    out = {"n_steps": int(step.size), "median": qs["p50"], "max": float(step.max()),
           **qs}
    if radius_used_px:
        radius = float(radius_used_px)
        out["radius_used_px"] = radius
        # How much of the distribution sits against the ceiling. High means the pilot
        # radius, not the molecules, decided the answer.
        out["censored_fraction"] = float(np.mean(step > 0.9 * radius))
        out["at_the_cap_fraction"] = float(np.mean(step > 0.99 * radius))
    return out


def linking_ambiguity(table, *, radius_px: float, meshes=None) -> dict:
    """How often a second detection sits within one linking radius, per frame.

    THE CEILING, where the step distribution gives the floor. A radius wide enough to
    reach the next molecule makes the link a guess rather than a measurement, and unlike
    the floor this is measurable BEFORE any tracking: it needs detections, not tracks.

    A poisson model over the whole field understates it about fivefold, which is why this
    counts rather than computes. Molecules sit in cells covering ~4.5 % of the field, so
    the density that matters is the in-cell one. Measured at 5 px on three lab conditions: **5.4 %**,
    4.1 % and 2.5 %, against a whole-field Poisson estimate of
    ~0.95 %.
    """
    import numpy as np

    # IN-CELL WHEN THE OUTLINES ARE THERE, and it matters more than it looks. Molecules
    # sit in cells covering about 4.5 per cent of the field, so detections OUTSIDE them
    # are scattered, have no neighbours, and DILUTE the count. Measured on the lab's data drive that is
    # negligible in 5 groups of 6 -- and tenfold on one lab condition, which reads 36.6 per
    # cent whole-field against 3.3 in-cell because ONE movie carries a dense pile of
    # out-of-cell signal. Without meshes there is no way to tell those two cases apart,
    # which is why this reports WHICH it measured rather than quietly doing one.
    scope = "whole field"
    if meshes is not None:
        from .cellmesh import points_in_polygon

        by_movie = meshes if isinstance(meshes, dict) else None
        inside = np.zeros(len(table), dtype=bool)
        x = table["x"].to_numpy(dtype=float)
        y = table["y"].to_numpy(dtype=float)
        movies = (table["movie"].to_numpy() if "movie" in table.columns
                  else np.full(len(table), None))
        pool = []
        if by_movie is None:
            pool = [(None, list(meshes))]
        else:
            pool = [(name, list(group)) for name, group in by_movie.items()]
        for name, group in pool:
            sel = slice(None) if name is None else (movies == name)
            if name is not None and not np.any(sel):
                continue
            for mesh in group:
                polygon = getattr(mesh, "polygon", mesh)
                hit = points_in_polygon(x[sel], y[sel], polygon)
                if name is None:
                    inside |= hit
                else:
                    idx = np.flatnonzero(sel)
                    inside[idx[hit]] = True
        if not inside.any():
            raise TrackMateRunError(
                "no detection falls inside any outline, so an in-cell ambiguity cannot "
                "be measured: check that the outlines belong to these movies (a "
                "TrackMate movie is named by its file, outlines by the movie)")
        table = table[inside]
        scope = "in cell"

    key = ["movie", "frame"] if "movie" in table.columns else ["frame"]
    radius = float(radius_px)
    if radius <= 0:
        raise TrackMateRunError(f"radius must be positive, got {radius_px!r}")
    hits = total = 0
    for _label, sub in table.groupby(key, sort=False):
        xy = sub[["x", "y"]].to_numpy(dtype=float)
        total += len(xy)
        if len(xy) < 2:
            continue
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        hits += int((d.min(axis=1) < radius).sum())
    if not total:
        raise TrackMateRunError("no detections, so ambiguity cannot be measured")
    return {"radius_px": radius, "n_detections": int(total),
            "fraction_with_neighbour": hits / total,
            "n_with_neighbour": int(hits),
            # NAMED, because the two differ tenfold on one real condition and a number
            # without its scope would be read as the other one.
            "scope": scope}


#: **REFERENCE POINTS FROM ONE DATASET. NOT A RULE, AND NOTHING READS THIS.** Each entry
#: is `true D_fast (um^2/s) -> share of steps beyond 5 px`, measured by tracking
#: simulated movies at 8 px through a fixed-radius LAP linker, with the cells PLACED
#: APART in a 126 x 130 px field, 24.2 per cent of localisations missed (this project's
#: own recall ceiling) and gap closing 3. The simulated crowding was matched to the real
#: data first: 2.3 per cent of localisations with a neighbour inside 5 px, against a real
#: median of 2.0 per cent over 26 MSD groups.
#:
#: **IT SATURATES, so it is a floor on D and never a reading of it.** Above about D = 5
#: the cell wall rather than the molecule decides the longest step a track can show, so
#: 8 and 12 read barely above 5. The curve separates "the radius covers this population"
#: from "the radius is cutting it off"; it does not return a D.
#:
#: **IT IS KEPT ONLY AS A CROSS-CHECK ON THE ANALYTIC INVERSION `widening_yield` USES,
#: and the function does not consult it.** A stored table maps a yield onto a D at ONE
#: frame interval, ONE cell size and ONE crowding; applied to a 51.75 ms acquisition it
#: would be wrong by the ratio of the two taus, which is this project's most-repeated
#: defect -- a number measured in one regime applied to another. So the verdict is
#: computed from the radius and tau of the project in hand instead.
#:
#: The narrow/wide pair, the frame interval, the pixel size and the cell are all part of
#: these numbers. Quote them together or not at all.
WIDENING_YIELD_CALIBRATION = {
    "narrow_radius_px": 5.0,
    "wide_radius_px": 8.0,
    "tau_ms": 20.47,
    "pixel_size_nm": 106.0,
    "cell_um": (3.0, 0.9),
    "slow_fraction": 0.35,
    "by_true_D_fast": {0.85: 0.015, 2.0: 0.076, 3.0: 0.113,
                       5.0: 0.172, 8.0: 0.193, 12.0: 0.200},
}

#: The three verdicts. Which one applies is DERIVED per project (see `widening_yield`),
#: never looked up: the yield is inverted into the diffusion coefficient that would
#: produce it at THIS radius and THIS frame interval, and that is read against what the
#: radius can measure -- `max_measurable_D` and `apparent_D_ceiling`, both already here.
WIDENING_VERDICTS = {
    "covered": "the narrow radius measures this population to within 10 per cent",
    "partial": "above what the narrow radius measures well, below the D at which a fit "
               "returns the setting rather than the molecule",
    "truncated": "the implied D exceeds the narrow radius's own ceiling, so that radius "
                 "reports its setting rather than the molecules",
    # A FOURTH STATE, and not a failure. Too few steps to separate the three above is a
    # statement about the sample, and reporting one of them anyway would be the confident
    # wrong answer a real run produced on twelve steps.
    "inconclusive": "too few steps to separate the verdicts above",
}


def implied_D_from_yield(*, beyond_fraction: float, radius_px: float, tau_s: float,
                         pixel_size_nm: float) -> float:
    """The diffusion coefficient a step population would need to leave this many steps
    beyond ``radius_px``. A FLOOR, for two reasons, both named below.

    NO CALIBRATION. For 2-D Brownian motion the step length is Rayleigh, so the share
    exceeding R is ``exp(-R^2 / (4 D tau))`` and the inversion is exact:

        D = R^2 / (4 tau ln(1 / beyond))

    Everything in it is a property of the acquisition in hand -- the radius, the frame
    interval, the pixel size -- so it transfers between projects where a stored table
    does not. VERIFIED against the planted simulation: a true fast D of 0.85 gives a
    yield of 1.5 per cent, which inverts to **0.82**, within 4 per cent of the truth.

    **IT IS A FLOOR AND NOT AN ESTIMATE**, and the gap grows with D:

      * the yield is taken over ALL steps, and the slow population contributes almost
        none of the long ones, so a mixture reads lower than its fast component alone;
      * the cell CONFINES, so the longest steps a track can show are capped by the cell
        rather than by D -- which is why the same simulation gives 1.3 for a planted 2.0
        and 1.9 for a planted 5.0.

    Both push the answer DOWN, so a verdict of "truncated" reached through this number is
    conservative: the true D is at least this large.
    """
    import numpy as np

    beyond = float(beyond_fraction)
    if not 0.0 < beyond < 1.0:
        # 0 is "no step exceeded the radius", which implies no lower bound on D at all,
        # and 1 is impossible. Neither is an error; neither is a number.
        return 0.0 if beyond <= 0.0 else float("inf")
    radius_um = float(radius_px) * float(pixel_size_nm) / 1000.0
    return radius_um ** 2 / (4.0 * float(tau_s) * np.log(1.0 / beyond))


def _wilson(proportion: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """A Wilson score interval, which stays inside [0, 1] where the normal one does not.

    The `covered` verdict is exactly the case of a proportion near zero, and that is
    precisely where a normal interval runs negative and reports a width it does not have.
    """
    import numpy as np

    if n <= 0:
        return 0.0, 1.0
    p, z2 = float(proportion), z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = (z / (1 + z2 / n)) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def widening_yield(narrow, wide, *, narrow_radius_px: float, wide_radius_px: float,
                   tau_s: float, pixel_size_nm: float,
                   min_localizations: int = 5) -> dict:
    """What share of steps the narrow radius could not see, from two tracking runs.

    THE ONE STATISTIC HERE THAT THE CENSORING CANNOT MANUFACTURE. Every attempt to read
    the fast state out of a 5 px run fits a distribution that was cut off at 5 px, so the
    fit reports the setting: the radius sweep found `D_fast / radius` dead constant at
    0.29 above 8 px. Tracking the SAME movies at a wider radius and COUNTING what appears
    beyond the old one is not a fit. It needs no model, no cell outlines and no ground
    truth, and it costs one extra pass over a couple of movies.

    MEASURED ON THE TWELVE SWEEP GROUPS (12 px against 5 px, same movies, same detection
    threshold, same gap policy): **9.8 to 22.6 per cent, median 16.3**. Every group is
    truncating at 5 px -- the narrowest still hides a tenth of its steps, and
    one of them nearly doubles its step count when the radius opens.

    **THIS IS A DISCLOSURE, NOT AN INSTRUCTION TO WIDEN.** A wider radius buys those
    steps and pays in wrong links, and the fractions -- which is what this project's
    claims rest on -- were validated at 5 px to +1.4 pp against the deposited references
    while 12 px inflated them by 12.8 pp in 11 of 12 groups. Report the yield beside a
    fast D as the measure of how far that D is from identified; do not widen because it
    is large.

    Only steps ONE FRAME APART are counted, in both runs. A bridged gap is two or more
    frames of travel, so it lands beyond the narrow radius for a reason that is not the
    molecule's speed: measured on a lab project the yield is 17.5 per cent on adjacent steps
    against 18.9 per cent when gap-crossing steps are let in.

    Parameters
    ----------
    narrow, wide
        Raw tables (``movie, track_id, frame, x, y`` in PIXELS) from tracking the SAME
        movies at the two radii.
    narrow_radius_px, wide_radius_px
        The radii those runs used. ``wide`` must exceed ``narrow``.
    min_localizations
        Tracks shorter than this are dropped from BOTH runs first, matching the canonical
        filter. Applied to both, or the two runs would not describe the same population.

    Returns
    -------
    dict
        Step counts at each radius, ``extra_fraction`` (what the wider run gained),
        ``beyond_narrow`` (the yield), a ``profile`` at several thresholds, and a
        ``band``/``verdict`` DERIVED from this project's own radius and tau
        (see :func:`implied_D_from_yield`), plus the ``implied_D_floor``
        and the two limits it was judged against.
    """
    import numpy as np

    narrow_r, wide_r = float(narrow_radius_px), float(wide_radius_px)
    if not wide_r > narrow_r:
        raise TrackMateRunError(
            f"the wide radius must exceed the narrow one, got wide {wide_r} against "
            f"narrow {narrow_r} - with no widening there is no yield to measure")

    def _steps(table, label):
        needed = {"track_id", "frame", "x", "y"}
        missing = needed - set(table.columns)
        if missing:
            raise TrackMateRunError(
                f"widening_yield needs {sorted(needed)} in the {label} table; "
                f"missing {sorted(missing)}")
        key = ["movie", "track_id"] if "movie" in table.columns else ["track_id"]
        frame = table.sort_values(key + ["frame"])
        if min_localizations > 1:
            size = frame.groupby(key, sort=False)["frame"].transform("size")
            frame = frame[size >= int(min_localizations)]
        if frame.empty:
            raise TrackMateRunError(
                f"no track in the {label} table reaches {min_localizations} "
                f"localisations, so it has no steps to compare")
        grouped = frame.groupby(key, sort=False)
        dx, dy = grouped["x"].diff(), grouped["y"].diff()
        lag = grouped["frame"].diff()
        keep = (lag == 1) & dx.notna()
        step = np.hypot(dx[keep].to_numpy(), dy[keep].to_numpy())
        if step.size == 0:
            raise TrackMateRunError(
                f"no consecutive-frame steps in the {label} table - a widening yield "
                f"cannot be measured from single localisations or bridged gaps alone")
        movies = (set(map(str, frame["movie"].unique()))
                  if "movie" in frame.columns else set())
        return step, movies

    step_narrow, movies_narrow = _steps(narrow, "narrow")
    step_wide, movies_wide = _steps(wide, "wide")

    # THE COMPARISON IS BETWEEN RADII, so it must not also be between movie sets. Two
    # runs over different movies differ for a reason this statistic cannot see, and that
    # difference would be read as truncation.
    if movies_narrow and movies_wide and movies_narrow != movies_wide:
        only_narrow = sorted(movies_narrow - movies_wide)
        only_wide = sorted(movies_wide - movies_narrow)
        raise TrackMateRunError(
            "the two runs cover different movies, so what differs between them is not "
            "the linking radius: only in narrow "
            f"{only_narrow if only_narrow else 'none'}, only in wide "
            f"{only_wide if only_wide else 'none'}. Track the same movies at both "
            "radii, or subset them before calling.")

    beyond = float(np.mean(step_wide > narrow_r))

    # THE VERDICT IS DERIVED HERE, FROM THIS PROJECT'S OWN NUMBERS. A stored band would
    # carry one dataset's frame interval onto every other: at 51.75 ms the same 5 px
    # covers D <= 0.34 where at 20.47 ms it covers 0.86, so one yield means two different
    # things and a lookup table cannot know which.
    implied = implied_D_from_yield(beyond_fraction=beyond, radius_px=narrow_r,
                                   tau_s=tau_s, pixel_size_nm=pixel_size_nm)
    measurable = max_measurable_D(radius_px=narrow_r, tau_s=tau_s,
                                  pixel_size_nm=pixel_size_nm)
    ceiling = apparent_D_ceiling(radius_px=narrow_r, tau_s=tau_s,
                                 pixel_size_nm=pixel_size_nm)

    # THE YIELD IS A PROPORTION, SO IT CARRIES AN INTERVAL, AND THE VERDICT DEFERS TO IT.
    # FOUND BY RUNNING IT: one real movie of a sparse condition gave TWELVE steps and a
    # confident "covered", where the twelve deposited groups rest on 2524 to 13430. A
    # verdict is only as good as the count under it, and a hard minimum would be a
    # constant somebody invented -- this asks the data instead. Wilson, because the
    # normal interval is useless at a proportion near zero, which is exactly the
    # "covered" case.
    lo, hi = _wilson(beyond, len(step_wide))
    implied_lo = implied_D_from_yield(beyond_fraction=lo, radius_px=narrow_r,
                                      tau_s=tau_s, pixel_size_nm=pixel_size_nm)
    implied_hi = implied_D_from_yield(beyond_fraction=hi, radius_px=narrow_r,
                                      tau_s=tau_s, pixel_size_nm=pixel_size_nm)

    def _band_of(value):
        if value <= measurable:
            return "covered"
        return "partial" if value <= ceiling else "truncated"

    band = _band_of(implied)
    # A point estimate that lands in one band while its interval spans two has not
    # decided anything, and saying so is the whole reason the interval is computed.
    if _band_of(implied_lo) != _band_of(implied_hi):
        band, verdict = "inconclusive", (
            f"{len(step_wide)} steps put the implied D between {implied_lo:.2f} and "
            f"{implied_hi:.2f} um2/s, which spans more than one verdict: track more "
            f"movies")
    else:
        verdict = WIDENING_VERDICTS[band]

    thresholds = sorted({narrow_r, (narrow_r + wide_r) / 2.0, wide_r})
    return {
        "narrow_radius_px": narrow_r,
        "wide_radius_px": wide_r,
        "tau_s": float(tau_s),
        "pixel_size_nm": float(pixel_size_nm),
        "implied_D_floor": implied,
        "beyond_narrow_interval": (lo, hi),
        "implied_D_interval": (implied_lo, implied_hi),
        "max_measurable_D": measurable,
        "apparent_D_ceiling": ceiling,
        "min_localizations": int(min_localizations),
        "n_steps_narrow": int(step_narrow.size),
        "n_steps_wide": int(step_wide.size),
        "extra_fraction": float(step_wide.size / step_narrow.size - 1.0),
        "beyond_narrow": beyond,
        "profile": {f"beyond_{t:g}px": float(np.mean(step_wide > t))
                    for t in thresholds},
        "movies": sorted(movies_wide) if movies_wide else [],
        "band": band,
        "verdict": verdict,
    }


#: **WHAT u-track ACTUALLY DID ABOUT THE LINKING RADIUS, read from `movieData.mat`
#: across 400 MSD movies rather than remembered.** Asked directly, "linking radius
#: derived from D, while D is not known yet. what linking radius did we use in utrack?"
#:, and the answer is that u-track never picked one:
#:
#:     minSearchRadius   0 px      (2 px in 10 of 400 movies)
#:     maxSearchRadius   5 px      uniform across all 400
#:     brownStdMult      3.0       uniform
#:     useLocalDensity   1         uniform
#:
#: The radius used on any given link is `brownStdMult` x the Kalman-estimated
#: PER-COORDINATE displacement s.d. of that track, clipped into the bracket and shrunk
#: further where the field is crowded. **So 5 px is a CAP, not a setting**, and this
#: file's earlier "u-track uses 5 px search radius" was the ceiling reported as the
#: value. That is how u-track escapes the circularity: it estimates each track's own
#: displacement scale as it goes instead of being told a D.
#:
#: TrackMate's LAP tracker has NO adaptive radius: one number is applied to every link
#:, so the choice cannot be dodged the same way. The cap is the right number to take,
#: and 3 per-coordinate s.d. is 3/sqrt(2) = 2.12 RMS steps, within 6 % of the 2x this
#: module's helper defaults to.
UTRACK_LINKING_POLICY = {
    "min_search_radius_px": 0.0,
    "max_search_radius_px": 5.0,
    "brown_std_mult": 3.0,
    "use_local_density": True,
    "adaptive": True,
    "n_movies_read": 400,
    "note": ("u-track sets the radius per track from its own Kalman-estimated "
             "displacement s.d.; 5 px is the cap, not the radius used"),
}


#: **THE FRAME GAP IS NOT DERIVABLE AND IS NOT PRETENDED TO BE.** It answers "how many
#: frames may a molecule vanish for and still be the same molecule?", which is a property
#: of the DETECTOR's reliability, not of the optics or the physics. Nothing in a movie's
#: header implies it.
#:
#: What it has instead is a MEASUREMENT, recorded here rather than in a comment nobody
#: reads: u-track needs none (`timeWindow = 1`) because its mixture-model fit re-finds a
#: molecule reliably; TrackMate's LoG flickers near threshold, and at gap 0 its median
#: raw track length is 2 against u-track's 6. At gap 3 the counts converge and the median
#: reaches 7. So 3 is the measured value for THIS pairing, on a lab project, 8 movies per condition.
#:
#: A bridged gap is safe downstream because `observables.squared_displacements` pairs by
#: FRAME DIFFERENCE, so the gap is excluded rather than counted as one long step.
FRAME_GAP_NOTE = (
    "The maximum frame gap depends on the detector and cannot be derived from the movie. "
    "Measured against u-track: at 0 the median raw TrackMate track length is 2 "
    "(u-track 6); at 3 the counts converge and the median reaches 7. Confirm it on "
    "your own data by comparing track-length distributions."
)

#: **THE VALIDATED LINKING RADIUS, AND IT IS A SEPARATE CLAIM FROM u-track's CAP.**
#:
#: The two numbers coincide at 5 px and that is not a coincidence: u-track's cap was
#: matched to this data's fast state, but they are different assertions and must not
#: share a constant. `UTRACK_EQUIVALENT` says what u-track did. This says what was
#: measured to be right, and it would still be 5 if u-track's cap had been 7.
#:
#: MEASURED before being hard-wired, over every group with a deposited reference,
#: 6 proteins x 2 carbon sources, 6 movies each, detector adaptive per condition and
#: frozen across radii (`msd_out/radius_sweep/validation_vs_deposited.csv`):
#:
#:              mean error   mean |error|   within 5 pp   corr. with deposited
#:     5 px       +1.4 pp       4.0 pp         9 / 12            +0.820
#:    12 px      +12.8 pp      12.8 pp         1 / 12            +0.639
#:
#: 5 px is nearly UNBIASED; 12 px is inflated by 13 pp and positive in 11 of 12 groups.
#: The two runs differ in nothing but the radius, so that comparison is clean. The
#: condition difference keeps its sign at 5 px in all but one protein, and that one has
#: a deposited difference too small to carry a reliable sign either way.
#:
#: **THE RESIDUAL DISAGREEMENTS TRACK SAMPLE SIZE, NOT THE RADIUS**: mean |error| is
#: 5.5 pp below 650 tracks against 2.5 pp above, and the worst group (+11.6 pp)
#: has the fewest tracks of the twelve.
#:
#: **WHAT IS NOT HARD-WIRED: ANY FAST-STATE D.** Sweeping the radius moves it by a factor
#: of six and it never plateaus (`D_fast / radius` constant at 0.29 above 8 px), so it is
#: not identifiable at tau = 20.47 ms at ANY radius. Widening buys a D that cannot be
#: trusted and destroys the fraction comparison that can. See `docs/linking_radius.md`.
LINKING_RADIUS_PX = 5.0

LINKING_RADIUS_NOTE = (
    "5 px was validated against a reference analysis over 6 proteins x 2 carbon "
    "sources: mean error +1.4 pp, against +13 pp at 12 px. It matches u-track's cap. "
    "The validation holds for 106 nm pixels and 20.47 ms frames, and no fast-state D "
    "is trustworthy at any radius. The threshold is per-dataset and not part of this "
    "claim: it is measured from each condition's own noise."
)

#: The measured starting point for a TrackMate run on this kind of data. The THRESHOLD is
#: absent on purpose: it is the one number that does not transfer between datasets, and
#: the window computes it from the movies' own noise.
ADAPTIVE_BASE = {
    "radius_px": 3.0,
    # THE VALIDATED CONSTANT, not `UTRACK_EQUIVALENT`. They agree today; conflating them
    # would silently re-derive this project's default from what another tracker happened
    # to cap at.
    "linking_max_distance_px": LINKING_RADIUS_PX,
    "max_frame_gap": 3,
}


def preset_choices():
    """What the window's Preset box offers: two ways of choosing, then saved sets.

    Condition names are not presets. This box used to offer "Glucose" and "Malate",
    one lab's conditions, and meaningless to anybody else opening the tool. Those two
    thresholds are real measurements and `UTRACK_COMPARABLE` keeps them; what they are
    not is a general setting. See `smtracker.presets`.
    """
    from .presets import BUILT_IN, list_presets

    return tuple(BUILT_IN) + tuple(p.name for p in list_presets())


def settings_from_preset(name, **overrides):
    """A preset by condition name, or ``None`` for the free choice.

    PURE, so the tracking window computes nothing: it reads the numbers back out of
    the widgets and calls this. ``overrides`` lets a user move one number off a
    preset without losing the rest.
    """
    if name not in UTRACK_COMPARABLE:
        if name == 'Custom':
            return None
        raise TrackMateRunError(
            f'no measured preset for {name!r}; have '
            + ', '.join(sorted(UTRACK_COMPARABLE)) + ' and Custom')
    values = dict(UTRACK_COMPARABLE[name])
    values.update(overrides)
    return TrackMateSettings(**values)


#: The analysis needs at least this many localisations in a track. u-track enforces it as
#: `minTrackLen`; TrackMate does not, so a batch can look successful and still leave
#: almost nothing the analysis can use. That number is the one worth printing.
MIN_USABLE_LOCALISATIONS = 5


def track_counts(raw) -> tuple[int, int]:
    """``(tracks, usable)`` in a read-back raw table: every track, and those of at least
    `MIN_USABLE_LOCALISATIONS` localisations -- the ones the canonical stage keeps. One
    source for the run summary and for the run's outcome record."""
    if raw is None or not len(raw):
        return 0, 0
    key = ["movie", "track_id"] if "movie" in raw.columns else ["track_id"]
    lengths = raw.groupby(key, observed=True).size()
    return int(len(lengths)), int((lengths >= MIN_USABLE_LOCALISATIONS).sum())


def summarise_run(raw, settings, *, table=None) -> str:
    """What a finished tracking run actually produced, in the terms that matter.

    Spots and tracks are not the result. A run reporting "1738 tracks" sounds
    successful while leaving 457 the analysis can use, because TrackMate does not enforce
    a minimum track length and the canonical stage discards anything under
    `MIN_USABLE_LOCALISATIONS`. So the usable count is printed FIRST and the raw count
    beside it.

    The step distribution is printed because it is the only on-screen check that the
    linking radius was not the binding constraint: steps pressed against the radius mean
    real motion was cut, and this project has measured what that costs.

    PURE. The window prints this; nothing here touches Qt, so the same text appears in a
    script and in a test.
    """
    import numpy as np

    lines = ["Tracking done."]
    if table is not None and len(table):
        ok = table[table["ok"]] if "ok" in table else table
        failed = len(table) - len(ok)
        lines.append(f"  movies: {len(ok)} tracked"
                     + (f", {failed} failed (named above)" if failed else ""))
        if "n_spots" in ok and len(ok):
            lines.append(f"  spots detected: {int(ok['n_spots'].sum())}")

    if raw is None or not len(raw):
        lines.append("  No localisations were read back: the XMLs exist but hold nothing.")
        return chr(10).join(lines)

    key = ["movie", "track_id"] if "movie" in raw.columns else ["track_id"]
    lengths = raw.groupby(key, observed=True).size()
    usable = track_counts(raw)[1]
    lines.append(f"  tracks: {len(lengths)} total, "
                 f"{usable} usable (at least {MIN_USABLE_LOCALISATIONS} localisations, "
                 f"{100 * usable / len(lengths):.0f} %)")
    lines.append(f"  track length: median {int(lengths.median())}, "
                 f"max {int(lengths.max())}")
    if usable < 0.25 * len(lengths):
        lines.append(
            "  ! Most tracks are too short to use. TrackMate's LoG flickers near "
            "threshold, so the detection threshold is usually too strict or "
            "the frame gap too small; measured against u-track, gap 3 reaches median 7.")

    frame = raw.sort_values(key + ["frame"])
    grouped = frame.groupby(key, observed=True)
    dx, dy = grouped["x"].diff(), grouped["y"].diff()
    lag = grouped["frame"].diff()
    keep = (lag == 1) & dx.notna()
    step = np.hypot(dx[keep].to_numpy(), dy[keep].to_numpy())
    if step.size:
        radius = float(settings.linking_max_distance_px)
        against = float(np.mean(step > 0.9 * radius))
        lines.append(f"  steps: {step.size}, median {np.median(step):.2f} px, "
                     f"p99 {np.percentile(step, 99):.2f}, max {step.max():.2f} "
                     f"(radius {radius:g})")
        lines.append(
            f"  {against * 100:.1f} % of steps sit in the top tenth of the radius"
            + ("  ! the radius is cutting real motion; widen it only if you are "
               "reporting D, never for fractions" if against > 0.05 else ""))
    lines.append("")
    lines.append(f"  settings: {settings.stamp()}")
    return chr(10).join(lines)


#: What a run writes beside its XMLs so the run can be repeated.
RUN_RECORD_FILENAME = "run.json"


def record_run(output_dir, *, settings, movies, crops=None, pixel_size_nm=None,
               frame_interval_s=None, source=None, threshold_provenance=None,
               per_movie_thresholds=None,
               outcome=None, at_once: int = 1):
    """Write what a tracking run was GIVEN, beside what it produced.

    The XMLs do not make a run reproducible, and a real run proved it. The
    experimenter tracked a 4500-frame acquisition and the XML recorded `nframes = 2588`,
    but not WHICH 2588, because the crop range lived in the window's memory and was never
    saved. Nothing on disk said which frames were tracked, at what pixel size, or at what
    frame interval. The tracks were fine; the run could not be repeated.

    The three things the XML cannot carry, and the reason each matters:

    * **the crop range**, per movie. A raw acquisition still holds the dense early frames
      the crop exists to remove, so "2588 frames" describes a different dataset depending
      on where it started.
    * **the pixel size**, because the generated script FORCES the calibration to
      1 px = 1 unit (so a micron-calibrated TIFF cannot reinterpret every radius). That
      is right for tracking and it means the XML's `pixelwidth` is 1.0, not a record.
    * **the frame interval**, which the XML does happen to carry correctly but which
      every D scales with: this project has corrected 20.0-vs-20.472 twice, and found a
      folder named "20 ms" whose acquisition measures 21.7387.

    What is missing is named, not omitted. A key present and null reads as "not recorded
    at run time"; a key absent reads as "this version did not track that", and six months
    on nobody can tell those apart.

    PURE. Takes values and writes a file; the window supplies what is on screen.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "smtracker.trackmate_run",
        "source": source,
        "settings": {
            "radius_px": settings.radius_px,
            # THE MEDIAN WHERE THEY DIFFER, and `threshold_applied` is what stops it
            # being read as the value every movie got. It is not a new statistic:
            # `batch_threshold` already returns exactly this median. A null here would
            # break every reader that expects a number, and would say "not set" where
            # the truth is "varies" -- two different things.
            "threshold": settings.threshold,
            "threshold_provenance": threshold_provenance,
            "threshold_applied": ("per_movie" if per_movie_thresholds
                                  else "one_for_the_batch"),
            "per_movie_thresholds": (
                {str(k): float(v) for k, v in sorted(per_movie_thresholds.items())}
                if per_movie_thresholds else None),
            "linking_max_distance_px": settings.linking_max_distance_px,
            "gap_closing_max_distance_px": settings.gap_closing_max_distance_px,
            "max_frame_gap": settings.max_frame_gap,
            "allow_splitting": settings.allow_splitting,
            "allow_merging": settings.allow_merging,
            "subpixel": settings.subpixel,
            "median_filter": settings.median_filter,
            "stamp": settings.stamp(),
        },
        # null rather than absent, see the docstring. Tracking itself works in pixels
        # and frames, so a run CAN complete without these; the record must then say they
        # were not set rather than imply they were never a concept.
        "calibration": {
            "pixel_size_nm": pixel_size_nm,
            "frame_interval_s": frame_interval_s,
            "note": ("null means not set at run time. Tracking does not need them: it "
                     "works in pixels and frames, but every D downstream scales with "
                     "the frame interval, and the XML cannot carry the pixel size "
                     "because the script forces 1 px = 1 unit."),
        },
        "movies": [str(Path(m).name) for m in movies],
        "crops": ({str(k): list(v) for k, v in crops.items()} if crops else None),
        "crops_note": ("1-based inclusive frame ranges, one per movie. null means the "
                       "movies were tracked whole: correct for a prepared stack, which "
                       "was already cropped, and impossible for a raw acquisition, "
                       "which run_batch refuses without a range."),
        "outcome": outcome,
        # Fijis run side by side (`run_batch`'s ``at_once``); the tracks do not depend on it
        "movies_at_once": int(at_once or 1),
    }
    path = output_dir / RUN_RECORD_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def describe_batch(qc):
    """The batch verdict as one block of text, so the window and a script agree.

    Names the failures FIRST. A batch table that lists only what succeeded reads as
    'these are the movies', which is the shape this module refuses everywhere else.
    """
    n_failed = int(qc.attrs.get('n_failed', 0))
    failed = list(qc.attrs.get('failed', []))
    lines = []
    if n_failed:
        lines.append(f'{n_failed} movie(s) failed and produced nothing: '
                     + ', '.join(failed[:6])
                     + (' ...' if len(failed) > 6 else ''))
    else:
        lines.append('no movie failed')
    flagged = qc[qc['flagged']] if 'flagged' in qc else qc.iloc[:0]
    lines.append(f'{len(qc)} movie(s) tracked, {len(flagged)} flagged as unlike their neighbours')
    for _, row in flagged.iterrows():
        lines.append(f"    {row['movie']}: {int(row['n_spots'])} spots, "
                     f"{int(row['n_tracks'])} tracks  [{row['flagged_on']}]")
    lines.append(str(qc.attrs.get('caveat', '')))
    return chr(10).join(lines)


#: The error a movie carries when `run_batch` was stopped before reaching it.
STOPPED_ERROR = "not run: the batch was stopped before this movie"


@dataclass(frozen=True)
class TrackMateSettings:
    """What TrackMate is asked to do. The two decisive numbers have NO default."""

    radius_px: float
    threshold: float
    linking_max_distance_px: float = UTRACK_EQUIVALENT["linking_max_distance_px"]
    gap_closing_max_distance_px: float = UTRACK_EQUIVALENT["gap_closing_max_distance_px"]
    max_frame_gap: int = UTRACK_EQUIVALENT["max_frame_gap"]
    allow_splitting: bool = UTRACK_EQUIVALENT["allow_splitting"]
    allow_merging: bool = UTRACK_EQUIVALENT["allow_merging"]
    subpixel: bool = True
    median_filter: bool = False
    target_channel: int = 1

    def __post_init__(self):
        if not (self.radius_px and self.radius_px > 0):
            raise TrackMateRunError(
                "radius_px has no default and must be positive: it is the expected spot "
                "radius in pixels and it decides what counts as a molecule")
        # ZERO IS NOT A SETTING, IT IS THE ABSENCE OF ONE, and this used to allow it.
        # TrackMate's threshold cuts on each spot's quality, so zero keeps everything:
        # measured on one lab movie, threshold 20 already gives 200 456 spots
        # against u-track's 2 325, and zero is below that. The tracking window shows an
        # UNCOMPUTED adaptive threshold as 0, reported as "it keeps normal settings",
        # so without this a batch could be started on it: half an hour producing a folder
        # of noise, with nothing having said no.
        if self.threshold is None or self.threshold <= 0:
            raise TrackMateRunError(
                "threshold has no default and must be > 0; zero keeps every detection. "
                "Compute it from the movies' own noise (the adaptive preset), or sweep "
                "it against the false-positive rate outside cell outlines "
                "(smtracker.detection) instead of choosing it by eye")
        if self.max_frame_gap < 0:
            raise TrackMateRunError("max_frame_gap must be >= 0")

    @property
    def matches_utrack_policy(self) -> bool:
        """Is the linking POLICY byte-for-byte u-track's, 5 px, no gap closing, no
        merge or split?

        NOT the same question as whether the RESULTS are comparable, and conflating the
        two was the original mistake here. u-track reaches median track length 6 with no
        gap closing because its mixture-model fit re-finds a molecule reliably; with the
        same policy TrackMate's LoG flickers near threshold and reaches only 2. Matching
        the policy therefore GUARANTEES a mismatch in the result.
        """
        return (self.linking_max_distance_px
                == UTRACK_EQUIVALENT["linking_max_distance_px"]
                and self.max_frame_gap == UTRACK_EQUIVALENT["max_frame_gap"]
                and not self.allow_splitting and not self.allow_merging)

    @property
    def matches_utrack_results(self) -> bool:
        """Are these the settings MEASURED to reproduce u-track's fractions and track
        counts? See `UTRACK_COMPARABLE`: gap closing ON, 5 px linking, and a per-condition
        threshold chosen by matching in-cell purity."""
        return any(self.radius_px == v["radius_px"]
                   and self.threshold == v["threshold"]
                   and self.max_frame_gap == v["max_frame_gap"]
                   and self.linking_max_distance_px
                   == UTRACK_EQUIVALENT["linking_max_distance_px"]
                   for v in UTRACK_COMPARABLE.values())

    def stamp(self) -> str:
        if self.matches_utrack_results:
            policy = ("u-track-comparable (measured: fractions within 1.5 pp, "
                      "0.72 to 0.88 times the tracks)")
        elif self.matches_utrack_policy:
            policy = ("u-track's linking policy, which is not the same as comparable "
                      "results: without gap closing TrackMate fragments to median "
                      "length 2 against u-track's 6")
        elif (self.linking_max_distance_px == LINKING_RADIUS_PX
              and self.max_frame_gap == ADAPTIVE_BASE["max_frame_gap"]):
            # THE VALIDATED SET IS NOT "CUSTOM", and calling it that was wrong the moment
            # 5 px stopped being inherited. `matches_utrack_results` additionally demands
            # one of two hand-matched THRESHOLDS, which an adaptively-computed threshold
            # never equals, so a correctly-configured run was being stamped "not
            # measured against u-track" while using the one linking radius that HAS been
            # measured, over 6 proteins. Two different claims, now said separately.
            # THE VERDICT ONLY. The evidence behind it -- 6 proteins x 2 carbon
            # sources, +1.4 pp against the deposited fractions, and that the THRESHOLD
            # is per-dataset and not part of the claim -- is in `LINKING_RADIUS_NOTE`,
            # on the linking-radius box's own tooltip, where it is read when the number
            # is being chosen rather than reprinted on every run. Trimmed on request:
            # a paragraph that repeats itself in a console stops being read.
            policy = (f"validated linking ({LINKING_RADIUS_PX:g} px, gap "
                      f"{ADAPTIVE_BASE['max_frame_gap']})")
        else:
            policy = "custom linking, not measured against u-track"
        return (f"TrackMate LoG r={self.radius_px:g} px, threshold={self.threshold:g}, "
                f"link {self.linking_max_distance_px:g} px, gap {self.max_frame_gap} "
                f"frames | {policy}")


_SCRIPT = '''# generated by smtracker.trackmate_run -- headless TrackMate
import sys
from ij import IJ, ImagePlus
from ij.plugin import FolderOpener
from fiji.plugin.trackmate import Model, Settings, TrackMate, Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.io import TmXmlWriter
from java.io import File

IMAGE = {image!r}
OUTPUT = {output!r}
STATUS = {status!r}


def say(line):
    """Print a marker AND append it to STATUS: the current Fiji launcher returns
    nothing through a pipe, so the file is what the caller reads."""
    print(line)
    handle = open(STATUS, "a")
    handle.write(line + "\\n")
    handle.close()

# THREE WAYS IN, AND THE RAW ONE NEEDS BIO-FORMATS. A folder of TIFFs is the prepared
# path; a single .tif opens directly; a raw acquisition (.nd2 from the Nikon/NIS scope,
# .stk from MetaMorph) needs the Bio-Formats importer, opened VIRTUAL so a 4500-frame
# file is not pulled into memory at once.
RAW_SUFFIXES = (".nd2", ".stk", ".lif", ".czi", ".ome.tif")
lower = IMAGE.lower()
if __IS_DIR__:
    imp = FolderOpener.open(IMAGE, "")
elif lower.endswith(RAW_SUFFIXES):
    from loci.plugins import BF
    from loci.plugins.in import ImporterOptions
    opts = ImporterOptions()
    opts.setId(IMAGE)
    opts.setVirtual(True)
    opened = BF.openImagePlus(opts)
    imp = opened[0] if opened else None
else:
    imp = IJ.openImage(IMAGE)
if imp is None:
    say("SMTRACKER_TM_FAIL could not open " + IMAGE)
    sys.exit(1)

# THE FRAME RANGE, 1-BASED INCLUSIVE, matching how a crop is written in a lab book and
# how movieprep's PrepConfig takes it. Applied BEFORE anything is configured, so every
# later count refers to the frames actually tracked.
FIRST_FRAME = {first_frame}
LAST_FRAME = {last_frame}
if FIRST_FRAME > 0 or LAST_FRAME > 0:
    from ij.plugin import Duplicator
    total = imp.getStackSize()
    lo = FIRST_FRAME if FIRST_FRAME > 0 else 1
    hi = LAST_FRAME if LAST_FRAME > 0 else total
    if lo < 1 or hi > total or hi < lo:
        say("SMTRACKER_TM_FAIL frame range %d-%d outside 1-%d" % (lo, hi, total))
        sys.exit(1)
    imp.setDimensions(1, imp.getStackSize(), 1)
    imp = Duplicator().run(imp, 1, 1, lo, hi, 1, 1)

# PIXELS, FORCED. A micron calibration on the TIFF would reinterpret every radius and
# distance below; this project's raw stage is pixels because cell outlines are.
cal = imp.getCalibration()
cal.pixelWidth = 1.0
cal.pixelHeight = 1.0
cal.pixelDepth = 1.0
cal.setUnit("pixel")

# THE STACK IS TIME, NOT Z, AND IMAGEJ DOES NOT KNOW THAT. A folder of 1001 single-frame
# TIFFs opens as 1001 Z-SLICES, and TrackMate links across T only -- so detection finds
# every spot (75 274 of them on a real movie) and the tracker returns ZERO tracks, with
# no error anywhere. Measured, not guessed at. Forcing (channels, slices, frames) is what
# turns a pile of slices into a movie.
n_images = imp.getStackSize()
imp.setDimensions(1, 1, n_images)

model = Model()
model.setLogger(Logger.VOID_LOGGER)
settings = Settings(imp)
settings.detectorFactory = LogDetectorFactory()
settings.detectorSettings = {{
    "DO_SUBPIXEL_LOCALIZATION": {subpixel},
    "RADIUS": {radius},
    "TARGET_CHANNEL": {channel},
    "THRESHOLD": {threshold},
    "DO_MEDIAN_FILTERING": {median},
}}

try:
    from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
except ImportError:
    from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
settings.trackerSettings["LINKING_MAX_DISTANCE"] = {link}
# GAP CLOSING IS A FLAG, NOT A DISTANCE OF ZERO. Setting MAX_FRAME_GAP = 0 while
# ALLOW_GAP_CLOSING stays at its default True produced 75 274 spots and ZERO tracks on a
# real movie: the linker is asked to close gaps of at most zero frames and gives up.
# u-track's `timeWindow = 1` means no gap closing at all, which is this flag being False.
settings.trackerSettings["ALLOW_GAP_CLOSING"] = {gap_closing}
settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = {gap_dist}
settings.trackerSettings["MAX_FRAME_GAP"] = {max_gap_java}
settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = {splitting}
settings.trackerSettings["ALLOW_TRACK_MERGING"] = {merging}

# FEATURE ANALYZERS. Without these TrackMate writes only QUALITY, position, frame,
# radius and the link cost - which is what every XML this project produced until now
# carried. The analyzers run at detection time and are cheap, and they supply three
# things nothing else here can:
#
#   * a PHOTOMETRIC value per spot (mean/median/total intensity), which is comparable
#     with u-track's fitted amplitude and with `spot_intensity`'s integrated value.
#     QUALITY is the LoG filter response and is NOT that.
#   * SNR and CONTRAST per spot, which is the purity question measured directly rather
#     than inferred from where detections fall relative to cell outlines.
#   * NUMBER_GAPS per track, a direct check on the gap-aware dwell code - which exists
#     precisely because TrackMate closes gaps where u-track does not.
#
# Registered defensively: TrackMate moved several of these between packages across 7.x,
# so each import is optional and the script reports which ones it got rather than dying.
_added = []
try:
    from fiji.plugin.trackmate.features.spot import SpotIntensityMultiCAnalyzerFactory
    settings.addSpotAnalyzerFactory(SpotIntensityMultiCAnalyzerFactory())
    _added.append("spot_intensity")
except Exception as exc:
    say("SMTRACKER_TM_NOTE no spot intensity analyzer: " + str(exc))
try:
    from fiji.plugin.trackmate.features.spot import SpotContrastAndSNRAnalyzerFactory
    settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())
    _added.append("spot_snr")
except Exception as exc:
    say("SMTRACKER_TM_NOTE no SNR analyzer: " + str(exc))
try:
    from fiji.plugin.trackmate.features.spot import SpotFitEllipseAnalyzerFactory
    settings.addSpotAnalyzerFactory(SpotFitEllipseAnalyzerFactory())
    _added.append("spot_ellipse")
except Exception as exc:
    pass
try:
    from fiji.plugin.trackmate.features.edges import EdgeSpeedAnalyzer
    settings.addEdgeAnalyzer(EdgeSpeedAnalyzer())
    _added.append("edge_speed")
except Exception as exc:
    say("SMTRACKER_TM_NOTE no edge speed analyzer: " + str(exc))
try:
    from fiji.plugin.trackmate.features.edges import EdgeTimeLocationAnalyzer
    settings.addEdgeAnalyzer(EdgeTimeLocationAnalyzer())
    _added.append("edge_time")
except Exception as exc:
    pass
try:
    from fiji.plugin.trackmate.features.track import TrackDurationAnalyzer
    settings.addTrackAnalyzer(TrackDurationAnalyzer())
    _added.append("track_duration")
except Exception as exc:
    say("SMTRACKER_TM_NOTE no track duration analyzer: " + str(exc))
try:
    from fiji.plugin.trackmate.features.track import TrackSpotQualityFeatureAnalyzer
    settings.addTrackAnalyzer(TrackSpotQualityFeatureAnalyzer())
    _added.append("track_quality")
except Exception as exc:
    pass
try:
    from fiji.plugin.trackmate.features.track import TrackIndexAnalyzer
    settings.addTrackAnalyzer(TrackIndexAnalyzer())
    _added.append("track_index")
except Exception as exc:
    pass
try:
    from fiji.plugin.trackmate.features.track import TrackBranchingAnalyzer
    settings.addTrackAnalyzer(TrackBranchingAnalyzer())
    _added.append("track_branching")
except Exception as exc:
    say("SMTRACKER_TM_NOTE no branching analyzer: " + str(exc))
try:
    from fiji.plugin.trackmate.features.track import TrackMotilityAnalyzer
    settings.addTrackAnalyzer(TrackMotilityAnalyzer())
    _added.append("track_motility")
except Exception as exc:
    pass
say("SMTRACKER_TM_ANALYZERS " + ",".join(_added))

tm = TrackMate(model, settings)
if {threads} > 0:
    tm.setNumThreads({threads})
if not tm.checkInput():
    say("SMTRACKER_TM_FAIL checkInput: " + str(tm.getErrorMessage()))
    sys.exit(1)
if not tm.process():
    say("SMTRACKER_TM_FAIL process: " + str(tm.getErrorMessage()))
    sys.exit(1)

writer = TmXmlWriter(File(OUTPUT))
writer.appendModel(model)
writer.appendSettings(settings)
writer.writeToFile()
say("SMTRACKER_TM_OK spots=" + str(model.getSpots().getNSpots(True)) +
      " tracks=" + str(model.getTrackModel().nTracks(True)))
'''

#: Printed by the generated script. Looked for rather than parsing Fiji's output, which
#: is noise all the way down, see `fiji.check_fiji` for what `--version` returns.
OK_MARKER = "SMTRACKER_TM_OK"
FAIL_MARKER = "SMTRACKER_TM_FAIL"


def build_script(image, output, settings: TrackMateSettings, *,
                 frame_range: tuple[int, int] | None = None, status=None,
                 threads: int | None = None) -> str:
    """The Jython TrackMate script, with every value baked in.

    Values are baked rather than passed as arguments because Fiji's `--run script arg=v`
    quoting is its own adventure on Windows, and a script written to a temp file is
    inspectable when a run goes wrong.

    ``threads`` caps TrackMate's own threads (it takes every core by default), so Fijis run side by side
    (`run_batch`'s ``at_once``) share the cores instead of each claiming all of them. None or 0: TrackMate's own.
    """
    image_path = Path(image)
    return _SCRIPT.format(
        image=str(image_path).replace("\\", "/"),
        output=str(Path(output)).replace("\\", "/"),
        # Where the markers go besides stdout (see `fiji.status_lines`); beside the
        # output by default, so a script built for inspection still runs as written.
        status=str(Path(status) if status else Path(output).with_suffix(".status")
                   ).replace("\\", "/"),
        subpixel=bool(settings.subpixel), radius=float(settings.radius_px),
        channel=int(settings.target_channel), threshold=float(settings.threshold),
        median=bool(settings.median_filter),
        link=float(settings.linking_max_distance_px),
        gap_dist=float(settings.gap_closing_max_distance_px),
        gap_closing=bool(settings.max_frame_gap > 0),
        # TrackMate wants a POSITIVE integer here even when gap closing is off; the flag
        # above is what actually disables it, and 1 is the value that means "adjacent
        # frames only" if it is ever switched back on.
        max_gap_java=max(1, int(settings.max_frame_gap)),
        splitting=bool(settings.allow_splitting), merging=bool(settings.allow_merging),
        # 0 means "no bound on this end"; the range is 1-BASED INCLUSIVE, matching how a
        # crop is written in a lab book and how `movieprep.PrepConfig` takes it.
        first_frame=int(frame_range[0]) if frame_range else 0,
        last_frame=int(frame_range[1]) if frame_range else 0,
        threads=int(threads or 0),
    ).replace("__IS_DIR__", "True" if image_path.is_dir() else "False")


def run_trackmate(image, output_xml, *, settings: TrackMateSettings, install=None,
                  timeout: float = 3600.0, keep_script: bool = False,
                  frame_range: tuple[int, int] | None = None, threads: int | None = None) -> dict:
    """Detect and track one movie with TrackMate, headless, and write TrackMate XML.

    ``image`` is a FOLDER of single-frame TIFFs (this project's prepared layout,
    `tif/<movie>/<movie>/*.tif`), a multi-page TIFF, or a **RAW acquisition**, `.nd2`
    from the Nikon/NIS scope or `.stk` from MetaMorph, opened through Bio-Formats as a
    VIRTUAL stack so a 4500-frame file is not pulled into memory.

    ``frame_range`` is ``(first, last)``, 1-based inclusive, and is what makes the raw
    path usable: a raw acquisition holds the dense early frames the crop exists to
    remove, so tracking one without a range would track a regime the experiment does not
    occupy. Applied before anything is configured, so every reported count refers to the
    frames actually tracked.

    The raw path is an alternative, not a replacement. u-track cannot read `.nd2`, so the
    prepared stacks remain the canonical route for this project's existing data; this
    exists so new data can be tracked without writing a second copy to disk.

    Returns the spot and track counts TrackMate reported, so a caller can see the run
    produced something before reading the XML back with
    `smtracker.trackmate.read_trackmate_xml`.
    """
    from .fiji import fiji_command, find_fiji, status_lines

    if install is None:
        install = find_fiji()
    if install is None:
        from .fiji import describe_absence

        raise TrackMateRunError(describe_absence())
    if not install.has_trackmate:
        raise TrackMateRunError(
            f"{install.root} has no TrackMate jar. It is normally part of Fiji; a "
            "plain ImageJ or a trimmed install will not do")

    image = Path(image)
    if not image.exists():
        raise TrackMateRunError(f"no such image: {image}")
    output_xml = Path(output_xml)
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    folder = Path(tempfile.mkdtemp(prefix="smtracker_tm_"))
    script_path = folder / "run_trackmate.py"
    status_path = folder / "run_trackmate.status"
    script = build_script(image, output_xml, settings, frame_range=frame_range,
                          status=status_path, threads=threads)
    script_path.write_text(script, encoding="utf-8")
    try:
        try:
            done = subprocess.run(
                fiji_command(install, script_path),
                capture_output=True, text=True, timeout=timeout, check=False, creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired as exc:
            raise TrackMateRunError(
                f"TrackMate did not finish within {timeout:g} s on {image.name}. The "
                f"script is at {script_path} if it should be inspected") from exc

        lines = ((done.stdout or "").splitlines() + (done.stderr or "").splitlines()
                 + status_lines(status_path))
        failed = next((ln for ln in lines if FAIL_MARKER in ln), None)
        if failed:
            raise TrackMateRunError(
                f"TrackMate refused {image.name}: {failed.split(FAIL_MARKER, 1)[-1].strip()}")
        ok = next((ln for ln in lines if OK_MARKER in ln), None)
        if ok is None:
            tail = os.linesep.join(ln for ln in lines[-8:] if ln.strip())
            raise TrackMateRunError(
                f"TrackMate produced no result marker for {image.name} "
                f"(exit {done.returncode}). Last output:{os.linesep}{tail}")
        if not output_xml.exists():
            raise TrackMateRunError(
                f"TrackMate reported success but wrote no file at {output_xml}")
        counts = dict(part.split("=", 1) for part in ok.split()
                      if "=" in part and not part.startswith(OK_MARKER))
        return {"xml": output_xml,
                "n_spots": int(counts.get("spots", 0)),
                "n_tracks": int(counts.get("tracks", 0)),
                "settings": settings.stamp(),
                "fiji": str(install.root),
                "script": str(script_path) if keep_script else None}
    finally:
        if not keep_script:
            try:
                script_path.unlink()
                status_path.unlink(missing_ok=True)
                folder.rmdir()
            except OSError:                            # pragma: no cover - best effort
                pass


#: **`tif/<movie>/<movie>/` IS u-track'S OWN OUTPUT, NOT THE PREPARED STACK.** The
#: prepared movie is the single multi-page `tif/<movie>.tif`, what the lab's Fiji macro
#: wrote and what `movieprep.prepare_movie` writes. u-track then IMPORTS that and
#: explodes it into its MovieData layout: `tif/<movie>/movieData.mat` beside a
#: per-channel folder `tif/<movie>/<movie>/` holding one TIFF per frame.
#:
#: VERIFIED on one lab condition: **30 flat `tif/seriesNNN.tif` and 30 `tif/seriesNNN/`
#: folders**, the inner one holding 1001 files named `series001_c1_t0001.tif`, u-track's
#: naming, not this project's.
#:
#: So the default used to point TrackMate at u-track's private intermediate: 1001 files
#: instead of 1, and only present AFTER u-track had already run. The flat stack is tried
#: FIRST and the exploded folder is kept as a fallback, because a project whose flat
#: TIFFs were deleted after tracking still has to be openable.
PREPARED_PATTERNS = ("tif/*.tif", "tif/*/*")


def discover_movies(root, *, pattern: str | None = None):
    """Every prepared movie under a condition folder, sorted so a batch is reproducible.

    With no ``pattern`` the flat multi-page stack `tif/<movie>.tif` is preferred and
    u-track's exploded `tif/<movie>/<movie>/` folder is the fallback: see
    `PREPARED_PATTERNS`. An explicit pattern is honoured as given.

    Returns files or folders; `build_script` opens either.
    """
    root = Path(root)
    patterns = (pattern,) if pattern is not None else PREPARED_PATTERNS
    for candidate in patterns:
        found = sorted(p for p in root.glob(candidate)
                       if (p.is_dir() and any(p.glob("*.tif")))
                       or (p.is_file() and p.suffix.lower() in (".tif", ".tiff")))
        if found:
            return found
    tried = ", ".join(repr(p) for p in patterns)
    raise TrackMateRunError(
        f"no prepared movies under {root} matching {tried}. The prepared stack is the "
        f"single multi-page tif/<movie>.tif that the preparation stage writes; "
        f"tif/<movie>/<movie>/ is u-track's own exploded copy and exists only after "
        f"u-track has run")


def run_batch(movies, output_dir, *, settings: TrackMateSettings, install=None,
              timeout: float = 3600.0, on_progress=None, crops=None,
              thresholds=None, stop=None, at_once: int = 1):
    """Track many movies, then judge them TOGETHER rather than one at a time.

    Eyeballing every movie does not scale (there are 119 here) and it is also the less
    sensitive check. A movie that produced half the usual spots looks perfectly normal on
    its own and is obvious beside its neighbours, which is why this returns a table meant
    to be read as a set and passed to :func:`batch_qc`.

    A movie that FAILS is kept as a row with its error rather than dropped. A short table
    reads as "these are the movies", not as "some are missing".

    ``crops`` maps movie stem -> (first, last), 1-based inclusive, for the RAW path: a raw
    acquisition still holds the dense early frames the crop exists to remove, so tracking
    one without a range would track a regime the experiment does not occupy. A movie with
    no entry is refused by name rather than tracked whole, the same rule as
    `movieprep.prepare_batch`, and for the same reason: a silently-inherited or absent
    range produces a dataset nobody looked at, under a name that says otherwise.

    ``thresholds`` maps movie stem -> detection threshold, so each movie is detected
    against ITS OWN noise. One threshold for a batch is `k x sigma_LoG` of the MEDIAN
    movie, which is right while the movies are alike and wrong the moment they are not:
    measured on the reference group, day 1 sits at sigma 38.5 against 18.3
    and 19.9 for the other two sessions, so a batch median set by the twenty quiet movies
    would detect the nine noisy ones at about 1.6 sigma where the rest get 3.2.

    A movie with no entry keeps ``settings.threshold``, which is what makes this an
    OVERRIDE rather than a second way to configure a run: pass nothing and the behaviour
    is exactly as before. The threshold applied to each movie is recorded in the returned
    table, because a run where every movie had a different setting cannot be described by
    the settings object alone.

    ``stop`` is a callable asked BEFORE each movie; once it answers True, the movies not
    yet run are kept as rows with ``error`` = `STOPPED_ERROR`, never dropped. A stopped
    run must not read as one that finished: a short table would say "these are the
    movies", and a stopped movie is not one that was tracked and found nothing. The movie
    in hand always finishes -- Fiji is a subprocess, and killing it would leave a half
    XML where a reader expects a whole one.

    ``at_once`` Fijis run side by side, one movie each (23 Oct 2026, `prep_model.MAX_AT_ONCE`), each capped at
    its share of the cores (`build_script`'s ``threads``). The table keeps the movies' order; ``on_progress``'s
    ``index`` is then the number of movies FINISHED, called one at a time from whichever finished.
    """
    import pandas as pd

    from .fiji import find_fiji

    if install is None:
        install = find_fiji()
    if install is None:
        from .fiji import describe_absence

        raise TrackMateRunError(describe_absence())

    movies = [Path(m) for m in movies]
    if crops is not None:
        from .rawmovie import movie_name

        missing = [movie_name(m) for m in movies if movie_name(m) not in crops]
        if missing:
            raise TrackMateRunError(
                f"no frame range for {len(missing)} movie(s): "
                + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
                + ". A raw acquisition without a range would be tracked whole, including "
                  "the dense early frames the crop exists to remove. Record ranges in the "
                  "preparation stage first")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import threading
    import time

    from .rawmovie import movie_name

    lanes = max(1, min(int(at_once or 1), len(movies)))
    threads = max(1, (os.cpu_count() or 2) // lanes) if lanes > 1 else None
    lock = threading.Lock()
    finished = [0]

    def one(movie):
        started = time.perf_counter()
        # ITS OWN THRESHOLD, OR THE BATCH'S. `replace` on a frozen dataclass, so the
        # settings object the caller passed is never mutated -- it is also the one
        # `record_run` stamps, and a run that quietly edited it would misdescribe every
        # movie but the last.
        here = settings
        applied = settings.threshold
        name = movie_name(movie)
        if thresholds and name in thresholds:
            from dataclasses import replace

            applied = float(thresholds[name])
            here = replace(settings, threshold=applied)
        row = {"movie": movie.name, "path": str(movie),
               "xml": None, "n_spots": None, "n_tracks": None,
               "threshold": applied,
               "seconds": None, "ok": False, "error": ""}
        if stop is not None and stop():
            row.update(error=STOPPED_ERROR, seconds=0.0)
        else:
            try:
                got = run_trackmate(movie, output_dir / f"{name}.xml",
                                    settings=here, install=install, timeout=timeout,
                                    frame_range=crops.get(name) if crops else None,
                                    threads=threads)
                row.update(xml=str(got["xml"]), n_spots=got["n_spots"],
                           n_tracks=got["n_tracks"], ok=True)
            except Exception as exc:                       # noqa: BLE001 - kept and named
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter() - started
        with lock:
            finished[0] += 1
            if on_progress is not None:
                on_progress(finished[0], len(movies), row)
        return row

    if lanes > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=lanes) as pool:
            rows = list(pool.map(one, movies))
    else:
        rows = [one(movie) for movie in movies]

    table = pd.DataFrame(rows)
    table.attrs["settings"] = settings.stamp()
    table.attrs["fiji"] = str(install.root)
    return table


def batch_qc(table, *, k: float = 3.5):
    """Which movies in a batch do not look like the others?

    Flagged, never dropped, and by the same median ± k·MAD rule `smtracker.qc` uses for
    cells: including its fallback when the MAD is zero, which is the case a small batch
    of near-identical movies actually hits. A flag means "look at this one", and whether
    it is a real problem is not a question a threshold can answer.

    Judged on spots and tracks per movie, and on the tracks-per-spot RATIO: a movie with
    normal detections and few tracks has a LINKING problem rather than a detection one,
    and the two want different fixes.
    """
    import numpy as np
    import pandas as pd

    from .qc import flag_outliers

    ok = table[table["ok"]].copy()
    if ok.empty:
        raise TrackMateRunError(
            "every movie in this batch failed; there is nothing to compare. "
            + "; ".join(table["error"].head(3)))
    ok["tracks_per_spot"] = np.where(ok["n_spots"] > 0,
                                     ok["n_tracks"] / ok["n_spots"], np.nan)

    out = ok[["movie", "n_spots", "n_tracks", "tracks_per_spot", "seconds"]].copy()
    reasons = {name: [] for name in out["movie"]}
    for column in ("n_spots", "n_tracks", "tracks_per_spot"):
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        flags = flag_outliers(values, k=k)
        out[f"{column}_flagged"] = flags.flagged
        out.attrs[f"{column}_scale_estimator"] = flags.scale_estimator
        for name, is_flagged in zip(out["movie"], flags.flagged):
            if is_flagged:
                reasons[name].append(column)
    out["flagged_on"] = [", ".join(reasons[name]) for name in out["movie"]]
    out["flagged"] = out["flagged_on"].astype(bool)

    failed = table[~table["ok"]]
    out.attrs["n_failed"] = int(len(failed))
    out.attrs["failed"] = list(failed["movie"])
    out.attrs["caveat"] = (
        "flags mark movies that differ from their neighbours; nothing is excluded. A "
        "movie flagged on tracks_per_spot but not on n_spots was detected normally and "
        "linked badly, which needs a different fix from a detection problem.")
    return out


#: How far from the batch median a movie's LoG noise must be before a flag means
#: anything. NOT a statistical threshold — a PHYSICAL one, and it exists because the
#: statistical test alone is useless here. Measured across eight real conditions, the
#: within-condition spread is 1.03-1.20x, i.e. the movies agree almost exactly; the MAD
#: is then tiny and `median +- 3.5*MAD` is a razor-thin band that 2 per cent of ordinary
#: variation trips. The first version of this flag fired on movies at **0.95x** and
#: **1.02x** the median, which is noise wearing an alarm's clothes.
#:
#: 1.5 sits above every within-session spread this project has measured (ceiling 1.22)
#: and below the mixed-session signature it exists to catch (2.3). It is the same shape
#: as the MAD-ZERO TRAP already recorded for cells, from the other side: there a zero
#: MAD flagged nothing, here a near-zero one flags everything.
NOISE_FLAG_MIN_RATIO = 1.5


def noise_outliers(per_movie, *, k: float = 3.5,
                   min_ratio: float = NOISE_FLAG_MIN_RATIO):
    """Which movies' LoG noise does not look like their neighbours'?

    WHY THIS IS WORTH FLAGGING AT ALL. The batch threshold is `k x sigma_LoG` of the
    MEDIAN movie, so a condition holding two sessions is detected at two different
    effective sensitivities and nothing on screen says so: the spot counts look normal
    because the threshold absorbed the difference. Measured elsewhere in this project,
    within-session spread is tight (CV 0.02-0.08, ratio 1.01-1.22) while a mixed pair
    reaches **2.3x** -- so a movie far from its neighbours is the signature of two
    sessions filed as one condition.

    `batch_threshold` already measures `sigma_log` per movie and reports only the
    min/max RATIO, which one very quiet movie can set on its own and which says nothing
    about WHICH movie. This applies the same median +- k*MAD rule `qc` uses for cells,
    so a flag names the movie.

    ``per_movie`` is `batch_threshold`'s own list of rows. Rows that could not be
    measured are carried through unflagged rather than dropped: a movie that failed to
    measure is a different problem, and a short list reads as "these are the movies".

    FLAGGED, NEVER DROPPED, and a flag is not a verdict: a genuinely dimmer sample is
    the same signature as a second session, and only the experimenter knows which.
    """
    import numpy as np
    import pandas as pd

    from .qc import flag_outliers

    rows = list(per_movie or ())
    if not rows:
        raise TrackMateRunError("no movies to compare")
    out = pd.DataFrame(rows)
    for column in ("movie", "sigma_log", "ok"):
        if column not in out:
            raise TrackMateRunError(
                f"per_movie rows have no {column!r}; expected `batch_threshold`'s own "
                f"rows")
    out["noise_flagged"] = False
    usable = out["ok"].astype(bool) & pd.to_numeric(
        out["sigma_log"], errors="coerce").notna()
    values = pd.to_numeric(out.loc[usable, "sigma_log"],
                           errors="coerce").to_numpy(dtype=float)
    estimator = "none"
    if values.size:
        flags = flag_outliers(values, k=k)
        median = float(np.median(values))
        out["sigma_ratio_to_median"] = pd.to_numeric(
            out["sigma_log"], errors="coerce") / (median or float("nan"))
        # BOTH TESTS, and the physical one is what makes the flag mean something. A
        # movie must be BOTH unlike its neighbours statistically AND far enough from
        # them to matter; see `NOISE_FLAG_MIN_RATIO`.
        ratio = out.loc[usable, "sigma_ratio_to_median"].to_numpy(dtype=float)
        far = (ratio >= float(min_ratio)) | (ratio <= 1.0 / float(min_ratio))
        out.loc[usable, "noise_flagged"] = np.asarray(flags.flagged) & far
        estimator = flags.scale_estimator
        out.attrs["sigma_median"] = median
        out.attrs["min_ratio"] = float(min_ratio)
    out.attrs["scale_estimator"] = estimator
    out.attrs["n_flagged"] = int(out["noise_flagged"].sum())
    out.attrs["n_unmeasured"] = int((~usable).sum())
    out.attrs["caveat"] = (
        f"flagged means both beyond median +- {k}*MAD and {fold_words(min_ratio)} or more away from "
        f"the median, either way. LoG noise far from the batch median usually means two "
        "sessions filed as one condition, which one batch threshold detects at different "
        "sensitivities; a dimmer sample gives the same signature. "
        "Nothing is excluded.")
    return out


def read_batch(table, *, pixel_size_nm: float | None = None, filtered_only: bool = True):
    """Every XML a batch produced, as ONE raw table, the same one u-track fills.

    This is the join between the two halves: `run_batch` writes TrackMate XML, and
    `smtracker.trackmate.read_trackmate_xml` turns each into `ingest.RAW_COLUMNS`. From
    here nothing downstream knows or cares which tracker produced the data, which is the
    whole point of the reader having been built first.

    The movie name comes from the batch, not from the XML's filename. TrackMate records
    the image it was given, which for this project is a folder that repeats the movie's
    name: recoverable, but one indirection more than necessary when the batch already
    knows.

    A movie that FAILED to track is skipped and COUNTED on `.attrs`, never silently
    absent: a short table reads as "this is the data", which is a different statement
    from "some of it did not run".
    """
    import pandas as pd

    from .trackmate import read_trackmate_xml

    ok = table[table["ok"]] if "ok" in table.columns else table
    if ok.empty:
        raise TrackMateRunError("no successful runs in this batch; nothing to read")

    frames, unreadable = [], []
    for row in ok.itertuples():
        path = Path(row.xml)
        if not path.exists():
            unreadable.append(f"{row.movie}: {path} is gone")
            continue
        try:
            frames.append(read_trackmate_xml(
                path, movie=row.movie, pixel_size_nm=pixel_size_nm,
                filtered_only=filtered_only))
        except Exception as exc:                       # noqa: BLE001 - kept and named
            unreadable.append(f"{row.movie}: {type(exc).__name__}: {exc}")

    if not frames:
        raise TrackMateRunError(
            "every XML in this batch failed to read: " + "; ".join(unreadable[:3]))

    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(getattr(table, "attrs", {}))
    out.attrs["source"] = "trackmate_xml"
    out.attrs["n_movies_read"] = len(frames)
    out.attrs["n_movies_failed_to_track"] = int((~table["ok"]).sum()) \
        if "ok" in table.columns else 0
    out.attrs["unreadable"] = unreadable
    return out


#: Threshold used for the sweep's single permissive run. Low enough that the sweep can
#: see BELOW any threshold anyone would ship, which is the whole point: the old optimizer
#: could only rank detections the upstream detector had already kept, so it measured
#: purity and was blind to completeness. Running detection ourselves is what lifts that.
SWEEP_FLOOR_THRESHOLD = 5.0


def sweep_detection_threshold(image, meshes, *, radius_px: float,
                              image_shape: tuple, install=None,
                              floor_threshold: float = SWEEP_FLOOR_THRESHOLD,
                              target_fdr: float = 0.05, n_thresholds: int = 40,
                              timeout: float = 3600.0, keep_xml=None):
    """Choose a detection threshold from the data, in ONE TrackMate run.

    One run, not one per threshold. TrackMate's THRESHOLD is a cut on each spot's
    QUALITY, so detecting once at a permissive floor and then filtering by quality
    reproduces any higher threshold exactly. VERIFIED on a real movie: filtering the
    floor run at quality >= 100 gives 6 493 detections against 6 493 from a direct run
    at threshold 100, and at 300 gives 960 against 960. Both exact.

    The point is completeness, not just purity. `smtracker.detection` could always
    estimate the false-positive rate from detections outside the cell outlines, but only
    among spots the upstream detector had already kept, so it measured purity and could
    not see what had been discarded. Running the detector ourselves at a floor below any
    sensible setting is what makes the discarded half visible, and it is the reason this
    lives here rather than in `detection`.

    Returns the sweep table and the recommendation, plus the floor run's own numbers so a
    caller can see how much was on the table to begin with.
    """
    from .detection import detections_from_trackmate, recommend_threshold, threshold_sweep

    settings = TrackMateSettings(radius_px=radius_px, threshold=float(floor_threshold))
    xml = Path(keep_xml) if keep_xml is not None else None
    folder = None
    if xml is None:
        folder = Path(tempfile.mkdtemp(prefix="smtracker_sweep_"))
        xml = folder / "sweep.xml"
    try:
        run = run_trackmate(image, xml, settings=settings, install=install,
                            timeout=timeout)
        detections = detections_from_trackmate(xml, movie=Path(image).name)
        sweep = threshold_sweep(detections, meshes, image_shape=image_shape,
                                score="amp", n_thresholds=n_thresholds)
        choice = recommend_threshold(sweep, target_fdr=target_fdr)
    finally:
        if folder is not None:
            import shutil

            shutil.rmtree(folder, ignore_errors=True)

    return {
        "sweep": sweep,
        "choice": choice,
        "floor_threshold": float(floor_threshold),
        "n_detections_at_floor": int(len(detections)),
        "n_spots_reported": run["n_spots"],
        "settings": settings.stamp(),
        "note": (
            f"one run at threshold {floor_threshold:g} then filtered by quality; a "
            "direct run at any higher threshold gives an identical detection set "
            "(verified exactly at 100 and 300 on real data)"),
    }


# ============================================================ adaptive threshold
#: The LoG threshold that matches u-track's yield, in units of the movie's OWN noise.
#:
#: WHY A CONSTANT HERE TRANSFERS WHERE A THRESHOLD DOES NOT. u-track thresholds against
#: LOCAL noise (`alphaLocMax`, a statistical criterion), so one u-track setting travels
#: between datasets. TrackMate's LoG threshold is an ABSOLUTE number on the filtered
#: image, so it does not: the same molecules on a noisier session need a larger number
#: for the same purity. Expressing it as a multiple of the filtered image's own noise
#: puts it back on u-track's footing.
#:
#: MEASURED on a lab project, 59 movies over six sessions, against the two thresholds that were
#: yield-matched to u-track by hand (60 and 130). Per session, sigma_LoG is
#: tight (CV 0.02-0.08) and the implied constant is:
#:
#:     condition A block 1 / 2 / 3   sigma 18.3 / 20.5 / 19.3   ->  k = 3.28 / 2.93 / 3.10
#:     condition B block 1           sigma 37.8                 ->  k = 3.44
#:     condition B block 4 / 5       sigma 18.4 / 20.2          ->  k = 7.07 / 6.43
#:
#: FOUR OF SIX SESSIONS AGREE AT k = 3.0-3.4, ACROSS BOTH CONDITIONS. The two that do not
#: are not a failure of the rule: condition B's 130 was calibrated on block 1, whose noise is
#: twice that of blocks 4 and 5, and then applied to them unchanged. So the fixed preset
#: over-thresholds 20 of the 29 movies of condition B, and the adaptive rule CORRECTS that rather
#: than reproducing it.
#:
#: THIS CONSTANT IS CALIBRATED ON THIS MICROSCOPE. The FORM transfers to another lab; the
#: number should be re-derived there by yield-matching once, which is what `k=` is for.
#:
#: **ON TRACKMATE'S OWN NOISE SCALE SINCE 13 OCT 2026 (ruled).** The k values in the table
#: above, and the shipped 3.2, were measured with a LoG normalised by sigma^2, which is
#: sigma^2 / 2 times TrackMate's own (measured: TrackMate's QUALITY = that response x
#: 2/sigma^2, every spot of a real run). All of them were taken at a 3.0 px spot radius, so
#: on TrackMate's scale they are x2.25: the shipped 3.2 is **7.2**. `log_noise_sigma` now
#: measures on TrackMate's scale, so ONE k is ONE stringency at every spot radius (before,
#: 3.2 meant 6.3 TrackMate-noise SD at 2.8 px and 9.8 at 3.5 px). Thresholds at 3.0 px
#: are unchanged to the digit; at the derived 2.8 px they rise ~13 %, back to calibrated.
UTRACK_MATCHED_K = 7.2

#: Frames sampled to measure the noise. It is a per-session property and tight within a
#: session, so a handful is plenty; the median over frames guards against a bright one.
_NOISE_FRAMES = 8


def log_noise_sigma(frames, *, radius_px: float) -> float:
    """The robust noise level of the LoG-filtered image.

    ``frames`` is an iterable of 2-D arrays. Returns the median over frames of a
    MAD-based sigma, which is what makes it a measure of the BACKGROUND rather than of
    the molecules: the LoG image is full of real spots, and an ordinary standard
    deviation would measure those instead of the noise the threshold has to clear.

    The filter: sigma = radius / sqrt(2) in 2-D, the response scale-normalised by
    sigma^2 and negated so a bright spot gives a positive value.

    **ON TRACKMATE'S OWN SCALE (13 Oct 2026).** The response is ``-2 x`` the Laplacian of
    Gaussian, which is what TrackMate reports as a spot's QUALITY -- MEASURED on a real run
    (one lab movie, radius 3.0 px): every detected spot's QUALITY equals it, no
    spread over 200 spots. Until then this was ``-sigma^2 x`` the LoG and documented as
    "TrackMate's own units", which it was not: it was sigma^2/2 times TrackMate's (2.25x at
    3.0 px, 1.96x at 2.8, 3.06x at 3.5), so one k meant a different stringency at every
    spot radius. The noise and `UTRACK_MATCHED_K` moved to TrackMate's scale together;
    at 3.0 px every threshold is unchanged.

    Where it breaks, measured. The MAD is robust but not immune: molecules inflate it,
    and a threshold derived from an inflated noise would rise with the density, which is
    backwards. On a 128x128 field the inflation is

        spots/frame     1      2      3      5     10     20     40
        inflation    1.02x  1.04x  1.06x  1.11x  1.25x  1.59x  2.46x

    This project's data runs about 2.3 detections per frame on 126x130, i.e. ~1.04x -
    negligible. Above roughly 10 per frame it is not, and the estimate should be taken on
    sparser frames or a different measure used. A dense field is exactly where this
    number should be distrusted.
    """
    import numpy as np
    from scipy.ndimage import gaussian_laplace

    sigma = float(radius_px) / np.sqrt(2.0)
    values = []
    for frame in frames:
        response = -2.0 * gaussian_laplace(np.asarray(frame, dtype=np.float64), sigma)
        median = np.median(response)
        values.append(1.4826 * np.median(np.abs(response - median)))
    if not values:
        raise TrackMateRunError("no frames given, so the noise cannot be measured")
    return float(np.median(values))


def adaptive_threshold(movie, *, radius_px: float, k: float = UTRACK_MATCHED_K,
                       n_frames: int = _NOISE_FRAMES, frame_range=None) -> dict:
    """A per-movie LoG threshold, computed from that movie's own noise.

    ``movie`` is either a folder of single-frame TIFFs (the prepared layout) or a raw
    acquisition (`.nd2`/`.stk`), the same two inputs `run_trackmate` takes.

    Returns the threshold AND what it was derived from, because a computed setting that
    does not say how it was computed is worse than a typed one: nobody can check it.

    What this does and does not claim. It reproduces a hand-tuned, u-track-matched
    threshold on the sessions those were calibrated on, and it disagrees with them where
    a fixed number was carried onto a session with different noise. It is NOT a
    measurement of the right threshold in an absolute sense: no such thing exists
    without either u-track's answer or ground truth. `k` is the transferable part and
    the place to re-calibrate.
    """
    from pathlib import Path

    import numpy as np

    path = Path(movie)
    if path.is_dir():
        files = sorted(p for p in path.iterdir()
                       if p.suffix.lower() in (".tif", ".tiff"))
        if not files:
            raise TrackMateRunError(f"{path}: no TIFF frames to measure noise on")
        if frame_range is not None:
            first, last = frame_range
            files = files[first - 1:last]
        import tifffile

        picked = files[:: max(1, len(files) // n_frames)][:n_frames]
        frames = [tifffile.imread(p) for p in picked]
        source = f"{len(files)} frame(s) in {path.name}"
    else:
        from .rawmovie import read_raw_movie

        raw = read_raw_movie(path)
        total = int(raw.meta.n_frames)
        first, last = (1, total) if frame_range is None else frame_range
        indices = np.linspace(first - 1, last - 1, num=min(n_frames, last - first + 1))
        frames = [raw.frame(int(i)) for i in indices]
        source = f"{last - first + 1} frame(s) of {path.name}"

    sigma = log_noise_sigma(frames, radius_px=radius_px)
    return {
        "threshold": float(k * sigma),
        "sigma_log": sigma,
        "k": float(k),
        "radius_px": float(radius_px),
        "n_frames_measured": len(frames),
        "source": source,
        "basis": (f"threshold = {k} x the movie's own LoG noise ({sigma:.1f}); "
                  f"k calibrated against u-track's yield"),
    }


#: Above this ratio between the noisiest and quietest movie, one threshold for the batch
#: is a compromise rather than a setting. MEASURED: one lab project spans 17.9 to 41.4 (2.3x)
#: because it holds three sessions, one of which is twice as noisy as the others - and the
#: shipped preset gives all 29 movies the number matched on the noisy one.
MIXED_SESSION_RATIO = 1.5


def batch_threshold(movies, *, radius_px: float, k: float = UTRACK_MATCHED_K,
                    crops=None, on_progress=None) -> dict:
    """One LoG threshold for a batch, computed from the movies' own noise.

    `run_batch` takes ONE settings object for the whole batch, so this returns the
    MEDIAN over movies rather than a per-movie value - and reports the spread, because
    that is the part a single number hides. Within one session the noise is tight
    (CV 0.02-0.08 measured over six sessions); across sessions it is not.

    A movie that cannot be measured is kept as a row with its error, never dropped: a
    short list reads as "these are the movies".
    """
    from pathlib import Path

    from .rawmovie import movie_name

    rows = []
    for index, movie in enumerate(movies, start=1):
        path = Path(movie)
        name = movie_name(path)
        window = None if crops is None else crops.get(name)
        rows.append(movie_threshold_row(path, name=name, radius_px=radius_px, k=k,
                                        frame_range=window))
        if on_progress is not None:
            on_progress(index, len(movies), rows[-1])
    return summarise_thresholds(rows, radius_px=radius_px, k=k)


def movie_threshold_row(path, *, name: str, radius_px: float,
                        k: float = UTRACK_MATCHED_K, frame_range=None) -> dict:
    """One movie's row of `batch_threshold`: its LoG noise and threshold, or the error.

    SPLIT OUT so a caller can measure movies in parallel processes (the workspace-style
    preparation window runs three at once) and still summarise them with exactly the
    arithmetic `batch_threshold` uses. A movie that cannot be measured is a row with its
    error, never dropped. Module-level, so a process pool can run it.
    """
    try:
        got = adaptive_threshold(path, radius_px=radius_px, k=k, frame_range=frame_range)
        return {"movie": name, "sigma_log": got["sigma_log"],
                "threshold": got["threshold"], "ok": True, "error": ""}
    except Exception as exc:                       # noqa: BLE001 - kept and named
        return {"movie": name, "sigma_log": None, "threshold": None,
                "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def summarise_thresholds(rows, *, radius_px: float, k: float = UTRACK_MATCHED_K) -> dict:
    """`batch_threshold`'s answer from its per-movie rows: the MEDIAN threshold, the
    noise spread, and whether the batch looks like more than one session."""
    from statistics import median

    rows = list(rows)
    usable = [r for r in rows if r["ok"]]
    if not usable:
        raise TrackMateRunError(
            "no movie could be measured, so no threshold can be computed: "
            + "; ".join(f"{r['movie']}: {r['error']}" for r in rows[:4]))

    sigmas = [r["sigma_log"] for r in usable]
    lo, hi = min(sigmas), max(sigmas)
    ratio = hi / lo if lo > 0 else float("inf")
    return {
        "threshold": float(median(r["threshold"] for r in usable)),
        "k": float(k), "radius_px": float(radius_px),
        "n_movies": len(usable), "n_failed": len(rows) - len(usable),
        "sigma_median": float(median(sigmas)),
        "sigma_min": float(lo), "sigma_max": float(hi),
        "sigma_ratio": float(ratio),
        "mixed_sessions": bool(ratio > MIXED_SESSION_RATIO),
        "per_movie": rows,
    }


#: The tracking settings a condition needs, in the order the window shows them.
#:
#: THRESHOLD FIRST, because it is the one that is MEASURED rather than chosen: it is
#: `k x sigma_LoG` of that condition's own movies and does not transfer between datasets.
#: The two radii follow, and they are different quantities (how big a molecule LOOKS
#: against how far it MOVES), which is why they are separate rows rather than one.
TRACKING_FIELDS = (
    # NAMED AS WHAT IT IS. The cell holds the MEDIAN of the per-movie thresholds, and a
    # bare "LoG detection threshold" reads as "the value every movie got" -- which it is
    # not, since each movie is detected against its own noise. Asked directly: "is the
    # table the median (average?), then we should say so."
    ("threshold", "LoG threshold (median)", ""),
    ("radius_px", "spot radius", "px"),
    ("linking_max_distance_px", "linking radius", "px"),
    ("max_frame_gap", "frame gap", "frames"),
)


def settings_per_condition(conditions, *, base=None) -> dict:
    """``{condition: {field: value}}`` seeded from one settings object.

    ``base`` is what the window currently holds; every condition starts from it, because
    the radii and the gap are properties of the OPTICS and the KINETICS and are shared
    across a session imaged the same way. The threshold is not seeded: it is measured per
    condition and a copied one would report one condition's noise over another's data.
    """
    seed = {}
    if base is not None:
        for key, _label, _unit in TRACKING_FIELDS:
            if key == "threshold":
                continue
            value = getattr(base, key, None)
            if value is not None:
                seed[key] = value
    return {name: dict(seed) for name in conditions}


def spot_radius_per_condition(calibrations: dict, *, sigmas: float = 3.0) -> dict:
    """``{condition: radius_px}`` derived from EACH condition's own optics.

    THE SPOT RADIUS IS NOT A SETTING, IT IS A CONSEQUENCE. It is how big a molecule
    LOOKS -- a LoG kernel matched to the PSF -- so it follows the emission wavelength,
    the numerical aperture and the pixel size, all three of which the calibration grid
    already holds per condition. The tracking grid was seeded from the preset instead,
    so every column read a flat 3 while the window's own optics box computed 2.80 from
    the same numbers: two answers to one question, on one screen.

    A condition missing any of the three is ABSENT from the result rather than given a
    default, because 3.0 px is right only for this project's own optics and reads as a
    measurement once it is printed in a column. `optics.detection_radius_px` explains
    where `sigmas = 3` comes from: it is what reproduces the 3.0 px in use here, chosen
    so switching to the derived value does not silently move existing results.

    A red dye at 646 nm gives 3.50 px on the same scope and NA 1.2 gives 4.34 -- the
    differences a single number cannot express, and the reason this is per condition.
    """
    from .optics import OpticsError, detection_radius_px

    out = {}
    for name, values in (calibrations or {}).items():
        emission = (values or {}).get("emission_nm")
        aperture = (values or {}).get("numerical_aperture")
        pixel = (values or {}).get("pixel_size_nm")
        if not emission or not aperture or not pixel:
            continue
        try:
            out[name] = detection_radius_px(emission_nm=float(emission),
                                            numerical_aperture=float(aperture),
                                            pixel_size_nm=float(pixel),
                                            sigmas=sigmas)
        except (OpticsError, TypeError, ValueError):
            continue
    return out


def missing_tracking_settings(settings: dict) -> dict:
    """``{condition: [fields]}`` for conditions that cannot be tracked yet.

    The threshold and both radii must be POSITIVE: `TrackMateSettings` refuses a
    threshold of zero because it keeps every detection, and a radius of zero has no
    meaning.

    ZERO IS A REAL FRAME GAP, and treating it as missing was wrong: u-track ran
    `timeWindow = 1`, i.e. no gap closing at all, and that is the setting this project
    reproduces. A field whose valid range includes zero cannot be checked for
    truthiness, which is the same mistake as reading a measured 0 as "not set".
    """
    positive = {"threshold", "radius_px", "linking_max_distance_px"}
    out = {}
    for name, values in settings.items():
        gaps = []
        for key, _label, _unit in TRACKING_FIELDS:
            value = values.get(key)
            if value is None or (key in positive and not value):
                gaps.append(key)
        if gaps:
            out[name] = gaps
    return out
