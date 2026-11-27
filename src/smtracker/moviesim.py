"""Synthetic single-molecule MOVIES with known truth, the only way to measure RECALL.

Why this exists
---------------
`smtracker.detection` estimates a detector's false-positive rate from spots landing
OUTSIDE cell outlines. That measures **purity** and is structurally blind to
**completeness**: it only ever sees detections the detector already kept, so it cannot
count the molecules it discarded. This project has recorded that limitation since the
detection work began, and it became load-bearing when TrackMate and u-track disagreed:
at matched in-cell purity TrackMate finds **2.35 to 5.63x** u-track's detections
(median 3.13x). Either u-track is conservative and those extras are real molecules, or
the metric is too coarse to tell.

**No measurement on real data answers that**, because real data has no ground truth. A
movie you generated does. This module renders one.

What it does NOT re-implement
-----------------------------
The MOTION. `smtracker.simulate` already produces ground-truth tracks with switching
states inside cells, including bleaching, motion blur and defocalisation. This module
consumes that table and adds only image formation, so there is one source of truth for
what the molecules did.

**Do not also set `miss_probability` OR `dZ_um` ON THE SIMULATION.** Those model the
DETECTOR failing, which is the very thing being measured here; leaving them on would
delete molecules before rendering and then credit the detector for not finding them.
`render_movie` refuses a table whose attrs say either was used.

The photon budget has no default, and it is the answer
------------------------------------------------------
Recall is almost entirely a function of how many photons a molecule emits per frame
against the background. Invent that number and the recall figure is a measurement of the
invention. So `PhotonSpec` refuses to default it, and `calibrate_photons` solves for the
budget that reproduces a **measured** spot signal-to-noise from the real movies, which
is what anchors the answer to this microscope rather than to an assumption.

The camera model is the EMCCD this project measured: offset ~507 counts, EM gain 140,
and the EM register's **excess noise factor** of sqrt(2), which is the thing naive
simulations omit: electron multiplication roughly doubles the variance, so a simulated
movie without it is about 1.4x cleaner than the real one and flatters every detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MovieSimError",
    "OpticsSpec",
    "CameraSpec",
    "calibrate_camera",
    "calibrate_photons",
    "PhotonSpec",
    "stagger_starts",
    "render_movie",
    "match_to_truth",
    "recall_curve",
]

#: The EM register multiplies stochastically, which inflates the variance by very nearly
#: a factor of two, so the noise is sqrt(2) larger than shot noise alone. Omitting it is
#: the classic way a simulated movie comes out too clean.
EMCCD_EXCESS_NOISE = float(np.sqrt(2.0))


class MovieSimError(RuntimeError):
    """Raised loud. Nothing here invents a photon budget or a camera setting."""


@dataclass(frozen=True)
class OpticsSpec:
    """The PSF. ``psf_sigma_px`` comes from `movieData.mat` (0.932 px here)."""

    psf_sigma_px: float
    pixel_size_nm: float

    def __post_init__(self):
        if not self.psf_sigma_px > 0:
            raise MovieSimError("psf_sigma_px must be > 0; u-track records it per movie")
        if not self.pixel_size_nm > 0:
            raise MovieSimError(
                "pixel_size_nm must be > 0, and must come from movieData.mat (106 nm "
                "here) rather than from the .nd2, which reports the bare objective "
                "magnification")


@dataclass(frozen=True)
class CameraSpec:
    """An EMCCD.

    ``sensitivity_e_per_count`` is the OUTPUT sensitivity, electrons per count AFTER the
    EM register, so::

        counts = offset + N x em_gain / sensitivity

    **Only the ratio matters, and saying otherwise was my own error.** An earlier version
    of this docstring claimed the two "must be kept apart" because N alone sets the shot
    noise, but N is fixed by `PhotonSpec`, so two cameras with the same
    ``counts_per_photon`` are indistinguishable in both mean AND noise. A test asserting
    they differ failed, and it was the claim that was wrong, not the code.

    The real defect was the ratio's VALUE. The first draft wrote
    ``counts = offset + N / conversion_gain`` with ``conversion_gain = 3``, i.e. 0.33
    counts per photon where the truth is **33**, so for an observed count level it
    inferred 100x too many photons, and shot noise came out sqrt(100) = 10x small. The
    rendered movie measured **8.2x too quiet** against a real one (temporal sd 12.4
    counts against 100.9), and any detector benchmarked on it would have looked far
    better than it is.

    So the lesson is not a parameterisation, it is that ``counts_per_photon`` must be
    **measured**, `calibrate_camera` gets it by photon transfer, and its residual
    against a real movie is about 1 per cent.
    """

    offset_counts: float
    em_gain: float
    sensitivity_e_per_count: float
    read_noise_e: float = 0.0
    excess_noise: float = EMCCD_EXCESS_NOISE

    def __post_init__(self):
        for name in ("offset_counts", "em_gain", "sensitivity_e_per_count"):
            if not getattr(self, name) > 0:
                raise MovieSimError(f"{name} must be > 0")
        if self.read_noise_e < 0:
            raise MovieSimError("read_noise_e must be >= 0")
        if self.excess_noise < 1.0:
            raise MovieSimError(
                "excess_noise < 1 is not physical: the EM register ADDS variance. Use "
                f"{EMCCD_EXCESS_NOISE:.4f} (sqrt 2) unless you have measured otherwise")

    @property
    def counts_per_photon(self) -> float:
        """How many counts one detected photon becomes, on average."""
        return self.em_gain / self.sensitivity_e_per_count

    def photons_from_counts(self, counts) -> float:
        """Invert the mean relation. The ONLY place counts become photons."""
        return max(float(counts) - self.offset_counts, 0.0) / self.counts_per_photon


def calibrate_camera(stack, *, em_gain: float, offset_counts: float,
                     excess_noise: float = EMCCD_EXCESS_NOISE,
                     low_percentile: float = 2.0,
                     high_percentile: float = 90.0) -> CameraSpec:
    """Measure the camera from a real stack by photon transfer. No assumed sensitivity.

    Across pixels, temporal variance is linear in temporal mean with slope
    ``excess^2 x em_gain / sensitivity``, so fitting variance against mean gives the
    ratio the mean alone cannot, and the sensitivity follows from a known EM gain.

    The fit uses the BULK of pixels (``low_percentile`` to ``high_percentile``): the dim
    tail is dominated by the read floor and the bright tail by cells and molecules, and
    neither obeys the background's transfer curve.

    VERIFIED on lab data (one 60-frame movie): the fit gives
    ``sensitivity = 4.24`` e-/count at EM gain 140, and predicts a per-pixel temporal sd
    of **102.5 counts** against a measured **101.6**, about 1 per cent. It also puts the
    background at **4.8 photons per pixel per 20 ms**, where the collapsed-gain model
    implied 477.
    """
    block = np.asarray(stack, dtype=np.float64)
    if block.ndim != 3 or block.shape[0] < 8:
        raise MovieSimError(
            f"photon transfer needs a (frames, rows, cols) stack of at least 8 frames, "
            f"got {block.shape}")
    if not em_gain > 0:
        raise MovieSimError("em_gain must be > 0 and is read from the .nd2 metadata")

    mean = block.mean(axis=0).ravel()
    var = block.var(axis=0).ravel()
    lo, hi = np.percentile(mean, [low_percentile, high_percentile])
    keep = (mean > lo) & (mean < hi)
    if keep.sum() < 100:
        raise MovieSimError("too few pixels survive the percentile window to fit")

    design = np.vstack([mean[keep], np.ones(int(keep.sum()))]).T
    slope, intercept = np.linalg.lstsq(design, var[keep], rcond=None)[0]
    if slope <= 0:
        raise MovieSimError(
            f"the photon-transfer slope came out {slope:.3g}, which is not physical: "
            "variance must rise with mean. Check the offset and that this is a raw stack")

    sensitivity = excess_noise ** 2 * em_gain / slope
    read_var = intercept + slope * offset_counts
    read_counts = float(np.sqrt(max(read_var, 0.0)))
    spec = CameraSpec(offset_counts=float(offset_counts), em_gain=float(em_gain),
                      sensitivity_e_per_count=float(sensitivity),
                      read_noise_e=read_counts * float(sensitivity),
                      excess_noise=float(excess_noise))
    return spec


@dataclass(frozen=True)
class PhotonSpec:
    """How bright a molecule is. Neither field has a default: see the module docstring.

    ``photons_per_frame`` is photons collected from ONE molecule in one exposure;
    ``background_photons_per_px`` is everything else reaching a pixel (autofluorescence,
    stray light, out-of-focus emission). Their RATIO is what sets recall.
    """

    photons_per_frame: float
    background_photons_per_px: float

    def __post_init__(self):
        if not self.photons_per_frame > 0:
            raise MovieSimError(
                "photons_per_frame has NO DEFAULT: recall is almost entirely a function "
                "of it, so a guessed value makes the recall figure a measurement of the "
                "guess. Use calibrate_photons against a real movie")
        if self.background_photons_per_px < 0:
            raise MovieSimError("background_photons_per_px must be >= 0")

    @property
    def snr(self) -> float:
        """Peak-pixel photons over the background's shot noise. A rough guide only."""
        return self.photons_per_frame / max(np.sqrt(self.background_photons_per_px), 1e-9)


def calibrate_photons(stack, *, camera: CameraSpec, optics: OpticsSpec,
                      n_frames: int = 40, spot_percentile: float = 99.9) -> PhotonSpec:
    """Solve for the photon budget that reproduces a REAL movie's spot brightness.

    This is what anchors a recall figure to this microscope instead of to a guess, and it
    is why `PhotonSpec` refuses to default.

    The inversion is the camera model run backwards, through
    `CameraSpec.photons_from_counts` so the relation lives in ONE place::

        photons_per_pixel = (counts - offset) / (em_gain / sensitivity)

    The background level is the frame median. The spot peak is a high percentile of the
    background-subtracted frame, and a 2-D Gaussian puts ``1 / (2 pi sigma^2)`` of a
    molecule's photons in its brightest pixel, so the total follows.

    ``spot_percentile`` is a CHOICE and it matters: too low and it measures background,
    too high and it measures the brightest outlier or a cosmic ray. 99.9 on a 126x130
    field is about the 16 brightest pixels per frame, which is the right order for a
    single-molecule field and is stated so it can be argued with.

    Not a measurement of one molecule. It assumes the brightest spots ARE single
    molecules, which is the same assumption the crop already rests on. On an uncropped
    or early-frame stack it will read multi-fluorophore spots and overstate the budget.
    """
    block = np.asarray(stack, dtype=np.float64)
    if block.ndim != 3 or block.shape[0] < 1:
        raise MovieSimError(f"expected a (frames, rows, cols) stack, got {block.shape}")
    if not 50.0 < spot_percentile < 100.0:
        raise MovieSimError("spot_percentile must be in (50, 100)")
    use = block[:min(int(n_frames), block.shape[0])]

    background_counts = float(np.median(use))
    peak_counts = float(np.percentile(use, spot_percentile))
    if peak_counts <= background_counts:
        raise MovieSimError(
            f"the {spot_percentile} percentile ({peak_counts:.1f}) is not above the "
            f"median ({background_counts:.1f}): this stack has no spots to calibrate on")

    background_photons = camera.photons_from_counts(background_counts)
    peak_photons = (peak_counts - background_counts) / camera.counts_per_photon
    per_molecule = peak_photons * 2.0 * np.pi * optics.psf_sigma_px ** 2

    spec = PhotonSpec(photons_per_frame=float(per_molecule),
                      background_photons_per_px=float(background_photons))
    return spec


def stagger_starts(tracks, *, n_frames: int, seed: int | None = 0):
    """Spread track start times across a longer movie, and REPORT the density.

    `simulate.simulate_condition` starts every molecule at frame 1, which is right for a
    kinetics simulation and wrong for a rendered movie: it puts the whole population in
    the first frames and then bleaches them together.

    Density is not a cosmetic detail here: it is the variable that decides whether a
    detector's extra detections are real molecules or merged neighbours, which is the
    entire question this module exists to answer. Rendering the un-staggered table gave
    **49.9 localisations per frame** against the **~2.3** u-track reports on this field, a
    22-fold excess at which spots overlap and any recall figure would describe a regime
    the experiment never occupies.

    Each track keeps its own length, kinetics and positions; only its frame numbers move.
    The returned table carries ``attrs['localisations_per_frame']`` so the density is
    stated rather than assumed.
    """
    if n_frames < 2:
        raise MovieSimError("n_frames must be >= 2")
    out = tracks.copy()
    rng = np.random.default_rng(seed)
    shifted = []
    for _uid, group in out.groupby("track_id", sort=False):
        span = int(group["frame"].max() - group["frame"].min())
        latest = max(int(n_frames) - span, 1)
        start = int(rng.integers(1, latest + 1))
        block = group.copy()
        block["frame"] = block["frame"] - block["frame"].min() + start
        shifted.append(block)
    out = pd.concat(shifted, ignore_index=True)
    out = out[out["frame"] <= int(n_frames)].copy()
    if out.empty:
        raise MovieSimError("staggering removed every localisation; n_frames too small")
    density = len(out) / float(out["frame"].nunique())
    out.attrs.update(tracks.attrs)
    out.attrs["localisations_per_frame"] = float(density)
    out.attrs["n_frames_requested"] = int(n_frames)
    return out


def _refuse_detector_knobs(tracks) -> None:
    """The simulation must not have deleted molecules before we render them."""
    used = getattr(tracks, "attrs", {}) or {}
    spec = used.get("imaging") or used.get("params") or {}
    for name in ("miss_probability", "dZ_um"):
        value = getattr(spec, name, None) if not isinstance(spec, dict) \
            else spec.get(name)
        if value:
            raise MovieSimError(
                f"the simulation used {name}={value!r}, which models the DETECTOR "
                "failing: the very quantity being measured here. Rendering it would "
                "delete molecules first and then credit the detector for not finding "
                "them. Re-simulate with it off")


def render_movie(tracks, *, optics: OpticsSpec, camera: CameraSpec, photons: PhotonSpec,
                 field_shape=(126, 130), margin_px: float = 6.0,
                 seed: int | None = 0, centre: bool = True, n_frames: int | None = None,
                 background_map=None, extra_background=None):
    """Render ground-truth tracks as a movie. Returns ``(stack, truth)``.

    ``stack`` is ``(frames, rows, cols)`` uint16, exactly what a camera writes.
    ``truth`` is one row per rendered localisation with its pixel position, so recall is
    counted against what was actually put in the image, not against the input table,
    which may contain molecules that fell outside the field.

    Positions arrive in the cell frame (centred on zero), so they are shifted into the
    field and any molecule landing outside it is dropped and REPORTED in
    ``truth.attrs['dropped_outside_field']`` rather than silently discarded.

    ``centre=False`` takes ``x``/``y`` as FIELD pixels already (a synthetic project that
    placed its cells in a field, `synthetic`), and ``n_frames`` renders EVERY frame
    1..n_frames -- empty ones as background -- so the stack is a whole movie whose frame k
    is simulation frame k (``rendered_frame == frame``). Without it only frames holding a
    molecule are rendered, which the recall benchmark wants and a movie does not.

    An optional ``photons`` column gives each ROW its own photon count instead of
    ``photons.photons_per_frame``: MOTION BLUR is then a molecule's sub-step positions in one
    exposure, each a spot carrying its share of the frame's photons, so a fast molecule is
    smeared and its peak dimmed as a camera records it (`synthetic.build_movie_project`).
    An optional ``sigma_px`` column gives each row its own PSF width (a molecule out of focus:
    wider, so dimmer at its peak, the photons kept).

    ``background_map`` (rows x cols, photons per pixel per frame) replaces the flat
    ``photons.background_photons_per_px`` (an illumination profile); ``extra_background`` is
    ``(map, factors)``: ``map`` (rows x cols) added to frame k times ``factors[k]`` -- a
    cell's autofluorescence fading as it bleaches.
    """
    required = {"frame", "x", "y", "track_id"}
    missing = required - set(getattr(tracks, "columns", ()))
    if missing:
        raise MovieSimError(f"ground-truth table is missing {sorted(missing)}")
    _refuse_detector_knobs(tracks)

    rng = np.random.default_rng(seed)
    rows, cols = int(field_shape[0]), int(field_shape[1])

    t = tracks.copy()
    if centre:
        # Centre the population in the field.
        t["px"] = t["x"].to_numpy(float) - t["x"].mean() + cols / 2.0
        t["py"] = t["y"].to_numpy(float) - t["y"].mean() + rows / 2.0
    else:
        t["px"] = t["x"].to_numpy(float)
        t["py"] = t["y"].to_numpy(float)
    inside = ((t["px"] >= margin_px) & (t["px"] < cols - margin_px)
              & (t["py"] >= margin_px) & (t["py"] < rows - margin_px))
    dropped = int((~inside).sum())
    t = t[inside].copy()
    if t.empty:
        raise MovieSimError(
            "every molecule fell outside the field; check field_shape against the "
            "cell geometry the tracks were simulated in")

    if n_frames is not None:
        if t["frame"].max() > n_frames or t["frame"].min() < 1:
            raise MovieSimError(f"a molecule's frame lies outside 1..{n_frames}")
        frames = np.arange(1, int(n_frames) + 1)
    else:
        frames = np.sort(t["frame"].unique())
    frame_index = {f: i for i, f in enumerate(frames)}
    photon_image = np.zeros((len(frames), rows, cols), dtype=np.float64)
    if background_map is not None:
        background_map = np.asarray(background_map, dtype=np.float64)
        if background_map.shape != (rows, cols):
            raise MovieSimError(f"background_map is {background_map.shape}, the field {(rows, cols)}")
        photon_image += background_map[None, :, :]
    else:
        photon_image += float(photons.background_photons_per_px)
    if extra_background is not None:
        # ONE (map, factors) pair, or a LIST of them (25 Nov 2026, `generator`: autofluorescence fading and a
        # background that changes over the movie, each with its own time course)
        pairs = [extra_background] if isinstance(extra_background, tuple) else list(extra_background)
        for extra, factors in pairs:
            extra = np.asarray(extra, dtype=np.float64)
            factors = np.asarray(factors, dtype=np.float64)
            if extra.shape != (rows, cols) or len(factors) != len(frames):
                raise MovieSimError("extra_background needs a field-sized map and one factor per frame")
            photon_image += factors[:, None, None] * extra[None, :, :]

    sigma = float(optics.psf_sigma_px)
    per_sigma = (t["sigma_px"].to_numpy(float) if "sigma_px" in t.columns
                 else np.full(len(t), sigma))
    reach = int(np.ceil(4 * max(sigma, float(per_sigma.max()) if len(per_sigma) else sigma)))
    yy = np.arange(-reach, reach + 1)
    per_row = (t["photons"].to_numpy(float) if "photons" in t.columns
               else np.full(len(t), float(photons.photons_per_frame)))
    for f, px, py, n_photons, sigma in zip(t["frame"].to_numpy(), t["px"].to_numpy(),
                                           t["py"].to_numpy(), per_row, per_sigma):
        k = frame_index[f]
        r0, c0 = int(round(py)), int(round(px))
        rs = np.clip(np.arange(r0 - reach, r0 + reach + 1), 0, rows - 1)
        cs = np.clip(np.arange(c0 - reach, c0 + reach + 1), 0, cols - 1)
        # Integrated Gaussian over each pixel would be exacter; the sampled form is
        # within a per cent at sigma ~ 1 px and is what matters here (relative SNR).
        gr = np.exp(-0.5 * ((rs - py) / sigma) ** 2)
        gc = np.exp(-0.5 * ((cs - px) / sigma) ** 2)
        psf = np.outer(gr, gc)
        total = psf.sum()
        if total > 0:
            photon_image[k, rs[:, None], cs[None, :]] += n_photons * psf / total

    # --- the camera ---------------------------------------------------------
    # PHOTOELECTRONS first, and they are what carries the shot noise. The EM gain and
    # the output sensitivity appear together in the MEAN (as counts_per_photon) but only
    # the electron count sets the variance, which is exactly the distinction the first
    # draft lost, leaving the movie 8.2x too quiet.
    electrons = rng.poisson(photon_image).astype(np.float64)
    amplified = electrons * camera.em_gain
    # THE EM REGISTER ADDS (F^2 - 1) x G^2 n, NOT F^2 x G^2 n (fixed 25 Oct 2026). The excess noise factor F^2 = 2
    # is the WHOLE variance over the shot noise's: Var = F^2 G^2 lambda, the Poisson term included. Adding F x
    # sqrt(n) x G on top of the Poisson draw counted the shot noise twice -- variance 3 G^2 lambda, measured at an
    # excess^2 of 2.93 on rendered pixels -- so every set rendered before was 1.22x noisier per photon than its camera
    # (`docs/synthetic_synr.md`, "FREE mNG ON NIKON 1"). Asked: "Fix yes".
    spread = (np.sqrt(max(camera.excess_noise ** 2 - 1.0, 0.0))
              * np.sqrt(np.maximum(electrons, 0.0)) * camera.em_gain)
    amplified = amplified + rng.normal(0.0, 1.0, amplified.shape) * spread
    amplified += rng.normal(0.0, camera.read_noise_e, amplified.shape)
    counts = camera.offset_counts + amplified / camera.sensitivity_e_per_count
    stack = np.clip(np.rint(counts), 0, 65535).astype(np.uint16)

    # THE FRAME A DETECTOR WILL REPORT IS THE STACK INDEX, NOT THE SIMULATION'S FRAME
    # NUMBER. Only frames containing a molecule are rendered, and staggered starts mean
    # the simulation's numbering has gaps, so the two differ - here 4..1000 against
    # 1..978. Dropping the mapping is the same defect class this project already recorded
    # between u-track (1-based) and TrackMate (0-based): it does not crash, it silently
    # matches nothing. `rendered_frame` is 1-BASED, matching how the frames are written.
    t = t.copy()
    t["rendered_frame"] = t["frame"].map(lambda f: frame_index[f] + 1)
    truth = t[["track_id", "frame", "rendered_frame", "px", "py"]].copy()
    for extra in ("true_state", "true_D", "cell"):
        if extra in t.columns:
            truth[extra] = t[extra].to_numpy()
    truth.attrs["dropped_outside_field"] = dropped
    truth.attrs["n_frames"] = int(len(frames))
    truth.attrs["field_shape"] = (rows, cols)
    truth.attrs["photons"] = (photons.photons_per_frame,
                              photons.background_photons_per_px)
    truth.attrs["snr"] = photons.snr
    return stack, truth


def match_to_truth(truth, detections, *, tolerance_px: float = 2.0,
                   frame_column: str = "frame", x_column: str = "x",
                   y_column: str = "y", truth_frame_column: str | None = None,
                   with_pairs: bool = False):
    """Match detections to ground truth per frame, nearest first, one-to-one.

    ONE-TO-ONE is the point. Letting two detections both claim one molecule would score a
    detector that splits every spot in half as perfect recall, and letting one detection
    claim two molecules would hide exactly the merging that high density causes. Matching
    is greedy by distance within each frame, which is optimal enough at these densities
    and is stated rather than assumed.

    Returns a dict with ``recall``, ``precision``, ``n_truth``, ``n_detected``,
    ``n_matched`` and the median localisation error of the matched pairs; with
    ``with_pairs``, also ``pairs``: ``[(truth index label, detection index label)]``, the
    matching itself, which linking accuracy needs (are two linked detections the same
    molecule?) and recall alone cannot give.
    """
    if tolerance_px <= 0:
        raise MovieSimError("tolerance_px must be > 0")
    det = detections
    for col in (frame_column, x_column, y_column):
        if col not in getattr(det, "columns", ()):
            raise MovieSimError(f"detections are missing {col!r}")

    # Match on the RENDERED index when the truth carries one: that is the frame a
    # detector reads off the stack. Defaulting to it silently would hide a mismatch, so
    # it is chosen explicitly and falls back only when absent.
    if truth_frame_column is None:
        truth_frame_column = ("rendered_frame" if "rendered_frame" in truth.columns
                              else "frame")
    if truth_frame_column not in truth.columns:
        raise MovieSimError(f"truth is missing {truth_frame_column!r}")

    n_truth = int(len(truth))
    n_det = int(len(det))
    matched, errors, pairs = 0, [], []
    for frame, tf in truth.groupby(truth_frame_column):
        df = det[det[frame_column] == frame]
        if df.empty:
            continue
        tx = tf["px"].to_numpy(float); ty = tf["py"].to_numpy(float)
        dx = df[x_column].to_numpy(float); dy = df[y_column].to_numpy(float)
        dist = np.hypot(tx[:, None] - dx[None, :], ty[:, None] - dy[None, :])
        used_t, used_d = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
        for i, j in order:
            if dist[i, j] > tolerance_px:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(int(i)); used_d.add(int(j))
            matched += 1
            errors.append(float(dist[i, j]))
            if with_pairs:
                pairs.append((tf.index[int(i)], df.index[int(j)]))

    out = {
        "n_truth": n_truth, "n_detected": n_det, "n_matched": matched,
        "recall": matched / n_truth if n_truth else float("nan"),
        "precision": matched / n_det if n_det else float("nan"),
        "median_error_px": float(np.median(errors)) if errors else float("nan"),
        "tolerance_px": float(tolerance_px),
    }
    if with_pairs:
        out["pairs"] = pairs
    return out


def recall_curve(truth, detections, *, quality_column: str, thresholds,
                 tolerance_px: float = 2.0, **match_kwargs):
    """Recall and precision as the detector's threshold is swept.

    One detection run, filtered: the same trick `trackmate_run.sweep_detection_threshold`
    verified exactly on real data: a threshold is a cut on a per-spot quality, so
    detecting once permissively and filtering reproduces any stricter run.

    This is the curve the purity-only sweep could never draw, because it needs the
    molecules the detector MISSED and only ground truth has those.
    """
    rows = []
    for value in thresholds:
        kept = detections[detections[quality_column] >= value]
        got = match_to_truth(truth, kept, tolerance_px=tolerance_px, **match_kwargs)
        got["threshold"] = float(value)
        rows.append(got)
    table = pd.DataFrame(rows)
    table.attrs["caveat"] = (
        "recall here is against molecules RENDERED INTO THE IMAGE, so it is bounded by "
        "the photon budget in PhotonSpec: calibrate that against a real movie before "
        "carrying any number across to real data")
    return table
