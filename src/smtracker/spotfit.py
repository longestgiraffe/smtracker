"""SPOT FIT: every detected spot refitted with a 2D Gaussian on its own frame, for its PER-SPOT PRECISION.

WHY (6 Nov 2026; asked: "is it possible in the tracking part for trackmate to make this an option to do the precision
etc?"). u-track fits each spot and writes its fit's formal uncertainty (``dx, dy``), which the per-spot precision fit
reads (`smtracker.precision`). TrackMate's LoG detector writes none, so a TrackMate project could not use it. Here each
TrackMate spot is fitted on the frame it was found in, in a (2h+1) x (2h+1) window centred on TrackMate's position:

    I(x, y) = B + A exp(-((x - x0)^2 + (y - y0)^2) / (2 s^2))

s the PSF width, GIVEN (from the optics, `optics.psf_sigma_nm`), as u-track fixes its psfSigma; x0, y0, A, B free;
least squares (Gauss-Newton, every spot of a frame at once). The precision is the fit's formal covariance,
s_res^2 (J^T J)^-1 for x0 and y0: the quantity u-track reports, and like u-track's a LOWER bound that reaches a fit
only through a calibration measured on immobile spots tracked the same way.

TRACKMATE'S POSITIONS ARE KEPT: the tracks are TrackMate's, and the fitted sigma is an index of how well each spot
could be placed. Kept beside the XML as ``<xml stem>.spotfit.csv`` (spot ID, frame, sigma_x_px, sigma_y_px and the
fit's own position for a check), which `trackmate.read_trackmate_xml` reads into ``dx, dy``.

Interfaces:
  fit_spots(stack, spots, psf_sigma_px, half_window=3)   per-spot fit of one movie's spots; ``stack(lo, hi)`` returns
                                                          frames as a (k, h, w) array (`trackclip.MovieFrames.read`)
  sidecar_path(xml), write_sidecar(xml, table), read_sidecar(xml)

Invariants: a spot whose fit fails is NaN, never a number (the analysis then refuses per-spot precision for its
condition, by name); coordinates in pixels with pixel centres at integers from 0 (TrackMate's origin); a window that
would cross the image edge is moved inside it, never cut.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SpotFitError", "fit_spots", "tracked_spots", "fit_xml", "fit_condition", "sidecar_path", "write_sidecar",
           "read_sidecar", "SIDECAR_SUFFIX"]

SIDECAR_SUFFIX = ".spotfit.csv"
#: Gauss-Newton iterations per frame's spots; converged long before on a spot of a few hundred photons.
_ITERATIONS = 12


class SpotFitError(RuntimeError):
    """Raised when spots cannot be fitted (no frames, malformed input)."""


def _fit_windows(windows, x0, y0, psf_sigma_px, gx, gy):
    """Gauss-Newton on N windows at once: (N, w, w) pixels, start (x0, y0) in window coordinates. Returns the fitted
    x0, y0 and their formal SDs (NaN where a fit fails)."""
    n, w, _ = windows.shape
    data = windows.reshape(n, -1)
    background = np.percentile(data, 20, axis=1)
    amplitude = np.maximum(data.max(axis=1) - background, 1e-6)
    theta = np.stack([x0, y0, amplitude, background], axis=1).astype(float)
    xs, ys = gx.ravel()[None, :], gy.ravel()[None, :]
    s2 = float(psf_sigma_px) ** 2
    ok = np.ones(n, dtype=bool)

    def evaluate(t):
        ddx, ddy = xs - t[:, 0:1], ys - t[:, 1:2]
        gg = np.exp(-(ddx ** 2 + ddy ** 2) / (2 * s2))
        res = data - (t[:, 3:4] + t[:, 2:3] * gg)
        jac = np.stack([t[:, 2:3] * gg * ddx / s2, t[:, 2:3] * gg * ddy / s2, gg, np.ones_like(gg)], axis=2)
        return res, jac

    # LEVENBERG-MARQUARDT, every spot its own damping: a step is taken only where it lowers that spot's residual
    resid, J = evaluate(theta)
    cost = np.sum(resid ** 2, axis=1)
    damping = np.full(n, 1e-3)
    eye = np.eye(4)[None]
    for _ in range(4 * _ITERATIONS):
        JtJ = np.einsum("npi,npj->nij", J, J)
        Jtr = np.einsum("npi,np->ni", J, resid)
        diag = np.einsum("nii->ni", JtJ)
        A = JtJ + damping[:, None, None] * (eye * diag[:, None, :] + 1e-12 * eye)
        try:
            step = np.linalg.solve(A, Jtr[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            ok[:] = False
            break
        step = np.where(np.isfinite(step), step, 0.0)
        trial = theta + step
        t_resid, t_J = evaluate(trial)
        t_cost = np.sum(t_resid ** 2, axis=1)
        better = t_cost < cost
        theta = np.where(better[:, None], trial, theta)
        resid = np.where(better[:, None], t_resid, resid)
        J = np.where(better[:, None, None], t_J, J)
        cost = np.where(better, t_cost, cost)
        damping = np.where(better, damping / 3.0, damping * 4.0)
        if np.all(np.abs(step[:, :2]).max(axis=1) < 1e-4):
            break
    dx, dy = xs - theta[:, 0:1], ys - theta[:, 1:2]
    g = np.exp(-(dx ** 2 + dy ** 2) / (2 * s2))
    resid = data - (theta[:, 3:4] + theta[:, 2:3] * g)
    J = np.stack([theta[:, 2:3] * g * dx / s2, theta[:, 2:3] * g * dy / s2, g, np.ones_like(g)], axis=2)
    JtJ = np.einsum("npi,npj->nij", J, J)
    rss = np.sum(resid ** 2, axis=1) / max(w * w - 4, 1)
    cov = np.full((n, 4, 4), np.nan)
    good = ok & (theta[:, 2] > 0) & np.isfinite(theta).all(axis=1)
    if good.any():
        try:
            cov[good] = np.linalg.inv(JtJ[good]) * rss[good][:, None, None]
        except np.linalg.LinAlgError:
            good[:] = False
    sx, sy = np.sqrt(cov[:, 0, 0]), np.sqrt(cov[:, 1, 1])
    inside = (np.abs(theta[:, 0] - x0) <= 1.5) & (np.abs(theta[:, 1] - y0) <= 1.5)   # stayed on its spot
    bad = ~(good & inside & np.isfinite(sx) & np.isfinite(sy) & (sx > 0) & (sy > 0))
    sx[bad], sy[bad] = np.nan, np.nan
    theta[bad, :2] = np.nan                                     # a failed fit has no position either
    return theta[:, 0], theta[:, 1], sx, sy


def fit_spots(stack, spots: pd.DataFrame, *, psf_sigma_px: float, half_window: int = 3,
              block: int = 100) -> pd.DataFrame:
    """Fit every spot of one movie on its own frame.

    ``stack(lo, hi)`` returns tracker frames lo..hi (1-based, inclusive) as a (k, h, w) array; ``spots`` has ``frame``
    (the tracker's, 1-based), ``x``, ``y`` (pixels, centres at integers from 0) and optionally ``spot_id``. Returns
    ``spots`` with ``sigma_x_px``, ``sigma_y_px`` (the fit's formal SD; NaN where it failed) and ``fit_x``, ``fit_y``.
    Frames are read ``block`` at a time."""
    if not (np.isfinite(psf_sigma_px) and psf_sigma_px > 0):
        raise SpotFitError(f"the PSF width must be a positive number of pixels, got {psf_sigma_px}")
    for column in ("frame", "x", "y"):
        if column not in spots.columns:
            raise SpotFitError(f"spots need a {column!r} column")
    out = spots.copy().reset_index(drop=True)
    for column in ("sigma_x_px", "sigma_y_px", "fit_x", "fit_y"):
        out[column] = np.nan
    if len(out) == 0:
        return out
    w = 2 * int(half_window) + 1
    gy, gx = np.mgrid[0:w, 0:w].astype(float)
    frames = out["frame"].to_numpy(dtype=int)
    first, last = int(frames.min()), int(frames.max())
    for lo in range(first, last + 1, block):
        hi = min(lo + block - 1, last)
        images = np.asarray(stack(lo, hi), dtype=float)
        height, width = images.shape[1:]
        if height < w or width < w:
            raise SpotFitError(f"frames of {width} x {height} px are smaller than the {w} px window")
        for f in range(lo, hi + 1):
            rows = np.flatnonzero(frames == f)
            if len(rows) == 0:
                continue
            image = images[f - lo]
            cx = np.clip(np.rint(out.loc[rows, "x"].to_numpy(dtype=float)).astype(int), half_window,
                         width - 1 - half_window)
            cy = np.clip(np.rint(out.loc[rows, "y"].to_numpy(dtype=float)).astype(int), half_window,
                         height - 1 - half_window)
            windows = np.stack([image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                                for x, y in zip(cx, cy)])
            x0 = out.loc[rows, "x"].to_numpy(dtype=float) - (cx - half_window)
            y0 = out.loc[rows, "y"].to_numpy(dtype=float) - (cy - half_window)
            fx, fy, sx, sy = _fit_windows(windows, x0, y0, psf_sigma_px, gx, gy)
            out.loc[rows, "fit_x"] = fx + (cx - half_window)
            out.loc[rows, "fit_y"] = fy + (cy - half_window)
            out.loc[rows, "sigma_x_px"] = sx
            out.loc[rows, "sigma_y_px"] = sy
    return out


def tracked_spots(xml) -> pd.DataFrame:
    """The spots of a TrackMate XML's kept tracks: ``spot_id``, ``frame`` (the tracker's, 1-based), ``x``, ``y`` in
    PIXELS (centres at integers from 0, TrackMate's origin), as `trackmate.read_trackmate_xml` reads them."""
    from . import trackmate as tm

    root = tm._parse(xml)
    model = root.find("Model")
    if model is None:
        raise SpotFitError(f"{xml}: no <Model> element: is this a TrackMate XML?")
    scale = tm._pixels_per_unit(tm.read_trackmate_metadata(xml), None, xml)
    spots = tm._read_spots(model, xml)
    kept = {spot_id for edges in tm._read_tracks(model, xml, filtered_only=True).values()
            for chain in tm._linear_segments(edges) for spot_id in chain}
    rows = [(spot_id, frame, x * scale, y * scale) for spot_id, (frame, x, y, _amp, _extras) in spots.items()
            if spot_id in kept]
    return pd.DataFrame(rows, columns=["spot_id", "frame", "x", "y"])


def fit_xml(xml, *, movie_root, movie: str, psf_sigma_px: float, half_window: int = 3,
            check_alignment: bool = True) -> dict:
    """Fit the tracked spots of one TrackMate XML on the movie's recorded frames (`trackclip.open_movie`, which
    checks that the spots are brightest on their own frames) and write the result beside the XML. Returns a summary:
    spots, fitted, the median sigma in pixels, the frames' source.

    ``check_alignment`` False skips that check, ONLY for immobile molecules (a stationary set): one is as bright on
    its neighbouring frames, and bleaching makes the earlier ones brighter, so the check fails by chance there (1 of
    the stationary set's 10 movies, 6 Nov 2026) while it decides nothing about the frames' order."""
    from .trackclip import open_movie

    spots = tracked_spots(xml)
    if spots.empty:
        raise SpotFitError(f"{xml}: no tracked spots to fit")
    frames = open_movie(movie_root, movie,
                        spots=(list(spots[["frame", "x", "y"]].itertuples(index=False, name=None))
                               if check_alignment else None))
    fitted = fit_spots(frames.read, spots, psf_sigma_px=psf_sigma_px, half_window=half_window)
    write_sidecar(xml, fitted, psf_sigma_px=psf_sigma_px)
    ok = fitted["sigma_x_px"].notna()
    return {"xml": str(xml), "spots": int(len(fitted)), "fitted": int(ok.sum()),
            "median_sigma_px": float(np.sqrt((fitted.loc[ok, "sigma_x_px"] ** 2 + fitted.loc[ok, "sigma_y_px"] ** 2)
                                             / 2).median()) if ok.any() else float("nan"),
            "frames": frames.words}


def fit_condition(condition_dir, *, psf_sigma_px: float, on_progress=None, replace: bool = False,
                  check_alignment: bool = True) -> list:
    """Every TrackMate XML of a condition (``<condition>/trackmate/*.xml``) fitted, each into its own sidecar; an
    XML whose sidecar exists is kept unless ``replace``. Returns one summary per XML (an error kept as its row)."""
    condition_dir = Path(condition_dir)
    xmls = sorted((condition_dir / "trackmate").glob("*.xml"))
    if not xmls:
        raise SpotFitError(f"{condition_dir}: no TrackMate XML in trackmate/")
    rows = []
    for index, xml in enumerate(xmls, start=1):
        if sidecar_path(xml).is_file() and not replace:
            rows.append({"xml": str(xml), "skipped": "already fitted"})
        else:
            try:
                rows.append(fit_xml(xml, movie_root=condition_dir, movie=xml.stem, psf_sigma_px=psf_sigma_px,
                                    check_alignment=check_alignment))
            except Exception as exc:                       # noqa: BLE001 - kept and named, as a batch keeps errors
                rows.append({"xml": str(xml), "error": f"{type(exc).__name__}: {exc}"})
        if on_progress is not None:
            on_progress(index, len(xmls), rows[-1])
    return rows


def sidecar_path(xml) -> Path:
    xml = Path(xml)
    return xml.with_name(xml.stem + SIDECAR_SUFFIX)


def write_sidecar(xml, table: pd.DataFrame, *, psf_sigma_px: float) -> Path:
    """Each spot's fitted precision beside its TrackMate XML (`read_sidecar` reads it back)."""
    if "spot_id" not in table.columns:
        raise SpotFitError("the spot table needs TrackMate's spot_id to be read back with its XML")
    path = sidecar_path(xml)
    columns = ["spot_id", "frame", "x", "y", "sigma_x_px", "sigma_y_px", "fit_x", "fit_y"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# smtracker spot fit 1; psf_sigma_px {float(psf_sigma_px):.6g}; positions are TrackMate's, "
                     f"sigma is each spot's 2D Gaussian fit's formal SD (a lower bound; calibrate before use)\n")
        table[columns].to_csv(handle, index=False, float_format="%.6g")
    return path


def read_sidecar(xml) -> "pd.DataFrame | None":
    """The spot fit beside ``xml``, or None where there is none."""
    path = sidecar_path(xml)
    if not path.is_file():
        return None
    table = pd.read_csv(path, comment="#", dtype={"spot_id": str})
    missing = [c for c in ("spot_id", "sigma_x_px", "sigma_y_px") if c not in table.columns]
    if missing:
        raise SpotFitError(f"{path} is not a spot fit (missing {missing})")
    return table
