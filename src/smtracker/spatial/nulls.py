"""Matched null tables for ``smtracker.spatial``, same four files, same schema,
different ``source``, so a downstream script can ``rbind`` observed + nulls and get
its baseline by grouping. No analysis has to build its own null.

``null_uniform``: randomise WHERE each track is, keep WHAT it is: keep the track's
displacement sequence exactly and place it, rigidly, at a uniform position +
orientation in its own cell's 3-D volume (rejection-tested against a drawn ``z``),
then project to ``(u, v)``. Length, internal correlation and apparent D are
preserved (so ``population`` is unchanged); only location is destroyed. Known bias,
stated honestly: rejection under-populates the CAPS for long/fast tracks, because a
long track physically cannot sit near a pole: a real effect, which is exactly why
``null_kinetic`` (P6) also exists.

``null_kinetic``: a confined two-state reflecting-Brownian simulation (P6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as _g
from .coords import cell_to_lab, to_cell_frame

__all__ = ["NullError", "null_uniform", "reflect_into_spherocylinder",
           "simulate_positions", "null_kinetic"]


class NullError(RuntimeError):
    """Raised when a null cannot be formed (a track that fits no placement, etc.)."""


def _sample_uniform_3d(n, L, W, rng):
    """``n`` points (u, y, z) uniform in the spherocylinder VOLUME (rejection)."""
    R = W / 2.0
    us, ys, zs = [], [], []
    have = 0
    while have < n:
        m = 2 * (n - have) + 16
        u = rng.uniform(-L / 2, L / 2, m)
        y = rng.uniform(-R, R, m)
        z = rng.uniform(-R, R, m)
        keep = y ** 2 + z ** 2 <= _g.half_width(u, L, W) ** 2
        us.append(u[keep]); ys.append(y[keep]); zs.append(z[keep])
        have += int(keep.sum())
    return (np.concatenate(us)[:n], np.concatenate(ys)[:n], np.concatenate(zs)[:n])


def _place_track_uniform(ou, ov, L, W, rng, *, batch=64, max_batches=64):
    """Place a rigid 2-D track (centroid offsets ``ou, ov``) at a uniform 3-D
    position + 2-D orientation, rejecting if any localisation leaves the volume.

    Returns placed ``(u, v)`` arrays, or ``None`` if no placement was found (a
    track longer than the cell can hold near a pole, the documented cap bias).
    """
    for _ in range(max_batches):
        au, ay, az = _sample_uniform_3d(batch, L, W, rng)
        theta = rng.uniform(0, 2 * np.pi, batch)
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        pu = au[:, None] + ou[None, :] * c - ov[None, :] * s
        pv = ay[:, None] + ou[None, :] * s + ov[None, :] * c
        hh = _g.half_width(pu, L, W)
        inside = (np.abs(pu) <= L / 2) & (pv ** 2 + az[:, None] ** 2 <= hh ** 2)
        ok = inside.all(axis=1)
        if ok.any():
            k = int(np.argmax(ok))
            return pu[k], pv[k]
    return None


def _one_uniform_rep(coords, cells_idx, rng):
    """One null_uniform replicate as a LAB-frame localisation table (to be fed
    through ``build_tables`` so every derived column is recomputed consistently)."""
    rows = []
    for tid, tr in coords.groupby("track_id", sort=False):
        L = float(tr["cell_length_um"].iloc[0])
        W = float(tr["cell_width_um"].iloc[0])
        u = tr["u_um"].to_numpy()
        v = tr["v_um"].to_numpy()
        ou, ov = u - u.mean(), v - v.mean()               # rigid shape (preserves D)
        placed = _place_track_uniform(ou, ov, L, W, rng)
        if placed is None:
            continue                                       # cap bias: track dropped
        pu, pv = placed
        cid = tr["cell_id"].iloc[0]
        cx, cy, ang = cells_idx.loc[cid, ["centroid_x", "centroid_y", "angle"]]
        x, y = cell_to_lab(pu, pv, cx, cy, ang)            # angle already in table units
        base = {c: tr[c].iloc[0] for c in
                ("cell_id", "enzyme", "condition", "replicate", "session")}
        for i, f in enumerate(tr["frame"].to_numpy()):
            rows.append({"track_id": tid, "frame": int(f), "x_um": x[i], "y_um": y[i],
                         **base})
    return pd.DataFrame(rows)


def null_uniform(locs: pd.DataFrame, cells: pd.DataFrame, *, dataset: str,
                 sigma_um: float, tau_s: float, edges=(0.0, 0.15, np.inf),
                 min_localisations: int = 5, angle_units: str = "rad",
                 n_rep: int = 20, seed: int = 0) -> dict:
    """``n_rep`` uniform-placement null replicates → the four schema tables with
    ``source='null_uniform'`` and a ``null_rep`` column (0…n_rep−1).

    Randomises WHERE each track is (uniform in its cell's 3-D volume, rejection
    against a drawn z) and keeps WHAT it is (its displacement sequence, so length,
    correlation and apparent D (hence ``population``) are preserved). Bias:
    rejection under-populates the caps for long/fast tracks (documented, real).
    """
    from .io import build_tables

    coords = to_cell_frame(locs, cells, angle_units=angle_units)
    # the null's own angle convention: build_tables receives lab coords + the SAME
    # cell table, so we place with the cell's stated angle_units throughout.
    cells_idx = cells.set_index("cell_id")
    per_kind = {k: [] for k in ("localisations", "steps", "tracks", "cells")}
    counts = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep)
        null_locs = _one_uniform_rep(coords, cells_idx, rng)
        if null_locs.empty:
            continue
        tab = build_tables(null_locs, cells, dataset=dataset, sigma_um=sigma_um,
                           tau_s=tau_s, edges=edges,
                           min_localisations=min_localisations,
                           angle_units=angle_units, source="null_uniform",
                           extra_columns={"null_rep": rep})
        counts.append(tab.pop("_counts"))
        for k in per_kind:
            per_kind[k].append(tab[k])
    out = {k: pd.concat(v, ignore_index=True) for k, v in per_kind.items()}
    out["_counts"] = {"n_rep": n_rep, "per_rep": counts,
                      "note": "cap under-population for long/fast tracks is a real "
                              "placement effect, not only an artefact"}
    return out


# --------------------------------------------------------------------------- #
# null_kinetic, confined two-state reflecting-Brownian simulation
# --------------------------------------------------------------------------- #

_MAX_MICRO_UM = 0.025


def reflect_into_spherocylinder(pos: np.ndarray, L: float, W: float,
                                max_iter: int = 40) -> np.ndarray:
    """Reflect 3-D points ``(u, y, z)`` back inside the spherocylinder, ITERATED to
    convergence (not once). Radial reflection in the cylinder body, spherical
    reflection about the nearest cap centre in the caps. Vectorised over rows.

    A single reflection can overshoot for a point far outside; iterating is what
    makes the stationary density of reflecting Brownian motion exactly uniform (the
    invariant test 4 catches a one-shot bug).
    """
    R = W / 2.0
    a = L / 2.0 - R
    pos = np.array(pos, dtype=float, copy=True)
    for _ in range(max_iter):
        u, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        au = np.abs(u)
        body = au <= a
        moved = False
        # cylinder body: radial reflection in (y, z)
        r = np.sqrt(y ** 2 + z ** 2)
        over = body & (r > R) & (r > 0)
        if over.any():
            f = (2 * R - r[over]) / r[over]
            pos[over, 1] *= f
            pos[over, 2] *= f
            moved = True
        # caps: spherical reflection about the nearest cap centre (±a, 0, 0)
        cap = ~body
        cu = np.sign(u) * a
        du = u - cu
        d = np.sqrt(du ** 2 + y ** 2 + z ** 2)
        over_c = cap & (d > R) & (d > 0)
        if over_c.any():
            f = (2 * R - d[over_c]) / d[over_c]
            pos[over_c, 0] = cu[over_c] + du[over_c] * f
            pos[over_c, 1] = y[over_c] * f
            pos[over_c, 2] = z[over_c] * f
            moved = True
        if not moved:
            break
    return pos


def simulate_positions(n_tracks: int, n_frames, L: float, W: float, *,
                       D=(0.057, 1.04), k_on: float = 0.0, k_off: float = 0.0,
                       tau_s: float, rng, max_micro_um: float = _MAX_MICRO_UM):
    """Confined two-state reflecting-Brownian tracks in a spherocylinder.

    Returns ``(u, v)`` arrays, shape ``(n_tracks, max_frames)`` with ``np.nan`` past
    each track's own length when ``n_frames`` is an array. Initial positions are
    rejection-sampled UNIFORM in the 3-D volume (NOT Gaussian about the axis, a
    Gaussian initialiser invents a membrane depletion). Motion is sub-stepped so each
    micro-step is ≤ ``max_micro_um``; the boundary is reflected to convergence; the
    two states (``D`` slow/fast) switch at spatially UNIFORM per-frame rates set by
    ``k_on`` (→slow) and ``k_off`` (→fast). ``v`` is the projection (z dropped).
    """
    D = np.asarray(D, dtype=float)
    lengths = np.broadcast_to(np.asarray(n_frames), (n_tracks,)).astype(int)
    max_f = int(lengths.max())
    n_sub = max(1, int(np.ceil(tau_s / (max_micro_um ** 2 / (2.0 * D.max())))))
    dt = tau_s / n_sub
    sd = np.sqrt(2.0 * D * dt)                              # per-state micro-step SD

    ux, uy, uz = _sample_uniform_3d(n_tracks, L, W, rng)
    pos = np.stack([ux, uy, uz], axis=1)
    p_stat_slow = k_on / (k_on + k_off) if (k_on + k_off) > 0 else 0.5
    state = (rng.random(n_tracks) >= p_stat_slow).astype(int)   # 0 slow, 1 fast
    p_off = 1.0 - np.exp(-k_off * dt)                      # slow → fast
    p_on = 1.0 - np.exp(-k_on * dt)                        # fast → slow

    out_u = np.full((n_tracks, max_f), np.nan)
    out_v = np.full((n_tracks, max_f), np.nan)
    for f in range(max_f):
        for _ in range(n_sub):
            step = rng.standard_normal((n_tracks, 3)) * sd[state][:, None]
            pos = reflect_into_spherocylinder(pos + step, L, W)
            r = rng.random(n_tracks)
            to_fast = (state == 0) & (r < p_off)
            to_slow = (state == 1) & (r < p_on)
            state = np.where(to_fast, 1, np.where(to_slow, 0, state))
        live = f < lengths
        out_u[live, f] = pos[live, 0]
        out_v[live, f] = pos[live, 1]
    return out_u, out_v


def null_kinetic(cells: pd.DataFrame, template_tracks: pd.DataFrame, *,
                 dataset: str, sigma_um: float, tau_s: float,
                 D=(0.057, 1.04), k_on: float | None = None,
                 k_off: float | None = None,
                 edges=(0.0, 0.15, np.inf), min_localisations: int = 5,
                 angle_units: str = "rad", n_rep: int = 20, seed: int = 0,
                 link_radius_um: float | None = None) -> dict:
    """Confined two-state null: the four schema tables with ``source='null_kinetic'``.

    ``k_on`` / ``k_off`` are REQUIRED and must be the MEASURED rates for the condition
    being nulled. They have no defaults on purpose: a null built on another
    condition's rates encodes the wrong confinement, and the observed-minus-null
    residual is then not interpretable. Two conditions can switch at very different
    rates, so one shared pair cannot serve both.

    For each real cell, simulate its real track COUNT and track-LENGTH distribution
    (from ``template_tracks``) as confined reflecting-Brownian two-state motion
    (:func:`simulate_positions`), add ``sigma_um`` noise, then run the SAME
    track-length filter, D estimator and ``edges``. This null exists because
    confinement in a hemispherical cap lowers a fast molecule's APPARENT D below the
    threshold, so it is labelled low-D near the poles with no biology involved: the
    artefact ``null_uniform`` cannot show because it preserves each track's D.

    ``link_radius_um`` (None: off, the null as AB3 ran it) BREAKS a track wherever a
    noisy localisation lies further than the tracker's linking radius from the one
    before (u-track at 5 px x 106 nm = 0.53 um), and each piece then meets the same
    ``min_localisations`` filter. Without it the null keeps every step whatever its
    length, so it reproduces the GEOMETRY of confinement but not the SELECTION a
    tracker adds to it: near a wall a step is reflected and short, so it survives the
    cut more often than one in the middle, and a free protein's tracked steps are
    over-represented at the periphery and read slower there (asked 30 Oct 2026). It
    draws nothing from ``rng``, so the positions are the same with and without it.
    """
    from .io import build_tables

    if k_on is None or k_off is None:
        raise NullError(
            "null_kinetic requires the measured per-condition k_on and k_off (1/s); "
            "they have no defaults. A null built on another condition's rates encodes "
            "the wrong confinement and its residual cannot be interpreted.")

    # per cell: real track count + the list of its track lengths
    tt = template_tracks
    lengths_by_cell = (tt.groupby(["cell_id", "track_id"]).size()
                       .groupby("cell_id").apply(list).to_dict())
    ident = {c: (tt[c].iloc[0] if tt[c].nunique() == 1 else None)
             for c in ("enzyme", "condition", "replicate", "session")}
    cells_idx = cells.set_index("cell_id")

    per_kind = {k: [] for k in ("localisations", "steps", "tracks", "cells")}
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + 7919 * (rep + 1))
        rows, tid = [], 0
        for cid, lens in lengths_by_cell.items():
            if cid not in cells_idx.index:
                continue
            c = cells_idx.loc[cid]
            L, W = float(c["length_um"]), float(c["width_um"])
            lens = np.array(lens)
            u, v = simulate_positions(len(lens), lens, L, W, D=D, k_on=k_on,
                                      k_off=k_off, tau_s=tau_s, rng=rng)
            u = u + rng.normal(0, sigma_um, u.shape)       # localisation noise
            v = v + rng.normal(0, sigma_um, v.shape)
            x, y = cell_to_lab(u, v, c["centroid_x"], c["centroid_y"], c["angle"])
            for t in range(len(lens)):
                tid += 1
                for f in range(int(lens[t])):
                    if np.isnan(u[t, f]):
                        continue
                    if (link_radius_um is not None and f and not np.isnan(u[t, f - 1])
                            and np.hypot(u[t, f] - u[t, f - 1], v[t, f] - v[t, f - 1]) > link_radius_um):
                        tid += 1                                # the tracker breaks the track here
                    rows.append({"track_id": tid, "frame": f, "x_um": x[t, f],
                                 "y_um": y[t, f], "cell_id": cid,
                                 "enzyme": ident["enzyme"], "condition": ident["condition"],
                                 "replicate": ident["replicate"], "session": ident["session"]})
        null_locs = pd.DataFrame(rows)
        if null_locs.empty:
            continue
        tab = build_tables(null_locs, cells, dataset=dataset, sigma_um=sigma_um,
                           tau_s=tau_s, edges=edges, min_localisations=min_localisations,
                           angle_units=angle_units, source="null_kinetic",
                           extra_columns={"null_rep": rep})
        tab.pop("_counts")
        for kk in per_kind:
            per_kind[kk].append(tab[kk])
    out = {k: pd.concat(v, ignore_index=True) for k, v in per_kind.items()}
    out["_counts"] = {"n_rep": n_rep, "D": list(D), "k_on": k_on, "k_off": k_off,
                      "link_radius_um": link_radius_um}
    return out
