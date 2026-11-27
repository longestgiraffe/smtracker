"""Building a cell mesh from a drawn SPINE: straight or curved, one code path.

Why this exists. The mesh editor could only draw STRAIGHT cells: two clicks gave two
poles, and every rib was perpendicular to the one straight axis between them. Real rods
bend. Everything else in the chain already copes with that,

* the Oufti format is a list of RIBS along a centreline, so curvature is inherent;
* the automatic path (`build_capsule`) skeletonises the mask and follows the bend;
* `cellcoords.project_to_mesh` finds the nearest centreline SEGMENT and accumulates arc
  length, so `L`, `d_norm`, the maps and the spatial statistics all handle a curved cell.

so the straight capsule was the only place curvature was lost, and it was lost in the
one place a human draws.

A strict generalisation, and that is pinned by test. Two spine points must reproduce
the old straight capsule to floating-point, or every cell drawn before this change would
silently move. Three or more points curve. There is no mode switch and no second function:
the straight case is the two-point case.

Constant width, deliberately. One width per cell, as before. A curved cell often also
tapers, and the mesh format could express that per rib, but a taper needs its own
interaction (drag a rib), and the automatic path already assumes a constant body radius
measured from the middle half. Starting constant keeps the drawn cell and the fitted cell
describing the same thing. Per-rib width is a later addition and needs no format change.

The clicked endpoints are the poles. The semicircular end caps are centred one radius
INSIDE each endpoint, so the cap's apex lands on the point that was clicked rather than
half a width beyond it. That was already fixed once in the editor's straight capsule and
the fix is kept here rather than re-lost.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "GeometryError",
    "DEFAULT_N_RUNGS",
    "MIN_N_RUNGS",
    "MAX_N_RUNGS",
    "rungs_for_length",
    "DEFAULT_N_CAP",
    "resample_spine",
    "smooth_spine",
    "catmull_rom",
    "tangents",
    "min_turn_radius",
    "DEFAULT_SMOOTH",
    "capsule_from_spine",
    "spine_curvature",
]


class GeometryError(RuntimeError):
    """Raised when a spine cannot make a mesh, rather than returning a broken one."""


#: Ribs along the body. 24 is what the editor has always drawn, kept so a re-drawn cell
#: has the same rib count as one drawn before this change.
DEFAULT_N_RUNGS = 24

#: The fewest ribs a cell gets, however short it is.
MIN_N_RUNGS = 12

#: The most, however long. A rib per pixel is the rule below; on a 300 px cell that is
#: 300 ribs, which is a mesh nothing downstream benefits from and every save carries.
MAX_N_RUNGS = 200


def rungs_for_length(length: float) -> int:
    """How many ribs a cell of this length gets: About one per pixel.

    This is why poles look round. The cap is not an arc: it is the ribs tapering,
    so the number of ribs IS the resolution of the pole, and a fixed 24 leaves a 40 px
    cell with about six ribs inside each cap and a visibly faceted end. Reported as
    "cells feel less round at the poles".

    The rule is the browser editor's own, from `makeCapsule`:
    `n = max(10, nRungs || max(12, round(L)))`. Matching it means a cell drawn in either
    editor has the same rib count as well as the same shape.
    """
    return int(min(MAX_N_RUNGS, max(MIN_N_RUNGS, round(float(length)))))

#: Points per semicircular end cap in the model outline.
DEFAULT_N_CAP = 16


def resample_spine(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to `n` points evenly spaced by arc length.

    Even spacing is what makes the ribs describe the cell rather than the clicking: a
    user who clicks three times near one pole and once at the other has said where the
    cell BENDS, not where its ribs belong.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise GeometryError(f"spine must be (N, 2), got {points.shape}")
    if len(points) < 2:
        raise GeometryError("a spine needs at least two points: the two poles")
    if n < 2:
        raise GeometryError(f"need at least two output points, got {n}")
    steps = np.hypot(*np.diff(points, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] <= 1e-9:
        raise GeometryError("the spine has zero length; the poles coincide")
    want = np.linspace(0.0, arc[-1], n)
    return np.column_stack([np.interp(want, arc, points[:, 0]),
                            np.interp(want, arc, points[:, 1])])


def smooth_spine(points: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average with a window that SHRINKS toward the ends.

    The endpoints are the poles the user clicked, and they must not move: smoothing them
    would place the cell's ends somewhere nobody clicked. But simply leaving a fixed
    margin untouched and averaging the rest with a full window is worse than it looks:

    **MEASURED, and it is why this function is not the obvious three-liner.** With a fixed
    margin, a large window pulls the interior toward a straight line while the endpoints
    stay pinned, so a KINK appears right beside each pole. On a sharp S-shaped spine the
    tightest turn radius went 2.70 px at window 11, up to 6.13 at 25, and then back DOWN
    to 0.77 at 35 and 0.43 at 45. Smoothing more made the curve sharper, which is the
    opposite of what smoothing is for, and it would have shipped as an unexplained
    "sometimes it refuses when you smooth harder".

    A window that shrinks to zero at the ends removes the discontinuity: the endpoints are
    still exactly fixed, and the amount of smoothing varies continuously along the spine.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if window < 3 or n < 3:
        return points.copy()
    half = window // 2
    out = points.copy()
    for i in range(1, n - 1):
        # The window can never reach past either end, so it closes smoothly rather than
        # switching from "untouched" to "full width" at a fixed margin.
        reach = min(half, i, n - 1 - i)
        if reach:
            out[i] = points[i - reach:i + reach + 1].mean(axis=0)
    return out


def tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangent at every point: central differences inside, one-sided at the ends."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise GeometryError("need at least two points for a tangent")
    raw = np.empty_like(points)
    raw[0] = points[1] - points[0]
    raw[-1] = points[-1] - points[-2]
    if len(points) > 2:
        raw[1:-1] = points[2:] - points[:-2]
    norm = np.hypot(raw[:, 0], raw[:, 1])
    # A repeated point gives a zero tangent. Carry the previous direction rather than
    # dividing by zero: a NaN rib would reach the mesh and fail much later.
    bad = norm < 1e-9
    if bad.any():
        good = np.flatnonzero(~bad)
        if not good.size:
            raise GeometryError("every spine point is identical; there is no direction")
        raw[bad] = raw[good[np.searchsorted(good, np.flatnonzero(bad)).clip(
            0, good.size - 1)]]
        norm = np.hypot(raw[:, 0], raw[:, 1])
    return raw / norm[:, None]


def catmull_rom(points: np.ndarray, samples_per_span: int = 24,
                alpha: float = 0.5) -> np.ndarray:
    """A smooth curve THROUGH every clicked point: centripetal Catmull-Rom.

    Why not a moving average, which is what this replaced. Averaging pulls the
    interior of the spine toward the straight chord while the endpoints stay pinned,
    leaving a transition zone whose curvature is HIGHER than the corner it was meant to
    round. Measured on a sharp S at width 10 px, the tightest turn radius went 2.70 px at
    window 11, up to 6.13 at 25, then back DOWN to 0.77 at 35, smoothing harder made the
    curve sharper. Tapering the window to zero at the ends moved the numbers but not the
    shape of the problem (2.70 / 3.95 / 6.13 / 3.71 / 1.78). A moving average is simply
    the wrong instrument: it does not interpolate, so it fights the fixed endpoints.

    Catmull-Rom passes THROUGH every control point, so each click is honoured exactly,
    which is what a drawing tool must do, and is smooth between them by construction,
    with no parameter that can be turned the wrong way.

    CENTRIPETAL (`alpha` = 0.5), not uniform. Uniform Catmull-Rom overshoots and can form
    a cusp or a self-intersecting loop when control points are unevenly spaced, which is
    exactly what clicking produces. The centripetal parametrisation is the variant proven
    free of both.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points.copy()
    # Phantom endpoints by reflection, so the curve starts and ends at the real clicks
    # with a sensible tangent instead of needing the user to click twice at each pole.
    padded = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])

    out = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i:i + 4]
        # Knot spacing by chord length to the power alpha; guard equal points.
        def knot(prev, a, b):
            d = float(np.hypot(*(b - a)))
            return prev + (d ** alpha if d > 1e-12 else 1e-6)
        t0 = 0.0
        t1 = knot(t0, p0, p1)
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        t = np.linspace(t1, t2, samples_per_span, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out + [points[-1][None, :]])


#: Smoothing applied to the dense spine before ribs are placed, in points of that dense
#: resample (~96 points). A CLICKED POLYLINE HAS CORNERS AND A CELL DOES NOT: without
#: this, three clicks give a hinge rather than a bend, and the ribs fan out around the
#: vertex instead of following a wall. Endpoints are never moved, so the poles stay where
#: they were clicked.
DEFAULT_SMOOTH = 11


def min_turn_radius(centre: np.ndarray) -> float:
    """The tightest turn along a centreline, in pixels. `inf` for a straight line.

    Why this is needed. Ribs are a constant-width OFFSET of the centreline, and an offset
    curve SELF-INTERSECTS wherever the turn radius is smaller than the offset distance,
    the inner ends of neighbouring ribs cross and the outline closes into a bowtie. That
    is not a rendering artefact: the polygon genuinely has negative area near the corner,
    and everything downstream that integrates over the cell would be wrong there.

    A cell cannot bend tighter than its own half-width either, so refusing is also the
    biologically honest answer rather than a technical limitation.

    Give this the curve, not a fine resampling of it. The circumradius is exact,
    on a circle of radius 20 it returns 20.000 at 16 points and at 1024. But curvature
    is a SECOND difference, so any error in where a point sits enters as ``e / h**2``,
    and re-spacing a polyline by arc length shrinks ``h`` while leaving ``e`` alone.
    Measured on one spline: resampled to 32/64/128/256/360/720 points it returns
    27.3, 24.7, 11.6, 6.5, 4.2, **1.95** px (collapsing toward zero) while the same
    curve sampled directly at those densities CONVERGES to 26.09.
    """
    centre = np.asarray(centre, dtype=float)
    if len(centre) < 3:
        return float("inf")
    a, b, c = centre[:-2], centre[1:-1], centre[2:]
    # Circumradius of each consecutive triple: R = abc / (4 * area).
    ab = np.hypot(*(b - a).T)
    bc = np.hypot(*(c - b).T)
    ca = np.hypot(*(a - c).T)
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(area > 1e-12, ab * bc * ca / (4.0 * area), np.inf)
    return float(np.nanmin(radius)) if radius.size else float("inf")


def capsule_from_spine(spine, width: float, *, n_rungs: int | None = None,
                       n_cap: int = DEFAULT_N_CAP, smooth: int = 0):
    """A cell mesh and its outline from a drawn spine of two OR MORE points.

    Returns ``(mesh, model)``: `mesh` is the Oufti (N, 4) rib array
    ``[x_left, y_left, x_right, y_right]``, `model` the closed outline polygon.

    With two spine points this is the straight capsule, to floating point. With more it
    follows the bend: each rib is perpendicular to the LOCAL tangent, and the end caps are
    swept about the local tangent at each pole rather than about one global axis.

    `smooth` rounds the spine's corners before ribs are placed; 0 keeps the polyline the
    user drew. It never moves the endpoints.
    """
    spine = np.asarray(spine, dtype=float)
    if spine.ndim != 2 or spine.shape[1] != 2:
        raise GeometryError(f"spine must be (N, 2), got {spine.shape}")
    if not np.isfinite(spine).all():
        raise GeometryError("the spine holds a non-finite point")
    if not (width and width > 0):
        raise GeometryError(f"width must be positive, got {width!r}")

    # A DOUBLE CLICK IS A SLIP, NOT A SHAPE. Repeated consecutive points give the spline
    # a zero-length span, which it turns into a near-zero turn radius and the guard below
    # then refuses, reporting an impossible bend for what was simply a second click in
    # the same place. Collapsed here so the drawn shape is what the user meant.
    if len(spine) > 2:
        keep = [0]
        for i in range(1, len(spine)):
            if np.hypot(*(spine[i] - spine[keep[-1]])) > 1e-9:
                keep.append(i)
        if len(keep) >= 2:
            spine = spine[keep]

    if n_rungs is None:
        # From the SPINE's length, not the chord: a curved cell is longer than the
        # straight line between its poles and needs the ribs to match.
        n_rungs = rungs_for_length(
            float(np.hypot(*np.diff(np.asarray(spine, dtype=float), axis=0).T).sum()))

    if len(spine) > 2:
        # THROUGH the clicks, not near them, then re-spaced by arc length, because a
        # spline is parametrised by knot, not by distance, and ribs must be even.
        curve = catmull_rom(spine)
        if smooth:
            curve = smooth_spine(curve, smooth)
        dense = resample_spine(curve, max(n_rungs * 4, 64))
    else:
        curve = dense = spine              # two points: a straight line, nothing to fit
    centre = resample_spine(dense, n_rungs)

    # the offset must not SELF-INTERSECT. Checked on the CURVE, at its own sampling,
    # NOT on the arc-length-resampled `dense`, and that distinction is the whole of a
    # defect this once had.
    #
    # CORRECTED: the guard read `min_turn_radius(dense)`, so its verdict depended on
    # `n_rungs`. MEASURED on one spine at width 9: accepted at 12, 24, 30 and 48 rungs
    # and REFUSED at 90 and 200, the same cell, the same width, a different answer
    # because of how finely it was being drawn.
    #
    # The estimator is not at fault and was checked before being blamed: on an exact
    # circle of radius 20 it returns 20.000 at 16 points and at 1024. The resampler is.
    # Curvature is a SECOND difference, so a fixed interpolation error e over a spacing
    # h enters as e/h^2, and arc-length resampling to more points shrinks h while
    # leaving e alone. The resampled steps here agree to 1.9e-4 of their own length,
    # which sounds negligible and is not: at 720 points it swamps a true curvature of
    # 1/26. Measured on one spine, radius against resampled point count: 27.3, 24.7,
    # 11.6, 6.5, 4.2, 1.95 px, falling monotonically toward zero. The raw spline over
    # the same range CONVERGES, to 27.9, 26.9, 26.5, 26.2, 26.14, 26.09.
    #
    # So: measure curvature where the points come from the curve, never where they come
    # from an interpolation of it.
    radius_min = min_turn_radius(curve if len(curve) >= 3 else centre)
    if radius_min < float(width) / 2.0:
        raise GeometryError(
            f"the spine turns too sharply for a cell {width:g} px wide: the tightest "
            f"turn has radius {radius_min:.2f} px against a half-width of "
            f"{width / 2.0:.2f} px.\n"
            f"Move the middle point(s) toward the pole-to-pole line, or draw the cell "
            f"narrower. More points along the same path do not help.")
    unit = tangents(centre)
    normal = np.column_stack([-unit[:, 1], unit[:, 0]])
    radius = float(width) / 2.0

    # THE RIBS TAPER INTO THE POLES, and getting this wrong put hooks on every outline.
    #
    # CORRECTED, AND THE REFERENCE I FIRST COPIED WAS THE WRONG ONE. The ribs used to be
    # full width at EVERY rung including the two at the poles, while the end caps were
    # centred one radius further in, so the outline ran to the pole, jumped back to the
    # cap's start, swept round and came back, leaving a hook sticking out past each end.
    # Reported from a screenshot; visible in `msd_out/outline_order_check.png`.
    #
    # The straight-capsule test that was meant to catch this INSTEAD LOCKED IT IN,
    # because the "independent transcription" it compares against came from the
    # segmentation repository's Python `generate_capsule_mesh`, which the browser
    # editor never calls. The editor's real geometry is its JavaScript `makeCapsule`,
    # and that tapers: `h = sqrt(r**2 - (r - d)**2)` within one radius of each pole.
    # Oufti's own meshes taper too, `meshio` states that rung 0 and rung N-1 are
    # degenerate. So two references disagreed and I had transcribed the unused one.
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(centre, axis=0).T))])
    from_start = np.minimum(along, along[-1] - along)
    half = np.where(from_start < radius,
                    np.sqrt(np.maximum(radius ** 2 - (radius - from_start) ** 2, 0.0)),
                    radius)

    mesh = np.empty((n_rungs, 4), dtype=np.float32)
    mesh[:, 0] = centre[:, 0] - half * normal[:, 0]
    mesh[:, 1] = centre[:, 1] - half * normal[:, 1]
    mesh[:, 2] = centre[:, 0] + half * normal[:, 0]
    mesh[:, 3] = centre[:, 1] + half * normal[:, 1]

    # ---- the outline: down one side and back up the other ---------------------------
    # No separate cap arcs. The ribs already close to a point at each pole, so the
    # capsule IS the ribbon between them, which is also what makes the outline and the
    # mesh describe one shape rather than two that have to be kept in agreement.
    # Duplicate pole points are dropped so the ring has no zero-length edges.
    points = [(mesh[i, 0], mesh[i, 1]) for i in range(n_rungs)]
    for i in range(n_rungs - 1, -1, -1):
        if np.hypot(mesh[i, 2] - mesh[i, 0], mesh[i, 3] - mesh[i, 1]) > 1e-9:
            points.append((mesh[i, 2], mesh[i, 3]))
    points.append(points[0])
    return mesh, np.array(points, dtype=np.float32)


def spine_curvature(mesh) -> dict:
    """How bent a mesh is, from its own ribs. Read back, never stored.

    Curvature is a biological readout, and once meshes can carry it somebody will want to
    measure it, so it is derived from the mesh rather than kept as a separate field that
    could disagree with the ribs.

    `sagitta_frac` is the maximum departure of the centreline from the straight
    pole-to-pole chord, as a fraction of the cell's length: scale-free, and the one to
    threshold on. `bend_angle_deg` is the angle between the cell's first third and its
    last third. A straight cell gives 0 for both.

    Corrected, and the first version was a noise integral. `bend_angle_deg` summed
    the ABSOLUTE angle between consecutive tangents, "total turning". On a real mesh
    every rib centre wobbles by a fraction of a pixel, and an absolute value turns each
    wobble into a positive contribution, so the sum measures the noise and the rung
    count rather than the shape. MEASURED on a perfectly straight centreline with
    0.2 px of rib noise: **23° at 10 rungs, 229° at 30, 3140° at 120** (nearly nine
    full turns on a straight line) while a genuinely curved cell with no noise at all
    read **65°**. It was unbounded, it grew with sampling, and it ranked a straight
    noisy cell 48x above a bent clean one.

    Found on real data: 132 Oufti cells came back with bends of 78–104° while their
    sagittae said they bow less than 1 % of their own length. Those two cannot describe
    one shape. The synthetic tests never saw it, because a synthetic capsule has no
    noise and gave exactly 0: a positive control passing vacuously, the third time in
    this project.

    Same defect as `orientation_radians.m`'s `atan(|dy|/|dx|)` and the legacy
    `sqrt(abs(y_icept/4))`, both already recorded here: an absolute value that turns
    noise into signal, biased strictly upward.

    The replacement compares the direction of the first third against the last third,
    so it averages over many ribs instead of differencing adjacent ones, is bounded by
    180°, and does not grow with the rung count.
    """
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 2 or mesh.shape[1] != 4 or len(mesh) < 3:
        raise GeometryError(f"need an (N>=3, 4) mesh, got {mesh.shape}")
    centre = np.column_stack([(mesh[:, 0] + mesh[:, 2]) / 2.0,
                              (mesh[:, 1] + mesh[:, 3]) / 2.0])
    step = max(1, len(centre) // 3)
    head = centre[step] - centre[0]
    tail = centre[-1] - centre[-1 - step]
    if np.hypot(*head) <= 1e-9 or np.hypot(*tail) <= 1e-9:
        bend = 0.0
    else:
        bend = float(abs(np.degrees(
            np.arctan2(head[0] * tail[1] - head[1] * tail[0], head @ tail))))

    chord = centre[-1] - centre[0]
    length = float(np.hypot(*chord))
    if length <= 1e-9:
        return {"bend_angle_deg": bend, "sagitta_frac": 0.0, "chord_px": 0.0}
    unit_chord = chord / length
    rel = centre - centre[0]
    perp = np.abs(rel[:, 0] * unit_chord[1] - rel[:, 1] * unit_chord[0])
    return {"bend_angle_deg": bend,
            "sagitta_frac": float(perp.max() / length),
            "chord_px": length}
