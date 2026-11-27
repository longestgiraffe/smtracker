"""A binary MASK becomes a capsule mesh: skeleton, midline, half-width, rungs.

The other direction from `geometry.py`, and that is why it is a separate module.
`geometry.capsule_from_spine` builds a mesh from a SPINE a human dragged, the editor's
operation. This builds one from a MASK a network produced. They share a mesh
representation and nothing else, and the integration brief's proposed layout put both in
one file; keeping them apart is what stops "geometry" meaning two things.

Ported from `hada/geometry.py`, which is the newer of the two copies. The segmentation
tool `segmentation_tool_v3.py` carries the same ten functions and they have DRIFTED: all
ten differ. HADA's is the one to take, and its own header says why:

  * `build_capsule` guarded against `R_body == 0`, which raised ZeroDivisionError and
    aborted a whole batch mid-run;
  * `split_chain_mask` uses a KD-tree rather than an (n_mask_px x n_spine_px) dense
    distance matrix, which was the memory ceiling on large fields;
  * search distances scale with the configured cell width instead of a hardcoded 25/35 px,
    so a different magnification is a config change;
  * and `extend_to_poles` keeps the midline's CURVATURE. The older copy returned a
    straight line between the two extended poles, which for a bent cell or a chain put
    the mesh largely outside the cell.

A rejected mask is named, not silently dropped. The original returns
`(None, None, None)` for every failure, so a batch cannot tell "too small" from "no
skeleton" from "zero width", and a field that quietly yields no cells is
indistinguishable from an empty field, which this project has already recorded as the
worse failure. `build_capsule` returns a `CapsuleFit` carrying the reason, and the caller
counts them.

The pole convention is OUFTI'S: the first and last rungs collapse to single points, so
rung 0 and rung N-1 are degenerate. `cellmesh` reads that, `editing.cell_from_ribs`
relies on it, and `geometry.capsule_from_spine` writes it, one convention, three places.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = [
    "labels_from_outlines",
    "MaskConfig",
    "CapsuleFit",
    "longest_skeleton_path",
    "smooth_points",
    "resample_1px",
    "tangent_at",
    "measure_half_width",
    "extend_to_poles",
    "capsule_outline",
    "build_capsule",
]


@dataclass(frozen=True)
class MaskConfig:
    """The tunables, with the units they are in.

    `cell_width_px` is the ONE number the search distances scale from, so a different
    magnification is a config change rather than an edit. It is not a measurement of
    anything: it sets how far the width probe and the pole extension are allowed to look.
    """

    cell_width_px: float = 10.5
    #: Reject a mask smaller than this before doing any work. Below ~20 px there is no
    #: skeleton to speak of.
    min_mask_area_px: int = 20
    #: The cap radius is this multiple of the measured body radius. Straight from the
    #: original; it makes the drawn cap slightly rounder than the body is wide.
    cap_radius_factor: float = 1.10
    #: Points per pole cap in the outline polygon.
    n_cap: int = 6
    #: The midline FOLLOWS THE MASK (changed 14 Oct 2026). A straight rod was the default --
    #: *B. subtilis* is straight, and a bent mask is usually two cells -- but measured against
    #: hand-drawn outlines on nine datasets (one user's protein I LB / MM-fructose / MM-malate, protein A,
    #: protein D, protein E glucose and malate; sum images), forcing it straight displaced the capsule from
    #: its own mask (median overlap 0.73; 0.85 following it) and cost every segmenter: Omnipose's
    #: masks F1 0.42 straight against 0.49 following the curve (raw masks 0.48), never worse on
    #: any dataset; the classical segmenter's 0.31 against 0.37. Hand outlines and Oufti's meshes
    #: follow a cell's slight curve. True keeps the old straight rod.
    straight_mesh: bool = False

    @property
    def max_search(self) -> float:
        return max(3.0, self.cell_width_px * 2.5)

    @property
    def pole_search(self) -> float:
        return max(4.0, self.cell_width_px * 3.5)


@dataclass
class CapsuleFit:
    """What a mask became, or why it did not become anything.

    `ok` is False with a `reason` rather than a bare `None`, so a run can report
    "12 masks rejected: 5 too small, 4 no skeleton, 3 zero width" instead of a field
    that merely came back empty.
    """

    ok: bool
    reason: str = ""
    mesh: np.ndarray | None = None          # (N, 4) float32, Oufti rung order
    outline: np.ndarray | None = None       # (M, 2) float32, closed
    box: np.ndarray | None = None
    body_radius_px: float = 0.0
    length_px: float = 0.0

    @classmethod
    def rejected(cls, reason: str) -> "CapsuleFit":
        return cls(ok=False, reason=reason)


def labels_from_outlines(cells, shape, *, cell_ids=None) -> np.ndarray:
    """Outlines -> a LABELLED image, one integer per cell. The inverse of this module.

    Everything else here goes mask -> capsule, which is the segmentation direction. This
    goes back, because a corrected mesh is also the best training label anyone will ever
    have for this data: a human looked at it.

    **IT USES `cellmesh.points_in_polygon`, and that is the whole point.** That function
    is MATLAB `inpolygon`'s `in | on` and is what the cell filter uses to decide which
    localisations belong to a cell. A mask drawn with any other rule: skimage's polygon
    rasteriser, a matplotlib path, an even-odd fill with different edge handling: would
    disagree with the analysis about which pixels are cell, by a boundary pixel here and
    there. Two answers to "is this inside the cell" is exactly the kind of split this
    project keeps having to close.

    Pixel centres are tested at integer coordinates, matching the convention the mesh
    files use (`PIXEL_ORIGIN_OFFSET` is applied at the viewer, not here).

    ``shape`` is ``(height, width)``. ``cell_ids`` labels each cell; the default numbers
    them from 1 in the order given, which is what `save_field` writes into the ``.mat``.

    Overlaps go to the LAST cell that claims the pixel, and that is a real if rare case,
    two hand-drawn outlines can touch. It is left rather than refused because a mask is a
    training label, not a measurement; the `.mat` remains the primary record.
    """
    from ..cellmesh import points_in_polygon

    height, width = int(shape[0]), int(shape[1])
    labels = np.zeros((height, width), dtype=np.uint16)
    if cell_ids is None:
        cell_ids = range(1, len(cells) + 1)
    for cell, label in zip(cells, cell_ids):
        poly = np.asarray(getattr(cell, "outline", cell), dtype=float)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        # ONLY THE BOUNDING BOX, so a large field costs the cell's own area rather than
        # the whole image per cell.
        x0 = max(int(np.floor(poly[:, 0].min())) - 1, 0)
        x1 = min(int(np.ceil(poly[:, 0].max())) + 2, width)
        y0 = max(int(np.floor(poly[:, 1].min())) - 1, 0)
        y1 = min(int(np.ceil(poly[:, 1].max())) + 2, height)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        inside = points_in_polygon(xs.ravel(), ys.ravel(), poly[:, :2])
        block = labels[y0:y1, x0:x1]
        block[inside.reshape(ys.shape)] = int(label)
    return labels


def longest_skeleton_path(skel) -> np.ndarray | None:
    """The longest path through a skeleton, as `(N, 2)` in `(x, y)`.

    Double BFS: the farthest node from an arbitrary endpoint, then the farthest from
    that: the standard graph-diameter trick. `None` when the skeleton is too small to
    define a path at all.
    """
    pts = np.column_stack(np.where(skel))
    if len(pts) < 2:
        return None
    present = {(r, c) for r, c in pts}

    def neighbours(r, c):
        return [(r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in present]

    ends = [(r, c) for r, c in pts if len(neighbours(r, c)) == 1] or [tuple(pts[0])]

    def bfs(start):
        distance = {start: 0}
        previous = {start: None}
        queue = deque([start])
        farthest = start
        while queue:
            current = queue.popleft()
            for node in neighbours(*current):
                if node not in distance:
                    distance[node] = distance[current] + 1
                    previous[node] = current
                    queue.append(node)
                    if distance[node] > distance[farthest]:
                        farthest = node
        return farthest, previous

    first, _ = bfs(ends[0])
    second, previous = bfs(first)
    path, current = [], second
    while current is not None:
        path.append(current)
        current = previous[current]
    return np.array(path)[:, [1, 0]].astype(float)


def smooth_points(points: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average, endpoints untouched: they are the poles."""
    out = np.asarray(points, dtype=float).copy()
    half = window // 2
    for i in range(half, len(out) - half):
        out[i] = np.asarray(points)[i - half:i + half + 1].mean(axis=0)
    return out


def resample_1px(points: np.ndarray) -> np.ndarray:
    """Resample a polyline to about one-pixel spacing, by arc length."""
    points = np.asarray(points, dtype=float)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    if arc[-1] < 2:
        return points
    want = np.arange(0.0, arc[-1], 1.0)
    return np.column_stack([np.interp(want, arc, points[:, 0]),
                            np.interp(want, arc, points[:, 1])])


def tangent_at(midline: np.ndarray, i: int) -> np.ndarray:
    """Unit tangent by central difference, one-sided at the ends."""
    n = len(midline)
    if i < 0:
        i = n + i
    if i == 0:
        t = midline[1] - midline[0]
    elif i == n - 1:
        t = midline[-1] - midline[-2]
    else:
        t = midline[i + 1] - midline[i - 1]
    norm = float(np.linalg.norm(t))
    return t / norm if norm > 1e-9 else np.array([1.0, 0.0])


def measure_half_width(point, normal, mask, max_search: float) -> float:
    """Half-width of the mask at `point`, probed along both directions of `normal`."""
    height, width = mask.shape
    found = []
    for sign in (1, -1):
        for d in np.arange(0.5, max_search, 0.5):
            x = int(round(point[0] + sign * normal[0] * d))
            y = int(round(point[1] + sign * normal[1] * d))
            if x < 0 or x >= width or y < 0 or y >= height or not mask[y, x]:
                found.append(d - 0.5)
                break
        else:
            found.append(float(max_search))
    return sum(found) / 2.0


def extend_to_poles(midline, mask, max_search: float, global_axis=None) -> np.ndarray:
    """Push the midline out to the mask boundary at both ends.

    Skeletonisation erodes the rounded poles, so the raw midline stops short and
    under-measures the cell.

    The curvature is kept. The older copy of this function returned a straight line
    between the two extended poles, discarding the midline's shape; for a bent cell that
    put the mesh largely outside the cell and any downstream (l, d) sampling then read
    background. Here only short straight extensions are appended along the LOCAL end
    tangent.

    The two seams where those extensions meet the skeleton are smoothed, and that is not
    cosmetic: measured over 1863 cells, **71 % of all cells called "bent" were this
    artefact**, with the kink at l ~ 0.15 and l ~ 0.85 and never mid-cell.
    `smooth_points` leaves the first and last points alone, so the poles do not move.
    """
    midline = np.asarray(midline, dtype=float)
    height, width = mask.shape

    def push(point, direction):
        last = point
        for d in np.arange(0.5, max_search, 0.5):
            x = int(round(point[0] + direction[0] * d))
            y = int(round(point[1] + direction[1] * d))
            if x < 0 or x >= width or y < 0 or y >= height or not mask[y, x]:
                return point + direction * (d - 0.5)
            last = point + direction * d
        return last

    if global_axis is not None:
        axis = np.asarray(global_axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        start_dir = axis if np.dot(axis, midline[0] - midline[-1]) > 0 else -axis
        end_dir = -start_dir
    else:
        start_dir = -tangent_at(midline, 0)
        end_dir = tangent_at(midline, len(midline) - 1)

    first = push(midline[0], start_dir)
    last = push(midline[-1], end_dir)

    def bridge(a, b):
        gap = float(np.linalg.norm(b - a))
        if gap < 1.0:
            return np.zeros((0, 2))
        k = int(np.floor(gap))
        return a + np.linspace(0.0, 1.0, k + 1)[1:-1, None] * (b - a)

    parts = [first[None, :], bridge(first, midline[0]), midline,
             bridge(midline[-1], last), last[None, :]]
    extended = resample_1px(np.vstack([p for p in parts if len(p)]))
    if len(extended) >= 7:
        extended = smooth_points(extended, 5)
    return extended


def capsule_outline(mesh: np.ndarray, n_cap: int = 6) -> np.ndarray:
    """The closed outline of a mesh, with semicircular pole caps."""
    mesh = np.asarray(mesh, dtype=float)
    left, right = mesh[:, :2], mesh[:, 2:]
    reference = mesh[len(mesh) // 2, :2]

    def arc(start, end, ref, n):
        centre = (start + end) / 2.0
        radial = start - centre
        radius = float(np.linalg.norm(radial))
        if radius < 1e-9:
            return np.array([start])
        unit = radial / radius
        perpendicular = np.array([-unit[1], unit[0]])
        if np.dot(perpendicular, ref - centre) > 0:
            perpendicular = -perpendicular
        angles = np.linspace(0.0, np.pi, n + 2)[1:-1]
        return np.array([centre + radius * (np.cos(a) * unit + np.sin(a) * perpendicular)
                         for a in angles])

    cap_a = arc(right[0], left[0], reference, n_cap)
    cap_b = arc(left[-1], right[-1], reference, n_cap)
    ring = np.vstack([cap_a, left[1:-1], cap_b, right[-2:0:-1]])
    return np.vstack([ring, ring[0]]).astype(np.float32)


def build_capsule(mask, cfg: MaskConfig | None = None, *, offset_x: float = 0.0,
                  offset_y: float = 0.0, global_axis=None) -> CapsuleFit:
    """Fit a capsule mesh to one binary cell mask.

    Returns a `CapsuleFit`; check `.ok` and report `.reason` when it is False. The four
    rejections are all legitimate outcomes for a mask a network produced, not errors:
    a mask too small to carry a skeleton, a skeleton too short to define a path, a
    midline that left the mask (zero measured width), and a resampled midline that
    collapsed.
    """
    from scipy.ndimage import binary_dilation, binary_fill_holes
    from skimage.morphology import skeletonize

    cfg = cfg or MaskConfig()
    filled = binary_fill_holes(np.asarray(mask, dtype=bool))
    if filled.sum() < cfg.min_mask_area_px:
        return CapsuleFit.rejected(
            f"mask is {int(filled.sum())} px, below the {cfg.min_mask_area_px} px floor")

    # DILATE AND FILL BEFORE SKELETONISING so a partial septum does not fork the
    # skeleton; the widths are then measured against the UNDILATED mask, so the dilation
    # cannot inflate the cell.
    source = binary_fill_holes(binary_dilation(filled, iterations=2))
    midline = longest_skeleton_path(skeletonize(source))
    if midline is None or len(midline) < 3:
        return CapsuleFit.rejected("no skeleton path long enough to define a midline")
    midline = smooth_points(resample_1px(smooth_points(midline, 5)), 3)
    if len(midline) < 3:
        return CapsuleFit.rejected("the midline collapsed on resampling")

    # THE BODY RADIUS COMES FROM THE MIDDLE HALF ONLY, because the poles taper and a
    # median over the whole midline would report a width no part of the cell has.
    n = len(midline)
    widths = [measure_half_width(midline[i],
                                 np.array([-tangent_at(midline, i)[1],
                                           tangent_at(midline, i)[0]]),
                                 filled, cfg.max_search)
              for i in range(n // 4, 3 * n // 4)]
    if not widths:
        return CapsuleFit.rejected("no rungs in the middle half to measure a width")
    r_body = float(np.median(widths))
    if r_body <= 1e-6:
        # The original divided by the cap radius here and took the whole batch down.
        return CapsuleFit.rejected("measured body radius is zero; the midline left the "
                                   "mask")
    r_cap = r_body * cfg.cap_radius_factor

    midline = extend_to_poles(midline, filled, cfg.pole_search, global_axis=global_axis)

    if cfg.straight_mesh:
        a, b = midline[0], midline[-1]
        count = max(int(np.ceil(float(np.linalg.norm(b - a)))) + 1, 8)
        midline = a + np.linspace(0.0, 1.0, count)[:, None] * (b - a)

    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(midline, axis=0).T))])
    total = float(arc[-1])

    def full_width(d: float) -> float:
        """Cylindrical body, hemispherical caps, taken as the tighter of the two ends."""
        other = total - d
        w1 = 2 * r_body * np.sqrt(
            max(0.0, 1 - ((r_cap - min(d, r_cap)) / r_cap) ** 2))
        w2 = 2 * r_body * np.sqrt(
            max(0.0, 1 - ((r_cap - min(other, r_cap)) / r_cap) ** 2))
        return float(min(w1, w2))

    rungs = []
    for i, (point, d) in enumerate(zip(midline, arc)):
        t = tangent_at(midline, i)
        normal = np.array([-t[1], t[0]])
        half = full_width(d) / 2.0
        rungs.append([point[0] + normal[0] * half + offset_x,
                      point[1] + normal[1] * half + offset_y,
                      point[0] - normal[0] * half + offset_x,
                      point[1] - normal[1] * half + offset_y])

    # OUFTI'S POLE CONVENTION: the end rungs collapse to single points, so rung 0 and
    # rung N-1 are degenerate. `cellmesh` reads that and `editing.cell_from_ribs`
    # depends on it.
    first = (midline[0, 0] + offset_x, midline[0, 1] + offset_y)
    last = (midline[-1, 0] + offset_x, midline[-1, 1] + offset_y)
    rungs[0] = [first[0], first[1], first[0], first[1]]
    rungs[-1] = [last[0], last[1], last[0], last[1]]

    mesh = np.array(rungs, dtype=np.float32)
    outline = capsule_outline(mesh, cfg.n_cap)
    box = np.array([mesh[:, [0, 2]].min(), mesh[:, [1, 3]].min(),
                    mesh[:, [0, 2]].max(), mesh[:, [1, 3]].max()], dtype=np.float64)
    return CapsuleFit(ok=True, mesh=mesh, outline=outline, box=box,
                      body_radius_px=r_body, length_px=total)
