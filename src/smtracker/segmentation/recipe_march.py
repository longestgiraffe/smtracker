"""The experimenter's March 2026 segmentation pipeline, as a recipe beside SMTracker's own.

Asked for 14 Oct 2026: "in my users folder the segmentation pipeline, the model that i used
there worked best ... can u import it in the tool so i can test it"; "it was in
segmentation_pipeline.py". That is ``~/segmentation_pipeline.py`` (7 Mar 2026).
It is NOT a trained model: it is `bact_phase_omni` read with its own settings and its own
mask -> outline steps, and those steps are where it differs from `segment.segment_field`:

* the model is read at ``mask_threshold`` -1.0 (SMTracker: 0.0), flow 0.4, ``cluster`` and
  ``resample`` on, the diameter left to the model, the phase image min-max scaled to 8 bit
  (`segment.normalise`, the same);
* a mask is dropped below 60 px, or touching the border (3 px) unless over 300 px, or rounder
  than 1.2 (aspect), or under 60 px once filled;
* a mask is split at a CONSTRICTION (`split_dumbbell`, min/max half-width below 0.55), else
  as a CHAIN (`split_chain_mask`) when its skeleton is over 40 px with an area over 420 px, or
  its aspect over 4.5 with an area over 300 px -- SMTracker's own path splits nothing;
* each piece becomes a STRAIGHT pole-to-pole capsule along the mask's principal axis
  (`extend_to_poles` here returns a straight line; the split pieces of one mask keep the
  parent's axis), searching 25 px for the width and 35 px for the poles;
* a capsule is kept when its body is 6.5-16 px wide and at least 10 px long; one still over
  50 px long is split once more; otherwise it is kept as it is when its width passes.

The functions below are COPIED VERBATIM from that file (generated from its source, so they
cannot drift from what was run), and `meshes_from_masks` is its per-series loop with the file
writing, previews and an unused fluorescence mask left out. They duplicate parts of
`masks.py` ON PURPOSE, as legacy mode keeps v2's arithmetic: this is the recipe being
compared, and "improving" it would test something else. If it wins, its steps move into
the one implementation; if it does not, this module goes.

Checked against the original script on the same masks: `tests/test_recipe_march.py`.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

__all__ = ["NAME", "LABEL", "MODEL", "MASK_THRESHOLD", "FLOW_THRESHOLD", "meshes_from_masks"]

#: The recipe's name in the model list (`segment.RECIPES`), and how the window shows it.
NAME = "pipeline_2026_03"
LABEL = "Phase pipeline: bact_phase_omni, chains split"
MODEL = "bact_phase_omni"
MASK_THRESHOLD = -1.0
FLOW_THRESHOLD = 0.4


# -- copied verbatim from segmentation_pipeline.py ---------------------------------------- #
# noqa: E501  (its own line lengths and spacing, kept)

def longest_skeleton_path(skel):
    pts = np.column_stack(np.where(skel))
    if len(pts) < 2: return None
    pt_set = {(r,c) for r,c in pts}
    def nb(r,c):
        return [(r+dr,c+dc) for dr in[-1,0,1] for dc in[-1,0,1]
                if (dr,dc)!=(0,0) and (r+dr,c+dc) in pt_set]
    ep = [(r,c) for r,c in pts if len(nb(r,c))==1]
    if not ep: ep = [tuple(pts[0])]
    def bfs(start):
        dist={start:0}; prev={start:None}; q=deque([start]); far=start
        while q:
            cur=q.popleft()
            for n in nb(*cur):
                if n not in dist:
                    dist[n]=dist[cur]+1; prev[n]=cur; q.append(n)
                    if dist[n]>dist[far]: far=n
        return far,dist,prev
    far1,_,_ = bfs(ep[0])
    far2,_,p2 = bfs(far1)
    path=[]; cur=far2
    while cur is not None: path.append(cur); cur=p2[cur]
    return np.array(path)[:,[1,0]].astype(float)


def smooth_pts(pts, w=5):
    r=pts.copy(); hw=w//2
    for i in range(hw, len(pts)-hw): r[i]=pts[i-hw:i+hw+1].mean(0)
    return r


def resample_1px(pts):
    diffs=np.diff(pts,axis=0); lens=np.sqrt((diffs**2).sum(1))
    cum=np.concatenate([[0],np.cumsum(lens)]); total=cum[-1]
    if total<2: return pts
    nd=np.arange(0,total,1.0)
    return np.column_stack([np.interp(nd,cum,pts[:,0]),
                            np.interp(nd,cum,pts[:,1])])


def get_tangent(mid, i):
    n=len(mid)
    if i<0: i=n+i
    if i==0: t=mid[1]-mid[0]
    elif i==n-1: t=mid[-1]-mid[-2]
    else: t=mid[i+1]-mid[i-1]
    l=np.linalg.norm(t); return t/l if l>1e-9 else np.array([1.,0.])


def measure_half_width(pt, normal, mask):
    H,W=mask.shape; hw=[]
    for sign in [1,-1]:
        for d in np.arange(0.5, 25, 0.5):
            nx=int(round(pt[0]+sign*normal[0]*d))
            ny=int(round(pt[1]+sign*normal[1]*d))
            if nx<0 or nx>=W or ny<0 or ny>=H or not mask[ny,nx]:
                hw.append(d-0.5); break
        else: hw.append(25.0)
    return sum(hw)/2


def extend_to_poles(midline, mask, global_axis=None):
    """Push both endpoints outward along the cell long axis to the mask boundary.

    global_axis: optional pre-computed unit vector (e.g. from parent cell before split).
    If None, computes PCA of the midline itself.
    """
    H, W = mask.shape

    if global_axis is not None:
        pca_axis = np.array(global_axis, dtype=float)
        pca_axis /= np.linalg.norm(pca_axis)
    else:
        mid_c   = midline - midline.mean(0)
        cov     = mid_c.T @ mid_c
        _, vecs = np.linalg.eigh(cov)
        pca_axis = vecs[:, -1]

    def push(pt, direction):
        for d in np.arange(0.5, 35, 0.5):
            nx = int(round(pt[0] + direction[0]*d))
            ny = int(round(pt[1] + direction[1]*d))
            if nx < 0 or nx >= W or ny < 0 or ny >= H or not mask[ny, nx]:
                return pt + direction*(d - 0.5)
        return pt

    # Align pca_axis with the outward direction of midline[-1]
    t_end = get_tangent(midline, -1)
    if np.dot(pca_axis, t_end) < 0:
        pca_axis = -pca_axis

    p1 = push(midline[0],  -pca_axis)
    p2 = push(midline[-1],  pca_axis)

    total = np.linalg.norm(p2 - p1)
    n     = max(int(np.ceil(total)) + 1, 4)
    return p1 + np.linspace(0, 1, n)[:, None] * (p2 - p1)


def capsule_outline(mesh, n_cap=6):
    s1,s2=mesh[:,:2],mesh[:,2:]; mb=mesh[len(mesh)//2,:2]
    def arc(pf, pt, mref, n):
        c=(pf+pt)/2; rv=pf-c; r=np.linalg.norm(rv)
        if r<1e-9: return np.array([pf])
        rh=rv/r; p=np.array([-rh[1],rh[0]])
        if np.dot(p,mref-c)>0: p=-p
        ang=np.linspace(0,np.pi,n+2)[1:-1]
        return np.array([c+r*(np.cos(a)*rh+np.sin(a)*p) for a in ang])
    c1=arc(s2[0],  s1[0],  mb, n_cap)
    c2=arc(s1[-1], s2[-1], mb, n_cap)
    out=np.vstack([c1, s1[1:-1], c2, s2[-2:0:-1]])
    return np.vstack([out, out[0]]).astype(np.float32)


def split_dumbbell(mask, min_max_ratio_threshold=0.55):
    """
    Detect and split dumbbell-shaped masks (two cells touching at a septum).

    Strategy:
      1. Compute distance transform on the mask (gives half-width at every point).
      2. Walk the skeleton, recording dist-transform value at each point.
      3. If the ratio of min/max width along the skeleton < threshold, there is a
         constriction (septum). Split the mask by eroding a thin slab perpendicular
         to the skeleton at the constriction point.
      4. Return list of 1 or 2 masks.

    threshold=0.65: works well for E.coli dividing cells where the septum
    reduces the half-width to ~45-50% of the full lobe width.
    """
    from skimage.morphology import skeletonize as _skel
    from scipy.ndimage import distance_transform_edt as _dt

    mask = binary_fill_holes(mask)
    if mask.sum() < 30:
        return [mask]

    skel_src = binary_fill_holes(binary_dilation(mask, iterations=2))
    dist     = _dt(mask)
    skel     = _skel(skel_src)

    # Build ordered path along skeleton
    pts = np.column_stack(np.where(skel))
    if len(pts) < 4:
        return [mask]
    pt_set = {(r, c) for r, c in pts}
    def nb(r, c):
        return [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1]
                if (dr,dc) != (0,0) and (r+dr,c+dc) in pt_set]
    ep = [(r, c) for r, c in pts if len(nb(r,c)) == 1]
    if not ep: ep = [tuple(pts[0])]
    def bfs(start):
        d = {start:0}; prev = {start:None}; q = deque([start]); far = start
        while q:
            cur = q.popleft()
            for n in nb(*cur):
                if n not in d:
                    d[n] = d[cur]+1; prev[n] = cur; q.append(n)
                    if d[n] > d[far]: far = n
        return far, d, prev
    far1, _, _  = bfs(ep[0])
    far2, _, pv = bfs(far1)
    path = []; cur = far2
    while cur is not None: path.append(cur); cur = pv[cur]
    path = np.array(path)   # shape (N,2) in row,col order

    if len(path) < 6:
        return [mask]

    # Smooth the distance profile (5-pt moving average)
    dvals = np.array([dist[r, c] for r, c in path], dtype=float)
    w = 3
    dvals_s = dvals.copy()
    for i in range(w, len(dvals)-w):
        dvals_s[i] = dvals[i-w:i+w+1].mean()

    d_min = dvals_s.min()
    d_max = dvals_s.max()
    if d_max < 1e-6 or d_min / d_max >= min_max_ratio_threshold:
        return [mask]   # no clear constriction

    # Find constriction: deepest minimum that is NOT within 15% of either end
    margin = max(3, int(0.15 * len(path)))
    interior = dvals_s[margin:-margin]
    if len(interior) == 0:
        return [mask]
    split_idx = margin + int(np.argmin(interior))

    # Split: erase a 1-pixel-thick cut perpendicular to skeleton at split_idx
    H, W = mask.shape
    split_mask = mask.copy()

    # Tangent at split point → normal = cut direction
    i = split_idx
    if i == 0: t = np.array([path[1,1]-path[0,1], path[1,0]-path[0,0]], dtype=float)
    elif i == len(path)-1: t = np.array([path[-1,1]-path[-2,1], path[-1,0]-path[-2,0]], dtype=float)
    else: t = np.array([path[i+1,1]-path[i-1,1], path[i+1,0]-path[i-1,0]], dtype=float)
    l = np.linalg.norm(t)
    if l < 1e-9: return [mask]
    t /= l
    nv = np.array([-t[1], t[0]])   # normal (x,y)

    # Thin cut — just enough to separate the two lobes
    pr, pc = path[split_idx]
    cut_width = max(1, int(d_min))
    for dist_along_normal in np.arange(-cut_width, cut_width+1, 0.5):
        rx = int(round(pr - nv[1]*dist_along_normal))
        cx = int(round(pc + nv[0]*dist_along_normal))
        if 0 <= rx < H and 0 <= cx < W:
            split_mask[rx, cx] = False

    # Separate the two halves
    lbl = label(split_mask)
    regions = [r for r in regionprops(lbl) if r.area >= 30]
    if len(regions) < 2:
        return [mask]   # split failed, return original

    # Re-expand each lobe back into the original mask to recover pixels lost at the cut
    result = []
    for r in sorted(regions, key=lambda r: -r.area)[:2]:
        lobe = binary_dilation(lbl == r.label, iterations=int(d_min) + 1)
        lobe = lobe & mask   # constrain to original cell boundary
        result.append(binary_fill_holes(lobe))
    return result


def build_capsule(omnipose_mask, signal_mask=None, offset_x=0, offset_y=0, global_axis=None):
    """
    Build capsule mesh from omnipose segmentation mask.

    global_axis: optional (2,) unit vector — the long axis of the parent cell before
    any dumbbell splitting.  When provided, pole extension uses this axis instead of
    the sub-mask PCA, giving correct orientation for split half-lobes.
    """
    raw_mask = binary_fill_holes(omnipose_mask)
    if raw_mask.sum() < 20: return None, None, None

    # Skeleton source: dilate+fill to bridge any septum gap
    skel_src  = binary_fill_holes(binary_dilation(raw_mask, iterations=2))
    # Pole boundary = raw_mask only (no extra dilation) so poles stop at real cell wall
    pole_mask = binary_fill_holes(raw_mask)

    skel    = skeletonize(skel_src)
    midline = longest_skeleton_path(skel)
    if midline is None or len(midline) < 3: return None, None, None

    midline = smooth_pts(resample_1px(smooth_pts(midline, 5)), 3)
    if len(midline) < 3: return None, None, None

    # Measure true cell width perpendicular to skeleton (robust for curved cells)
    # Use middle 50% of skeleton to avoid pole taper
    n_mid = len(midline)
    mid_start, mid_end = n_mid // 4, 3 * n_mid // 4
    half_widths = []
    for i in range(mid_start, mid_end):
        tang = get_tangent(midline, i)
        nv   = np.array([-tang[1], tang[0]])
        half_widths.append(measure_half_width(midline[i], nv, raw_mask))
    if not half_widths:
        return None, None, None
    R_body = float(np.median(half_widths))   # true radius, perpendicular to axis
    R_cap  = R_body * 1.10

    # Extend poles to halo boundary, using parent cell axis if provided
    midline = extend_to_poles(midline, pole_mask, global_axis=global_axis)

    diffs   = np.diff(midline, axis=0)
    cum_d   = np.concatenate([[0], np.cumsum(np.sqrt((diffs**2).sum(1)))])
    total_L = cum_d[-1]

    def cw(d):
        d2=total_L-d
        w1=2*R_body*np.sqrt(max(0., 1-((R_cap-min(d, R_cap))/R_cap)**2))
        w2=2*R_body*np.sqrt(max(0., 1-((R_cap-min(d2,R_cap))/R_cap)**2))
        return min(w1, w2)

    rungs=[]
    for i,(pt,d) in enumerate(zip(midline, cum_d)):
        tang=get_tangent(midline,i); nv=np.array([-tang[1],tang[0]]); hw=cw(d)/2
        rungs.append([pt[0]+nv[0]*hw+offset_x, pt[1]+nv[1]*hw+offset_y,
                      pt[0]-nv[0]*hw+offset_x, pt[1]-nv[1]*hw+offset_y])

    p1x,p1y=midline[0,0]+offset_x,  midline[0,1]+offset_y
    p2x,p2y=midline[-1,0]+offset_x, midline[-1,1]+offset_y
    rungs[0]  = [p1x,p1y,p1x,p1y]
    rungs[-1] = [p2x,p2y,p2x,p2y]

    mesh  = np.array(rungs, dtype=np.float32)
    model = capsule_outline(mesh)
    box   = np.array([mesh[:,[0,2]].min(), mesh[:,[1,3]].min(),
                      mesh[:,[0,2]].max(), mesh[:,[1,3]].max()], dtype=np.float64)
    return mesh, model, box


def split_chain_mask(mask, min_cell_len=18, max_cell_len=55):
    """Split a chain mask into individual cell masks.
    
    Uses perpendicular cuts at junction points along the skeleton rather than
    Voronoi assignment, which distorts angles on curved/angled chains.
    Allows a small overlap (1px dilation) at each pole so cells touch naturally.
    """
    from skimage.morphology import skeletonize
    from scipy.ndimage import binary_fill_holes, binary_dilation

    mask = binary_fill_holes(mask)
    skel = skeletonize(mask)
    spine = longest_skeleton_path(skel)
    if spine is None or len(spine) < 2:
        return [mask]

    spine = resample_1px(smooth_pts(spine, 5))
    total_len = len(spine)

    hws = []
    for i in range(total_len // 4, 3 * total_len // 4):
        tang = get_tangent(spine, i)
        nv   = np.array([-tang[1], tang[0]])
        hws.append(measure_half_width(spine[i], nv, mask))
    R = np.median(hws) if hws else 4.0
    est_cell_len = R * 2 * np.pi * 0.8

    n_cells = max(1, round(total_len / max(est_cell_len, min_cell_len)))
    if n_cells <= 1:
        return [mask]

    H, W = mask.shape

    # Find junction indices evenly spaced along skeleton
    junctions = [int(round(total_len * i / n_cells)) for i in range(1, n_cells)]

    # Build sub-masks by making perpendicular cuts at each junction
    # Strategy: label each skeleton pixel by which segment it belongs to,
    # then flood-fill the mask pixels from the skeleton labels.
    skel_labels = np.zeros(total_len, dtype=int)
    seg = 0
    boundaries = [0] + junctions + [total_len]
    for si in range(n_cells):
        skel_labels[boundaries[si]:boundaries[si+1]] = si

    # Map skeleton pixel coords → label
    label_map = np.full((H, W), -1, dtype=int)
    for idx, pt in enumerate(spine):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= y < H and 0 <= x < W:
            label_map[y, x] = skel_labels[idx]

    # Propagate labels to all mask pixels via distance to labelled skeleton pts
    ys_mask, xs_mask = np.where(mask)
    if len(ys_mask) == 0:
        return [mask]

    # For each mask pixel, find nearest labelled skeleton point
    skel_pts = np.column_stack([spine[:,0], spine[:,1]])  # (N, 2) x,y
    px_all   = np.stack([xs_mask, ys_mask], axis=1).astype(float)
    dists    = np.sum((px_all[:, None, :] - skel_pts[None, :, :]) ** 2, axis=2)
    nearest  = np.argmin(dists, axis=1)
    assigned = skel_labels[nearest]

    sub_masks = []
    for ci in range(n_cells):
        sub = np.zeros((H, W), bool)
        idx = assigned == ci
        if idx.sum() < 20:
            continue
        sub[ys_mask[idx], xs_mask[idx]] = True
        # Dilate by 1px so poles of adjacent cells overlap slightly — 
        # avoids gaps and allows natural touching without distortion
        sub = binary_dilation(sub, iterations=1) & mask
        sub_masks.append(binary_fill_holes(sub))

    return sub_masks if sub_masks else [mask]


# -- its per-series loop (process_series), without the file writing ------------------------ #
def meshes_from_masks(masks) -> tuple:
    """``(cells, stats)`` from an omnipose label image, as `process_series` built them:
    ``cells`` is ``[(mesh, outline, parent_label)]`` (mesh ``(N, 4)`` rungs, outline the
    capsule polygon, both in image pixels), ``stats`` ``{"raw", "inside", "border",
    "failed"}``."""
    masks = np.asarray(masks).astype(np.int32)
    H, W = masks.shape
    n_raw = int(masks.max())

    BORDER = 3
    valid_labels = []
    border_labels = []
    for r in regionprops(masks):
        minr, minc, maxr, maxc = r.bbox
        at_border = minr < BORDER or minc < BORDER or maxr > H - BORDER or maxc > W - BORDER
        if r.area < 60:
            continue
        if at_border:
            if r.area > 300:
                border_labels.append(r.label)
        else:
            valid_labels.append(r.label)
    all_labels = valid_labels + border_labels

    cells = []
    failed = 0
    for cid in all_labels:
        cell_mask = (masks == cid)
        props = regionprops(cell_mask.astype(int))[0]
        minr, minc, maxr, maxc = props.bbox
        pad = 8
        r0, c0 = max(0, minr - pad), max(0, minc - pad)
        r1, c1 = min(H, maxr + pad), min(W, maxc + pad)
        local_raw = binary_fill_holes(cell_mask[r0:r1, c0:c1])

        lprops = regionprops(local_raw.astype(int))[0]
        aspect = lprops.major_axis_length / max(lprops.minor_axis_length, 1)
        if aspect < 1.2 or lprops.area < 60:
            failed += 1
            continue

        _mid_c = np.column_stack(np.where(local_raw)).astype(float)
        _mid_c -= _mid_c.mean(0)
        _cov = _mid_c.T @ _mid_c
        _, _vecs = np.linalg.eigh(_cov)
        _pca_rc = _vecs[:, -1]
        global_axis = np.array([_pca_rc[1], _pca_rc[0]])

        sub_masks = split_dumbbell(local_raw)

        if len(sub_masks) == 1:
            _skel = skeletonize(binary_fill_holes(local_raw))
            _sp = longest_skeleton_path(_skel)
            _skel_len = 0.0
            if _sp is not None and len(_sp) > 1:
                _diffs = np.diff(_sp.astype(float), axis=0)
                _skel_len = float(np.sum(np.sqrt((_diffs ** 2).sum(1))))
            do_chain = (_skel_len > 40 and lprops.area > 420) or \
                       (aspect > 4.5 and lprops.area > 300)
            if do_chain:
                chain = split_chain_mask(local_raw)
                if len(chain) > 1:
                    sub_masks = chain

        def _mesh_len(mesh):
            mid = (mesh[:, :2] + mesh[:, 2:]) / 2
            return float(np.sum(np.sqrt(((np.diff(mid, axis=0)) ** 2).sum(axis=1))))

        def _body_width(mesh):
            w = np.sqrt(((mesh[:, 2:] - mesh[:, :2]) ** 2).sum(axis=1))
            bw = w[2:-2] if len(w) > 4 else w
            return bw.mean() if len(bw) > 0 else 0.0

        def _valid(mesh):
            if mesh is None:
                return False
            bw = _body_width(mesh)
            if bw < 6.5 or bw > 16.0:
                return False
            ml = _mesh_len(mesh)
            if ml < 10:
                return False
            return True

        is_split = len(sub_masks) > 1
        for sub in sub_masks:
            mesh, model, box = build_capsule(sub, offset_x=c0, offset_y=r0,
                                             global_axis=None if is_split else global_axis)
            if mesh is None:
                failed += 1
                continue
            if _valid(mesh):
                cells.append((mesh, model, cid))
            else:
                ml = _mesh_len(mesh)
                bw = _body_width(mesh)
                if ml > 50 and bw >= 5.0:
                    sub2 = split_chain_mask(sub)
                    if len(sub2) > 1:
                        added = 0
                        for s2 in sub2:
                            m2, mo2, b2 = build_capsule(s2, offset_x=c0, offset_y=r0)
                            if m2 is not None and _valid(m2):
                                cells.append((m2, mo2, cid))
                                added += 1
                        if added > 0:
                            continue
                if bw >= 6.5 and bw <= 16.0 and ml >= 10:
                    cells.append((mesh, model, cid))
                else:
                    failed += 1
    stats = {"raw": n_raw, "inside": len(valid_labels), "border": len(border_labels),
             "failed": failed}
    return cells, stats
