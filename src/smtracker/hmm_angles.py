"""Jump angles per HMM state, each beside its chance at the same places (22 Nov 2026).

Asked: "what about the angle map we planned where u see how tracks change in their angle? with which tool? sqd not
possible? or just for steps, HMM? appd?", then "both ja" (the HMM section's Jump angles page first, then the Spatial
Map's method). This module is the computation both read.

WHAT IT MEASURES. The angle between two steps in a row of one track, at the localisation they share: 0 degrees the
molecule carried straight on, 180 it turned back (`jumpangles`' quantity; TrackIt, Kuhn et al. 2021; Burov et al.
2013). The number quoted is the FOLD ANISOTROPY, f(180 +- 30) / f(0 +- 30): turning back over carrying on, 1 for no
preference, above 1 for more turning back (`WINDOW_DEG`).

WHY THE HMM LABELS THE PAIRS. An angle needs two steps in a row, so its label must be the PAIR's. The HMM is the only
fit here that labels each step from its whole track; a pair counts for state k with weight P(both steps in k), the
two-slice posterior of the same forward-backward (`hmm.decode_states`'s ``p_stay_k``), never the product of the two
steps' own posteriors, which at a switch gives each state a quarter where the chain says the pair straddles it. Pairs
that straddle a switch are counted out and their share said (``switch_share``). SQD's per-step weights come from step
length alone, the same length that drives the wall's reflection, so a split by them builds the wall into the split;
apparent D labels whole tracks of about five angles, across switches.

THE CHANCE AT THE SAME PLACES. Three things turn a free molecule back or forward with no biology in it:
- the cell WALL reflects it back, and a fast state's step is about half the half-width;
- LOCALISATION ERROR: two steps in a row share a localisation, so their errors pull opposite ways (covariance
  -sigma^2 per axis), a backward bias that is largest for the slow state, whose steps are about the size of the error;
- camera BLUR: a localisation is the average position over the exposure, which makes two steps in a row overlap in
  time (covariance +2 R D tau per axis, R = 1/6 for a camera exposed the whole frame; Berglund 2010), a FORWARD bias
  largest for the fast state.
`jumpangles.angle_null` simulates the first two in one box. Here every observed pair gets its own free molecules
(`CHANCE_DRAWS` a state), started where its turn was seen (the middle localisation, in its cell's own frame), in 3D
inside a capsule of that cell's length and width, at a depth drawn uniformly through the cell there; walked forward
and back 1.5 frames in `SUBSTEPS` sub-steps a frame or more with reflection at the wall (a reflected free walk is
reversible, so walking back from the vertex is walking forward to it); averaged over each frame's exposure; projected
onto the image; given each localisation's own error; and linked through the tracker's radius (below). Observed and
chance then share positions, cells, weights and
replicates; only the angles differ, and the EXCESS (observed minus chance) is what is quoted.

THE CHANCE'S D AND NOISE. The free molecule must make steps as long as the state's observed ones: its D is found
(`match_D`) so that its own simulated steps -- walls, blur and these localisation errors included, at these places --
have the mean squared length of the state's pairs (each at its weight). Converting the fitted D without the wall made
a fast chance molecule too slow (D 0.69 for 0.85 in a 0.9 um cell). Where even D 0 makes longer steps, the state's steps
are all localisation error, walked at D 0 and said (``at_floor``). The error is each spot's own through a per-spot
calibration MEASURED on immobile spots where the tracks carry the tracker's per-spot precision (the HMM's own under
Per spot, `hmm_view.precision_of` otherwise), else the HMM's or the project's recorded sigma (`hmm_view._sigma_um`, a
theoretical one named as such), else the chance is refused: a chance computed from an imagined error is not a chance.

RESOLUTION, MEASURED (22 Nov 2026): free molecules walked forward over whole tracks in a 3 x 0.9 um capsule (24,000
tracks, a separate walk from this one, 40 sub-steps a frame) read an excess of -0.002 (D 0.85, 30 nm; SE 0.009),
-0.003 (D 0.2; SE 0.011), -0.004 (D 0.05; SE 0.021) and +0.007 (D 0.85, no error; SE 0.007) in fold anisotropy, the
chance's D within 0.6 % of the truth. Four corrections got there, each a false excess before it: the chance's D
matched to the steps rather than converted from the fit (+0.06 for the fast state); the true start drawn around the
seen one and weighted by the cell's depth (+0.07 clamped to the wall, +0.04 with every in-plane place equally likely);
moved out by the wall's pull on the exposure average (+0.04 for the fast state); and the D matched on every pair, not a
sample (its D moved by 6 % between draws). An excess under `RESOLUTION` is not read as a difference from free.

THROUGH THE LINKING RADIUS (22 Nov 2026). A tracker links no step longer than its search radius, so every observed
pair has two steps shorter than it, and the observed steps of a molecule near the detection ceiling are shorter than
its own. A chance linking every step then walked at the D of the linked steps (0.8 um2/s for free mNG simulated at 2
and 5) and met the wall far less than the molecules did: +0.7 to +1.2 beyond chance on the lab's data drive after u-track at 5 px. So
each free pair is LINKED as the observed ones were (a pair with a step past the radius drawn again, `_LINK_TRIES`), at
the radius the condition's tracker recorded (`linking_for`; none for a track table, said), and its D is matched on the
linked steps (bisected: the linked steps' mean levels off with D). A fast walk takes finer sub-steps (`_substeps_for`).
Measured on free molecules in a 3 x 0.9 um capsule cut at 5 px (96,000 pairs, 30 nm): -0.016, -0.019, -0.016, +0.009
at D 0.05, 0.85, 2, 5 (the matched D 0.050, 0.860, 2.025, 5.07); without the radius +0.053 at D 2 and +0.427 at D 5.
On the synthetic free mNG's TRUE positions (no camera, no tracker; the lab's data drive, 90,000 pairs a set) the walk reads +0.02 (D 2)
and +0.05 (D 5) in a capsule 0.95 um wide, but +0.35 and +0.46 in one as wide as the outlines (1.15 um): THE OUTLINE
IS WIDER THAN THE SPACE THE MOLECULES FILL (below).

THE CHANCE'S CAMERA AND CELL (22 Nov 2026, asked "yes both"). The camera is open for the share of each frame the
camera files record (`exposure_for`: exposure over measured interval, 20 ms of 25.0 in one user's movies, 20 of 20.3 in
protein F's; the whole frame where no file says), centred in the frame (`_frame_steps`). The cell is each outline shrunk
evenly all round by ONE inset per condition (`chance_by_state`, `measure_inset`): the inset at which free pairs,
filling the cell uniformly and linked as the observed ones were, lie as far across it as the fastest state's vertices
(mean squared offset from the axis); the fastest state past the linking limit is passed over. A state PAST THE LINKING
LIMIT (its linked steps as long as no free molecule's up to `_MATCH_D_CAP`) gets no chance and says so. Measured on
free molecules in a 3.0 x 0.9 um cell drawn as a 3.2 x 1.1 um outline, the camera open 80 % of 25 ms frames, tracks
cut at 5 px (78,000-94,000 pairs, 30 nm): the width read 0.901 and 0.898 um, beyond chance +0.007 (D 0.85) and +0.033
(D 2; SE 0.011, 0.014), against +0.296 and +0.398 at the outline with a whole-frame camera; the whole frame open, D 2:
0.901 um, +0.020 (+0.266 before); D 5 past the limit. On the lab's data drive the cells read 0.99-1.00 um on the synthetic free mNG
(outlines 1.15), 0.88-0.90 on SynG (built 0.9; outlines 1.05), 0.79 and 0.73-0.78 on two sets of real cells
(outlines 1.06 and 0.95-1.01).

THE UNIT. Angles within a track share steps and tracks within a cell share a cell, so the unit is the replicate: one
excess per replicate (``MIN_WEIGHT`` pairs at least), its mean +- SD shown, Welch between two conditions on those
(`uncertainty.welch_test`; RULE 0: replicates are never paired).

HONEST LIMITS. The cell is a straight capsule (a bent cell's centreline is straightened), one width a condition
read from where its fastest state lies (a state that avoids the wall, or crowds it, moves the width with it); the
exposure is one share of the frame, the median over the movies, a uniform illumination while open (a stroboscopic
acquisition would need its own); the linking radius is one hard cut, where u-track's adapts to each track below its
cap (free mNG simulated at D 2 and 5 and tracked by u-track still read +0.23-0.28 and +0.67-0.75 beyond chance with
the width and the exposure read); the depth is drawn, not seen; the state's D is one value for the condition, so a
state that holds several speeds gets the chance of their average.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .jumpangles import DEFAULT_N_BINS, DEFAULT_WINDOW_DEG

__all__ = [
    "AngleError", "WINDOW_DEG", "N_BINS", "EXPOSURE_R", "RESOLUTION", "SUBSTEPS", "CHANCE_DRAWS", "MIN_WEIGHT",
    "state_names", "pair_table", "match_D", "chance_angles", "chance_by_state", "fold_ratio", "summarise",
    "fit_from_result", "noise_for", "linking_for", "exposure_for", "measure_inset", "cell_words",
    "compute_hmm_angles",
]

#: Half-width of the forward and backward windows of the fold anisotropy, degrees (TrackIt's 30; `jumpangles`).
WINDOW_DEG = DEFAULT_WINDOW_DEG
#: Bins of the angle histogram over 0-180 degrees (10 degrees each).
N_BINS = DEFAULT_N_BINS
#: Berglund's blur coefficient for a camera exposed during the whole frame interval (stream acquisition): the
#: chance's blur. 0 would be an instantaneous exposure.
EXPOSURE_R = 1.0 / 6.0
#: An excess (fold anisotropy) free molecules can read by the method alone (module head, "RESOLUTION"): smaller is not
#: called a difference from free.
RESOLUTION = 0.05
#: Sub-steps a frame of the chance walk, at least: its reflection at the wall and its exposure average are taken over
#: these. A faster walk takes more, so that no sub-step spreads by more than `_SUBSTEP_UM` (`_substeps_for`).
SUBSTEPS = 8
#: The largest spread of one sub-step (um), and the most sub-steps a frame. At 8 sub-steps a molecule at D 5 um2/s
#: moved 160 nm a sub-step in a cell of 450 nm radius, reflected too coarsely, and its chance read 0.10 too high (free
#: molecules in a 3 x 0.9 um capsule, through a 5 px linking radius); at 32 it read -0.013 (SE 0.018). 80 nm keeps 8
#: sub-steps up to D 1.28 at 20 ms, so a slow or middle state's chance is walked as before.
_SUBSTEP_UM = 0.08
_MAX_SUBSTEPS = 64
#: Free molecules a pair and a state.
CHANCE_DRAWS = 4
#: The weighted pairs a replicate needs for its own excess (`jumpangles.angles_by_unit`'s 20).
MIN_WEIGHT = 20.0
#: A pair with a smaller weight for a state is not walked for it (it adds nothing that can be seen).
_MIN_PAIR_WEIGHT = 1e-3
#: An observed turn outside its cell's capsule (an outline is not the membrane) is pulled to this share of the radius.
_INSIDE = 0.98
#: Draws of a chance molecule's true start before a vertex seen far outside its cell is pulled in (`_chance_steps`).
_START_TRIES = 50
#: The chance's D (`match_D`): draws a pair, the relative miss in mean squared step it stops at, the walks it may take.
_MATCH_DRAWS = 2
_MATCH_TOLERANCE = 0.002
_MATCH_WALKS = 8
#: Through a linking radius: the bisection's steps, and the largest D a chance may walk at (um2/s; past it the linked
#: steps of a molecule in a bacterium barely lengthen).
_MATCH_BISECT = 24
_MATCH_D_CAP = 50.0
#: Draws of a free pair before one the tracker would not have linked keeps its last (`_chance_steps`).
_LINK_TRIES = 30
#: The chance's cell (`measure_inset`): no radius below `_MIN_RADIUS_UM`; at most `_MAX_WIDER_UM` wider than the outline
#: (a radius); measured only from a fastest state at `_WIDTH_MIN_D` um2/s or more (a slower one's places are where it
#: stays, not the cell); the relative miss in mean squared offset the search stops at, and the molecules a place.
_MIN_RADIUS_UM = 0.1
_MAX_WIDER_UM = 0.15
_WIDTH_MIN_D = 0.1
_WIDTH_TOLERANCE = 0.002
_WIDTH_DRAWS = 2
_WIDTH_PAIRS = 20000
#: An exposure shorter than the frame is walked in sub-steps that hold it to within this share of the frame.
_EXPOSURE_TOLERANCE = 0.02
#: Free molecules a place, and rounds, that measure what the wall does to the exposure average there (`_wall_shifted`).
_PILOT_DRAWS = 8
_PILOT_ROUNDS = 1

_NAMES = {1: ("all",), 2: ("slow", "fast"), 3: ("slow", "middle", "fast")}


class AngleError(RuntimeError):
    """Raised when the angles or their chance cannot be computed honestly."""


def state_names(n_states: int) -> list:
    """The states' names, slowest first, as the HMM maps name them."""
    return list(_NAMES.get(int(n_states), tuple(f"state {k + 1}" for k in range(int(n_states)))))


# -- the pairs --------------------------------------------------------------------------------------------------- #
def pair_table(projected: pd.DataFrame, decoded: pd.DataFrame, n_states: int) -> pd.DataFrame:
    """Every pair of steps in a row of one track with its angle, its vertex (the shared localisation) and each state's
    weight. ``projected`` is a condition's tracks in cell coordinates (`Project.projected`), ``decoded`` the steps
    `hmm.decode_states` labelled on it. Three localisations in consecutive frames make a pair; a pair whose steps were
    not decoded (a step without per-spot precision under Per spot) or that has a step of length 0 (a synthetic truth's
    bound step: no direction) is left out and counted in ``attrs``."""
    from .canonical import LOC_SIGMA_COLUMN, without_attrs

    need = ["uid", "frame", "x_um", "y_um", "cell", "day", "movie", "condition"]
    missing = [c for c in need if c not in projected.columns]
    if missing:
        raise AngleError(f"the tracks lack {missing}")
    stay = [f"p_stay_{k + 1}" for k in range(int(n_states))]
    if any(c not in decoded.columns for c in stay):
        raise AngleError("the decoded steps carry no pair posteriors (p_stay_k); decode with this version")
    extra = [c for c in ("L_um", "D_um", "l_norm", "d_norm", LOC_SIGMA_COLUMN) if c in projected.columns]
    loc = (without_attrs(projected)[need + extra].sort_values(["uid", "frame"], kind="stable")
           .reset_index(drop=True))
    uid = loc["uid"].to_numpy()
    frame = loc["frame"].to_numpy(dtype=float)
    if len(loc) < 3:
        raise AngleError("fewer than three localisations: no pair of steps")
    i = np.arange(len(loc) - 2)
    ok = ((uid[i] == uid[i + 1]) & (uid[i + 1] == uid[i + 2])
          & np.isclose(frame[i + 1] - frame[i], 1.0) & np.isclose(frame[i + 2] - frame[i + 1], 1.0))
    i = i[ok]
    xy = loc[["x_um", "y_um"]].to_numpy(dtype=float)
    first, second = xy[i + 1] - xy[i], xy[i + 2] - xy[i + 1]
    n1, n2 = np.hypot(*first.T), np.hypot(*second.T)
    moving = (n1 > 0) & (n2 > 0)
    zero_step = int((~moving).sum())
    i, first, second, n1, n2 = i[moving], first[moving], second[moving], n1[moving], n2[moving]
    cosine = np.clip((first * second).sum(axis=1) / (n1 * n2), -1.0, 1.0)
    out = pd.DataFrame({
        "uid": uid[i], "step_frame": frame[i].astype(int), "frame": frame[i + 1].astype(int),
        "angle_deg": np.degrees(np.arccos(cosine)), "step_prev_um": n1, "step_next_um": n2,
        "x_um": xy[i + 1, 0], "y_um": xy[i + 1, 1],
    })
    for column in ("condition", "day", "movie", "cell"):
        out[column] = loc[column].to_numpy()[i + 1]
    for column in ("L_um", "D_um", "l_norm", "d_norm"):
        if column in loc.columns:
            out[column] = loc[column].to_numpy(dtype=float)[i + 1]
    if LOC_SIGMA_COLUMN in loc.columns:
        sigma = loc[LOC_SIGMA_COLUMN].to_numpy(dtype=float)
        for j in range(3):
            out[f"reported_sigma_{j}"] = sigma[i + j]

    weights = without_attrs(decoded)[["uid", "frame", *stay]].rename(columns={"frame": "step_frame"})
    merged = out.merge(weights, on=["uid", "step_frame"], how="left", validate="many_to_one")
    decoded_ok = merged[stay].notna().all(axis=1)
    undecoded = int((~decoded_ok).sum())
    merged = merged[decoded_ok].reset_index(drop=True)
    for k in range(int(n_states)):
        merged[f"w_{k + 1}"] = merged.pop(f"p_stay_{k + 1}").clip(lower=0.0, upper=1.0)
    merged["w_switch"] = (1.0 - merged[[f"w_{k + 1}" for k in range(int(n_states))]].sum(axis=1)).clip(lower=0.0)
    if merged.empty:
        raise AngleError("no pair of steps in a row was decoded")
    merged.attrs["zero_step"] = zero_step
    merged.attrs["undecoded"] = undecoded
    return merged


# -- the chance -------------------------------------------------------------------------------------------------- #
def _reflect(p: np.ndarray, half: np.ndarray, radius: np.ndarray) -> None:
    """Reflect ``p`` (N, 3; x along the axis) into the capsule of cylinder half-length ``half`` and radius ``radius``,
    in place: a point past the wall goes back as far inside, along the line from the nearest axis point."""
    axis = np.clip(p[:, 0], -half, half)
    rel = p.copy()
    rel[:, 0] -= axis
    rho = np.linalg.norm(rel, axis=1)
    out = rho > radius
    if out.any():
        scale = np.maximum(2.0 * radius[out] - rho[out], 0.0) / rho[out]
        p[out] = rel[out] * scale[:, None]
        p[out, 0] += axis[out]


def _walk_average(start, sd, half, radius, rng, *, substeps, open_half, vertex_only=False):
    """Walk from ``start`` (the middle of the vertex frame's exposure) with reflection, ``substeps`` sub-steps a frame;
    return the vertex exposure's half from here (sub-steps 0 to ``open_half``) and the next frame's whole exposure
    (``substeps`` -+ ``open_half``), each by the trapezoid rule and divided by the exposure's ``2 * open_half``
    sub-steps, so the vertex's two halves (one walk each way) add up to its average. ``open_half`` is ``substeps / 2``
    for a camera open the whole frame (`_frame_steps`). ``vertex_only``: stop after the vertex's half."""
    p = start.copy()
    h = int(open_half)
    norm = 2.0 * h
    vertex = 0.5 * p / norm                       # the vertex exposure's half from here: weights 1/2, 1, ..., 1/2
    after = np.zeros_like(p)
    first, last = int(substeps) - h, int(substeps) + h
    for j in range(1, (h if vertex_only else last) + 1):
        p = p + rng.normal(0.0, 1.0, size=p.shape) * sd[:, None]
        _reflect(p, half, radius)
        if j < h:
            vertex += p / norm
        elif j == h:
            vertex += 0.5 * p / norm
        if j == first or j == last:
            after += 0.5 * p / norm
        elif first < j < last:
            after += p / norm
    return vertex, after


def chance_angles(pairs: pd.DataFrame, geometry: dict, D: np.ndarray, sigma_um: np.ndarray, *, tau_s: float,
                  draws: int = CHANCE_DRAWS, substeps: int = SUBSTEPS, seed: int = 0, link_um=None,
                  exposure: float = 1.0, inset_um: float = 0.0) -> np.ndarray:
    """``draws`` free molecules' angles for every pair (module head): ``(len(pairs), draws)`` degrees. ``D`` is each
    pair's free D (µm²/s), ``sigma_um`` each pair's three localisation errors ``(n, 3)``; ``geometry`` maps a cell to
    its `cellcoords.CellGeometry`. The pairs need ``L_um`` and ``D_um`` (`Project.projected`). ``link_um``: the
    tracker's linking radius, every free pair LINKED as the observed ones were (`_chance_steps`); None, no tracker.
    ``exposure``: the share of each frame the camera is open (`exposure_for`); ``inset_um``: how much narrower in radius
    than its outline each cell is walked (`measure_inset`)."""
    n = len(pairs)
    if n == 0:
        return np.empty((0, draws))
    a, b = _chance_steps(pairs, geometry, D, sigma_um, tau_s=tau_s, draws=draws, substeps=substeps, seed=seed,
                         link_um=link_um, exposure=exposure, inset_um=inset_um)
    na, nb = np.hypot(*a.T), np.hypot(*b.T)
    cosine = np.clip((a * b).sum(axis=1) / np.maximum(na * nb, 1e-300), -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).reshape(n, draws)


def match_D(pairs: pd.DataFrame, geometry: dict, sigma_um: np.ndarray, target_um2: float, *, tau_s: float,
            weights=None, draws: int = _MATCH_DRAWS, substeps: int = SUBSTEPS, seed: int = 0, link_um=None,
            exposure: float = 1.0, inset_um: float = 0.0) -> float:
    """The free D whose chance steps -- walls, blur, these localisation errors and the tracker's linking radius
    (``link_um``) included, at these places -- are on average as long as the state's observed ones: mean squared step
    ``target_um2`` (the mean of its pairs' two steps, each pair at its ``weights``). Every pair is walked at its weight,
    ``draws`` times. 0 where even D 0 makes longer steps: the state's steps are all localisation error (``at_floor``).
    Without a linking radius the mean grows smoothly with D and a secant finds it in a few walks (`_MATCH_TOLERANCE`);
    with one it LEVELS OFF (the linked steps cannot be longer than the radius), so it is bracketed and bisected, and a
    state whose linked steps are as long as no free molecule's up to `_MATCH_D_CAP` are is PAST THE LINKING LIMIT:
    ``math.inf``, no chance (free molecules at D 5 um2/s, 25 ms frames, 5 px, a 0.9 um cell: their linked steps no
    longer tell their D, and a chance walked at the cap read -0.24).

    WHY NOT FROM THE FITTED D. Converting the fit's D through the wall-free relation ignores that the wall
    shortens a fast molecule's steps: checked on free molecules simulated forward in a 3 x 0.9 um capsule, D 0.85,
    30 nm, the converted chance (D tau + s_fit - s_null) / (tau (1 - 2R)) walked at 0.69 and read an excess of +0.06
    where there is none. WHY EVERY PAIR: matched on 2,000 of them once each, the D moved by about 6 % from one draw to
    the next, which a slow state's excess reads (0.0507-0.0527 for 0.05 over four seeds). WHY THE LINKING RADIUS (22
    Nov 2026, on the lab's data drive): a tracker drops a fast molecule's steps longer than its radius, so the observed steps, and a D
    matched to them without the radius, are below the molecule's; the chance then met the wall far less than the
    molecules did and free mNG simulated at D 2 and 5 um2/s read +0.7 to +1.2 beyond chance after u-track at 5 px."""
    n = len(pairs)
    if n == 0 or not math.isfinite(target_um2) or target_um2 <= 0:
        raise AngleError("no steps to match the chance's step size to")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    sig = np.asarray(sigma_um, dtype=float)

    def mean_r2(D: float) -> float:
        a, b = _chance_steps(pairs, geometry, np.full(n, D), sig, tau_s=tau_s, draws=draws, substeps=substeps,
                             seed=seed + 1, link_um=link_um, exposure=exposure, inset_um=inset_um)
        r2 = (0.5 * ((a ** 2).sum(axis=1) + (b ** 2).sum(axis=1))).reshape(n, draws)
        linked = np.isfinite(r2)
        if not linked.any():
            return math.inf                              # nothing linked: longer than anything the tracker keeps
        ww = np.repeat(w[:, None], draws, axis=1)
        return float(np.sum(np.where(linked, ww * r2, 0.0)) / np.sum(ww[linked]))

    floor = mean_r2(0.0)
    if floor >= target_um2:
        return 0.0
    guess = (target_um2 - floor) / (4.0 * tau_s * (1.0 - 2.0 * EXPOSURE_R * float(exposure)))
    if link_um is None:
        tried = [(0.0, floor)]
        for _ in range(_MATCH_WALKS):
            tried.append((guess, mean_r2(guess)))
            (d0, f0), (d1, f1) = tried[-2], tried[-1]
            if abs(f1 - target_um2) <= _MATCH_TOLERANCE * target_um2:
                return guess
            if f1 == f0:
                break
            guess = max(d1 + (target_um2 - f1) * (d1 - d0) / (f1 - f0), 0.0)
            if guess > 1e4:
                raise AngleError("no free D makes steps this long inside these cells")
        return min(tried[1:], key=lambda t: abs(t[1] - target_um2))[0]
    # THROUGH THE LINKING RADIUS: bracket, then bisect on log D
    lo, hi = 0.0, max(guess, 1e-4)
    while mean_r2(hi) < target_um2:
        lo, hi = hi, hi * 2.0
        if hi > _MATCH_D_CAP:
            return math.inf
    for _ in range(_MATCH_BISECT):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 2.0
        f = mean_r2(mid)
        if abs(f - target_um2) <= _MATCH_TOLERANCE * target_um2:
            return mid
        lo, hi = (mid, hi) if f < target_um2 else (lo, mid)
    return math.sqrt(lo * hi) if lo > 0 else hi / 2.0


def _wall_shifted(x, y, half, radius, sd, rng, substeps, open_half):
    """The places ``(x, y)`` moved out by what the wall does to a free molecule's exposure average there (sub-step
    spread ``sd``; `_chance_steps`), measured on `_PILOT_DRAWS` free molecules a place, `_PILOT_ROUNDS` times."""
    if not np.any(np.asarray(sd) > 0):
        return x, y
    m = _PILOT_DRAWS
    many = lambda a: np.repeat(np.asarray(a, dtype=float), m)                  # noqa: E731
    centre_x, centre_y = x, y
    for _ in range(_PILOT_ROUNDS):
        axis = np.clip(centre_x, -half, half)
        rho = np.hypot(centre_x - axis, centre_y)
        pull = np.where(rho > _INSIDE * radius, _INSIDE * radius / np.maximum(rho, 1e-12), 1.0)
        px, py = axis + (centre_x - axis) * pull, centre_y * pull
        depth = np.sqrt(np.maximum(radius ** 2 - (px - axis) ** 2 - py ** 2, 0.0))
        start = np.stack([many(px), many(py), rng.uniform(-1.0, 1.0, size=len(px) * m) * many(depth)], axis=1)
        forward, _ = _walk_average(start, many(sd), many(half), many(radius), rng, substeps=substeps,
                                   open_half=open_half, vertex_only=True)
        backward, _ = _walk_average(start, many(sd), many(half), many(radius), rng, substeps=substeps,
                                    open_half=open_half, vertex_only=True)
        shift = (forward + backward - start).reshape(len(px), m, 3).mean(axis=1)
        centre_x, centre_y = x - shift[:, 0], y - shift[:, 1]
    return centre_x, centre_y


def _chance_steps(pairs, geometry, D, sigma_um, *, tau_s, draws, substeps, seed, link_um=None, exposure=1.0,
                  inset_um=0.0):
    """The free molecules' two steps in the image, ``(n * draws, 2)`` each (`chance_angles`). With ``link_um`` (the
    tracker's linking radius) every pair is one the tracker would have LINKED, as every observed pair was: a free pair
    with a step longer than the radius is drawn again, from its own start, up to `_LINK_TRIES` times (a pair still not
    linked is LEFT OUT, its steps NaN: there a free molecule is almost never linked, and the observed pairs are all
    linked ones; a draw kept with its long steps made the chance's mean step grow past any linked one's, so a state
    past the linking limit read a D). The camera is open for the share
    ``exposure`` of each frame, centred in it; each cell is walked ``inset_um`` narrower in radius than its outline,
    its cylinder as long (`measure_inset`)."""
    if substeps < 2 or substeps % 2:
        raise AngleError("the sub-steps a frame must be even and at least 2")
    substeps, open_half = _frame_steps(D, tau_s, substeps, exposure)
    length, width = _cells(pairs, geometry)
    radius = np.maximum(width / 2.0 - float(inset_um), _MIN_RADIUS_UM)
    half = np.maximum(length / 2.0 - width / 2.0, 0.0)
    rng = np.random.default_rng(seed)
    rep = lambda a: np.repeat(np.asarray(a, dtype=float), draws, axis=0)      # noqa: E731
    hr, rr = rep(half), rep(radius)
    sig = rep(sigma_um)
    # WHERE THE MOLECULE WAS, NOT WHERE IT WAS SEEN. The vertex is a localisation: the molecule's average position over
    # the exposure, plus its error. Started there, a fast chance molecule starts further from the wall than the real
    # one was (the average of a walk that touched the wall lies inside it), meets the wall less and turns back less:
    # checked on free molecules simulated forward in a 3 x 0.9 um capsule, D 0.85 and 30 nm, a false excess of
    # +0.06. So the start is the vertex plus the spread of the true position around it, per axis R D tau from the
    # blur (the exposure average around the middle of a free walk; Dtau/6 for a whole-frame exposure) and the
    # vertex's own error squared, drawn and kept inside the cell, then a depth drawn uniformly through it there.
    # Drawn, not clamped, and weighted by the cell's DEPTH there: the true position given the seen one is that spread
    # times how likely the place is, and a free molecule fills the cell's VOLUME, so an in-plane place is as likely
    # as the cell is deep there -- deepest on the axis, no depth at the edge. A draw is kept at the depth's share of
    # the radius and drawn again otherwise (outside the cell it has none); only a vertex seen far outside its outline
    # keeps no draw and is pulled in. Clamping outside draws to the wall piled molecules up there (D 0.05, 30 nm: a
    # false excess of +0.07), and every in-plane place equally likely still put too many at the edge (+0.04, its
    # matched D 8 % high).
    seen_x = pairs["L_um"].to_numpy(dtype=float) - length / 2.0
    seen_y = pairs["D_um"].to_numpy(dtype=float)
    # AND SHIFTED BY WHAT THE WALL DID TO THE EXPOSURE AVERAGE. Near the wall a fast molecule's average over the
    # exposure lies INSIDE where it was (the wall turns the walk back during the exposure), so the true position is
    # further out than the seen one by that shift. Measured here for each pair at its place: free molecules started
    # there (`_PILOT_DRAWS` each, their depth drawn), the mean of their exposure average minus their start, taken off
    # the seen place, ONCE. Without it a fast free state read +0.04 (D 0.85, 30 nm); measured again over D on 22 Nov
    # 2026 (free molecules in a 3 x 0.9 um capsule, 30 nm, 96,000 pairs, SE 0.008-0.012): no shift +0.046 at D 2 and
    # +0.023 at 0.85; taken off twice (the shift measured again at the shifted place) -0.045 at D 2; once -0.011,
    # -0.012, -0.019 and +0.024 at D 0.2, 0.85, 2 and 5.
    seen_x, seen_y = _wall_shifted(seen_x, seen_y, half, radius,
                                   np.sqrt(2.0 * np.asarray(D, dtype=float) * tau_s / substeps), rng, substeps,
                                   open_half)
    seen_x, seen_y = rep(seen_x), rep(seen_y)
    D_rep = rep(D)
    a, b = _snippets(seen_x, seen_y, hr, rr, D_rep, sig, tau_s, substeps, open_half, rng)
    if link_um is None:
        return a, b
    link = float(link_um)
    todo = (np.hypot(*a.T) > link) | (np.hypot(*b.T) > link)
    for _ in range(_LINK_TRIES):
        rows = np.flatnonzero(todo)
        if not len(rows):
            break
        ra, rb = _snippets(seen_x[rows], seen_y[rows], hr[rows], rr[rows], D_rep[rows], sig[rows], tau_s, substeps,
                           open_half, rng)
        a[rows], b[rows] = ra, rb
        todo[rows] = (np.hypot(*ra.T) > link) | (np.hypot(*rb.T) > link)
    a[todo], b[todo] = np.nan, np.nan
    return a, b


def _substeps_for(D, tau_s: float, substeps: int) -> int:
    """The sub-steps a frame a walk at ``D`` takes: ``substeps`` at least, more where a sub-step would spread by more
    than `_SUBSTEP_UM`, even, at most `_MAX_SUBSTEPS` (`SUBSTEPS`)."""
    top = float(np.max(np.asarray(D, dtype=float), initial=0.0))
    need = math.ceil(2.0 * top * float(tau_s) / _SUBSTEP_UM ** 2) if math.isfinite(top) else _MAX_SUBSTEPS
    need += need % 2
    return int(min(max(int(substeps), need), max(int(substeps), _MAX_SUBSTEPS)))


def _frame_steps(D, tau_s: float, substeps: int, exposure: float = 1.0) -> tuple:
    """``(substeps, open_half)``: the sub-steps a frame (`_substeps_for`) and the half of the exposure in them. A camera
    open the whole frame: half the frame. Open for a share ``exposure`` of it: the fewest further sub-steps (even) that
    hold ``2 open_half / substeps`` within `_EXPOSURE_TOLERANCE` of the share (20 ms of 25: 10 sub-steps, 4 a half)."""
    n = _substeps_for(D, tau_s, substeps)
    share = float(exposure)
    if not math.isfinite(share) or share >= 1.0 - _EXPOSURE_TOLERANCE / 2.0:
        return n, n // 2
    if share <= 0.0:
        raise AngleError(f"an exposure of {share:g} of the frame cannot be walked")
    best = None
    for m in range(n, 4 * max(n, _MAX_SUBSTEPS) + 1, 2):
        h = max(1, int(round(share * m / 2.0)))
        miss = abs(2.0 * h / m - share)
        if best is None or miss < best[2]:
            best = (m, h, miss)
        if miss <= _EXPOSURE_TOLERANCE:
            break
    return best[0], best[1]


def _cells(pairs, geometry) -> tuple:
    """Each pair's cell's outline length and greatest width (um)."""
    for column in ("L_um", "D_um", "cell"):
        if column not in pairs.columns:
            raise AngleError(f"the pairs lack '{column}': give tracks in cell coordinates (Project.projected)")
    cells = pairs["cell"].astype(str).to_numpy()
    try:
        length = np.array([geometry[c].length_um for c in cells], dtype=float)
        width = np.array([geometry[c].max_width_um for c in cells], dtype=float)
    except KeyError as exc:
        raise AngleError(f"no outline geometry for cell {exc.args[0]!r}") from exc
    return length, width


def _uniform_in(half, radius, rng) -> np.ndarray:
    """One place a row drawn uniformly through the volume of its capsule (cylinder half-length ``half``, radius
    ``radius``; x along the axis)."""
    out = np.empty((len(half), 3))
    todo = np.arange(len(half))
    while len(todo):
        h, r = half[todo], radius[todo]
        x = rng.uniform(-1.0, 1.0, size=len(todo)) * (h + r)
        y = rng.uniform(-1.0, 1.0, size=len(todo)) * r
        z = rng.uniform(-1.0, 1.0, size=len(todo)) * r
        inside = (x - np.clip(x, -h, h)) ** 2 + y ** 2 + z ** 2 <= r ** 2
        out[todo[inside]] = np.stack([x[inside], y[inside], z[inside]], axis=1)
        todo = todo[~inside]
    return out


def measure_inset(pairs: pd.DataFrame, geometry: dict, sigma_um: np.ndarray, D: float, *, tau_s: float,
                  weights=None, exposure: float = 1.0, substeps: int = SUBSTEPS, seed: int = 0, link_um=None) -> float:
    """How much narrower in radius than their outlines (um; below 0 wider) these pairs' cells are where the molecules
    go: the inset at which free PAIRS -- molecules filling each pair's cell uniformly, walked at ``D`` over three
    exposures, given each localisation's error and LINKED through ``link_um`` as the observed ones were -- have their
    vertices as far across the cell, on average, as the pairs' vertices are (mean squared offset from the axis, each
    pair at its ``weights``). One number for the pairs; bisected, the offsets fall as the inset grows; held between
    `_MAX_WIDER_UM` wider and `_MIN_RADIUS_UM` of radius left; the cells and errors of at most `_WIDTH_PAIRS` pairs,
    drawn at the weights, walked.

    WHY (22 Nov 2026, on the lab's data drive): an outline is not the space the molecules fill. On the synthetic free mNG the outlines
    are 1.15 um wide and 99 % of the true positions lie within 0.475 um of the axis; the chance walked in the outline
    met the wall too little and read +0.35 (D 2) and +0.46 (D 5) on the true positions, +0.02 and +0.05 in a cell
    0.95 um wide. WHY LINKED: a pair the tracker linked is one whose molecule moved little in both frames, whose
    exposure average the wall pulled in less; measured over every molecule, a state at D 5 past a 5 px radius read
    its 0.9 um cell as wider than the 1.1 um outline."""
    n = len(pairs)
    if n == 0:
        raise AngleError("no pairs to measure the cells' width from")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    y = pairs["D_um"].to_numpy(dtype=float)
    target = float(np.sum(w * y ** 2) / np.sum(w))
    sig = np.asarray(sigma_um, dtype=float)
    pick = np.random.default_rng(seed).choice(n, size=min(n, _WIDTH_PAIRS), replace=n > _WIDTH_PAIRS,
                                              p=w / w.sum()) if n > _WIDTH_PAIRS else np.arange(n)
    wm = np.ones(len(pick)) if n > _WIDTH_PAIRS else w
    length, width = _cells(pairs.iloc[pick], geometry)
    m = _WIDTH_DRAWS
    many = lambda a: np.repeat(np.asarray(a, dtype=float), m, axis=0)          # noqa: E731
    half, wide, wm = many(np.maximum(length / 2.0 - width / 2.0, 0.0)), many(width), many(wm)
    noise = many(sig[pick])
    n_sub, open_half = _frame_steps(np.array([D]), tau_s, substeps, exposure)
    D_rep = np.full(len(half), max(float(D), 0.0))

    def offsets(inset: float) -> float:
        rng = np.random.default_rng(seed + 1)
        radius = np.maximum(wide / 2.0 - inset, _MIN_RADIUS_UM)
        shots = _three_shots(_uniform_in(half, radius, rng), half, radius, D_rep, noise, tau_s, n_sub, open_half, rng)
        if link_um is not None:
            todo = _unlinked(shots, link_um)
            for _ in range(_LINK_TRIES):
                rows = np.flatnonzero(todo)
                if not len(rows):
                    break
                again = _three_shots(_uniform_in(half[rows], radius[rows], rng), half[rows], radius[rows],
                                     D_rep[rows], noise[rows], tau_s, n_sub, open_half, rng)
                for shot, new in zip(shots, again):
                    shot[rows] = new
                todo[rows] = _unlinked(again, link_um)
            shots[1][todo] = np.nan                       # never linked: no observed pair is like it
        seen = shots[1][:, 1]
        kept = np.isfinite(seen)
        return float(np.sum(wm[kept] * seen[kept] ** 2) / np.sum(wm[kept])) if kept.any() else 0.0

    lo, hi = -_MAX_WIDER_UM, max(float(np.min(width)) / 2.0 - _MIN_RADIUS_UM, -_MAX_WIDER_UM)
    if offsets(lo) <= target:
        return lo
    if offsets(hi) >= target:
        return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        f = offsets(mid)
        if abs(f - target) <= _WIDTH_TOLERANCE * target or hi - lo < 1e-3:
            return mid
        lo, hi = (mid, hi) if f > target else (lo, mid)
    return 0.5 * (lo + hi)


def _unlinked(shots, link_um) -> np.ndarray:
    """The rows of three localisations with a step past the linking radius."""
    a, b = shots[1] - shots[0], shots[2] - shots[1]
    return (np.hypot(*a.T) > float(link_um)) | (np.hypot(*b.T) > float(link_um))


def _snippets(seen_x, seen_y, hr, rr, D_rep, sig, tau_s, substeps, open_half, rng):
    """One free molecule a row: its true start drawn around the seen place (the depth-weighted spread, module head),
    walked back and forward 1.5 frames, averaged over each frame's exposure, given each localisation's error; its two
    steps in the image."""
    spread = np.sqrt(EXPOSURE_R * (2.0 * open_half / substeps) * D_rep * tau_s + sig[:, 1] ** 2)
    x, y = seen_x.copy(), seen_y.copy()
    todo = np.ones(len(x), dtype=bool)
    for _ in range(_START_TRIES):
        k = int(todo.sum())
        if not k:
            break
        x[todo] = seen_x[todo] + rng.normal(0.0, 1.0, size=k) * spread[todo]
        y[todo] = seen_y[todo] + rng.normal(0.0, 1.0, size=k) * spread[todo]
        nearest = np.clip(x[todo], -hr[todo], hr[todo])
        reach = np.hypot(x[todo] - nearest, y[todo]) / rr[todo]
        kept = rng.uniform(size=k) < np.sqrt(np.clip(1.0 - reach ** 2, 0.0, None))
        todo[np.flatnonzero(todo)[kept]] = False
    axis = np.clip(x, -hr, hr)
    rho = np.hypot(x - axis, y)
    pull = np.where(rho > _INSIDE * rr, _INSIDE * rr / np.maximum(rho, 1e-12), 1.0)
    x, y = axis + (x - axis) * pull, y * pull
    depth = np.sqrt(np.maximum(rr ** 2 - (x - axis) ** 2 - y ** 2, 0.0))
    start = np.stack([x, y, rng.uniform(-1.0, 1.0, size=len(x)) * depth], axis=1)
    shots = _three_shots(start, hr, rr, D_rep, sig, tau_s, substeps, open_half, rng)
    return shots[1] - shots[0], shots[2] - shots[1]


def _three_shots(start, hr, rr, D_rep, sig, tau_s, substeps, open_half, rng) -> list:
    """A free molecule from ``start`` (the middle of the vertex exposure) walked back and forward 1.5 frames: the
    three localisations in the image (before, the vertex, after), each its exposure's average plus its own error."""
    sd = np.sqrt(2.0 * D_rep * tau_s / substeps)
    forward_vertex, after = _walk_average(start, sd, hr, rr, rng, substeps=substeps, open_half=open_half)
    backward_vertex, before = _walk_average(start, sd, hr, rr, rng, substeps=substeps, open_half=open_half)
    middle = forward_vertex + backward_vertex        # the vertex's own exposure: a half from each walk
    # each localisation's error, in the image's two axes (x along the cell, y across: any frame keeps the angle)
    shots = [before[:, :2], middle[:, :2], after[:, :2]]
    return [s + rng.normal(0.0, 1.0, size=s.shape) * sig[:, j:j + 1] for j, s in enumerate(shots)]


# -- the numbers ------------------------------------------------------------------------------------------------- #
def fold_ratio(angles, weights=None, *, window_deg: float = WINDOW_DEG) -> tuple:
    """``(ratio, backward, forward)``: the weighted share within ``window_deg`` of 180 over that within it of 0. The
    ratio is NaN where nothing carried on."""
    a = np.asarray(angles, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    backward = float(w[a >= 180.0 - window_deg].sum())
    forward = float(w[a <= window_deg].sum())
    return (backward / forward if forward > 0 else math.nan), backward, forward


def _density(angles, weights, edges) -> list:
    counts, _ = np.histogram(np.asarray(angles, dtype=float), bins=edges, weights=np.asarray(weights, dtype=float))
    total = counts.sum()
    width = edges[1] - edges[0]
    return [float(v) for v in (counts / (total * width) if total > 0 else counts * math.nan)]


def summarise(pairs: pd.DataFrame, chance, names: list, *, D_fit, D_chance, unit: str = "day",
              window_deg: float = WINDOW_DEG, n_bins: int = N_BINS) -> list:
    """Each state's numbers: its observed histogram (a density over ``n_bins`` bins of 0-180 degrees) and fold
    anisotropy, pooled and per replicate (``unit``), and -- where ``chance`` is given, ``chance[k]`` state k's
    ``(n, draws)`` angles (NaN where the pair was not walked for it) -- the same for chance and the excess. ``chance``
    None (the page's standard since 22 Nov 2026: "make comparison with simulation optional ... just show pure data"):
    the chance fields are NaN and ``D_chance`` may be None. A replicate's observed value needs `MIN_WEIGHT` pairs
    (``ok_observed``), its excess the chance too (``ok``)."""
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    angles = pairs["angle_deg"].to_numpy(dtype=float)
    units = pairs[unit].astype(str).to_numpy() if unit in pairs.columns else np.full(len(pairs), "")

    def spread(values):
        return (float(np.mean(values)) if values else math.nan,
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan)

    states = []
    for k, name in enumerate(names):
        w = pairs[f"w_{k + 1}"].to_numpy(dtype=float)
        obs_ratio, obs_back, obs_fwd = fold_ratio(angles, w, window_deg=window_deg)
        if chance is not None:
            walked = chance[k]
            draws = walked.shape[1]
            cw = np.repeat(w, draws) / draws
            ca = walked.ravel()
            good = np.isfinite(ca)
            ch_ratio = fold_ratio(ca[good], cw[good], window_deg=window_deg)[0]
            chance_density = _density(ca[good], cw[good], edges)
        else:
            ch_ratio, chance_density = math.nan, [math.nan] * n_bins
        replicates = []
        for key in sorted(set(units), key=lambda v: (len(v), v)):
            mask = units == key
            weight = float(w[mask].sum())
            o = fold_ratio(angles[mask], w[mask], window_deg=window_deg)[0]
            if chance is not None:
                cm = np.repeat(mask, draws) & good
                c = fold_ratio(ca[cm], cw[cm], window_deg=window_deg)[0]
            else:
                c = math.nan
            seen = weight >= MIN_WEIGHT and math.isfinite(o)
            usable = seen and math.isfinite(c)
            replicates.append({"unit": key, "weight": weight, "observed": o, "chance": c,
                               "excess": (o - c) if usable else math.nan, "ok": bool(usable),
                               "ok_observed": bool(seen)})
        excess_mean, excess_sd = spread([r["excess"] for r in replicates if r["ok"]])
        observed_mean, observed_sd = spread([r["observed"] for r in replicates if r["ok_observed"]])
        D_free = math.nan if D_chance is None else float(np.asarray(D_chance)[k])
        states.append({
            "name": name, "index": k + 1,
            "D_fit": float(np.asarray(D_fit, dtype=float)[k]), "D_chance": D_free,
            "at_floor": bool(math.isfinite(D_free) and D_free <= 0.0),
            "weight": float(w.sum()),
            "observed_density": _density(angles, w, edges),
            "chance_density": chance_density,
            "observed": obs_ratio, "chance": ch_ratio,
            "excess": (obs_ratio - ch_ratio) if math.isfinite(obs_ratio) and math.isfinite(ch_ratio) else math.nan,
            "n_backward": obs_back, "n_forward": obs_fwd,
            "replicates": replicates,
            "excess_mean": excess_mean, "excess_sd": excess_sd,
            "n_replicates": sum(1 for r in replicates if r["ok"]),
            "observed_mean": observed_mean, "observed_sd": observed_sd,
            "n_observed": sum(1 for r in replicates if r["ok_observed"]),
        })
    return states


# -- one condition, from the HMM section's result ---------------------------------------------------------------- #
def fit_from_result(result: dict, project=None, condition: str | None = None):
    """The HMM section's fit, rebuilt EXACTLY from its result (`hmm_view._fit_dict`: the off-diagonal rates are
    A / tau, so A is rates x tau with its rows completed to 1), with the noise model it was fitted under, for
    `hmm.decode_states`. ``project`` and ``condition`` resolve a Per spot fit's calibration when the result does not
    carry the Run's own (`hmm_view.precision_of`)."""
    from .hmm import HMMFit

    fit = result.get("fit") or {}
    if not fit:
        raise AngleError("the HMM result carries no fit")
    tau = float(result["tau_s"])
    K = int(fit["n_states"])
    A = np.asarray(fit["rates"], dtype=float) * tau
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(A, 1.0 - A.sum(axis=1))
    settings = result.get("settings") or {}
    sigma_um, precision = 0.0, None
    if settings.get("sigma") == "per_spot" and not settings.get("precision_off"):
        from .hmm_view import precision_of

        given = settings.get("precision")
        calibration = precision_of(project, condition, "per_spot", given=tuple(given) if given else None)
        precision = (float(calibration.c_nm), float(calibration.p))
    elif settings.get("sigma") == "tracker":
        from .hmm_view import _sigma_um

        sigma_um = _sigma_um(project, "tracker")[0]
    initial = fit.get("initial")
    return HMMFit(n_states=K, D=np.asarray(fit["D"], dtype=float), transition_matrix=A,
                  occupancy=np.asarray(fit["occupancy"], dtype=float),
                  stationary=np.asarray(fit["stationary"], dtype=float), tau_s=tau,
                  n_steps=int(fit.get("n_steps", 0)), n_sequences=int(fit.get("n_tracks", 0)),
                  log_likelihood=math.nan, bic=float(fit.get("bic", math.nan)), localization_sigma_um=float(sigma_um),
                  initial_distribution=None if initial is None else np.asarray(initial, dtype=float),
                  precision_calibration=precision)


def noise_for(project, condition: str, pairs: pd.DataFrame, fit) -> tuple:
    """``(sigma (n, 3) µm, words)``: each pair's three localisation errors for the chance, and the words naming
    them (module head). Raises AngleError where no error is known."""
    from .precision import Calibration

    reported = [f"reported_sigma_{j}" for j in range(3)]
    calibration = None
    if fit.precision_calibration is not None:
        calibration = Calibration(c_nm=fit.precision_calibration[0], p=fit.precision_calibration[1],
                                  source="the HMM fit's own")
    elif all(c in pairs.columns for c in reported) and np.isfinite(pairs[reported].to_numpy(dtype=float)).any():
        from .hmm_view import precision_of

        try:
            calibration = precision_of(project, condition, "per_spot")
        except ValueError:
            calibration = None
    if calibration is not None and all(c in pairs.columns for c in reported):
        sigma = calibration.true_sigma_um(pairs[reported].to_numpy(dtype=float))
        typical = float(np.nanmedian(sigma)) if np.isfinite(sigma).any() else math.nan
        if not math.isfinite(typical):
            raise AngleError("the spots carry no usable per-spot precision")
        sigma = np.where(np.isfinite(sigma), sigma, typical)
        words = (f"each spot's own, through the per-spot calibration "
                 f"(median {1000 * typical:.0f} nm)")
    else:
        if fit.localization_sigma_um > 0:
            static = float(fit.localization_sigma_um)
            words = f"the HMM's, {1000 * static:.0f} nm"
        else:
            from .hmm_view import _sigma_um

            try:
                static, provenance = _sigma_um(project, "tracker")
            except ValueError as exc:
                raise AngleError("the chance needs the localisation error, and these tracks carry no per-spot "
                                 "precision and the project records no sigma") from exc
            words = f"the project's, {provenance}"
        sigma = np.full((len(pairs), 3), static)
    return sigma, words


def linking_for(project, condition: str) -> tuple:
    """``(radius_um, words)``: the largest step ``condition``'s tracker linked (its recorded search radius, u-track's
    cap on its adaptive radius, TrackMate's LINKING_MAX_DISTANCE; `AcquisitionParams.search_radius_px_for`), which the
    chance's free pairs go through as the observed ones did (`_chance_steps`); ``(None, words)`` where none is recorded
    (a track table): the chance then links every step, and a fast state reads too high beyond chance."""
    from .canonical import CanonicalError

    params = project.params
    try:
        radius_px = params.search_radius_px_for(condition)
    except CanonicalError:
        radius_px = None
    if radius_px is None or not params.um_per_pixel:
        return None, "no linking radius recorded: every free step linked"
    radius_um = float(radius_px) * float(params.um_per_pixel)
    return radius_um, f"the tracker's linking radius, {radius_px:g} px ({1000 * radius_um:.0f} nm)"


def exposure_for(project, condition: str) -> tuple:
    """``(share, words)``: the share of each frame ``condition``'s camera was open, from its movies' camera files
    (`project.camera_exposures`: the exposure over each movie's measured frame interval, the median over its movies),
    which the chance's camera copies; ``(None, words)`` where no file records one (a track table, a movie without its
    raw acquisition beside it): the chance's camera is then open the whole frame. One user's movies: 20 ms of each 25."""
    from .project import camera_exposures

    root = (getattr(project, "roots", None) or {}).get(condition)
    if root is None:
        return None, "the whole frame (no camera file to read the exposure from)"
    found = camera_exposures(root)
    try:
        movies = set(project.projected(condition)["movie"].astype(str))
    except (KeyError, AttributeError):
        movies = set()
    if movies and movies & set(found):
        found = {k: v for k, v in found.items() if k in movies}
    tau_ms = float(getattr(project.params, "frame_interval_ms", 0.0) or 0.0)
    rows = [(float(e), float(i or tau_ms)) for e, i, _file in found.values() if e and (i or tau_ms)]
    if not rows:
        return None, "the whole frame (the camera files record no exposure)"
    shares = np.array([min(e / i, 1.0) for e, i in rows])
    share = float(np.median(shares))
    exposure_ms = float(np.median([e for e, _ in rows]))
    interval_ms = float(np.median([i for _, i in rows]))
    words = f"{exposure_ms:g} ms of each {interval_ms:.1f} ms frame, from the camera files"
    if shares.max() - shares.min() > 0.05:
        words += f" (its share {shares.min():.2f}-{shares.max():.2f} across movies; the median taken)"
    return share, words


def chance_by_state(pairs: pd.DataFrame, geometry: dict, sigma: np.ndarray, fit, *, draws: int = CHANCE_DRAWS,
                    seed: int = 0, link_um=None, exposure: float = 1.0, width: str = "measured") -> tuple:
    """``(chance, D, cell)``: for each state k, ``chance[k]`` the ``(n, draws)`` free molecules' angles at every pair
    (NaN where its weight for k is under `_MIN_PAIR_WEIGHT`, not walked), the free D each state was walked at -- its
    steps as long as the state's observed ones, at these places (`match_D`) -- and the cell they were walked in
    (``{"inset_um", "outline_width_um", "width_um", "state", "D", "measured"}``). The one chance the Jump angles page
    and the HMM map's Turning back both read.

    THE CELL (``width`` "measured", the standard since 22 Nov 2026): every state is walked in cells narrower than their
    outlines by the inset at which free molecules lie as far across the cell as the FASTEST state's do (`measure_inset`,
    at its matched D, then the D matched again in that cell, then the inset once more); the fastest state past the
    linking limit is passed over for the next; one slower than `_WIDTH_MIN_D` keeps the outlines (its places are where
    it stays). ``width`` "outline": the outlines as drawn. A state PAST THE LINKING LIMIT (`match_D` inf) has no chance:
    its angles NaN, its D NaN, listed in ``cell["past_limit"]``."""
    squared = 0.5 * (pairs["step_prev_um"].to_numpy(dtype=float) ** 2
                     + pairs["step_next_um"].to_numpy(dtype=float) ** 2)
    n_states = int(fit.n_states)
    weights = {k: pairs[f"w_{k + 1}"].to_numpy(dtype=float) for k in range(n_states)}
    walks = {k: weights[k] > _MIN_PAIR_WEIGHT for k in range(n_states)}

    def matched(k, inset):
        w, walk = weights[k], walks[k]
        target = float(np.sum(w * squared) / np.sum(w))
        return match_D(pairs[walk], geometry, sigma[walk], target, tau_s=fit.tau_s, weights=w[walk],
                       seed=seed + 100 + k, link_um=link_um, exposure=exposure, inset_um=inset)

    if width not in ("measured", "outline"):
        raise AngleError(f"the chance's cell width must be 'measured' or 'outline', got {width!r}")
    outline = float(np.median(_cells(pairs, geometry)[1]))
    cell = {"inset_um": 0.0, "outline_width_um": outline, "width_um": outline, "state": None, "D": None,
            "measured": False, "past_limit": []}
    D_fit = np.asarray(getattr(fit, "D", np.arange(n_states)), dtype=float)
    moving = [k for k in range(n_states) if walks[k].any()]
    known = {}
    if width == "measured":
        for k in sorted(moving, key=lambda j: -D_fit[j]):
            D0 = matched(k, 0.0)
            if not math.isfinite(D0):
                continue                                  # past the linking limit: its places are not the cell's
            cell.update(state=k, D=float(D0))
            if D0 >= _WIDTH_MIN_D:
                w, walk = weights[k], walks[k]
                inset, D_now = 0.0, D0
                for _ in range(2):
                    inset = measure_inset(pairs[walk], geometry, sigma[walk], D_now, tau_s=fit.tau_s,
                                          weights=w[walk], exposure=exposure, seed=seed + 300, link_um=link_um)
                    D_now = matched(k, inset)
                    if not math.isfinite(D_now):
                        break
                if not math.isfinite(D_now):              # past the limit in its own cell: its places are not the
                    cell.update(state=None, D=None)        # cell's either; the next state is asked
                    continue
                cell.update(inset_um=float(inset), width_um=outline - 2.0 * float(inset), D=float(D_now),
                            measured=True)
                known[k] = D_now
            else:
                known[k] = D0
            break
    D_free = np.zeros(n_states)
    chance = {}
    for k in range(n_states):
        walk = walks[k]
        angles = np.full((len(pairs), int(draws)), np.nan)
        if walk.any():
            D_free[k] = known[k] if k in known else matched(k, cell["inset_um"])
            if math.isfinite(D_free[k]):
                angles[walk] = chance_angles(pairs[walk], geometry, np.full(int(walk.sum()), D_free[k]),
                                             sigma[walk], tau_s=fit.tau_s, draws=int(draws), seed=seed + k,
                                             link_um=link_um, exposure=exposure, inset_um=cell["inset_um"])
            else:
                cell["past_limit"].append(k)
                D_free[k] = math.nan
        chance[k] = angles
    return chance, D_free, cell


def cell_words(cell: dict, names) -> str:
    """The chance's cell in words (`chance_by_state`'s ``cell``)."""
    if not cell:
        return "each cell's outline"
    if not cell.get("measured"):
        if cell.get("state") is None:
            return f"each cell's outline (median {cell['outline_width_um']:.2f} µm wide)"
        return (f"each cell's outline (median {cell['outline_width_um']:.2f} µm wide; the {names[cell['state']]} "
                f"state, D {cell['D']:.2f} µm²/s, is too slow to show the cell's width)")
    return (f"{cell['width_um']:.2f} µm wide (median), where the {names[cell['state']]} state's molecules lie; the "
            f"outlines {cell['outline_width_um']:.2f} µm")


def compute_hmm_angles(project, condition: str, *, hmm: dict, chance: bool = False,
                       draws: int = CHANCE_DRAWS, seed: int = 0) -> dict:
    """One condition's jump angles per HMM state, from the HMM section's result ``hmm`` for it; with ``chance`` (the
    page's Against chance, off by default since 22 Nov 2026) beside free molecules at the same places too.

    Picklable, run on the process pool (`sectioncache.cached_job` section "hmm_angles"). An expected refusal comes
    back as ``{"condition": ..., "error": "<Type>: <message>"}``; anything else raises."""
    from .hmm import HMMError, decode_states

    settings = {"hmm_settings": dict(hmm.get("settings") or {}), "fit": hmm.get("fit"),
                "tau_s": hmm.get("tau_s"), "chance": bool(chance), "draws": int(draws), "seed": int(seed)}
    if hmm.get("error"):
        return {"condition": condition, "settings": settings, "error": f"HMM: {hmm['error']}"}
    try:
        fit = fit_from_result(hmm, project, condition)
        projected = project.projected(condition)
        geometry = projected.attrs.get("geometry") or {}
        decoded = decode_states(projected, fit)
        pairs = pair_table(projected, decoded, fit.n_states)
        names = state_names(fit.n_states)
        link_um, link_words, share, exposure_words, cell = None, "", None, "", None
        if chance:
            sigma, noise_words = noise_for(project, condition, pairs, fit)
            link_um, link_words = linking_for(project, condition)
            share, exposure_words = exposure_for(project, condition)
            walked, D_free, cell = chance_by_state(pairs, geometry, sigma, fit, draws=draws, seed=seed,
                                                   link_um=link_um, exposure=1.0 if share is None else share)
        else:
            walked, D_free, noise_words = None, None, ""
        states = summarise(pairs, walked, names, D_fit=fit.D, D_chance=D_free)
        for k in (cell or {}).get("past_limit", ()):
            states[k]["past_limit"] = True
    except (AngleError, HMMError, ValueError) as exc:
        return {"condition": condition, "settings": settings, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "condition": condition,
        "settings": settings,
        "n_states": int(fit.n_states),
        "chance": bool(chance),
        "names": names,
        "states": states,
        "edges_deg": [float(e) for e in np.linspace(0.0, 180.0, N_BINS + 1)],
        "window_deg": float(WINDOW_DEG),
        "n_pairs": int(len(pairs)),
        "n_tracks": int(pairs["uid"].nunique()),
        "switch_share": float(pairs["w_switch"].sum() / len(pairs)),
        "zero_step": int(pairs.attrs.get("zero_step", 0)),
        "undecoded": int(pairs.attrs.get("undecoded", 0)),
        "noise_words": noise_words,
        "link_um": link_um, "link_words": link_words,
        "exposure": share, "exposure_words": exposure_words,
        "cell": cell, "cell_words": cell_words(cell, names) if cell else "",
        "d_word": "per-spot" if fit.precision_calibration is not None else (
            "σ-corrected" if fit.localization_sigma_um > 0 else "apparent"),
        "exposure_r": float(EXPOSURE_R) * (1.0 if share is None else share), "substeps": int(SUBSTEPS),
        "draws": int(draws),
        "tau_s": float(fit.tau_s),
    }
