"""Confinement: dwell events inside a moving ROI, with the chance rate attached.

The legacy method (`confTracks.m` → `dwellCount2.m` → `tauCounter3.m`)
------------------------------------------------------------------------
For one track, build the boolean matrix ``inside[i, j] = |p_i − p_j| < radius``.
Column ``j`` is a circle of that radius centred on localization ``j``. Walking
down a column gives the runs of *consecutive* localizations that stay inside
that circle; a run of ``m`` points is a dwell event of ``m − 1`` steps. The
column with the longest dwell wins, that dwell is recorded, its interior points
are masked out, and the search repeats until no run of ``min_steps`` survives.
A track is then

- **free**: no dwell event at all,
- **confined**, a single dwell spanning the whole track,
- **mixed**, dwells plus free excursions (the interesting case).

This is reproduced faithfully, including two quirks worth naming:

1. **The one-gap allowance is shared across candidate circles.** `tauCounter3`
   initialises `counterOut` once, *outside* the column loop, and never resets
   it, so within one search pass only the first single-point excursion
   encountered is bridged, and which circle that is depends on scan order, not
   on the data. The comment says "allow 1 gap"; the code allows one gap per
   pass. ``gap_mode="per_run"`` implements the documented intent so the gap can
   be measured (on free Brownian tracks the confined-segment fraction moves
   0.74 → 0.79 → 0.80 for ``none`` → ``legacy`` → ``per_run``); ``"legacy"``
   (default) reproduces the shipped behaviour.
2. **Masking keeps the run's endpoints.** Only interior points are removed
   (`i_remove(2:end-1)`), so two dwell events may share a boundary localization.
   Deliberate, and kept.

Why a null model is not optional here
-------------------------------------
A free random walk revisits its own neighbourhood. Run this detector on purely
Brownian tracks and it *still* reports confinement, the rate rises as the
radius grows and as tracks get longer, with no biology anywhere in the
simulation. A confined fraction quoted without its chance rate is therefore
uninterpretable. ``confinement_null`` runs the identical detector on free
Brownian tracks matched to the real track-length distribution, and
``radius_sensitivity`` shows how the answer moves with the two hand-set
parameters. Both report **excess over chance**, which is the only part of the
number that can carry a claim.

What this method is FOR, now
----------------------------
Not kinetics. ``smtracker.hmm`` measures state switching properly (per-step,
with a likelihood), so dwell *times* are its job. Confinement earns its place on
the **spatial** axis: it labels *which localizations* sit in a dwell zone, which
feeds ``smtracker.maps`` and answers "where does the molecule stop?".

Invariants
----------
- ``radius_um > 0``, ``min_steps >= 1``.
- Tracks are split on frame gaps before detection: a bridged gap is not a step.
- Nothing is returned as "confined" that the same detector does not also flag on
  the matched Brownian null more rarely: the excess is always reported.
- Fails loud on a non-canonical table or an empty result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS, AcquisitionParams
from .simulate import matched_brownian_null
from .canonical import without_attrs

__all__ = [
    "ConfinementError",
    "GAP_MODES",
    "TRACK_CLASSES",
    "dwell_events",
    "confinement_events",
    "label_confinement",
    "ConfinementNull",
    "confinement_null",
    "confined_segments",
    "track_dwells",
    "dwell_index",
    "track_transitions",
    "transitions",
    "transition_null",
    "LEGACY_MIN_STEPS_BETWEEN",
    "radius_sensitivity",
]

#: How the single-point excursion allowance behaves. See the module head.
GAP_MODES = ("legacy", "per_run", "none")

#: Per-track verdicts.
TRACK_CLASSES = ("free", "mixed", "confined")

_MASKED = np.inf


class ConfinementError(RuntimeError):
    """Raised when confinement detection is asked for something impossible."""


# --------------------------------------------------------------------------- #
# The detector, one track at a time
# --------------------------------------------------------------------------- #


def _column_runs(column: np.ndarray, min_steps: int, gaps_left: list[int], gap_mode: str):
    """Runs of consecutive in-circle localizations down one column.

    Returns ``(taus, index_arrays)`` for runs of at least ``min_steps`` steps.
    ``gaps_left`` is a one-element list so the legacy call-wide counter can be
    shared across columns exactly as MATLAB's un-reset ``counterOut`` is.
    """
    n = column.size
    nonzero = np.flatnonzero(column != 0)
    if nonzero.size == 0:
        return [], []
    start = int(nonzero[0])

    per_run = gap_mode == "per_run"
    if per_run:
        gaps_left = [1]

    runs: list[list[int]] = [[]]
    for i in range(start, n):
        value = column[i]
        keep = value == 1
        if not keep and value == 0 and gaps_left[0] > 0 and 0 < i < n - 1:
            # The legacy bridges a single point that is outside a circle whose
            # two neighbours are inside (their three values sum to exactly 2).
            if column[i - 1] + column[i] + column[i + 1] == 2:
                gaps_left[0] -= 1
                keep = True
        if keep:
            runs[-1].append(i)
        elif runs[-1]:
            runs.append([])
            if per_run:
                gaps_left[0] = 1        # the allowance is per run, as documented

    taus, indexes = [], []
    for run in runs:
        tau = len(run) - 1
        if tau >= min_steps:
            taus.append(tau)
            indexes.append(np.asarray(run, dtype=int))
    return taus, indexes


def _best_column(inside: np.ndarray, min_steps: int, gap_mode: str):
    """The circle centre whose longest dwell is greatest (first wins on ties)."""
    gaps_left = [1 if gap_mode != "none" else 0]
    best_taus: list[int] = []
    best_indexes: list[np.ndarray] = []
    for col in range(inside.shape[1]):
        taus, indexes = _column_runs(inside[:, col], min_steps, gaps_left, gap_mode)
        if taus and (not best_taus or max(taus) > max(best_taus)):
            best_taus, best_indexes = taus, indexes
    return best_taus, best_indexes


def dwell_events(
    xy: np.ndarray,
    *,
    radius_um: float,
    min_steps: int = 3,
    gap_mode: str = "legacy",
) -> tuple[list[int], list[np.ndarray]]:
    """Dwell events of one track: ``(taus_in_steps, localization_indexes)``.

    Greedy, exactly as the legacy: take the longest dwell, mask its interior,
    repeat. Indexes are 0-based positions within the track.
    """
    if gap_mode not in GAP_MODES:
        raise ConfinementError(f"gap_mode must be one of {GAP_MODES}, got {gap_mode!r}")
    if radius_um <= 0:
        raise ConfinementError(f"radius_um must be > 0, got {radius_um}")
    if min_steps < 1:
        raise ConfinementError(f"min_steps must be >= 1, got {min_steps}")

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ConfinementError(f"xy must be (n, 2), got {xy.shape}")
    n = xy.shape[0]
    if n < min_steps + 1:
        return [], []

    distance = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    inside = (distance < radius_um).astype(float)

    taus, indexes = _best_column(inside, min_steps, gap_mode)
    if not taus:
        return [], []
    # A single dwell spanning the whole track: nothing left to search.
    if len(taus) == 1 and taus[0] == n - 1:
        return list(taus), list(indexes)

    kept_taus: list[int] = []
    kept_indexes: list[np.ndarray] = []
    while taus:
        if sum(kept_taus) > n - 1:
            raise ConfinementError(
                "more dwell steps than the track has steps: detector is inconsistent"
            )
        best = int(np.argmax([len(idx) for idx in indexes]))
        kept_taus.append(taus[best])
        kept_indexes.append(indexes[best])

        remove = indexes[best]
        if remove.size >= 3:
            remove = remove[1:-1]        # endpoints stay available (legacy)
        inside[remove, :] = _MASKED
        inside[:, remove] = _MASKED

        taus, indexes = _best_column(inside, min_steps, gap_mode)

    order = np.argsort([idx[0] for idx in kept_indexes])
    return [kept_taus[i] for i in order], [kept_indexes[i] for i in order]


# --------------------------------------------------------------------------- #
# Whole-dataset detection
# --------------------------------------------------------------------------- #


def _contiguous_runs(frames: np.ndarray) -> list[np.ndarray]:
    """Split positions into runs of consecutive frames."""
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    return [r for r in np.split(np.arange(frames.size), breaks) if r.size]


#: Key under which ``dwell_index`` records the settings it was built with, so a consumer
#: handed an index built at another radius refuses instead of quietly mixing two.
_INDEX_SETTINGS = "__settings__"


def track_dwells(xy, frames, *, radius_um, min_steps=3, gap_mode="legacy"):
    """Dwell events of ONE track, gap-aware, in the track's own row positions.

    The one source of truth for "where does this track dwell". Three callers used to ask
    that question and two of them asked it differently: ``confinement_events`` split the
    track on frame gaps first (``_contiguous_runs``), while ``transitions`` and
    ``confined_segments`` ran ``dwell_events`` over the whole track. A frame gap is not a
    step, so a molecule seen at frames 3 and 9 is not dwelling for six steps between them
    the split is the correct treatment and the whole-track call was the loose one.

    It did not bite on this data, and that is not the same as being harmless. Measured on
    the reference project, **no track carries a frame gap**, because u-track ran with
    ``timeWindow = 1``. So the three paths agreed here by a property of the tracker, not
    by construction. TrackMate does close gaps, and the readers are deliberately
    source-agnostic downstream, so the divergence was waiting for the second tracker.

    Returns ``(taus_in_steps, indexes)`` with indexes as positions in the track as passed.
    """
    frames = np.asarray(frames)
    xy = np.asarray(xy, dtype=float)
    if frames.shape[0] != xy.shape[0]:
        raise ConfinementError(
            f"frames and xy disagree: {frames.shape[0]} against {xy.shape[0]}")
    taus: list[int] = []
    indexes: list[np.ndarray] = []
    for run in _contiguous_runs(frames):
        run_taus, run_indexes = dwell_events(xy[run], radius_um=radius_um,
                                             min_steps=min_steps, gap_mode=gap_mode)
        taus.extend(run_taus)
        indexes.extend(run[idx] for idx in run_indexes)
    return taus, indexes


def dwell_index(canonical, *, radius_um, min_steps=3, gap_mode="legacy"):
    """``{uid: (taus, indexes)}`` for every track, computed ONCE.

    The detection is the expensive part: measured on the reference project,
    ``confinement_events`` took 9.44 s and ``transitions`` a further 7.40 s recomputing
    the identical events, 16.84 s for one radius. Pass this to both and the second is
    nearly free.

    ``.attrs``-style provenance travels on the returned mapping's companion dict so the
    consumers can refuse a mismatched index rather than silently mixing radii.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise ConfinementError(f"not a canonical table (missing {missing})")
    if canonical.empty:
        raise ConfinementError("no localizations to analyse")

    out: dict = {}
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    for uid, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        out[uid] = track_dwells(track[["x_um", "y_um"]].to_numpy(),
                                track["frame"].to_numpy(), radius_um=radius_um,
                                min_steps=min_steps, gap_mode=gap_mode)
    out[_INDEX_SETTINGS] = (float(radius_um), int(min_steps), str(gap_mode))
    return out


def _dwells_for(canonical, dwells, *, radius_um, min_steps, gap_mode):
    """Return a ``{uid: (taus, indexes)}`` mapping, computing it only if not supplied."""
    if dwells is None:
        return dwell_index(canonical, radius_um=radius_um, min_steps=min_steps,
                           gap_mode=gap_mode)
    got = dwells.get(_INDEX_SETTINGS)
    want = (float(radius_um), int(min_steps), str(gap_mode))
    if got is None:
        raise ConfinementError(
            "the dwell index carries no settings, so it cannot be checked against this "
            "call. Build it with dwell_index()")
    if got != want:
        raise ConfinementError(
            f"the dwell index was built at radius {got[0]} um, min_steps {got[1]}, "
            f"gap_mode {got[2]!r} and this call asks for {want[0]} um, {want[1]}, "
            f"{want[2]!r}. Reusing it would report one radius under another's name")
    return dwells


def confinement_events(
    canonical: pd.DataFrame,
    *,
    radius_um: float,
    min_steps: int = 3,
    gap_mode: str = "legacy",
    dwells: dict | None = None,
) -> pd.DataFrame:
    """Per-track confinement verdicts and dwell events.

    ``dwells`` is an optional ``dwell_index`` result. Supplying one shares the detection
    with ``transitions`` / ``confined_segments`` instead of repeating it.

    Returns one row per track: ``uid``, ``cell``, ``n_localizations``,
    ``n_dwells``, ``longest_dwell_steps``, ``n_confined_points``,
    ``confined_fraction`` (of that track's localizations) and ``track_class``.
    Dataset-level fractions land in ``.attrs``.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise ConfinementError(f"not a canonical table (missing {missing})")
    if canonical.empty:
        raise ConfinementError("no localizations to analyse")

    index = _dwells_for(canonical, dwells, radius_um=radius_um, min_steps=min_steps,
                        gap_mode=gap_mode)
    rows = []
    confined_positions: dict[str, np.ndarray] = {}
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    for uid, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        frames = track["frame"].to_numpy()
        taus, indexes = index[uid]
        flagged: list[int] = []
        for idx in indexes:
            flagged.extend(np.asarray(idx).tolist())

        unique_flagged = np.unique(flagged) if flagged else np.empty(0, dtype=int)
        confined_positions[uid] = unique_flagged
        n = len(frames)
        if not taus:
            verdict = "free"
        elif max(taus) + 1 == n:
            verdict = "confined"
        else:
            verdict = "mixed"
        rows.append(
            {
                "uid": uid,
                "cell": track["cell"].iloc[0],
                "condition": track["condition"].iloc[0],
                "day": track["day"].iloc[0],
                "n_localizations": n,
                "n_dwells": len(taus),
                "longest_dwell_steps": int(max(taus)) if taus else 0,
                "n_confined_points": int(unique_flagged.size),
                "confined_fraction": float(unique_flagged.size / n),
                "track_class": verdict,
            }
        )

    table = pd.DataFrame(rows)
    total_points = int(table["n_localizations"].sum())
    table.attrs.update(
        {
            "radius_um": float(radius_um),
            "min_steps": int(min_steps),
            "gap_mode": gap_mode,
            "n_tracks": int(len(table)),
            "confined_segment_fraction": float(
                table["n_confined_points"].sum() / total_points
            ),
            "confined_positions": confined_positions,
        }
    )
    for verdict in TRACK_CLASSES:
        table.attrs[f"fraction_{verdict}"] = float(
            (table["track_class"] == verdict).mean()
        )
    return table


def label_confinement(
    canonical: pd.DataFrame,
    *,
    radius_um: float,
    min_steps: int = 3,
    gap_mode: str = "legacy",
) -> pd.DataFrame:
    """``canonical`` plus a per-localization ``confined`` flag and ``track_class``.

    This is the spatial handle: feed it to ``smtracker.maps`` to ask *where* the
    dwell zones are.
    """
    events = confinement_events(
        canonical, radius_um=radius_um, min_steps=min_steps, gap_mode=gap_mode
    )
    positions = events.attrs["confined_positions"]
    classes = dict(zip(events["uid"], events["track_class"]))

    out = canonical.sort_values(["uid", "frame"], kind="stable").reset_index(drop=True)
    confined = np.zeros(len(out), dtype=bool)
    offset = 0
    for uid, track in without_attrs(out).groupby("uid", sort=False, observed=True):
        idx = positions.get(uid, np.empty(0, dtype=int))
        if idx.size:
            confined[offset + idx] = True
        offset += len(track)
    out["confined"] = confined
    out["track_class"] = out["uid"].map(classes)
    out.attrs.update(canonical.attrs)
    out.attrs["confinement"] = {k: v for k, v in events.attrs.items()
                               if k != "confined_positions"}
    return out


# --------------------------------------------------------------------------- #
# The chance rate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfinementNull:
    """What the same detector finds on free Brownian tracks.

    ``excess`` is observed − chance. If it is not comfortably positive, the
    confinement measurement carries no claim, whatever the raw fraction says.
    """

    radius_um: float
    min_steps: int
    D_um2_s: float
    sigma_um: float
    observed: dict[str, float]
    chance: dict[str, float]
    chance_sd: dict[str, float]
    n_replicates: int
    caveat: str = field(default="")

    @property
    def excess(self) -> dict[str, float]:
        return {k: self.observed[k] - self.chance[k] for k in self.observed}

    def report(self) -> str:
        lines = [
            f"Confinement at radius {1000 * self.radius_um:.0f} nm, "
            f"min_steps {self.min_steps} "
            f"(null: free Brownian, D = {self.D_um2_s:.3g} µm²/s, "
            f"{self.n_replicates} replicates)",
            f"{'quantity':<28}{'observed':>10}{'chance':>10}{'± sd':>8}{'excess':>10}",
        ]
        for key in self.observed:
            lines.append(
                f"{key:<28}{self.observed[key]:>10.3f}{self.chance[key]:>10.3f}"
                f"{self.chance_sd[key]:>8.3f}{self.excess[key]:>+10.3f}"
            )
        if self.caveat:
            lines.append(f"CAVEAT: {self.caveat}")
        return "\n".join(lines)


_NULL_KEYS = ("fraction_confined", "fraction_mixed", "confined_segment_fraction")


def confinement_null(
    canonical: pd.DataFrame,
    *,
    radius_um: float,
    D_um2_s: float,
    min_steps: int = 3,
    sigma_um: float = 0.0,
    n_replicates: int = 5,
    gap_mode: str = "legacy",
    seed: int = 0,
) -> ConfinementNull:
    """Run the detector on the data and on matched free Brownian tracks.

    ``D_um2_s`` has no default on purpose: the right null is "a molecule that is
    genuinely free", so the number must come from the analysis (typically the
    fast state of the SQD or HMM fit), not from this function's imagination.
    """
    if n_replicates < 1:
        raise ConfinementError(f"n_replicates must be >= 1, got {n_replicates}")
    params = canonical.attrs.get("params")
    if not isinstance(params, AcquisitionParams):
        raise ConfinementError("canonical table carries no AcquisitionParams")

    real = confinement_events(
        canonical, radius_um=radius_um, min_steps=min_steps, gap_mode=gap_mode
    )
    observed = {key: float(real.attrs[key]) for key in _NULL_KEYS}

    draws: dict[str, list[float]] = {key: [] for key in _NULL_KEYS}
    for replicate in range(n_replicates):
        null_tracks = matched_brownian_null(
            canonical, D_um2_s=D_um2_s, sigma_um=sigma_um, seed=seed + replicate
        )
        null_events = confinement_events(
            null_tracks, radius_um=radius_um, min_steps=min_steps, gap_mode=gap_mode
        )
        for key in _NULL_KEYS:
            draws[key].append(float(null_events.attrs[key]))

    chance = {k: float(np.mean(v)) for k, v in draws.items()}
    chance_sd = {k: float(np.std(v, ddof=1)) if n_replicates > 1 else float("nan")
                 for k, v in draws.items()}

    caveat = ""
    step_um = np.sqrt(4.0 * D_um2_s * params.tau_s)
    if radius_um > step_um:
        caveat = (
            f"the confinement radius ({1000 * radius_um:.0f} nm) exceeds one free "
            f"step at D = {D_um2_s:.3g} ({1000 * step_um:.0f} nm), so a free "
            "molecule rarely leaves the circle and the chance rate is high"
        )
    return ConfinementNull(
        radius_um=float(radius_um),
        min_steps=int(min_steps),
        D_um2_s=float(D_um2_s),
        sigma_um=float(sigma_um),
        observed=observed,
        chance=chance,
        chance_sd=chance_sd,
        n_replicates=int(n_replicates),
        caveat=caveat,
    )


def radius_sensitivity(
    canonical: pd.DataFrame,
    *,
    radii_um,
    D_um2_s: float,
    min_steps_values=(3,),
    sigma_um: float = 0.0,
    n_replicates: int = 3,
    gap_mode: str = "legacy",
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep the two hand-set parameters and report observed, chance and excess.

    The radius and ``min_steps`` are not measured quantities: they are choices.
    A confinement result that only exists at one radius is a property of the
    choice, not of the molecule, and this table is what makes that visible.
    """
    radii = np.atleast_1d(np.asarray(radii_um, dtype=float))
    if radii.size == 0:
        raise ConfinementError("no radii to sweep")

    rows = []
    for min_steps in np.atleast_1d(min_steps_values):
        for radius in radii:
            null = confinement_null(
                canonical,
                radius_um=float(radius),
                D_um2_s=D_um2_s,
                min_steps=int(min_steps),
                sigma_um=sigma_um,
                n_replicates=n_replicates,
                gap_mode=gap_mode,
                seed=seed,
            )
            for key in _NULL_KEYS:
                rows.append(
                    {
                        "radius_um": float(radius),
                        "min_steps": int(min_steps),
                        "quantity": key,
                        "observed": null.observed[key],
                        "chance": null.chance[key],
                        "chance_sd": null.chance_sd[key],
                        "excess": null.excess[key],
                    }
                )
    table = pd.DataFrame(rows)
    table.attrs["D_um2_s"] = float(D_um2_s)
    table.attrs["n_replicates"] = int(n_replicates)
    return table


# =====================================================================================
# TRANSITIONS BETWEEN FREE AND CONFINED (``Help_fcn/nTrans.m``)
# =====================================================================================
#
# THE LEGACY RULE, reproduced. A track has changed state when the stretch between two
# dwells -- or before the first, or after the last -- has a root-mean-square single-step
# displacement LARGER THAN THE DWELL RADIUS. Leaving a dwell and entering the next counts
# as one of each.
#
# **THE MINIMUM LENGTH IS APPLIED IN THE MIDDLE AND NOT AT THE ENDS.** ``nTrans.m:52``
# guards an interior gap with ``first - last > 2``, at least three steps, matching its own
# comment. The head and tail branches have NO such guard, so a single step before the
# first dwell can register a transition while three steps between two dwells may not.
# Both are reproduced: ``strict=False`` is the legacy, ``strict=True`` is consistent, and
# both counts are reported so the gap is measured rather than argued.
#
# ``timeAvgMsd`` inside that file is misnamed: it is the mean SQUARED single-step
# displacement, not a time-averaged MSD over lags. Its square root is an RMS step, which
# is then compared against a RADIUS -- a choice, not a measurement, reproduced as such.
#
# AND THERE IS NO CHANCE RATE, exactly as for the confined fraction. A free Brownian track
# wanders in and out of any circle, so it produces transitions with no state change in it.

#: ``nTrans.m:52``, ``first - last > 2``.
LEGACY_MIN_STEPS_BETWEEN = 3


def _rms_step(xy):
    """``timeAvgMsd`` reproduced: the root mean squared SINGLE-STEP displacement."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return float("nan")
    dr2 = np.nansum(np.diff(xy, axis=0) ** 2, axis=1)
    finite = np.isfinite(dr2)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.sum(dr2[finite]) / finite.sum()))


def track_transitions(xy, dwell_indexes, *, radius_um,
                      min_steps_between=LEGACY_MIN_STEPS_BETWEEN, strict=False):
    """Free-to-confined and confined-to-free counts for ONE track.

    ``strict=False`` is the legacy: the minimum-length guard applies to interior gaps
    only. ``strict=True`` applies it at the head and tail as well.
    """
    xy = np.asarray(xy, dtype=float)
    if radius_um <= 0:
        raise ConfinementError("radius_um must be > 0, got " + str(radius_um))
    n = len(xy)
    n_fc = n_cf = 0
    dwell_indexes = [np.asarray(d, dtype=int) for d in dwell_indexes if len(d)]

    if not dwell_indexes or (len(dwell_indexes) == 1 and len(dwell_indexes[0]) == n):
        return {"n_free_to_confined": 0, "n_confined_to_free": 0, "n_transitions": 0,
                "n_dwells": len(dwell_indexes), "n_localisations": n,
                "strict": bool(strict)}

    order = sorted(dwell_indexes, key=lambda d: int(d[0]))
    head = int(order[0][0])
    tail = int(order[-1][-1])

    # Head: before the first dwell. The legacy applies no length guard here.
    if head != 0 and (not strict or head >= min_steps_between):
        if _rms_step(xy[: head + 1]) > radius_um:
            n_fc += 1
    # Tail: after the last dwell. Again no guard in the legacy.
    if tail != n - 1 and (not strict or (n - 1 - tail) >= min_steps_between):
        if _rms_step(xy[tail:]) > radius_um:
            n_cf += 1
    # Interior gaps: the legacy DOES guard these.
    for a, b in zip(order[:-1], order[1:]):
        last, first = int(a[-1]), int(b[0])
        if first - last > min_steps_between - 1:
            if _rms_step(xy[last: first + 1]) > radius_um:
                n_fc += 1
                n_cf += 1
    return {"n_free_to_confined": n_fc, "n_confined_to_free": n_cf,
            "n_transitions": n_fc + n_cf, "n_dwells": len(order),
            "n_localisations": n, "strict": bool(strict)}


def transitions(canonical, *, radius_um, min_steps=3,
                min_steps_between=LEGACY_MIN_STEPS_BETWEEN, gap_mode="legacy",
                strict=False, dwells=None):
    """One row per track: its dwell count and its transition counts.

    ``.attrs`` carries the dataset totals and the same total under the OTHER setting of
    ``strict``, so the legacy's inconsistent minimum is a measured difference.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise ConfinementError("not a canonical table (missing " + str(missing) + ")")
    if canonical.empty:
        raise ConfinementError("no localizations to analyse")

    index = _dwells_for(canonical, dwells, radius_um=radius_um, min_steps=min_steps,
                        gap_mode=gap_mode)
    rows, rows_other = [], []
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    for uid, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        xy = track[["x_um", "y_um"]].to_numpy()
        _, idx = index[uid]
        for target, sink in ((strict, rows), (not strict, rows_other)):
            got = track_transitions(xy, idx, radius_um=radius_um,
                                    min_steps_between=min_steps_between, strict=target)
            got["uid"] = uid
            if "cell" in track.columns:
                got["cell"] = track["cell"].iloc[0]
            sink.append(got)
    if not rows:
        raise ConfinementError("no track produced a verdict")

    out = pd.DataFrame(rows)
    other = pd.DataFrame(rows_other)
    out.attrs["radius_um"] = float(radius_um)
    out.attrs["min_steps"] = int(min_steps)
    out.attrs["min_steps_between"] = int(min_steps_between)
    out.attrs["n_tracks"] = int(len(out))
    out.attrs["n_transitions"] = int(out["n_transitions"].sum())
    out.attrs["transitions_per_track"] = float(out["n_transitions"].mean())
    out.attrs["n_transitions_other_strictness"] = int(other["n_transitions"].sum())
    out.attrs["strict"] = bool(strict)
    out.attrs["strictness_note"] = (
        "the legacy guards an interior gap with at least three steps and guards the head "
        "and tail with nothing, so a single step before the first dwell can register a "
        "transition. Both counts are reported and neither is called the difference")
    out.attrs["no_chance_rate_warning"] = (
        "a free Brownian track wanders in and out of any circle, so it produces "
        "transitions with no state change in it. Run transition_null before quoting this")
    return out


def transition_null(canonical, *, radius_um, D_um2_s, min_steps=3,
                    min_steps_between=LEGACY_MIN_STEPS_BETWEEN, strict=False,
                    sigma_um=0.0, gap_mode="legacy", n_replicates=5, seed=0):
    """Observed transitions against the rate a FREE Brownian population produces.

    ``D_um2_s`` has NO DEFAULT: the null is one Brownian population and which D that is
    must come from the analysis, the same rule as ``confinement_null``.
    """
    if D_um2_s <= 0:
        raise ConfinementError("D_um2_s must be positive and must come from the analysis")
    if n_replicates < 1:
        raise ConfinementError(f"n_replicates must be >= 1, got {n_replicates}")
    observed = transitions(canonical, radius_um=radius_um, min_steps=min_steps,
                           min_steps_between=min_steps_between, gap_mode=gap_mode,
                           strict=strict)
    obs_rate = float(observed["n_transitions"].mean())

    # The SAME null the confined fraction uses: lengths resampled from the real data, so
    # the null inherits the real length distribution including its tail.
    chance = []
    for replicate in range(n_replicates):
        sim = matched_brownian_null(canonical, D_um2_s=D_um2_s, sigma_um=sigma_um,
                                    seed=seed + replicate)
        got = transitions(sim, radius_um=radius_um, min_steps=min_steps,
                          min_steps_between=min_steps_between, gap_mode=gap_mode,
                          strict=strict)
        chance.append(float(got["n_transitions"].mean()))
    chance = np.asarray(chance, dtype=float)
    return {
        "observed_per_track": obs_rate,
        "chance_per_track": float(chance.mean()),
        "chance_sd": float(chance.std(ddof=1)) if len(chance) > 1 else float("nan"),
        "excess_per_track": obs_rate - float(chance.mean()),
        "n_replicates": int(n_replicates),
        "radius_um": float(radius_um),
        "D_um2_s": float(D_um2_s),
        "n_tracks": int(len(observed)),
        "caveat": (
            "the transition count inherits every parameter the dwell detector has plus "
            "the radius the step size is compared against. Quote the excess with its "
            "chance rate, never the observed count alone"),
    }


def confined_segments(canonical, *, radius_um, min_steps=3, gap_mode="legacy",
                      dwells=None):
    """The confined stretches, each as its own track. ``getConfinedSegments.m``.

    This is the population the SLA classification is about, and it is not the same as the
    tracks. ``confinedMSDMotion.m:11`` reads ``npTracks_segments`` -- the confined
    stretches only -- so classifying every track in a group answers a different question
    with an identical-looking answer. That is why ``legacy.sla`` requires the population
    to be named on the figure.

    Returns a canonical-shaped table in which each SEGMENT carries its own ``uid``,
    ``"<uid>#seg<k>"``, so every downstream function that groups by track works unchanged.
    ``.attrs`` records how much of the data reached it, because the answer is usually
    "not much" and that belongs beside any verdict computed from it.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise ConfinementError(f"not a canonical table (missing {missing})")
    if canonical.empty:
        raise ConfinementError("no localizations to analyse")

    index = _dwells_for(canonical, dwells, radius_um=radius_um, min_steps=min_steps,
                        gap_mode=gap_mode)
    pieces = []
    n_tracks_in = 0
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    for uid, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        n_tracks_in += 1
        _, indexes = index[uid]
        for k, idx in enumerate(indexes):
            idx = np.asarray(idx, dtype=int)
            if len(idx) < 2:
                continue
            piece = track.iloc[idx].copy()
            piece["uid"] = f"{uid}#seg{k + 1}"
            piece["source_uid"] = uid
            pieces.append(piece)

    if not pieces:
        raise ConfinementError(
            f"no confined segment at radius {radius_um} um with a minimum of "
            f"{min_steps} steps. The classification has nothing to run on, which is "
            f"itself the result: nothing here dwells at this radius.")

    out = pd.concat(pieces, ignore_index=True)
    out.attrs.update(canonical.attrs)
    out.attrs["radius_um"] = float(radius_um)
    out.attrs["min_steps"] = int(min_steps)
    out.attrs["n_tracks_in"] = int(n_tracks_in)
    out.attrs["n_tracks_with_a_segment"] = int(out["source_uid"].nunique())
    out.attrs["n_segments"] = int(out["uid"].nunique())
    out.attrs["n_localisations_in"] = int(len(canonical))
    out.attrs["n_localisations_out"] = int(len(out))
    out.attrs["percent_of_localisations"] = float(
        100.0 * len(out) / max(len(canonical), 1))
    out.attrs["population"] = (
        f"confined segments only: {out.attrs['n_segments']} segments from "
        f"{out.attrs['n_tracks_with_a_segment']} of {n_tracks_in} tracks, "
        f"{out.attrs['n_localisations_out']} of {len(canonical)} localisations "
        f"({out.attrs['percent_of_localisations']:.1f} per cent), at radius "
        f"{radius_um * 1000:.0f} nm and a minimum of {min_steps} steps")
    return out
