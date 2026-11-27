"""WHERE state switching happens: capture and release as located EVENTS.

Distinct from `smtracker.confinement.transitions`, which counts ROI entries and exits
from a geometric detector. This module works on the HMM's per-step decoding, so an event
is a change in the inferred state and carries the position it happened at.

What this answers that the state map does not. `maps.labels_from_hmm` shows where the
slow state IS. This shows where molecules ENTER and LEAVE it, which are different
questions with different answers: a slow state can be enriched somewhere because molecules
stay stopped longer there, or because more of them stop there, and only the events tell
the two apart.

Three things this got wrong before it got them right, each preserved as a guard:

1. **The state labels are 1-based and ``fit.D`` is 0-indexed.** ``argmin(fit.D)`` names
   no state at all, so every transition came out as a release -- which reads as a finding
   rather than an off-by-one. ``slow_state_label`` does the mapping once.
2. **The denominator is the ELIGIBLE state, not every step.** Capture can only happen
   from fast and release only from slow, and the slow share changes across the cell, so a per-total-step map mostly redraws where the states are.
   Computed that way the release map came out with the OPPOSITE sign to the radial
   profile of the same quantity. ``rate_map`` REFUSES to run without being told.
3. **The grid is set by counting statistics, not by taste.** At a typical release rate
   per slow step and an effect of a few points, a bin needs about 400 slow steps for the effect
   to clear two SD; with ~14 700 slow steps that caps the grid near 36 bins. Drawn at
   16x8 = 128 the map was Poisson speckle that looked like structure.
   ``resolvable_bins`` computes the cap so it is chosen rather than assumed.

THE CONVENTION. A transition spans TWO localisations. It is placed at the step's START,
matching ``hmm.decode_states``; the midpoint is equally defensible and moves the
magnitude, as it does 2.70x against 3.90x on the state map. Quote the convention.

The null is not in this module yet, and the numbers below came from a script rather
than from code that ships. Recording that distinction rather than naming a function that
does not exist: the procedure is to simulate the same cells with spatially uniform
switching at the measured rates (``spatial.nulls.simulate_positions``) and decode them
with the SAME fit, so that whatever the null produces is decoder geometry rather than
kinetics.

MEASURED on one lab group, the release gradient (axis minus membrane) was compared with the
null's 5 realisations and with a relaxed cell filter: the null's spread covered zero and
the observed gradient lay well outside it, so the decoder explains none of the gradient
and the cell filter about 16 per cent. The
null does NOT model that filter -- it keeps every simulated track, while the real filter
drops whole tracks that stray outside an outline and its loss is D-selective at the
boundary -- which is why the filter had to be tested separately by re-running.

What survives, and what does not: on that group, capture per fast step was flat across
shells and release per slow step changed toward the membrane, on all three replicate days.
The 2-D map additionally suggested a pole effect, but the pole bins are the thinnest in the
map and that is not established. One group only, three days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "SwitchingError",
    "EVENTS",
    "STEP_POSITIONS",
    "slow_state_label",
    "transition_steps",
    "rate_map",
    "radial_profile",
    "fold_map",
    "resolvable_bins",
    "measured_rates",
]


class SwitchingError(RuntimeError):
    """Raised when a switching map is asked for something it cannot honestly draw."""


#: The two directions, named for what happens rather than for the state.
EVENTS = ("capture", "release")

#: Which of a step's two localisations the event is placed at.
STEP_POSITIONS = ("start", "midpoint")


def slow_state_label(decoded, fit) -> int:
    """The label ``decode_states`` uses for the slowest state.

    ``decode_states`` labels states 1..n; ``fit.D`` is 0-indexed. Mapping one to the
    other by ``argmin(fit.D)`` alone names no state, and every transition then comes out
    in one direction.
    """
    D = np.asarray(fit.D, dtype=float)
    n_states = len(D)
    # `HMMFit` sorts states by ascending D and `decode_states` labels them index + 1 in
    # that order, so the slow state is argmin(D) + 1 -- which is 1 for a sorted fit.
    # Computed rather than hard-coded, so a fit that ever stopped sorting would be caught
    # instead of silently relabelling every event.
    label = int(np.argmin(D)) + 1

    # NOT EVERY STATE HAS TO APPEAR. A group where nothing ever stops decodes entirely to
    # one label, and that is a result rather than an error; the first version required
    # the label count to match the fit and refused exactly that case.
    present = set(int(v) for v in pd.unique(np.asarray(decoded["state"])))
    unknown = sorted(present - set(range(1, n_states + 1)))
    if unknown:
        raise SwitchingError(
            f"the decoding carries state label(s) {unknown}, which a {n_states}-state "
            f"fit cannot have produced: these are not the same model")
    return label


def transition_steps(decoded, fit, *, position: str = "start"):
    """One row per step pair within a track, labelled ``none``/``capture``/``release``.

    Returns ``(steps, slow_state)``. ``steps`` carries the decoding's own columns plus
    ``event``, so the position of an event is the position the decoding recorded.
    """
    if position not in STEP_POSITIONS:
        raise SwitchingError(
            f"position must be one of {STEP_POSITIONS}, got {position!r}. A transition "
            f"spans two localisations and the choice is a convention that moves the "
            f"magnitude, so you must state it")
    for column in ("uid", "frame", "state"):
        if column not in decoded.columns:
            raise SwitchingError(
                f"missing {column!r}: this takes the output of hmm.decode_states")
    if decoded.empty:
        raise SwitchingError("nothing decoded")

    frame = decoded.sort_values(["uid", "frame"], kind="stable").reset_index(drop=True)
    state = np.asarray(frame["state"])
    uid = np.asarray(frame["uid"])
    same_track = uid[1:] == uid[:-1]
    previous, following = state[:-1], state[1:]
    slow = slow_state_label(frame, fit)

    changed = same_track & (previous != following)
    event = np.full(len(previous), "none", dtype=object)
    event[changed & (following == slow)] = "capture"
    event[changed & (following != slow)] = "release"

    out = (frame.iloc[:-1] if position == "start" else frame.iloc[1:]).copy()
    out["event"] = event
    out["step_position"] = position
    out = out[same_track].reset_index(drop=True)
    out.attrs.update(decoded.attrs)
    out.attrs["slow_state"] = slow
    out.attrs["step_position"] = position
    out.attrs["n_capture"] = int((out["event"] == "capture").sum())
    out.attrs["n_release"] = int((out["event"] == "release").sum())
    return out, slow


def measured_rates(fit) -> dict:
    """``{capture, release}`` per second, from the fitted transition matrix.

    Ordered by ascending D, so ``capture`` is fast to slow whatever the label order.
    """
    D = np.asarray(fit.D, dtype=float)
    order = np.argsort(D)
    T = np.asarray(fit.transition_matrix, dtype=float)[np.ix_(order, order)]
    tau = float(fit.tau_s)
    return {"capture": float(T[1, 0] / tau), "release": float(T[0, 1] / tau),
            "tau_s": tau, "D": tuple(np.round(D[order], 6))}


def resolvable_bins(steps, *, event: str, slow_state: int, effect: float,
                    n_sigma: float = 4.0) -> dict:
    """How many bins the counts support for an effect of size ``effect``.

    A grid finer than this shows Poisson noise, and Poisson noise in a heat map looks
    exactly like structure. Returns the cap and the per-bin requirement so the choice can
    be made rather than assumed.
    """
    eligible = _eligible(steps, event, slow_state)
    n = int(eligible.sum())
    if n == 0:
        raise SwitchingError(f"no steps eligible for {event!r}")
    p = float((np.asarray(steps["event"]) == event).sum()) / n
    if not 0 < p < 1:
        raise SwitchingError(f"{event!r} rate is {p}, so no interval can be formed")
    per_bin = p * (1 - p) / ((effect / n_sigma) ** 2)
    return {"rate": p, "n_eligible": n, "steps_per_bin_needed": float(per_bin),
            "max_bins": int(n // per_bin) if per_bin else 0,
            "n_sigma": float(n_sigma), "effect": float(effect)}


def _eligible(steps, event: str, slow_state: int) -> np.ndarray:
    if event not in EVENTS:
        raise SwitchingError(f"event must be one of {EVENTS}, got {event!r}")
    state = np.asarray(steps["state"])
    return (state == slow_state) if event == "release" else (state != slow_state)


def _bin_index(steps, bins):
    l = np.clip(np.asarray(steps["l_norm"], dtype=float), 0.0, 1.0)
    d = np.clip(np.asarray(steps["d_norm"], dtype=float), -1.0, 1.0)
    li = np.clip((l * bins[0]).astype(int), 0, bins[0] - 1)
    di = np.clip(((d + 1.0) / 2.0 * bins[1]).astype(int), 0, bins[1] - 1)
    return li * bins[1] + di


def rate_map(steps, *, event: str, slow_state: int | None = None,
             bins: tuple[int, int] = (8, 4), n_min: int = 120):
    """Events per step of the eligible state, per bin.

    ``slow_state`` is REQUIRED. Dividing by every step in the bin makes this a map of
    where the states are: computed that way the release map came out with the opposite
    sign to the radial profile of the same quantity.
    """
    if slow_state is None:
        raise SwitchingError(
            "slow_state is required. The denominator is the eligible state (capture "
            "happens from fast, release from slow); dividing by every step would map "
            "where the states are instead")
    for column in ("l_norm", "d_norm", "state", "event"):
        if column not in steps.columns:
            raise SwitchingError(
                f"missing {column!r}: this takes the output of transition_steps on a "
                f"cell-frame table")

    flat = _bin_index(steps, bins)
    n_bins = bins[0] * bins[1]
    exposure = np.bincount(flat[_eligible(steps, event, slow_state)],
                           minlength=n_bins).astype(float)
    events = np.bincount(flat[np.asarray(steps["event"]) == event],
                         minlength=n_bins).astype(float)
    rate = np.full(n_bins, np.nan)
    enough = exposure >= n_min
    rate[enough] = events[enough] / exposure[enough]

    grid = rate.reshape(bins)
    out = {"rate": grid, "exposure": exposure.reshape(bins),
           "events": events.reshape(bins), "bins": tuple(bins),
           "n_min": int(n_min), "event": event, "slow_state": int(slow_state),
           "n_bins_kept": int(enough.sum()), "n_bins": n_bins}
    out["caption"] = (
        f"{event} per {'slow' if event == 'release' else 'fast'} step | "
        f"{bins[0]}x{bins[1]} bins, {int(enough.sum())} of {n_bins} over n_min={n_min} | "
        f"{int(events.sum())} events over {int(exposure.sum())} eligible steps")
    return out


def radial_profile(steps, *, event: str, slow_state: int, n_shells: int = 5):
    """Event rate against distance from the axis, pooled along the cell.

    Pooling along the long axis is what makes the shells well populated; it also hides
    any axial structure, so a flat profile is not a flat map.
    """
    d = np.abs(np.clip(np.asarray(steps["d_norm"], dtype=float), -1.0, 1.0))
    shell = np.clip((d * n_shells).astype(int), 0, n_shells - 1)
    eligible = _eligible(steps, event, slow_state)
    is_event = np.asarray(steps["event"]) == event
    rows = []
    for s in range(n_shells):
        here = shell == s
        n_elig = int((here & eligible).sum())
        n_event = int((here & is_event).sum())
        rows.append({"shell": s, "n_eligible": n_elig, "n_events": n_event,
                     "rate": n_event / n_elig if n_elig else np.nan})
    table = pd.DataFrame(rows)
    table.attrs["event"] = event
    table.attrs["n_shells"] = int(n_shells)
    table.attrs["gradient"] = float(table["rate"].iloc[0] - table["rate"].iloc[-1])
    table.attrs["caption"] = (
        f"{event} per {'slow' if event == 'release' else 'fast'} step, shell 0 = axis to "
        f"{n_shells - 1} = membrane; pooled along the cell, so a flat profile is not a "
        f"flat map")
    return table


def fold_map(grid):
    """Fold both ways and POOL the folds.

    Which pole is ``l = 0`` and which side is ``d > 0`` are arbitrary, so unfolded
    asymmetry is almost always noise. The legacy takes ``max`` of the reflections, which
    is biased upward and worst where counts are lowest; this takes the mean.
    """
    grid = np.asarray(grid, dtype=float)
    with np.errstate(invalid="ignore"):
        along = np.nanmean(np.stack([grid, grid[::-1, :]]), axis=0)
        return np.nanmean(np.stack([along, along[:, ::-1]]), axis=0)
