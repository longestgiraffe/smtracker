"""Execution tiers: opt-in methods, not a forced panel.

Running every estimator on every open is wasteful: most questions need only the
fraction, and the expensive methods (HMM, saSPT, Spot-On, σ-sweep, recovery)
should idle until asked for. This defines three cumulative tiers and a runner
that executes **only** the requested methods:

- ``quick``: SQD bound fraction + cell bootstrap. The headline, fast.
- ``cross_check``: adds HMM, saSPT, Spot-On, confinement, clustering. The
  corroboration set.
- ``everything``: adds the spatial map and the heavy validation views.

The runner drives the same pure controllers the method browser uses, so there is
one source of truth for what each method computes; the tier only chooses *which*
run. A method not in the requested tier is not touched, the whole point.
"""

from __future__ import annotations

from . import browser_controller as ctrl

__all__ = ["TIERS", "TIER_ORDER", "methods_for", "run_tier"]

#: Methods introduced *at* each tier (each tier also runs the earlier tiers').
_INTRODUCED = {
    "quick": ["fractions", "superplot_distribution", "superplot_fraction", "qc"],
    "cross_check": ["kinetics", "state_count", "confinement", "clustering", "spoton"],
    "everything": ["spatial"],
}

#: Coarsest → finest. A tier includes every method from itself and earlier tiers.
TIER_ORDER = ["quick", "cross_check", "everything"]

#: Cumulative membership, resolved once.
TIERS: dict[str, list[str]] = {}
for _i, _tier in enumerate(TIER_ORDER):
    TIERS[_tier] = [m for t in TIER_ORDER[: _i + 1] for m in _INTRODUCED[t]]


class TierError(RuntimeError):
    """Raised when an unknown tier or method is requested."""


def methods_for(tier: str) -> list[str]:
    """The methods a tier runs (cumulative)."""
    if tier not in TIERS:
        raise TierError(f"unknown tier {tier!r}; choose from {TIER_ORDER}")
    return list(TIERS[tier])


def run_tier(project, condition, *, tier: str = "quick", methods=None, n_boot=50,
             cache=None, settings=None):
    """Compute only the requested methods; return ``{method: status}``.

    ``methods`` overrides the tier with an explicit opt-in list (a method runs
    only if named). With a ``cache`` (a ``ResultCache``) each method is memoized
    by its own provenance, so a re-run of a tier is instant where nothing
    changed. Failures are captured per method (``"failed: …"``) rather than
    sinking the whole tier: one broken method must not hide the others.
    """
    selected = methods if methods is not None else methods_for(tier)
    unknown = [m for m in selected if not hasattr(ctrl, m)]
    if unknown:
        raise TierError(f"unknown method(s): {unknown}")

    out: dict[str, str] = {}
    for method in selected:
        try:
            _, status = _run_one(project, condition, method, n_boot=n_boot,
                                 cache=cache, settings=settings)
        except Exception as exc:                        # noqa: BLE001 - reported
            status = f"failed: {exc}"
        out[method] = status
    return out


def _run_one(project, condition, method, *, n_boot, cache, settings):
    fn = getattr(ctrl, method)
    if cache is None:
        return fn(project, condition, n_boot=n_boot)

    from .provenance import capture

    provenance = capture(
        settings={"method": method, "condition": condition, "n_boot": n_boot,
                  **(settings or {})},
        seeds={"bootstrap": 0, "hmm": 0},
        canonical=project.condition(condition),
        unit="cell",
    )
    # The status string is what a headless tier run reports; it is cheap and
    # already carries the numbers, so it is what we memoize.
    def compute():
        _draw, status = fn(project, condition, n_boot=n_boot)
        return status

    status = cache.get_or_compute(f"tier_{method}", provenance, compute)
    return None, status
