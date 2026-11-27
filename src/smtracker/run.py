"""One call: acquisition folder → deposited table. The GUI and the CLI both use it.

Why this exists (AT1). A user with three days and their own data must be able to go from
a raw acquisition folder to a deposited table **without a terminal and without reading the
source**. That needs one packaged function with a typed result, because:

- **Thin-Qt is not relaxed for it.** The widget calls this; it computes nothing itself.
  A GUI path that computes a different number from the CLI path is worse than no GUI path,
  so there is exactly one implementation and both callers pass the same arguments.
- The four states that have bitten this project silently must be *visible* in the result,
  not buried in a log: gagging, the state count, the counts behind a number, and
  provenance. :class:`RunResult` therefore carries each as a field the interface can
  render, rather than printing them.

What it does not do. No cross-condition comparison (Amendment 4), one condition per run.
Comparing conditions or enzymes happens in the firewalled figure script, outside the
package, reading only deposited tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .deposit import build_deposit, write_deposit
from .hmm import HMMError, fit_hmm, step_sequences
from .spatial.maps import MIN_INDEPENDENT_VISITS
from .spatial.pipeline import SpatialConfig, spatial_localisations

__all__ = ["RunConfig", "RunResult", "StateFit", "run_to_deposit",
           "config_from_inputs", "GAG_REASON"]

#: What the interface must say when a condition is gagged. An empty panel meaning
#: "insufficient data" and one meaning "not yet run" must not look alike.
GAG_REASON = ("{condition}: {visits} independent visits of {population} in the shell, "
              "against a floor of {floor}. No enrichment number is emitted.")


@dataclass(frozen=True)
class RunConfig:
    """Everything that decides the answer. No defaults: see :class:`SpatialConfig`."""

    enzyme: str
    condition: str
    spatial: SpatialConfig
    dataset: str
    n_states: tuple = (2, 3)
    hmm_starts: int = 8
    hmm_sigma_um: float = 0.0
    min_visits: int = MIN_INDEPENDENT_VISITS


@dataclass
class StateFit:
    """One HMM fit, with the K that produced it attached to every number.

    K travels WITH the fractions, deliberately: the interface must never display a
    fraction without the state count beside it.
    """

    n_states: int
    D: tuple
    occupancy: tuple
    k_lo_to_hi: float
    k_hi_to_lo: float
    bic: float
    n_steps: int
    n_tracks: int

    def label(self) -> str:
        return f"K = {self.n_states}"


@dataclass
class RunResult:
    """The typed result. Every field exists because the interface must show it."""

    enzyme: str
    condition: str
    fits: dict                       # {K: StateFit}: BOTH are produced, never one
    n_cells: int
    n_days: int
    n_tracks: int
    n_shell_visits: int
    gagged: bool
    gag_reason: str | None
    tables: dict                     # the deposited frames
    written: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def counts_line(self) -> str:
        """Unit and population, in the same string as the numbers (standing rule)."""
        return (f"{self.n_cells} cells · {self.n_days} replicates · "
                f"{self.n_tracks} tracks · {self.n_shell_visits} independent shell "
                f"visits: {self.enzyme}/{self.condition}")


def config_from_inputs(*, enzyme: str, condition: str, sigma_nm: float,
                      erosion_px: float, d_threshold: float,
                      min_localisations: int = 5,
                      require_all_inside: bool = True,
                      dataset: str | None = None,
                      n_states=(2, 3), hmm_starts: int = 8) -> RunConfig:
    """Build a :class:`RunConfig` from the user-visible inputs. ONE implementation.

    AU3: the GUI and the CLI must not merely call the same function, they must call it
    with the same arguments. A widget that calls the right function with
    ``d_threshold=0.5`` instead of ``0.15`` passes every structural test. Routing both
    callers through this one constructor makes the parameter sets equal by construction,
    and ``tests/test_run_gui_cli_identity.py`` asserts the dicts match for identical
    user input.
    """
    missing = [n for n, v in (("enzyme", enzyme), ("condition", condition),
                              ("sigma_nm", sigma_nm), ("erosion_px", erosion_px),
                              ("d_threshold", d_threshold)) if v is None or v == ""]
    if missing:
        raise ValueError(f"missing required inputs: {missing} - none has a safe default")
    return RunConfig(
        enzyme=str(enzyme), condition=str(condition),
        spatial=SpatialConfig(
            sigma_nm=float(sigma_nm), erosion_px=float(erosion_px),
            d_threshold=float(d_threshold),
            min_localisations=int(min_localisations),
            require_all_inside=bool(require_all_inside)),
        dataset=str(dataset) if dataset else str(enzyme).lower(),
        n_states=tuple(n_states), hmm_starts=int(hmm_starts))


def _provenance() -> dict:
    """Commit, dirty flag and load-bearing versions, for the deposit.

    AU2: this function previously wrote ``unknown`` for every run. It called
    ``capture(inputs=...)``, which is not a parameter of ``capture``; the resulting
    ``TypeError`` was swallowed by a broad ``except Exception`` and returned
    ``commit=None``. The deposit then carried a provenance column that reported
    nothing, which is worse than no column because the next reader believes it.

    So it now fails loud. A deposit that cannot say which code produced it is not a
    deposit, and silently degrading to ``unknown`` is the recurring defect class in
    this project: a guard that reports success while doing nothing.
    """
    from . import provenance as _p

    cap = _p.capture(settings={}, seeds={})
    code = dict(cap.code)
    commit = code.get("commit")
    if not commit:
        raise RuntimeError(
            "provenance capture returned no commit hash. The deposit would carry a "
            "column that reports nothing; refusing to write it. Run inside the git "
            "repository, or pass repo= explicitly.")
    return {"commit": commit,
            "dirty": str(code.get("dirty")).lower() == "true",
            "git_available": str(code.get("git_available")).lower() == "true",
            "versions": code.get("load_bearing_versions"),
            "timestamp": cap.timestamp,
            "warnings": list(cap.warnings)}


def run_to_deposit(condition_root, config: RunConfig, outdir, *,
                   day_map=None) -> RunResult:
    """Acquisition folder → HMM fits at every K → deposited table.

    ``condition_root`` is the directory holding one condition's movie folders.
    Returns a :class:`RunResult`; the deposit is written under ``outdir``.
    """
    root, outdir = Path(condition_root), Path(outdir)
    locs, params = spatial_localisations(
        root, config.condition, enzyme=config.enzyme, config=config.spatial,
        dataset=config.dataset, day_map=day_map, with_params=True)
    if locs.empty:
        raise ValueError(f"no localisations for {config.enzyme}/{config.condition} "
                         f"under {root}: refusing to deposit an empty table")

    # ---- HMM at EVERY requested K. Both are produced; none is selected here. ----
    # A minimal canonical-shaped frame for the HMM. Built explicitly rather than by
    # patching columns onto the localisation table, so a schema change fails loudly.
    tau = float(params.tau_s)
    canon = pd.DataFrame({
        "uid": locs["track_id"].to_numpy(),
        "frame": locs["frame"].to_numpy(),
        "x_um": locs["x_um"].to_numpy(),
        "y_um": locs["y_um"].to_numpy(),
        "cell": locs["cell_id"].to_numpy(),
        "movie": locs.get("replicate", pd.Series(index=locs.index, dtype=object)),
        "day": locs.get("day", pd.Series(index=locs.index, dtype=object)),
        "protein": config.enzyme,
        "condition": config.condition,
        "track_id": locs["track_id"].to_numpy(),
        "t_s": locs["frame"].to_numpy() * tau,
        "amp": np.nan,
        "tau_s": tau,
    })
    fits: dict = {}
    for K in config.n_states:
        try:
            seqs = step_sequences(canon, min_steps=1)
            f = fit_hmm(seqs, n_states=K, tau_s=tau,
                        localization_sigma_um=config.hmm_sigma_um,
                        n_starts=config.hmm_starts, seed=0)
        except (HMMError, KeyError, ValueError):
            continue
        order = np.argsort(f.D)
        rates = np.asarray(f.rates_per_s)
        fits[K] = StateFit(
            n_states=K, D=tuple(np.asarray(f.D)[order]),
            occupancy=tuple(np.asarray(f.occupancy)[order]),
            k_lo_to_hi=float(rates[order[0], order[-1]]),
            k_hi_to_lo=float(rates[order[-1], order[0]]),
            bic=float(f.bic), n_steps=int(f.n_steps),
            n_tracks=int(f.n_sequences))

    # ---- gagging, stated not silent ----
    pops = [p for p in locs["population"].astype(str).unique() if p.startswith("D<")]
    pop = pops[0] if pops else None
    in_shell = locs["r_norm"].to_numpy() > config.spatial.d_threshold * 0 + 0.75
    visits = int(pd.unique(
        locs.loc[in_shell & (locs["population"].astype(str) == str(pop)),
                 "track_id"]).size) if pop else 0
    gagged = visits < config.min_visits
    reason = GAG_REASON.format(condition=config.condition, visits=visits,
                               population=pop, floor=config.min_visits) if gagged else None

    day_of = locs.groupby("cell_id")["day"].first().to_dict()
    tables = build_deposit(
        locs, locs.drop_duplicates("cell_id"), params=params,
        sigma_um=config.spatial.sigma_nm / 1000.0,
        d_threshold=config.spatial.d_threshold, shell_r_norm=0.75,
        erosion_px=config.spatial.erosion_px,
        min_localisations=config.spatial.min_localisations,
        day_of=day_of, enzyme=config.enzyme, condition=config.condition)

    prov = _provenance()
    for name, frame in tables.items():
        if isinstance(frame, pd.DataFrame):
            frame["provenance_commit"] = prov.get("commit")
            frame["provenance_dirty"] = prov.get("dirty")
    written = write_deposit(tables, outdir,
                            stem=f"{config.enzyme}_{config.condition}")
    return RunResult(
        enzyme=config.enzyme, condition=config.condition, fits=fits,
        n_cells=int(locs["cell_id"].nunique()),
        n_days=int(pd.Series(list(day_of.values())).nunique()),
        n_tracks=int(locs["track_id"].nunique()),
        n_shell_visits=visits, gagged=gagged, gag_reason=reason,
        tables=tables, written=written, provenance=prov)
