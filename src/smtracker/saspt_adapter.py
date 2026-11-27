"""saSPT adapter: a state count nobody had to assert.

saSPT (Heckert et al. 2022) infers a **Bayesian nonparametric** mixture over a
fixed grid of diffusion coefficients: instead of fitting ``K`` states and asking
BIC to choose ``K``, it puts a Dirichlet prior over a large grid of D values and
reports the posterior occupation of each. Nothing asserts a state count, nothing
is discarded by a goodness-of-fit filter.

Why it is the right third vote
------------------------------
The open question on the reference dataset is whether a third state is real.
The three estimators already built all *assume* a count:

- ``sqd`` : mixture of ``K`` exponentials, ``K`` chosen by BIC;
- ``appd``: mixture of ``K`` Gammas, ``K`` chosen by BIC (and, in the legacy
  form, on a filtered subpopulation);
- ``hmm`` : ``K`` states plus transitions, ``K`` chosen by BIC.

saSPT does not, and its assumptions differ in kind (variational Bayes over a
grid, with a defocalization correction). So it is an independent vote rather
than a same-family echo. It cannot settle whether a state at the localization
floor is biology or measurement heterogeneity (no method can, from this data
alone) but it can say whether the *data* supports resolving more than two
occupied modes.

What it does NOT do
-------------------
saSPT models each trajectory as having a single diffusion coefficient. It is
therefore **switching-blind**, like SQD and appD, and unlike ``smtracker.hmm``.
On data with the switching measured here (~6–8 s⁻¹), expect its occupations to
be blurred in the same direction as any static estimator. Read it as a vote on
*how many modes the data resolves*, not as a switching-corrected result.

Units
-----
saSPT wants raw pixel coordinates plus ``pixel_size_um``. The canonical table is
already in µm, so the adapter passes µm with ``pixel_size_um = 1.0`` rather than
converting back: one conversion, done once, in ``smtracker.canonical``.

Packaging note (a real risk for the installer)
----------------------------------------------
``saspt`` 0.4.0 imports ``matplotlib.cm.get_cmap``, removed in matplotlib 3.9.
This module installs a narrow compatibility shim at import time instead of
pinning matplotlib globally, because napari will need a modern matplotlib.
If saSPT stays unmaintained, the state-array model is small enough to
reimplement natively: the same rule already applied to vbSPT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS

__all__ = [
    "SASPTError",
    "SASPTResult",
    "to_saspt_detections",
    "run_saspt",
]

#: saSPT's required detection columns.
SASPT_COLUMNS = ["frame", "trajectory", "y", "x"]


class SASPTError(RuntimeError):
    """Raised when saSPT is unavailable or cannot be run."""


@dataclass
class SASPTResult:
    """Posterior occupations over saSPT's diffusion-coefficient grid."""

    diffusion_grid: np.ndarray          # µm²/s
    occupations: np.ndarray             # posterior occupancy, sums to 1
    n_trajectories: int
    n_detections: int
    likelihood_type: str

    def modes(self, *, min_occupancy: float = 0.05,
              min_prominence: float = 0.25) -> pd.DataFrame:
        """INTERIOR local maxima of the occupancy profile: the resolved modes.

        This is the closest thing saSPT gives to "how many states": peaks in a
        continuous posterior, not a fitted count. A broad shoulder is not a state,
        which is exactly the ambiguity a ``K``-state fit hides.

        **A GRID EDGE IS NOT A MODE, and treating it as one was a real defect here.**
        The first version compared each point against ``-inf`` on its missing side, so
        index 0 qualified as a peak whenever the profile merely *decreased* from it.
        saSPT's default grid starts at 0.01 µm²/s, and its RBME likelihood reports a
        LOCALIZATION-ERROR-CORRECTED D while this project's SQD reports an apparent one
        -- an apparent 0.055 less σ²/τ at σ = 17-30 nm lands at ~0.01 -- so the slow
        population's mass piles against the lowest grid point rather than forming a peak
        inside it. Measured on nine deposited groups, the profile fell ELEVENFOLD in one
        grid step from that edge and then decayed monotonically, and every group was
        reported as "2 modes" when eight of the nine resolve exactly ONE.

        Boundary mass is therefore reported by :attr:`edge_occupancy` and
        :meth:`at_grid_floor`, never as a mode. A population sitting at the edge is not
        resolved by this grid, and the honest response is to widen the grid and re-run.

        ``min_prominence`` is a fraction of the peak's own height: a bump must rise that
        far above the higher of its flanking minima to count. Without it a single noisy
        grid point on a broad shoulder is a "state".

        ``min_occupancy`` is relative to the total (which is 1), not to the tallest
        peak. Relative-to-max silently drops a genuine small mode whenever one mode
        dominates -- exactly the low-occupancy case where the state count is least
        certain and most worth reporting.
        """
        occupancy = np.asarray(self.occupations, dtype=float)
        rows = []
        for i in range(1, len(occupancy) - 1):
            if not (occupancy[i] > occupancy[i - 1] and occupancy[i] >= occupancy[i + 1]):
                continue
            if occupancy[i] < min_occupancy:
                continue
            # Prominence against the deeper of the two flanking valleys.
            left_min = occupancy[:i].min()
            right_min = occupancy[i + 1:].min()
            base = max(left_min, right_min)
            if occupancy[i] - base < min_prominence * occupancy[i]:
                continue
            rows.append({"D_um2_per_s": self.diffusion_grid[i],
                         "occupancy": occupancy[i]})
        return pd.DataFrame(rows, columns=["D_um2_per_s", "occupancy"])

    @property
    def edge_occupancy(self) -> float:
        """Posterior mass in the LOWEST grid point: unresolved, not a mode."""
        return float(self.occupations[0])

    def at_grid_floor(self, *, threshold: float = 0.02) -> bool:
        """Is mass piling against the bottom of the grid rather than resolving?

        True means the grid does not reach low enough for this data and a state count
        read off it is not trustworthy. Widen ``diff_coefs`` and re-run.
        """
        return self.edge_occupancy >= threshold and (
            self.occupations[0] > self.occupations[1])

    def occupancy_below(self, D_threshold: float) -> float:
        """Total posterior occupancy below a D: e.g. the localization floor."""
        return float(self.occupations[self.diffusion_grid < D_threshold].sum())

    def summary(self) -> str:
        modes = self.modes()
        described = ", ".join(
            f"D ≈ {row.D_um2_per_s:.3g} ({row.occupancy * 100:.0f} %)"
            for row in modes.itertuples()
        )
        # THE EDGE MASS IS ALWAYS SAID, because a resolved count means nothing while a
        # population is piled against the bottom of the grid. Silence here is what let
        # "2 modes" stand for nine groups that resolve one.
        edge = ""
        if self.at_grid_floor():
            edge = (f"; {self.edge_occupancy * 100:.0f} % of the mass sits in the LOWEST "
                    f"grid point (D = {self.diffusion_grid[0]:.3g}) and is NOT resolved "
                    f": widen the grid before reading a state count")
        return (
            f"saSPT ({self.likelihood_type}, {self.n_trajectories} trajectories, "
            f"{self.n_detections} detections): {len(modes)} resolved mode(s): "
            f"{described or '(none)'}{edge}"
        )


def _import_saspt():
    """Import saSPT, shimming the matplotlib API it relies on."""
    try:
        import matplotlib.cm
        import matplotlib.pyplot as plt

        if not hasattr(matplotlib.cm, "get_cmap"):
            # Removed in matplotlib 3.9; saspt 0.4.0 still imports it.
            matplotlib.cm.get_cmap = lambda name=None, lut=None: plt.get_cmap(name)
        import saspt
    except ImportError as exc:
        raise SASPTError(
            "saspt is not installed. Install it with `pip install saspt` "
            "(it is a pure-Python dependency and bundles cleanly)."
        ) from exc
    return saspt


def to_saspt_detections(canonical: pd.DataFrame) -> pd.DataFrame:
    """Canonical table → saSPT detection table.

    Coordinates stay in **µm** and the caller passes ``pixel_size_um = 1.0``,
    so no unit conversion happens twice.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise SASPTError(f"not a canonical table (missing {missing})")
    if len(canonical) == 0:
        raise SASPTError("canonical table is empty")

    # saSPT identifies trajectories by integer id.
    codes = pd.Categorical(canonical["uid"]).codes
    out = pd.DataFrame(
        {
            "frame": canonical["frame"].to_numpy(dtype=int),
            "trajectory": codes.astype(int),
            "y": canonical["y_um"].to_numpy(dtype=float),
            "x": canonical["x_um"].to_numpy(dtype=float),
        }
    )
    if (out["trajectory"] < 0).any():
        raise SASPTError("some tracks have no uid; cannot build trajectory ids")
    return out


def run_saspt(
    canonical: pd.DataFrame,
    *,
    likelihood_type: str = "rbme",
    focal_depth: float = 0.7,
    progress_bar: bool = False,
    **kwargs,
) -> SASPTResult:
    """Run saSPT on a canonical table and return its occupancy profile.

    Parameters
    ----------
    canonical
        Canonical table (µm, seconds), carrying ``AcquisitionParams`` in
        ``.attrs`` so τ is read rather than guessed.
    likelihood_type
        ``"rbme"`` (regular Brownian motion with localization error, the saSPT
        default) or another of saSPT's likelihoods.
    focal_depth
        Depth of field in µm, for the defocalization correction. Fast molecules
        leave the focal volume sooner: the same survivorship effect documented
        throughout this project, which saSPT can partially correct for.

    Raises
    ------
    SASPTError
        If saSPT is unavailable, or τ is unknown.
    """
    saspt = _import_saspt()

    params = canonical.attrs.get("params")
    if params is None:
        raise SASPTError(
            "canonical table carries no AcquisitionParams in .attrs['params']; τ "
            "is required."
        )

    detections = to_saspt_detections(canonical)
    settings = {
        "likelihood_type": likelihood_type,
        "pixel_size_um": 1.0,          # coordinates are already µm
        "frame_interval": float(params.tau_s),
        "focal_depth": float(focal_depth),
        "progress_bar": progress_bar,
        **kwargs,
    }
    try:
        state_array = saspt.StateArray.from_detections(detections, **settings)
        occupations = np.asarray(state_array.posterior_occs, dtype=float)
        grid = np.asarray(state_array.parameter_values[0], dtype=float)
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
        raise SASPTError(f"saSPT failed: {exc}") from exc

    # RBME marginalises over localization error: collapse to the D axis.
    while occupations.ndim > 1:
        occupations = occupations.sum(axis=-1)
    total = occupations.sum()
    if not np.isfinite(total) or total <= 0:
        raise SASPTError("saSPT returned a degenerate occupancy profile")
    occupations = occupations / total

    if occupations.shape != grid.shape:
        raise SASPTError(
            f"saSPT occupancy shape {occupations.shape} does not match its "
            f"diffusion grid {grid.shape}"
        )

    return SASPTResult(
        diffusion_grid=grid,
        occupations=occupations,
        n_trajectories=int(detections["trajectory"].nunique()),
        n_detections=len(detections),
        likelihood_type=likelihood_type,
    )
