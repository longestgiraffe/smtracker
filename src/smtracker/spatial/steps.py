"""Cell-frame steps, and the confinement-free long-axis D built on them.

``observables.squared_displacements`` gives ``r^2``; the spatial arc needs the two
CELL-frame components separately (``du`` along the cell long axis and ``dv``
transverse) because the whole argument of the long-axis estimator is that those two
directions are confined differently. There was no packaged version: the code lived in
``scripts/msd_part3_decomposition.py`` and was reused from there (ON2).

THE START-point convention. A step spans two positions and is located at its **start**,
matching ``hmm.decode_states`` and everything else in this project. The midpoint is
equally defensible and changes contrast magnitudes (2.70x -> 3.90x on the reference
data); the start point is the conservative choice. Never mix the two in one figure.

WHAT ``long_axis_D`` ASSUMES, stated because it is the estimator's whole value:
a transverse reflecting wall does not restrict motion PARALLEL to it, so the variance
of ``du`` is confinement-free at every radius: no interior-only sample is needed, and
the long axis lies in the imaging plane so there is no projection problem. It assumes
the underlying diffusion is ISOTROPIC. If the motion is genuinely anisotropic the
estimate is wrong, and this module does not test that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["cell_frame_steps", "long_axis_D", "MIN_STEPS_FOR_D"]

#: Below this many steps ``long_axis_D`` returns NaN rather than a variance estimated
#: from a handful of points. Not derived: a guard, and it says so.
MIN_STEPS_FOR_D = 50


def cell_frame_steps(localisations: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-frame steps in the cell frame, located at the START point.

    Requires the spatial localisation table (``spatial.io.build_tables``): ``u_um`` /
    ``v_um`` cell-frame coordinates, ``r_norm``, ``cell_length_um``, ``cell_width_um``.

    Only CONSECUTIVE frames form a step: a gap does not become one long step. (On this
    project's u-track data no gaps exist at all, `timeWindow = 1`, measured; the check
    is kept because TrackMate data may differ.)

    Returns one row per step with ``du`` (along the long axis), ``dv`` (transverse),
    ``r_start`` (the START point's ``r_norm``), and ``in_body``.

    ``in_body`` marks steps whose start lies in the CYLINDRICAL body, ``|u| < L/2 -
    W/2``. In the polar caps the local long axis is not the cell long axis and the wall
    confines along ``u`` too (the one place the long-axis argument fails) so callers
    estimating a confinement-free D must filter on it. It is returned rather than
    applied, because a density map legitimately wants the caps.
    """
    need = ["track_id", "frame", "u_um", "v_um", "r_norm",
            "cell_length_um", "cell_width_um"]
    missing = [c for c in need if c not in localisations.columns]
    if missing:
        raise ValueError(f"not a spatial localisation table (missing {missing})")

    d = localisations.sort_values(["track_id", "frame"]).reset_index(drop=True)
    nxt = d.groupby("track_id", sort=False).shift(-1)
    ok = nxt["frame"].notna().to_numpy() & (
        (nxt["frame"].to_numpy() - d["frame"].to_numpy()) == 1)
    cur, nx = d[ok], nxt[ok]
    L = cur["cell_length_um"].to_numpy()
    W = cur["cell_width_um"].to_numpy()
    u0 = cur["u_um"].to_numpy()
    out = {
        "track_id": cur["track_id"].to_numpy(),
        "r_start": cur["r_norm"].to_numpy(),
        "du": nx["u_um"].to_numpy() - u0,
        "dv": nx["v_um"].to_numpy() - cur["v_um"].to_numpy(),
        "in_body": np.abs(u0) < (0.5 * L - 0.5 * W),
    }
    for extra in ("cell_id", "day", "session", "population"):
        if extra in cur.columns:
            out[extra] = cur[extra].to_numpy()
    return pd.DataFrame(out)


def long_axis_D(steps: pd.DataFrame, *, tau_s: float, sigma_um: float,
                min_steps: int = MIN_STEPS_FOR_D) -> float:
    """Confinement-free apparent D from the long-axis component, µm²/s.

    ``Var(du) = 2*D*tau + 2*sigma^2`` in one dimension, so
    ``D = (Var(du) - 2*sigma^2) / (2*tau)``.

    Note the asymmetry with ``appd.apparent_D``, which subtracts NOTHING: this removes
    the localisation term and that one does not, so the two differ by ``sigma^2/tau``
    before any physics (ON3c). Quote which is in use.

    ``sigma_um`` is a fixed INPUT, not fitted, and it is not measured for this project
    (30-76 nm standing range, provenance ``theoretical``). Over-subtracting drives the
    estimate down; the value therefore travels with its provenance.

    Callers wanting the confinement-free reading must pass only ``in_body`` steps,
    this function does not filter, because it cannot know whether the caller has
    already done so.
    """
    du = np.asarray(steps["du"] if isinstance(steps, pd.DataFrame) else steps,
                    dtype=float)
    du = du[np.isfinite(du)]
    if len(du) < min_steps:
        return float("nan")
    return float((du.var(ddof=1) - 2.0 * sigma_um**2) / (2.0 * tau_s))
