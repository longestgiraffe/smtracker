"""Single-track inspection, built to make the resolution floor visible.

The question this answers
-------------------------
The project's central caveat is that the slow state's D sits *at or below* what
the microscope resolves, so D₁ is a bound and not a measurement (CLAUDE.md).
Every aggregate view states that in words. This view states it per track, in
pixels: the step-size trace is drawn against a **localization-floor band**, so
for any track flagged "slow" the reader can see whether the small steps are a
real binding event or jitter indistinguishable from an immobile molecule.

Nothing else in the tool makes the floor tangible at the single-molecule level.
That is why the floor band, not the trajectory, is the load-bearing element.

σ is uncertain, so the band is a RANGE
--------------------------------------
u-track records no localization precision. The active σ is DERIVED from the pixel
size (``AcquisitionParams.localization_sigma_nm`` = pixel/√12, the pixelation lower
bound, ~30 nm at 106 nm pixels); the true precision is higher, since photon noise
and background add on top. So ``floor_band`` accepts *two* σ values and returns two
bands: the pixelation lower bound and a plausible photon-limited upper. A track
whose steps sit below the upper band is not resolvable under either assumption; one
between the bands is a judgement call that depends on a number nobody has measured.
(The earlier 76 nm "MSD-intercept" figure is RETIRED, confounded, biased high; see
``docs/constants_provenance.md``. The legacy 30 nm lives only in the MATLAB GMM
classifier, not its SQD path, and is not used as σ here.)

What the band means
-------------------
An immobile molecule localized with precision σ per coordinate still appears to
move: the two independent localization errors give a displacement with
E[r²] = 4σ², i.e. r ~ Rayleigh(σ√2). ``floor_band`` returns the quantile of that
distribution, so "90 % of steps from a truly immobile molecule fall below this
line". That is a statement about *steps*, which is what the trace shows,
related to but not the same as ``AcquisitionParams.localization_floor_D``, which
is the legacy σ²/(4τ) expressed as a diffusion coefficient. Both are reported.

Method disagreement is a first-class readout
--------------------------------------------
A suspicious track is usually one the methods label differently. ``inspect_track``
collects the HMM's per-step states, appD's single per-track D and the confinement
flag side by side, because appD collapsing a switching molecule into one averaged
D *while* the HMM splits it is the switching-blur diagnostic made concrete, the
same effect that gives the HMM map 2.7x dynamic range against appD's 1.3x.

Invariants
----------
- Data-first: pure numpy/pandas, no Qt, no napari. The napari widget and the
  matplotlib figure are both consumers.
- Gaps are counted and never bridged: a step across a frame gap is not a step.
- Fails loud on an unknown uid or a track too short to have a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS, AcquisitionParams

__all__ = [
    "InspectorError",
    "SIGMA_RANGE_NM",
    "floor_band",
    "TrackInspection",
    "inspect_track",
]

#: Display band for the localization-floor σ: from the pixelation lower bound
#: (~pixel/√12 ≈ 30 nm at 106 nm pixels, the physical minimum, derived not
#: assumed) up to a plausible photon-limited ~50 nm. The 76 nm MSD-intercept figure
#: is RETIRED (confounded, biased high, see docs/constants_provenance.md); the
#: legacy 30 nm belongs to the MATLAB GMM classifier, not σ here.
SIGMA_RANGE_NM = (30.0, 50.0)


class InspectorError(RuntimeError):
    """Raised when a track cannot be inspected."""


def floor_band(sigma_um: float, *, quantile: float = 0.9) -> float:
    """Step size below which a truly immobile molecule lands ``quantile`` of the time.

    Two independent localization errors of σ per coordinate give a displacement
    r ~ Rayleigh(σ√2), so the quantile is ``σ√2 · sqrt(-2 ln(1 − q))``.
    """
    if sigma_um <= 0:
        raise InspectorError(f"sigma_um must be > 0, got {sigma_um}")
    if not 0.0 < quantile < 1.0:
        raise InspectorError(f"quantile must be in (0, 1), got {quantile}")
    return float(sigma_um * np.sqrt(2.0) * np.sqrt(-2.0 * np.log(1.0 - quantile)))


@dataclass
class TrackInspection:
    """One track, with everything needed to judge its labels."""

    uid: str
    cell: str
    condition: str
    frames: np.ndarray                  # per localization
    x_um: np.ndarray
    y_um: np.ndarray
    steps_um: np.ndarray                # per step, len = n_localizations - 1
    step_frames: np.ndarray             # frame the step STARTS on (matches hmm.decode_states)
    step_is_contiguous: np.ndarray      # False where a frame gap was bridged
    tau_s: float
    sigma_bands_um: dict[float, float]  # σ (µm) -> step-size quantile
    quantile: float
    states: np.ndarray | None = None    # per step, from the HMM
    posteriors: np.ndarray | None = None
    appd_um2_per_s: float | None = None
    confined: np.ndarray | None = None  # per localization
    track_class: str | None = None
    localization_floor_D: float | None = None
    detection_ceiling_D: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_localizations(self) -> int:
        return len(self.frames)

    @property
    def n_steps(self) -> int:
        return len(self.steps_um)

    @property
    def n_gaps(self) -> int:
        return int((~self.step_is_contiguous).sum())

    @property
    def instantaneous_D(self) -> np.ndarray:
        """Per-step apparent D, ``r²/(4τ)``. Single-step D is enormously noisy,
        this is for *looking*, never for quoting."""
        return self.steps_um**2 / (4.0 * self.tau_s)

    @property
    def fraction_below_floor(self) -> dict[float, float]:
        """Share of steps under each σ band: the "is this resolvable?" number."""
        return {
            sigma: float(np.mean(self.steps_um < band))
            for sigma, band in self.sigma_bands_um.items()
        }

    @property
    def n_state_switches(self) -> int:
        if self.states is None:
            return 0
        return int((np.diff(self.states) != 0).sum())

    def disagreement(self) -> str:
        """How the methods labelled this track, side by side.

        The interesting case is a track the HMM splits and appD averages: appD
        reports one D belonging to neither state, which is the switching blur
        that makes a per-track speed map structurally weaker than a per-step map.
        """
        parts = [f"N = {self.n_localizations} localizations ({self.n_steps} steps)"]
        if self.n_gaps:
            parts.append(f"{self.n_gaps} frame gap(s): not bridged")
        if self.states is not None:
            slow = float(np.mean(self.states == 1))
            parts.append(
                f"HMM: {100 * slow:.0f} % slow, {self.n_state_switches} switch(es)"
            )
        if self.appd_um2_per_s is not None:
            parts.append(f"appD (one D per track): {self.appd_um2_per_s:.3g} µm²/s")
        if self.track_class is not None:
            parts.append(f"confinement: {self.track_class}")

        if self.states is not None and self.appd_um2_per_s is not None:
            if self.n_state_switches > 0:
                parts.append(
                    "Disagreement: the HMM splits this track, while appD averages "
                    "it into a single D that belongs to neither state"
                )
        return " | ".join(parts)

    def floor_verdict(self) -> str:
        """Whether this track's motion is resolvable at all."""
        shares = self.fraction_below_floor
        low_sigma, high_sigma = min(shares), max(shares)
        low, high = shares[low_sigma], shares[high_sigma]
        text = (
            f"{100 * low:.0f} % of steps below the {1000 * low_sigma:.0f} nm floor, "
            f"{100 * high:.0f} % below the {1000 * high_sigma:.0f} nm floor "
            f"(q = {self.quantile:.2f})"
        )
        if high > 0.5:
            return (
                text + ": not resolvable. Most steps are consistent with an "
                "immobile molecule under the larger σ, so any D from this track "
                "is an upper bound, not a measurement"
            )
        if low > 0.5:
            return (
                text + ": resolvable only if σ is really 30 nm. σ is not measured, "
                "so treat this track's D as uncertain"
            )
        return text + ": resolvable under both σ assumptions"

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.uid}  (cell {self.cell}, condition {self.condition})",
                f"  {self.disagreement()}",
                f"  floor: {self.floor_verdict()}",
            ]
        )


def inspect_track(
    canonical: pd.DataFrame,
    uid: str,
    *,
    hmm_fit=None,
    confinement_radius_um: float | None = None,
    sigma_range_nm: tuple[float, float] = SIGMA_RANGE_NM,
    quantile: float = 0.9,
    appd_n_steps: int = 4,
) -> TrackInspection:
    """Everything about one track: geometry, steps, labels and the floor.

    Every label is optional and degrades to ``None`` rather than being invented:
    an inspector that silently shows no HMM states is honest, one that fakes them
    is not.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise InspectorError(f"not a canonical table (missing {missing})")
    params = canonical.attrs.get("params")
    if not isinstance(params, AcquisitionParams):
        raise InspectorError("canonical table carries no AcquisitionParams")

    track = canonical[canonical["uid"] == uid].sort_values("frame", kind="stable")
    if track.empty:
        raise InspectorError(f"no track with uid {uid!r}")
    if len(track) < 2:
        raise InspectorError(f"track {uid!r} has {len(track)} localization(s); no step")

    frames = track["frame"].to_numpy()
    x = track["x_um"].to_numpy(dtype=float)
    y = track["y_um"].to_numpy(dtype=float)
    steps = np.hypot(np.diff(x), np.diff(y))
    contiguous = np.diff(frames) == 1

    bands = {
        nm / 1000.0: floor_band(nm / 1000.0, quantile=quantile)
        for nm in sigma_range_nm
    }

    notes: list[str] = []
    states = posteriors = None
    if hmm_fit is not None:
        states, posteriors = _hmm_labels(track, hmm_fit, contiguous, notes)

    appd = None
    try:
        from .appd import apparent_D

        table = apparent_D(track, n_steps=appd_n_steps, mode="split")
        if not table.empty:
            appd = float(table["appd_um2_per_s"].mean())
    except Exception as exc:                       # noqa: BLE001 - reported, not swallowed
        notes.append(f"appD unavailable for this track: {exc}")

    confined = track_class = None
    if confinement_radius_um is not None:
        from .confinement import label_confinement

        labelled = label_confinement(track, radius_um=confinement_radius_um)
        confined = labelled["confined"].to_numpy()
        track_class = str(labelled["track_class"].iloc[0])

    return TrackInspection(
        uid=uid,
        cell=str(track["cell"].iloc[0]),
        condition=str(track["condition"].iloc[0]),
        frames=frames,
        x_um=x,
        y_um=y,
        steps_um=steps,
        step_frames=frames[:-1],
        step_is_contiguous=contiguous,
        tau_s=params.tau_s,
        sigma_bands_um=bands,
        quantile=quantile,
        states=states,
        posteriors=posteriors,
        appd_um2_per_s=appd,
        confined=confined,
        track_class=track_class,
        localization_floor_D=params.localization_floor_D,
        # the ceiling of the radius THIS track's condition was linked at (tester D10, 4 Nov 2026)
        detection_ceiling_D=params.detection_ceiling_D_for(str(track["condition"].iloc[0])),
        notes=notes,
    )


def _hmm_labels(track, fit, contiguous, notes):
    """Per-step HMM states for one track, aligned to the step array.

    ``decode_states`` works on contiguous runs, so steps across a gap have no
    state. They are left as ``-1`` rather than guessed.
    """
    from .hmm import decode_states

    decoded = decode_states(track, fit)
    n_steps = len(contiguous)
    states = np.full(n_steps, -1, dtype=int)
    posteriors = np.full(n_steps, np.nan)

    if "frame" in decoded.columns:
        # decode_states keys a step by the frame it STARTS on, so align on
        # frames[:-1]. Using the second point would shift the whole trace by one
        #, a bug that looks like a plausible time offset rather than an error.
        frame_to_index = {int(f): i for i, f in enumerate(track["frame"].to_numpy()[:-1])}
        for _, row in decoded.iterrows():
            index = frame_to_index.get(int(row["frame"]))
            if index is not None:
                states[index] = int(row["state"])
                posteriors[index] = float(row["posterior"])
    else:                                           # pragma: no cover - defensive
        notes.append("HMM decoding returned no frame column; states not aligned")

    if np.any(states < 0):
        notes.append(
            f"{int((states < 0).sum())} step(s) have no HMM state (frame gap or "
            "outside a decoded run)"
        )
    return states, posteriors
