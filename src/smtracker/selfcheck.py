"""The analysis self-test: one declared-truth set through the SQD fit and the HMM, against stored numbers.

WHY (10 Nov 2026, the software review's item 3). `smtracker --selftest` built a napari viewer and tore it down: it
proved that Qt, vispy and napari were bundled, and nothing about whether the build COMPUTES what the code it was
built from computes. A frozen numpy or scipy that differs from the source environment's, a missing compiled module
that a fallback quietly covers, a changed default: each would ship with every check passing. `--dry-run` reads a
real project inside the bundle but compares its numbers with nothing. This module is the missing comparison, and it
needs no data: the set is simulated here, from a seed.

WHAT IT RUNS. `simulated()` draws `N_TRACKS` Brownian tracks from a two-population mixture whose D and shares are
DECLARED (`TRUTH_D`, `TRUTH_FRACTIONS`), each localisation blurred by `SIGMA_UM`, at the lab's acquisition (106 nm
pixels, 20 ms). `measure()` puts it through the two engines the results rest on, with the library's defaults:

- `sqd.fit_sqd`, two populations, the localisation error given;
- `hmm.fit_hmm`, two states, the same error given.

TWO COMPARISONS, NEVER MIXED (`run`):

- against `STORED`, the numbers this code gave in the environment the bundle is built from: TIGHT
  (`STORED_D_RTOL`, `STORED_SHARE_PP`). It asks "does this build compute what the checked code computed"; it is the
  one a broken bundle fails. A fit is deterministic given its seed, so the tolerance covers floating-point
  differences between machines and nothing else.
- against the declared truth: LOOSE (`TRUTH_D_RTOL`, `TRUTH_SHARE_PP`), the sampling error of this many tracks with
  room to spare (the spread over 20 other seeds is in `docs/claude/qt_and_packaging.md`, "THE SELF-TEST COMPUTES").
  It asks "is what it computes right", and it keeps the stored numbers honest: a wrong number stored once would
  otherwise pass for ever.

The set's own fingerprint (step count, sum of r²) is checked FIRST. Where it differs, the random generator or the
simulator changed and the stored comparison compares two different data sets; the check says so by name instead of
leaving a reader to work out why every number is off. numpy does not promise a `Generator`'s stream across
versions: after a numpy upgrade this is the line to expect, and `measure()` prints what to store.

The simulated molecules NEVER SWITCH (`simulate.mixture_tracks`), so the HMM's last check is that it finds no
switching (`NO_SWITCH_MAX` per frame): a self-test of the transition matrix with a known answer.

INVARIANTS
- Headless: imports with Qt blocked; numpy, pandas and the spine only.
- It never raises and never stops at the first failure. A stage that raises is a FAILED check carrying the error
  and its traceback, and the stages after it still run, so one report names everything that is wrong. That is
  reporting, not swallowing: the run fails (`passed` is False, `--selftest` exits non-zero).
- `STORED` changes only together with a behaviour change that is meant, in the same commit, with the reason in the
  changelog. Never to make a failing self-test pass.
- Fast: a few seconds (the least-squares SQD fit and the HMM; the SQD maximum-likelihood fit takes about 40 s on
  this set and is left to the test suite).

INTERFACE
- `simulated()` -> the canonical table; `measure()` -> the numbers as a dict (what `STORED` holds).
- `run()` -> list of `Check(name, ok, detail)`; `passed(checks)`; `report(checks)` -> the text printed.
- `main(verbose=True)` -> bool: run, print, say whether every check passed (`app`'s `--selftest`, the build's
  verify).
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Check", "N_TRACKS", "NO_SWITCH_MAX", "SEED", "SIGMA_UM", "STORED", "STORED_D_RTOL", "STORED_SHARE_PP",
    "TRUTH_D", "TRUTH_D_RTOL", "TRUTH_FRACTIONS", "TRUTH_SHARE_PP", "main", "measure", "passed", "report", "run",
    "simulated",
]

#: The declared truth: D in µm²/s and each population's share (of tracks, and so the expected share of steps).
TRUTH_D = (0.05, 0.8)
TRUTH_FRACTIONS = (0.35, 0.65)
#: Localisation error of every coordinate, µm (30 nm), given to both fits as known.
SIGMA_UM = 0.03
PIXEL_NM = 106.0
TAU_MS = 20.0
N_TRACKS = 1500
SEED = 20261110

#: Against the stored numbers: floating-point differences between machines, nothing else.
STORED_D_RTOL = 0.005
STORED_SHARE_PP = 0.2
#: Against the declared truth: the sampling error of `N_TRACKS` tracks, with room. Over 20 other seeds the largest
#: error was 13.8 % in D (the SQD fit's slow D, which sits beside the localisation offset of 0.045 µm²/s; SD 7 %) and
#: 2.5 pp in a share (SD 1.2 pp), so these are about three SDs.
TRUTH_D_RTOL = 0.20
TRUTH_SHARE_PP = 4.0
#: The most switching per frame the HMM may report where none was simulated (0.0016 at most over those 20 seeds).
NO_SWITCH_MAX = 0.01
#: The fingerprint's relative tolerance on the sum of r²: summation order, not data.
_SAME_SET_RTOL = 1e-9

#: What `measure()` gave in the environment the bundle is built from (Python 3.13, the versions in `about.txt`),
#: 10 Nov 2026. See the invariant above before touching it.
STORED = {
    "steps": 14154,
    "sum_r2_um2": 638.4403087944916,
    "sqd_D": (0.05065786600487134, 0.7858264330595495),
    "sqd_fractions": (0.3604914146110952, 0.6395085853889049),
    "hmm_D": (0.048589815101303, 0.7775145097329951),
    "hmm_occupancy": (0.3548798574681116, 0.6451201425318884),
}


@dataclass(frozen=True)
class Check:
    """One line of the self-test: what was asked, whether it held, and the numbers."""

    name: str
    ok: bool
    detail: str


def _params():
    from .canonical import AcquisitionParams

    return AcquisitionParams(pixel_size_nm=PIXEL_NM, frame_interval_ms=TAU_MS)


def simulated():
    """The declared-truth set: `N_TRACKS` tracks of 6 to 40 localisations, each one population for its whole length."""
    from . import simulate

    # The lengths take their OWN seed: drawn from the seed the simulator uses, the first draws of one stream would
    # set both a track's length and its population, and the share of steps would no longer be the declared share.
    lengths = np.clip(5 + np.random.default_rng(SEED + 1).geometric(0.18, size=N_TRACKS), 5, 40)
    return simulate.mixture_tracks(lengths, D=list(TRUTH_D), fractions=list(TRUTH_FRACTIONS), params=_params(),
                                   sigma_um=SIGMA_UM, seed=SEED)


def _observable(tracks):
    from . import observables

    return observables.squared_displacements(tracks)


def _sqd(r2) -> dict:
    from . import sqd

    fit = sqd.fit_sqd(r2, n_pop=2, localization_sigma_um=SIGMA_UM)
    return {"sqd_D": tuple(float(v) for v in fit.D), "sqd_fractions": tuple(float(v) for v in fit.fractions)}


def _hmm(tracks) -> dict:
    from . import hmm

    fit = hmm.fit_hmm(hmm.step_sequences(tracks), n_states=2, tau_s=TAU_MS / 1000.0, localization_sigma_um=SIGMA_UM)
    A = np.asarray(fit.transition_matrix, dtype=float)
    return {"hmm_D": tuple(float(v) for v in fit.D), "hmm_occupancy": tuple(float(v) for v in fit.occupancy),
            "hmm_switching": float(max(A[0, 1], A[1, 0]))}


def measure() -> dict:
    """Every number the self-test compares, computed now. Raises what the engines raise: this is the developer's
    call (what to put in `STORED` after a change that is meant), `run` is the one that never raises."""
    tracks = simulated()
    r2 = _observable(tracks)
    values = r2["sq_disp_um2"].to_numpy()
    out = {"steps": int(values.size), "sum_r2_um2": float(values.sum())}
    out.update(_sqd(r2))
    out.update(_hmm(tracks))
    return out


def _error(exc: BaseException) -> str:
    return f"raised {type(exc).__name__}: {exc}\n{traceback.format_exc().rstrip()}"


def _numbers(values, percent: bool = False) -> str:
    if percent:
        return " / ".join(f"{100.0 * v:.2f}" for v in values) + " %"
    return " / ".join(f"{v:.4f}" for v in values) + " µm²/s"


def _close(D, shares, D_ref, shares_ref, rtol: float, pp: float) -> bool:
    D, shares, D_ref, shares_ref = (np.asarray(v, dtype=float) for v in (D, shares, D_ref, shares_ref))
    if D.shape != D_ref.shape or shares.shape != shares_ref.shape:
        return False
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(shares))):
        return False
    return bool(np.all(np.abs(D - D_ref) <= rtol * np.abs(D_ref)) and np.all(100.0 * np.abs(shares - shares_ref) <= pp))


def _pair(label: str, share_word: str, D, shares, stored_D, stored_shares) -> list[Check]:
    """The two comparisons of one engine's result: the stored numbers (tight), the declared truth (loose)."""
    found = f"D {_numbers(D)}, {share_word} {_numbers(shares, percent=True)}"
    return [
        Check(f"{label} gives the stored numbers", _close(D, shares, stored_D, stored_shares, STORED_D_RTOL,
                                                          STORED_SHARE_PP),
              f"{found}; stored D {_numbers(stored_D)}, {share_word} {_numbers(stored_shares, percent=True)} "
              f"(within {100 * STORED_D_RTOL:g} % and {STORED_SHARE_PP:g} pp)"),
        Check(f"{label} recovers the declared truth", _close(D, shares, TRUTH_D, TRUTH_FRACTIONS, TRUTH_D_RTOL,
                                                             TRUTH_SHARE_PP),
              f"{found}; declared D {_numbers(TRUTH_D)}, shares {_numbers(TRUTH_FRACTIONS, percent=True)} "
              f"(within {100 * TRUTH_D_RTOL:g} % and {TRUTH_SHARE_PP:g} pp)"),
    ]


def run() -> list[Check]:
    """Every check, in order. Never raises: a stage that raises is a failed check, and the next stage still runs."""
    try:
        tracks = simulated()
        r2 = _observable(tracks)
        values = r2["sq_disp_um2"].to_numpy()
        steps, total = int(values.size), float(values.sum())
    except Exception as exc:  # noqa: BLE001 -- reported as a failed check, with its traceback; the run fails
        return [Check("the declared-truth set is simulated", False, _error(exc))]

    same = steps == STORED["steps"] and abs(total - STORED["sum_r2_um2"]) <= _SAME_SET_RTOL * abs(STORED["sum_r2_um2"])
    detail = f"{steps:,} steps, sum of r² {total:.9f} µm²; stored {STORED['steps']:,} and {STORED['sum_r2_um2']:.9f}"
    if not same:
        detail += (". The random generator or the simulator changed, so the stored numbers below belong to "
                   "another data set (a numpy upgrade can do this)")
    checks = [Check("the simulated set is the stored one", same, detail)]

    try:
        got = _sqd(r2)
        checks += _pair("the SQD fit", "fractions", got["sqd_D"], got["sqd_fractions"], STORED["sqd_D"],
                        STORED["sqd_fractions"])
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("the SQD fit runs", False, _error(exc)))

    try:
        got = _hmm(tracks)
        checks += _pair("the HMM", "occupancy", got["hmm_D"], got["hmm_occupancy"], STORED["hmm_D"],
                        STORED["hmm_occupancy"])
        checks.append(Check("the HMM finds no switching where none was simulated",
                            bool(np.isfinite(got["hmm_switching"]) and got["hmm_switching"] <= NO_SWITCH_MAX),
                            f"largest switching probability {got['hmm_switching']:.5f} per frame "
                            f"(at most {NO_SWITCH_MAX:g})"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("the HMM runs", False, _error(exc)))
    return checks


def passed(checks) -> bool:
    """Whether every check held. No checks is a failure, never a pass."""
    checks = list(checks)
    return bool(checks) and all(c.ok for c in checks)


def report(checks) -> str:
    """The text `--selftest` prints: one line per check, then one verdict line a script can look for."""
    checks = list(checks)
    lines = [f"  {'ok  ' if c.ok else 'FAIL'}  {c.name}: {c.detail}" for c in checks]
    failed = sum(not c.ok for c in checks)
    verdict = "OK" if passed(checks) else "FAILED"
    count = f"{len(checks)} checks" if not failed else f"{failed} of {len(checks)} checks failed"
    lines.append(f"analysis self-test: {verdict} ({count}; {N_TRACKS:,} simulated tracks of declared D and shares, "
                 "through the SQD fit and the HMM)")
    return "\n".join(lines)


def main(*, verbose: bool = True) -> bool:
    """Run the self-test, print its report, and say whether every check passed."""
    checks = run()
    if verbose:
        print(report(checks))
    return passed(checks)
