"""Treated against a mock addition: does the treated condition change BEYOND the mock imaged on the same schedule?

WHY (8 Nov 2026). On real data with no addition at all, each movie's mean squared jump falls over a 20 to 35 min
imaging session: most replicates of a lab project, in both its conditions (`manuscript/kinetics_plan/real_data_null/`), with no
drift of the localisation error. So "after the addition against before it" on one pad is confounded with time in the session. The
answer is the treated condition's change against a mock addition's change (the same handling, nothing added),
recorded on the same schedule.

THE UNIT IS THE REPLICATE (Rule 0). Each replicate (culture) gives ONE change, its readout in an AFTER window against
its BEFORE window. The treated replicates and the mock replicates are two independent samples, compared with Welch
(`uncertainty.welch_test`), never paired by replicate number: a culture of the treated condition and one of the mock
share nothing but a label. Dropping a pairing narrows the interval through Welch's df, the direction that flatters an
effect; the sentences say so.

THE READOUTS:
- ``"mean_D"``: one D, the mean squared jump / (4 tau), less each step's localisation noise where per-spot precision
  is on (``induction.NOISE_COLUMN``; the mean of an exponential r^2 is 4 D tau + 2 noise). This is the maximum
  likelihood D of ONE population, what the Kinetics section's "Mean mobility" readout fits over time. A replicate's
  change is the log ratio after / before, reported as a percentage.
- ``"bound_share"``: the slowest population's share, two populations with their D's HELD at one pooled fit over every
  step of both conditions (so the treated and mock shares are read against the same D's), the shares refitted per
  replicate and window by maximum likelihood (EM on the weights). A replicate's change is after minus before, in
  percentage points. THE ALIAS: with the D's held, a fall of the fast D is read partly as a rise of this share
  (`ALIAS_SENTENCES`, said under the comparison as the mean D's is).

THE WINDOWS. Before: every step with a negative time since the addition (the baseline movies). After: every step from
``after_from_s`` on. `after_start` gives a default: where the treated condition's fitted response is 90 % complete,
else the second half of the record after the addition. A replicate with fewer than `MIN_STEPS` steps in a window is
left out and named, never filled in.

THE SMALLEST CHANGE FOUND (8 Nov 2026). Beside the test, one sentence says what change this comparison finds four
times in five, and with twice the replicates a side (`detectable_at`: Welch's test simulated at the replicates' own
SDs, seeded). A comparison that calls nothing so says how large a change could hide in it, and a plan reads how many
cultures it needs. For the SynK course data set's design (three baseline movies and two after a culture, 3 against 3) the
movie-to-movie scatter there, which is the real data's, gives about 38 pp of bound share and a 45 % fall in mean
mobility (`manuscript/kinetics_plan/real_data_null/synk_design_table.py`). With fewer than five replicates a side the
sentence says it is rough, the SDs being estimates themselves.

Headless; imports no Qt (`tests/test_ui_thin_qt.py`).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .induction import CLOCK, NOISE_COLUMN

READOUTS = ("mean_D", "bound_share")
READOUT_WORDS = {"mean_D": "mean mobility (one D)", "bound_share": "bound share"}
#: What each readout CANNOT tell apart, the last sentence under its comparison. The bound share's (10 Nov 2026): its
#: D's are held at one pooled fit, so a fast population that slows down is read partly as molecules binding. Measured
#: on the SynK course data set's D drop, declared truth "no change of the bound share": plus 9.4 pp per culture and plus
#: 22.1 pp beyond the mock (p 0.067). The mean D's sentence stood alone until then.
ALIAS_SENTENCES = {
    "mean_D": "A change of the mean D comes from a change in D or in the bound share alike; Populations over time "
              "tells which.",
    "bound_share": "The bound share is read at D values held at one pooled fit, so a fast population that slows down "
                   "reads in part as a rise of the bound share; Populations over time tells which.",
}
#: The fewest steps a replicate needs in a window for its value to count.
MIN_STEPS = 200
#: The share of the response complete where the after window starts, by default.
AFTER_COMPLETE = 0.9
_EM_ITER = 500
_EM_TOL = 1e-9
#: The smallest change a comparison FINDS (`detectable_at`): Welch's test, two-sided at `DETECT_ALPHA`, finding it
#: in `DETECT_POWER` of experiments ("four times in five"), simulated (`DETECT_DRAWS` seeded experiments) rather than
#: from the textbook shortcut (t_0.975 + t_0.80) SE: at three replicates a side Welch's own df is lower and random,
#: and the shortcut's size is found only 70 % of the time (Welch reaches 80 % at 3.48 SD, the shortcut says 3.04;
#: at five a side 2.08 against 2.02; 8 Nov 2026).
DETECT_POWER = 0.8
DETECT_ALPHA = 0.05
DETECT_DRAWS = 4000


class MockError(RuntimeError):
    """Raised when a treated-against-mock comparison cannot be made honestly."""


def after_start(steps: pd.DataFrame, fit=None) -> tuple:
    """``(seconds, why)``: where the after window starts. With ``fit`` (an `induction.InductionFit` of the TREATED
    condition that found a change), the first time its response is `AFTER_COMPLETE` complete; without one, or where
    the response never gets there inside the data, the middle of the record after the addition."""
    t = steps[CLOCK].to_numpy(float)
    t = t[np.isfinite(t)]
    post = t[t >= 0]
    if post.size == 0:
        raise MockError("no step after the addition: there is no after window")
    middle = float(0.5 * (post.min() + post.max()))
    if fit is not None and getattr(fit, "change", "none") != "none":
        grid = np.linspace(0.0, float(post.max()), 400)
        h = fit.response(grid)
        done = np.flatnonzero(h >= AFTER_COMPLETE)
        if done.size:
            at = float(grid[done[0]])
            return at, f"from {at:.0f} s, where the treated response is {100 * AFTER_COMPLETE:.0f} % complete"
    return middle, f"from {middle:.0f} s, the second half of the record after the addition"


def _noise(part: pd.DataFrame) -> np.ndarray:
    if NOISE_COLUMN in part.columns:
        return part[NOISE_COLUMN].to_numpy(float)
    return np.zeros(len(part))


def mean_D(part: pd.DataFrame, tau_s: float) -> float:
    """One population's maximum-likelihood D on these steps: (mean r^2 - 2 mean noise) / (4 tau)."""
    r2 = part["sq_disp_um2"].to_numpy(float)
    return float((np.mean(r2) - 2.0 * np.mean(_noise(part))) / (4.0 * float(tau_s)))


def shares_at_D(part: pd.DataFrame, tau_s: float, D) -> np.ndarray:
    """The populations' shares on these steps with their D's HELD at ``D`` (slowest first), by maximum likelihood (EM
    on the weights; each step's population mean 4 D tau + 2 noise, as `induction` and `sqd` have it)."""
    from .sqd import _log_mixture_density

    r2 = part["sq_disp_um2"].to_numpy(float)
    D = np.sort(np.asarray(D, float))
    means = 4.0 * float(tau_s) * D[None, :] + 2.0 * _noise(part)[:, None]
    f = np.full(len(D), 1.0 / len(D))
    for _ in range(_EM_ITER):
        log_g, a = _log_mixture_density(r2, means, np.broadcast_to(f, means.shape))
        if log_g is None:
            raise MockError("the mixture density is not positive on these steps")
        resp = np.exp(a + np.log(f)[None, :] - log_g[:, None])
        new = resp.mean(axis=0)
        new = new / new.sum()
        if np.max(np.abs(new - f)) < _EM_TOL:
            f = new
            break
        f = new
    return f


def pooled_D(parts, tau_s: float, n_pop: int = 2) -> tuple:
    """Two populations' D's from ONE maximum-likelihood fit over every step given (both conditions, both windows),
    slowest first: the D's every share is read against."""
    from .sqd import fit_sqd

    steps = pd.concat(list(parts), ignore_index=True)
    noise = _noise(steps)
    fit = fit_sqd(steps["sq_disp_um2"].to_numpy(float), n_pop=int(n_pop), tau_lag_s=float(tau_s), method="mle",
                  **({"step_noise_um2": noise} if np.any(noise) else {}))
    return tuple(float(v) for v in np.sort(np.asarray(fit.D, float)))


def replicate_changes(steps: pd.DataFrame, *, readout: str, after_from_s: float, tau_s: float,
                      D=None) -> pd.DataFrame:
    """One row per replicate (the step table's ``day`` column): its value before and after, the steps in each
    window, its change (``mean_D``: the log ratio after / before; ``bound_share``: after minus before, in pp), and
    why a replicate was left out (empty where it counts)."""
    if readout not in READOUTS:
        raise MockError(f"readout {readout!r}: choose one of {READOUTS}")
    if readout == "bound_share" and D is None:
        raise MockError("the bound share is read against held D's: give D (`pooled_D`)")
    where = {"before": "before the addition", "after": "in the after window"}
    rows = []
    for day, part in steps.groupby("day", sort=True):
        tt = part[CLOCK].to_numpy(float)
        before, after = part[tt < 0], part[tt >= float(after_from_s)]
        row = {"replicate": day, "n_before": len(before), "n_after": len(after), "before": math.nan,
               "after": math.nan, "change": math.nan, "left_out": ""}
        short = [w for w, p in (("before", before), ("after", after)) if len(p) < MIN_STEPS]
        if short:
            row["left_out"] = f"fewer than {MIN_STEPS} steps " + " and ".join(where[w] for w in short)
            rows.append(row)
            continue
        if readout == "mean_D":
            b, a = mean_D(before, tau_s), mean_D(after, tau_s)
            if not (b > 0 and a > 0):
                row["left_out"] = "a mean D at or below 0 once the noise is taken out"
            else:
                row.update(before=b, after=a, change=math.log(a / b))
        else:
            b = 100.0 * float(shares_at_D(before, tau_s, D)[0])
            a = 100.0 * float(shares_at_D(after, tau_s, D)[0])
            row.update(before=b, after=a, change=a - b)
        rows.append(row)
    return pd.DataFrame(rows)


def _one_sample(values, confidence: float = 0.95) -> tuple:
    """Mean and its t interval over the replicates; (mean, nan, nan) below two."""
    from scipy import stats

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan, math.nan
    m = float(v.mean())
    if v.size < 2:
        return m, math.nan, math.nan
    half = float(stats.t.ppf(0.5 + confidence / 2, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size))
    return m, m - half, m + half


def detectable_at(sd_a: float, n_a: int, sd_b: float, n_b: int, *, power: float = DETECT_POWER,
                  alpha: float = DETECT_ALPHA, draws: int = DETECT_DRAWS, seed: int = 0) -> float:
    """The smallest true difference ``mean(b) - mean(a)`` that Welch's test (two-sided ``alpha``) finds in ``power``
    of experiments with ``n_a`` and ``n_b`` values of SDs ``sd_a`` and ``sd_b`` (normal), by seeded simulation: the
    same experiments at every shift (common random numbers), the shift found by bisection. nan below two values a
    side or with no spread on either side."""
    from scipy import stats

    n_a, n_b = int(n_a), int(n_b)
    sd_a, sd_b = float(sd_a), float(sd_b)
    if n_a < 2 or n_b < 2 or not (math.isfinite(sd_a) and math.isfinite(sd_b)) or (sd_a <= 0 and sd_b <= 0):
        return math.nan
    rng = np.random.default_rng(seed)
    za = rng.standard_normal((int(draws), n_a)) * sd_a
    zb = rng.standard_normal((int(draws), n_b)) * sd_b
    va, vb = za.var(axis=1, ddof=1) / n_a, zb.var(axis=1, ddof=1) / n_b
    se = np.sqrt(va + vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1))
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    base = zb.mean(axis=1) - za.mean(axis=1)

    def found(shift: float) -> float:
        return float(np.mean(np.abs((base + shift) / se) > crit))

    scale = math.sqrt(sd_a ** 2 / n_a + sd_b ** 2 / n_b)
    lo, hi = 0.0, 4.0 * scale
    while found(hi) < power:
        lo, hi = hi, 2.0 * hi
        if hi > 1e3 * scale:
            return math.nan
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if found(mid) < power else (lo, mid)
    return float(hi)


def detectable_change(a, b, *, times: int = 1) -> float:
    """`detectable_at` with the SDs of the replicate values ``a`` (mock) and ``b`` (treated) taken as the truth and
    ``times`` as many replicates a side: what this comparison would find four times in five. Rough with few
    replicates, their SDs being estimates themselves (at three a side the estimate falls between 0.6 and 1.6 times
    the true value in 8 of 10 experiments)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return math.nan
    return detectable_at(float(a.std(ddof=1)), int(times) * a.size, float(b.std(ddof=1)), int(times) * b.size)


def _left_out(tables) -> list:
    """Each replicate left out and why, as words: ``[(side, replicate, reason)]``."""
    return [(side, r["replicate"], r["left_out"]) for side, table in tables for _, r in table.iterrows()
            if r["left_out"]]


def _noise_usable(part: pd.DataFrame) -> bool:
    return NOISE_COLUMN in part.columns and bool(np.isfinite(part[NOISE_COLUMN].to_numpy(float)).all())


def compare(treated: pd.DataFrame, mock: pd.DataFrame, *, readout: str = "mean_D", after_from_s: float,
            tau_s: float | None = None, n_pop: int = 2) -> dict:
    """The treated condition's change against the mock's, replicate against replicate (Welch, unpaired).

    ``treated`` / ``mock``: each ONE condition's steps on the induction clock (`induction.induction_steps`, with
    ``day`` and, for per-spot precision, `NOISE_COLUMN`), recorded on the same schedule. ``n_pop``: the populations
    the bound share is read with (the Run's own count; the bound share is the slowest's). Returns the per-replicate
    tables, each condition's mean change with its t interval, the Welch test of treated minus mock, and the sentences.

    BOTH ARMS ARE READ ALIKE (8 Nov 2026, review): per-spot noise is taken out only where BOTH conditions carry it for
    every step; where only one does, it is left out of both and the sentences say so (taking it out of one arm alone
    biased the mean D's ratio, and NaNs in the pooled fit crashed the bound share).
    """
    from .uncertainty import welch_test

    tau = float(tau_s if tau_s is not None else treated.attrs.get("tau_s", math.nan))
    if not tau > 0:
        raise MockError("no frame interval: give tau_s")
    use_t, use_m = _noise_usable(treated), _noise_usable(mock)
    noise_note = ""
    if use_t != use_m:
        treated = treated.drop(columns=[NOISE_COLUMN], errors="ignore")
        mock = mock.drop(columns=[NOISE_COLUMN], errors="ignore")
        noise_note = (f"Per-spot localisation error left out of this comparison: only the "
                      f"{'treated condition' if use_t else 'mock'} carries it, and both are read alike.")
    D = None
    if readout == "bound_share":
        t_t, t_m = treated[CLOCK].to_numpy(float), mock[CLOCK].to_numpy(float)
        D = pooled_D([treated[(t_t < 0) | (t_t >= after_from_s)], mock[(t_m < 0) | (t_m >= after_from_s)]], tau,
                     n_pop=int(n_pop))
    tr = replicate_changes(treated, readout=readout, after_from_s=after_from_s, tau_s=tau, D=D)
    mk = replicate_changes(mock, readout=readout, after_from_s=after_from_s, tau_s=tau, D=D)
    a, b = mk["change"].dropna().to_numpy(float), tr["change"].dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        left = _left_out((("treated", tr), ("mock", mk)))
        why = ("; left out: " + "; ".join(f"replicate {r} of the {side}: {reason}" for side, r, reason in left)
               if left else "")
        raise MockError(f"{len(b)} treated and {len(a)} mock replicates with both windows: Welch needs two a side "
                        f"(each replicate one value){why}")
    w = welch_test(a, b)
    out = {"readout": readout, "after_from_s": float(after_from_s), "D": D, "treated": tr, "mock": mk,
           "treated_change": _one_sample(b), "mock_change": _one_sample(a), "welch": w, "n_pop": int(n_pop),
           "noise_note": noise_note,
           "detectable": (detectable_change(a, b), detectable_change(a, b, times=2))}
    out["lines"] = lines(out)
    return out


def _pct(log_ratio: float) -> float:
    return 100.0 * (math.exp(log_ratio) - 1.0)


def detectable_line(readout: str, detectable, n_t: int, n_m: int) -> str:
    """The smallest change this comparison finds four times in five, and with twice the replicates; "" where it
    cannot be said (`detectable_change` nan). A mean D's is a log ratio, said as the fall AND the rise it means."""
    if not detectable:
        return ""
    now, doubled = (float(v) for v in detectable)
    if not math.isfinite(now):
        return ""

    def size(v: float) -> str:
        if readout == "mean_D":
            return (f"a fall of about {100.0 * (1.0 - math.exp(-v)):.0f} % (or a rise of "
                    f"{100.0 * (math.exp(v) - 1.0):.0f} %)")
        return f"about {v:.0f} pp either way"

    more = f"; with twice the replicates a side, {size(doubled)}" if math.isfinite(doubled) else ""
    rough = (" Rough with so few replicates: their scatter is itself an estimate." if min(n_t, n_m) < 5 else "")
    return (f"Smallest change this comparison finds four times in five, at its replicates' own scatter: "
            f"{size(now)}{more}.{rough}")


def lines(result: dict) -> list:
    """The sentences: each condition's own change, then the change beyond the mock with its interval and p."""
    readout, w = result["readout"], result["welch"]
    n_t, n_m = int(w.n_b), int(w.n_a)

    def say(v, lo, hi):
        if readout == "mean_D":
            core = f"{_pct(v):+.0f} %"
            span = "" if not math.isfinite(lo) else f" [{_pct(lo):+.0f}, {_pct(hi):+.0f}]"
        else:
            core = f"{v:+.1f} pp"
            span = "" if not math.isfinite(lo) else f" [{lo:+.1f}, {hi:+.1f}]"
        return core + span

    what = READOUT_WORDS[readout]
    if readout == "bound_share" and int(result.get("n_pop", 2)) != 2:
        what = f"bound share (the slowest of {int(result['n_pop'])} populations)"
    out = [f"Treated, {what}: {say(*result['treated_change'])} from before to after ({n_t} replicates).",
           f"Mock, {what}: {say(*result['mock_change'])} ({n_m} replicates).",
           (f"Beyond the mock: {say(w.difference, w.ci_low, w.ci_high)}, Welch over {n_t} against {n_m} "
            f"replicates, p = {w.p:.2g}" if math.isfinite(w.p) else
            f"Beyond the mock: {say(w.difference, w.ci_low, w.ci_high)}; no p ({w.reason})")
           + (" (unpaired: Welch's df narrows the interval, the direction that flatters an effect)."
              if math.isfinite(w.p) else ".")]
    found = detectable_line(readout, result.get("detectable"), n_t, n_m)
    if found:
        out.append(found)
    left = _left_out((("treated", result["treated"]), ("mock", result["mock"])))
    if left:
        out.append("Left out: " + "; ".join(f"replicate {r} of the {side}: {reason}" for side, r, reason in left) + ".")
    if result.get("noise_note"):
        out.append(result["noise_note"])
    out.append(ALIAS_SENTENCES[readout])
    return out
