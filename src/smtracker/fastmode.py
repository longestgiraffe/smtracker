"""Fast dev profile: recompute quickly during UI work, but NEVER silently.

UI iteration must not wait on full-rigor fits, so ``--fast`` trades statistical
rigor for speed: fewer bootstrap resamples, fewer multistart seeds, capped HMM
iterations, and an optional track subsample. It exists ONLY for developing the
interface: a fast-mode number is not an analysis number.

The honesty guard (non-negotiable)
----------------------------------
A fast-mode result must be impossible to mistake for, or export as, a real one:
- **Stamped into provenance.** ``fast`` is written into the settings that form
  the cache KEY, so a fast result has a different key than the full result: it
  can never be served in place of a real one, and its exported
  ``.provenance.json`` says ``fast: {...}`` plainly.
- **Bannered on every figure and in the GUI.** The controller paints
  ``BANNER`` across each view and the window shows the same warning, so a
  screenshot carries the caveat with it.

Full rigor is the default; fast mode is opt-in and loud.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["FastProfile", "BANNER"]

#: The words shown on every fast-mode figure and in the console. Deliberately
#: AVOIDS "fast": that word is on the figure-neutrality banned list (it names a
#: diffusive state), and a strict substring gate cannot tell "fast mode" apart
#: from "fast state". The GUI checkbox, which is not gated, still reads "DEV FAST
#: MODE"; the on-figure marker says the same thing without the collision.
BANNER = "DEV MODE: reduced rigor, not for analysis"


@dataclass(frozen=True)
class FastProfile:
    """Reduced-rigor knobs for fast UI iteration. Presence ITSELF is the flag.

    Defaults are deliberately cheap, not merely smaller: ~50 resamples, a handful
    of seeds, capped EM iterations. ``track_subsample`` (a fraction in (0, 1]) is
    off by default: turn it on for the largest conditions when even the reduced
    fits drag.
    """

    n_boot: int = 50
    sqd_n_starts: int = 6
    hmm_n_starts: int = 3
    hmm_max_iter: int = 60
    track_subsample: float | None = None

    def as_settings(self) -> dict:
        """The dict written into provenance settings: the honesty stamp."""
        return {"fast": True, **asdict(self)}

    @property
    def banner(self) -> str:
        return BANNER
