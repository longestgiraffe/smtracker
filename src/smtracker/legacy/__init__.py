"""Legacy mode: the OLD SMTracker method, run on the current data.

What this is for. The old tool produced maps and fits that the lab knows. This package
reproduces that output from the same data, so the old and the new can be looked at
without an argument about which one is being shown.

What this is not. It is not a second analysis. It is a display of the old method. No
number from this package may enter the manuscript deposit.

Four rules, and each one is enforced in code, not by convention.

1. ONE SWITCH. The CLI flag is ``--legacy``.
2. One output directory. Every write goes through ``legacy_path``, which refuses any
   path outside ``legacy_out/``. The guard has a positive control in
   ``tests/test_legacy_guard.py``.
3. Every output is marked. The word ``legacy`` is in each file name and in each figure
   title. ``legacy_title`` builds the title and refuses one without the word.
4. No side by side. There is no comparison function here and none is to be added. The
   two output directories keep the old and the new apart on disk.

The central distinction. A design choice is reproduced. A defect is fixed. A defect is
code that does not do what its own comment, label or variable name says. Every fix is
one row in ``legacy_out/legacy_deviations.csv``, written by ``deviations``.
"""
from .guard import (LegacyGuardError, LEGACY_ROOT, legacy_path, legacy_title,
                    assert_inside_legacy, assert_neutral)
from .deviations import DEVIATIONS, write_deviations
from .limitations import LIMITATION_LINES, limitation_line

__all__ = ["LegacyGuardError", "LEGACY_ROOT", "legacy_path", "legacy_title",
           "assert_inside_legacy", "assert_neutral", "DEVIATIONS", "write_deviations",
           "LIMITATION_LINES", "limitation_line"]
