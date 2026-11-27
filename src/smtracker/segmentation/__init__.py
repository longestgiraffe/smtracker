"""Segmentation: brightfield images to Oufti cell meshes, and the editor that corrects them.

Ported, not rewritten. The working code came from an earlier, separate segmentation project
where it produced a real analysis. What is brought across is the general rod-shaped-bacteria
machinery; that project's own analysis and scoring are deliberately left behind, and its
historical naming is stripped.

No second mesh I/O layer. `smtracker.cellmesh` reads Oufti outlines and is the
stricter reader of the two: it refuses a ribless outline, names a non-finite mesh, and
skips `movieList.mat` by name rather than by sniffing for a variable. So reading stays
there and this package never grows its own. What SMTracker lacked is a WRITER, which the
editor needs; that is the one genuinely new capability.

**Dependency split: inference in-process, training as a subprocess.** Measured, torch's
CUDA build is 3.1 GB against a 407 MB frozen bundle, and two thirds of it is CUDA that a
trained model does not need: the experimenter measured CPU inference at nearly GPU speed.
So the installer carries CPU-only torch and never CUDA, and GPU training is a FETCHED
component driven as a subprocess, exactly as `smtracker.fiji` handles Fiji. One torch per
process, always. See `docs/segmentation_integration.md`.
"""
from __future__ import annotations

__all__ = ["geometry"]
