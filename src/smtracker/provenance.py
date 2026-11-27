"""Provenance: a stamp of exactly what produced each result.

Now that real numbers are paper-bound, every result the tool emits carries a
record of the code, environment, randomness, settings and input data behind it,
so any figure can be traced back to the exact state that produced it, and so a
result computed on an uncommitted tree is *flagged*, never mistaken for a
reproducible one.

One source of truth
-------------------
``capture`` is the only function that builds a provenance record; every writer
calls it. There is no per-writer reimplementation, so a field added here appears
everywhere at once.

The cache-key substrate
-----------------------
The persistence cache (next step) will key results by a hash of the
*reproducibility-relevant* material: input data + settings + seeds + code
identity. That material is exactly what a provenance record holds, so
``Provenance.key()`` derives a stable SHA-256 over that subset. Two fields are
deliberately **excluded** from the key because they do not change the result:

- ``timestamp``: when it was written, not what it computed;
- ``environment``: the *display* dump of the full interpreter and package set.

A dependency change that *could* alter results is not excluded: the versions of
the load-bearing numeric packages live under ``code`` (code identity), so they
DO enter the key. The distinction is "could this change the number?", a git
commit and numpy's version can; the wall clock cannot.

Fail-loud
---------
If git is unavailable, or the working tree is dirty, the record says so
explicitly (``git_available``, ``dirty``) and ``reproducibly_stamped`` is
False. A result is never silently stamped as clean.

Form
----
- A structured JSON block travelling on ``result.attrs["provenance"]``.
- A companion ``<file>.provenance.json`` beside any exported file
  (``write_beside``).
- A commented header block for CSV exports (``csv_header``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from .childproc import NO_WINDOW

__all__ = [
    "ProvenanceError",
    "Provenance",
    "LOAD_BEARING_PACKAGES",
    "capture",
    "hash_units",
    "hash_frame",
]


class ProvenanceError(RuntimeError):
    """Raised when a result cannot capture honest provenance."""


#: Packages whose version could change a numeric result. Their versions are
#: **key material** (part of code identity), not just display: a scipy that
#: changes an optimizer default can move a fit. Anything not here (matplotlib,
#: napari, …) affects rendering, not results, and stays display-only.
LOAD_BEARING_PACKAGES = ("numpy", "scipy", "pandas", "saspt")

#: The full set recorded for the human-readable environment block (display only).
_DISPLAY_PACKAGES = LOAD_BEARING_PACKAGES + ("matplotlib", "mat73", "tifffile")


# --------------------------------------------------------------------------- #
# Content hashing, the input-membership fingerprint
# --------------------------------------------------------------------------- #


def hash_frame(frame: pd.DataFrame, *, columns=None) -> str:
    """Order-independent SHA-256 of a frame's content.

    The rows are sorted by their canonical key first, so the same data hashes
    the same regardless of how it happened to be ordered when it arrived. Only
    the content columns are hashed: grouping labels that merely *name* a unit
    are excluded, so relabelling a condition does not change what its data
    fingerprints to.
    """
    if columns is None:
        columns = [c for c in ("uid", "frame", "x_um", "y_um", "amp")
                   if c in frame.columns]
    if not columns:
        raise ProvenanceError("no hashable content columns in frame")
    sort_keys = [c for c in ("uid", "frame") if c in frame.columns] or columns
    ordered = frame.sort_values(sort_keys, kind="stable")[columns]
    row_hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def hash_units(canonical: pd.DataFrame, *, by: str = "cell") -> dict:
    """``{unit_key: content_hash}``, which units fed a result, each fingerprinted.

    ``by`` is the membership level to record (``condition``, ``day``, ``movie``
    or ``cell``). Recording it at the cell level makes the finest unit auditable;
    the cache can hash the whole mapping.
    """
    from .canonical import without_attrs

    if by not in canonical.columns:
        raise ProvenanceError(f"table has no {by!r} column to record membership")
    # OVER A VIEW WITHOUT `.attrs` (4 Nov 2026): pandas deep-copied the project table's attrs (acquisition, filter
    # records, the cells' geometry) into every cell's slice and every column read off it -- 4.8 million deepcopy calls
    # for one condition of a lab project's 175 cells, 13-29 s a cache key at idle priority, and an SQD Run under Whole
    # data set makes a dozen keys. The hash reads the content columns only, so every key is the same as before.
    return {
        str(key): hash_frame(part)
        for key, part in without_attrs(canonical).groupby(by, observed=True)
    }


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Provenance:
    """A reproducibility stamp. ``key()`` is the cache-relevant subset, hashed."""

    code: dict
    environment: dict
    seeds: dict
    settings: dict
    inputs: dict
    timestamp: str
    warnings: tuple = ()
    #: Full reasoned exclusion records, display only, excluded from the key.
    #: The key material is the *set of excluded unit IDs* (in ``inputs``); a
    #: typo-fix in a reason must not invalidate the cache.
    exclusions_detail: tuple = ()

    @property
    def reproducibly_stamped(self) -> bool:
        """Was this produced by identifiable, committed code?"""
        return bool(self.code.get("git_available") and not self.code.get("dirty"))

    def key_material(self) -> dict:
        """The subset that determines the RESULT: the cache's key inputs.

        Excludes ``timestamp`` (when, not what) and ``environment`` (display).
        Includes ``code`` in full, because a commit change or a load-bearing
        package version change can move a number.
        """
        return {
            "code": self.code,
            "seeds": self.seeds,
            "settings": self.settings,
            "inputs": self.inputs,
        }

    def key(self) -> str:
        """Stable SHA-256 over the key material, the persistence cache key."""
        blob = json.dumps(
            self.key_material(), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def content_key_material(self) -> dict:
        """Key material for a COMMIT-INSENSITIVE cache.

        Same as ``key_material`` but with ``code`` reduced to just the
        load-bearing package versions, the git commit and dirty flag are
        dropped. Two results with identical data / settings / seeds / package
        versions therefore share a content key even across ordinary commits.

        This exists ONLY for the committed *warm cache*, whose whole purpose is to
        survive the next commit (a cache invalidated by every commit is useless as
        a shipped convenience). The per-user ``ResultCache`` keeps using ``key()``
        and stays commit-sensitive by design, a code change there rightly misses.
        The honesty cost (a fit-logic change that keeps the same class structure
        could be served stale) is bounded: warm entries keep their full stamp
        (recording the commit they were computed at), and ``--refresh`` forces a
        rebuild.
        """
        return {
            "code": {"load_bearing_versions":
                     self.code.get("load_bearing_versions")},
            "seeds": self.seeds,
            "settings": self.settings,
            "inputs": self.inputs,
        }

    def content_key(self) -> str:
        """Stable SHA-256 over the commit-insensitive content material."""
        blob = json.dumps(
            self.content_key_material(), sort_keys=True, separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "environment": self.environment,
            "seeds": self.seeds,
            "settings": self.settings,
            "inputs": self.inputs,
            "exclusions_detail": list(self.exclusions_detail),
            "timestamp": self.timestamp,
            "warnings": list(self.warnings),
            "reproducibly_stamped": self.reproducibly_stamped,
            "key": self.key(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Provenance":
        return cls(
            exclusions_detail=tuple(data.get("exclusions_detail", ())),
            code=data["code"],
            environment=data["environment"],
            seeds=data["seeds"],
            settings=data["settings"],
            inputs=data["inputs"],
            timestamp=data["timestamp"],
            warnings=tuple(data.get("warnings", ())),
        )

    # -- exports ---------------------------------------------------------- #

    def write_beside(self, path: str | Path) -> Path:
        """Write ``<path>.provenance.json`` next to an exported file."""
        companion = Path(str(path) + ".provenance.json")
        companion.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return companion

    def csv_header(self, *, comment: str = "# ") -> str:
        """A commented header block for the top of a CSV export."""
        lines = [
            f"{comment}SMTracker result: provenance key {self.key()}",
            f"{comment}code: {self.code.get('commit', 'unknown')}"
            f"{' (DIRTY, not reproducibly stamped)' if self.code.get('dirty') else ''}",
            f"{comment}written: {self.timestamp}",
            f"{comment}settings: "
            f"{json.dumps(self.settings, sort_keys=True, default=str)}",
        ]
        for warning in self.warnings:
            lines.append(f"{comment}WARNING: {warning}")
        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        commit = self.code.get("commit", "unknown")
        stamp = "reproducible" if self.reproducibly_stamped else "NOT reproducible"
        # σ and its source ride in the on-figure stamp (item D): every floor and
        # σ-sensitivity is then traceable to a derived-or-measured value, and a
        # figure made with a manually-entered σ says so.
        sigma_nm = self.inputs.get("localization_sigma_nm")
        sigma_src = self.inputs.get("localization_sigma_source")
        sigma_clause = ""
        if sigma_nm is not None:
            sigma_clause = f" | σ {sigma_nm:.1f} nm ({sigma_src or 'unknown'})"
        return (
            f"provenance {self.key()[:12]} | {commit} ({stamp}) | "
            f"written {self.timestamp}{sigma_clause}"
        )


# --------------------------------------------------------------------------- #
# Capture, the one source of truth
# --------------------------------------------------------------------------- #


def _git(*args, repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _code_identity(repo: Path, warnings: list) -> dict:
    commit = _git("rev-parse", "HEAD", repo=repo)
    git_available = commit is not None
    dirty = None
    if git_available:
        status = _git("status", "--porcelain", repo=repo)
        dirty = bool(status)
        if dirty:
            warnings.append(
                "working tree is dirty (uncommitted changes): this result is "
                "NOT reproducibly stamped"
            )
    else:
        warnings.append("git is unavailable: code identity could not be captured")

    versions = {}
    for name in LOAD_BEARING_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "commit": commit,
        "git_available": git_available,
        "dirty": dirty,
        "load_bearing_versions": versions,
    }


def _environment() -> dict:
    versions = {}
    for name in _DISPLAY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "packages": versions}


def _clean(value):
    """Coerce to a reproducibly-serialisable value, or fail loud.

    ``json.dumps(..., default=str)`` is deliberately NOT used for settings: it
    would turn an arbitrary object into a string containing its memory address,
    which changes run to run and would silently break the cache key. Common
    scientific types (numpy scalars/arrays, Path) are converted explicitly;
    anything else raises, because a value that cannot be represented stably has
    no business in a reproducibility stamp.
    """
    import numpy as np

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    raise ProvenanceError(
        f"settings value {value!r} of type {type(value).__name__} is not "
        "reproducibly serialisable: a memory-address repr would break the key"
    )


def _canonical_settings(settings: dict) -> dict:
    """Settings as reproducibly-serialisable primitives, sorted for a stable key."""
    return _clean(dict(settings))


def capture(
    *,
    settings: dict,
    seeds=None,
    canonical: pd.DataFrame | None = None,
    unit: str = "cell",
    extra_inputs: dict | None = None,
    environment_extra: dict | None = None,
    now: datetime | None = None,
    repo: str | Path | None = None,
) -> Provenance:
    """Build the provenance record for a result. The one source of truth.

    Parameters
    ----------
    settings
        The full analysis settings that produced the result, n_pop, weighting,
        σ, bootstrap count, unit, thresholds, exclusion set, etc. Whatever the
        caller varied belongs here; it is key material.
    seeds
        The RNG seed(s). ``int``, ``dict`` or ``None`` (recorded as ``None`` so
        a deterministic, seed-free result is distinguishable from a forgotten
        one).
    canonical
        The input table; its per-``unit`` content hashes and its pixel size / τ
        are recorded as input membership. Omit only for results with no tabular
        input.
    unit
        Membership level to fingerprint (default ``cell``).
    extra_inputs
        Extra key-material inputs a caller wants recorded (e.g. a simulation
        truth manifest hash).
    now
        Injected timestamp, so tests are deterministic and the analysis never
        calls the clock itself.

    Fails loud only on unserialisable settings; a missing git or dirty tree is
    *recorded*, not raised, the result still ships, flagged.
    """
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    warnings: list = []

    code = _code_identity(repo, warnings)

    inputs: dict = {}
    if canonical is not None:
        params = canonical.attrs.get("params")
        inputs["pixel_size_nm"] = getattr(params, "pixel_size_nm", None)
        inputs["frame_interval_ms"] = getattr(params, "frame_interval_ms", None)
        # the ACTIVE localization σ and where it came from, so any floor or
        # σ-sensitivity is traceable to a derived-or-measured value (consequence 1)
        inputs["localization_sigma_nm"] = getattr(params, "localization_sigma_nm", None)
        inputs["localization_sigma_source"] = getattr(
            params, "localization_sigma_source", None)
        inputs["unit"] = unit
        inputs["n_localizations"] = int(len(canonical))
        try:
            inputs["units"] = hash_units(canonical, by=unit)
        except ProvenanceError as exc:
            warnings.append(str(exc))
        for level in ("condition", "day", "movie", "cell"):
            if level in canonical.columns:
                inputs[f"n_{level}"] = int(canonical[level].nunique())
    if extra_inputs:
        inputs.update(extra_inputs)

    # Exclusions: the KEY material is the SET of excluded unit IDs, sorted, with
    # no reason prose, fixing a typo in a reason must not invalidate the cache.
    # The full reasoned records ride along display-only, out of the key.
    exclusions_detail: tuple = ()
    if canonical is not None and canonical.attrs.get("exclusions"):
        records = canonical.attrs["exclusions"]
        inputs["excluded_units"] = sorted(
            f"{e['kind']}:{e['key']}" for e in records
        )
        exclusions_detail = tuple(records)

    if seeds is None:
        seed_record = {"seed": None}
    elif isinstance(seeds, dict):
        seed_record = seeds
    else:
        seed_record = {"seed": seeds}

    timestamp = (now or datetime.now(timezone.utc)).isoformat()

    return Provenance(
        code=code,
        environment={**_environment(), **(environment_extra or {})},
        seeds=seed_record,
        settings=_canonical_settings(settings),
        inputs=inputs,
        timestamp=timestamp,
        warnings=tuple(warnings),
        exclusions_detail=exclusions_detail,
    )
