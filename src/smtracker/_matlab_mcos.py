"""Read MATLAB class objects (MCOS) out of a ``.mat`` file. Low-level, internal.

Why this exists
---------------
u-track stores its ``MovieData`` as a MATLAB **class object**, not a struct.
``scipy.io.loadmat`` returns it as an opaque placeholder (``MatlabOpaque``) and
``mat73`` does not help (these are v7 files, not v7.3). Everything we need to
stop *guessing* acquisition parameters (pixel size, frame interval, numerical
aperture, and the tracking settings) lives inside that object. So we decode it.

Format (MAT-file v7 "MCOS" / FileWrapper__), as observed
--------------------------------------------------------
1. The file holds an undocumented variable ``__function_workspace__``: a byte
   blob that is itself a headerless mini MAT-file. Prepending a synthetic
   128-byte header makes it readable with the normal reader.
2. It contains one struct with a single field ``MCOS``, an opaque of class
   ``FileWrapper__`` whose object metadata is a cell array. Cell 0 is a binary
   *metadata blob*; cells 3.. hold the actual property values.
3. The metadata blob:
   ``uint32 version, uint32 n_names, uint32 offsets[8]`` (40 bytes), then a
   run of NUL-terminated names (class names + property names), then the
   regions delimited by those offsets. Versions 2 and 3 leave the last two
   offsets 0; version 4 (newer MATLAB, 2025) sets them to one more region at
   the end of the blob, which nothing here reads. The regions read:

   - region 1 (``offsets[0]:offsets[1]``), class table, 4×uint32 per class;
     entry 1 is the class-name index into the names list.
   - region 2 (``offsets[1]:offsets[2]``), "type 1" property maps.
   - region 3 (``offsets[2]:offsets[3]``), object table, 6×uint32 per object:
     ``[class_idx, _, _, type1_idx, type2_idx, object_id]``.
   - region 4 (``offsets[3]:offsets[4]``), "type 2" property maps.

   A property map is ``uint32 n_props`` followed by ``n_props`` triples
   ``[name_idx, kind, value]``, padded to 8-byte alignment. ``kind`` 0 → the
   value is itself a name (a string), 1 → an index into the value cells
   (``cells[value + 2]``), 2 → a logical.

Scope and honesty
-----------------
This is a **reverse-engineered, undocumented** format. It is written to fail
loud on anything it does not recognise rather than return a plausible-looking
wrong number: a wrong pixel size or τ would silently rescale every diffusion
coefficient downstream. Validated on the u-track v7 ``movieData.mat`` files in
this project (MATLAB R2014b-era, MCOS version 2) and on a newer-MATLAB file
writing version 3 (see ``_VALIDATED_VERSIONS`` for exactly what was checked).

The gate is on the LAYOUT, not just the version stamp: ``_check_layout`` verifies
every structural invariant the parsing relies on, offsets monotonic and inside
the blob, class/object tables a whole number of rows, name count as declared,
and names the invariant that failed. A version number is only a proxy for "is the
layout what I expect"; checking the layout directly is both more permissive for a
benign version bump and stricter for a genuinely different writer.

Only what the analysis spine needs is decoded: property *values*. Nested object
references (a property whose value is another object) are returned as
``ObjectRef``, not followed automatically: the caller decides.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["McosError", "McosUnvalidatedVersion", "ObjectRef", "McosObject", "read_mcos_objects"]

_HEADER_SIZE = 128
#: The metadata blob's header: version, number of names, eight region offsets.
_HEAD_BYTES = 40
#: MCOS layout versions this decoder has been validated against.
#:
#: 2, MATLAB R2014b-era, the reference project.
#: 3, written by newer MATLAB. Validated on
#:     ``D:/data/project/condition/tif/series001/movieData.mat`` by
#:     decoding it and checking the result: the name table parses to genuine
#:     MovieData property names, and the values are self-consistent and match the
#:     same microscope's other data (pixel 106.000003 nm, NA 1.49, search radius
#:     5 px, minTrackLen 5, gapWindow 1, mergeSplit 0, probDim 2; tau 20.3 ms,
#:     1501 frames, 176x176 px for that acquisition). ``psfSigma`` and the
#:     emission wavelength are ABSENT in that file, so the theoretical sigma there
#:     falls back to the pixelation basis, visible in ``sigma_basis``, not silent.
#: 4, written by a newer MATLAB in March 2025 (all 80 files of a lab folder; asked
#:     20 Oct 2026, "MCOS version 4 has not been validated"). The header holds EIGHT
#:     region offsets, as it always did: versions 2 and 3 leave the last two 0 (all
#:     1 188 version-2 and 160 version-3 files on E:, F: and G: checked), version 4
#:     sets them to one more region at the end of the blob (the eighth offset is the
#:     blob's end in every file), so the names start at byte 40. Read from byte 32
#:     they began with two of those offsets' bytes as names, and the count failed.
#:     Regions 1-5 are laid out as in version 2, byte for byte on the same
#:     MovieData. Validated by decoding ``19.03.2025 SMT/1903/exp/free`` series001:
#:     values checked in ``tests/test_utrack_meta.py`` (TestVersion4).
#:
#: THE VERSION NUMBER IS A PROXY; THE LAYOUT IS THE THING. A benign version bump
#: that changes nothing we read should not block a project, and a genuinely
#: different writer that happens to stamp a familiar version must not slip through.
#: So the version list is paired with :func:`_check_layout`, which verifies the
#: structural invariants directly. Widen this list only after decoding a real file
#: of that version and checking its values, and record what was checked here.
_VALIDATED_VERSIONS = (2, 3, 4)
#: bytes per row in the class table (4 x uint32) and the object table (6 x uint32)
_CLASS_ROW_BYTES = 16
_OBJECT_ROW_BYTES = 24
# An object reference is stored as a uint32 vector whose first entry is this
# marker, followed by [ndims, dim..., class_idx, object_id].
_OBJECT_MARKER = 0xDD000000


class McosError(RuntimeError):
    """Raised when a .mat file's MCOS payload cannot be decoded."""


class McosUnvalidatedVersion(McosError):
    """The file's MCOS version is not in ``_VALIDATED_VERSIONS``; nothing else is known to be wrong. A caller that
    can check the decoded values against something independent may read it with ``unvalidated=True``
    (`utrack_meta.read_movie_metadata` checks the frame count and image size against the movie's own stack)."""

    def __init__(self, message: str, version: int):
        super().__init__(message)
        self.version = version


def _check_layout(meta: bytes, version: int, n_names: int, offsets: list[int],
                  path, *, unvalidated: bool = False) -> list[str]:
    """Verify the metadata blob is laid out the way this decoder reads it.

    The version number alone is a weak gate: it blocks benign bumps that change
    nothing we read, and it would wave through a genuinely different writer that
    happened to stamp a familiar number. So the version is checked against the
    VALIDATED list *and* the structure is checked directly, every invariant the
    parsing below relies on:

    - the region offsets in use (six, or eight in version 4) are non-decreasing and lie
      inside the blob;
    - the names region starts after the 40-byte header;
    - the class table is a whole number of 4-uint32 rows and the object table a
      whole number of 6-uint32 rows (the two reshapes that would otherwise raise a
      numpy error far from the cause, or worse, silently succeed on a wrong shape);
    - the name table yields exactly the declared number of names.

    Returns the parsed names. Raises :class:`McosError` naming which invariant
    failed, because "layout not understood" without the specific is not actionable.
    """
    if version not in _VALIDATED_VERSIONS and not unvalidated:
        raise McosUnvalidatedVersion(
            f"{path}: MCOS version {version} has not been validated "
            f"(validated: {_VALIDATED_VERSIONS}). SMTracker does not guess: a wrong pixel "
            "size or frame interval rescales every diffusion coefficient. "
            "To add a version: decode one real file of it, check the values against "
            "a known acquisition, and record what was checked at _VALIDATED_VERSIONS.",
            version)
    # the last two offsets are 0 (versions 2, 3) or one more region after the sixth (version 4)
    used = offsets if any(offsets[6:]) else offsets[:6]
    if used != sorted(used):
        raise McosError(f"{path}: MCOS region offsets are not monotonic: {offsets}")
    if not (_HEAD_BYTES <= used[0] <= len(meta)) or used[-1] > len(meta):
        raise McosError(
            f"{path}: MCOS region offsets {offsets} fall outside the {len(meta)}-byte "
            "metadata blob")
    for label, lo, hi, row in (("class", offsets[0], offsets[1], _CLASS_ROW_BYTES),
                               ("object", offsets[2], offsets[3], _OBJECT_ROW_BYTES)):
        if (hi - lo) % row:
            raise McosError(
                f"{path}: MCOS {label} table is {hi - lo} bytes, not a whole number "
                f"of {row}-byte rows: the layout is not the one this decoder reads.")

    names = [b.decode("latin1") for b in meta[_HEAD_BYTES:offsets[0]].split(b"\x00") if b]
    if len(names) != n_names:
        raise McosError(
            f"{path}: MCOS name table declares {n_names} names but {len(names)} "
            "were parsed; layout not understood."
        )
    return names


@dataclass(frozen=True)
class ObjectRef:
    """A property value that points at another MCOS object."""

    class_index: int
    object_ids: tuple[int, ...]


@dataclass
class McosObject:
    """One decoded MATLAB class object."""

    class_name: str
    object_id: int
    props: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default=None):
        return self.props.get(name, default)


def read_mcos_objects(path: str | Path, *, unvalidated: bool = False) -> list[McosObject]:
    """Decode every MATLAB class object stored in ``path``.

    Returns them in file order; ``object_id`` matches the ids used by
    ``ObjectRef`` so callers can resolve references themselves.

    ``unvalidated=True`` reads a version outside ``_VALIDATED_VERSIONS`` with every layout
    check still applied; only a caller that then checks the values may ask for it.

    Raises
    ------
    McosError
        If the file holds no MCOS payload, or its layout is not one this
        decoder was validated against (naming the invariant that failed);
        :class:`McosUnvalidatedVersion` when only the version is unknown.
    """
    path = Path(path)
    if not path.exists():
        raise McosError(f"file not found: {path}")

    blob_cells = _read_filewrapper_cells(path)
    meta = np.asarray(blob_cells[0]).tobytes()
    if len(meta) < _HEAD_BYTES:
        raise McosError(f"{path}: MCOS metadata blob is too short ({len(meta)} bytes)")

    head = np.frombuffer(meta[:_HEAD_BYTES], dtype="<u4")
    version, n_names = int(head[0]), int(head[1])
    offsets = [int(x) for x in head[2:10]]
    names = _check_layout(meta, version, n_names, offsets, path, unvalidated=unvalidated)

    classes = _u32(meta, offsets[0], offsets[1]).reshape(-1, 4)
    objects = _u32(meta, offsets[2], offsets[3]).reshape(-1, 6)
    maps_type1 = _parse_property_maps(_u32(meta, offsets[1], offsets[2]))
    maps_type2 = _parse_property_maps(_u32(meta, offsets[3], offsets[4]))

    out: list[McosObject] = []
    for row in objects[1:]:  # row 0 is the null object
        class_idx, _, _, t1, t2, object_id = (int(v) for v in row)
        class_name = _lookup(names, int(classes[class_idx][1]), "class name", path)
        props_raw = _select_map(maps_type2, t2) if t2 else _select_map(maps_type1, t1)
        props = {
            _lookup(names, name_idx, "property name", path): _value(
                kind, value, blob_cells, names, path
            )
            for name_idx, kind, value in props_raw
        }
        out.append(McosObject(class_name=class_name, object_id=object_id, props=props))
    return out


# scipy names this field, not the file, so it is LOOKED UP rather than hardcoded.
PAYLOAD_FIELDS = ("_ObjectMetadata", "arr")


def _payload_field(wrapper, path: Path) -> str:
    """Which field of scipy's MATLAB-opaque wrapper holds the FileWrapper__ cells.

    scipy builds the wrapper with its own `OPAQUE_DTYPE`, which on 1.15.3 is
    ('s0', 's1', 's2', 'arr'). Asking for "_ObjectMetadata" therefore raised
    `ValueError: no field of name _ObjectMetadata` on EVERY v7 .mat holding class
    objects -- which is every u-track `movieData.mat`. It was reported as one
    project that would not open, and it was a property of the installed scipy
    rather than of that project's files: nothing about the data was involved.

    A wrapper carrying no known field is REFUSED BY NAME. Taking the last field
    positionally would decode whatever happened to sit there and report it as
    acquisition metadata, which is the silent-wrong-number failure this module
    exists to avoid.
    """
    fields = np.asarray(wrapper).dtype.names or ()
    for key in PAYLOAD_FIELDS:
        if key in fields:
            return key
    raise McosError(
        f"{path}: the MCOS wrapper carries none of the known payload fields "
        f"{PAYLOAD_FIELDS}; scipy gave {list(fields)}"
    )


def _read_filewrapper_cells(path: Path) -> np.ndarray:
    """Pull the FileWrapper__ cell array out of ``__function_workspace__``."""
    try:
        from scipy.io.matlab._mio5 import MatFile5Reader
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise McosError("scipy is required to decode MATLAB class objects") from exc

    with open(path, "rb") as fh:
        reader = MatFile5Reader(fh)
        reader.initialize_read()
        try:
            workspace = reader.get_variables(["__function_workspace__"])
        except Exception as exc:  # noqa: BLE001 - re-raised as our own error
            raise McosError(f"{path}: cannot read the MAT-file subsystem") from exc
        raw = workspace.get("__function_workspace__")
    if raw is None:
        raise McosError(
            f"{path}: no __function_workspace__: the file stores no MATLAB "
            "class objects (or is v7.3, which this decoder does not handle)."
        )

    # The blob is a headerless mini MAT-file: skip its 8-byte lead-in and
    # prepend a synthetic 128-byte header (little-endian, version 0x0100).
    body = raw.tobytes()[8:]
    header = b"MATLAB 5.0 MAT-file, subsystem".ljust(116, b" ") + b"\x00" * 8 + b"\x00\x01IM"
    stream = io.BytesIO(header + body)
    sub = MatFile5Reader(stream, struct_as_record=False, squeeze_me=True)
    sub.initialize_read()
    stream.seek(_HEADER_SIZE)
    var_header, _ = sub.read_var_header()
    payload = sub.read_var_array(var_header, process=True)

    wrapper = getattr(payload, "MCOS", None)
    if wrapper is None:
        raise McosError(f"{path}: subsystem holds no 'MCOS' field")
    cells = np.asarray(np.asarray(wrapper)[_payload_field(wrapper, path)]).ravel()[0]
    cells = np.asarray(cells).ravel()
    if cells.size < 3:
        raise McosError(f"{path}: FileWrapper__ cell array is too small ({cells.size})")
    return cells


def _u32(buf: bytes, start: int, stop: int) -> np.ndarray:
    return np.frombuffer(buf[start:stop], dtype="<u4")


def _parse_property_maps(seg: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """Split a region into per-object property maps (8-byte aligned records)."""
    maps: list[list[tuple[int, int, int]]] = []
    i = 0
    while i < len(seg):
        n = int(seg[i])
        i += 1
        if i + 3 * n > len(seg):
            raise McosError(
                f"MCOS property map claims {n} properties but the region ends first"
            )
        maps.append([tuple(int(v) for v in seg[i + 3 * k : i + 3 * k + 3]) for k in range(n)])
        i += 3 * n
        if i % 2:  # records are padded to an 8-byte boundary
            i += 1
    return maps


def _select_map(maps: list, index: int) -> list[tuple[int, int, int]]:
    if index >= len(maps):
        raise McosError(f"MCOS property-map index {index} out of range ({len(maps)})")
    return maps[index]


def _lookup(names: list[str], index: int, what: str, path: Path) -> str:
    if not 1 <= index <= len(names):
        raise McosError(f"{path}: {what} index {index} outside the name table")
    return names[index - 1]


def _value(kind: int, value: int, cells: np.ndarray, names: list[str], path: Path):
    """Resolve one property value: a name, a cell payload, or a logical."""
    if kind == 0:
        return _lookup(names, value, "string value", path)
    if kind == 2:
        return bool(value)
    if kind != 1:
        raise McosError(f"{path}: unknown MCOS property kind {kind}")
    idx = value + 2
    if idx >= len(cells):
        raise McosError(f"{path}: property value cell {idx} out of range ({len(cells)})")
    return _as_python(cells[idx])


def _as_python(cell):
    """Unwrap a value cell: scalars, empties, strings and object references."""
    if isinstance(cell, str):
        return cell
    arr = np.asarray(cell)
    if arr.dtype == np.uint32 and arr.size >= 6 and int(arr.ravel()[0]) == _OBJECT_MARKER:
        flat = arr.ravel()
        n_dims = int(flat[1])
        n_objs = int(np.prod([int(v) for v in flat[2 : 2 + n_dims]])) if n_dims else 1
        class_index = int(flat[2 + n_dims])
        ids = tuple(int(v) for v in flat[3 + n_dims : 3 + n_dims + n_objs])
        return ObjectRef(class_index=class_index, object_ids=ids)
    if arr.dtype == object and arr.size and all(
        isinstance(x, np.ndarray) and x.dtype == np.uint32 for x in arr.ravel()
    ):
        refs = [_as_python(x) for x in arr.ravel()]
        if all(isinstance(r, ObjectRef) for r in refs):
            return refs
    if arr.shape == ():
        return arr.item()
    if arr.size == 0:
        return None
    if arr.size == 1:
        return arr.ravel()[0].item() if arr.dtype.kind in "fiub" else arr.ravel()[0]
    return arr
