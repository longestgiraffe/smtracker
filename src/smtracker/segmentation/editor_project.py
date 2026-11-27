"""Finding fields to edit, and writing one back. Pure: no Qt, no drawing.

The window calls these two functions and adds nothing to them, so "does saving keep the
automatic meshes" and "does scanning find every condition" are answered by headless
tests rather than by clicking.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

__all__ = ["CURATED_SUFFIX", "AUTO_SUFFIX", "scan_fields", "NOT_SCANNED", "save_field", "DEFAULT_CHANNELS",
           "MAX_FLUOR", "FLUOR_VIEWS", "normalise_channels", "channels_problem",
           "channel_views", "fields_of", "field_condition", "field_label"]

#: Written beside a ``.mat`` the moment a human saves it, so training and QC can tell a
#: curated field from automatic output. The browser editor's marker, kept: the two
#: editors must not disagree about what "curated" looks like on disk.
CURATED_SUFFIX = ".curated.json"

#: The sibling folder the pre-edit outlines are copied into.
AUTO_SUFFIX = "_auto"


#: The folder outlines are drawn on, and the fluorescence folders shown beside it. A lab
#: that names them ``BF`` / ``gfp`` / ``rfp`` sets that once in the editor's Channels dialog;
#: it is remembered per user (`ui.json`, key ``editor_channels``).
DEFAULT_CHANNELS = {"bright": "phase", "fluor": ["sum"]}
#: Fluorescence folders the editor offers a view and an overlay for.
MAX_FLUOR = 3
#: Each fluorescence channel's two view keys, in order. Channel 1 keeps the classic
#: editor's ``sum`` / ``overlay``, so its shortcuts, tests and saved habits are unchanged.
FLUOR_VIEWS = (("sum", "overlay"), ("fluor2", "overlay2"), ("fluor3", "overlay3"))


def _clean(name) -> str:
    return str(name or "").strip().strip("/\\").strip()


def channels_problem(bright, fluors) -> str:
    """Why a channel set cannot be used, or "" when it can. Folder NAMES only: a path is
    refused, since the same names are looked up beside every condition's images."""
    bright = _clean(bright)
    names = [_clean(f) for f in fluors if _clean(f)]
    if not bright:
        return "Name the brightfield folder (the images the outlines are drawn on)."
    for name in [bright, *names]:
        if any(c in name for c in "/\\:"):
            return f"'{name}' is a path; give the folder's name only."
    if len(names) > MAX_FLUOR:
        return f"At most {MAX_FLUOR} fluorescence folders."
    lowered = [n.lower() for n in [bright, *names]]
    if len(set(lowered)) != len(lowered):
        return "Each folder can be named once."
    return ""


def normalise_channels(raw) -> dict:
    """A saved or typed channel set as ``{"bright": name, "fluor": [names]}``; anything
    unusable (absent, malformed, refused by `channels_problem`) is the default."""
    if not isinstance(raw, dict):
        return {"bright": DEFAULT_CHANNELS["bright"], "fluor": list(DEFAULT_CHANNELS["fluor"])}
    bright = _clean(raw.get("bright"))
    fluors = [_clean(f) for f in (raw.get("fluor") or []) if _clean(f)]
    if channels_problem(bright, fluors):
        return {"bright": DEFAULT_CHANNELS["bright"], "fluor": list(DEFAULT_CHANNELS["fluor"])}
    return {"bright": bright, "fluor": fluors}


#: Folder names that are acronyms, shown upper-case on their buttons (``gfp`` -> GFP).
ACRONYMS = {"bf", "dic", "gfp", "egfp", "sfgfp", "rfp", "yfp", "cfp", "bfp", "mng", "tl"}


def _title(name: str) -> str:
    """A folder name as a button: a known acronym upper-case, else its first letter."""
    if name.lower() in ACRONYMS:
        return name.upper()
    return name[:1].upper() + name[1:]


def channel_views(channels: dict) -> list[tuple]:
    """The view pills a channel set offers, in order (and keys 1, 2, 3 ... follow it):
    ``[(view key, label, fluorescence index or None)]`` -- the brightfield, then each
    fluorescence folder and its overlay on the brightfield."""
    fluors = channels["fluor"]
    views = [("phase", _title(channels["bright"]), None)]
    for k, name in enumerate(fluors):
        single, overlay = FLUOR_VIEWS[k]
        views.append((single, _title(name), k))
        views.append((overlay, "Overlay" if len(fluors) == 1 else f"{_title(name)} overlay", k))
    return views


def fields_of(fields: list, condition, root=None) -> list[int]:
    """Indexes of the fields that belong to ``condition`` (a condition folder's own name);
    every field when ``condition`` is None. A field scanned straight from the folder that
    was opened (label ``.``) belongs to that folder's name."""
    if condition is None:
        return list(range(len(fields)))
    return [i for i, field in enumerate(fields) if field_condition(field, root) == condition]


def field_condition(field: dict, root=None) -> str:
    """The condition a field belongs to: its condition folder's own name."""
    label = str(field.get("condition", "")).replace("\\", "/")
    if label in ("", ".") and root is not None:
        return Path(root).name
    return label.split("/")[-1]


def field_label(field: dict) -> str:
    """A field as the list shows it once its condition is picked: the image's own name,
    marked when it has no outlines yet."""
    stem = Path(field["phase"]).stem
    return stem if field.get("has_mat") else f"{stem}   (no outlines yet)"


#: Folders the scan never enters: what SMTracker and the trackers write INSIDE a condition, never a condition and
#: never a field. Compared lower-case. u-track's per-frame images under ``tif/`` were 439 000 of the 443 000 files in
#: a user's analysis folder, and walking them froze the outlines window for 20 s (15 Oct 2026).
NOT_SCANNED = frozenset({"tif", "trackingpackage", "cell_meshes", "cell_meshes_auto", "cell_meshes_preview",
                         "cell_masks", "bleaching", "trackmate", "track_tables", "backups", ".smtracker_warm",
                         "__pycache__"})


def scan_fields(root, *, bright: str = "phase", on_progress=None) -> list[dict]:
    """Every brightfield image under ``root`` (a folder named ``bright``, any case), with
    the ``.mat`` that belongs to it.

    Walks the WHOLE tree rather than one condition, so a project laid out as
    ``<root>/<condition>/phase/`` opens in one session instead of one per condition.
    Each name carries its condition, so the list is unambiguous when two conditions hold
    a ``series001``. A folder with no ``phase/`` but phase snapshots beside its movies
    (nothing filed) gives those (`channels.loose_snapshots`, the outlines window's rule).
    Never enters `NOT_SCANNED` folders, nor the ``bright`` folder below its own images.

    ``on_progress(done, total, name)``, when given, is called as the walk enters each of
    ``root``'s own folders (``name``) and once at the end (``done == total``), so a window
    can show a bar while a large tree is read; it may be called from a worker thread.
    """
    import os

    from .channels import loose_snapshots, movie_names_beside, snapshot_field
    from ..rawmovie import same_movie
    from ..organise import ACQUISITION_SUFFIXES

    root = Path(root)
    if not root.is_dir():
        return []
    from ..cellmesh import is_mesh_folder_name

    acquisitions = tuple(s.lower() for s in ACQUISITION_SUFFIXES)
    fields: list[dict] = []
    tops: list[str] = []
    entered = None
    for folder, subfolders, _files in _walk(root):
        subfolders[:] = [d for d in subfolders if not d.startswith((".", "_"))
                         and d.lower() not in NOT_SCANNED and not is_mesh_folder_name(d)]
        phase = next((folder / d for d in subfolders if d.lower() == bright.lower()), None)
        if phase is not None:
            subfolders[:] = [d for d in subfolders if d.lower() != bright.lower()]
        if folder == root:
            tops = list(subfolders)
        elif on_progress is not None and tops:
            top = folder.relative_to(root).parts[0]
            if top != entered:
                entered = top
                on_progress(tops.index(top) if top in tops else 0, len(tops), top)
        if phase is not None:
            try:
                images = sorted(p for p in phase.iterdir()
                                if p.suffix.lower() in (".tif", ".tiff"))
            except OSError:
                continue
        elif bright.lower() == "phase" and any(
                os.path.splitext(f)[1].lower() in acquisitions for f in _files):
            images = loose_snapshots(folder)
        else:
            continue
        from ..cellmesh import mesh_folder, outline_file

        meshes = mesh_folder(folder)
        label = folder.relative_to(root).as_posix()
        others = None
        for image in images:
            # THE MOVIE'S NAME, not the file's stem: ``series001.nd2.tif`` is field
            # ``series001``, whose outlines the analysis reads (`channels.snapshot_field`).
            stem = snapshot_field(image)
            snapshot = stem
            if stem.isdigit():
                # A SNAPSHOT NAMED BY ITS NUMBER ONLY (one user's MetaMorph ``001.tif``, 23 Oct 2026) takes the
                # name its movie goes by, so its outlines are filed where the analysis reads them
                others = movie_names_beside(folder) if others is None else others
                stem = next((hit for names in others if (hit := same_movie(stem, names))), stem)
            # THE MOVIE'S OUTLINE FILE, under its name or another with its number (`cellmesh.outline_file`, 25 Oct
            # 2026): the segmenter may have filed it under the sum image's name, and editing a second file left the
            # hand-drawn outlines unread
            mat = outline_file(meshes, stem)
            if not mat.exists() and (meshes / (snapshot + ".mat")).exists():
                mat = meshes / (snapshot + ".mat")     # outlines already filed under the number: edit THAT file
            fields.append({
                "name": f"{label}/{stem}" if label != "." else stem,
                "condition": label,
                "phase": image,
                "mat": mat,
                "has_mat": mat.exists(),
            })
    if on_progress is not None:
        on_progress(len(tops), len(tops), "")
    return sorted(fields, key=lambda f: f["name"])


def _walk(root: Path):
    import os

    for folder, subfolders, files in os.walk(root):
        yield Path(folder), subfolders, files


#: Where the label masks go, a SIBLING of `cell_meshes/`, not inside it, so a glob for
#: `cell_meshes/*.mat` cannot pick one up. That glob has already been surprised twice:
#: by u-track's own `movieList.mat` and by macOS `._` sidecars.
MASK_FOLDER = "cell_masks"


def field_image_shape(field) -> tuple | None:
    """``(height, width)`` for a field, from its image HEADER. ``None`` if unreadable.

    Save all is why this exists. The editor knows the shape of the field it is DISPLAYING
    (it has the pixels in hand) and nothing about the others, so a "Save all" would
    have written one mask and silently skipped the rest. Reading a TIFF header is a few
    hundred bytes and costs nothing next to the save itself.

    Returns None rather than raising: a missing or odd image must not stop the `.mat`
    being written, and `save_field` records the reason there is no mask.
    """
    path = field.get("phase") if hasattr(field, "get") else None
    if path is None:
        return None
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as handle:
            shape = handle.pages[0].shape
    except Exception:                                 # noqa: BLE001 - reported as None
        return None
    if len(shape) < 2:
        return None
    return (int(shape[-2]), int(shape[-1]))


def save_field(mat_path, document, *, image_shape=None):
    """Write one field's cells. Returns ``(path, backup_or_None)``.

    The automatic meshes are kept before a human first writes over them, into a
    sibling ``cell_meshes_auto/`` and only if absent, so repeated saves never touch the
    pristine copy and curation stays reversible. Re-taking it on the second save would
    preserve the human's first edit under the name "automatic", which is worse than not
    keeping one at all.

    An empty field is a result, not a refusal. Asked for: "sometimes i wanna be
    able to delte ALL meshes from a series file ... make it so that even an emtpy mat is
    what it is -> no cell mesh". Deleting every cell and saving now writes a file with an
    empty `cellList`, which is how this project already represents an unsegmented movie:
    `read_project_cell_meshes` passes `on_empty="empty"` by default, `assign_cells`
    records the movie under `.attrs["unsegmented_movies"]` and drops its tracks, and the
    QC N-per-group panel is where it becomes visible.

    The old refusal said the reader would reject such a file. That was true of the
    reader's non-default mode only, and stopped being the shipped behaviour when
    unsegmented movies became survivable: a refusal outliving its reason.

    The curated marker is still written, with `n_cells: 0`, which is what separates
    "a human looked and there are none" from "never segmented".
    """
    from ..cellmesh import CellMeshError, CellMesh, write_cell_meshes

    mat_path = Path(mat_path)
    meshes = [CellMesh(movie=mat_path.stem, cell_id=str(i + 1),
                       polygon=cell.outline, mesh=cell.ribs)
              for i, cell in enumerate(document.cells)]
    # RIBLESS OUTLINES STAY (the UI designer, 26 Nov 2026: "saving drops ribless outlines from the file
    # without asking"). The editor cannot show a contour with no ribs, but the CELL FILTER
    # reads only the polygon (`assign_cells`), so those outlines keep tracks in the analysis;
    # dropping them on save removed those cells' tracks from every later run, silently.
    # They are written back unchanged, numbered after the editable cells.
    kept = _ribless_outlines(mat_path)
    meshes += [CellMesh(movie=mat_path.stem, cell_id=str(len(meshes) + 1 + k),
                        polygon=polygon, mesh=None)
               for k, polygon in enumerate(kept)]
    backup = _backup_once(mat_path)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    # `allow_empty` is opt-in in the writer so a stray empty list cannot erase a
    # field; here it is exactly what the user asked for, so it is passed deliberately.
    write_cell_meshes(mat_path, meshes, overwrite=True, image_shape=image_shape,
                      allow_empty=True)

    # a mask beside the .mat, ASKED FOR, and the `.mat` stays primary.
    #
    # "meshes need to be the go to output in mat format", so this is an ADDITION rather
    # than a format change: the mesh is what every downstream stage reads, and the mask
    # is what a segmentation model can be trained on. A corrected mesh is the best
    # training label this data will ever have, because a human looked at it.
    #
    # WITHOUT AN IMAGE SHAPE THERE IS NO MASK, and that is recorded rather than raised:
    # the `.mat` is the thing the analysis needs and must not fail for want of a shape.
    mask_path = None
    if image_shape is not None:
        from .masks import labels_from_outlines

        mask_path = (mat_path.parent.parent / MASK_FOLDER / (mat_path.stem + ".tif"))
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        labels = labels_from_outlines(
            [*document.cells, *kept], image_shape,
            cell_ids=[int(m.cell_id) for m in meshes])
        import tifffile

        tifffile.imwrite(mask_path, labels)

    marker = mat_path.with_name(mat_path.name + CURATED_SUFFIX)
    marker.write_text(json.dumps({
        "curated": True,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "n_cells": len(meshes),
        "n_ribless_kept": len(kept),
        "mask": str(mask_path.name) if mask_path is not None else None,
        "mask_skipped": None if mask_path is not None else
            "no image shape was supplied, so the field size is unknown",
        "n_regenerated": sum(1 for c in document.cells if c.regenerated),
        "n_curved": sum(1 for c in document.cells if c.is_curved),
        "auto_backup": str(backup) if backup else None,
    }, indent=1), encoding="utf-8")
    return mat_path, backup


def _ribless_outlines(mat_path: Path) -> list:
    """The contour-only outlines already in ``mat_path`` (none when it does not exist)."""
    import numpy as np

    from ..cellmesh import read_cell_meshes

    if not mat_path.exists():
        return []
    return [np.asarray(m.polygon, dtype=float)
            for m in read_cell_meshes(mat_path, on_empty="empty") if m.mesh is None]


def _backup_once(mat_path: Path) -> Path | None:
    folder = mat_path.parent
    backup_dir = folder.parent / (folder.name + AUTO_SUFFIX)
    backup = backup_dir / mat_path.name
    if not backup.exists() and mat_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mat_path, backup)
        return backup
    return backup if backup.exists() else None
