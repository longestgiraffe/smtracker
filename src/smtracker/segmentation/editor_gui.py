"""The mesh editor's WINDOW. Qt, and nothing else.

THIN QT, the house rule: no numpy, no pandas, no scipy at module level, and no geometry
anywhere. This file routes events. Everything it shows was computed by
:mod:`smtracker.segmentation.editing` (the model) or drawn by
:mod:`smtracker.segmentation.editor_view` (the view), both covered headless, so a click
that picks the wrong cell fails in `test_segmentation_editor_view`, not by looking
subtly wrong on screen. Same split as `widgets.py` against `plots.py`, kept by an AST
test.

**Parity with the browser editor is the requirement, and it is a CHECKLIST rather than
an impression.** The first version of this window shipped as the default with no channel
switch, no brightness, contrast or alpha, no draw preview, no merge button, keyboard
shortcuts that never reached the handler, and an outline drawn with hooks past both
poles. It went out because "the tests pass" was mistaken for "it works". Every control
in `editor.html` is now enumerated in `PARITY` below and asserted present by
`test_segmentation_editor_gui`, so a missing feature fails a test instead of being
noticed in use.

What this has that the browser editor does not is curvature: a cell is a spine and a
width, so `B` adds a bend and dragging it curves the cell, and every other edit preserves
it. In the browser a cell is two poles and every edit rebuilds it straight.
"""
from __future__ import annotations

__all__ = ["MeshEditorWindow", "launch", "PARITY", "SHORTCUTS"]

#: EVERY control the browser editor has, as data, so "no function is missing" is a test
#: rather than a memory. Read out of `editor.html`: its buttons, its sliders, its keys.
#:
#: `applyAlign` is the one deliberate omission: it shifted meshes onto the fluorescence
#: channel by a per-field offset that was a HADA measurement, and that analysis is not
#: ported. Omitted by decision and named here, which is not the same as forgotten.
PARITY = {
    "open_folder": "pick a folder to scan for phase/ + cell_meshes/",
    "save_field": "write this field's outlines",
    "save_all": "write every field edited in this session",
    "zoom_in": "magnify",
    "zoom_out": "shrink",
    "zoom_fit": "fit the field to the window",
    "view_phase": "show the phase channel",
    "view_sum": "show the summed fluorescence channel",
    "view_overlay": "blend the two",
    "overlay_opacity": "how strongly the fluorescence is tinted onto the phase",
    "overlay_colour": "the tint the fluorescence takes in the overlay",
    "brightness": "display brightness, per cent",
    "contrast": "display contrast, per cent",
    "reset_display": "brightness and contrast back to 100 %",
    "stretch_mode": "percentile window: septa / imagej / outline",
    "width": "the width a drawn cell gets, and the selected cell's width",
    "tool_select": "select and edit",
    "tool_draw": "draw a new cell, with a live preview",
    "tool_split": "split at the septum",
    "merge": "merge the selected cell with its nearest neighbour",
    "delete": "delete the selected cell",
    "undo": "step back",
    "redo": "step forward",
    "prev_field": "previous field",
    "next_field": "next field",
    "cell_list": "the cells in this field",
    "cell_stats": "number, length, width and rib count of the selected cell",
    "add_bend": "add a bend to the selected cell (NOT in the browser editor)",
    "remove_bend": "straighten the selected cell (NOT in the browser editor)",
}

#: Keyboard, carried over from the browser editor VERBATIM where it had one: these are
#: already in the experimenter's fingers and changing them would be a change nobody
#: asked for. `A` was the HADA alignment toggle and is free; `B`/`N` are the new bends.
SHORTCUTS = [
    ("S", "select and edit", "tool_select"),
    ("D", "draw a cell", "tool_draw"),
    ("X", "split at the septum", "tool_split"),
    ("M", "merge with the nearest neighbour", "merge"),
    ("B", "add a bend", "add_bend"),
    ("N", "straighten", "remove_bend"),
    ("Del", "delete the selected cell", "delete"),
    ("Ctrl+Z", "undo", "undo"),
    ("Ctrl+Y", "redo", "redo"),
    ("Ctrl+S", "save this field", "save_field"),
    ("1 / 2 / 3", "phase / sum / overlay", "view_phase"),
    ("C", "cycle the percentile stretch", "stretch_mode"),
    ("R", "reset brightness and contrast", "reset_display"),
    ("[ / ]", "nudge contrast by 15 %", "contrast"),
    ("+ / -", "zoom", "zoom_in"),
    ("F", "fit to window", "zoom_fit"),
    ("← / →", "previous / next field", "prev_field"),
    ("wheel", "previous / next field", "next_field"),
    ("Ctrl+wheel", "zoom", "zoom_in"),
    ("Esc", "cancel a draw, or deselect", "tool_select"),
]

TOOLS = ("select", "draw", "split")


def _load_channels(phase_path, bright: str = "phase", fluors=("sum",)):
    """The brightfield image and, for each fluorescence folder, its partner beside it or
    None: ``(image, [partner per folder])``.

    A partner is the same image name in a sibling folder named after the channel (any
    case), or the name with ``_<bright>`` swapped for ``_<channel>`` there, ``.tif`` or
    ``.tiff``. One unreadable partner is None; the others still load."""
    from pathlib import Path

    import tifffile

    phase_path = Path(phase_path)
    image = tifffile.imread(phase_path)
    # BOTH LAYOUTS, because the brightfield image sits in EITHER place (ruled 18 Oct
    # 2026, of the classical segmenter: "they can be either in phase subfolder or in
    # main folder"). `<cond>/phase/series001.tif` puts the channel folders one level UP;
    # a loose snapshot at `<cond>/series001.tif` puts them BESIDE it. Looking only one
    # level up read the PROJECT folder for a loose snapshot -- whose subdirectories are
    # the other conditions, never `sum` -- so the overlay was silently dead on exactly
    # the folders that keep their snapshots loose, and worked on the ones with `phase/`.
    # `channels.resolve_channels` already knew both layouts; this helper does its own
    # discovery, which is this project's own lesson about one fix not reaching the next.
    # NEAREST FIRST: a folder beside the image belongs to THIS condition, while one a
    # level up may belong to a sibling if a project ever nests differently.
    siblings = {}
    for folder in (phase_path.parent, phase_path.parent.parent):
        try:
            for d in folder.iterdir():
                if d.is_dir():
                    siblings.setdefault(d.name.lower(), d)
        except OSError:
            continue
    partners = []
    for fluor in fluors:
        target = siblings.get(str(fluor).lower())
        found = None
        if target is not None and target != phase_path.parent:
            stems = [phase_path.stem]
            swapped = phase_path.stem.replace(f"_{phase_path.parent.name}", f"_{fluor}")
            swapped = swapped.replace(f"_{bright}", f"_{fluor}")
            if swapped not in stems:
                stems.append(swapped)
            # THE SAME MOVIE UNDER ANOTHER NAME (23 Oct 2026): one user's ``phase/001.tif`` has its partner in
            # ``sum/series001_YFPLaser.tif`` -- the one image there with that series number (`rawmovie.same_movie`)
            from ..rawmovie import same_movie
            from .channels import image_stems, snapshot_field

            hit = same_movie(snapshot_field(phase_path), image_stems(target))
            if hit is not None and hit not in stems:
                stems.append(hit)
            for stem in stems:
                for suffix in dict.fromkeys((phase_path.suffix, ".tif", ".tiff")):
                    probe = target / f"{stem}{suffix}"
                    if probe.exists():
                        try:
                            found = tifffile.imread(probe)
                        except (OSError, ValueError):
                            found = None
                        break
                if found is not None:
                    break
        partners.append(found)
    return image, partners


def _load_images(phase_path):
    """The phase image and its `sum/` partner if there is one."""
    image, partners = _load_channels(phase_path, "phase", ("sum",))
    return image, partners[0]


def _load_field_channels(mat_path, phase_path, bright: str = "phase", fluors=("sum",)):
    """Read one field into a `Document`, plus its brightfield image and one partner per
    fluorescence folder: ``(document, image, partners, skipped)``."""
    from ..cellmesh import read_cell_meshes
    from .editing import Document, cell_from_ribs

    cells, skipped = [], []
    if mat_path is not None and mat_path.exists():
        for mesh in read_cell_meshes(mat_path, on_empty="empty"):
            if mesh.mesh is None:
                # A contour with no ribs has no centreline, so there is nothing to drag.
                # NAMED rather than silently dropped: a cell vanishing on open reads as
                # data loss, and this is the ribless case `cellmesh` already reports.
                skipped.append(mesh.cell_id)
                continue
            cells.append(cell_from_ribs(mesh.mesh, mesh.polygon, mesh.cell_id))
    image, partners = ((_load_channels(phase_path, bright, fluors)) if phase_path is not None
                       else (None, [None] * len(tuple(fluors))))
    return Document(cells=cells), image, partners, skipped


def _load_field(mat_path, phase_path):
    """Read one field into a `Document`, plus its images (phase and `sum/`)."""
    document, image, partners, skipped = _load_field_channels(mat_path, phase_path)
    return document, image, partners[0], skipped


#: Windows this module has opened, and the reason it must hold them.
#:
#: **A TOP-LEVEL PySide6 WINDOW WITH NO PYTHON REFERENCE IS DESTROYED AT ONCE.** The
#: moment `launch`'s caller discards the return value its refcount hits zero, CPython
#: deletes the wrapper, and the C++ widget goes with it, taking any QThread it owns.
#: MEASURED on the real route (`smtracker` -> Segmentation): "QThread: Destroyed while
#: thread '' is still running", then a **segfault, exit 139**, reported from a terminal.
#: Qt does NOT own a parentless widget on Python's behalf, and being visible does not
#: save it.
#:
#: THE DOOR OWNS WHAT IT OPENS, rather than every caller having to remember: the launch
#: sites in `app.py` all discarded, and so would the next one. Dropped again in
#: `closeEvent`, so a session that opens several windows does not accumulate them.
_OPEN_WINDOWS: list = []


def _keep(window):
    """Hold `window` until it closes. See `_OPEN_WINDOWS`."""
    if window not in _OPEN_WINDOWS:
        _OPEN_WINDOWS.append(window)
    return window


def _release(window) -> None:
    """Drop `window` on close, by IDENTITY: a QWidget's `==` can be overridden."""
    for i, held in enumerate(_OPEN_WINDOWS):
        if held is window:
            del _OPEN_WINDOWS[i]
            return


def launch(root=None, *, show: bool = True):
    """Open the editor. Returns the window.

    A `QApplication` is made if none exists: constructing a widget before one EXISTS
    aborts the process. Qt calls `abort()`, so there is no exception and nothing to
    catch: `smtracker` with no arguments once died at exit 127 having printed nothing.
    """
    from qtpy.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ..appicon import apply as _apply_icon

    _apply_icon()
    # `_build_window_class()`, NOT the bare name `MeshEditorWindow`: a module-level
    # `__getattr__` serves ATTRIBUTE ACCESS on the module and NOT a global lookup inside
    # a function, and the bare name raised NameError for every real user while every
    # test passed, because the tests reach it as `editor_gui.MeshEditorWindow`.
    window = _keep(_build_window_class()(root))
    if show:
        window.show()
    return window


_CORE = None


def _core_class():
    """`EditorCore`: every handler of the editor, and NO construction.

    SHARED by the classic window below and the workspace-style editor page
    (`ui.outlines.editor_page`), so every shortcut, click and drag is ONE piece of code
    whichever window it runs in. A subclass builds the widgets the handlers name
    (`_field_list`, `_cell_list`, `_stats`, `_tool_buttons`, `_view_buttons`,
    `_mode_box`, `_opacity`/`_bright`/`_contrast`, `_align_button`/`_align_note`,
    `_status`, `_hint`, `_canvas`/`_axes`/`_figure`, `_controls`), then calls
    `_init_state` before building and `_start` after. Built once and cached, so the
    class a subclass inherits from is the class every window uses."""
    global _CORE
    if _CORE is not None:
        return _CORE
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
                                QLineEdit,
                                QMessageBox, QSlider)

    def _slider(low, high, start, step=1):
        bar = QSlider(Qt.Horizontal)
        bar.setRange(int(low), int(high))
        bar.setValue(int(start))
        bar.setSingleStep(int(step))
        return bar


    class EditorCore:
        """The mesh editor's behaviour: see `_core_class`."""

        def _init_state(self, root) -> None:
            from pathlib import Path

            from .editing import DEFAULT_WIDTH_PX, Document

            self._root = Path(root) if root else None
            self._fields: list[dict] = []
            self._index = -1
            # EVERY field edited this session stays in memory, keyed by its .mat path.
            # Without it "Save ALL modified" cannot exist and switching fields would
            # throw away edits, the browser editor keeps the same cache for the same
            # two reasons.
            self._documents: dict[str, Document] = {}
            self._document = Document()
            self._phase = None
            self._sum = None
            #: The folder outlines are drawn on, the fluorescence folders shown beside it,
            #: their images for the field on screen (None where a field has none), and the
            #: views in key order (1, 2, 3, ...). The classic editor keeps phase + sum.
            self._bright_folder = "phase"
            self._fluor_folders: tuple = ("sum",)
            self._fluors: list = []
            self._view_order: tuple = ("phase", "sum", "overlay")
            #: The measured mesh-to-fluorescence offset, or None when there is none to
            #: apply. Set by `_check_alignment`, consumed by `_apply_offset`.
            self._offset = None
            self._selected: int | None = None
            self._tool = "select"
            self._mode = "imagej"          # what ImageJ shows on opening: the ask
            self._view = "phase"
            self._colour = "red"
            self._drag = None
            self._pan = None
            self._draw_from = None
            self._preview = None
            self._width = DEFAULT_WIDTH_PX
            self._artists: list = []
            self._background = None
            self._grab = None
            self._controls: dict[str, object] = {}

        def _start(self) -> None:
            # ON THE APPLICATION, NOT THE WINDOW. `keyPressEvent` on a QMainWindow never
            # saw a single key: focus lives in the canvas or a list, and those consume
            # it. Every shortcut was dead and the window looked like it had none. An
            # application-level filter is what the browser's `document.addEventListener`
            # actually is: including its guard, which is the next line.
            QApplication.instance().installEventFilter(self)

            self._set_tool("select")
            if self._root is not None:
                self._scan(self._root)
            else:
                self._say("Open a folder that holds phase/ and cell_meshes/.")

        def _release_focus(self) -> None:
            """Hand the keyboard back to the picture, so the shortcuts work again.

            Called from `editingFinished`, which Qt emits on Enter and on focus leaving
            the box. Without this a single click in a numeric box disabled 1/2/3 and
            every letter tool for good, because `eventFilter` declines keys while a box
            has focus -- correctly, so that typing a number is not a tool change.
            """
            canvas = getattr(self, "_canvas", None)
            if canvas is not None:
                canvas.setFocus()

        def _sync_value_labels(self) -> None:
            """Keep the readouts with their sliders when the value is set in code.

            `_reset_display` blocks the sliders' signals so the image is redrawn once
            rather than twice, and blocking them also skips the label update, which
            would leave "300 %" printed beside a slider sitting at 100.
            """
            for key, (label, suffix) in getattr(self, "_value_labels", {}).items():
                label.setText(f"{self._controls[key].value()}{suffix}")

        # ── plumbing ─────────────────────────────────────────────────────────
        def _say(self, message: str) -> None:
            self._status.showMessage(message)

        def _set_hint(self, text: str) -> None:
            self._hint.setText(text)

        def _complain(self, exc: Exception) -> None:
            """Every refusal reaches the user as a SENTENCE, never as a traceback.

            The model raises one exception type on purpose and its messages are written
            for the person making the edit: the turn-radius refusal names the two fixes
            that were measured to work.

            The status bar, not a modal dialog, for two reasons that agree. Parity:
            the browser editor reports a refused edit with `setStatus(msg, true)`, and a
            dialog would be a different tool that interrupts. And this project already
            records that a `QMessageBox` on a path a test can reach hangs the suite for
            its whole 900 s timeout, because headless there is nobody to click it, the
            first version of this method did exactly that and hung on the first refused
            merge. A refusal is information, not an event needing consent.

            The full text goes to the tooltip so a multi-line message (the turn-radius
            one is three sentences) is readable without a dialog.
            """
            first = str(exc).splitlines()[0]
            self._say(first)
            self._status.setToolTip(str(exc))
            self._set_hint("⚠ " + first[:70])

        def eventFilter(self, obj, event):             # noqa: N802 - Qt virtual
            from qtpy.QtCore import QEvent

            if event.type() != QEvent.KeyPress:
                return False
            # THE BROWSER'S OWN GUARD: `if (e.target.tagName === 'INPUT') return;`
            # Without it, typing a folder path would trigger the tool shortcuts inside
            # the letters being typed.
            #
            # BUT A GUARD THAT NEVER LIFTS IS A WINDOW WITHOUT SHORTCUTS. Reported
            # 18 Oct 2026: "after some time the buttons 1 2 3 and s and d shortcuts etc
            # dont work anymore". Nothing gave the focus back, so ONE click in a numeric
            # box or either combo (tint, stretch) killed every shortcut for the rest of
            # the session -- and "after some time" is exactly how that reads, since it
            # depends on happening to touch one.
            #
            # A NON-EDITABLE COMBO NEEDS NO TEXT. It uses a letter only to jump to an
            # item, which is worth far less here than `s`/`d`/`1`/`2`/`3`, so it is not
            # a reason to decline. A line edit and a spin box do need their characters,
            # and they hand the focus back when editing finishes (`_release_focus`).
            focus = QApplication.focusWidget()
            if isinstance(focus, (QLineEdit, QDoubleSpinBox)):
                return False
            if isinstance(focus, QComboBox) and focus.isEditable():
                return False
            if not self.isActiveWindow():
                return False
            return bool(self._handle_key(event))

        def _handle_key(self, event) -> bool:
            key, text = event.key(), event.text().lower()
            control = bool(event.modifiers() & Qt.ControlModifier)
            # BY KEY, NOT BY TEXT, for the Ctrl chords: on Windows Ctrl+S arrives with
            # the text "\x13", a control character, so `text == "s"` never matched and
            # Ctrl+S / Ctrl+Z / Ctrl+Y did nothing from the keyboard.
            if control and key == Qt.Key_S:
                self._save()
            elif control and key == Qt.Key_Z:
                self._undo()
            elif control and key == Qt.Key_Y:
                self._redo()
            elif control:
                return False
            elif text == "s":
                self._set_tool("select")
            elif text == "d":
                self._set_tool("draw")
            elif text == "x":
                self._set_tool("split")
            elif text == "m":
                self._merge()
            elif text == "b":
                self._add_bend()
            elif text == "n":
                self._remove_bend()
            elif text == "c":
                order = [self._mode_box.itemText(i)
                         for i in range(self._mode_box.count())]
                self._mode_box.setCurrentText(
                    order[(order.index(self._mode) + 1) % len(order)])
            elif text == "r":
                self._reset_display()
            elif text == "f":
                self._fit()
            elif len(text) == 1 and text.isdigit() and 1 <= int(text) <= len(self._view_order):
                self._set_view(self._view_order[int(text) - 1])
            elif text in ("[", "]"):
                # nudge contrast without leaving the canvas, as the browser does
                step = 15 if text == "]" else -15
                self._contrast.setValue(self._contrast.value() + step)
            elif text in ("+", "="):
                self._zoom(1 / 1.4)
            elif text == "-":
                self._zoom(1.4)
            elif key in (Qt.Key_Delete, Qt.Key_Backspace):
                self._delete()
            elif key == Qt.Key_Escape:
                self._escape()
            elif key == Qt.Key_Left:
                self._step_field(-1)
            elif key == Qt.Key_Right:
                self._step_field(+1)
            else:
                return False
            return True

        def _escape(self) -> None:
            if self._tool == "draw" and self._draw_from is not None:
                self._draw_from = None
                self._preview = None
                self._set_hint("click the first pole to start")
                self._refresh()
                return
            self._set_tool("select")
            self._selected = None
            self._refresh()

        # ── project ──────────────────────────────────────────────────────────
        def _pick_folder(self) -> None:
            from pathlib import Path

            chosen = QFileDialog.getExistingDirectory(self, "Folder with phase/")
            if chosen:
                self._scan(Path(chosen))

        def _scan(self, root) -> None:
            from .editor_project import scan_fields

            self._root = root
            self._fields = scan_fields(root, bright=self._bright_folder)
            self._documents.clear()
            self._field_list.clear()
            for field in self._fields:
                mark = "" if field["has_mat"] else "   (no outlines yet)"
                self._field_list.addItem(field["name"] + mark)
            if not self._fields:
                self._say(f"No {self._bright_folder}/ folder with images under {root}.")
                return
            self._say(f"{len(self._fields)} fields under {root}")
            self._field_list.setCurrentRow(0)

        def _step_field(self, delta: int) -> None:
            target = self._index + delta
            if 0 <= target < len(self._fields):
                self._field_list.setCurrentRow(target)

        def _open_index(self, index: int) -> None:
            if not 0 <= index < len(self._fields):
                return
            self._stash()
            field = self._fields[index]
            key = str(field["mat"])
            try:
                channels = (self._bright_folder, self._fluor_folders)
                if key in self._documents:
                    # EDITS SURVIVE A FIELD SWITCH. Re-reading would discard them
                    # silently, and the arrow keys make switching a one-key action.
                    self._document = self._documents[key]
                    _doc, self._phase, self._fluors, skipped = _load_field_channels(
                        None, field["phase"], *channels)
                else:
                    (self._document, self._phase,
                     self._fluors, skipped) = _load_field_channels(
                        field["mat"], field["phase"], *channels)
                    self._documents[key] = self._document
                self._sum = self._fluors[0] if self._fluors else None
            except Exception as exc:                   # noqa: BLE001 - reported
                self._complain(exc)
                return
            self._index = index
            self._selected = None
            self._draw_from = None
            self._preview = None
            self._refresh(full=True, fit=True)
            note = (f": {len(skipped)} outline(s) without ribs cannot be edited and are "
                    f"kept as they are on save: " + ", ".join(skipped)) if skipped else ""
            present = [name for name, image in zip(self._fluor_folders, self._fluors)
                       if image is not None]
            channels = (f" ({', '.join(present)} channel"
                        f"{'s' if len(present) > 1 else ''} available)" if present else "")
            self._say(f"{field['name']}: {len(self._document.cells)} cells{channels}{note}")
            if self._view != "phase" and self._fluor_image(self._view) is None:
                self._view = "phase"
                for name, button in getattr(self, "_view_buttons", {}).items():
                    button.setChecked(name == "phase")
            self._check_alignment()

        def _check_alignment(self) -> None:
            """Do these outlines sit where the fluorescence is?

            The editor is where this matters most. The segmentation run checks it once,
            but this window is where a human DRAWS on phase, so a registration offset
            between the phase and fluorescence paths is introduced or inherited here, and
            it is silent everywhere downstream: every number still computes, the cell
            filter just keeps the wrong localisations.

            Reports only. Moving a mesh is the experimenter's decision, which is what the
            button is for.
            """
            self._offset = None
            if self._sum is None or not self._document.cells:
                self._align_button.setEnabled(False)
                self._align_note.setText("")
                return
            try:
                from .alignment import mesh_image_offset
                from .masks import labels_from_outlines

                frame = self._sum
                if getattr(frame, "ndim", 2) > 2:
                    frame = frame.reshape(-1, *frame.shape[-2:]).mean(axis=0)
                labels = labels_from_outlines(self._document.cells, frame.shape)
                got = mesh_image_offset(labels, frame)
            except Exception as exc:                   # noqa: BLE001 - reported
                self._align_note.setText(f"alignment not checked: {type(exc).__name__}")
                self._align_button.setEnabled(False)
                return
            self._align_note.setText(got["verdict"])
            if got["ok"] is False:
                self._offset = (got["dx"], got["dy"])
                self._align_button.setEnabled(True)
                self._align_button.setText(
                    f"Shift all by ({got['dx']:+d}, {got['dy']:+d}) px")
                self._say(f"Alignment: {got['verdict']}")
            else:
                self._align_button.setEnabled(False)
                self._align_button.setText("Shift to fluorescence")

        def _apply_offset(self) -> None:
            """Move every cell by the measured offset. ONE undo step.

            A GESTURE, so the whole field moves or none of it does: a half-applied
            shift would leave two registrations in one file, which is worse than the
            offset it was fixing.
            """
            if self._offset is None:
                return
            dx, dy = self._offset
            self._document.begin_gesture()
            try:
                for index in range(len(self._document.cells)):
                    self._document.move_cell(index, dx, dy)
            finally:
                self._document.end_gesture()
            self._say(f"Shifted {len(self._document.cells)} cell(s) by "
                      f"({dx:+d}, {dy:+d}) px. Ctrl+Z undoes it in one step.")
            self._refresh(full=True)
            self._check_alignment()

        def _stash(self) -> None:
            if 0 <= self._index < len(self._fields):
                self._documents[str(self._fields[self._index]["mat"])] = self._document

        # ── drawing ──────────────────────────────────────────────────────────
        def _display_image(self):
            from .editor_view import compose

            second = self._fluor_image(self._view)
            if second is None and self._view == "phase":
                second = self._sum
            if self._phase is None and second is None:
                return None
            view = ("phase" if self._view == "phase" else
                    "overlay" if self._view.startswith("overlay") else "sum")
            return compose(self._phase, second, view=view,
                           opacity=self._opacity.value() / 100.0, mode=self._mode,
                           brightness=self._bright.value(),
                           contrast=self._contrast.value(), colour=self._colour)

        def _redraw_image(self) -> None:
            self._refresh(full=True)

        def _refresh(self, *, full: bool = False, fit: bool = False,
                     lists: bool = True) -> None:
            """Redraw. BLITTED unless the image itself changed.

            This is why dragging felt laggy beside the browser. The first version
            redrew the WHOLE canvas on every mouse-move (the phase image included) and
            rebuilt both list widgets while it was at it. The browser never paid either
            cost: its image is a separate canvas layer and its cell list only changes
            when the cells do.

            Here the image is captured once as a background and only the cell artists
            are redrawn over it, which is what `playback.TrackPlayer` does to reach
            1530–2884 fps. `lists=False` during a drag stops two QListWidgets being
            rebuilt sixty times a second for a selection that has not changed.
            """
            from .editor_view import draw_document

            rebuild = full or self._background is None
            limits = None
            if rebuild and not fit:
                limits = (self._axes.get_xlim(), self._axes.get_ylim())
            if rebuild:
                self._axes.clear()
            else:
                for artist in self._artists:
                    artist.remove()
            self._artists = draw_document(
                self._axes, self._document,
                self._display_image() if rebuild else None,
                selected=self._selected, mode=self._mode, already_scaled=True)
            if rebuild and limits is not None and limits[0] != (0.0, 1.0):
                self._axes.set_xlim(*limits[0])
                self._axes.set_ylim(*limits[1])
            self._draw_preview()
            # An ANIMATED artist is left out of an ordinary draw, which is what lets the
            # background be captured WITHOUT the cells and restored under them after.
            for artist in self._artists:
                artist.set_animated(True)
            if rebuild:
                # The DRAW EVENT captures the background and puts the cells on top of
                # it; asking for the draw here and doing the rest there is what makes a
                # repaint Qt starts behave the same as one we start.
                self._background = None
                self._canvas.draw_idle()
            else:
                self._blit()
            if lists:
                self._refresh_cell_list()
                self._refresh_stats()

        def _on_draw(self, event=None) -> None:
            """Re-capture the background and re-draw the cells after ANY canvas draw.

            This is why the first field opened with no outlines. Blitting marks the
            cell artists `animated`, which means matplotlib leaves them out of a normal
            draw: they are meant to be painted afterwards by an explicit blit. But Qt
            starts draws of its own, on show, on resize, on any expose, and each one
            wiped the cells until something happened to blit again. Switching the view
            from 1 to 2 and back was that something, which is exactly how it was
            reported.

            Hooking the draw event fixes it at the source: every draw, whoever asked for
            it, ends with the background captured and the cells drawn over it.
            """
            try:
                self._background = self._canvas.copy_from_bbox(self._figure.bbox)
            except Exception:                  # noqa: BLE001 - no renderer yet
                self._background = None
                return
            for artist in self._artists:
                self._axes.draw_artist(artist)

        def _blit(self) -> None:
            if self._background is None:
                self._canvas.draw_idle()
                return
            self._canvas.restore_region(self._background)
            for artist in self._artists:
                self._axes.draw_artist(artist)
            self._canvas.blit(self._figure.bbox)

        def _invalidate(self) -> None:
            """The captured background is only valid for THESE axes limits. Zoom, pan
            and fit all move them, and restoring a stale background draws the cells over
            an image that has shifted underneath."""
            self._background = None

        def _draw_preview(self) -> None:
            if self._preview is None:
                return
            import matplotlib.lines as mlines

            outline, length = self._preview
            if outline is None:
                return
            line = mlines.Line2D(outline[:, 0], outline[:, 1], color="#ff6b35",
                                 lw=1.4, ls="-", alpha=0.9, zorder=7)
            self._artists.append(self._axes.add_line(line))
            text = self._axes.text(outline[:, 0].mean(), outline[:, 1].mean(),
                                   f"L {length:.1f}  W {self._width:.1f}",
                                   color="#ff6b35", fontsize=8, ha="center", zorder=8)
            self._artists.append(text)

        def _refresh_cell_list(self) -> None:
            self._cell_list.blockSignals(True)
            self._cell_list.clear()
            for i, cell in enumerate(self._document.cells):
                shape = "curved" if cell.is_curved else "straight"
                mark = " *" if cell.regenerated else ""
                self._cell_list.addItem(
                    f"{i + 1}: {cell.length_px:.0f} × {cell.width:.1f} px  "
                    f"{shape}{mark}")
            if self._selected is not None:
                self._cell_list.setCurrentRow(self._selected)
            self._cell_list.blockSignals(False)

        def _refresh_stats(self) -> None:
            if self._selected is None or self._selected >= len(self._document.cells):
                self._stats.setText("–")
                return
            cell = self._document.cells[self._selected]
            self._stats.setText(
                f"<b>Cell #</b> {self._selected + 1}<br>"
                f"<b>Length</b> {cell.length_px:.1f} px<br>"
                f"<b>Width</b> {cell.width:.1f} px<br>"
                f"<b>Ribs</b> {len(cell.ribs)}<br>"
                f"<b>Bends</b> {max(0, len(cell.spine) - 2)}")

        def _scale(self) -> float:
            try:
                box = self._axes.get_window_extent()
                lo, hi = self._axes.get_xlim()
                span = abs(hi - lo)
                return float(box.width / span) if span else 1.0
            except Exception:                          # noqa: BLE001 - pre-layout
                return 1.0

        # ── view ─────────────────────────────────────────────────────────────
        def _fluor_index(self, view: str):
            """Which fluorescence folder a view shows, or None for the brightfield."""
            from .editor_project import FLUOR_VIEWS

            for k, keys in enumerate(FLUOR_VIEWS):
                if view in keys:
                    return k
            return None

        def _fluor_image(self, view: str):
            k = self._fluor_index(view)
            if k is None or k >= len(self._fluors):
                return None
            return self._fluors[k]

        def _set_view(self, view: str) -> None:
            k = self._fluor_index(view)
            if k is not None and self._fluor_image(view) is None:
                folder = (self._fluor_folders[k] if k < len(self._fluor_folders)
                          else "fluorescence")
                self._say(f"This field has no {folder}/ image."
                          if not view.startswith("overlay") else
                          f"The overlay needs a {folder}/ image, and this field has none.")
                return
            self._view = view
            for name, button in getattr(self, "_view_buttons", {}).items():
                button.setChecked(name == view)
            self._refresh(full=True)

        @staticmethod
        def _overlay_colours():
            """The tints, from the view module so the two cannot drift apart."""
            from .editor_view import OVERLAY_COLOURS

            return OVERLAY_COLOURS

        def _set_colour(self, colour: str) -> None:
            self._colour = colour
            if self._view == "overlay":
                self._refresh(full=True)

        def _set_mode(self, mode: str) -> None:
            self._mode = mode
            self._refresh(full=True)

        def _reset_display(self) -> None:
            self._bright.blockSignals(True)
            self._contrast.blockSignals(True)
            self._bright.setValue(100)
            self._contrast.setValue(100)
            self._bright.blockSignals(False)
            self._contrast.blockSignals(False)
            self._sync_value_labels()
            self._refresh(full=True)
            self._say("Brightness and contrast reset.")

        def _zoom(self, factor: float, centre=None) -> None:
            x0, x1 = self._axes.get_xlim()
            y0, y1 = self._axes.get_ylim()
            cx = centre[0] if centre else (x0 + x1) / 2.0
            cy = centre[1] if centre else (y0 + y1) / 2.0
            self._axes.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
            self._axes.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
            self._invalidate()
            self._refresh()

        def _fit(self) -> None:
            image = self._display_image()
            if image is None:
                return
            height, width = image.shape[:2]
            self._axes.set_xlim(0.0, float(width))
            self._axes.set_ylim(float(height), 0.0)
            self._invalidate()
            self._refresh()

        def _on_scroll(self, event) -> None:
            """Wheel walks the FIELDS; Ctrl+wheel zooms.

            Asked for, and it puts the common action on the bare gesture: curating a
            project is walking a list of thirty fields, and zoom is what you reach for
            occasionally within one. Wheel down is the NEXT field, matching how a wheel
            scrolls a document downward through it.

            The wheel is taken even OUTSIDE the axes, so a wheel over the margin still
            steps; zoom needs a cursor position and is ignored where there is none.
            """
            # THE KEY COMES FROM THE WHEEL EVENT ITSELF. matplotlib's `event.key` is the last
            # key the CANVAS saw, and a canvas that never has the focus (the outlines
            # window's) never sees one, so Ctrl+wheel read as a plain wheel and stepped the
            # field (the UI designer, 26 Nov 2026). `guiEvent` is Qt's own QWheelEvent, whose modifiers
            # are the keys held at that instant; `event.key` stays the fallback.
            gui = getattr(event, "guiEvent", None)
            if gui is not None and hasattr(gui, "modifiers"):
                zooming = bool(gui.modifiers() & Qt.ControlModifier)
            else:
                modifiers = (event.key or "")
                zooming = "control" in modifiers or "ctrl" in modifiers
            if zooming:
                if event.inaxes is not self._axes or event.xdata is None:
                    return
                self._zoom(1 / 1.2 if event.button == "up" else 1.2,
                           centre=(event.xdata, event.ydata))
                return
            self._step_field(-1 if event.button == "up" else +1)

        def _set_width(self, value: float) -> None:
            self._width = float(value)
            if self._selected is not None:
                try:
                    self._document.set_width(self._selected, self._width)
                except Exception as exc:               # noqa: BLE001 - reported
                    self._complain(exc)
                    return
                self._refresh()

        # ── mouse ────────────────────────────────────────────────────────────
        def _on_press(self, event) -> None:
            if event.inaxes is not self._axes or event.xdata is None:
                return
            point = (float(event.xdata), float(event.ydata))
            if event.button == 2:
                self._pan = point                      # middle-drag pans
                return
            if event.button == 3:
                # RIGHT-CLICK DELETES THE CELL UNDER IT, as the browser editor does.
                # The first version put PANNING here, which quietly took a gesture that
                # was already in the experimenter's hands. Panning is the middle button.
                self._delete_at(point)
                return
            if self._tool == "draw":
                self._draw_click(point)
                return
            if self._tool == "split":
                self._split_at(point)
                return

            from .editor_view import hit_test

            hit = hit_test(self._document, point, scale=self._scale(),
                           selected=self._selected)
            if hit is None:
                self._selected = None
                self._refresh()
                return
            self._selected = hit.cell
            # A BODY DRAG MOVES THE WHOLE CELL, which the browser has done all along and
            # this window had simply lost: `body` only selected, so grabbing a cell and
            # pulling did nothing at all.
            self._drag = hit if hit.kind in ("pole", "bend", "width", "body") else None
            self._grab = point
            if self._drag is not None:
                # ONE undo step for the whole gesture. Every edit method checkpoints, so
                # without this a drag pushed one entry per pointer move and Ctrl+Z
                # walked back through it a pixel at a time.
                self._document.begin_gesture()
            # SELECTING A CELL DOES NOT CHANGE THE PRESET either, and for the same
            # reason the drag does not: the box is the width the NEXT cell is drawn at,
            # so clicking a narrow cell to inspect it must not quietly make every cell
            # drawn afterwards narrow. The selected cell's own width is on the stats
            # panel, which is where a per-cell number belongs.
            self._refresh()

        def _delete_at(self, point) -> None:
            from .editor_view import nearest_cell

            index = nearest_cell(self._document, point, scale=self._scale())
            if index is None:
                self._say("Right-click a cell to delete it.")
                return
            self._document.delete_cell(index)
            self._selected = None
            self._refresh()
            self._say(f"Cell {index + 1} deleted. Ctrl+Z restores it.")

        def _on_move(self, event) -> None:
            if event.inaxes is not self._axes or event.xdata is None:
                return
            point = (float(event.xdata), float(event.ydata))
            if self._pan is not None:
                dx, dy = point[0] - self._pan[0], point[1] - self._pan[1]
                x0, x1 = self._axes.get_xlim()
                y0, y1 = self._axes.get_ylim()
                self._axes.set_xlim(x0 - dx, x1 - dx)
                self._axes.set_ylim(y0 - dy, y1 - dy)
                self._invalidate()
                self._refresh(lists=False)
                return
            if self._tool == "draw" and self._draw_from is not None:
                from .editor_view import preview_capsule

                self._preview = preview_capsule(self._draw_from, point, self._width)
                self._refresh(lists=False)
                return
            if self._drag is None:
                return
            try:
                self._apply_drag(point)
            except Exception:                          # noqa: BLE001
                # A refusal DURING a drag is not reported: the pointer is still moving
                # and a dialog would fire on every mouse event. The cell stops following
                # until the pointer is somewhere legal again.
                return
            self._refresh(lists=False)

        def _on_release(self, event) -> None:
            was_dragging = self._drag is not None
            if was_dragging:
                self._document.end_gesture()
            self._drag = None
            self._grab = None
            self._pan = None
            if was_dragging:
                # the lists were held still during the drag; catch them up once
                self._refresh_cell_list()
                self._refresh_stats()

        def _apply_drag(self, point) -> None:
            hit = self._drag
            if hit is None:
                return
            if hit.kind == "body":
                # BY DELTA, not to an absolute point: the cell must not jump so its
                # centre lands under the cursor the moment you touch it.
                self._document.move_cell(hit.cell, point[0] - self._grab[0],
                                         point[1] - self._grab[1])
                self._grab = point
            elif hit.kind == "pole":
                self._document.move_pole(hit.cell, 0 if hit.point == 0 else -1, point)
            elif hit.kind == "bend":
                self._document.move_spine_point(hit.cell, hit.point, point)
            elif hit.kind == "width":
                from .editor_view import width_axis

                # across the cell, along the middle rib's normal, not the distance from
                # a spine point. The projection is what makes the handle follow the
                # pointer: dragging ALONG the cell leaves the width alone, and on a
                # straight cell (two spine points) the old code measured from the far
                # POLE, which is why the handle answered at ninety degrees to the pull.
                #
                # EACH HANDLE MOVES ITS OWN EDGE. Asked for after using it: the earlier
                # version measured from the CENTRELINE and set a symmetric width, so
                # pulling the right edge moved the left one too and the cell grew on a
                # side nobody had hold of. The width is now measured from the OPPOSITE
                # edge, which is what stays put.
                cell = self._document.cells[hit.cell]
                centre, normal = width_axis(cell)
                side = 1 if hit.point == 1 else 0
                sign = -1.0 if side == 0 else 1.0
                fixed = centre - sign * normal * (cell.width / 2.0)
                span = abs(float((point[0] - fixed[0]) * normal[0]
                                 + (point[1] - fixed[1]) * normal[1]))
                self._document.set_width_from_edge(hit.cell, side, max(span, 1.0))
                # THE PRESET IS NOT TOUCHED. The box is the width the NEXT cell is drawn
                # at; dragging one cell's edge is an edit to that cell, and writing it
                # back would mean every correction silently re-set the default for
                # everything drawn afterwards.

        # ── tools ────────────────────────────────────────────────────────────
        def _set_tool(self, name: str) -> None:
            self._tool = name
            self._draw_from = None
            self._preview = None
            for key, button in self._tool_buttons.items():
                button.setChecked(key == name)
            self._set_hint({"select": "drag the cell to move it, or drag a pole, a bend "
                                      "or the edge",
                            "draw": "click one pole, then the other",
                            "split": "click a cell where it should be cut"}[name])
            self._refresh()

        def _draw_click(self, point) -> None:
            if self._draw_from is None:
                self._draw_from = point
                self._set_hint("now click the other pole")
                return
            try:
                index = self._document.add_cell([self._draw_from, point], self._width)
            except Exception as exc:                   # noqa: BLE001 - reported
                self._draw_from = None
                self._preview = None
                self._complain(exc)
                return
            self._draw_from = None
            self._preview = None
            self._selected = index
            self._refresh()
            self._say(f"Cell {index + 1} drawn.")

        def _split_at(self, point) -> None:
            # THE CELL UNDER THE CLICK is cut, selected or not (asked 15 Oct 2026: "needs to
            # select cell first before it works, using this tool should just cut where u
            # click"). Bodies only: a neighbour's pole handle at the septum must not take the
            # cut. Among touching cells the selected one wins; a click on no cell still cuts
            # the selected one at the septum nearest the click, as before.
            from .editor_view import hit_test

            hit = hit_test(self._document, point, scale=self._scale(), handle_px=0.0,
                           selected=self._selected)
            index = hit.cell if hit is not None else self._selected
            if index is None:
                self._say("Click a cell where it should be cut.")
                return
            try:
                a, _b = self._document.split_cell(index, point)
            except Exception as exc:                   # noqa: BLE001 - reported
                self._complain(exc)
                return
            self._selected = a
            self._refresh()
            self._say(f"Split into cells {a + 1} and {a + 2}.")

        def _select_from_list(self, row: int) -> None:
            self._selected = row if 0 <= row < len(self._document.cells) else None
            self._refresh()

        def _undo(self) -> None:
            if self._document.undo():
                self._clamp_selection()
                self._refresh()
            else:
                self._say("Nothing to undo.")

        def _redo(self) -> None:
            if self._document.redo():
                self._clamp_selection()
                self._refresh()
            else:
                self._say("Nothing to redo.")

        def _clamp_selection(self) -> None:
            if (self._selected is not None
                    and self._selected >= len(self._document.cells)):
                self._selected = None

        def _delete(self) -> None:
            if self._selected is None:
                self._say("select a cell first")
                return
            self._document.delete_cell(self._selected)
            self._selected = None
            self._refresh()

        def _merge(self) -> None:
            if self._selected is None:
                self._say("Select a cell first, then press M.")
                return
            other = self._nearest_other(self._selected)
            if other is None:
                self._say("There is no other cell to merge with.")
                return
            try:
                keep = self._document.merge_cells(self._selected, other)
            except Exception as exc:                   # noqa: BLE001 - reported
                self._complain(exc)
                return
            self._selected = keep
            self._refresh()
            self._say(f"Merged into cell {keep + 1}.")

        def _nearest_other(self, index: int):
            """The neighbour whose nearest POLE is closest. Only picks a candidate,
            `merge_cells` owns the geometry and the refusal."""
            import math

            cell = self._document.cells[index]
            best, best_gap = None, float("inf")
            for other, candidate in enumerate(self._document.cells):
                if other == index:
                    continue
                for a in (cell.spine[0], cell.spine[-1]):
                    for b in (candidate.spine[0], candidate.spine[-1]):
                        gap = math.hypot(a[0] - b[0], a[1] - b[1])
                        if gap < best_gap:
                            best, best_gap = other, gap
            return best

        def _add_bend(self) -> None:
            """A new control point goes in the middle of the longest span, ON the spine,
            so pressing B changes nothing until it is dragged, an edit that moves the
            cell the moment you ask for a handle is one you have to undo to see."""
            if self._selected is None:
                self._say("Select a cell first, then press B.")
                return
            cell = self._document.cells[self._selected]
            spans = [(float(((cell.spine[i + 1] - cell.spine[i]) ** 2).sum()), i)
                     for i in range(len(cell.spine) - 1)]
            _length, k = max(spans)
            middle = (cell.spine[k] + cell.spine[k + 1]) / 2.0
            try:
                self._document.insert_spine_point(self._selected, k, middle)
            except Exception as exc:                   # noqa: BLE001 - reported
                self._complain(exc)
                return
            self._refresh()
            self._say("Bend added. Drag the square handle.")

        def _remove_bend(self) -> None:
            if self._selected is None:
                self._say("select a cell first")
                return
            cell = self._document.cells[self._selected]
            if len(cell.spine) <= 2:
                self._say("This cell is already straight.")
                return
            try:
                self._document.delete_spine_point(self._selected, len(cell.spine) // 2)
            except Exception as exc:                   # noqa: BLE001 - reported
                self._complain(exc)
                return
            self._refresh()

        # ── saving ───────────────────────────────────────────────────────────
        def _save(self) -> None:
            if self._index < 0:
                return
            self._save_one(self._index, announce=True)

        def _save_one(self, index: int, *, announce: bool) -> bool:
            from .editor_project import field_image_shape, save_field

            field = self._fields[index]
            document = self._documents.get(str(field["mat"]), self._document)
            if not field.get("has_mat") and not document.modified:
                # NOTHING WAS DRAWN AND NOTHING EXISTS: writing now would make an EMPTY
                # outline file and mark the field corrected, which says "a human looked
                # and there are no cells" and drops the movie's tracks (the UI designer, 26 Nov
                # 2026). A real empty field is made by deleting cells (that is an edit).
                if announce:
                    self._say(f"{field['name']} has no outlines and nothing drawn, so "
                              f"nothing was saved. Draw a cell first.")
                return False
            # THE DISPLAYED FIELD'S SHAPE IS IN HAND; every other field's comes from
            # its own header. Without the second branch "Save all" wrote a mask for the
            # field on screen and silently none for the rest, the shape of defect this
            # project keeps finding, a capability wired at one end.
            shape = None
            if index == self._index and self._phase is not None:
                shape = (self._phase.shape[-2], self._phase.shape[-1])
            else:
                shape = field_image_shape(field)
            try:
                path, backup = save_field(field["mat"], document, image_shape=shape)
            except Exception as exc:                   # noqa: BLE001 - reported
                if announce:
                    self._complain(exc)
                return False
            document.modified = False
            field["has_mat"] = True
            if announce:
                self._say(f"Saved {len(document.cells)} cells to {path.name}"
                          + (f"  (originals kept in {backup.parent.name}/)"
                             if backup else ""))
            return True

        def _save_all(self) -> None:
            """Every field edited this session, not just the one on screen.

            The browser editor has this and a curation run is dozens of fields, so
            without it a session ends with the arrow keys having quietly discarded work.
            """
            self._stash()
            pending = [i for i, field in enumerate(self._fields)
                       if self._documents.get(str(field["mat"]))
                       and self._documents[str(field["mat"])].modified]
            if not pending:
                self._say("Nothing has been modified.")
                return
            written = sum(1 for i in pending if self._save_one(i, announce=False))
            self._refresh_cell_list()
            self._say(f"Saved {written} of {len(pending)} modified field(s).")

        def _close_event(self, event):                   # noqa: N802 - Qt virtual
            self._stash()
            unsaved = [f["name"] for f in self._fields
                       if self._documents.get(str(f["mat"]))
                       and self._documents[str(f["mat"])].modified]
            if not unsaved:
                event.accept()
                return
            answer = QMessageBox.question(
                self, "Unsaved outlines",
                f"{len(unsaved)} field(s) have edits that are not saved:\n  "
                + "\n  ".join(unsaved[:8])
                + ("\n  …" if len(unsaved) > 8 else "")
                + "\n\nLeave them?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            event.accept() if answer == QMessageBox.Yes else event.ignore()


    _CORE = EditorCore
    return EditorCore


def _build_window_class():
    """Qt imported inside, so this module is importable without a GUI stack."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton,
                                QSlider, QVBoxLayout, QWidget)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    def _slider(low, high, start, step=1):
        bar = QSlider(Qt.Horizontal)
        bar.setRange(int(low), int(high))
        bar.setValue(int(start))
        bar.setSingleStep(int(step))
        return bar


    class MeshEditorWindow(_core_class(), QMainWindow):
        """Draw and correct the cell outlines the analysis assigns tracks to."""

        def __init__(self, root=None, parent=None):
            super().__init__(parent)


            from ..buildstamp import build_stamp

            self.setWindowTitle(f"SMTracker · Mesh Editor  ·  {build_stamp()}")
            self.resize(1320, 830)

            self._init_state(root)

            central = QWidget()
            layout = QHBoxLayout(central)
            layout.addLayout(self._build_sidebar(), 0)
            layout.addLayout(self._build_canvas_column(), 1)
            self.setCentralWidget(central)
            self._status = self.statusBar()
            self._hint = QLabel("")
            self._status.addPermanentWidget(self._hint)

            self._start()

        # ── construction ─────────────────────────────────────────────────────
        def _build_sidebar(self):
            side = QVBoxLayout()
            self._controls["open_folder"] = button = QPushButton("Open folder…")
            button.clicked.connect(self._pick_folder)
            side.addWidget(button)

            side.addWidget(QLabel("Fields"))
            self._field_list = QListWidget()
            self._field_list.currentRowChanged.connect(self._open_index)
            side.addWidget(self._field_list, 3)

            side.addWidget(QLabel("Cells"))
            self._cell_list = QListWidget()
            self._cell_list.currentRowChanged.connect(self._select_from_list)
            side.addWidget(self._cell_list, 2)
            self._controls["cell_list"] = self._cell_list

            self._stats = QLabel("–")
            self._stats.setTextFormat(Qt.RichText)
            self._stats.setFrameShape(QFrame.StyledPanel)
            self._stats.setMargin(6)
            self._controls["cell_stats"] = self._stats
            side.addWidget(self._stats)

            # ALIGNMENT: a note that always says where things stand, and a button that
            # is only offered when there is a shift worth applying.
            self._align_note = QLabel("")
            self._align_note.setWordWrap(True)
            self._align_note.setStyleSheet("color: #666; font-size: 11px;")
            self._align_button = QPushButton("Shift to fluorescence")
            self._align_button.setEnabled(False)
            self._align_button.setToolTip(
                "Move every outline by the measured offset between the outlines and the "
                "fluorescence channel. One undo step. Available only when the offset is "
                "larger than the tolerance.")
            self._align_button.clicked.connect(self._apply_offset)

            self._controls["save_field"] = save = QPushButton("Save field  (Ctrl+S)")
            save.clicked.connect(self._save)
            side.addWidget(save)
            side.addWidget(self._align_button)
            side.addWidget(self._align_note)
            self._controls["save_all"] = save_all = QPushButton("Save ALL modified")
            save_all.clicked.connect(self._save_all)
            side.addWidget(save_all)
            return side

        def _build_canvas_column(self):
            right = QVBoxLayout()
            right.addLayout(self._build_tool_row())
            right.addLayout(self._build_view_row())

            self._figure = Figure(figsize=(7.6, 7.6), layout="constrained")
            self._canvas = FigureCanvasQTAgg(self._figure)
            # CLICKING THE PICTURE MUST TAKE THE FOCUS, or focus stays in
            # whichever box was last typed in and the shortcuts stay dead.
            self._canvas.setFocusPolicy(Qt.StrongFocus)
            self._axes = self._figure.add_subplot(111)
            self._canvas.mpl_connect("button_press_event", self._on_press)
            self._canvas.mpl_connect("motion_notify_event", self._on_move)
            self._canvas.mpl_connect("button_release_event", self._on_release)
            self._canvas.mpl_connect("scroll_event", self._on_scroll)
            # AFTER EVERY REAL DRAW, including the ones Qt starts on its own. See
            # `_on_draw`: without this the cells are invisible until something forces
            # an explicit blit.
            self._canvas.mpl_connect("draw_event", self._on_draw)
            right.addWidget(self._canvas, 1)
            right.addWidget(self._shortcut_label())
            return right

        def _shortcut_label(self):
            text = "   ".join(f"<b>{key}</b> {what}" for key, what, _c in SHORTCUTS)
            label = QLabel(text)
            label.setTextFormat(Qt.RichText)
            label.setWordWrap(True)
            label.setStyleSheet("font-size: 10px;")
            return label

        def _build_tool_row(self):
            row = QHBoxLayout()
            self._tool_buttons = {}
            for name, label, key in (("select", "✦ Select / edit  (S)", "tool_select"),
                                     ("draw", "✚ Draw cell  (D)", "tool_draw"),
                                     ("split", "✂ Split at septum  (X)", "tool_split")):
                b = QPushButton(label)
                b.setCheckable(True)
                b.clicked.connect(lambda _c=False, n=name: self._set_tool(n))
                row.addWidget(b)
                self._tool_buttons[name] = b
                self._controls[key] = b
            for key, label, slot in (("merge", "⇔ Merge  (M)", self._merge),
                                     ("delete", "🗑 Delete  (Del)", self._delete),
                                     ("add_bend", "⌒ Add bend  (B)", self._add_bend),
                                     ("remove_bend", "╱ Straighten  (N)",
                                      self._remove_bend),
                                     ("undo", "↩ Undo  (Ctrl+Z)", self._undo),
                                     ("redo", "↪ Redo  (Ctrl+Y)", self._redo)):
                b = QPushButton(label)
                b.clicked.connect(slot)
                row.addWidget(b)
                self._controls[key] = b
            row.addStretch(1)
            return row

        def _build_view_row(self):
            """The controls above the canvas, in TWO rows.

            One row was the bug. Everything lived in a single QGridLayout row of
            about twenty columns, so each QSlider was allotted a few pixels and rendered
            as a nub: reported as "alpha brightness contrast just a button and not a
            bar". They were sliders the whole time; there was no room to draw one.

            So the image adjustments get a row of their own, each slider a real minimum
            width, and each its numeric value beside it as the browser editor shows
            (`imgBrightVal` and friends). A slider whose value you cannot read is only
            half a control.
            """
            rows = QVBoxLayout()
            rows.setSpacing(4)

            top = QHBoxLayout()
            self._view_buttons = {}
            for key, label, view in (("view_phase", "Phase  (1)", "phase"),
                                     ("view_sum", "Sum  (2)", "sum"),
                                     ("view_overlay", "Overlay  (3)", "overlay")):
                button = QPushButton(label)
                button.setCheckable(True)
                button.clicked.connect(lambda _c=False, v=view: self._set_view(v))
                top.addWidget(button)
                self._controls[key] = button
                self._view_buttons[view] = button

            top.addSpacing(12)
            self._colour_box = QComboBox()
            self._colour_box.addItems(list(self._overlay_colours()))
            self._colour_box.setCurrentText(self._colour)
            self._colour_box.currentTextChanged.connect(self._set_colour)
            self._colour_box.activated.connect(self._release_focus)
            self._controls["overlay_colour"] = self._colour_box
            top.addWidget(QLabel("tint"))
            top.addWidget(self._colour_box)

            self._mode_box = QComboBox()
            self._mode_box.addItems(["imagej", "septa", "outline"])
            self._mode_box.currentTextChanged.connect(self._set_mode)
            self._mode_box.activated.connect(self._release_focus)
            self._controls["stretch_mode"] = self._mode_box
            top.addWidget(QLabel("stretch"))
            top.addWidget(self._mode_box)

            self._width_box = QDoubleSpinBox()
            self._width_box.setRange(1.0, 60.0)
            self._width_box.setSingleStep(0.5)
            self._width_box.setValue(self._width)
            self._width_box.valueChanged.connect(self._set_width)
            # A BOX THAT KEEPS FOCUS KEEPS THE SHORTCUTS. `eventFilter` deliberately
            # declines every key while focus is in a spin box, line edit or combo (the
            # browser's own `if (e.target.tagName === 'INPUT') return;`) -- which is
            # right, since the digits must reach the box. But nothing gave the focus
            # back, so one click here killed 1/2/3 and every letter shortcut for the
            # rest of the session. `editingFinished` fires on Enter AND on focus-out,
            # so both of the experimenter's conditions ("after pressing enter or after
            # pressing somewhere else") are the same signal.
            self._width_box.editingFinished.connect(self._release_focus)
            self._controls["width"] = self._width_box
            top.addWidget(QLabel("width px"))
            top.addWidget(self._width_box)

            top.addStretch(1)
            for key, label, slot, wide in (
                    ("zoom_out", "−", lambda: self._zoom(1.4), 34),
                    ("zoom_in", "+", lambda: self._zoom(1 / 1.4), 34),
                    ("zoom_fit", "Fit  (F)", self._fit, 70),
                    ("prev_field", "◀", lambda: self._step_field(-1), 34),
                    ("next_field", "▶", lambda: self._step_field(+1), 34)):
                button = QPushButton(label)
                button.clicked.connect(slot)
                button.setMaximumWidth(wide)
                top.addWidget(button)
                self._controls[key] = button
            rows.addLayout(top)

            adjust = QHBoxLayout()
            self._value_labels = {}
            for key, text, low, high, start, suffix in (
                    ("overlay_opacity", "alpha", 0, 100, 50, " %"),
                    ("brightness", "bright", 20, 300, 100, " %"),
                    ("contrast", "contrast", 20, 400, 100, " %")):
                bar = _slider(low, high, start)
                # A MINIMUM WIDTH, which is the whole fix: without one, the layout is
                # free to squeeze a slider down to its handle.
                bar.setMinimumWidth(130)
                readout = QLabel(f"{start}{suffix}")
                readout.setMinimumWidth(46)
                bar.valueChanged.connect(
                    lambda value, lbl=readout, sfx=suffix: (
                        lbl.setText(f"{value}{sfx}"), self._redraw_image()))
                adjust.addWidget(QLabel(text))
                adjust.addWidget(bar, 1)
                adjust.addWidget(readout)
                adjust.addSpacing(10)
                self._controls[key] = bar
                self._value_labels[key] = (readout, suffix)

            self._opacity = self._controls["overlay_opacity"]
            self._bright = self._controls["brightness"]
            self._contrast = self._controls["contrast"]

            self._controls["reset_display"] = reset = QPushButton("Reset  (R)")
            reset.clicked.connect(self._reset_display)
            adjust.addWidget(reset)
            rows.addLayout(adjust)
            return rows

        def closeEvent(self, event):                  # noqa: N802
            # RELEASED ONLY IF THE CLOSE ACTUALLY HAPPENS. `segment_gui` asks
            # "a run is still going, stop it?" and IGNORES the event when the
            # answer is no: dropping the reference there would leave a window
            # on screen that nothing holds, which is the exact segfault this
            # list exists to prevent, reintroduced by the fix for it.
            self._close_event(event)
            if event.isAccepted():
                _release(self)

    return MeshEditorWindow


def __getattr__(name):
    """`MeshEditorWindow` built on first use, so importing this module needs no Qt."""
    if name == "MeshEditorWindow":
        return _build_window_class()
    raise AttributeError(name)
