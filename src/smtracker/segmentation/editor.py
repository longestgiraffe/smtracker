"""The mesh editor: a browser window for drawing and correcting cell outlines.

Ported from the segmentation repository, with its private mesh I/O removed. The
original carried its own ``load_mat``/``save_mat``, a fourth copy of the Oufti
format, beside `hada/meshio.py`, beside `cellmesh`, beside the three the segmentation
repo's own module head records as having already drifted. Reading and writing now go
through :mod:`smtracker.cellmesh`, so a cell the editor writes is a cell this tool
reads, by construction rather than by agreement.

Why flask and a browser, inside a Qt application. Because it works, and because a
napari port is a rewrite of 1300 lines of canvas code that would buy integration and
nothing else. Nothing here is on the analysis spine and nothing here computes a number.

**It blocks the process that starts it, and that is stated rather than dressed up.**
:func:`serve` runs the server in the calling process until it is stopped, so choosing
this door from the launcher gives you the editor and a terminal you end with Ctrl+C.
That is honest and it is not where this should end: a Qt window that opened the editor
and stayed usable would need it as a child process, the way `smtracker.fiji` drives
Fiji. The pieces for that are here (:func:`free_port` and :func:`make_token` exist so
a parent can hand a child both) and the parent does not, yet.

The drawing is in javascript, and this module does not pretend otherwise. The
editor builds each capsule in the browser (`makeCapsule`) so a dragged handle follows
the mouse without a round trip. `segmentation.geometry.capsule_from_spine` is the
Python one, used by the automatic path and available here: the two are separate
implementations of one shape, which is the thing this project has just spent a commit
removing everywhere else. It is tolerated ONLY for the live preview, and only because
the alternative is a network round trip per mouse move. What lands ON DISK is not left
to the preview: :func:`rebuild_mesh` recomputes it in Python from the same endpoints,
so the stored file always comes from the one implementation. See `docs/`.

Three things the original did that a shipped tool must not:

* it ran ``pip install`` at startup, which a frozen application cannot do and a
  managed environment must not have done to it;
* it bound the fixed port 5001, so a second editor, or anything else on that port,
  failed with an error naming neither;
* it served an unauthenticated API over HTTP that reads and WRITES arbitrary paths.
  It binds loopback only, but any page in any browser tab can reach a loopback port,
  so the API now requires a per-run token that only the served page knows.
"""
from __future__ import annotations

import base64
import io
import json
import os
import secrets
import socket
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

__all__ = [
    "EditorError",
    "CURATED_SUFFIX",
    "DISPLAY_MODES",
    "IDLE_TIMEOUT_S",
    "HEARTBEAT_CHECK_S",
    "free_port",
    "make_token",
    "build_app",
    "serve",
    "main",
]


class EditorError(RuntimeError):
    """Raised when the editor cannot start or cannot honour a request."""


#: Written beside a ``.mat`` the moment a human saves it, so training and QC can tell
#: a curated field from automatic output. The original's marker, kept.
CURATED_SUFFIX = ".curated.json"

#: Stop after this long with no page checking in.
#:
#: 90 s, AND THE FLOOR IS NOT ARBITRARY: browsers throttle `setInterval` in a
#: BACKGROUND tab to roughly once a minute, so anything under ~60 s would shut the
#: editor down on someone who switched windows to look at a phase image. The cost of
#: waiting is one idle process for a minute and a half; the cost of being too eager is
#: losing unsaved outlines.
IDLE_TIMEOUT_S = 90.0

#: How often the watchdog looks. Short, because it also carries the goodbye beacon,
#: closing the tab should stop the process in a second or two, not in ninety.
HEARTBEAT_CHECK_S = 1.0

#: Percentile windows for the phase display. No single stretch serves both jobs: a
#: wide window keeps the cell/background edge crisp but saturates the interior, hiding
#: the septum; a window ending at the background level spreads the cell's own dark
#: range over the whole display, which is where septal contrast lives.
DISPLAY_MODES = {
    "imagej": (0.1, 99.9),    # faithful, what ImageJ shows on opening a 16-bit image
    "outline": (2.0, 98.0),   # punchy cell/background edge
    "septa": (0.2, 50.0),     # min..background over the full range; reveals septa
}


def free_port() -> int:
    """Ask the OS for a port nothing is using.

    The original hard-coded 5001. A second editor, or anything else already on it,
    then failed with a bind error naming neither the port nor the other process, and
    on Windows a stale handle can hold it after the window is closed.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_token() -> str:
    """A per-run secret the served page knows and nothing else does.

    The server binds loopback, which stops a machine on the network reaching it and
    does NOT stop a web page in the user's own browser: any origin can issue requests
    to 127.0.0.1. This API reads and writes arbitrary paths, so it is not a thing to
    leave open for the lifetime of the window.
    """
    return secrets.token_urlsafe(24)


def _html_path() -> Path:
    here = Path(__file__).resolve().parent / "editor.html"
    if not here.exists():                       # pragma: no cover - packaging error
        raise EditorError(f"the editor page is missing: {here}")
    return here


def _render_tiff(path, lo: float, hi: float):
    """One phase TIFF to a base64 PNG plus its shape.

    The percentile default is near min/max, which is what ImageJ shows on opening.
    A 2–98 stretch clips 4 % of pixels; on these images the background sits about two
    noise SD from those cuts, so the mottle gets stretched over the whole display and
    the field looks far noisier than it is.
    """
    import numpy as np
    import tifffile
    from PIL import Image

    img = tifffile.imread(path).astype("float32")
    if img.ndim == 3:
        img = img[0]
    a, b = np.percentile(img, (lo, hi))
    img = np.clip((img - a) / max(b - a, 1e-9), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype("uint8"), mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), img.shape


def _cells_for_browser(path):
    """Read one ``.mat`` through :mod:`smtracker.cellmesh` into the page's shape.

    ONE READER. The original parsed the MATLAB struct itself and so accepted files
    `cellmesh` refuses (a ribless outline, a non-finite mesh) meaning the editor
    could open and re-save a cell the analysis would later drop without saying so.
    """
    import numpy as np

    from ..cellmesh import read_cell_meshes

    cells = []
    for cell in read_cell_meshes(path, on_empty="empty"):
        mesh = cell.mesh if cell.mesh is not None else np.zeros((0, 4))
        xs, ys = cell.polygon[:, 0], cell.polygon[:, 1]
        cells.append({
            "id": cell.cell_id,
            "mesh": np.asarray(mesh, dtype=float).tolist(),
            "model": np.asarray(cell.polygon, dtype=float).tolist(),
            "box": [float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min())],
        })
    return cells


def rebuild_mesh(spine, width: float):
    """The Python capsule, for a client that sends a SPINE rather than a mesh.

    This is the one implementation (`segmentation.geometry`) and it is what makes
    the browser's `makeCapsule` a PREVIEW rather than a second source of truth. A
    client that sends only ``mesh``/``model`` is taken at its word; one that sends a
    spine gets the mesh rebuilt here, including the turn-radius refusal.
    """
    from .geometry import capsule_from_spine

    return capsule_from_spine(spine, width)


def _backup_once(mat_path: Path) -> str | None:
    """Keep the automatic meshes before a human first writes over them.

    Written to a sibling ``<folder>_auto/`` and only if absent, so repeated saves
    never touch the pristine copy and curation stays reversible.
    """
    import shutil

    mat_path = Path(mat_path)
    backup_dir = mat_path.parent.parent / (mat_path.parent.name + "_auto")
    backup = backup_dir / mat_path.name
    if not backup.exists() and mat_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mat_path, backup)
        return str(backup)
    return str(backup) if backup.exists() else None


def build_app(token: str):
    """The Flask application. Separated from :func:`serve` so it is testable.

    Flask is imported HERE, not at module scope, so importing this module costs
    nothing on a machine that will never open the editor, and so the analysis-only
    build does not acquire a web framework as a hard dependency.
    """
    import numpy as np
    from flask import Flask, jsonify, request, send_from_directory

    from ..cellmesh import CellMesh, CellMeshError, write_cell_meshes
    from .geometry import GeometryError

    app = Flask(__name__, static_folder=None)
    page = _html_path()

    @app.before_request
    def _require_token():
        # The page itself is served without one (the browser has to be able to open
        # the URL); every API call must carry it. Any tab on this machine can reach a
        # loopback port, and this API reads and writes arbitrary paths.
        if request.path in ("/", "/favicon.ico") or request.path.startswith("/static"):
            return None
        supplied = request.args.get("token") or request.headers.get("X-Editor-Token")
        if supplied is None and request.is_json:
            supplied = (request.get_json(silent=True) or {}).get("token")
        if not secrets.compare_digest(str(supplied or ""), token):
            return jsonify({"error": "bad or missing editor token"}), 403
        return None

    @app.route("/")
    def index():
        return send_from_directory(page.parent, page.name)

    @app.route("/api/alive", methods=["GET", "POST"])
    def alive():
        """The page saying it still has a tab open. See :data:`IDLE_TIMEOUT_S`."""
        app.config["last_seen"] = time.monotonic()
        return jsonify({"ok": True})

    @app.route("/api/bye", methods=["GET", "POST"])
    def bye():
        """The tab closing. Best effort: `sendBeacon` on unload usually arrives and is
        not guaranteed to, which is exactly why the idle timeout exists as well."""
        app.config["last_seen"] = 0.0
        return jsonify({"ok": True})

    @app.route("/favicon.ico")
    def favicon():
        # The browser asks for one on every load and it does not carry the token, so
        # it logged a 403 beside every page view, an authentication failure in the
        # log that is nothing of the kind, which is how a real one gets ignored.
        return "", 204

    @app.route("/api/browse")
    def browse():
        path = request.args.get("path") or str(Path.home())
        try:
            entries = [
                {"name": n, "isdir": os.path.isdir(os.path.join(path, n)),
                 "path": os.path.join(path, n)}
                for n in sorted(os.listdir(path))
            ]
        except OSError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"path": path, "entries": entries,
                        "parent": os.path.dirname(path)})

    @app.route("/api/browse_native")
    def browse_native():
        """The folder picker Explorer uses; the page browses in itself if it fails."""
        from ._folderpicker import pick_folder

        try:
            folder = pick_folder(request.args.get("initial") or str(Path.home()))
        except Exception as exc:                       # noqa: BLE001 - ctypes/COM
            return jsonify({"error": str(exc)}), 400
        if folder and os.path.isdir(folder):
            return jsonify({"path": folder})
        return jsonify({"cancelled": True})

    @app.route("/api/load")
    def load():
        phase_path = request.args.get("phase")
        mat_path = request.args.get("mat")
        if not phase_path or not os.path.exists(phase_path):
            return jsonify({"error": f"phase image not found: {phase_path}"}), 400
        window = DISPLAY_MODES.get(request.args.get("mode", "septa"),
                                   DISPLAY_MODES["imagej"])
        try:
            image, shape = _render_tiff(phase_path, *window)
            cells = (_cells_for_browser(mat_path)
                     if mat_path and os.path.exists(mat_path) else [])
        except (OSError, ValueError, CellMeshError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

        if not mat_path or not os.path.exists(mat_path):
            # the condition's outlines folder however it is spelled (`cell meshes` too)
            from ..cellmesh import mesh_folder, outline_file

            phase_file = Path(phase_path)
            condition = (phase_file.parent.parent if phase_file.parent.name.lower() == "phase"
                         else phase_file.parent)                  # a snapshot loose beside the movies
            # the movie's outline file under whatever name it is filed (`cellmesh.outline_file`)
            mat_path = str(outline_file(mesh_folder(condition), phase_file.stem))

        sum_image = sum_path = None
        candidate = phase_path.replace(f"{os.sep}phase{os.sep}", f"{os.sep}sum{os.sep}")
        for variant in (candidate, candidate.replace("_phase.", "_sum.")):
            for ext in ("", ".tif", ".tiff"):
                probe = variant if not ext else os.path.splitext(variant)[0] + ext
                if os.path.exists(probe):
                    try:
                        sum_image, _ = _render_tiff(probe, *DISPLAY_MODES["imagej"])
                        sum_path = probe
                    except (OSError, ValueError):
                        pass
                    break
            if sum_image:
                break

        return jsonify({"cells": cells, "image": image, "sum_image": sum_image,
                        "height": int(shape[0]), "width": int(shape[1]),
                        "mat_path": mat_path, "phase_path": phase_path,
                        "sum_path": sum_path,
                        # The channel offset was a HADA measurement and is not ported;
                        # zero here means "not applied", not "measured as zero".
                        "align_dx": 0, "align_dy": 0})

    @app.route("/api/load_sum")
    def load_sum():
        path = request.args.get("path")
        lo, hi = DISPLAY_MODES.get(request.args.get("mode", "septa"),
                                   DISPLAY_MODES["imagej"])
        lo = float(request.args.get("lo", lo))
        hi = float(request.args.get("hi", hi))
        try:
            image, shape = _render_tiff(path, lo, hi)
        except (OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"image": image, "height": int(shape[0]),
                        "width": int(shape[1])})

    @app.route("/api/save", methods=["POST"])
    def save():
        payload = request.get_json(silent=True) or {}
        mat_path = Path(payload.get("mat_path", ""))
        raw_cells = payload.get("cells") or []
        # an empty field is a result, not a refusal, the same change as the Qt
        # editor's `save_field`. Asked for: "sometimes i wanna be able to delte ALL
        # meshes from a series file ... make it so that even an emtpy mat is what it
        # is -> no cell mesh". The old reason ("read_cell_meshes refuses a file
        # holding none") describes that reader's non-default mode only, and stopped
        # being the shipped behaviour when unsegmented movies became survivable:
        # `read_project_cell_meshes` passes `on_empty="empty"`, `assign_cells` records
        # the movie under `.attrs["unsegmented_movies"]`, and the QC N-per-group panel
        # is where it shows up.
        try:
            meshes = []
            for i, cell in enumerate(raw_cells, start=1):
                spine = cell.get("spine")
                if spine:
                    mesh, model = rebuild_mesh(spine, float(cell["width"]))
                else:
                    mesh = np.asarray(cell["mesh"], dtype=float)
                    model = np.asarray(cell["model"], dtype=float)
                    if mesh.size == 0:
                        mesh = None
                meshes.append(CellMesh(movie=mat_path.stem, cell_id=str(i),
                                       polygon=model, mesh=mesh))
            mat_path.parent.mkdir(parents=True, exist_ok=True)
            backup = _backup_once(mat_path)
            write_cell_meshes(
                mat_path, meshes, overwrite=True,
                image_shape=(int(payload["height"]), int(payload["width"]))
                if payload.get("height") and payload.get("width") else None,
                # Deliberate: the writer keeps refusing an ACCIDENTAL empty list, so
                # only a caller that means it can erase a field.
                allow_empty=True,
            )
            marker = mat_path.with_name(mat_path.name + CURATED_SUFFIX)
            marker.write_text(json.dumps({
                "curated": True,
                "saved": datetime.now().isoformat(timespec="seconds"),
                "n_cells": len(meshes),
                "auto_backup": backup,
            }, indent=1), encoding="utf-8")
        except (CellMeshError, GeometryError, KeyError, TypeError, ValueError,
                OSError) as exc:
            # GeometryError IS THE USER'S MESSAGE, not an internal fault. It is what
            # a spine bending tighter than its own half-width raises, and it names the
            # two fixes that were measured to work. Left uncaught it came back as a
            # 500 with a traceback in the console and nothing on screen.
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify({"ok": True, "n_cells": len(meshes), "curated": True})

    @app.route("/api/scan")
    def scan():
        """Every condition folder under a root, with all their series.

        A tree like ``<root>/<condition>/phase/`` would otherwise be one editor
        session per condition. Each series name carries its condition so the sidebar
        stays unambiguous.

        THE LISTING IS `editor_project.scan_fields`, the desktop editor's, so both layouts
        are read here as there: a ``phase/`` folder, or the snapshots beside the movies
        (ruled 18 Oct 2026), and ``series001.nd2.tif`` is field ``series001``. This route
        had its own walk that knew only ``phase/``, so a loose condition gave an empty
        sidebar and no reason (handoff 18 Oct 2026, place 1).
        """
        from .editor_project import scan_fields

        root = request.args.get("root")
        if not root or not os.path.isdir(root):
            return jsonify({"error": "root is not a directory"}), 400
        return jsonify(_field_rows(scan_fields(root)))

    @app.route("/api/series")
    def series():
        """The fields of ONE condition folder (its own ``phase/`` or its loose snapshots),
        named without a condition prefix. The page's fallback when a scan found nothing."""
        from .editor_project import scan_fields

        folder = request.args.get("condition")
        if not folder or not os.path.isdir(folder):
            return jsonify({"error": "condition is not a directory"}), 400
        return jsonify(_field_rows(f for f in scan_fields(folder) if f["condition"] == "."))

    return app


def _field_rows(fields) -> list[dict]:
    """`scan_fields` rows as the page reads them: paths as strings."""
    return [{"name": f["name"], "condition": f["condition"], "phase": str(f["phase"]),
             "mat": str(f["mat"]), "has_mat": bool(f["has_mat"])} for f in fields]


def serve(*, port: int | None = None, token: str | None = None,
          open_browser: bool = True, root: str | None = None,
          idle_timeout_s: float | None = IDLE_TIMEOUT_S) -> str:
    """Run the editor until its page goes away. Returns the URL it served.

    ``root`` pre-fills the folder the sidebar scans, so launching from SMTracker with
    a project already open does not ask for the path a second time.

    **It stops when the tab does, and that is not free with a browser UI.** A window
    has a close event; a tab has none the server can rely on. `sendBeacon` on unload
    usually arrives and is not guaranteed to: a crashed browser, a killed process or a
    closed laptop send nothing at all. So the page also HEARTBEATS, and the server
    stops when it stops hearing. Without this, closing the tab left the process
    listening forever: found by the experimenter, with a port still open and a terminal
    still running after the editor was visibly gone.

    ``idle_timeout_s`` is generous ON PURPOSE. Browsers throttle `setInterval` in a
    BACKGROUND tab to about once a minute, so a timeout under ~60 s would kill the
    editor of anyone who switched windows to look at something. Pass ``None`` to
    disable the timeout entirely (the beacon still works).
    """
    port = free_port() if port is None else int(port)
    token = make_token() if token is None else token
    url = f"http://127.0.0.1:{port}/?token={token}"
    if root:
        from urllib.parse import quote
        url += f"&root={quote(str(root))}"

    from werkzeug.serving import make_server

    app = build_app(token)
    app.config["last_seen"] = time.monotonic()
    # `make_server`, NOT `app.run`: Werkzeug 2.1 REMOVED the old
    # `werkzeug.server.shutdown` environ hook, so a route can no longer stop the
    # server from inside a request. A server object can be shut down from another
    # thread, which is what the watchdog needs.
    server = make_server("127.0.0.1", port, app, threaded=True)

    stop = threading.Event()

    def watchdog():
        # A GRACE PERIOD, because the first heartbeat cannot arrive before the browser
        # has opened, and on a cold start that is seconds.
        while not stop.wait(HEARTBEAT_CHECK_S):
            last = app.config.get("last_seen", 0.0)
            if last == 0.0:                       # the page said goodbye
                break
            if idle_timeout_s is not None and time.monotonic() - last > idle_timeout_s:
                print("no page has checked in for "
                      f"{idle_timeout_s:.0f} s: stopping the mesh editor.")
                break
        server.shutdown()

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"Mesh editor -> {url}")
    print("It stops on its own when you close the tab. Ctrl+C also works.")
    try:
        server.serve_forever()
    finally:
        stop.set()
    print("Mesh editor stopped.")
    return url


def main(argv=None) -> int:
    """``python -m smtracker.segmentation.editor [--root FOLDER]``.

    NO ``pip install`` HERE. The original shelled out to pip when Flask was missing,
    which a frozen application cannot do and a managed environment must not have done
    to it. A missing dependency is reported, with the command to fix it.
    """
    import argparse

    parser = argparse.ArgumentParser(description="SMTracker mesh editor")
    parser.add_argument("--root", help="folder to scan for phase/ + cell_meshes/")
    parser.add_argument("--port", type=int, help="fixed port (default: a free one)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    try:
        import flask  # noqa: F401
    except ImportError:
        print("The mesh editor needs Flask, which is not installed.\n"
              "    pip install flask\n"
              "Nothing else in SMTracker requires it.")
        return 2
    serve(port=args.port, root=args.root, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
