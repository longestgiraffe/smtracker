"""The write guard, the title guard, and the vocabulary gate for legacy mode.

The guard is structural. A legacy function cannot write outside ``legacy_out/``, because
every path it uses comes from ``legacy_path``, and that function raises on any path that
resolves outside the legacy root. A test plants a violation and asserts that the guard
fires, so the guard is known to work and is not merely present.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["LegacyGuardError", "LEGACY_ROOT", "legacy_path", "assert_inside_legacy",
           "legacy_title", "FORBIDDEN_WORDS", "assert_neutral_text", "assert_neutral"]

#: The one output directory. Nothing in legacy mode writes anywhere else.
LEGACY_ROOT = Path("legacy_out")

#: Words that must not appear in rendered text. Rename; do not add an exception.
FORBIDDEN_WORDS = ("bound", "free", "slow", "fast", "mobile", "immobile", "binding",
                   "diffusing", "residence", "dwell", "reside")

#: Directories a legacy write must never touch, whatever else is true.
PROTECTED = ("msd_out", "zenodo")


class LegacyGuardError(RuntimeError):
    """Raised when legacy mode tries to write outside its own directory, or when a
    rendered string breaks the marking rule or the vocabulary rule."""


def assert_inside_legacy(path, *, root=None) -> Path:
    """Return ``path`` resolved, after proving it is inside the legacy root.

    The check is on the RESOLVED path, so ``legacy_out/../msd_out`` is caught. It is not
    a string test on the prefix: ``legacy_out_other/`` must not pass because its name
    starts with the root's name, so the comparison walks the parent parts.
    """
    base = Path(LEGACY_ROOT if root is None else root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = (Path.cwd() / target)
    target = Path(target).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise LegacyGuardError(
            f"legacy mode tried to write to {target}, which is outside {base}. Legacy "
            f"output goes to {LEGACY_ROOT}/ only, so a legacy number can never reach "
            f"the manuscript deposit by accident.") from None
    for protected in PROTECTED:
        if protected in target.parts:
            raise LegacyGuardError(
                f"legacy mode tried to write to {target}, which contains the protected "
                f"directory {protected!r}. That directory holds the new analysis.")
    return target


def legacy_path(*parts, root=None, make_parent: bool = True) -> Path:
    """Build a path under ``legacy_out/`` and prove it is inside.

    This is the only way legacy mode makes a path. Every write in this package calls it.
    The file name must carry the word ``legacy`` (rule 3), so a file cannot be mistaken
    for new output after it leaves the directory.
    """
    base = Path(LEGACY_ROOT if root is None else root)
    target = base.joinpath(*[str(p) for p in parts])
    checked = assert_inside_legacy(target, root=base)
    name = checked.name
    if name and checked.suffix and "legacy" not in name.lower():
        raise LegacyGuardError(
            f"the file name {name!r} does not carry the word 'legacy'. Every legacy "
            "output is marked in its name, so it stays identifiable once it is copied "
            "out of the directory.")
    if make_parent:
        checked.parent.mkdir(parents=True, exist_ok=True)
    return checked


def assert_neutral_text(text: str, *, what: str = "text") -> str:
    """Refuse a rendered string that carries interpretive vocabulary."""
    lowered = str(text).lower()
    hits = [w for w in FORBIDDEN_WORDS if w in lowered]
    if hits:
        raise LegacyGuardError(
            f"{what} uses interpretive vocabulary {hits}: {text!r}. Use low-D and "
            "high-D, shell and interior, short step and long step. Rename; do not add "
            "an exception.")
    if "—" in str(text) or "–" in str(text):
        raise LegacyGuardError(f"{what} contains a dash character: {text!r}")
    return str(text)


def assert_neutral(fig) -> None:
    """The FIGURE gate for legacy output. Same contract as the nine gates already in the
    project: it walks every text surface a figure has, it proves it can detect a planted
    violation before it trusts a pass, and it RAISES rather than warning.

    A legacy figure is still a rendered figure, so it is held to the same vocabulary
    rule as every other figure here. Reproducing the old METHOD does not license
    reproducing old labels.

    Three title artists are read, not one. ``ax.title`` is only the centred title; a
    ``loc="left"`` or ``loc="right"`` title is a separate artist and is in neither
    ``ax.title`` nor ``ax.texts``. Every panel title in this project is left-aligned.
    """
    import matplotlib.pyplot as plt

    def hits(f):
        out = []
        for ax in f.axes:
            arts = [ax.xaxis.label, ax.yaxis.label]
            arts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
            arts += list(ax.texts)
            for loc in ("left", "center", "right"):
                for w in FORBIDDEN_WORDS:
                    if w in ax.get_title(loc=loc).lower():
                        out.append((w, ax.get_title(loc=loc)))
            leg = ax.get_legend()
            if leg is not None:
                arts += list(leg.get_texts())
                if leg.get_title() is not None:
                    arts.append(leg.get_title())
            for a in arts:
                for w in FORBIDDEN_WORDS:
                    if w in a.get_text().lower():
                        out.append((w, a.get_text()))
        for a in f.texts:
            for w in FORBIDDEN_WORDS:
                if w in a.get_text().lower():
                    out.append((w, a.get_text()))
        return out

    for planted in ("the bound fraction", "distance to the shell boundary"):
        probe = plt.figure()
        probe.add_subplot(111).set_title(planted, loc="left")
        caught = hits(probe)
        plt.close(probe)
        if not caught:
            raise AssertionError(f"assert_neutral cannot detect {planted!r}")
    bad = hits(fig)
    if bad:
        raise AssertionError("legacy figure uses interpretive vocabulary: "
                             + "; ".join(f"{w!r} in {t!r}" for w, t in bad))


def legacy_title(text: str) -> str:
    """Build a figure title that is marked as legacy and passes the vocabulary gate."""
    marked = text if "legacy" in text.lower() else f"legacy: {text}"
    return assert_neutral_text(marked, what="figure title")
