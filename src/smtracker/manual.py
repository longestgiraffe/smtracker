"""The user manual, as a page a window's Help button opens (headless).

ONE SOURCE: `docs/MANUAL.md`, written for the people who run the experiments. A source
checkout reads it there; the installer ships it as `smtracker/resources/manual/MANUAL.md`
(`packaging/build.py`). `write_pages` renders it ONCE per call into a folder of the user's
(`manual_folder`), one page per stage that opens at that stage's chapter -- a page per
chapter rather than a `#fragment`, because Windows drops the fragment when it hands a
`file:` URL to the default browser.

Rendered with `markdown_it` (CommonMark plus tables), every heading given an `id` from its
text, the page styled for light and dark (the reader's own setting decides). No Qt here: the
windows call `open_manual(stage)` in `ui.help`, which only opens the path this returns.

TWO WAYS TO READ IT (ruled 27 Nov 2026: "shouldnt help the manual open the website manual? or at least ask which
manual, website or local. make sure both are same"): each Help button is a small menu, `help_choices(stage)`: the
manual on the website (`website_url`, the page of that stage under `SITE_URL`, which `scripts/manual_from_site.py`
reads too) and the manual in SMTracker (this offline copy). Sameness is CHECKED, not hoped: the copy is generated
from the site, says which website version it is, and `tests/test_manual_matches_site.py` and the build's verify
compare it with a fresh copy wherever the site's checkout is found (`site_checkout`).

Fails loud: a manual that cannot be found raises `ManualError` naming where it looked, so
a packaging mistake says so at the first click instead of opening an empty page.
"""
from __future__ import annotations

import base64
import html
import os
import re
import sys
from pathlib import Path

__all__ = ["ManualError", "STAGE_HEADINGS", "SITE_URL", "STAGE_ROUTES", "HELP_CHOICES", "manual_source",
           "manual_folder", "render", "headings", "anchor_for", "write_pages", "website_url", "help_choices",
           "site_checkout"]


class ManualError(RuntimeError):
    """The manual could not be found or written."""


#: Which chapter each window's Help opens: the first level-2 heading containing these words.
STAGE_HEADINGS = {"start": None, "prep": "Image preparation", "outlines": "Cell outlines",
                  "analysis": "Analysis"}


#: Where the website is published (GitHub Pages: https://<owner>.github.io/<repository>/; its workflow
#: `.github/workflows/pages.yml`, `next.config.mjs` with trailingSlash).
SITE_URL = "https://longestgiraffe.github.io/smtracker-site"
#: Each stage's chapter on the website: its folder's index page (`(group)` folders are not in an address).
STAGE_ROUTES = {"start": "/docs", "prep": "/docs/tracking", "outlines": "/docs/cell-outlines",
                "analysis": "/docs/analysis"}
#: The two ways to read the manual and the update check, in the order a Help menu lists them: ``(key, label, tip)``.
HELP_CHOICES = (("website", "Manual on the website", "Opens the newest manual in your web browser (needs a network)."),
                ("local", "Manual in SMTracker", "Opens the copy that came with this program (no network needed)."),
                ("updates", "Check for updates", "Asks GitHub whether a newer SMTracker is published, now."))


def website_url(stage: str) -> str:
    """The website's page for ``stage``'s chapter, ending in a slash as the site serves it."""
    if stage not in STAGE_ROUTES:
        raise ManualError(f"no manual chapter for {stage!r}")
    return SITE_URL + STAGE_ROUTES[stage] + "/"


def help_choices(stage: str) -> list:
    """What ``stage``'s Help menu offers: ``[(key, label, tip, target)]``; ``target`` is the website address for
    "website" and None for "local" (the window opens its own copy)."""
    return [(key, label, tip, website_url(stage) if key == "website" else None)
            for key, label, tip in HELP_CHOICES]


def site_checkout() -> Path | None:
    """The website's checkout where one is at hand (``SMTRACKER_SITE``, else ``D:/smtracker-site``), else None."""
    for raw in (os.environ.get("SMTRACKER_SITE"), "D:/smtracker-site"):
        if raw and (Path(raw) / "content" / "docs" / "meta.json").is_file():
            return Path(raw)
    return None


def manual_source() -> Path:
    """`MANUAL.md`: the bundled copy in a frozen build, else the repository's `docs/`."""
    here = Path(__file__).resolve().parent
    candidates = [here / "resources" / "manual" / "MANUAL.md",
                  here.parent.parent / "docs" / "MANUAL.md"]
    if getattr(sys, "_MEIPASS", None):
        candidates.insert(0, Path(sys._MEIPASS) / "smtracker" / "resources" / "manual" / "MANUAL.md")
    for path in candidates:
        if path.is_file():
            return path
    raise ManualError("the manual is not where it should be: looked in "
                      + ", ".join(str(p) for p in candidates))


def manual_folder() -> Path:
    """Where the rendered pages go: the user's local application data, never the data drives."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "SMTracker" / "manual"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render(markdown_text: str) -> str:
    """The manual's body as HTML: CommonMark with tables, every heading with an id."""
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(markdown_text)
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            token.attrSet("id", _slug(tokens[i + 1].content))
    return md.renderer.render(tokens, md.options, {})


def headings(markdown_text: str) -> list:
    """``[(level, text, id)]`` of every heading, in order."""
    out = []
    for line in markdown_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            text = m.group(2).strip()
            out.append((len(m.group(1)), text, _slug(text)))
    return out


def anchor_for(stage: str, markdown_text: str) -> str | None:
    """The id of ``stage``'s chapter (`STAGE_HEADINGS`), None for the top of the manual."""
    if stage not in STAGE_HEADINGS:
        raise ManualError(f"no manual chapter for {stage!r}")
    words = STAGE_HEADINGS[stage]
    if words is None:
        return None
    for level, text, slug in headings(markdown_text):
        if level == 2 and words.lower() in text.lower():
            return slug
    raise ManualError(f"the manual has no chapter on {words!r}")


_STYLE = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --rule:#d1d9e0; --code:#f6f8fa; --accent:#0969da; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --rule:#3d444d; --code:#151b23; --accent:#4493f8; }
}
html { background:var(--bg); }
body { background:var(--bg); color:var(--fg); font:16px/1.6 "Inter","Segoe UI",system-ui,sans-serif;
       max-width:72ch; margin:0 auto; padding:32px 16px 96px; }
h1 { font-size:1.9rem; line-height:1.2; margin:0 0 .6em; }
h2 { font-size:1.35rem; margin:2.2em 0 .6em; padding-top:.6em; border-top:1px solid var(--rule); }
h3 { font-size:1.08rem; margin:1.8em 0 .4em; }
h1, h2, h3 { text-wrap:balance; }
p, li { color:var(--fg); }
a { color:var(--accent); }
hr { border:none; border-top:1px solid var(--rule); margin:2em 0; }
code { background:var(--code); padding:.1em .35em; border-radius:4px; font-size:.9em; }
table { border-collapse:collapse; margin:1em 0; font-size:.92rem; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--rule); padding:.45em .7em; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; }
strong { font-weight:650; }
.masthead { margin:0 0 28px; }
.masthead img { display:block; height:32px; width:auto; }
"""


def _data_uri(path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _branding() -> tuple:
    """``(masthead, favicon)``: the wordmark over the page, in the reader's theme, and the icon for the browser's
    tab, both inlined so a page needs no file beside it; empty where the files did not ship."""
    from .appicon import ICON_STEM, logo_path

    light, dark = logo_path(False), logo_path(True)
    masthead = ""
    if light is not None:
        source = (f'<source media="(prefers-color-scheme: dark)" srcset="{_data_uri(dark)}">'
                  if dark is not None else "")
        masthead = (f'<header class="masthead"><picture>{source}'
                    f'<img src="{_data_uri(light)}" alt="SMTracker" height="32"></picture></header>')
    tab = Path(__file__).resolve().parent / "resources" / f"{ICON_STEM}-32.png"
    favicon = f'<link rel="icon" type="image/png" href="{_data_uri(tab)}">' if tab.is_file() else ""
    return masthead, favicon


def write_pages(folder: Path | None = None) -> dict:
    """Render the manual into ``folder`` (default `manual_folder`): ``{stage: page path}``, one
    page per `STAGE_HEADINGS` stage, each opening at its chapter."""
    source = manual_source()
    text = source.read_text(encoding="utf-8")
    body = render(text)
    title = html.escape(headings(text)[0][1] if headings(text) else "SMTracker manual")
    folder = Path(folder) if folder is not None else manual_folder()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ManualError(f"cannot write the manual into {folder}: {exc}") from exc
    masthead, favicon = _branding()
    pages = {}
    for stage in STAGE_HEADINGS:
        anchor = anchor_for(stage, text)
        jump = ("" if anchor is None else
                f"<script>window.addEventListener('load',function(){{var e=document.getElementById("
                f"'{anchor}');if(e)e.scrollIntoView();}});</script>")
        page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                f"<title>{title}</title>{favicon}<style>{_STYLE}</style>{jump}</head>"
                f"<body>{masthead}{body}</body></html>")
        path = folder / f"manual-{stage}.html"
        path.write_text(page, encoding="utf-8")
        pages[stage] = path
    return pages
