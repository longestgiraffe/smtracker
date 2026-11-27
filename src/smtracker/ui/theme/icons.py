"""Lucide icons, drawn INTO A PAINTER -- never into a pixmap.

shadcn's icon set IS lucide, so these are lucide's own ``d`` attributes rather
than shapes drawn by hand: the same strings the Paper artboard uses, which is
what lets the two be compared without arguing about whether a glyph differs.

**Why :func:`paint` and not a QPixmap.** The first version of this module built
a pixmap at an assumed device-pixel-ratio, stamped that ratio onto it, and
handed it to a ``QLabel``. That is three separate places to get the ratio wrong
plus a resample, and it went wrong in all of them: glyphs came out 1.6x
oversized on a 125 % screen, and ``setScaledContents`` -- added to stop them
overflowing -- then softened and fattened every stroke.

The fix was to stop making pixmaps at all. :func:`paint` draws straight into the
live painter of whatever widget wants the glyph, so there is no ratio to assume:
the painter already carries the screen's transform, strokes land at the correct
subpixel weight, and Qt antialiases them natively. The geometry arrives as SVG
and ``QSvgRenderer`` renders into a painter as happily as into a device.

**One box, one weight**, borrowed from the same place: the stroke is expressed
in viewBox units, so it scales with the glyph and a 10 px icon has the same
apparent weight as a 20 px one. Getting that wrong is the usual reason an icon
set looks mixed.

**Recoloured by string substitution.** Lucide strokes inherit ``currentColor``
in the browser; Qt's SVG renderer has no cascade to inherit from, so the colour
goes into the document before parsing -- there is no point at which a pen could
reach the stroke.
"""

from __future__ import annotations

import functools

from qtpy.QtCore import QByteArray, QRectF, Qt
from qtpy.QtGui import QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer

__all__ = ["PATHS", "pixmap"]

#: name -> the inner markup of lucide's 24x24 viewBox.
PATHS = {
    # -- the right-hand tool rail and its popovers (see `shadcn.toolbar`).
    #
    # `sliders-horizontal` for Visualization, `layers` for Visibility,
    # `flask-conical` for Conditions and `table` for the data tables -- four
    # nouns, in lucide's drawing of them.
    # `eye` / `eye-off` are the per-row state inside the
    # Visibility and Conditions popovers: one thing's on/off, where the rail's
    # `layers` is the SET of them.
    "sliders-horizontal": '<line x1="21" x2="14" y1="4" y2="4"/>'
                          '<line x1="10" x2="3" y1="4" y2="4"/>'
                          '<line x1="21" x2="12" y1="12" y2="12"/>'
                          '<line x1="8" x2="3" y1="12" y2="12"/>'
                          '<line x1="21" x2="16" y1="20" y2="20"/>'
                          '<line x1="12" x2="3" y1="20" y2="20"/>'
                          '<line x1="14" x2="14" y1="2" y2="6"/>'
                          '<line x1="8" x2="8" y1="10" y2="14"/>'
                          '<line x1="16" x2="16" y1="18" y2="22"/>',
    "flask-conical": '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18'
                     ' 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/>'
                     '<path d="M6.453 15h11.094"/><path d="M8.5 2h7"/>',
    "table": '<path d="M12 3v18"/>'
             '<rect width="18" height="18" x="3" y="3" rx="2"/>'
             '<path d="M3 9h18"/><path d="M3 15h18"/>',
    "eye": '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0'
           ' 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/>'
           '<circle cx="12" cy="12" r="3"/>',
    "eye-off": '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0'
               ' 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/>'
               '<path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/>'
               '<path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0'
               ' 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/>'
               '<path d="m2 2 20 20"/>',
    "rows-3": '<rect width="18" height="18" x="3" y="3" rx="2"/>'
              '<path d="M3 9h18"/><path d="M3 15h18"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
            '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
            '<circle cx="12" cy="12" r="3"/>',
    "scan-search": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
                   '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
                   '<circle cx="12" cy="12" r="3"/><path d="m16 16-1.9-1.9"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "grid-2x2": '<rect width="18" height="18" x="3" y="3" rx="2"/>'
                '<path d="M3 12h18"/><path d="M12 3v18"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "aperture": '<circle cx="12" cy="12" r="10"/><path d="m14.31 8 5.74 9.94"/>'
                '<path d="M9.69 8h11.48"/><path d="m7.38 12 5.74-9.94"/>'
                '<path d="M9.69 16 3.95 6.06"/><path d="M14.31 16H2.83"/>'
                '<path d="m16.62 12-5.74 9.94"/>',
    "crosshair": '<line x1="12" x2="12" y1="2" y2="6"/><line x1="12" x2="12" y1="18" y2="22"/>'
                 '<line x1="2" x2="6" y1="12" y2="12"/><line x1="18" x2="22" y1="12" y2="12"/>'
                 '<circle cx="12" cy="12" r="7"/>',
    "ruler": '<rect width="20" height="10" x="2" y="7" rx="2"/><path d="M6 7v3"/>'
             '<path d="M10 7v5"/><path d="M14 7v3"/><path d="M18 7v5"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/>'
                '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    # ``columns-3`` already belongs to the Analysis groups section header, so
    # the tab that OPENS that section cannot reuse it -- an icon that means two
    # things at once on one screen means neither.
    "layers": '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83'
              'l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>'
              '<path d="m6.08 9.5-3.48 1.59a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 '
              '0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83L17.92 9.5"/>'
              '<path d="m6.08 14.5-3.48 1.59a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 '
              '0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83l-3.48-1.59"/>',
    "columns-3": '<rect width="18" height="18" x="3" y="3" rx="2"/>'
                 '<path d="M9 3v18"/><path d="M15 3v18"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 16v-5"/><path d="M12 16V8"/>'
             '<path d="M17 16v-3"/>',
    # lucide "chart-spline": a curve on axes -- the MSD section's glyph (a rising curve).
    "chart-spline": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/>'
                    '<path d="M7 16c.5-2 1.5-7 4-7 2 0 2 3 4 3 2.5 0 4.5-5 5-7"/>',
    "rotate": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
              '<path d="M3 3v5h5"/>',
    # lucide "history": the legacy window's row (SMTracker 2.0's tabs, 26 Nov 2026)
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
               '<path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "grid": '<rect width="7" height="7" x="3" y="3" rx="1"/>'
            '<rect width="7" height="7" x="14" y="3" rx="1"/>'
            '<rect width="7" height="7" x="14" y="14" rx="1"/>'
            '<rect width="7" height="7" x="3" y="14" rx="1"/>',
    "list": '<line x1="3" x2="21" y1="6" y2="6"/><line x1="3" x2="21" y1="12" y2="12"/>'
            '<line x1="3" x2="21" y1="18" y2="18"/>',
    "grip": '<circle cx="9" cy="6" r="1"/><circle cx="9" cy="12" r="1"/>'
            '<circle cx="9" cy="18" r="1"/><circle cx="15" cy="6" r="1"/>'
            '<circle cx="15" cy="12" r="1"/><circle cx="15" cy="18" r="1"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "folder-open": '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/>'
            '<path d="M12 8h.01"/>',
    # A tip beside a chart's title (the Spatial Map Test page's: how to select a spot, how
    # to move it). lucide's `lightbulb`.
    "lightbulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0'
                 ' 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/>'
                 '<path d="M10 22h4"/>',
    # The "Record" tab. `info` was here first and did not survive 16 px: its
    # meaning lives in a 4-unit bar and a dot inside a 20-unit circle, so at a
    # 16 px box only 20 % of its ink reaches full opacity -- the WORST of ten
    # glyphs measured, against 38 % for this one. Everything that carries the
    # sense of it falls under a device pixel and turns to grey mush, which is
    # exactly what "doesn't look good in small size" is. A document reads at
    # 16 px because its OUTLINE is the meaning, and the outline is the whole
    # box.
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 '
                 '0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
                 '<path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    # The Metadata tool: Lucide's receipt-text. NOT file-text, which already names the
    # Record tab -- one glyph meaning two things on one screen means neither.
    "receipt-text": '<path d="M4 2v20l2-1 2 1 2-1 2 1 2-1 2 1 2-1 2 1V2l-2 1-2-1-2 1-2-1'
                    '-2 1-2-1-2 1Z"/><path d="M14 8H8"/><path d="M16 12H8"/>'
                    '<path d="M13 16H8"/>',
    # The light/dark toggle: the glyph names the theme a click SWITCHES TO.
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/>'
           '<path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/>'
           '<path d="M2 12h2"/><path d="M20 12h2"/>'
           '<path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    # lucide "git-fork": one set splitting into branches -- the Clustering section's glyph.
    "git-fork": '<circle cx="12" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/>'
                '<path d="M18 9v2c0 .6-.4 1-1 1H7c-.6 0-1-.4-1-1V9"/><path d="M12 12v3"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "arrow-left-right": '<path d="M8 3 4 7l4 4"/><path d="M4 7h16"/>'
                        '<path d="m16 21 4-4-4-4"/><path d="M20 17H4"/>',
    # lucide "shuffle": QC's "Do the replicates differ?" card (movies dealt out at random)
    "shuffle": '<path d="m18 14 4 4-4 4"/><path d="m18 2 4 4-4 4"/>'
               '<path d="M2 18h1.973a4 4 0 0 0 3.3-1.7l5.454-7.6a4 4 0 0 1 3.3-1.7H22"/>'
               '<path d="M2 6h1.972a4 4 0 0 1 3.6 2.2"/><path d="M22 18h-6.041a4 4 0 0 1-3.3-1.8l-.359-.45"/>',
    "minus": '<path d="M5 12h14"/>',
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "chevron-up": '<path d="m18 15-6-6-6 6"/>',
    # lucide "pin": the HMM Test page's cards pin with it
    "pin": ('<path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 '
            '15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 '
            '15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/>'),
    # lucide "image": the Images & tracks section.
    "image": '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>'
             '<circle cx="9" cy="9" r="2"/>'
             '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    # lucide "panels-top-left": one frame cut into panels -- the Composer section. Not
    # "layout-dashboard": four rounded squares read as the Spatial Map's "grid" in the rail.
    "panels-top-left": '<rect width="18" height="18" x="3" y="3" rx="2"/>'
                       '<path d="M3 9h18"/><path d="M9 21V9"/>',
    # lucide "sparkles": the rail's Test button, the redesigned page of HMM and the Spatial Map (4 Nov 2026). It
    # wore "panels-top-left", the Composer's own glyph in the sidebar, so on HMM it read as a way to the Composer.
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 '
                '9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 '
                '0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>'
                '<path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    #: lucide `chevrons-up-down` -- the switcher's affordance. NOT a plain
    #: chevron: this one says the control cycles through a SET, which is
    #: exactly what picking a workspace does.
    "chevrons-up-down": '<path d="m7 15 5 5 5-5"/><path d="m7 9 5-5 5 5"/>',
    #: lucide `panel-left`, the sidebar trigger's glyph in the registry's own
    #: block. The filled-looking left column is a second rect, not a fill.
    "panel-left": '<rect width="18" height="18" x="3" y="3" rx="2"/>'
                  '<path d="M9 3v18"/>',
    # -- the Compute mark's three states (see `toolbar.ComputeMark`).
    #
    # `play` / `loader-circle` / `square`, and they are ONE control's states
    # rather than three controls, which is why they share a size and a box.
    # `loader-circle` is lucide's `Loader2Icon` -- the glyph shadcn's own
    # `Spinner` spins, so the rotation here is the registry's `animate-spin`
    # and not a shape invented for it. Its 3/4 arc is what makes the turn
    # legible: a full ring rotating is indistinguishable from a still one.
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "loader-circle": '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
    "square": '<rect width="18" height="18" x="3" y="3" rx="2"/>',
    # lucide "square-check" and "asterisk": a checklist row (the Significance tool's pairs) and the Significance tool
    # itself (4 Nov 2026).
    "square-check": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="m9 12 2 2 4-4"/>',
    "asterisk": '<path d="M12 6v12"/><path d="M17.196 9 6.804 15"/><path d="m6.804 9 10.392 6"/>',
    "pause": '<rect x="14" y="4" width="4" height="16" rx="1"/>'
             '<rect x="6" y="4" width="4" height="16" rx="1"/>',
    # -- the mesh editor's tools (`ui.outlines.editor_page`): lucide's own drawings,
    # because the classic window's emoji and box characters are not in Inter, and a
    # glyph missing from Inter sends Qt through every installed font on the GUI thread.
    "mouse-pointer": '<path d="m4 4 7.07 17 2.51-7.39L21 11.07z"/>',
    "pencil": '<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83'
              'l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/>'
              '<path d="m15 5 4 4"/>',
    "scissors": '<circle cx="6" cy="6" r="3"/><path d="M8.12 8.12 12 12"/>'
                '<path d="M20 4 8.12 15.88"/><circle cx="6" cy="18" r="3"/>'
                '<path d="M14.8 14.8 20 20"/>',
    "merge": '<path d="m8 6 4-4 4 4"/><path d="M12 2v10.3a4 4 0 0 1-1.172 2.872L4 22"/>'
             '<path d="m20 22-5-5"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>'
             '<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
    "spline": '<circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/>'
              '<path d="M5 17A12 12 0 0 1 17 5"/>',
    "undo": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5'
            'a5.5 5.5 0 0 1-5.5 5.5H11"/>',
    "redo": '<path d="m15 14 5-5-5-5"/><path d="M20 9H9.5A5.5 5.5 0 0 0 4 14.5'
            'A5.5 5.5 0 0 0 9.5 20H13"/>',
    "save": '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5'
            'a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8'
            'a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
    "zoom-in": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>'
               '<path d="M11 8v6"/><path d="M8 11h6"/>',
    "zoom-out": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>'
                '<path d="M8 11h6"/>',
    "maximize": '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>'
                '<path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "chevron-left": '<path d="m15 18-6-6 6-6"/>',
    # lucide "download": export a figure.
    "download": '<path d="M12 15V3"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
                '<path d="m7 10 5 5 5-5"/>',
    # lucide "copy": copy an image to the clipboard, and the Metadata receipt's Copy.
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
            '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    # lucide "expand" / "shrink": the outline editor's focus mode, in and out.
    "expand": '<path d="m15 15 6 6"/><path d="m15 9 6-6"/><path d="M21 16v5h-5"/>'
              '<path d="M21 8V3h-5"/><path d="M3 16v5h5"/><path d="m3 21 6-6"/>'
              '<path d="M3 8V3h5"/><path d="M9 9 3 3"/>',
    "shrink": '<path d="m15 15 6 6m-6-6v4.8m0-4.8h4.8"/><path d="M9 19.8V15m0 0H4.2M9 15l-6 6"/>'
              '<path d="M15 4.2V9m0 0h4.8M15 9l6-6"/><path d="M9 4.2V9m0 0H4.2M9 9 3 3"/>',
    "keyboard": '<path d="M10 8h.01"/><path d="M12 12h.01"/><path d="M14 8h.01"/>'
                '<path d="M16 12h.01"/><path d="M18 8h.01"/><path d="M6 8h.01"/>'
                '<path d="M7 16h10"/><path d="M8 12h.01"/>'
                '<rect width="20" height="16" x="2" y="4" rx="2"/>',
}

#: Lucide's default stroke is 2.0 at a 24 viewBox. MEASURED: at that weight the
#: rendered ink fills 87-100 % of the icon box (the stroke extends a full unit
#: outside each path), which against Inter's 10.17 px cap height at 14 px makes
#: a 15 px icon 1.5x the height of the text beside it.
#:
#: A browser gets away with 2.0 because it lays the stroke down with subpixel
#: antialiasing and it reads as a hairline. Qt rasterises the same stroke onto
#: a small pixmap and it reads as a line -- so the identical numbers that looked
#: right in the Paper artboard came out heavy on screen.
#:
#: **These values were CHOSEN BY EYE against a calibration strip, not derived.**
#: Matching the artboard's own 15 px / stroke 2 reproduced its exact
#: icon-to-cap ratio of 1.42x, and that ratio was still too heavy on a real
#: 1.25-ratio screen -- twice. Which settles something worth writing down: the
#: web spec is not the target here, because the same numbers do not render the
#: same way. A strip of options at the real device ratio decided it in one pass
#: where two rounds of reasoning from the spec had not.
STROKE = 1.25

#: Icon sizes, named so there is ONE place to retune them. ``MD`` sits beside
#: 14 px text, ``SM`` beside 12 px, ``XS`` is the drag handle.
#:
#: Settled at 13/12/11 only AFTER the painter fix. The sizes chosen before it
#: (15, then 13, then 11) were all compensating for a pipeline that could not
#: draw a correct glyph at any size, so none of those judgements carried over
#: -- 11 px looked right against fattened strokes and too faint against clean
#: ones. A size chosen to cancel out a rendering bug is worth nothing once the
#: bug is gone.
MD, SM, XS = 13, 12, 11

#: ``size-4``, the registry's own icon size. Reserved for an ICON-ONLY control,
#: where the glyph IS the label and has to carry a word's weight.
LG = 16

#: One notch under ``size-4``, for a glyph that sits BESIDE text in a control.
#:
#: The registry uses 16 for both cases. Taking it literally in a button put a
#: 16 px glyph next to 14 px text and the pair read a size too large for this
#: screen -- a judgement, made by looking, and the honest label for it is a
#: deviation rather than a correction. What is NOT a judgement is that a
#: button's icon and its type move together: 13 px text with a 16 px glyph is
#: the mismatch that made the icon look bolted on.
CONTROL = 14

_DOC = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="{colour}" stroke-width="{stroke}" '
        'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


#: How many (glyph, colour, stroke) renderers to keep. The whole set this
#: interface can ask for is bounded and small -- about forty glyphs across a
#: handful of ink colours -- so this never evicts in practice; it is a ceiling
#: against a caller that generates colours rather than a working limit.
_RENDERER_CACHE = 512


@functools.lru_cache(maxsize=_RENDERER_CACHE)
def _renderer(name: str, colour: str, stroke: float | None) -> QSvgRenderer:
    """A renderer for one glyph in one colour. Raises on an unknown name.

    Raising rather than drawing nothing: a typo'd glyph renders as a blank
    square that looks like a layout bug, and then gets chased in the layout.

    CACHED, because this is called from `paint` and `paint` is called from a
    `paintEvent`. Uncached, every repaint re-formatted an SVG document and built
    a fresh `QSvgRenderer` to parse it -- and the Confirm screen carries about a
    hundred glyphs, so a single full repaint of it parsed a hundred documents.
    A renderer is immutable once built and `render` takes the painter, so one
    instance serves every widget that wants that glyph in that colour.
    """
    if name not in PATHS:
        raise KeyError(f"no lucide path for {name!r}; add it to PATHS")
    doc = _DOC.format(colour=colour, body=PATHS[name],
                      stroke=STROKE if stroke is None else stroke)
    return QSvgRenderer(QByteArray(doc.encode()))


def paint(p: QPainter, name: str, rect: QRectF, colour: str,
          stroke: float | None = None) -> None:
    """Draw ``name`` centred and square in ``rect``, straight into ``p``.

    The one way icons should reach the screen -- see the module docstring.
    """
    side = min(rect.width(), rect.height())
    box = QRectF(rect.center().x() - side / 2, rect.center().y() - side / 2,
                 side, side)
    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    _renderer(name, colour, stroke).render(p, box)
    p.restore()


def pixmap(name: str, px: int, colour: str, ratio: float = 1.0,
           stroke: float | None = None) -> QPixmap:
    """A glyph as a standalone pixmap. For proof sheets and tests ONLY.

    Kept because a contact sheet has to become a file, but deliberately off the
    path any widget takes: this is the shape that made the icons wrong.
    """
    out = QPixmap(round(px * ratio), round(px * ratio))
    out.setDevicePixelRatio(ratio)
    out.fill(Qt.transparent)
    painter = QPainter(out)
    paint(painter, name, QRectF(0, 0, px, px), colour, stroke)
    painter.end()
    return out
