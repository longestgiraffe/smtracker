"""Inter on Qt — the scale, and the family trap it turned out not to fall into.

shadcn's type scale is just Tailwind's, so these are Tailwind's ``text-*``
values verbatim. There is no token payload to resolve (see ``tokens``), so the
table is literal.

**THE FACE IS INTER, WHICH IS A DEPARTURE FROM shadcn**, whose own default is
Geist. It is the second deliberate one here (the ordinal day ramp in ``tokens``
is the first) and it was made for a measured reason rather than a visual one:
Geist's coverage gaps cost multi-second freezes on this stack. The number, the
mechanism and the superseded claim are recorded on :data:`FAMILY`.

**Inter registers its weights as separate families**, which is the shape that
lets a font resolve to something else entirely without ever raising::

    Inter, Inter Thin, Inter Light, Inter Medium, Inter SemiBold, Inter Black, ...

That is the shape that breaks ``QFont(family) + setWeight()``, so it was
MEASURED before any of this was written rather than assumed either way::

    request         resolved family   weight   advance of "Confirm project" @24px
    Inter @400      Inter                400   177
    Inter @500      Inter                500   179
    Inter @600      Inter                600   182
    Inter @700      Inter                700   184

Four distinct advances and four correct resolved weights: Qt DOES synthesise
the family group for Inter, so the plain ``QFont("Inter")`` route is safe here
and no per-weight family names are needed. :func:`report` re-checks it at
startup anyway, because "it worked on this machine" is not the same claim as
"it works", and a wrong answer shows up as text that is subtly too light
rather than as an error.

The bundled build carries an ``opsz`` axis as well as ``wght``. Nothing here
sets it, so every face is the axis default; ``wght`` is the only axis this
module drives (see ``ui(wght=…)``).

**Pixels, not points**: Qt's logical pixel is
the CSS px Tailwind specifies, and ``setPointSizeF`` would quietly inflate
every size by 4/3 at 96 dpi.
"""

from __future__ import annotations

import pathlib

from qtpy.QtGui import QFont, QFontDatabase, QFontInfo, QFontMetrics, QGuiApplication

__all__ = ["FAMILY", "FEATURES", "MONO_FAMILY", "RECEIPT_FAMILY", "SIZES", "USED_STEPS",
           "WEIGHTS",
           "feature_effect", "mono", "receipt", "report", "set_features", "ui", "value"]

#: Inter (Rasmus Andersson's face, SIL OFL). The fallbacks are the platform UI
#: stacks, so a machine without Inter still renders a sane screen — and
#: :func:`report` names what actually won.
#:
#: **A MISSING GLYPH IS A PERFORMANCE DEFECT HERE, NOT A COSMETIC ONE**, and
#: that is why this family changed. When the face lacks a character, Qt asks the
#: platform for a fallback by WALKING THE INSTALLED FONT DATABASE -- 2 942
#: families on the machine this was measured on -- and the walk is synchronous,
#: on the GUI thread, inside whatever call first needed the glyph. MEASURED,
#: first render of one character, everything else warm::
#:
#:     U+2009 THIN SPACE   in Geist   4 658 ms      in Inter    19 ms
#:     σ  τ  µ  ±  ×  ·  – —          under 3 ms in both
#:
#: One character, and it cost the Confirm screen SEVEN SECONDS: a cold 30-series
#: page built in 7 019 ms under Geist and 2 193 ms under Inter, one family
#: swapped and nothing else. The cost lands wherever the glyph is first drawn,
#: with no traceback and no error, and it scales with how many fonts the machine
#: has -- so it barely shows on a clean install and is brutal on a loaded one.
#:
#: Over the eleven characters this interface actually renders, Inter triggers
#: the walk ZERO times and Geist once. That is the whole argument for the swap;
#: it is a coverage decision, not a taste one.
#:
#: SUPERSEDED, kept so it cannot be requoted: this note used to read "Geist has
#: no Greek, and it ships a ``.notdef`` box -- so Qt considers the glyph
#: present, never consults a fallback, and draws an empty rectangle", and
#: concluded "SCREEN TEXT SPELLS THEM: 'PSF width', not 'PSF σ'". Under Inter,
#: σ, τ, µ and U+2009 all rasterise DIFFERENTLY from U+E000 (a private-use
#: codepoint no font has), i.e. they are real glyphs. The spelling-out rule is
#: therefore no longer forced by the font. It is left in place because it is
#: also a plain-language choice, but it is now a preference and not a
#: workaround, and a screen that writes σ will render it.
FAMILY = "Inter"
MONO_FAMILY = "Geist Mono"
#: IBM Plex Mono (IBM, SIL OFL), the Metadata panel's receipt face -- chosen by the UI designer over
#: Cutive Mono, Courier Prime, Space Mono, JetBrains Mono and Geist Mono after seeing all
#: six set as the receipt. Static Regular and Medium files, no variable axis.
#:
#: **IT LACKS σ AND U+2009**, measured on its cmap, and so does Geist Mono. The receipt
#: prints σ (the smoothing row), so :func:`receipt` names INTER, bundled and covering both,
#: as the explicit next family: a missing glyph then costs a lookup in a font already
#: loaded, never the walk over every installed family described above.
RECEIPT_FAMILY = "IBM Plex Mono"
_FALLBACK = ["Segoe UI Variable Text", "Segoe UI", "Arial"]

#: Tailwind's ``text-*`` ramp, which is the whole of shadcn's type scale.
#: Transcribed complete; the screen uses a SUBSET of it, named below.
SIZES = {
    "xs": 12, "sm": 14, "base": 16, "lg": 18, "xl": 20, "2xl": 24, "3xl": 30,
    #: 13 px. NOT a Tailwind step -- the one deliberate addition to the table.
    #:
    #: Tailwind skips 12 -> 14, and controls that must sit UNDER body text
    #: without dropping to meta size have nowhere to go. At 14 a toolbar button
    #: or a disclosure link matches its own section title exactly and reads
    #: louder than it; at 12 it reads as a caption rather than something you
    #: press.
    #:
    #: The gap is not peculiar to Tailwind -- every ramp built on the same
    #: ratios skips 12 -> 14 in the same place, and the symptom is always the
    #: one seen here: a disclosure link ranking level with the section title it
    #: sits under.
    #:
    #: 13 -> 14 is a ratio of 1.08, SMALLER than the 1.14 step this design
    #: rejected as too subtle to be deliberate. That is acceptable here for a
    #: reason that does not apply there: this is not a CONTENT step competing to
    #: be read at a glance, it is a control step whose whole job is to be nearly
    #: body-size and still lose. Weight and colour carry the rest.
    "control": 13,
}

#: The steps this design actually uses.
#:
#: **Chosen by auditing the screen, not by picking from the table.** The first
#: pass used five steps -- 12, 14, 16, 24, 30 -- whose ratios ran 1.17, 1.14,
#: 1.50, 1.25: one step too small to read as deliberate sitting next to a cliff.
#: Worse, ``base`` and ``3xl`` were each used exactly ONCE in the whole screen.
#: A step that exists for a single label is not a scale, it is an exception, and
#: two of them were producing the "some are big, others are small" feel.
#:
#: The four roles, and the reason each size is where it is:
#:
#: ===== ==== ======================================================
#: step  px   what it carries
#: ===== ==== ======================================================
#: 2xl   24   the page title, and EVERY card's headline number -- the
#:            counts and the acquisition values alike. Same size,
#:            different WEIGHT from the title, which is what shadcn's own
#:            stat cards do (``CardTitle text-2xl font-semibold
#:            tabular-nums``) and what keeps the largest number on screen
#:            from needing a step of its own. Reach for it through
#:            :func:`value`, never by hand.
#: lg    18   MODAL titles ONLY (``DialogTitle text-lg font-semibold``).
#:            It carried the home screen's heading too until an audit
#:            measured that ladder; see the ladder rule below. NOT card
#:            values either -- that changed earlier, for `value`.
#: base  16   TOP-LEVEL CARD TITLES. Straight off the registry:
#:            ``CardTitle`` carries ``leading-none font-semibold`` and NO
#:            ``text-*`` class, so it inherits the 16 px root. A card
#:            NESTED in one of those (a lane) steps back down to ``sm``,
#:            or the board reads as sections rather than as one section
#:            holding lanes -- see ``groups.Lane``. This step was retired
#:            once and came back for a reason; see below.
#: sm    14   labels, body, series names -- and every card DESCRIPTION,
#:            which is what a metric's name is (``CardDescription
#:            text-sm text-muted-foreground``).
#: ctrl  13   disclosure links and tooltips. NOT buttons any more: the
#:            registry puts those at ``text-sm`` and the 13 px reading was
#:            calibrated against a 14 px panel title that is now 16.
#: xs    12   notes, meta, day dates, tags.
#: ===== ==== ======================================================
#:
#: **``base`` was retired and then earned its way back, which is the rule
#: working rather than failing.** The first pass dropped it because it was used
#: exactly ONCE in the whole screen, and a step that exists for a single label
#: is an exception, not a scale. The note left behind said reaching for it again
#: would be the signal that a new ROLE had appeared and needed deciding.
#:
#: That is exactly what happened. Card titles were sitting at ``sm``/semibold,
#: level with body text, and read as small -- correctly, because the registry
#: puts them a step above it. So ``base`` returns as a ROLE with four sites, not
#: as one label wanting to be bigger, and the 14 -> 16 ratio of 1.14 (rejected
#: elsewhere here as too subtle to read as deliberate) is acceptable for the
#: same reason 13 is: it is not a content step competing to be read at a glance,
#: and semibold against normal carries the rest of the distance.
#:
#: ``3xl`` remains unused.
#:
#: THE LADDER RULE, from an audit of all 92 typography calls in this package.
#:
#: **24 and 16 are the heading pair. 18 exists only where 24 does not.**
#:
#: `HomePage` stacked three headings -- 24, 18 and 16, every one of them at
#: weight 600 -- and MEASURED that ladder does not read as three levels: the
#: 18 -> 16 step is a ratio of 1.125 and **1.45 px of cap height** (13.09
#: against 11.64), while weight contributes nothing because all three are
#: semibold. It is the same failure the launcher cards showed at 16 over a
#: 14 px body, 1.47 px, which is what prompted this audit. So a screen that
#: already carries a 24 goes STRAIGHT to 16, and ``lg`` is reserved for a
#: MODAL, where it is the top of that dialog's own ladder and has no 24 to
#: compete with (``dialogs.RegroupDialog`` is the case it exists for).
#:
#: **Below 16, hierarchy is carried by WEIGHT and COLOUR, never by size.**
#: 16 -> 14 is 1.14 and 14 -> 12 is 1.17, both far under any legibility
#: threshold, and both are correct as they stand: 600-against-400 and
#: foreground-against-muted do that work. This is written down so the next
#: reader does not "fix" a ratio by inventing a seventh step -- the failure
#: this table's own history records twice.
#:
#: **A title NESTED in a titled card steps down one.** ``groups.Lane`` sits at
#: ``sm``/semibold inside "Analysis groups" for exactly this reason, and
#: ``confirm.LoadingPage``'s summary card stays at ``sm`` under the project
#: name for the same one. Both are deliberate and neither is a violation.
#:
#: **ONE ROLE, ONE SPELLING.** The 16/600 card-and-section title resolves
#: through ``panels.section_font()`` everywhere -- "Analysis groups",
#: "Replicate days", "Recent projects", "Open a project folder" and the
#: launcher cards. It was reached three different ways (that function, a bare
#: ``ty.ui("base", "semibold")``, and a ``px=``/``wght=`` pair), all rendering
#: identically at the time and none of them tracking the others: moving
#: ``SECTION_WGHT`` would have moved two sites and silently left the third.
USED_STEPS = ("xs", "control", "sm", "base", "lg", "2xl")

#: Tailwind's ``font-*`` weights, restricted to the four this design uses.
WEIGHTS = {
    "normal": QFont.Weight(400),
    "medium": QFont.Weight(500),
    "semibold": QFont.Weight(600),
    "bold": QFont.Weight(700),
}

#: Tailwind's ``tracking-tight``. Applied only above 20px: it is a display
#: correction, and at 14px it closes counters that small text needs open.
TRACKING_TIGHT = -0.025


#: OpenType features applied to EVERY font this module builds.
#:
#: Set once, module-wide, because that is the only way an interface can be
#: consistent about them: a stylistic set applied at some call sites and not
#: others puts two shapes of the same letter on one screen. :func:`set_features`
#: is how an entry point overrides them.
#:
#: **A font accepts every feature request and silently ignores the ones it does
#: not have**, so asking for a tag proves nothing about whether it did anything.
#: :func:`feature_effect` renders with and without and compares the raster;
#: :func:`report` prints the verdict at startup.
#:
#: **BUT A TAG IS NOT A MEANING -- IT IS AN INDEX INTO ONE FAMILY'S OWN TABLE**,
#: and the sentence above quietly implied otherwise. It says an unknown tag
#: costs nothing, which is true; what it does not say is that a KNOWN tag in the
#: wrong family does something else entirely. The set here used to be Geist's
#: ``ss01/ss03/ss06``. Carried onto Inter unchanged, ``ss06`` is a live feature
#: rather than an ignored one, and MEASURED it blew the digits apart: the string
#: ``128`` went from 22.9 px to 58.0 px, i.e. every number in every table
#: silently deformed while the font resolved correctly and nothing raised.
#:
#: So this set is chosen PER FAMILY and re-measured when the family changes.
#: What each one does in the bundled Inter, by rendering every character with
#: the feature off and on and diffing the raster::
#:
#:     cv01   1
#:     cv06   u
#:     cv08   I
#:     ss07   i j . , ; : ! ?
#:
#: ``cv08`` is the disambiguation that is kept: it serifs the capital ``I``,
#: which is the glyph a reader has to tell apart from ``l`` and ``1`` in a
#: series name like ``series001`` or a cell id, where a misread is a wrong row
#: rather than an ugly one.
#:
#: **NEITHER THE ``l`` NOR THE ZERO IS DISAMBIGUATED**, and both are rulings
#: rather than oversights. This set was briefly ``ss02`` -- Inter's
#: disambiguation set, covering ``l``, ``I`` and ``0`` in one tag -- then split
#: into ``cv05`` + ``cv08``, and then ``cv05`` was dropped as well, leaving the
#: ``I`` alone. Each of the other two is a single entry away and MEASURED to
#: touch nothing else: ``cv05`` changes ``l``, and the standard ``zero`` feature
#: changes ``0``. Add one only if a misread actually costs a wrong row.
FEATURES: dict[str, int] = {"cv01": 1, "cv06": 1, "cv08": 1, "ss07": 1}


def set_features(**features: int) -> dict[str, int]:
    """Replace the module-wide feature set. Returns what is now active."""
    global FEATURES
    FEATURES = dict(features)
    return dict(FEATURES)


#: The sample :func:`feature_effect` uses. FULL alphabet and digits, not a
#: hand-picked string.
#:
#: The first version used ``"agilry 0123456789 Rt"`` -- the letters stylistic
#: sets "usually" touch -- and it reported ``ss06`` as inert. ss06 substitutes
#: exactly one glyph, capital ``G``, which that sample did not contain. So the
#: check said the font lacked a feature when the TEST lacked a character, which
#: is the more dangerous failure of the two: it reads as a confident negative.
#: Cheap enough to be exhaustive, so it is.
FEATURE_SAMPLE = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "abcdefghijklmnopqrstuvwxyz"
                  "0123456789.,:;?!*&@#%()[]{}/-")


def feature_effect(tag: str, sample: str = FEATURE_SAMPLE, px: int = 32) -> bool:
    """Does ``tag`` actually change how ``sample`` renders in this family?

    Rasterises the sample with the feature off and on and compares the images.
    A raster comparison rather than an advance comparison because a stylistic
    set is usually a GLYPH SUBSTITUTION at the same width -- the shape changes
    and the metrics do not, so measuring advances (which is the right test for
    ``tnum``) would report every stylistic set as inert.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QImage, QPainter

    def raster(on: bool) -> bytes:
        font = QFont(_family())
        font.setPixelSize(px)
        font.setHintingPreference(QFont.PreferNoHinting)
        if on:
            font.setFeature(QFont.Tag(tag), 1)
        img = QImage(px * len(sample), px * 2, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setFont(font)
        p.drawText(0, int(px * 1.2), sample)
        p.end()
        return img.constBits().tobytes()

    return raster(False) != raster(True)


#: The fonts this application SHIPS, beside this module so they travel with
#: the package rather than with the checkout.
#:
#: **The design must not depend on what is installed on the machine.** Until
#: now it did: `_family()` walked a fallback stack and took whatever it found,
#: so the same build rendered in Inter here, Geist on one colleague's laptop and
#: Segoe UI on another -- and the deliverable is a standalone installer, where
#: "whatever is installed" is not something we get to inspect. Two files settle
#: it for every machine.
#:
#: They are the VARIABLE builds (`Inter[opsz,wght].ttf`), which is one file per
#: family instead of nine and brings a continuous `wght` axis with it -- see
#: :func:`ui`'s ``wght`` argument for what that buys.
#:
#: SIL Open Font License 1.1, copied beside them in ``LICENSE.txt``. That
#: licence permits bundling and redistribution inside a commercial installer,
#: which matters here for the same reason the Qt binding does -- see "The Qt
#: backend is PySide6, and that is a LICENSING decision" in ``CLAUDE.md``.
FONT_DIR = pathlib.Path(__file__).resolve().parent / "fonts"

#: The resolved family, once. See :func:`_family`.
_RESOLVED: str | None = None

#: Whether the bundled faces have been registered this run.
_BUNDLED = False


def _load_bundled() -> list[str]:
    """Register every font we ship. Once, and LOUDLY if one will not load.

    Called from :func:`_family`, not from an entry point, so it cannot be
    forgotten: every path that builds a font goes through the family lookup
    first, including the check scripts. `addApplicationFont` needs a GUI
    application to exist, so this no-ops until one does -- the same condition
    `_family` already handles for its own cache.

    A font that fails to load returns -1 and is not reported anywhere else; the
    application would simply render in a fallback and look subtly wrong on the
    machine where it happened. So it raises.
    """
    global _BUNDLED
    if _BUNDLED or QGuiApplication.instance() is None:
        return []
    _BUNDLED = True
    families: list[str] = []
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id == -1:
            raise RuntimeError(
                f"the bundled font {path.name} would not load; the interface "
                f"would silently render in a fallback")
        families.extend(QFontDatabase.applicationFontFamilies(font_id))
    return families


def _family() -> str:
    """The first installed family in the stack. Asked, never assumed -- ONCE.

    **The question is cheap; asking it is not.** `QFontDatabase.families()`
    enumerates every family installed on the machine, and `ui()` called this on
    every font it built -- so opening one menu asked the system for the whole
    font list 34 times. MEASURED with cProfile on the lanes board's Filter
    menu: 16.3 ms per open inside `families()` alone, out of 25 ms in `ui()`.
    Nothing else in that menu's build came close.

    The answer cannot change while the process runs -- but only because the
    bundled faces are registered FIRST, on the line below. That used to read
    "nothing here calls `addApplicationFont`, so there is no moment after
    startup when a new family could appear", which stopped being true the day
    the fonts were bundled: had the load happened later, this cache would have
    pinned a fallback family chosen before the real one existed.

    Cached only once a GUI application exists, though. Before one does,
    `families()` answers with an empty list, and remembering that would pin the
    last fallback for the whole run -- a wrong answer that would then never be
    re-asked, which is exactly the failure a cache is prone to.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    _load_bundled()
    have = set(QFontDatabase.families())
    chosen = next((n for n in (FAMILY, *_FALLBACK) if n in have),
                  _FALLBACK[-1])
    if QGuiApplication.instance() is not None:
        _RESOLVED = chosen
    return chosen


def ui(step: str = "sm", weight: str = "normal", *, tight: bool | None = None,
       tabular: bool = False, family: str | None = None,
       px: int | None = None, wght: float | None = None) -> QFont:
    """A UI font at a named Tailwind step.

    ``tight`` defaults to "only if this is display-sized", which is the rule
    Tailwind's own docs describe and the reason it is not simply always on.

    ``tabular`` is Tailwind's ``tabular-nums``. Off by default because
    proportional figures set better in running text, and REQUIRED on anything
    that changes in place -- see :func:`value`.

    ``wght`` asks the VARIABLE font for a weight the named steps cannot reach --
    450, say, between `normal` and `medium`. It needs the bundled variable face
    (:data:`FONT_DIR`); with the static per-weight files an earlier bundle
    shipped it does nothing at all, silently, which is why it arrived
    with the bundle and not before. MEASURED: the axis WINS over ``weight`` when
    both are set (400+450 and 600+450 both render at 450), so ``weight`` stays
    meaningful as the graceful fallback if the family in use has no axis.

    ``px`` overrides the step's size and exists for ONE case: a surface drawn
    at a root other than 16 px, where a Tailwind class lands between steps.
    The workspace panel is at an 18 px root, so its ``text-sm`` is 15.75 -- the
    ramp's neighbouring steps are 14 and 16, and rounding up made the panel's
    type visibly heavier than the reference. The STEP is still passed, so the
    class a size came from is still readable at the call site; only the
    rounding changes.
    """
    px = SIZES[step] if px is None else px
    font = QFont(family or _family())
    font.setPixelSize(px)
    font.setWeight(WEIGHTS[weight])
    # NO HINTING, and this is the difference between "Inter" and "mangled".
    #
    # Windows' default is full hinting: stems are snapped to whole pixels so
    # small text stays crisp on low-resolution screens. Inter is
    # drawn for the opposite regime -- subpixel positioning with the outline
    # left alone -- so snapping distorts exactly the proportions the face was
    # designed around, and it lands hardest at the 12-14 px this interface is
    # mostly made of. It also breaks letter-spacing, because tracking asks for
    # fractional advances that a hinted font then rounds away.
    #
    # The web never shows this: browsers do not hint these faces. So a design
    # taken from a browser mock into Qt looks wrong for a reason that is not in
    # the design at all.
    font.setHintingPreference(QFont.PreferNoHinting)
    font.setStyleStrategy(QFont.PreferAntialias)
    for tag, setting in FEATURES.items():
        font.setFeature(QFont.Tag(tag), setting)
    if tabular:
        font.setFeature(QFont.Tag("tnum"), 1)
    if wght is not None:
        font.setVariableAxis(QFont.Tag("wght"), float(wght))
    if tight if tight is not None else px >= 24:
        font.setLetterSpacing(QFont.PercentageSpacing, 100 + TRACKING_TIGHT * 100)
    return font


def value(step: str = "2xl", weight: str = "bold") -> QFont:
    """The ONE font for a card's headline number. Always tabular.

    A single function rather than a spelling to remember, because the figures
    are not a matter of taste here -- these numbers CHANGE IN PLACE as series
    are ticked in and out of the fit, and Inter's default figures are
    proportional. MEASURED at 24 px, with this module's own feature set live::

        digit advance, as shipped    '1' 9.9 px   '4' 15.8 px   (spread 6.0)
        with tnum                    every digit 15.1 px        (spread 0.0)

    So a count crossing from a 1 to a 4 shifts the text 5.9 px sideways with
    the default figures and 0.0 px with ``tnum``. Re-measured on the family
    swap rather than carried over: under Geist the same table read '1' 10.5 px
    against '0' 16.4 px, so the widest digit is not even the same one. That is a number visibly
    jumping while you work, and it is why shadcn's own stat card spells out
    ``tabular-nums`` on its ``CardTitle``.

    Worth recording how this was missed: the table above already SAID "the
    counts values (bold, tabular)". The prose was right and the code never did
    it, which is the failure mode a docstring cannot catch -- so the fix is a
    function that cannot be called without the feature rather than a sentence
    asking for it.
    """
    return ui(step, weight, tight=True, tabular=True)


def mono(step: str = "xs", weight: str = "normal") -> QFont:
    """Geist Mono, for values that must not shuffle sideways when redrawn.

    `_family()` FIRST, return discarded -- it is what registers the
    bundled faces (`_load_bundled`), and `ui()` itself only reaches that
    call in its OWN fallback branch (`family or _family()`), which this
    function never takes since it always passes an explicit `family=`.
    Call this before anything else has asked `ui()` for an unqualified
    font and `QFontDatabase.families()` here answers without Geist Mono
    ever having been registered -- MEASURED failing exactly this way
    (called as the first typography request in a fresh process,
    `QFontDatabase.families()` came back without the bundled fonts
    registered at all, and this silently fell back to Consolas), not
    caught by inspection.
    """
    _family()
    have = set(QFontDatabase.families())
    return ui(step, weight,
              family=MONO_FAMILY if MONO_FAMILY in have else "Consolas")


def receipt(step: str = "xs", weight: str = "normal", *, px: int | None = None) -> QFont:
    """IBM Plex Mono for the Metadata receipt, falling back to Inter per glyph.

    Built WITHOUT :data:`FEATURES`: those tags are Inter's choices and mean other glyphs
    in another face (the ``ss06`` trap above), so the receipt draws Plex as Plex. Hinting
    and antialiasing follow :func:`ui`. `_family()` first, for the reason :func:`mono`
    gives: it is what registers the bundled faces.
    """
    _family()
    have = set(QFontDatabase.families())
    first = RECEIPT_FAMILY if RECEIPT_FAMILY in have else (MONO_FAMILY if MONO_FAMILY in have
                                                          else "Consolas")
    font = QFont(first)
    font.setFamilies([first, FAMILY])
    font.setPixelSize(SIZES[step] if px is None else px)
    font.setWeight(WEIGHTS[weight])
    font.setHintingPreference(QFont.PreferNoHinting)
    font.setStyleStrategy(QFont.PreferAntialias)
    return font


def report() -> str:
    """One line naming what actually resolved — printed at startup.

    Includes the weight check from the module docstring, so the claim that
    Qt synthesises Inter's family group is re-tested on the machine in front
    of you rather than quoted from the machine it was measured on.
    """
    resolved = _family()
    light, heavy = ui("2xl", "normal"), ui("2xl", "bold")
    spread = (QFontMetrics(heavy).horizontalAdvance("Confirm project")
              - QFontMetrics(light).horizontalAdvance("Confirm project"))
    actual = QFontInfo(ui("2xl", "semibold")).family()
    weights = "distinct" if spread > 0 else "COLLAPSED (weights not resolving)"
    # Name the features that are ASKED FOR and inert, rather than only the ones
    # that work: a tag silently doing nothing is the failure this check exists
    # to catch, so it has to be the part that is loud.
    live = [t for t in FEATURES if feature_effect(t)]
    dead = [t for t in FEATURES if t not in live]
    feat = "+".join(live) if live else "none"
    if dead:
        feat += f"  (INERT in this family: {'+'.join(dead)})"
    return (f"family={resolved!r} (semibold resolves to {actual!r})  "
            f"weights={weights} (+{spread}px 400->700)  mono={MONO_FAMILY!r}"
            + chr(10) + f"        features={feat}")
