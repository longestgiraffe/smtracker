"""Movie preparation: the raw acquisition to the stacks everything downstream reads.

Reproduces the experimenter's Fiji macro (`automationbackbonePB_V3.ijm`), which is the
step before u-track and therefore before every number in this project:

    raw/<movie>.nd2   ->  bleaching curve over the WHOLE acquisition
                      ->  tif/<movie>.tif   the cropped stack, ONE multi-page TIFF
                      ->  bleaching curve over the CROP
                      ->  sum/<movie>.tif   a projection, for the viewer's background

Two `tif/` layouts exist and only one is ours. The macro writes a SINGLE multi-page
TIFF per movie (`saveAs("tiff", pathB + title + fString)`); the
`tif/<movie>/<movie>/*.tif` folder of hundreds of single frames is produced by **u-track
itself** from that file, as its own working copy. An earlier version of this module
wrote the folder: impersonating the tracker, and skipping the file u-track is actually
given. The experimenter caught it.

The crop is the scientific decision in this pipeline, so it has no default. The macro
hard-codes `rangeStart`/`rangeEnd` and they are edited by hand per batch, 500-1500 in
one copy, 500-2500 in another, which is exactly why the frame counts in this project vary
(1001, 1501, 2001, 1751). The rule is physical rather than numeric: cut where the field
has bleached down to separable single molecules.

And no function here proposes one, because the images do not determine it. The first
draft of this module shipped a `suggest_crop` built on the bleaching curve. Checked
against twelve crops actually made, it was wrong by two orders of magnitude in its one
tuning constant, and fixing that constant did not save it: field mean, spot density and
spot amplitude ALL place the single-molecule floor a median 2124 frames (**50.5 per cent
of each acquisition**) before the crop chosen. See `CropFloor`, which reports each
criterion as a FLOOR and refuses to combine them into a recommendation.

Five things measured while reproducing it, each recorded because the macro's own names
say otherwise:

1.  `sum/` is a STANDARD-DEVIATION projection, not a sum (`Z Project... projection=
    [Standard Deviation]`). The name is kept because the viewer and the legacy layout
    both glob for `sum/`, but it is not a sum and must never be described as one. It is
    also why the image reads as a track envelope: an SD projection lights up the pixels
    that VARY, which is where molecules moved.

2.  That image is not quantitative. `Enhance Contrast saturated=0.2` sets the DISPLAY
    range and the `16-bit` conversion that follows BAKES that range into the pixel
    values. So its absolute numbers are a function of a display setting. Confirmed on the
    shipped file: the reference project's `sum/series001.tif` spans the full 0-65535 with
    median 8069, which a sum of 1001 frames at ~1100 counts could never do. This module
    therefore returns the FLOAT projection as the real quantity and writes the stretched
    16-bit copy only for compatibility.

3.  **`Enhance Contrast` does not touch the analysis stack.** Without `normalize` it only
    sets the lookup table, so the `tif/` stack u-track reads is untouched raw data. Not
    assumed: the cropped stack's last frame mean (1129) equals the raw `.nd2`'s frame
    4500 mean. This module writes the raw slice unmodified and a test asserts it is
    bit-identical.

4.  The frame interval is 20.472 ms, not 20 ms. The macro writes `frame=[0.020479
    sec]` and it is right: the camera's own timestamps give 20.4717-20.4733 ms across
    four lab conditions, one of them a WT control (sd 0.002 ms over 1999 intervals).
    `movieData.mat` carries 0.02, the nominal exposure, and that is what every D in this
    project was computed with. See `TAU_NOTE`: the consequence is a uniform +2.4 % on
    every D and every rate, and NOTHING that rests on a ratio moves.

One deliberate departure, and it is an addition rather than a replacement: the macro
saves each bleaching curve as a TIFF **picture of a plot**, so the numbers are discarded
and only a rendering survives. This writes the curve itself as CSV: the same firewall
the manuscript figures use, because a rendered plot cannot be re-scaled, re-read or
checked against anything, AND the picture beside it, drawn from that CSV. For a while
it wrote only the CSV, which this very paragraph described as "draws the picture from
it" while nothing did; the experimenter noticed the pictures had gone. Both ship.

The projection is a choice, defaulting to what the macro did. `Z Project...` offers Sum
Slices and Standard Deviation; the macro hard-codes Standard Deviation, and every `sum/`
file in this project's existing data IS one: the folder's NAME is the only thing that
says sum. `PrepConfig.projection` selects and `sd` is the default,
because changing it silently would make two differently-made images share one filename.
The macro saves ONE image per movie, so the quantitative float copy is OPT-IN
(`write_float_projection`) and named for what it holds (`_sd_float` / `_sum_float`); it
looks blank in most viewers, which is not a defect but float32 with a range of tens of
counts shown by a viewer assuming 0-255.

Pure and Qt-free: the preparation window calls these functions and computes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .chart_text import fold_words

__all__ = [
    "MoviePrepError",
    "TAU_NOTE",
    "PrepConfig",
    "CropFloor",
    "bleaching_curve",
    "density_curve",
    "CROWDING_RADII_PX",
    "MIN_SEPARATION_PX",
    "CROWDING_FLAG_RATIO",
    "CROWDING_RATIO_FLOOR",
    "DENSITY_FLAG_RATIO",
    "CROWDING_MIN_MOVIES",
    "resolve_peaks",
    "crowding_column",
    "crowding_summary",
    "crop_floor",
    "crop_quality",
    "sd_projection",
    "projection",
    "PROJECTIONS",
    "display_stretch",
    "prepare_movie",
    "prepare_batch",
    "CROPS_FILENAME",
    "save_crops",
    "load_crops",
]


class MoviePrepError(RuntimeError):
    """Raised loud. Nothing here guesses a crop, a pixel size or an interval."""


#: MEASURED, and stated so it cannot be read as alarming. Every D and every rate this
#: project reports used tau = 0.020 s; the camera's timestamps say 0.020472 s. So a
#: reported D is high by 0.020472/0.020 = 1.0236, i.e. **+2.4 %**, and a reported rate
#: likewise (a dwell time is 2.4 % low). NOTHING THAT RESTS ON A RATIO MOVES: the
#: localisation floor sigma^2/(4 tau) and the detection ceiling scale by the same factor,
#: so D1-against-the-floor, the sigma crossover at 66.2 nm and every FRACTION are
#: untouched. The frozen MATLAB reference used the same 0.02, so the validation to 3 s.f.
#: is unaffected: this is a shared convention, not a disagreement between two codes.
TAU_NOTE = (
    "u-track's movieData.mat holds the nominal exposure (for example 0.0200 s); the "
    "camera timestamps may give another (0.020472 s). Then D and rates read 2.4 % high "
    "and dwell times 2.4 % low. Fractions and ratios are unaffected."
)

#: The interval the macro writes and the camera confirms. NOT a default anywhere: it is
#: quoted so a user can recognise their own number.
MEASURED_FRAME_INTERVAL_S = 0.020472

#: ImageJ's `Enhance Contrast` saturated= values, per output, from the macro. Recorded
#: because they are display settings that a reader could mistake for analysis choices.
LEGACY_SATURATED = {"tif": 0.35, "cropped_curve": 0.3, "sum": 0.2}


@dataclass(frozen=True)
class PrepConfig:
    """What preparation needs, none of it guessable.

    ``crop_start`` / ``crop_end`` are 1-BASED and INCLUSIVE, matching the macro's
    ``duplicate range=START-END`` and therefore matching what is written in a lab book.
    """

    crop_start: int
    crop_end: int
    pixel_size_nm: float
    frame_interval_s: float
    sum_saturated: float = LEGACY_SATURATED["sum"]
    #: "sd" or "sum". SD is the macro's and is what every existing `sum/` file holds.
    projection: str = "sd"
    #: The quantitative float projection beside the 16-bit one. OFF: the macro writes a
    #: single image per movie, and the extra file reads as an output nobody asked for.
    write_float_projection: bool = False

    def __post_init__(self):
        if self.crop_start < 1:
            raise MoviePrepError(
                f"crop_start is 1-based and must be >= 1, got {self.crop_start}")
        if self.crop_end < self.crop_start:
            raise MoviePrepError(
                f"crop_end {self.crop_end} is before crop_start {self.crop_start}")
        if not self.pixel_size_nm > 0:
            raise MoviePrepError(
                "pixel_size_nm has no default. Take it from the tracking record (movieData.mat), "
                "not from the .nd2, which can report the bare objective magnification "
                "without the tube lens and would rescale every D")
        if not self.frame_interval_s > 0:
            raise MoviePrepError(
                "frame_interval_s has no default. " + TAU_NOTE)

    @property
    def n_frames(self) -> int:
        return self.crop_end - self.crop_start + 1

    def stamp(self) -> str:
        return (f"frames {self.crop_start}-{self.crop_end} ({self.n_frames}), "
                f"{self.pixel_size_nm:g} nm/px, {self.frame_interval_s * 1000:.4f} ms")


#: The camera's dark level, in counts, and how far above it a frame may sit and still be a
#: CANDIDATE for unlit. MEASURED across all 119 raw movies in this project: the dark plateau
#: lies at **506.0-510.4 counts** and is quiet to about one count, so this is a property of
#: the CAMERA rather than of any movie.
#:
#: AN ABSOLUTE LEVEL IS WHAT MAKES THE TEST SAFE, and a per-movie one is not. A first
#: attempt used the 1st percentile of each movie: on a movie with no dark window that is
#: simply the dimmest LIT frame, so the guard excluded the bleached tail -- the very
#: region a floor belongs in -- and pushed 21 of 106 movies to discard the whole
#: acquisition. Against a fixed level, a movie that was never dark has no frame below it
#: and is left exactly alone. Measured after the change: 124 of 126 such movies unmoved.
#:
#: BUT A LEVEL ALONE CANNOT DECIDE IT, which is why `lit_frames` now asks for a SWITCH
#: (12 Oct 2026). The old rule called a frame lit above 535 counts (510 + 25). one user's
#: dim mVenus movies are LIT at 520-540, so every frame read as dark, `crop_floor` threw
#: their probes away and `last_illumination_start` put the "last switch-on" at the movie's
#: end -- 22 movies were proposed the range 2000-2000. And no level separates the cases:
#: One user's PY79 movies whose laser went off in their LAST frames read 514-517 there
#: (1542 -> 851 -> 515), above the leading dark plateau. What does separate them is that a
#: laser SWITCHES -- the field falls or rises to the dark level within a frame or two --
#: while a dim field fades. 12 counts above the dark level keeps every dark frame seen
#: (up to 517) a candidate.
DARK_LEVEL_COUNTS = 510.0
DARK_MARGIN_COUNTS = 12.0
#: Where the dark plateau sits (the median of the 506.0-510.4 measured): the zero a
#: frame's EXCESS over dark is counted from when a switch is judged.
DARK_PLATEAU_COUNTS = 507.0
#: A switch: across SWITCH_SPAN frames the field changes by at least SWITCH_MIN_COUNTS and
#: its excess over the dark plateau by at least SWITCH_RATIO times. MEASURED: a laser going
#: off is 1000-fold on a bright PY79 field (1522 -> 515), ~100-fold on mCherry (637 -> 508)
#: and would be ~20-fold on dim mVenus (530 -> 508); a fading dim field crossing the
#: candidate level changes by 1-2 counts, about 1.2-fold.
SWITCH_SPAN = 3
SWITCH_MIN_COUNTS = 8.0
SWITCH_RATIO = 3.0


def _switch(dark_side, lit_side) -> bool:
    """Whether the field steps between these two short runs of frame means, from the
    dark side to the lit side (direction is the caller's)."""
    import numpy as np

    if len(dark_side) == 0 or len(lit_side) == 0:
        return False
    low, high = float(np.median(dark_side)), float(np.median(lit_side))
    excess_low = max(low - DARK_PLATEAU_COUNTS, 1.0)
    excess_high = high - DARK_PLATEAU_COUNTS
    return (high - low) >= SWITCH_MIN_COUNTS and excess_high >= SWITCH_RATIO * excess_low


def lit_frames(means) -> "np.ndarray":
    """Which frames saw illumination. See `DARK_LEVEL_COUNTS`.

    A frame at the camera's dark level did not image a thinned field; it imaged nothing.
    The distinction matters because a LoG detector still returns spots there -- MEASURED
    on `D:/data/project/condition/series001`, **200-320 spots per frame at
    amplitude ~100** while the laser was off, against 2200-6700 for a real molecule. Every
    criterion in `crop_floor` was then reading noise as data.

    A frame is UNLIT when it sits within `DARK_MARGIN_COUNTS` of the dark level AND the
    laser demonstrably switched around it: its run of such frames is the leading one,
    ended by the laser coming on; or follows a drop to dark (switched off); or precedes a
    rise from dark (switched back on); or the movie never leaves the dark level at all. A
    field fading slowly toward the dark level has no switch and stays lit. A lit island
    shorter than `SWITCH_SPAN` frames between two unlit runs is unlit too (a flicker).
    Frame means are expected in ORDER; on a subsample the steps are coarser but the rule
    holds.
    """
    import numpy as np

    m = np.asarray(means, dtype=float)
    n = m.size
    lit = np.ones(n, dtype=bool)
    near = m < (DARK_LEVEL_COUNTS + DARK_MARGIN_COUNTS)
    if not near.any():
        return lit
    if near.all():
        return ~lit
    span = SWITCH_SPAN
    i = 0
    while i < n:
        if not near[i]:
            i += 1
            continue
        j = i
        while j < n and near[j]:
            j += 1
        first, last = m[i:i + span], m[max(i, j - span):j]
        before, after = m[max(0, i - span):i], m[j:j + span]
        off_into = i > 0 and _switch(first, before)        # lit -> dark at i
        on_after = j < n and _switch(last, after)          # dark -> lit at j
        if off_into or on_after:
            lit[i:j] = False
        i = j
    # a lit flicker of a frame or two inside a dark window is part of the window
    k = 0
    while k < n:
        if not lit[k] or k == 0:
            k += 1
            continue
        stop = k
        while stop < n and lit[stop]:
            stop += 1
        if stop < n and (stop - k) < span and not lit[k - 1]:
            lit[k:stop] = False
        k = stop
    return lit


#: How far the frame mean may wander inside a stretch still called flat, as a FRACTION of
#: that stretch's own level. Ruled by the experimenter: "look at where the flat curve is
#: for the most frames, basically, allowing a few percentage fluctuation."
#:
#: RELATIVE, WHICH IS THE WHOLE POINT. Every earlier criterion here was absolute -- ten
#: spots per frame, a tail level in counts -- and an absolute number cannot transfer from a
#: bright mNeonGreen field to a dim mCherry one, nor between fields of different size. A
#: fraction of the curve's own level holds a movie at 680 counts and one at 1300 to the
#: same standard, with nothing to tune per label.
#:
#: TWO PER CENT, RULED BY THE EXPERIMENTER ON THE QUANTITY HE SAID DECIDES IT: "the
#: biggest problem is too much photo density". So the tolerances were compared on SPOTS
#: PER FRAME at the floor each one gives, not on how much of the movie they keep. Measured
#: over the eight MSD movies (median over the 400 frames after the floor):
#:
#:     2 %  4.5 spots/frame     3 %  5.2     5 %  9.5     the crop he chose  4.2
#:
#: 5 PER CENT IS THE ONE THAT FAILS: it more than doubles the density actually worked at,
#: and does it worst on two lab conditions (17.0 and 14.0). Between 2 and 3 the
#: median gap is 0.7 spots/frame, but it is not spread evenly -- one condition reads 3.0
#: against 6.0 and another 6.5 against 7.5, i.e. 3 per cent is worst exactly on the
#: low-occupancy movies this project already records as the fragile case.
#:
#: WHAT 2 PER CENT COSTS, RECORDED BECAUSE IT IS REAL AND WAS OVERRULED RATHER THAN
#: ANSWERED. The crop an analyst actually chose sits a median 50.5 per cent into the
#: acquisition, and a 2 per cent floor keeps 48.3 per cent -- so the floor now lands ON the
#: human choice rather than well below it, and its margin narrows to 125 frames on
#: one lab condition and 210 on another, against 473-1581 at 3 per cent. A floor that
#: reproduces the choice is closer to a proposal than to a bound, which is the one thing
#: this stage exists not to be. It is STILL BELOW the analyst's crop in 8 of 8, so it
#: remains a floor by measurement; the objection is that it is no longer a comfortable one.
#:
#: For scale, the floor keeps a median 48.3 per cent of each movie at 2 per cent, 63.6 at 3
#: and 80.7 at 5, over 154 saved curves. THE VALUE IS A CONTROL, not a constant: the crop
#: panel's "flat within %" box moves it per project, so a label or a field this was never
#: measured on is a setting away rather than a code change.
FLAT_TOLERANCE = 0.02

#: Frames. A flat stretch shorter than this is a coincidence, not a regime.
FLAT_MIN_FRAMES = 200

#: Above this share of the fluorescence, the flat band is mostly CAMERA OFFSET and the
#: floor says so. Found on a MetaMorph camera (one user's B. subtilis movies, 96 of them):
#: its dark level is ~1944 counts under only ~75-135 counts of fluorescence, so 2 % of
#: the raw level is ~40 counts, about HALF the signal, and the rule called the slow decay
#: flat from frame 98 (median; 1-213) where the macro cropped all 96 at 1000. On the
#: Nikon the dark level is ~507 under 140-650 counts, so the same 2 % is 4-16 % of the
#: signal, which is the regime the 2 % was ruled in. MEASURED on those 96 movies, the
#: start the rule gives at each tolerance (median, range): 2 % 98 (1-213), 1 % 230,
#: 0.75 % 332 (175-527), 0.5 % 586 (281-848).
OFFSET_WARN_SHARE = 0.25


def leading_dark_level(means, *, search: int = 600) -> float | None:
    """The camera's dark level from the frames before the laser fired, or ``None``.

    From the frame MEANS, so a floor rebuilt from saved curves can use it too. The dark
    frames are the leading run below 2 % of the way from the minimum to the maximum of
    the first ``search`` frames -- the threshold `rawmovie.detect_dark_frames` uses --
    and there must be one: a movie whose laser was already on starts at its BRIGHTEST
    frame, so its run is empty and nothing is guessed. No contrast floor here, unlike
    that detector, because a MetaMorph camera's step is only 1.1-1.2x its own offset.
    """
    means = np.asarray(means, dtype=float)
    head = means[:min(int(search), means.size)]
    if head.size < 2 or not np.isfinite(head).all():
        return None
    lo, hi = float(head.min()), float(head.max())
    lit = head > lo + 0.02 * (hi - lo)
    n_dark = int(np.argmax(lit)) if lit.any() else 0
    if n_dark == 0 or n_dark >= head.size:
        return None
    return float(np.median(head[:n_dark]))

#: Frames. A dark run at least this long is the laser being OFF rather than a flicker at
#: switch-on. The real interruptions here run 300-600 frames.
MIN_DARK_RUN = 50


def last_illumination_start(means) -> int:
    """First frame index of the FINAL illuminated episode. 0 when the laser never failed.

    AN INTERRUPTED ACQUISITION HOLDS MORE THAN ONE EXPERIMENT. Everything before the last
    laser-on event was imaged under a different illumination, of different molecules at a
    different bleaching depth, so it cannot supply the crop for what follows.

    FOUND BY A TEST RATHER THAN FORESEEN: masking only the dark FRAMES is not enough,
    because a flat stretch lying entirely BEFORE the interruption still competes on
    length, and at a tight tolerance it wins — a planted movie put its floor at frame 1,
    before a dark window at 600. Tightening the tolerance makes early episodes MORE
    competitive, not less, which is the opposite of the intuition.

    Safe by construction on ordinary data: a movie that was never dark has one episode,
    which is the whole movie. Measured, 126 of 126 such curves are untouched.
    """
    lit = lit_frames(means)
    n = lit.size
    index, start = 0, 0
    while index < n:
        if lit[index]:
            index += 1
            continue
        stop = index
        while stop < n and not lit[stop]:
            stop += 1
        if (stop - index) >= MIN_DARK_RUN:
            start = stop
        index = stop
    return int(start)


def longest_flat_stretch(values, *, tolerance: float = FLAT_TOLERANCE,
                         usable=None, min_length: int = FLAT_MIN_FRAMES):
    """``(start, stop)`` of the longest window whose spread is within ``tolerance``.

    Linear rather than quadratic: two pointers with monotonic deques, so a 4500-frame
    curve is scanned once. ``usable`` masks frames that may not take part -- unlit ones,
    see `lit_frames` -- and a window may not straddle them.

    TWO THINGS IT GETS RIGHT WITHOUT BEING TOLD, both of them defects reported against
    earlier rules. The DARK WINDOW of an interrupted movie is the flattest part of it
    (about one count on 507), but it is short beside the settled tail, so "for the most
    frames" rejects it. And the RE-BRIGHTENING after a laser comes back is steep, so it
    cannot lie inside a flat window -- which is why a floor of 726 was wrong on a movie
    whose field was still climbing 530 to 830 at that frame.
    """
    from collections import deque

    values = np.asarray(values, dtype=float)
    n = values.size
    ok = np.ones(n, dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    best = (0, 0)
    low: deque = deque()          # increasing values, so the front is the minimum
    high: deque = deque()         # decreasing values, so the front is the maximum
    left = 0
    for right in range(n):
        if not ok[right] or not np.isfinite(values[right]):
            low.clear()
            high.clear()
            left = right + 1
            continue
        while low and values[low[-1]] >= values[right]:
            low.pop()
        low.append(right)
        while high and values[high[-1]] <= values[right]:
            high.pop()
        high.append(right)
        while left <= right:
            lo_v, hi_v = values[low[0]], values[high[0]]
            middle = 0.5 * (lo_v + hi_v)
            if middle > 0 and (hi_v - lo_v) / middle <= tolerance:
                break
            if low[0] == left:
                low.popleft()
            if high[0] == left:
                high.popleft()
            left += 1
        if right - left + 1 > best[1] - best[0]:
            best = (left, right + 1)
    return best if (best[1] - best[0]) >= min_length else None


@dataclass(frozen=True)
class CropFloor:
    """The EARLIEST start each measurable criterion allows, never a recommendation.

    MEASURED, and the measurement is why there is no suggestion here. Twelve movies
    across four protein G batches, with the crop actually used recovered from the prepared
    stack by `rawmovie.find_crop_offset` (unambiguous in 16 of 16), against three
    independent criteria:

    | criterion | where it is satisfied |
    |---|---|
    | field mean | plateaus 500-1000 frames BEFORE the chosen start; a "within 1 sd of the plateau" rule cuts early in 11 of 16, median 500 frames, and every looser tolerance is worse |
    | spot density | already 1-5 spots/frame at the chosen start AND 500 frames earlier; u-track flags linking ambiguity only above ~209 on this field |
    | spot amplitude | flat from about a quarter of the way in, 954 counts at 25 %, 745 at start-500, 745 at the start; the chosen start sits at 1.11x the movie's end (0.86-1.29) |

    All three place the single-molecule floor far earlier than the crop. A median
    **2124 frames, 50.5 per cent of each acquisition (range 22-54 %)**, is held back
    beyond the point where density has already fallen below 10 spots per frame.

    That margin is not an error and nothing here argues against it. Overlapping PSFs
    bias localisation well before they defeat linking, and the molecules present early
    are the least-bleached, so a wider crop samples a different population as well as a
    denser one. The point is narrower: the margin is a judgement the images do not
    RECORD, so a tool that proposed a number would be proposing its own and dressing it
    as the analyst's.

    SCOPE: protein G, four batches, twelve movies. NOT confirmed on protein A. Generalising a number
    past the regime it was measured in is the mistake this project has now recorded five
    times, so this one states its regime.
    """

    by_density: int | None
    by_amplitude: int | None
    by_field_mean: int
    n_frames: int
    spots_per_frame_limit: float
    #: Why a criterion could not be evaluated, by name. Carried rather than dropped: a
    #: floor built from two criteria is not the same answer as one built from three, and
    #: the difference has to travel with the number.
    unavailable: dict = field(default_factory=dict)
    #: How many probe points were DROPPED as unlit. Carried because a floor read from
    #: 40 lit probes is not the same answer as one read from 180, and a movie whose laser
    #: was interrupted must be able to say so rather than look ordinary.
    n_unlit_probes: int = 0
    #: Where the longest flat stretch of the frame mean begins — THE FLOOR. The three
    #: criteria above are kept as diagnostics because each says something the others do
    #: not, but none of them decides any more: density needs an absolute spot count that
    #: does not transfer between labels (unavailable in 13 of 22 mCherry movies), and the
    #: field-mean rule asks when the mean reaches its FINAL level, which on a curve that
    #: creeps downward means "when the movie is nearly over" — latest of the three in 22
    #: of 22 of those movies, at 2108 to 3950 of 4500.
    by_flat: int | None = None
    flat_tolerance: float = FLAT_TOLERANCE
    #: The camera's dark level read off the leading unlit frames (`leading_dark_level`),
    #: or None when the laser was already on.
    dark_level: float | None = None
    #: The flat band (tolerance x the stretch's level) as a share of the fluorescence
    #: ABOVE the dark level. Near 1 the band is camera offset, not signal -- see
    #: `OFFSET_WARN_SHARE`.
    flat_band_share: float | None = None

    @property
    def offset_warning(self) -> str:
        """One line when the flat band is mostly camera offset, else ``""``."""
        if self.flat_band_share is None or self.flat_band_share <= OFFSET_WARN_SHARE:
            return ""
        return (f"the camera's dark level is {self.dark_level:.0f} counts, so "
                f"{100 * self.flat_tolerance:g} % of this curve is "
                f"{100 * self.flat_band_share:.0f} % of its fluorescence: most of the "
                f"decay counts as flat. Lower 'Flat within', or set one start for every "
                f"movie")

    @property
    def earliest_defensible(self) -> int:
        """Where the longest flat stretch begins, or the old conjunction without one.

        IT USED TO BE THE LATEST OF THREE, and taking a maximum means any one criterion
        behaving badly sets the answer alone. Measured on 22 mCherry movies, the
        field-mean rule was that criterion every single time. The flat rule replaces the
        three rather than joining them, because it subsumes what each was reaching for:
        a field that has stopped changing is thinned, settled and no longer bleaching
        down, and it says so in units of the curve's own level rather than in counts.

        MEASURED across 154 saved curves: the flat rule keeps a median 80.7 per cent of
        each movie against the old 79.2, so it is no stingier overall — but its range is
        38 to 97 per cent where the old one ran 1 to 95. The disasters are what went.
        """
        if self.by_flat is not None:
            return self.by_flat
        floors = [f for f in (self.by_density, self.by_amplitude, self.by_field_mean)
                  if f is not None]
        return max(floors)

    def describe(self) -> str:
        # `unlit` is a NOTE about frames that were dropped, not a criterion that could
        # not be measured, so it must not be listed as one — "unlit not measurable" reads
        # as a failure where the truth is that the guard did its job.
        missing = "; ".join(f"{k} not measurable" for k in self.unavailable
                            if k not in ("unlit",))
        warning = self.offset_warning
        return ((f"{missing}, so this floor rests on the rest. " if missing else "")
                + (f"{warning[0].upper()}{warning[1:]}. " if warning else "")
                + (f"the longest stretch flat to {100 * self.flat_tolerance:g} % "
                   f"begins at {self.by_flat}. "
                   if self.by_flat is not None else
                   "no stretch of this movie is flat for long enough, so the floor "
                   "falls back to the three criteria below. ")
                + f"diagnostics: density {self.by_density}, amplitude "
                f"{self.by_amplitude}, field mean {self.by_field_mean}, of "
                f"{self.n_frames} frames. "
                f"No earlier than {self.earliest_defensible}: this is a floor, not "
                f"a proposal. On twelve test movies the crop actually chosen started a "
                f"median 2124 frames (50.5 %) later, a margin the images do not "
                f"determine. Set the range yourself.")


def bleaching_curve(movie, *, crop: tuple[int, int] | None = None) -> pd.DataFrame:
    """Mean counts per frame: the curve the macro plots and then throws away.

    ``crop`` is (start, end), 1-based inclusive, reproducing the second plot the macro
    makes over the cropped range. Frame numbers stay in the ORIGINAL acquisition's
    numbering, so a crop can always be located back in the full curve.
    """
    means = np.asarray(movie.frame_means(), dtype=float)
    frames = np.arange(1, means.size + 1, dtype=int)
    if crop is not None:
        start, end = int(crop[0]), int(crop[1])
        if start < 1 or end > means.size:
            raise MoviePrepError(
                f"crop {start}-{end} outside the acquisition's 1-{means.size} frames")
        keep = (frames >= start) & (frames <= end)
        frames, means = frames[keep], means[keep]
    table = pd.DataFrame({"frame": frames, "mean_counts": means})
    table.attrs["n_frames_acquired"] = int(movie.meta.n_frames)
    table.attrs["movie"] = str(getattr(movie, "path", ""))
    return table


#: Radii, in camera pixels, at which crowding is recorded. The ladder is the one this
#: project already measured linking ambiguity on (`trackmate_run.linking_ambiguity`), so
#: the two are read on the same axis. They are NOT the same measurement -- see
#: `crowding_summary`, which says so at length and for a reason.
CROWDING_RADII_PX = (3.0, 5.0, 8.0, 12.0)

#: Two detections closer than this are ONE molecule, not two, and are merged before
#: anything is counted.
#:
#: THIS IS NOT A TUNING CONSTANT, IT IS THE RESOLUTION LIMIT, AND LEAVING IT OUT MADE THE
#: FIRST VERSION OF THIS MEASURE REPORT ITS OWN DETECTOR. The probe finds local maxima in a
#: 3x3 neighbourhood with no minimum separation, so the shoulder of a bright PSF becomes a
#: second "spot" two or three pixels from the first. MEASURED on the nearest-neighbour
#: distribution over the analysed crop:
#:
#:     bin (px)            2-3     3-4     detections/frame     after merging
#:     Lab set A   (mNG)  24.1 %   8.0 %        5.10                4.45   (1.15x)
#:     user 2   (mCherry) 60.4 %  15.2 %       24.30               16.30   (1.49x)
#:
#: The 2-3 px bin EMPTIES on merging in both, which is the signature of splitting rather
#: than of crowding: a genuine second molecule has no reason to prefer that separation.
#: The inflation matches this project's independently recorded 1.0-1.4x on mNeonGreen and
#: 1.35-1.79x on mCherry.
#:
#: **THE INFLATION DIFFERS BY LABEL, WHICH IS WHY THIS CANNOT BE LEFT UNCORRECTED.** A raw
#: peak count compares a 1.49x-inflated mCherry condition against a 1.15x-inflated
#: mNeonGreen one, so the two columns of any comparison would be different quantities --
#: this file's most-repeated defect. `n_spots` is left alone, because `crop_floor` is
#: calibrated on it; `n_resolved` is the count to read and to compare.
#:
#: 3 px is above the PSF here (sigma 0.93 px, FWHM 2.2 px for mNeonGreen; about 2.6 px for
#: a red dye) and below every real separation observed. `optics.detection_radius_px` gives
#: the label-specific value -- 2.80 px for mNeonGreen at NA 1.49, 3.50 for a red dye -- and
#: callers that know the fluorophore should pass it.
MIN_SEPARATION_PX = 3.0

#: A movie is NAMED when it is this many times its own condition's median, in resolved
#: detections per frame or in crowded share.
#:
#: **RELATIVE, AND THE ABSOLUTE RULE THAT PRECEDED IT WAS WRONG.** The first version
#: flagged any movie above 10 per cent crowded, a threshold taken from
#: `linking_ambiguity`'s recorded 1.2-4.5 per cent in-cell. Run on real movies it named
#: **three of four clean MSD movies and four of four at 8 px** -- because this probe's
#: detection set is not u-track's: at the same crop it finds 4.45 resolved detections per
#: frame against u-track's 2.33, and the extra ones sit inside cells, where the noise floor
#: is higher because cells autofluoresce at 928 counts against a 597 background. Even
#: after merging, a lab project reads 23.5 per cent here against `linking_ambiguity`'s 4.0.
#:
#: So the two numbers are NOT comparable and a threshold carried across is a threshold
#: measured in one regime applied to another -- the defect this file records five times.
#: What survives the difference is the COMPARISON BETWEEN MOVIES OF ONE CONDITION, which
#: share a detector, a label and a session. Same shape as `noise_outliers`, and the same
#: reason: a movie unlike its neighbours is worth opening, whatever the absolute level.
CROWDING_FLAG_RATIO = 1.5

#: Below this crowded share the ratio rule is not applied. A movie at 0.4 per cent beside a
#: condition median of 0.2 is twice its neighbours and is not a finding. Same MAD-zero trap
#: `noise_outliers` carries `NOISE_FLAG_MIN_RATIO` for.
CROWDING_RATIO_FLOOR = 0.02

#: The same rule on RESOLVED DETECTIONS PER FRAME, which is the quantity asked for: "a
#: number with median for each condition, warning if some movies have high loc/frame".
#: Looser than the crowding ratio because density varies more between fields than
#: ambiguity does -- a movie with more cells in frame is not a problem.
DENSITY_FLAG_RATIO = 2.0

#: Below this many movies the ratio rules do not run, and the summary says why.
#:
#: A RATIO IS AGAINST A REFERENCE, AND WITH THREE MOVIES THE REFERENCE IS ONE MOVIE. The
#: condition median is what "unlike its neighbours" is measured against, so at n = 2 it is
#: the mean of the pair and at n = 3 it is literally the middle movie -- naming the top one
#: then reports the ordering, which every set has. The counts and the medians are still
#: reported; only the flag is withheld, and the row says so rather than reading as clean.
#: Same distinction `qc.control_contamination` makes at ten units.
CROWDING_MIN_MOVIES = 4


def resolve_peaks(positions, values, *, min_separation_px: float = MIN_SEPARATION_PX):
    """Merge peaks closer than one PSF: brightest kept, its neighbours dropped.

    Greedy from the brightest, which is the convention every localisation package uses and
    the only one that is stable -- merging from the dimmest lets a noise peak absorb a real
    molecule. Returns the kept positions in their original order.

    It answers a physical question, not a statistical one: two maxima inside one
    diffraction-limited spot CANNOT be two molecules, so counting the pair as a neighbour
    invents the crowding it is meant to measure. See `MIN_SEPARATION_PX` for what that cost
    on real data before this existed.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2:
        return positions
    values = np.asarray(values, dtype=float)
    kept: list[int] = []
    for i in np.argsort(values)[::-1]:
        p = positions[i]
        if all(float(np.hypot(*(p - positions[j]))) >= min_separation_px for j in kept):
            kept.append(int(i))
    return positions[sorted(kept)]


def crowding_column(radius_px: float) -> str:
    """The `density_curve` column holding the crowded share at ``radius_px``.

    RAISES for a radius the curves do not carry, naming the ladder. Interpolating between
    two radii would be a fourth quantity wearing a measurement's name, and this project has
    been bitten by exactly that more than once.
    """
    for r in CROWDING_RADII_PX:
        if abs(float(radius_px) - r) < 1e-9:
            return f"crowded_{r:g}px"
    raise MoviePrepError(
        f"crowding is recorded at {', '.join(format(r, 'g') for r in CROWDING_RADII_PX)} "
        f"px and not at {float(radius_px):g}. Pass one of those, or re-read the movies "
        f"with a different CROWDING_RADII_PX. Values between two radii are not "
        f"interpolated, because the result would not be a measurement.")


def density_curve(movie, *, probe_every: int = 25, k: float = 6.0,
                  size: int = 3,
                  min_separation_px: float = MIN_SEPARATION_PX) -> pd.DataFrame:
    """Spots per frame, median amplitude, and how CROWDED each probed frame is.

    A deliberately plain local-maximum detector, NOT the analysis one. Its job is to show
    the analyst how the field thins; a detector carrying its own threshold policy would
    put a second unstated choice inside the plot meant to inform a first one.

    THE CROWDING COLUMNS ASK `linking_ambiguity`'s QUESTION, EARLIER. For each probed
    frame, the share of detections with another detection within r pixels -- "how often
    would a link at this radius be a guess". It COUNTS rather than modelling, because a
    whole-field Poisson model understates it about fivefold on this data: molecules sit in
    cells covering ~4.5 per cent of the field, so the density that matters is not the
    field's.

    IT IS A WHOLE-FIELD NUMBER AND IT SAYS SO. No outlines exist at this stage -- which is
    the point of measuring here, before tracking and while the crop can still be changed --
    so out-of-cell detections DILUTE it in the ordinary case and INFLATE it where a movie
    carries a pile of out-of-cell signal. Measured, that is the difference between 3.3 and
    36.6 per cent on one lab condition. Both readings are useful and neither is the in-cell
    number; `linking_ambiguity(meshes=...)` is that, once there are outlines.

    Computed from the SAME detections the counts come from, in the same pass, so the movie
    is read once and the two can never disagree about what a spot is.
    """
    from scipy.ndimage import maximum_filter, uniform_filter
    from scipy.spatial import cKDTree

    n = len(movie)
    frames = np.arange(0, n, max(1, int(probe_every)))
    counts, amps, resolved = [], [], []
    crowded = {r: [] for r in CROWDING_RADII_PX}
    for index in frames:
        image = movie.frame(int(index))
        resid = image - uniform_filter(image, 15)
        sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        peaks = (maximum_filter(image, size) == image) & (resid > k * sd)
        values = resid[peaks]
        counts.append(int(values.size))
        amps.append(float(np.median(values)) if values.size else np.nan)
        # RESOLVED FIRST, THEN COUNTED. Merging peaks inside one PSF is what separates a
        # measurement of the field from a measurement of the detector -- see
        # `MIN_SEPARATION_PX`. `n_spots` above is deliberately left uncorrected, because
        # `crop_floor` is calibrated on it and moving it would move every crop floor.
        rows_, cols_ = np.nonzero(peaks)
        pos = np.column_stack([rows_.astype(float), cols_.astype(float)])
        pos = resolve_peaks(pos, values, min_separation_px=min_separation_px)
        resolved.append(int(len(pos)))
        if len(pos) < 2:
            # 0 detections: nothing that COULD be ambiguous, and a 0 there would read as
            # "measured clean". 1: genuinely no competitor, which IS a measurement.
            share = float("nan") if len(pos) == 0 else 0.0
            for r in CROWDING_RADII_PX:
                crowded[r].append(share)
            continue
        nn = cKDTree(pos).query(pos, k=2)[0][:, 1]
        for r in CROWDING_RADII_PX:
            crowded[r].append(float(np.mean(nn <= r)))
    table = pd.DataFrame({"frame": frames + 1, "n_spots": counts,
                          "median_amplitude": amps, "n_resolved": resolved})
    for r in CROWDING_RADII_PX:
        table[f"crowded_{r:g}px"] = crowded[r]
    table.attrs["min_separation_px"] = float(min_separation_px)
    table.attrs["detector"] = (f"local maxima {size}x{size} standing {k:g} robust sd "
                               "above a 15 px local background")
    table.attrs["crowding"] = (
        "share of detections with another detection within r px, counted per probed "
        "frame, whole field (no outlines exist at this stage)")
    return table


def crowding_summary(curves, *, radius_px: float = 5.0, ranges=None):
    """Detections per frame and crowding per movie, over the frames that will be TRACKED.

    ``curves`` maps a movie name to its `MovieCurves` or to a bare `density_curve` table.
    One row per movie; the condition medians and the count of named movies are on
    ``.attrs``.

    THE RANGE IS NOT OPTIONAL IN SPIRIT, ONLY IN THE SIGNATURE. A raw acquisition opens
    with the dense phase the crop exists to remove -- hundreds of overlapping molecules at
    frame 50 -- so a number taken over the whole movie describes the part nobody analyses,
    and every movie would be named. ``ranges`` is ``{movie: (start, stop)}``, the crop set;
    where a movie has none its own crop FLOOR stands in and the row says so, so a number
    computed over one window can never be read as one computed over another.

    **THIS IS NOT `linking_ambiguity` AND MUST NOT BE QUOTED AS IT.** Both count how often
    a detection has another within one linking radius, and they run on different detection
    sets: over the same crop this probe finds **4.45 resolved detections per frame against
    u-track's 2.33** on a lab project, and reads **23.5 per cent crowded at 5 px against
    `linking_ambiguity`'s 4.0**. The extra detections sit inside cells, where the noise
    floor is genuinely higher -- cells autofluoresce at 928 counts against a 597 background
    -- so they are not removable by a threshold that keeps the real molecules. Read this as
    an UPPER BOUND and a relative measure; the number that decides anything is
    `trackmate_run.linking_ambiguity(meshes=...)`, after tracking and with outlines.

    **SO THE FLAG IS RELATIVE.** An absolute cut carried over from those in-cell figures
    named three of four clean MSD movies the first time it was run. What survives the
    difference between the two detectors is the comparison BETWEEN MOVIES OF ONE
    CONDITION, which share a probe, a label and a session -- the same argument
    `noise_outliers` rests on, and the same reason a movie unlike its neighbours is worth
    opening whatever the absolute level.

    AND IT IS WHOLE-FIELD, WHICH CUTS BOTH WAYS. Out-of-cell detections have no
    neighbours, so ordinarily they DILUTE the share; where a movie carries a pile of
    out-of-cell signal they INFLATE it. Measured, 36.6 per cent whole-field against 3.3
    in-cell on one lab condition -- a condition that was not crowded but had one bad movie,
    which is exactly the case this is here to surface.
    """
    column = crowding_column(radius_px)
    ranges = dict(ranges or {})

    rows = []
    for name in sorted(curves):
        entry = curves[name]
        table = getattr(entry, "density", entry)
        floor = getattr(getattr(entry, "floor", None), "earliest_defensible", None)
        if table is None or not len(table):
            continue
        missing = [c for c in (column, "n_resolved") if c not in table.columns]
        if missing:
            raise MoviePrepError(
                f"'{name}' carries no {' or '.join(missing)} column, so crowding cannot "
                f"be read from it. These curves were computed before crowding was "
                f"recorded; re-read the movie (the prep window's Read button) and they "
                f"will be there. Returning zero here would report an unmeasured movie as "
                f"a clean one.")

        window = ranges.get(name)
        if window is not None:
            start_frame = int(window[0])
            stop = int(window[1]) if len(window) > 1 and window[1] else None
            scope = "the crop set"
        elif floor is not None:
            start_frame, stop, scope = int(floor), None, "its crop floor, no range set"
        else:
            start_frame, stop = 1, None
            scope = "the whole acquisition, no range and no floor"

        frames = table["frame"].to_numpy(dtype=float)
        keep = frames >= start_frame
        if stop is not None:
            keep &= frames <= stop
        if not keep.any():
            continue
        share = table[column].to_numpy(dtype=float)[keep]
        found = table["n_resolved"].to_numpy(dtype=float)[keep]
        peaks = table["n_spots"].to_numpy(dtype=float)[keep]
        # AN ALL-EMPTY WINDOW IS UNDEFINED, NOT CLEAN. numpy returns NaN and warns; the
        # guard is here so a legitimate case does not print what reads like a defect.
        med = lambda a: (float("nan") if not np.isfinite(a).any()  # noqa: E731
                         else float(np.nanmedian(a)))
        n_res, n_peak = med(found), med(peaks)
        # THE SHARE IS POOLED, NOT AVERAGED OVER FRAMES, and the first version got this
        # wrong in a way that is invisible on a dense field. A per-frame share over 4
        # detections can only be 0, 0.25, 0.5, 0.75 or 1, so its MEDIAN across frames is 0
        # whenever fewer than half the frames hold a crowded pair -- which reported 0.0 %
        # for every clean MSD movie regardless of what was in them. The quantity meant is
        # "of all detections, how many had a neighbour", so detections are the denominator
        # and a frame with more of them counts for more. It is also what
        # `linking_ambiguity` pools, which is what makes the two readable on one axis.
        weights = np.where(np.isfinite(share), found, np.nan)
        usable = np.isfinite(share) & np.isfinite(weights) & (weights > 0)
        share_pooled = (float(np.sum(share[usable] * weights[usable])
                              / np.sum(weights[usable]))
                        if usable.any() else float("nan"))
        rows.append({
            "movie": name,
            "first_frame": int(start_frame),
            "n_probed": int(keep.sum()),
            "loc_per_frame": n_res,
            "peaks_per_frame": n_peak,
            "split_ratio": (n_peak / n_res) if n_res else float("nan"),
            "crowded_share": share_pooled,
            "measured_over": scope,
        })

    columns = ["movie", "first_frame", "n_probed", "loc_per_frame", "peaks_per_frame",
               "split_ratio", "crowded_share", "measured_over"]
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        for extra in ("density_ratio", "crowding_ratio", "flagged", "reason"):
            out[extra] = []
        out.attrs.update(radius_px=float(radius_px), n_flagged=0,
                         median_loc_per_frame=float("nan"),
                         median_crowded_share=float("nan"),
                         median_split_ratio=float("nan"),
                         scope="whole field", note="no curves to summarise")
        return out

    def _median(name):
        a = out[name].to_numpy(dtype=float)
        return float("nan") if not np.isfinite(a).any() else float(np.nanmedian(a))

    median_share = _median("crowded_share")
    median_loc = _median("loc_per_frame")
    median_split = _median("split_ratio")

    out["density_ratio"] = (out["loc_per_frame"] / median_loc
                            if median_loc and np.isfinite(median_loc) else np.nan)
    out["crowding_ratio"] = (out["crowded_share"] / median_share
                             if median_share and np.isfinite(median_share) else np.nan)

    enough = len(out) >= CROWDING_MIN_MOVIES
    reasons, flags = [], []
    for _, row in out.iterrows():
        if not enough:
            reasons.append(f"not compared: {len(out)} movie(s), and a ratio against the "
                           f"median of fewer than {CROWDING_MIN_MOVIES} is a ratio "
                           f"against one movie")
            flags.append(False)
            continue
        why = []
        dens_ratio = float(row["density_ratio"])
        share = float(row["crowded_share"])
        crowd_ratio = float(row["crowding_ratio"])
        if np.isfinite(dens_ratio) and dens_ratio >= DENSITY_FLAG_RATIO:
            why.append(f"{fold_words(dens_ratio, 1, of=True)} the condition's detections per frame "
                       f"({row['loc_per_frame']:.0f} against {median_loc:.0f})")
        if (np.isfinite(share) and np.isfinite(crowd_ratio)
                and median_share >= CROWDING_RATIO_FLOOR
                and crowd_ratio >= CROWDING_FLAG_RATIO):
            why.append(f"{fold_words(crowd_ratio, 1, of=True)} its crowding at {float(radius_px):g} px "
                       f"({100 * share:.0f} % against {100 * median_share:.0f} %)")
        reasons.append("; ".join(why))
        flags.append(bool(why))
    out["flagged"] = flags
    out["reason"] = reasons

    out.attrs.update(
        radius_px=float(radius_px),
        median_loc_per_frame=median_loc,
        median_crowded_share=median_share,
        median_split_ratio=median_split,
        n_flagged=int(out["flagged"].sum()),
        scope="whole field",
        note=(f"median {median_loc:.1f} resolved detections/frame over the tracked range, "
              f"{100 * median_share:.0f} % of them with another within "
              f"{float(radius_px):g} px. Whole field, this probe's own detector: an upper "
              f"bound on linking ambiguity, not comparable with the in-cell figure."))
    return out


def crop_floor(movie, *, spots_per_frame: float = 10.0,
               amplitude_tolerance: float = 1.3, flat_tolerance: float = FLAT_TOLERANCE,
               probe_every: int = 25, density=None, means=None) -> CropFloor:
    """The earliest start the images defend. Read `CropFloor` before using it.

    Returns floors, never a crop. ``spots_per_frame`` carries a stated default because
    u-track's own ambiguity criterion on this field is about 209, twenty times looser,
    so 10 is conservative by construction rather than tuned to anything.
    """
    # REUSED WHEN THE CALLER ALREADY HAS IT. Detecting spots over the movie is the
    # expensive half of this (3.1 s against 1.1 s on a 4500-frame .nd2), and the
    # window that draws the density curve has just computed the same thing.
    if density is None:
        density = density_curve(movie, probe_every=probe_every)
    frames = density["frame"].to_numpy()
    counts = density["n_spots"].to_numpy(dtype=float)
    amps = density["median_amplitude"].to_numpy(dtype=float)

    # THE FRAME MEANS ARE NEEDED BEFORE THE CRITERIA, not after them, because an UNLIT
    # probe point is not a measurement of anything and must be dropped from all three.
    # Reported: a movie whose laser failed mid-acquisition gave a floor of 376 -- inside
    # the dark window -- because the detector found 200-320 noise spots per frame there.
    # See `lit_frames`.
    means = (np.asarray(movie.frame_means(), dtype=float) if means is None
             else np.asarray(means, dtype=float))
    probe_index = np.clip(frames.astype(int) - 1, 0, means.size - 1)
    # Judged on the FULL curve, then read at the probes: a switch is a step between
    # neighbouring frames, which a 25-frame subsample blurs.
    all_lit = lit_frames(means)
    lit = all_lit[probe_index]
    n_unlit = int((~lit).sum())
    if not lit.any():
        raise MoviePrepError(
            "every probed frame sits at the camera's dark level, so this acquisition was "
            "never illuminated where it was sampled. No crop floor can be read from it")
    # Dropped rather than masked, so nothing downstream can take a maximum or a median
    # over a noise spike. `counts` keeps its shape for `frames` to index.
    counts = np.where(lit, counts, np.nan)
    amps = np.where(lit, amps, np.nan)

    # A DENSITY CRITERION THAT CANNOT BE EVALUATED IS NOT A REASON TO REFUSE THE FLOOR.
    # It used to be, and on real data that cost two movies in ten: their bleaching was
    # normal, their amplitude and field-mean floors were perfectly computable, and the
    # density criterion took both down with it.
    #
    # `strictest` is the conjunction of the criteria that CAN be measured. Refusing the
    # whole answer because one of three is unavailable throws away a floor the images do
    # support, so the missing one is NAMED on the result and the rest still stand. Only a
    # movie where nothing at all can be measured is refused, at the end.
    #
    # The distinction the old guard protected is kept: "I found no spots" and "the field
    # has thinned" produce the same array and must not produce the same answer. Neither
    # now yields a confident density floor; both say so instead.
    unavailable: dict = {}
    by_density = None
    peak_count = float(np.nanmax(counts)) if np.isfinite(counts).any() else 0.0
    # THE PEAK IS NEEDED BY THE AMPLITUDE CRITERION TOO, so it is found before the
    # density branch rather than inside it. Defining it only where density succeeds left
    # `after` undefined for exactly the movies this change exists to rescue.
    peak = int(np.nanargmax(counts)) if np.isfinite(counts).any() else 0
    after = np.arange(counts.size) > peak
    if not np.isfinite(counts).any() or peak_count <= 0:
        # NOTHING ANYWHERE IS STILL FATAL, and this is the distinction the original
        # guard was protecting: "I found no spots" and "the field has thinned" produce
        # the same array and must not produce the same answer. A sparse field is a real
        # measurement and keeps its other criteria; an empty one has no measurement at
        # all, and the field mean would otherwise hand back a confident floor of 1 for a
        # movie holding nothing.
        raise MoviePrepError(
            "no spots were detected anywhere in this movie, so nothing here describes "
            "a field thinning out. Either this is not a single-molecule acquisition, or "
            "the probe's sensitivity is far below what this data needs")
    if peak_count <= spots_per_frame:
        unavailable["density"] = (
            f"the detected spot count never exceeds {spots_per_frame}/frame (peak "
            f"{peak_count:.0f}), so no density can fall below it. This reflects how many "
            f"molecules were detected, not the movie. Lower `spots_per_frame`, or raise "
            f"the detector's sensitivity, for a field this sparse")
    else:
        hit = np.flatnonzero(after & (counts <= spots_per_frame))
        if hit.size == 0:
            unavailable["density"] = (
                f"spot density never falls back to {spots_per_frame}/frame, so the field "
                f"never bleached down to a linkable density within this movie")
        else:
            by_density = int(frames[hit[0]])

    # NAMED, NOT FATAL, for the same reason as density above: with no amplitudes the
    # other two criteria are still perfectly measurable, and a movie with no detections
    # at all is caught at the end where it belongs.
    finite_amps = amps[np.isfinite(amps)]
    by_amplitude = None
    if peak_count <= spots_per_frame:
        # NO DENSE PHASE MEANS NOTHING TO BE "AFTER". This criterion anchors on
        # `argmax(counts)`, and on a field that never exceeds the limit that index is
        # wherever the first tie happened to land -- arbitrary, and it moved a real
        # movie's floor from 26 to 3751 once a 13-frame noise spike stopped being the
        # maximum. The density branch above already refuses this case for its own
        # criterion; the amplitude one shares the anchor and must refuse it too.
        unavailable["amplitude"] = (
            f"the detected spot count never exceeds {spots_per_frame}/frame (peak "
            f"{peak_count:.0f}), so there is no dense phase for brightness to settle "
            f"after: this criterion anchors on the density peak and that peak is not "
            f"a real one here")
    elif finite_amps.size == 0:
        unavailable["amplitude"] = (
            "no spot amplitudes could be measured anywhere in this movie, so there is "
            "no brightness to watch settle")
    else:
        floor_amp = float(np.median(finite_amps[-max(4, finite_amps.size // 10):]))
        settled = np.flatnonzero(after & np.isfinite(amps)
                                 & (amps <= amplitude_tolerance * floor_amp))
        by_amplitude = int(frames[settled[0]]) if settled.size else None
        if by_amplitude is None:
            unavailable["amplitude"] = (
                "spot brightness never settles to within "
                f"{fold_words(amplitude_tolerance, of=True)} its end-of-movie level")

    # supplied or measured, exactly as `density` already is, and for the same reason.
    # Both are written to `bleaching/*.csv` when a crop is saved, so a folder opened a
    # second time can rebuild this floor from what it already computed instead of
    # reading every acquisition again at about 7.7 s each. Without this parameter the
    # cached path could rebuild two of the three floors and had to fall back to the
    # movie for the third, which is the whole cost.
    tail = means[-max(10, means.size // 20):]
    level = float(np.median(tail)) + float(np.std(tail))
    # ON LIT FRAMES ONLY, for the same reason as the other two: a dark frame is below
    # every level there is, so without this the rule reports the moment the laser died as
    # the moment the field thinned.
    below = np.flatnonzero((np.arange(means.size) > int(np.argmax(np.where(
        all_lit, means, -np.inf))))
        & (means <= level) & all_lit)
    by_field_mean = int(below[0]) + 1 if below.size else int(means.size)

    n_frames = int(means.size) if movie is None else len(movie)
    # NOTHING MEASURABLE IS STILL A REFUSAL. One criterion missing is an answer built
    # from the rest; all of them missing is not an answer at all, and returning a floor
    # there would be the confident number this function has always refused to invent.
    if by_density is None and by_amplitude is None and by_field_mean is None:
        raise MoviePrepError(
            "no crop floor can be read from this movie: "
            + "; ".join(f"{k}: {v}" for k, v in unavailable.items()))
    # THE FLOOR ITSELF. Smoothed first, because a single noisy frame would otherwise
    # break a genuinely flat stretch in two and halve it; the median window is short
    # against any real regime.
    smooth = pd.Series(means).rolling(51, center=True, min_periods=1).median().to_numpy()
    # ONLY THE FINAL ILLUMINATION EPISODE MAY SUPPLY IT. Masking the dark frames alone
    # stops a stretch STRADDLING the gap; it does not stop one lying wholly before it
    # from winning on length, which at a tight tolerance is exactly what happens.
    episode = last_illumination_start(means)
    usable = all_lit.copy()
    usable[:episode] = False
    flat = longest_flat_stretch(smooth, tolerance=flat_tolerance, usable=usable)
    by_flat = None if flat is None else int(flat[0]) + 1
    dark_level = leading_dark_level(means)
    flat_band_share = None
    if flat is not None and dark_level is not None:
        level = float(np.median(smooth[flat[0]:flat[1]]))
        if level > dark_level:
            flat_band_share = float(flat_tolerance) * level / (level - dark_level)
    if by_flat is None:
        unavailable["flat"] = (
            f"no stretch of this movie stays within {100 * flat_tolerance:g} % of its own "
            f"level for {FLAT_MIN_FRAMES} frames, so the floor falls back to the "
            f"criteria that could be measured")
    if n_unlit:
        unavailable["unlit"] = (
            f"{n_unlit} of {frames.size} probed frames sat at the camera's dark level "
            f"and were dropped: a LoG detector still returns spots in an unlit frame "
            f"(measured, 200-320 of them at a twentieth of a molecule's amplitude), so "
            f"they are noise rather than a thinned field")
    return CropFloor(by_density=by_density, by_amplitude=by_amplitude,
                     by_field_mean=by_field_mean, n_frames=n_frames,
                     spots_per_frame_limit=float(spots_per_frame),
                     unavailable=unavailable, n_unlit_probes=n_unlit,
                     by_flat=by_flat, flat_tolerance=float(flat_tolerance),
                     dark_level=dark_level, flat_band_share=flat_band_share)


def sd_projection(frames) -> np.ndarray:
    """The macro's `Z Project... [Standard Deviation]`, as float.

    The quantitative version. What the macro writes to `sum/` is this, then display-
    stretched and converted to 16-bit, which makes its numbers depend on a display
    setting. Keep this one for anything that measures.

    ImageJ's Z-projector uses the SAMPLE standard deviation (n-1), so this does too.
    """
    return projection(frames, kind="sd")


#: What `Z Project...` can make of a stack. The macro hard-codes Standard Deviation and
#: every `sum/` file in this project's existing data IS one: the folder's NAME is the
#: only thing that says sum. ImageJ's own dialog offers both, so both are offered here.
PROJECTIONS = ("sd", "sum")


def projection(frames, *, kind: str = "sd") -> np.ndarray:
    """A Z projection of a stack, as float.

    ``kind`` is "sd" (the macro's, and what every existing `sum/` file contains) or
    "sum". SD is the DEFAULT because it is what produced the data already on disk, and
    changing that silently would make two differently-made images share one filename.

    Why SD looks like tracks: it lights up the pixels that VARY, which is where
    molecules moved. A true sum is dominated by the static background instead.
    """
    if kind not in PROJECTIONS:
        raise MoviePrepError(f"projection kind must be one of {PROJECTIONS}, got "
                             f"{kind!r}")
    block = np.asarray(frames, dtype=np.float64)
    # A SAMPLE standard deviation needs two; a sum needs one.
    needed = 2 if kind == "sd" else 1
    if block.ndim != 3 or block.shape[0] < needed:
        raise MoviePrepError(
            f"a {kind} projection needs at least {needed} "
            f"frame{'s' if needed > 1 else ''}, got shape {block.shape}")
    if kind == "sd":
        return block.std(axis=0, ddof=1).astype(np.float32)
    return block.sum(axis=0).astype(np.float32)


def display_stretch(image, *, saturated: float) -> tuple[np.ndarray, float, float]:
    """ImageJ's `Enhance Contrast` + `16-bit`, returning the image AND its limits.

    ImageJ splits ``saturated`` (a PERCENT) between the two ends, so the cut is
    ``saturated / 200`` of the pixels at each end. The limits are returned because they
    are what makes the result reproducible: without them the output is a picture whose
    scale nobody recorded.

    Display only. Anything that measures should take the float image instead.
    """
    values = np.asarray(image, dtype=np.float64)
    if not 0.0 <= saturated < 100.0:
        raise MoviePrepError("saturated is a percent in [0, 100)")
    tail = saturated / 200.0
    lo = float(np.quantile(values, tail)) if tail > 0 else float(values.min())
    hi = float(np.quantile(values, 1.0 - tail)) if tail > 0 else float(values.max())
    if hi <= lo:
        raise MoviePrepError(
            f"the stretch limits collapsed ({lo:g} to {hi:g}): this image has no "
            "contrast to enhance")
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 65535.0
    return np.rint(scaled).astype(np.uint16), lo, hi


#: What this tool stamps into every TIFF it writes (the TIFF `Software` tag).
SOFTWARE_TAG = "SMTracker"
#: Software tags of files THIS TOOL wrote before it stamped its own name: tifffile's
#: default. Fiji's macro writes none (its files say `ImageJ=` in the description) and the
#: MetaMorph scope writes `VisiView`.
_OWN_SOFTWARE = (SOFTWARE_TAG, "tifffile")


def foreign_file(path) -> str | None:
    """Who wrote `path`, when it exists and was NOT written by this tool; None otherwise.

    THE TOOL OVERWRITES ONLY WHAT IT WROTE. Named after its file, a MetaMorph movie wrote
    `sum/series001_YFPLaser.tif`, which could never land on the macro's
    `sum/series001.tif`; named `series001` (`rawmovie.movie_name`), it would replace the
    macro's projection and, through `prepare_movie`, the macro's prepared STACK. Neither
    writer checked. A file another program wrote is kept and the skip is named.
    """
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return None
    try:
        import tifffile

        with tifffile.TiffFile(path) as handle:
            tags = handle.pages[0].tags
            software = str(tags["Software"].value) if "Software" in tags else ""
            description = (str(tags["ImageDescription"].value)
                           if "ImageDescription" in tags else "")
    except Exception:                                  # noqa: BLE001 - unknown = keep
        return "an unreadable or non-TIFF file"
    if software.startswith(_OWN_SOFTWARE):
        return None
    if description.startswith("ImageJ="):
        return "Fiji / ImageJ"
    return software or "another program"


def prepare_movie(raw_path, out_root, config: PrepConfig, *, name: str | None = None,
                  write: bool = True) -> dict:
    """One raw acquisition to the four outputs the macro produces.

    The cropped stack is written UNMODIFIED: `Enhance Contrast` in the macro sets a
    lookup table and not a pixel, so anything else here would silently change the data
    every downstream number is computed from.
    """
    import tifffile

    from .rawmovie import read_raw_movie

    raw_path = Path(raw_path)
    out_root = Path(out_root)
    movie = read_raw_movie(raw_path)
    if not name:
        from .rawmovie import movie_name

        name = movie_name(raw_path)

    n_acquired = int(movie.meta.n_frames)
    if config.crop_end > n_acquired:
        raise MoviePrepError(
            f"{name}: crop {config.crop_start}-{config.crop_end} runs past the "
            f"acquisition's {n_acquired} frames. The range is per movie, not per batch: "
            "these acquisitions differ in length")

    full = bleaching_curve(movie)
    cropped = bleaching_curve(movie, crop=(config.crop_start, config.crop_end))

    # 1-based inclusive -> python slice
    block = np.asarray(movie._frames[config.crop_start - 1:config.crop_end])
    projected = projection(block, kind=config.projection)
    stretched, lo, hi = display_stretch(projected, saturated=config.sum_saturated)

    result = {
        "movie": name,
        "raw": str(raw_path),
        "n_frames_acquired": n_acquired,
        "n_frames_kept": int(block.shape[0]),
        "crop_start": config.crop_start,
        "crop_end": config.crop_end,
        "sum_stretch_lo": lo,
        "sum_stretch_hi": hi,
        "config": config.stamp(),
        "tau_note": TAU_NOTE,
        "written": [],
    }
    if not write:
        result["curve_full"] = full
        result["curve_cropped"] = cropped
        result["projection"] = projection
        result["stack"] = block
        return result

    # ONE MULTI-PAGE TIFF PER MOVIE, which is what the macro writes:
    #     saveAs("tiff", pathB + title + fString)
    # after `Properties... unit=[um] pixel_width=0.106 frame=[0.020479 sec]`.
    #
    # THIS WAS WRONG AND THE EXPERIMENTER CAUGHT IT: an earlier version wrote a FOLDER
    # of single-frame TIFFs, `tif/<movie>/<movie>/*.tif`. That layout is real and this
    # project reads it, but **u-track produces it**, from the multi-page file, as its
    # own working copy. Writing it here made preparation impersonate the tracker, and
    # skipped the file u-track is actually given.
    #
    # THE CALIBRATION IS WRITTEN IN, for the same reason the macro sets Properties
    # before saving: it is where the pixel size and frame interval came from in the
    # first place, and `optics.pixel_size_from_project` reads them back out of exactly
    # this file. Measured: the single frames carry `(72, 1)` inch, the library default,
    # because that calibration did not survive a frame-by-frame export.
    stack_dir = out_root / "tif"
    stack_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stack_dir / f"{name}.tif"
    writer = foreign_file(stack_path)
    if writer is not None:
        raise MoviePrepError(
            f"{stack_path} was written by {writer}; not overwritten. Move it aside to "
            f"prepare this movie again")
    tifffile.imwrite(
        stack_path, block, imagej=True, software=SOFTWARE_TAG,
        resolution=(1000.0 / config.pixel_size_nm, 1000.0 / config.pixel_size_nm),
        metadata={"unit": "um", "finterval": config.frame_interval_s,
                  "frames": int(block.shape[0]), "axes": "TYX"})
    result["written"].append(str(stack_path))

    bleach_dir = out_root / "bleaching"
    bleach_dir.mkdir(parents=True, exist_ok=True)
    for label, table in (("full", full), ("cropped", cropped)):
        path = bleach_dir / f"{name}_{label}.csv"
        table.to_csv(path, index=False)
        result["written"].append(str(path))

    # AND THE PICTURE, because the CSV was meant to ADD to the macro's output rather
    # than replace it. Nobody opens a spreadsheet to see whether a movie bleached down.
    # Drawn FROM the CSVs just written, so the record and the view cannot disagree.
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from .plots import draw_bleaching_pair

        figure = Figure(figsize=(7.0, 4.4), dpi=130, tight_layout=True)
        draw_bleaching_pair(figure, full, cropped, movie=name,
                            crop=(config.crop_start, config.crop_end),
                            frame_interval_s=config.frame_interval_s)
        picture = bleach_dir / f"{name}.png"
        figure.savefig(picture)
        result["written"].append(str(picture))
    except Exception as exc:                       # noqa: BLE001 - named, never fatal
        # A missing plotting stack must not cost the stacks and the CSVs, which are the
        # part the analysis needs. The omission is REPORTED rather than silent.
        result["curve_picture_error"] = f"{type(exc).__name__}: {exc}"

    sum_dir = out_root / "sum"
    sum_dir.mkdir(parents=True, exist_ok=True)
    sum_path = sum_dir / f"{name}.tif"
    writer = foreign_file(sum_path)
    if writer is None:
        tifffile.imwrite(sum_path, stretched, software=SOFTWARE_TAG)
        result["written"].append(str(sum_path))
    else:
        result.setdefault("kept", []).append(f"{sum_path} ({writer})")
    # The quantitative copy, beside the compatibility one. Named so nobody has to know
    # that `sum/` is neither a sum nor a measurement.
    # the float copy is OPT-IN. The macro saves ONE image per movie: the projection,
    # display-stretched and 16-bit, and a second `_sd_float.tif` beside it read as an
    # extra output nobody asked for. It also LOOKS BLANK in most viewers, which is not a
    # defect: it is float32 with a range of tens of counts, and a viewer assuming 0-255
    # shows white. It is the only QUANTITATIVE version (the 16-bit one has a display
    # setting baked into its pixel values), so it stays available and stays off.
    if config.write_float_projection:
        float_path = sum_dir / f"{name}_{config.projection}_float.tif"
        tifffile.imwrite(float_path, projected, software=SOFTWARE_TAG)
        result["written"].append(str(float_path))
    return result


#: Where a set of per-movie crops is written, beside the raw acquisitions it describes.
CROPS_FILENAME = "crops.json"


def crops_locations(root, output_root=None):
    """Where a crop set may live, most-preferred first. Pure pathlib.

    `bleaching/` IS THE HOME, ruled by the experimenter: the ranges and the curves that
    justify them are one record, and scattering them puts a lone json in a folder full of
    unrelated things. The other two are READ for what already exists, beside the movies
    (before the outputs moved) and at the condition root (before they moved again).
    Ignoring either would re-ask a question already answered.
    """
    out = []
    for folder in (output_root, root):
        if folder is None:
            continue
        folder = Path(folder)
        for candidate in (folder / "bleaching", folder):
            if candidate not in out:
                out.append(candidate)
    return out


def find_crops(root, output_root=None):
    """An existing `crops.json` for this folder, or None.

    Moved out of the crop panel, which is a Qt widget: the TRACKING stage no longer
    chooses crops (it only READS the ranges the preparation stage recorded) so needing
    a widget alive to locate a file made a pure lookup depend on a window.
    """
    for folder in crops_locations(root, output_root):
        candidate = Path(folder) / CROPS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def save_crops(path, crops, *, pixel_size_nm: float, frame_interval_s: float,
               label: str | None = None, emission_nm: float | None = None,
               numerical_aperture: float | None = None) -> None:
    """Record the per-movie crop ranges, so the TRACKING stage can reuse them.

    The crop is the expensive decision in this pipeline - it is the one thing the images
    do not determine and a human has to look at every movie to make. Asking for it twice,
    once to prepare and again to track from raw, would invite the two to disagree, and a
    disagreement there is invisible downstream: two datasets, same name.

    The calibration travels with it for the same reason. Ranges alone would let a run be
    reproduced at the wrong pixel size or frame interval, which is exactly the pair this
    project keeps having to correct.
    """
    import json

    path = Path(path)
    if not crops:
        raise MoviePrepError("refusing to write an empty crop set")
    payload = {
        "crops": {str(k): [int(v[0]), int(v[1])] for k, v in crops.items()},
        "pixel_size_nm": float(pixel_size_nm),
        "frame_interval_s": float(frame_interval_s),
        # THE OPTICS TRAVEL TOO, and leaving them out was inconsistent rather than
        # deliberate: NA and the emission wavelength set the PSF WIDTH, which sets the
        # localisation floor, the number deciding whether a slow state's D is a
        # measurement or a bound. That is at least as consequential as the pixel size
        # already recorded here, and a run reproduced without it is reproduced without
        # knowing what it could resolve. `None` where nothing was set: absent and
        # guessed must stay distinguishable.
        "label": label,
        "emission_nm": None if emission_nm is None else float(emission_nm),
        "numerical_aperture": (None if numerical_aperture is None
                               else float(numerical_aperture)),
        "note": ("1-based inclusive frame ranges, one per movie. The crop is a per-movie",
                 "judgement the images do not determine; see CropFloor."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


#: The calibration half of a crop set: everything `save_crops` writes that is NOT a
#: frame range. Named once so the updater below and the window agree about what a
#: "settings change" is.
CROPS_CALIBRATION_KEYS = ("pixel_size_nm", "frame_interval_s", "label", "emission_nm",
                          "numerical_aperture")


def update_crops_calibration(path, **values) -> dict:
    """Rewrite ONLY the calibration in an existing crop set. Returns what changed.

    THE ORDER OF THE WORK SHOULD NOT DECIDE WHAT IS ON DISK. The calibration is written
    INTO `crops.json` at save time, so setting a fluorophore AFTER saving left the file
    claiming `null` while the window showed mNeonGreen -- reported exactly that way, and
    it is this project's own "two datasets, one name" with the two a few minutes apart
    rather than two folders.

    The RANGES ARE NEVER TOUCHED. They are the expensive human decision this file exists
    to record, and a calibration edit is not an occasion to rewrite them; that also
    means this cannot half-write a crop somebody is still deciding.

    Unknown keys are refused rather than added, so a typo becomes an error instead of a
    field nothing reads. A key whose value is unchanged is not reported as a change.
    """
    import json

    path = Path(path)
    unknown = sorted(set(values) - set(CROPS_CALIBRATION_KEYS))
    if unknown:
        raise MoviePrepError(
            f"not calibration fields: {', '.join(unknown)}. "
            f"Known: {', '.join(CROPS_CALIBRATION_KEYS)}")
    payload = load_crops(path)
    changed = {}
    for key, value in values.items():
        # A REQUIRED FIELD IS NEVER CLEARED FROM HERE. `load_crops` refuses a payload
        # with no pixel size or frame interval -- neither has a default anywhere in this
        # project -- so writing None over one turns a good crop set into an unreadable
        # file. An empty cell in a settings grid is "not filled in yet", not "delete what
        # is on disk", and the two must not be the same gesture.
        if value is None and key in ("pixel_size_nm", "frame_interval_s"):
            continue
        if key in ("pixel_size_nm", "frame_interval_s", "emission_nm",
                   "numerical_aperture"):
            value = None if value is None else float(value)
        elif value is not None:
            value = str(value)
        if payload.get(key) != value:
            changed[key] = (payload.get(key), value)
            payload[key] = value
    if not changed:
        return {}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return changed


def load_crops(path) -> dict:
    """Read back what `save_crops` wrote. Returns the whole payload, not just ranges,
    because a caller that ignores the calibration is the failure this guards against."""
    import json

    path = Path(path)
    if not path.exists():
        raise MoviePrepError(f"no crop set at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise MoviePrepError(f"{path}: not readable JSON ({exc})") from exc
    crops = payload.get("crops")
    if not isinstance(crops, dict) or not crops:
        raise MoviePrepError(f"{path}: no crops recorded")
    out = {}
    for movie, span in crops.items():
        try:
            lo, hi = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise MoviePrepError(f"{path}: {movie!r} has no usable range") from exc
        if hi < lo or lo < 1:
            raise MoviePrepError(
                f"{path}: {movie!r} range {lo}-{hi} is not 1-based increasing")
        out[movie] = (lo, hi)
    # UNDER MOVIE NAMES, so a set saved under a MetaMorph file's stem is found again.
    from .rawmovie import movie_named_crops

    out = movie_named_crops(out, path)
    for key in ("pixel_size_nm", "frame_interval_s"):
        if not payload.get(key):
            raise MoviePrepError(f"{path}: {key} missing; neither has a default")
    # THE WHOLE PAYLOAD, WHICH IS WHAT THE DOCSTRING ALWAYS CLAIMED. It used to rebuild
    # a dict of three known keys, so anything ADDED to the file was silently dropped on
    # read, the optics went in and never came back out, and the window looked as though
    # it had not saved them. A reader that filters to what it knew about when it was
    # written cannot be extended; the validated fields are overwritten onto a copy of
    # the payload instead, so new ones travel and the checked ones stay checked.
    result = dict(payload)
    result["crops"] = out
    result["pixel_size_nm"] = float(payload["pixel_size_nm"])
    result["frame_interval_s"] = float(payload["frame_interval_s"])
    return result


def prepare_batch(raw_dir, out_root, crops, *, pixel_size_nm: float,
                  frame_interval_s: float, projection: str = "sd",
                  on_progress=None) -> pd.DataFrame:
    """Prepare many movies, each with ITS OWN crop.

    ``crops`` maps movie stem -> (start, end). A mapping rather than one range, because
    the crop is a per-movie physical judgement and the batch that shares one is the
    exception. A movie with no entry is REFUSED by name rather than given the batch's
    range: silently preparing it under somebody else's crop is how a stack ends up with
    frames nobody looked at.

    A movie that FAILS is kept as a row with its error, the same rule as `run_batch`.
    """
    raw_dir = Path(raw_dir)
    # DERIVED, not repeated. The two microscopes here write .nd2 (Nikon/NIS) and .stk
    # (MetaMorph), and already-extracted stacks arrive as .tif/.tiff, but which suffixes
    # are READABLE is `rawmovie`'s fact, not this module's. A second hard-coded copy
    # drifts the moment a reader is added, and it drifts in the worst direction: the scan
    # finds a file that the reader then refuses, which reads as a corrupt movie rather
    # than as a missing reader.
    from .rawmovie import _READERS, discover_raw_movies

    readable = set(_READERS)
    found, _companions = discover_raw_movies(raw_dir)
    if not found:
        raise MoviePrepError(
            f"no raw movies in {raw_dir}; readable suffixes are "
            f"{', '.join(sorted(readable))}")

    from .rawmovie import movie_name

    names = {p: movie_name(p) for p in found}
    missing = [names[p] for p in found if names[p] not in crops]
    if missing:
        raise MoviePrepError(
            f"no crop given for {len(missing)} movie(s): {', '.join(missing[:8])}"
            + (" ..." if len(missing) > 8 else "")
            + ". The crop is a per-movie judgement: read each bleaching curve rather "
              "than reusing one range")

    rows = []
    for index, path in enumerate(found, start=1):
        start, end = crops[names[path]]
        row = {"movie": names[path], "raw": str(path), "ok": False, "error": "",
               "n_frames_kept": None, "crop_start": start, "crop_end": end}
        try:
            config = PrepConfig(crop_start=int(start), crop_end=int(end),
                                pixel_size_nm=pixel_size_nm,
                                frame_interval_s=frame_interval_s,
                                projection=projection)
            got = prepare_movie(path, out_root, config)
            row.update(ok=True, n_frames_kept=got["n_frames_kept"])
        except Exception as exc:                       # noqa: BLE001 - kept and named
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if on_progress is not None:
            on_progress(index, len(found), row)

    table = pd.DataFrame(rows)
    table.attrs["tau_note"] = TAU_NOTE
    table.attrs["n_failed"] = int((~table["ok"]).sum())
    return table


@dataclass(frozen=True)
class MovieCurves:
    """Everything the crop decision needs about one movie, computed once.

    The wait was the point of this. Measured on a 4500-frame `.nd2`: opening 1.2 s,
    bleaching curve 2.3 s, density curve 1.1 s, floor 3.1 s, and the floor was
    recomputing the density curve the window had just drawn. Bundling them removes the
    duplicate and gives the window one thing to cache, so a movie is slow once instead
    of on every click.
    """

    movie: str
    n_frames: int
    bleaching: "pd.DataFrame"
    density: "pd.DataFrame"
    floor: CropFloor


def movie_curves(movie, *, name=None, probe_every: int = 25,
                 spots_per_frame: float = 10.0,
                 amplitude_tolerance: float = 1.3,
                 flat_tolerance: float = FLAT_TOLERANCE) -> MovieCurves:
    """The bleaching curve, the density curve and the crop floor, in one pass."""
    density = density_curve(movie, probe_every=probe_every)
    return MovieCurves(
        movie=str(name if name is not None else getattr(movie, "path", "movie")),
        n_frames=int(len(movie)),
        bleaching=bleaching_curve(movie),
        density=density,
        floor=crop_floor(movie, spots_per_frame=spots_per_frame,
                         amplitude_tolerance=amplitude_tolerance,
                         flat_tolerance=flat_tolerance,
                         probe_every=probe_every, density=density),
    )


#: Which of the floor's three candidate starts a proposal uses. They DISAGREE, and by a
#: lot: measured on one 4500-frame movie, 176 (density), 276 (amplitude) and 1127 (field
#: mean): sixfold apart. That is why the criterion is named on every proposal and why
#: nothing applies one silently.
#: "strictest" FIRST, because it is the only one that satisfies all three
#: criteria; the individual rules stay so a floor can be read on its own.
#: "strictest" replaces "defensible", which said the right thing in a word nobody
#: reaches for. It is the LATEST of the three floors, i.e. the one that satisfies every
#: criterion rather than one of them. The old name is still accepted so a recorded
#: proposal keeps loading.
CROP_START_RULES = ("strictest", "density", "amplitude", "field_mean")

#: The fewest frames a proposed range may keep: the shortest track the analysis keeps has
#: five localisations (u-track's minTrackLen 5, `canonical`'s filter), so a range shorter
#: than that can hold no track at all.
MIN_PROPOSED_FRAMES = 5
CROP_START_RULE_ALIASES = {"defensible": "strictest"}


def propose_crop(curves: MovieCurves, *, start_rule: str,
                 end_min_spots_per_frame: float | None = None,
                 end_frame: int | None = None,
                 start_frame: int | None = None) -> dict:
    """A crop range PROPOSED from a stated rule, never applied on its own.

    The crop is the scientific decision in this pipeline, so this returns a suggestion
    with the rule that produced it and the two numbers a reader needs to judge it. The
    window shows it and the analyst edits or accepts; it is not a default, because the
    three rules disagree sixfold and picking one for somebody is picking their answer.

    ``end_min_spots_per_frame`` cuts the TAIL: past the point where the field has
    bleached below that density there is more noise than signal, and those frames add
    background and false links without adding molecules. ``None`` keeps the whole tail.

    ``end_frame`` cuts EVERY movie at one frame instead, and wins over the density rule.
    Asked for as the more reliable cut: one number for the whole set, chosen by eye on
    the curves, where a density threshold lands wherever each movie's noise puts it. A
    movie shorter than the frame keeps its own last frame and the reason says so; a frame
    at or before the proposed start is refused by name rather than inverted.

    ``start_frame`` is the same for the START, and wins over ``start_rule``: asked for
    beside the common end ("why no start cut option we can apply for all movies"). The
    rule's own floor is still reported under ``alternatives``, so a common start that
    sits BEFORE what the curve defends can be seen to. A movie that ends at or before
    the frame is refused by name.
    """
    import numpy as np

    start_rule = CROP_START_RULE_ALIASES.get(start_rule, start_rule)
    if start_rule not in CROP_START_RULES:
        raise MoviePrepError(
            f"start_rule={start_rule!r} is not one of {CROP_START_RULES}")
    # "strictest" IS THE CONJUNCTION, AND PROPOSING ANYTHING ELSE CONTRADICTED THE
    # PANEL'S OWN LABEL. `CropFloor.earliest_defensible` is the LATEST of the three
    # floors precisely because every criterion must be met, not merely one, and the
    # window drew "floor 1127 (earliest defensible)" beside a proposed start of 176,
    # taken from the density rule alone. One of those two numbers was wrong to show,
    # and it was the proposal: a start below the earliest defensible start is, by the
    # definition on the class, not defensible.
    if start_rule == "strictest":
        start = curves.floor.earliest_defensible
    else:
        start = getattr(curves.floor, f"by_{start_rule}")
    start_reason = ("the latest of all three floors" if start_rule == "strictest"
                    else f"crop floor by {start_rule.replace('_', ' ')}")
    if start_frame is not None:
        start_frame = int(start_frame)
        if start_frame >= curves.n_frames:
            raise MoviePrepError(
                f"{curves.movie}: the common start frame {start_frame} is not before its "
                f"last frame {curves.n_frames}")
        start = max(1, start_frame)
        start_reason = f"every movie starts at frame {start}"
    if start is None:
        raise MoviePrepError(
            f"the '{start_rule}' rule found no floor in {curves.movie}: it did not "
            f"settle anywhere in {curves.n_frames} frames. Try another rule or read the "
            f"curve.")

    end = curves.n_frames
    end_reason = "the last frame"
    if end_frame is not None:
        end_frame = int(end_frame)
        if end_frame <= int(start):
            raise MoviePrepError(
                f"{curves.movie}: the common end frame {end_frame} is not after its start "
                f"{int(start)}")
        end = min(end_frame, curves.n_frames)
        end_reason = (f"every movie cut at frame {end_frame}" if end == end_frame else
                      f"the movie ends at frame {curves.n_frames}, before {end_frame}")
    elif end_min_spots_per_frame is not None:
        frames = np.asarray(curves.density["frame"], dtype=float)
        spots = np.asarray(curves.density["n_spots"], dtype=float)
        alive = frames[(spots >= end_min_spots_per_frame) & (frames > start)]
        if alive.size:
            end = int(alive.max())
            end_reason = (f"density last reached "
                          f"{end_min_spots_per_frame:g}/frame here")
        else:
            end_reason = (f"density never reached {end_min_spots_per_frame:g}/frame "
                          f"after the start, so the tail was kept")

    # NO PROPOSED RANGE HOLDS A FRAME THE LASER DID NOT LIGHT (raised by the experimenter:
    # the red laser "sometimes turns off then on again"). Measured: 61 of 62 of one user's
    # mCherry movies go dark at least once, and four of one user's PY79 movies go dark in
    # their LAST 8-23 frames. A dark TAIL is cut off (the range ends at the last lit
    # frame, and says so); a dark stretch INSIDE the range refuses it by name. Built on
    # `lit_frames`' switch rule -- the level rule this first stood on read dim mVenus as
    # dark and flagged 40 ranges the lab had tracked.
    import numpy as np

    lit_at = curves.bleaching["frame"].to_numpy(dtype=int)
    lit = lit_frames(curves.bleaching["mean_counts"].to_numpy(dtype=float))
    if lit.any():
        last_lit = int(lit_at[lit].max())
        if int(end) > last_lit:
            end = last_lit
            end_reason = f"{end_reason}; the laser is off from frame {last_lit + 1}"
    dark = (lit_at >= int(start)) & (lit_at <= int(end)) & ~lit
    if dark.any():
        off = lit_at[dark]
        raise MoviePrepError(
            f"{curves.movie}: the laser is off over {off.size} frame(s) inside the proposed "
            f"range {int(start)}-{int(end)} (frames {int(off.min())}-{int(off.max())}); a "
            f"range must not hold them. Set the start after frame {int(off.max())} by hand, "
            f"or one start for every movie.")
    # A "RANGE" TOO SHORT FOR ONE TRACK IS NOT A PROPOSAL. Found on one user's 2000-frame
    # mVenus movies (12 Oct 2026): their curve never settles, the flat rule finds no
    # stretch, the fallback's field-mean floor is the LAST frame, and 22 movies were given
    # 2000-2000 -- saved like any proposal, and one frame forms no track. Refused by name,
    # as a rule finding no floor already is.
    if int(end) - int(start) + 1 < MIN_PROPOSED_FRAMES:
        raise MoviePrepError(
            f"{curves.movie}: the '{start_rule}' rule starts at frame {int(start)} and the "
            f"range ends at {int(end)}, {int(end) - int(start) + 1} frame(s), fewer than "
            f"a {MIN_PROPOSED_FRAMES}-localisation track needs; the curve does not settle "
            f"before the end. Set the range by hand, or one start for every movie.")
    return {
        "movie": curves.movie,
        "start": int(start), "end": int(end),
        "n_frames_kept": int(end - start + 1),
        "start_rule": "common start" if start_frame is not None else start_rule,
        "start_reason": start_reason,
        "end_reason": end_reason,
        "alternatives": {
            rule: (curves.floor.earliest_defensible if rule == "strictest"
                   else getattr(curves.floor, f"by_{rule}"))
            for rule in CROP_START_RULES},
    }


def crop_quality(curves, start: int, end: int, *,
                 tolerance_per_cent: float | None = None) -> dict:
    """Is the KEPT range flat, or still bleaching down? Measured, not eyeballed.

    The question the crop exists for. The whole-acquisition curve is dominated by the
    first few hundred frames, so a kept range that is still falling looks flat next to
    it: the legacy macro drew a SECOND curve over the crop alone for exactly this
    reason, and reading it is how an analyst decides the field has reached single
    molecules.

    The drop is compared against the curve's own scatter, so "still falling" is a
    measurement rather than a threshold somebody picked. A straight line is fitted over
    the kept range; the drop it predicts end-to-end is set against the residual scatter
    about that line. A drop smaller than the noise is not a trend.

    Returns the numbers and a short verdict; it does NOT decide the crop.
    """
    frames = np.asarray(curves.bleaching["frame"], dtype=float)
    values = np.asarray(curves.bleaching["mean_counts"], dtype=float)
    keep = (frames >= start) & (frames <= end) & np.isfinite(values)
    n = int(keep.sum())
    if n < 10:
        raise MoviePrepError(
            f"only {n} bleaching points inside {start}-{end}; too few to say whether "
            f"the kept range is flat")

    x, y = frames[keep], values[keep]
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    span = float(predicted[0] - predicted[-1])
    level = float(np.median(y))
    noise = float(np.std(y - predicted))

    drop_pct = 100.0 * span / level if level else float("nan")
    noise_pct = 100.0 * noise / level if level else float("nan")

    # NO BUILT-IN VERDICT, AND THE FIRST DRAFT'S WAS WRONG TWICE OVER. It called a range
    # "flat" when the end-to-end drop was under three times the PER-POINT scatter, on
    # protein I series006 that reported "flat" for a curve the eye plainly sees declining, and
    # the comparison was the wrong one anyway: a trend fitted through 3400 points is not
    # tested against single-point noise, it is tested against the standard error of its
    # own slope, which at that many points makes almost any real curve "significant".
    #
    # SIGNIFICANCE IS NOT THE QUESTION. The question is whether the drop is small enough
    # to live with, which is a judgement and belongs to the analyst, and the
    # experimenter had already named the control they wanted: "a setting like 'drop of
    # fluorescence is less than x'". So the drop is REPORTED, and a tolerance is applied
    # only when one is given.
    within = None
    if tolerance_per_cent is not None:
        within = bool(drop_pct <= float(tolerance_per_cent))

    spots_f = np.asarray(curves.density["frame"], dtype=float)
    spots_v = np.asarray(curves.density["n_spots"], dtype=float)
    inside = (spots_f >= start) & (spots_f <= end) & np.isfinite(spots_v)
    # THE MEAN, NOT THE MEDIAN. On a sparse tail most probe frames hold zero spots, so
    # the median reads 0.0 and looks like a broken measurement while the mean (0.4 on
    # series006) says what is actually there.
    spots = float(np.mean(spots_v[inside])) if inside.any() else float("nan")

    verdict = (f"drops {drop_pct:.1f} % across the kept range "
               f"(point scatter {noise_pct:.1f} %)")
    if within is not None:
        verdict += "  ✓" if within else f"  ✗ over {float(tolerance_per_cent):g} %"

    return {
        "start": int(start), "end": int(end), "n_points": n,
        "level": level, "drop_per_cent": drop_pct, "noise_per_cent": noise_pct,
        "within_tolerance": within, "mean_spots_per_frame": spots,
        "verdict": verdict,
    }


def discover_across_conditions(parent, *, report=None) -> dict:
    """``{condition name: [acquisition paths]}`` for every condition under ``parent``.

    Asked for so preparation is one session for a whole experiment rather than one window
    per condition, with the folder NAME telling them apart. The mesh editor already
    worked this way (`editor_project.scan_fields`) and segmentation gained it at the same
    time (`segment.segment_conditions`); this is the preparation third.

    Built on `organise.discover_project_folders`, so the rule for "what is a condition" is
    the same one the other two stages use rather than a third opinion. A parent that is
    itself a condition returns that one entry, so a caller never has to ask which case it
    is in.

    A condition holding no readable acquisition is LEFT OUT rather than returned empty:
    it has nothing to prepare, and an empty group would appear in a window as a condition
    whose movies failed to load.
    """
    from .organise import discover_project_folders
    from .rawmovie import discover_raw_movies

    groups: dict[str, list] = {}
    for state in discover_project_folders(parent):
        folder = state.folder
        # The acquisitions may be loose, or already filed into `raw/`. `acquisition_folder`
        # is what knows the difference, and it also says WHY, which matters here because
        # organising moves them and a scan that reported zero afterwards would read as
        # "my data disappeared".
        from .organise import acquisition_folder

        try:
            where, _why = acquisition_folder(folder)
            movies, _companions = discover_raw_movies(where, report=report)
        except Exception:                                     # noqa: BLE001 - skipped
            continue
        if movies:
            groups[folder.name] = list(movies)
    return groups


def prepare_conditions(groups, *, pixel_size_nm: float, frame_interval_s: float,
                       projection: str = "sd", on_progress=None) -> pd.DataFrame:
    """`prepare_batch` over several conditions, one table out.

    ``groups`` is ``[(condition_name, raw_dir, out_root, crops), ...]``. Each condition
    keeps ITS OWN crop mapping, which is the whole reason this is not one flat call:
    `series001` exists in every condition, so a single stem-keyed mapping would give one
    condition's movie another condition's crop range and there would be nothing on the
    result to show it had happened.

    The returned table carries a ``condition`` column so a failure can be read back to
    the folder it came from. Progress is continuous across the whole run rather than
    restarting per condition, because that is what a progress bar in front of a person
    is counting.

    A condition that raises is not allowed to end the run: `prepare_batch` already keeps
    a failed MOVIE as a row, and this keeps a failed CONDITION the same way, so four
    conditions always produce four conditions' worth of rows.
    """
    total = sum(len(crops) for _name, _raw, _out, crops in groups)
    done = 0
    tables = []
    for name, raw_dir, out_root, crops in groups:
        def relay(index, _sub_total, row, _done=done):
            if on_progress is not None:
                on_progress(_done + index, total, row)

        try:
            table = prepare_batch(raw_dir, out_root, crops,
                                  pixel_size_nm=pixel_size_nm,
                                  frame_interval_s=frame_interval_s,
                                  projection=projection, on_progress=relay)
        except Exception as exc:                          # noqa: BLE001 - kept as a row
            table = pd.DataFrame([{
                "movie": f"{name} (whole condition)", "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "n_frames_kept": 0, "crop_start": 0, "crop_end": 0,
            }])
        table = table.copy()
        table.insert(0, "condition", name)
        tables.append(table)
        done += len(crops)
    if not tables:
        raise MoviePrepError("no conditions to prepare")
    return pd.concat(tables, ignore_index=True)


@dataclass(frozen=True)
class ConditionProgress:
    """How far preparation has got in one condition, read from disk alone.

    Asked for so a parent of conditions shows an overview rather than only whatever is
    on screen: "does it read and propose just for the first folder, or all of them? we
    can show bars for all progresses".

    IT OPENS NO MOVIE. Every count comes from a directory listing or a small csv, so an
    overview of a whole session costs the same as listing it. That matters because
    reading a movie's bleaching curve is about 7.7 s, and an overview that had to read
    them would be the very wait it exists to explain.
    """

    name: str
    folder: Path
    n_movies: int
    n_curves: int
    n_crops: int
    #: The calibration this condition RECORDED, or None where it has none yet.
    #:
    #: Carried so a difference between conditions is visible without opening each one.
    #: Preparation and tracking are per movie, so conditions imaged differently are
    #: perfectly safe to process side by side; what is not safe is comparing their D
    #: values afterwards, and that is decided long before anyone reaches the analysis.
    pixel_size_nm: float | None = None
    frame_interval_s: float | None = None
    emission_nm: float | None = None

    def calibration(self) -> tuple:
        """The three numbers that must match before two conditions' D can be compared."""
        return (self.pixel_size_nm, self.frame_interval_s, self.emission_nm)

    @property
    def ready(self) -> bool:
        """Every movie has a crop range, which is what `prepare_batch` requires."""
        return bool(self.n_movies) and self.n_crops >= self.n_movies

    def caption(self) -> str:
        return (f"{self.name}: {self.n_curves}/{self.n_movies} read, "
                f"{self.n_crops}/{self.n_movies} cropped"
                + ("  ready" if self.ready else ""))


def condition_progress(parent, *, groups=None) -> list[ConditionProgress]:
    """Preparation progress for every condition under ``parent``, without reading movies.

    ``groups`` is what `discover_across_conditions` returns, passed in when the caller
    already has it so a window refreshing on every signal does not re-list the disk.

    Curves counted from ``bleaching/*.csv`` and crops from the recorded ``crops.json``,
    which are exactly the two things a re-opened folder rebuilds itself from, so this
    reports the same state the panel would find rather than a second opinion about it.
    """
    if groups is None:
        groups = discover_across_conditions(parent)

    out = []
    for name, movies in groups.items():
        folder = Path(movies[0]).parent
        # The condition root, not the acquisition folder: after organising the movies sit
        # in `<condition>/raw/` while the outputs stay one level up.
        from .organise import RAW_SUBFOLDER

        root = folder.parent if folder.name == RAW_SUBFOLDER else folder
        bleach = root / "bleaching"
        n_curves = (sum(1 for q in bleach.glob("*.csv") if q.is_file())
                    if bleach.is_dir() else 0)
        n_crops = 0
        px = tau = em = None
        try:
            found = find_crops(root)
            if found is not None:
                payload = load_crops(found)
                n_crops = len(payload.get("crops", {}))
                # The calibration is in the same file as the ranges, which is the whole
                # reason it was written there, so reading it costs nothing extra.
                px = payload.get("pixel_size_nm")
                tau = payload.get("frame_interval_s")
                em = payload.get("emission_nm")
        except Exception:                                 # noqa: BLE001 - advisory
            n_crops = 0
        out.append(ConditionProgress(name=name, folder=root, n_movies=len(movies),
                                     n_curves=min(n_curves, len(movies)),
                                     n_crops=min(n_crops, len(movies)),
                                     pixel_size_nm=px, frame_interval_s=tau,
                                     emission_nm=em))
    return out


def calibration_mismatch(rows) -> dict:
    """Which recorded calibrations differ across conditions, by field.

    ``rows`` is what `condition_progress` returns. Returns ``{field: {name: value}}`` for
    every field where the conditions that HAVE recorded one do not agree.

    THIS IS NOT A REASON TO STOP. Preparation and tracking are per movie: two conditions
    imaged differently are tracked with their own numbers and nothing pools them. What it
    decides is what may be done AFTERWARDS, and that decision is made long before anyone
    opens the analysis, which is why it is worth seeing here rather than there. A pixel
    size difference means the D values cannot be compared at all; a frame interval
    difference means they are strata; a fluorophore difference changes the PSF and so the
    spot radius each condition needs.

    A condition with nothing recorded is left OUT rather than counted as disagreeing:
    "not set yet" and "set to something else" are different, and only one is a mismatch.
    """
    fields = {"pixel_size_nm": {}, "frame_interval_s": {}, "emission_nm": {}}
    for row in rows:
        for name in fields:
            value = getattr(row, name, None)
            if value is not None:
                fields[name][row.name] = float(value)
    return {name: got for name, got in fields.items()
            if len({round(v, 6) for v in got.values()}) > 1}


#: The calibration a condition needs before it can be tracked, in the order a person
#: fills them in. Each carries the label the window shows and the unit it is in.
#: THE FLUOROPHORE COMES BEFORE THE EMISSION because it DETERMINES it: picking one
#: fills the wavelength, and the reverse is not true. Listing emission first asked for a
#: number that the next row was about to supply.
CALIBRATION_FIELDS = (
    ("pixel_size_nm", "pixel size", "nm"),
    ("frame_interval_s", "frame interval", "ms"),
    ("label", "fluorophore", ""),
    ("emission_nm", "emission", "nm"),
    ("numerical_aperture", "NA", ""),
)


def detect_calibration(movie_path) -> dict:
    """What the ACQUISITION itself can tell us, and nothing it cannot.

    The frame interval is MEASURED by the instrument per frame, and it is the one people
    most often get wrong by hand: this project has twice corrected a nominal 20 against a
    measured 20.4717, and a folder recording 20 ms whose acquisition measures 51.7477.
    The exposure setting is NOT the frame interval, because readout adds to it.

    THE PIXEL SIZE IS DELIBERATELY NOT READ, although the file carries one. Surveyed
    across 29 groups, 24 claim 160.0 nm and five claim 106.0, and the split runs BETWEEN
    CONDITIONS OF ONE PROTEIN. So a tool that read it would be right on a spot check and
    wrong on most of the data, which is worse than being wrong everywhere because a
    uniform error gets noticed. It is derived by NIS from a magnification that ignores
    the tube lens; 160 against the true 106 would rescale every D by 2.28x.
    """
    from .rawmovie import read_raw_movie

    meta = read_raw_movie(movie_path).meta
    out = {}
    if getattr(meta, "frame_interval_ms", None):
        out["frame_interval_s"] = float(meta.frame_interval_ms) / 1000.0
    if getattr(meta, "numerical_aperture", None):
        out["numerical_aperture"] = float(meta.numerical_aperture)
    return out


def read_calibrations(parent, *, groups=None, probe: bool = True) -> dict:
    """``{condition: {field: value}}`` for every condition under ``parent``.

    Seeded from each condition's own recorded `crops.json`, so a set already prepared
    opens with its numbers in place and one never prepared opens empty. A field nobody
    recorded is ``None`` rather than a default: this project does not invent a pixel size
    or a frame interval, and an empty cell is the honest way to ask for one.
    """
    if groups is None:
        groups = discover_across_conditions(parent)
    out = {}
    for name, movies in groups.items():
        folder = Path(movies[0]).parent
        from .organise import RAW_SUBFOLDER

        root = folder.parent if folder.name == RAW_SUBFOLDER else folder
        values = {key: None for key, _label, _unit in CALIBRATION_FIELDS}
        try:
            found = find_crops(root)
            if found is not None:
                payload = load_crops(found)
                for key, _label, _unit in CALIBRATION_FIELDS:
                    if payload.get(key) is not None:
                        values[key] = payload[key]
        except Exception:                                 # noqa: BLE001 - advisory
            pass
        # WHAT THE FILE DID NOT RECORD, ASKED OF THE ACQUISITION. A folder that has never
        # been prepared has no crops.json at all, so seeding only from it left every cell
        # empty on exactly the sessions this table is for. One metadata read per
        # condition, and it never overwrites a recorded value.
        if probe and movies and any(values[k] is None for k in
                                    ("frame_interval_s", "numerical_aperture")):
            try:
                for key, value in detect_calibration(movies[0]).items():
                    if values.get(key) is None:
                        values[key] = value
            except Exception:                             # noqa: BLE001 - advisory
                pass
        out[name] = values
    return out


def replicate_calibration(calibrations: dict, source: str) -> dict:
    """Copy one condition's calibration onto every other, and return the new mapping.

    Asked for directly: "write for all conditions what u set in 1, replicate it in 2, 3".
    Most sessions ARE imaged alike, so filling one column and copying is the ordinary
    case, and the differences that remain are the ones worth seeing.

    ONLY THE FIELDS THE SOURCE ACTUALLY HAS. Copying a `None` over a value somebody
    already entered would make the button destructive in a way its name does not say, so
    a source that has no fluorophore recorded leaves everyone else's alone.

    Returns a NEW mapping; the caller decides whether to keep it. Nothing is written to
    disk here, because the calibration is saved with the ranges and that is one press
    the user makes deliberately.
    """
    if source not in calibrations:
        raise MoviePrepError(
            f"{source!r} is not one of the conditions: {sorted(calibrations)}")
    template = {k: v for k, v in calibrations[source].items() if v is not None}
    if not template:
        raise MoviePrepError(
            f"{source} has no calibration to copy: fill at least one field first")
    out = {}
    for name, values in calibrations.items():
        merged = dict(values)
        merged.update(template)
        out[name] = merged
    return out


def missing_calibration(calibrations: dict, *, required=("pixel_size_nm",
                                                         "frame_interval_s")) -> dict:
    """``{condition: [fields]}`` for conditions that cannot be tracked yet.

    Pixel size and frame interval are required because every D scales with them. The
    emission, NA and label shape the DETECTION and are strongly advised, but a run
    without them is wrong in a way that can be corrected afterwards rather than a run
    whose coordinates mean nothing.
    """
    out = {}
    for name, values in calibrations.items():
        gaps = [f for f in required if values.get(f) is None]
        if gaps:
            out[name] = gaps
    return out
