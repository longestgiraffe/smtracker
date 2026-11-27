"""Canonical track table: units and the track-length filter, in ONE place.

Everything downstream (SQD, appD, MSD, saSPT, plots) consumes the canonical
table produced here and *never* converts units or filters by length again, the
project's "one source of truth" rule (``CLAUDE.md``).

Canonical columns
-----------------
``protein, condition, day, movie, cell, uid, track_id, frame, t_s, x_um, y_um, amp``

- ``uid``   : globally unique track key, ``"<movie>:<track_id>"`` (``track_id``
  is only unique within a movie).
- ``x_um, y_um`` : positions in **µm** = pixel coordinate × ``pixel_size_nm``/1000.
- ``t_s``   : time in **seconds** = ``frame`` × ``frame_interval_ms``/1000. Frame
  numbering is kept as-is (u-track frames are 1-based), so ``t_s`` is an absolute
  movie time, not a per-track time. Displacements use *differences*, so the
  origin does not matter.
- ``frame`` is retained: gaps are real (u-track can bridge them) and lag
  computations must use frame differences, never row differences.

Invariants
----------
- **Acquisition parameters are required, never guessed.** Every D scales with
  ``τ`` and with ``pixel_size²``; a default here would silently produce wrong
  physics. Both must be non-positive-free, finite, and supplied by the caller
  from acquisition metadata.
- **Length filter, once:** tracks with fewer than ``min_localizations``
  localizations are dropped (default 5: a track of 4 steps is KEPT, one of 3
  steps or fewer is removed). Counted as *rows* (localizations), which is
  unambiguous under gaps. The legacy analysis has no length filter of its own
  (`calcCDF.m` takes every track, `getDispl_gui.m:24` drops one-point tracks
  only): its 5 is u-track's ``minTrackLen = 5``, so on u-track data this filter
  removes nothing. What it does remove is recorded per condition
  (``attrs["length_floor_removed"]``) for QC, since those tracks have no rows.
- **Fail loud:** missing columns, non-finite coordinates, duplicate
  ``(uid, frame)`` rows, or an empty result after filtering all raise.
- Grouping must already be assigned (``smtracker.grouping.assign_groups``);
  ``cell`` may still be ``<NA>`` while segmentation is not wired up.

Measurement limits (report these with every result, see ``CLAUDE.md``)
---------------------------------------------------------------------
``localization_floor_D`` ≈ σ²/(4τ), the smallest apparent D distinguishable
from immobile; ``detection_ceiling_D`` ≈ r_search²/(4τ): above it, fast
molecules are truncated or lost by the tracker. Both are **bounds**, not
physical D values. ``AcquisitionParams`` computes them so no analysis has to
re-derive them.

**The search radius is carried PER CONDITION where the conditions were linked
at different radii** (tester D10, ruled 4 Nov 2026: free mNG linked at 5 px and
at 8 px). ``search_radius_px_by_condition`` holds each condition's radius (None
where its tracker recorded none) and the scalar ``search_radius_px`` is then
None: there is no one radius. Every reader asks for a condition's own
(:meth:`AcquisitionParams.search_radius_px_for`,
:meth:`~AcquisitionParams.detection_ceiling_D_for`, or :meth:`~AcquisitionParams.for_conditions`
on a table, `table_params`), and the scalar :attr:`~AcquisitionParams.detection_ceiling_D`
REFUSES on such a project, as :attr:`~AcquisitionParams.tau_s` refuses on a
mixed-tau one. A project linked at one radius keeps no map and reads exactly as
before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pandas as pd

__all__ = [
    "without_attrs",
    "CanonicalError",
    "AcquisitionParams",
    "table_params",
    "distinct_radii",
    "radius_words",
    "CANONICAL_COLUMNS",
    "CANONICAL_EXTRA_COLUMNS",
    "LOC_SIGMA_COLUMN",
    "to_canonical",
    "track_table",
    "cap_track_length",
    "track_cap_cost",
    "longest_removed_steps",
]

#: Older provenance names → the three states. Kept so a stored analysis loads with
#: its provenance intact instead of being silently re-defaulted:
#:   derived_pixelation → theoretical (it WAS the theoretical σ, on the pixelation
#:                        basis; the value changes only where a PSF width exists)
#:   manual, msd_intercept → configured (a supplied constant / an estimate from the
#:                        data: neither is a stationary-reference measurement)
#:   immobile_control → measured
_SIGMA_SOURCE_ALIASES = {
    "derived_pixelation": "theoretical",
    "manual": "configured",
    "msd_intercept": "configured",
    "immobile_control": "measured",
}

CANONICAL_COLUMNS = [
    "protein", "condition", "day", "movie", "cell", "uid", "track_id",
    "frame", "t_s", "x_um", "y_um", "amp",
]
#: EACH SPOT'S OWN LOCALISATION PRECISION (6 Nov 2026, the per-spot precision fit): u-track's per-localisation
#: uncertainty (``tracksCoordAmpCG``'s ``dx, dy``, pixels) as ONE per-axis sigma in um, sqrt((dx^2 + dy^2) / 2) x
#: the pixel size; NaN where u-track wrote none. ONLY where the raw table has ``dx, dy`` (u-track): a TrackMate
#: or track-table project has no such column, never a guessed one. It is u-track's formal fit covariance, a LOWER
#: bound on the true error (measured on SynR: 1.3-2.2 times small, docs/claude/sigma_and_floor.md), so a fit reads
#: it through a calibration, never raw. Not part of any cache's row digest (`sectioncache._rows_digest`): a result
#: that reads it keys on it itself.
LOC_SIGMA_COLUMN = "loc_sigma_um"
#: Written by :func:`to_canonical` alongside the required columns: this row's own
#: frame interval. Always present, so a consumer never has to ask whether the
#: project happens to be uniform: on a mixed project it is the movie's own tau, and
#: `params.tau_s` would have raised. Then each spot's precision, where the tracker recorded one.
CANONICAL_EXTRA_COLUMNS = ["tau_s", LOC_SIGMA_COLUMN]

#: Keep tracks with at least this many localizations (4 steps): u-track's ``minTrackLen``, which the legacy analysis
#: relied on.
DEFAULT_MIN_LOCALIZATIONS = 5


def longest_removed_steps(min_localizations: int = DEFAULT_MIN_LOCALIZATIONS) -> int:
    """The longest track the length floor removes, in steps: at 5 localisations a 4-step track is kept and a 3-step
    one removed (reported 28 Oct 2026: QC drew the cut at 4 and called every kept 4-step track removed)."""
    return int(min_localizations) - 2

#: Two pixel sizes (or search radii) are ONE setting when they agree to this RELATIVE tolerance (reported 26 Oct 2026:
#: u-track stored one user's 106 nm as 106.00000296800008 in some projects and 106.0 in others, and a folder holding both
#: was refused as "two pixel sizes"). 1e-6 of 106 nm is 0.0001 nm: rounding passes, a real difference (106 against 160,
#: or 106 against 106.1) is still refused.
SAME_SETTING_REL_TOL = 1e-6


#: Pixel sizes of two conditions this close (relative) are LOADED, with a warning, rather than refused (ruled 26 Oct
#: 2026: "i think 1 % pixel size difference is [allowed] with a warning"): every coordinate is converted at ONE pixel
#: size, so another condition's D reads off by about twice the difference (D scales with the pixel size squared), its
#: fractions unchanged. Further apart they are refused as two projects.
PIXEL_SIZE_WARN_REL = 0.01


def same_setting(a, b) -> bool:
    """Whether two recorded acquisition values are one setting (`SAME_SETTING_REL_TOL`, relative)."""
    return math.isclose(float(a), float(b), rel_tol=SAME_SETTING_REL_TOL, abs_tol=0.0)


class CanonicalError(RuntimeError):
    """Raised when a raw table cannot be turned into a valid canonical table."""


@dataclass(frozen=True)
class AcquisitionParams:
    """Acquisition metadata. No defaults: these must come from the microscope.

    Parameters
    ----------
    pixel_size_nm
        Camera pixel size in the sample plane, nm.
    frame_interval_ms
        Frame interval τ, ms. Every diffusion coefficient scales with 1/τ.
    localization_precision_nm
        1σ localization precision, nm. Optional: when None the active σ is the
        THEORETICAL one (see ``localization_sigma_nm``). Supply a value only for a
        lab-configured constant or a value from a stationary reference, and set
        ``localization_sigma_source`` accordingly.
    localization_sigma_source
        Where the active σ comes from: one of :data:`SIGMA_SOURCES`. The value and
        this state travel together as one pair: **a σ without a provenance state is
        not a σ**, and every saved analysis stamps both.
    psf_sigma_px
        The optical PSF width the tracker recorded (u-track ``psfSigma``, pixels).
        READ from ``movieData.mat``, never recomputed here, reimplementing
        u-track's fitted high-NA approximation from memory would be a guess, and
        the file already carries its output. Sets the theoretical σ when present.
    search_radius_px
        Tracker linking search radius, pixels (u-track). Sets the detection
        ceiling. Optional, as above. None on a project whose conditions were
        linked at different radii: see ``search_radius_px_by_condition``.
    search_radius_px_by_condition
        ``{condition: radius_px or None}`` when the conditions were linked at
        different radii (the loader's `_check_params_agree`), else None.

    σ configuration
    ---------------
    Four provenance states, no more:

    - ``"theoretical"``: computed from the optical parameters in the file. Usable
      with no action from the user, and the fallback when nothing better exists.
    - ``"tracker_reported"``: the median of the tracker's own per-localisation
      uncertainty (u-track's ``dx, dy``). See below.
    - ``"configured"``: a lab-supplied constant. Instrument/lab-level config, with
      a per-dataset override; never a per-run dialog.
    - ``"measured"``: from a stationary reference sample. Entered by hand, with
      this state, when such a measurement exists. Nothing is auto-promoted into it:
      in particular u-track's per-localization ``dx, dy`` are a lower bound on σ
      (see ``spatial/SCHEMA.md``) and never become a measured σ.

    The fourth state was added, rather than folding ``dx, dy`` into one of the three,
    because it is a different quantity from all of them and the entry above already
    says so. It is not ``"measured"`` (that rule stands, and is the reason this state
    exists); it is not ``"configured"`` (nobody supplied it); and it is not
    ``"theoretical"`` (it comes from the data, not from the optics).

    WHY IT IS PREFERRED OVER ``"theoretical"`` WHEN AVAILABLE. The theoretical σ on
    the PSF basis is ``psf_sigma_px × pixel_size_nm``, which is the WIDTH OF THE SPOT
    and not the precision of its centre: fitting a spot to N photons locates it far
    better than its own width. Measured over 26 groups and 219 582 localisations on
    this project, u-track reports a median **17.3 nm** against that basis's **98.8**,
    and Thompson 2002 at the recall benchmark's own photon budget (127 photons per
    molecule, 4.81 background per pixel) independently gives **17.2 nm** with the
    EMCCD excess-noise factor. Two routes sharing no arithmetic.

    IT IS STILL A LOWER BOUND, which is why it is not ``"measured"``: a fit's formal
    covariance knows photon noise and background, and knows nothing about stage
    drift, PSF model mismatch or a molecule out of the focal plane. Honest range on
    this data is ~17 nm floor, plausibly 20-35 nm in truth. The ensemble MSD
    intercept is NOT the alternative: simulated with σ = 0 exactly, a 1 x 3 µm cell
    alone produces an apparent 35.6 nm at 2 lags rising to 119.5 nm at 20, which
    reproduces this project's recorded window-dependence from confinement and
    nothing else.

    Older files used ``derived_pixelation`` / ``manual`` / ``msd_intercept`` /
    ``immobile_control``; those are accepted and migrated (see
    ``_SIGMA_SOURCE_ALIASES``) so a stored analysis still loads with its provenance
    intact rather than being silently re-defaulted.
    """

    #: valid provenance sources for the localization σ: value and state are a pair
    SIGMA_SOURCES = ("theoretical", "tracker_reported", "configured", "measured")

    pixel_size_nm: float
    frame_interval_ms: float
    localization_precision_nm: float | None = None
    localization_sigma_source: str = "theoretical"
    psf_sigma_px: float | None = None
    search_radius_px: float | None = None
    #: ``{movie: frame_interval_ms}`` when a project mixes acquisition rates.
    #:
    #: WHY THIS EXISTS AND WHY ``tau_s`` THEN RAISES. Every D scales with 1/tau, so a
    #: project imaged at 20.3 ms and 30 ms cannot be pooled under one tau: the wrong
    #: group comes out 1.478x off, which reclassifies tracks across any threshold.
    #: Rather than let a scalar tau be read by code that cannot handle the mixture,
    #: :attr:`tau_s` REFUSES on a mixed project and callers must use
    #: :meth:`tau_s_for` or the canonical ``tau_s`` column. That turns every unsafe
    #: path (the pooled-model fits whose likelihood has one tau in it) into a loud
    #: failure instead of a silently wrong number.
    frame_interval_ms_by_movie: dict | None = None
    #: ``{condition: search_radius_px or None}`` when a project's conditions were LINKED at different radii (tester
    #: D10, ruled 4 Nov 2026: free mNG linked at 5 px beside the same molecule linked at 8 px), else None.
    #:
    #: WHY PER CONDITION AND NOT THE FIRST ONE'S. The merge kept the FIRST condition's radius for every condition, so
    #: the 8 px conditions' apparent-D ceiling line, QC's crowded-frame flags and neighbour share, and the SDA map's
    #: limits were all drawn at 5 px. The fitted D and fractions never read the radius. With a map the scalar
    #: ``search_radius_px`` is None (there is no one radius), every reader asks for its condition's own
    #: (:meth:`search_radius_px_for`, :meth:`detection_ceiling_D_for`, :meth:`for_conditions`), and the scalar
    #: :attr:`detection_ceiling_D` refuses, as :attr:`tau_s` does on a mixed-tau project. None where the
    #: conditions agree, so a project linked at one radius reads exactly as before. A condition whose tracker
    #: recorded NO radius (a track table, TrackMate) beside one that did is a value of its own, None: it never
    #: borrows another condition's radius (before, it took the first condition's), and its ceiling is unknown.
    search_radius_px_by_condition: dict | None = None

    def __post_init__(self) -> None:
        for name in ("pixel_size_nm", "frame_interval_ms"):
            _require_positive(getattr(self, name), name)
        for name in ("localization_precision_nm", "search_radius_px", "psf_sigma_px"):
            val = getattr(self, name)
            if val is not None:
                _require_positive(val, name)
        if self.search_radius_px_by_condition is not None:
            radii = {}
            for condition, val in dict(self.search_radius_px_by_condition).items():
                if val is not None:
                    _require_positive(val, f"search_radius_px of {condition!r}")
                radii[str(condition)] = None if val is None else float(val)
            # A MAP ONLY WHERE THE RADII DIFFER: conditions linked alike are one radius, the scalar
            distinct = distinct_radii(radii.values())
            if len(distinct) <= 1:
                if distinct and self.search_radius_px is None:
                    object.__setattr__(self, "search_radius_px", distinct[0])
                radii = None
            else:
                object.__setattr__(self, "search_radius_px", None)
            object.__setattr__(self, "search_radius_px_by_condition", radii)
        source = _SIGMA_SOURCE_ALIASES.get(self.localization_sigma_source,
                                           self.localization_sigma_source)
        object.__setattr__(self, "localization_sigma_source", source)
        if source not in self.SIGMA_SOURCES:
            raise ValueError(
                f"localization_sigma_source must be one of {self.SIGMA_SOURCES}, "
                f"got {self.localization_sigma_source!r}")
        # A provided precision left at the default state is a CONFIGURED value; a
        # non-theoretical state REQUIRES a value; theoretical means no supplied one.
        if (self.localization_precision_nm is not None
                and source == "theoretical"):
            object.__setattr__(self, "localization_sigma_source", "configured")
        if (self.localization_precision_nm is None
                and source != "theoretical"):
            raise ValueError(
                f"localization_sigma_source={source!r} needs an explicit "
                "localization_precision_nm")

    @property
    def um_per_pixel(self) -> float:
        return self.pixel_size_nm / 1000.0

    @property
    def has_mixed_tau(self) -> bool:
        """True when this project was imaged at more than one frame interval."""
        by = self.frame_interval_ms_by_movie
        return bool(by) and len({round(float(v), 9) for v in by.values()}) > 1

    @property
    def tau_s(self) -> float:
        """Frame interval τ in seconds. RAISES on a mixed-τ project.

        Deliberately fatal rather than convenient: a scalar τ read on a project that
        mixes acquisition rates is the silent-wrong-number failure this whole module
        exists to prevent. Code that can handle the mixture uses :meth:`tau_s_for`
        or the canonical ``tau_s`` column; code that cannot (a pooled SQD or HMM
        likelihood has exactly one τ in it) stops here with the reason.
        """
        if self.has_mixed_tau:
            rates = sorted({round(float(v), 6) for v
                            in self.frame_interval_ms_by_movie.values()})
            from .chart_text import fold_words

            raise CanonicalError(
                f"this project mixes frame intervals ({rates} ms) so there is no "
                "single τ. Every D scales with 1/τ, so using one would be wrong by "
                f"up to {fold_words(max(rates) / min(rates), 3)}. Use tau_s_for(movie) or the "
                "canonical 'tau_s' column, or analyse each rate as its own group.")
        return self.frame_interval_ms / 1000.0

    def tau_s_for(self, movie: str) -> float:
        """This movie's τ in seconds: the per-movie route, always safe."""
        by = self.frame_interval_ms_by_movie or {}
        if movie in by:
            return float(by[movie]) / 1000.0
        if self.has_mixed_tau:
            raise CanonicalError(
                f"no frame interval recorded for movie {movie!r} in a mixed-τ "
                "project; refusing to fall back on another movie's rate")
        return self.frame_interval_ms / 1000.0

    def tau_strata(self) -> dict:
        """``{tau_s: [movies]}``: the strata a mixed project must be reported in.

        The strata are not merely an arithmetic inconvenience: at 20.3 vs 30 ms the
        static-error offset σ²/τ differs by ~0.014 µm²/s (≈10 % of a 0.15 threshold),
        the detection ceiling r²/(4τ) differs by ~48 %, and motion blur differs. So
        even with τ applied correctly per movie these are two slightly different
        measurements and belong in the report as strata.
        """
        by = self.frame_interval_ms_by_movie or {}
        if not by:
            return {self.frame_interval_ms / 1000.0: []}
        out: dict[float, list] = {}
        for movie, ms in sorted(by.items()):
            out.setdefault(round(float(ms) / 1000.0, 9), []).append(movie)
        return out

    @property
    def theoretical_basis(self) -> str:
        """Which optical quantity the theoretical σ rests on: ``"psf"`` or
        ``"pixelation"``. Recorded because the two differ by ~3× on this setup."""
        return "psf" if self.psf_sigma_px is not None else "pixelation"

    @property
    def theoretical_sigma_nm(self) -> float:
        """σ from the optics alone, nm: the default, needing no user action.

        ``psf_sigma_px × pixel_size_nm`` when the tracker recorded a PSF width (on
        the reference setup: 0.931681 px × 106.000003 nm = **98.8 nm**), else the
        pixelation bound ``pixel_size_nm / √12`` = **30.6 nm**, the standard
        deviation of a uniform distribution over one pixel.

        The two are different quantities and the gap is not reconciled here: the
        PSF width is how wide a single emitter's image is, the pixelation bound is
        how finely the camera samples it, and the achievable localization precision
        is smaller than the PSF width by roughly √N photons. Both numbers, and this
        caveat, are recorded in ``spatial/SCHEMA.md``.
        """
        if self.psf_sigma_px is not None:
            return float(self.psf_sigma_px) * self.pixel_size_nm
        return self.pixel_size_nm / math.sqrt(12.0)

    @property
    def localization_sigma_nm(self) -> float:
        """The ACTIVE 1σ localization precision, nm: always available.

        The configured/measured value if one was supplied, else the THEORETICAL one
        (above). ``localization_sigma_source`` records which of the three states it
        is; :meth:`sigma_stamp` returns the value and the state together, because a
        σ without its provenance is not a σ.
        """
        if self.localization_precision_nm is not None:
            return self.localization_precision_nm
        return self.theoretical_sigma_nm

    def sigma_stamp(self) -> dict:
        """The σ field pair (plus its basis) for provenance. One source of truth.

        Every saved analysis carries this; anything that loads one and finds the
        value or the provenance missing must FAIL rather than assume
        ``theoretical``: assuming a default is exactly what the provenance state
        exists to prevent.
        """
        return {
            "sigma_nm": float(self.localization_sigma_nm),
            "sigma_provenance": self.localization_sigma_source,
            "sigma_basis": (self.theoretical_basis
                            if self.localization_sigma_source == "theoretical"
                            else "tracker_dx_dy"
                            if self.localization_sigma_source == "tracker_reported"
                            else "supplied"),
        }

    @property
    def D_bound_legacy(self) -> float:
        """σ²/(4τ) µm²/s, the BOUND, in the legacy convention.

        The name matters (AZ ruling). There is no unique "localisation floor": two
        σ→D conversions are registered in this project and they are **different
        quantities, neither substituting for the other**:

        * ``D_bound_legacy`` = σ²/(4τ), the smallest apparent D the legacy
          convention calls distinguishable from immobile;
        * ``D_static_error_offset`` = σ²/τ, the additive BIAS a σ-correction
          removes, from the 2-D two-endpoint relation ⟨r²⟩ = 4Dτ + 4σ².

        They differ by 4x. (The often-quoted 6x gap to the PSF basis is 4 x 1.5:
        4 from that noise variance, 1.5 from Berglund blur at R = 1/6.)

        Both derive from the SAME σ and τ, so the single source of truth is the
        *inputs*, not a second stored constant.

        Which one a comparison wants is decidable, not a matter of taste: it
        depends on whether the D being compared is apparent or σ-corrected. Read it
        out of the estimator that produced the number. For the HMM here,
        ``hmm.py`` adds ``offset = 2 * _NDIM * sigma**2`` (= 4σ²) to the squared
        displacement, and ``scripts/msd_hmm.py`` sets ``SIGMA_UM_EMISSION = 0.0``,
        so its fitted D are APPARENT and carry +σ²/τ: the matching comparator is
        ``D_static_error_offset``.
        """
        sigma_um = self.localization_sigma_nm / 1000.0
        return sigma_um**2 / (4.0 * self.tau_s)

    @property
    def D_static_error_offset(self) -> float:
        """σ²/τ µm²/s, the additive bias a σ-correction removes. See ``D_bound_legacy``.

        From ⟨r²⟩ = 4Dτ + 4σ² (2-D, two endpoints): fitting with no error offset
        gives D_apparent = D_true + σ²/τ. So this is what an APPARENT D is inflated
        by, and the quantity an apparent D should be compared against.
        """
        sigma_um = self.localization_sigma_nm / 1000.0
        return sigma_um**2 / self.tau_s

    @property
    def localization_floor_D(self) -> float:
        """Deprecated alias for :attr:`D_bound_legacy`. Kept, not overwritten.

        The name implies a unique floor and there is not one, see
        ``D_bound_legacy``. Existing callers keep working and keep their meaning
        (this has always returned σ²/(4τ)); new code should name the quantity it
        actually means.
        """
        return self.D_bound_legacy

    @property
    def has_mixed_search_radius(self) -> bool:
        """True when this project's conditions carry their OWN search radii (``search_radius_px_by_condition``): they
        were linked at different radii, or some recorded none beside others that did (a track table, TrackMate)."""
        return bool(self.search_radius_px_by_condition)

    def search_radius_px_for(self, condition) -> float | None:
        """``condition``'s own linking search radius in pixels (None where its tracker recorded none): the per-condition
        route, always safe. RAISES on a project linked at several radii for a condition it holds no radius for, rather
        than lend it another condition's."""
        by = self.search_radius_px_by_condition or {}
        if str(condition) in by:
            return by[str(condition)]
        if self.has_mixed_search_radius:
            raise CanonicalError(
                f"no search radius recorded for condition {condition!r} in a project linked at several radii "
                f"({', '.join(f'{c} {radius_words(r)}' for c, r in by.items())}); refusing to fall back on another "
                "condition's")
        return self.search_radius_px

    def for_conditions(self, conditions) -> "AcquisitionParams":
        """These parameters as the rows of ``conditions`` read them: on a project linked at several search radii, the
        ONE radius those conditions share, as the scalar (the map dropped); ``self`` where there is no map. RAISES
        `CanonicalError` when the conditions were linked at different radii: a ceiling or a crowding disc over them
        has no one radius to be drawn at."""
        if not self.has_mixed_search_radius:
            return self
        radii = {str(c): self.search_radius_px_for(c) for c in dict.fromkeys(str(c) for c in conditions)}
        distinct = distinct_radii(radii.values())
        if len(distinct) > 1:
            raise CanonicalError(
                "these conditions were linked at different search radii ("
                + ", ".join(f"{c} {radius_words(r)}" for c, r in radii.items())
                + "), so no one radius or detection ceiling applies to them together; ask for each condition's own "
                "(search_radius_px_for, detection_ceiling_D_for)")
        return replace(self, search_radius_px=distinct[0] if distinct else None,
                       search_radius_px_by_condition=None)

    @property
    def detection_ceiling_D(self) -> float | None:
        """Apparent-D ceiling r_search²/(4τ) in µm²/s (``None`` if radius unknown). RAISES on a project linked at
        several search radii: there is no one ceiling, ask for a condition's (:meth:`detection_ceiling_D_for`)."""
        if self.has_mixed_search_radius:
            raise CanonicalError(
                "this project's conditions were linked at different search radii ("
                + ", ".join(f"{c} {radius_words(r)}" for c, r in self.search_radius_px_by_condition.items())
                + "), so there is no single detection ceiling. Use detection_ceiling_D_for(condition).")
        if self.search_radius_px is None:
            return None
        r_um = self.search_radius_px * self.um_per_pixel
        return r_um**2 / (4.0 * self.tau_s)

    def detection_ceiling_D_for(self, condition) -> float | None:
        """``condition``'s apparent-D ceiling r²/(4τ) in µm²/s, at its own search radius (None if it recorded none)."""
        return self.for_conditions([condition]).detection_ceiling_D

    def limits_report(self) -> str:
        """One-line human-readable statement of what this setup can resolve."""
        floor = self.localization_floor_D
        parts = [f"τ = {self.frame_interval_ms:g} ms", f"pixel = {self.pixel_size_nm:g} nm"]
        parts.append(
            f"σ = {self.localization_sigma_nm:.3g} nm "
            f"({self.localization_sigma_source}/{self.sigma_stamp()['sigma_basis']}), "
            f"localization floor D ≈ {floor:.4g} µm²/s"
        )
        if self.has_mixed_search_radius:
            # EACH CONDITION'S OWN CEILING, never one of them for all (tester D10)
            said = []
            for condition in self.search_radius_px_by_condition:
                ceil = self.detection_ceiling_D_for(condition)
                said.append(f"{condition} ≈ {ceil:.3g} µm²/s" if ceil is not None else f"{condition} unknown")
            parts.append("detection ceiling D per condition (linked at different search radii): " + ", ".join(said))
        else:
            ceil = self.detection_ceiling_D
            parts.append(
                f"detection ceiling D ≈ {ceil:.3g} µm²/s" if ceil is not None
                else "detection ceiling: unknown (search_radius_px not set)"
            )
        return "; ".join(parts) + ". Floor/ceiling are bounds, not physical D values."


def radius_words(radius) -> str:
    """A search radius in words: ``"5 px"``, or ``"none recorded"``."""
    return "none recorded" if radius is None else f"{float(radius):g} px"


def distinct_radii(values) -> list:
    """The distinct radii among ``values`` (None counted as one value of its own), one setting each
    (`same_setting`)."""
    out: list = []
    for v in values:
        if not any((v is None and o is None) or (v is not None and o is not None and same_setting(v, o))
                   for o in out):
            out.append(v)
    return out


def table_params(table) -> "AcquisitionParams":
    """The acquisition parameters ``table``'s rows read at: its ``attrs["params"]`` narrowed to the conditions in its
    ``condition`` column (`AcquisitionParams.for_conditions`), so a table of one condition has that condition's own
    search radius and detection ceiling. RAISES `CanonicalError` for a table without parameters, and for one whose
    rows come from conditions linked at different radii."""
    params = getattr(table, "attrs", {}).get("params")
    if not isinstance(params, AcquisitionParams):
        raise CanonicalError("the table carries no AcquisitionParams in .attrs['params']")
    if not params.has_mixed_search_radius:
        return params
    if "condition" not in getattr(table, "columns", ()):
        raise CanonicalError("the table has no condition column, and its project was linked at several search radii")
    return params.for_conditions(pd.unique(table["condition"].astype(str)))


def _require_positive(value, name: str) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(v) or v <= 0:
        raise CanonicalError(f"{name} must be finite and > 0, got {value!r}")


def to_canonical(
    df: pd.DataFrame,
    params: AcquisitionParams,
    *,
    min_localizations: int = DEFAULT_MIN_LOCALIZATIONS,
) -> pd.DataFrame:
    """Convert a grouped raw table to the canonical table (µm, seconds, filtered).

    Parameters
    ----------
    df
        Grouped raw table: ``movie, track_id, frame, x, y, amp`` plus
        ``condition, day`` assigned (``cell`` may be ``<NA>``).
    params
        Acquisition metadata (pixel size, τ).
    min_localizations
        Keep tracks with at least this many localizations (default 5 → ≥4 steps).
        ``1`` disables the filter.

    Returns
    -------
    pandas.DataFrame
        ``CANONICAL_COLUMNS``, sorted by ``(uid, frame)``.

    Raises
    ------
    CanonicalError
        On missing columns, unassigned grouping, non-finite coordinates,
        duplicate ``(uid, frame)`` rows, or an empty result.
    """
    if not isinstance(params, AcquisitionParams):
        raise CanonicalError(
            "params must be an AcquisitionParams (pixel size and τ are required "
            "and must not be guessed)."
        )
    if min_localizations < 1:
        raise CanonicalError(f"min_localizations must be >= 1, got {min_localizations}")

    required = ["movie", "track_id", "frame", "x", "y", "condition", "day"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise CanonicalError(f"raw table is missing columns {missing}")
    if len(df) == 0:
        raise CanonicalError("raw table is empty")
    for col in ("condition", "day"):
        if df[col].isna().any():
            raise CanonicalError(
                f"'{col}' is unassigned for some rows; run grouping.assign_groups first"
            )
    if not df[["x", "y", "frame"]].apply(lambda s: s.map(math.isfinite)).all().all():
        raise CanonicalError("raw table contains non-finite frame/x/y values")

    out = pd.DataFrame(index=df.index)
    out["protein"] = df["protein"].to_numpy() if "protein" in df.columns else pd.NA
    out["condition"] = df["condition"].to_numpy()
    out["day"] = df["day"].to_numpy()
    out["movie"] = df["movie"].to_numpy()
    out["cell"] = df["cell"].to_numpy() if "cell" in df.columns else pd.NA
    out["uid"] = df["movie"].astype(str) + ":" + df["track_id"].astype(str)
    out["track_id"] = df["track_id"].to_numpy()
    out["frame"] = df["frame"].astype("int64").to_numpy()
    # tau PER MOVIE when the project mixes acquisition rates, so t_s and every
    # downstream lag are that movie's own. The column always exists, so a consumer
    # can use it without asking whether the project happens to be uniform.
    if params.has_mixed_tau:
        taus = out["movie"].map(lambda m: params.tau_s_for(str(m)))
        if taus.isna().any():
            missing = sorted(set(out.loc[taus.isna(), "movie"]))[:5]
            raise CanonicalError(
                f"mixed-tau project has no frame interval for movies {missing}")
        out["tau_s"] = taus.astype(float).to_numpy()
    else:
        out["tau_s"] = float(params.tau_s)
    out["t_s"] = out["frame"] * out["tau_s"]
    out["x_um"] = df["x"].to_numpy() * params.um_per_pixel
    out["y_um"] = df["y"].to_numpy() * params.um_per_pixel
    out["amp"] = df["amp"].to_numpy() if "amp" in df.columns else float("nan")
    if "dx" in df.columns and "dy" in df.columns:
        sx = pd.to_numeric(df["dx"], errors="coerce").to_numpy(dtype=float)
        sy = pd.to_numeric(df["dy"], errors="coerce").to_numpy(dtype=float)
        sigma = ((sx ** 2 + sy ** 2) / 2.0) ** 0.5 * params.um_per_pixel
        out[LOC_SIGMA_COLUMN] = pd.Series(sigma, index=out.index).where(lambda s: s > 0)

    dup = out.duplicated(["uid", "frame"])
    if dup.any():
        offenders = sorted(pd.unique(out.loc[dup, "uid"]))[:5]
        raise CanonicalError(
            f"{int(dup.sum())} duplicate (uid, frame) rows, e.g. {offenders}. "
            "The same track cannot hold two localizations in one frame."
        )

    n_before = out["uid"].nunique()
    # WHAT THE FLOOR REMOVES, per condition as {steps: tracks}: those tracks have no rows afterwards, so QC's length
    # panel, drawn from this table, could not show them (on TrackMate data the whole real cost of the filter).
    floor_removed: dict = {}
    if min_localizations > 1:
        counts = out.groupby("uid")["frame"].transform("size")
        short = out.loc[counts < min_localizations]
        if len(short):
            per_track = short.groupby("uid").agg(condition=("condition", "first"), n=("frame", "size"))
            for condition, part in per_track.groupby("condition", sort=True, observed=True):
                hist = (part["n"] - 1).value_counts().sort_index()
                floor_removed[str(condition)] = {int(s): int(k) for s, k in hist.items()}
        out = out.loc[counts >= min_localizations]
    n_after = out["uid"].nunique()
    if len(out) == 0:
        raise CanonicalError(
            f"length filter (min_localizations={min_localizations}) removed all "
            f"{n_before} tracks. Check the filter, or the tracking itself."
        )

    out.attrs["params"] = params
    out.attrs["min_localizations"] = min_localizations
    out.attrs["n_tracks_before_filter"] = int(n_before)
    out.attrs["n_tracks_after_filter"] = int(n_after)
    out.attrs["length_floor_removed"] = floor_removed
    # CARRIED EXPLICITLY, NOT LEFT TO PANDAS. `.attrs` survives some operations here and
    # not others, and the cell filter's counts were being lost between `assign_cells` and
    # this table -- so `assignment_funnel` fed the canonical table read zeros and looked
    # like a clean pipeline. The counts describe tracks that no longer have rows, so
    # nothing downstream can recompute them; an explicit hand-over is the only kind that
    # can be checked, and `tests/test_cell_filter_funnel.py` checks it.
    for carried in ("cell_filter_by_movie", "n_tracks_before_cell_filter",
                    "n_tracks_after_cell_filter", "n_tracks_ambiguous_cell",
                    "n_cells", "unsegmented_movies", "cell_filter_tolerance_px",
                    "n_tracks_added_by_tolerance", "cell_filter_near_edge_px",
                    "cell_filter_near_edge_by_movie", "n_tracks_near_edge"):
        if carried in df.attrs:
            out.attrs[carried] = df.attrs[carried]
    cols = CANONICAL_COLUMNS + [c for c in CANONICAL_EXTRA_COLUMNS if c in out]
    return out[cols].sort_values(["uid", "frame"], kind="stable").reset_index(
        drop=True
    )


def track_table(canonical: pd.DataFrame) -> pd.DataFrame:
    """One row per track: grouping + length + frame span.

    A QC view: track-count per condition/day, and how many tracks have gaps
    (``n_localizations < frame_span``).
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise CanonicalError(f"not a canonical table; missing {missing}")
    g = canonical.groupby("uid", observed=True)
    tbl = g.agg(
        protein=("protein", "first"),
        condition=("condition", "first"),
        day=("day", "first"),
        movie=("movie", "first"),
        cell=("cell", "first"),
        n_localizations=("frame", "size"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
    ).reset_index()
    tbl["frame_span"] = tbl["last_frame"] - tbl["first_frame"] + 1
    tbl["has_gaps"] = tbl["n_localizations"] < tbl["frame_span"]
    return tbl


#: The shortest cap offered: a track of 4 steps is the shortest the length filter keeps.
MIN_CAP_STEPS = 4


def _steps_per_track(table):
    """``uid -> steps`` (localisations - 1, as `qc.track_summary` counts them)."""
    return table.groupby("uid", observed=True)["frame"].size() - 1


def track_cap_cost(table, max_steps: int) -> dict:
    """What leaving out every track LONGER than ``max_steps`` steps would remove, before it
    is done: tracks and localisations, in number and share, and the longest track there."""
    steps = _steps_per_track(table)
    over = steps[steps > int(max_steps)]
    locs = int((over + 1).sum())
    return {"max_steps": int(max_steps), "n_tracks": int(len(steps)),
            "n_tracks_removed": int(len(over)), "n_localizations": int(len(table)),
            "n_localizations_removed": locs,
            "longest": int(steps.max()) if len(steps) else 0}


def cap_track_length(table, max_steps):
    """``table`` without every track LONGER than ``max_steps`` steps -- a USER'S choice,
    asked for after QC showed a 91-step track ("set a filter so tracks with x steps filter
    out"). ``None`` returns the table as it is.

    THE CUT IS NOT NEUTRAL, and it is recorded so it can be said. A long track is
    preferentially a slow one (a fast molecule leaves the focus or the linking radius
    sooner), so capping length removes slow molecules first and LOWERS a slow fraction;
    some long tracks are also stuck objects or autofluorescent spots, which is the reason
    to want the cap. What left is kept on ``attrs["track_cap"]`` -- the cap, and every
    removed track's condition, steps and localisations -- so QC can draw them as removed
    and every caption can name the cap. The length FLOOR (`to_canonical`'s
    ``min_localizations``) is separate and unchanged.
    """
    if max_steps is None:
        return table
    max_steps = int(max_steps)
    if max_steps < MIN_CAP_STEPS:
        raise CanonicalError(f"a track-length cap below {MIN_CAP_STEPS} steps would remove "
                             f"every track the length filter keeps")
    steps = _steps_per_track(table)
    over = steps[steps > max_steps]
    keep = ~table["uid"].isin(over.index)
    if not keep.any():
        raise CanonicalError(f"capping at {max_steps} steps would remove every track")
    out = table.loc[keep].copy()
    out.attrs = dict(table.attrs)
    first = without_attrs(table).drop_duplicates("uid").set_index("uid")
    removed = [{"uid": str(uid), "condition": str(first.at[uid, "condition"]),
                "steps": int(n), "localizations": int(n) + 1}
               for uid, n in over.items()]
    out.attrs["track_cap"] = {"max_steps": max_steps, "removed": removed}
    return out


def without_attrs(table):
    """A SHALLOW copy of `table` with empty `.attrs`, for a loop over its tracks.

    pandas deep-copies `.attrs` into every slice a groupby hands out, every column read
    off that slice, and every row taken with `.iloc` -- and a project table carries its
    acquisition parameters, filter records and, once projected, a per-cell `geometry`
    record there (0.63 ms per copy on protein A/Malate, 0.53 of it `geometry`). A per-track
    loop pays that thousands of times: `appd.apparent_D` took 53.7 s instead of 2.1 s,
    and `hmm.decode_states` held the HMM spatial map for minutes, which is how this was
    found ("its stuck"). The data is shared, not copied, and the caller's table keeps its
    `.attrs`; only the loop's view drops them, which no per-track computation reads.
    """
    out = table.copy(deep=False)
    out.attrs = {}
    return out
