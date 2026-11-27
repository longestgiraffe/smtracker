"""Synthetic single-molecule PROJECTS with known truth: SynR, a carbon-source regulator.

Decided 13 Oct 2026 (the experimenter): a synthetic dataset that cleanly shows the new SMTracker's functions, for a software
paper aimed at Nucleic Acids Research. The design, the parameters and the reasons are `docs/synthetic_synr.md`;
this module builds it. SynR is synthetic and CcpA-inspired: no real protein is implied.

WHAT IS SIMULATED
-----------------
Rod-shaped cells as CAPSULES (two semicircular caps), placed without overlap in a field of view; inside each, a
NUCLEOID of one lobe (short cells) or two (long cells, replicated chromosomes, a gap at mid-cell). A molecule is in
one of three states (`STATES`, slowest first, the order SMTracker sorts its populations in):

- ``B`` bound: specifically bound at an operator, ANCHORED where it bound (D = 0; localisation error alone moves
  its reported position, so its apparent D is the floor sigma^2 / tau);
- ``S`` scanning: non-specific DNA contact, diffusing inside the nucleoid and reflected at its edge;
- ``F`` free: diffusing in the cytoplasm, reflected at the cell's edge.

Transitions are a continuous-time Markov chain F <-> S <-> B (facilitated diffusion: no direct F <-> B), propagated
with the EXACT per-sub-step matrix (`simulate.ConditionSpec.transition_matrix`, never a first-order step). One
spatial rule changes the chain: a free molecule can start scanning only while it is INSIDE the nucleoid, so the
F -> S rate is set higher inside (`rates_for`) to make the effective exchange the one asked for. The truth an
estimator is scored against is therefore the EMPIRICAL occupancy -- the true state of every localisation and every
step, counted -- reported per group and per day beside the target the rates were built for.

The camera: each frame is the average of ``n_subframes`` sub-step positions (motion blur), plus Gaussian
localisation error ``sigma``; photobleaching ends a track after a geometric number of frames; tracks start at
random frames; tracks under ``min_localizations`` are not written (u-track's own minimum). Stage 1 writes tracks
PERFECTLY LINKED: the analysis is judged, not a tracker (stage 2, rendered movies, judges detection and linking).

REPLICATE DAYS ARE INDEPENDENT (Rule 0): each day's S -> B rate is scaled by its own lognormal factor, so the
bound fraction moves between days; days of different groups share nothing.

SYNR v2 (13 Oct 2026, asked for: "hmm ... when for some reason in one state it binds longer to dna", and "the time
differential thing ... the TF stops binding and falls off (switch of medium)") adds two groups after v1's five, so
v1's groups come out identical under the same seed:

- ``SynR-tight_glucose``: an operator variant bound FIVE TIMES LONGER (residence 2.5 s against 0.5 s) and bound
  five times less often, so its bound fraction is glucose's. The SQD fractions cannot tell the two apart; the HMM's
  rates must (a fivefold lower k(B -> S)).
- ``SynR_switch``: glucose, then malate. From ``switch_frame`` every rate relaxes from glucose's to malate's with
  ``switch_half_life_s`` (``rates(t) = after + (before - after) * 2^(-t / t_half)``), as a co-activator would fall
  after a medium switch in a flow cell; 2000-frame movies with the switch at frame 500, and 48 molecules a cell so a
  one-second bin holds enough steps. The truth is the bound share of localisations per second around the switch.

WHAT IS WRITTEN (`build_project`), in the lab's prepared-project layout, so the app opens it like real data:
``<group>/trackmate/seriesNNN.xml`` (TrackMate XML, µm), ``<group>/cell_meshes/seriesNNN.mat`` (Oufti, through
`cellmesh.write_cell_meshes`), ``<group>/bleaching/crops.json`` (pixel size and frame interval, which a TrackMate
project reads), and ``truth/``: ``truth.json`` and one ``<group>_localisations.csv.gz`` with the true state of every
localisation, one ``<group>_cells.csv`` with every cell's placement and nucleoid. Series 001-010 are day 1,
011-020 day 2, ...: the default day rule.

SYNK (21 Oct 2026, asked: "plot the Ds over time to measure how fast it drops ... the event might trigger at
different timepoints for different molecules of the same cell, for different cells of the same movies and so on",
built for every direction, rate and timescale, and shown in the paper): ``GroupSpec(kind="synk", induction=...)``.
An INDUCTION at t = 0 makes each molecule respond from its OWN onset -- the common delay plus offsets drawn per day,
movie, cell and molecule (`Induction`) -- by a declared shape (step, linear, exponential, sigmoid). The response
``h`` in [0, 1] moves every state's D from its before to its after value, and the rates from those holding the
before occupancy to those holding the after one, so D, the fractions or both change, in either direction. Each
movie starts at its own time since the induction (`Induction.movie_start_s`): before it within one movie (a trigger
inside the movie) or minutes after it (movies across a session). No nucleoid: every state diffuses in the cell at
its own D(t) (D 0 stays where it is). The truth adds per localisation its time since the induction, its molecule's
onset, ``h`` and its state's true D.

SYNK THROUGH A TRACKER (26 Oct 2026, asked: "lets build synK as well. u know we need to show that we can show a drop
of D"): `build_movie_project` renders SynK as it renders SynR (`SYNK_GROUPS`: the free D dropping, the bound share
rising to the same mean D, and nothing, on one clock). Every stage-2 camera file is a real MetaMorph `.stk`
(`write_stk`: one IFD, the planes contiguous, a creation stamp per plane), recorded on a declared calendar
(`MOVIE_FIRST_DATE`, `MOVIE_CLOCK`): each replicate a week apart, a SynK group's addition at the clock and each
movie at the addition plus its `Induction.movie_start_s`, so the app's "Time after the addition" reads the movies'
own stamps as it reads one user's.

IN 3D (`GroupSpec.dims` 3; 30-31 Oct 2026): the free protein, SynR, SynG and SynK move in the cell's VOLUME in
micro-steps (`MICRO_STEP_UM`), SynN's own 3D model takes the same walk, and each records its heights (which set focus)
and the INSTANTANEOUS position the truth reports; `SYNR_3D_GROUPS`, `SYNG_V2_3D_GROUPS`, `SYNK_3D_GROUPS`,
`SYNN_3D_GROUPS` are the 2D declarations with ``dims`` 3. A group's 2D draws are unchanged.

INVARIANTS: no Qt; deterministic under a seed; fails loud on a cell that cannot be placed, an empty group, or an
existing output folder (nothing is overwritten).
"""
from __future__ import annotations

import datetime as _dt
import gzip
import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .induction import RESPONSE_SHAPES, response_shape

__all__ = ["STATES", "Lobe", "CapsuleCell", "GroupSpec", "SYNR_GROUPS", "SYNR_V1_GROUPS", "SYNR_TIGHT",
           "SYNR_SWITCH", "SynRDesign",
           "nucleoid_lobes", "place_cells", "rates_for", "simulate_movie", "write_trackmate_xml",
           "build_project", "SyntheticError", "MOVIE_CALIBRATION", "MOVIE_CALIBRATION_N1", "render_phase",
           "build_movie_project", "write_true_outlines", "SYNR_FREE_FAST", "MOVIE_RENDERING",
           "MovieRealism", "SYNR_V3_GROUPS", "SYNR_V3_LOWCOPY", "SYNR_V3_HIGHCOPY", "SYNR_V3_REALISM", "SYNG_N1_REALISM",
           "Induction", "INDUCTION_SHAPES", "SYNG_GROUPS", "SYNG_V2_GROUPS", "fork_share",
           "SYNR_FREE_3D", "SYNR_FREE_3D_FAST", "SYNR_3D_GROUPS", "SYNG_V2_3D_GROUPS", "SYNG_2STATE_3D_GROUPS", "SYNK_3D_GROUPS",
           "SYNN_3D_GROUPS", "nucleoid_share", "MICRO_STEP_UM"]

#: The states, slowest first.
STATES = ("B", "S", "F")


class SyntheticError(RuntimeError):
    """A synthetic project could not be built as asked."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Lobe:
    """One nucleoid lobe: an ellipse along the cell axis, in the cell's own frame (µm from the cell centre)."""

    centre_um: float
    half_length_um: float
    half_width_um: float

    def rho(self, u, v):
        """Normalised radius: < 1 inside the lobe."""
        return np.hypot((np.asarray(u) - self.centre_um) / self.half_length_um,
                        np.asarray(v) / self.half_width_um)

    @property
    def area(self) -> float:
        return math.pi * self.half_length_um * self.half_width_um

    # -- 3D (`GroupSpec.dims` 3): the lobe as a body of revolution about the cell's axis, an ellipsoid of the same
    # -- half-length and half-width, so its projection is the 2D lobe
    def rho3(self, u, v, z):
        """`rho` in 3D: < 1 inside the ellipsoid."""
        return np.sqrt(((np.asarray(u) - self.centre_um) / self.half_length_um) ** 2
                       + (np.asarray(v) ** 2 + np.asarray(z) ** 2) / self.half_width_um ** 2)

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.half_length_um * self.half_width_um ** 2


def _rho3(lobe: Lobe, u: float, v: float, z: float) -> float:
    """`Lobe.rho3` for one point, without numpy (the 3D simulation asks it every sub-step)."""
    a = (float(u) - lobe.centre_um) / lobe.half_length_um
    return math.sqrt(a * a + (float(v) ** 2 + float(z) ** 2) / lobe.half_width_um ** 2)


def nucleoid_lobes(length_um: float, *, split_length_um: float = 3.8, fraction: float = 0.72,
                   half_width_um: float = 0.33, two_lobe_fraction: float = 0.34) -> tuple:
    """The nucleoid of a cell of ``length_um``: one centred lobe over ``fraction`` of the length, or -- from
    ``split_length_um`` -- two lobes centred at +-L/4, each ``two_lobe_fraction`` of the length long, leaving a gap
    at mid-cell (replicated, segregating chromosomes)."""
    if length_um < split_length_um:
        return (Lobe(0.0, fraction * length_um / 2.0, half_width_um),)
    half = two_lobe_fraction * length_um / 2.0
    return (Lobe(-length_um / 4.0, half, half_width_um), Lobe(length_um / 4.0, half, half_width_um))


@dataclass(frozen=True)
class CapsuleCell:
    """A capsule-shaped cell placed in the field: centre (µm), axis angle (rad), length and width (µm)."""

    cell_id: int
    cx_um: float
    cy_um: float
    angle: float
    length_um: float
    width_um: float
    lobes: tuple

    @property
    def radius(self) -> float:
        return self.width_um / 2.0

    @property
    def half_axis(self) -> float:
        """Half the length of the straight segment between the caps' centres."""
        return self.length_um / 2.0 - self.radius

    def to_local(self, x, y):
        dx, dy = np.asarray(x) - self.cx_um, np.asarray(y) - self.cy_um
        c, s = math.cos(self.angle), math.sin(self.angle)
        return dx * c + dy * s, -dx * s + dy * c

    def to_field(self, u, v):
        c, s = math.cos(self.angle), math.sin(self.angle)
        u, v = np.asarray(u), np.asarray(v)
        return self.cx_um + u * c - v * s, self.cy_um + u * s + v * c

    def axis_distance(self, u, v):
        """Distance from (u, v) to the axis segment, and the nearest axis point's u."""
        q = np.clip(u, -self.half_axis, self.half_axis)
        return np.hypot(np.asarray(u) - q, v), q

    def inside(self, u, v) -> np.ndarray:
        d, _q = self.axis_distance(u, v)
        return d <= self.radius

    def in_nucleoid(self, u, v) -> np.ndarray:
        return np.any([lobe.rho(u, v) <= 1.0 for lobe in self.lobes], axis=0)

    def reflect_in_cell(self, u: float, v: float) -> tuple:
        """Reflect a point that left the capsule back across its edge (radially from the axis segment), exact for
        the caps and the walls."""
        d, q = self.axis_distance(u, v)
        d, q = float(d), float(q)
        if d <= self.radius:
            return u, v
        back = max(0.0, 2.0 * self.radius - d)
        return q + (u - q) * back / d, v * back / d

    # -- SynN's 3D: (u, v) as everywhere, z the height across the rod (the cell is a capsule of revolution) -- #
    def axis_distance3(self, u: float, v: float, z: float) -> tuple:
        """Distance from (u, v, z) to the axis segment, and the nearest axis point's u."""
        q = min(max(float(u), -self.half_axis), self.half_axis)
        return math.sqrt((float(u) - q) ** 2 + float(v) ** 2 + float(z) ** 2), q

    def reflect_in_cell3(self, u: float, v: float, z: float) -> tuple:
        """`reflect_in_cell` in 3D: radially from the axis segment, exact for the cylinder and the caps."""
        d, q = self.axis_distance3(u, v, z)
        if d <= self.radius:
            return u, v, z
        back = max(0.0, 2.0 * self.radius - d) / d
        return q + (u - q) * back, v * back, z * back

    def to_surface3(self, u: float, v: float, z: float) -> tuple:
        """The membrane point radially outward of (u, v, z)."""
        d, q = self.axis_distance3(u, v, z)
        if d == 0.0:
            return u, self.radius, 0.0
        s = self.radius / d
        return q + (u - q) * s, v * s, z * s

    def volume(self, radius: float | None = None) -> float:
        """The capsule's volume (um^3), or that of the capsule ``radius`` around the same axis segment."""
        r = self.radius if radius is None else float(radius)
        return math.pi * r * r * 2.0 * self.half_axis + 4.0 / 3.0 * math.pi * r ** 3

    def shell_share(self, depth_um: float) -> float:
        """The share of the cell's volume within ``depth_um`` of its membrane."""
        return 1.0 - self.volume(max(0.0, self.radius - float(depth_um))) / self.volume()

    # -- SynR and SynG in 3D (`GroupSpec.dims` 3): the nucleoid's lobes as ellipsoids about the axis (`Lobe.rho3`),
    # -- the forks on the axis --
    def in_nucleoid3(self, u: float, v: float, z: float) -> bool:
        return any(_rho3(lobe, u, v, z) <= 1.0 for lobe in self.lobes)

    def lobe_of3(self, u: float, v: float, z: float) -> "Lobe":
        """The lobe a point is in, else the nearest (in normalised radius)."""
        return min(self.lobes, key=lambda lobe: _rho3(lobe, u, v, z))

    def reflect_in_lobe3(self, u: float, v: float, z: float, lobe: "Lobe") -> tuple:
        """`reflect_in_lobe` in 3D: radially in the lobe's normalised frame; a step too long lands on the edge."""
        r = _rho3(lobe, u, v, z)
        if r <= 1.0:
            return u, v, z
        back = max(0.0, 2.0 - r) / r
        return lobe.centre_um + (u - lobe.centre_um) * back, v * back, z * back

    def nucleoid_volume(self) -> float:
        return sum(lobe.volume for lobe in self.lobes)

    def fork_sites3(self) -> list:
        """`fork_sites` in 3D, ``[(u, v, z), ...]``: on the axis, so at the focal plane's height."""
        return [(u, v, 0.0) for u, v in self.fork_sites()]

    def reflect_in_lobe(self, u: float, v: float, lobe: "Lobe") -> tuple:
        """Reflect a point that left ``lobe`` back inside it (radially in the lobe's normalised frame); a step too
        long to reflect lands on the edge."""
        r = float(lobe.rho(u, v))
        if r <= 1.0:
            return u, v
        back = max(0.0, 2.0 - r) / r
        return lobe.centre_um + (u - lobe.centre_um) * back, v * back

    def fork_sites(self) -> list:
        """SynG's replication forks, ``[(u, v), ...]`` in the cell's frame: two per nucleoid lobe, its centre
        +- `FORK_HALF_SPACING_UM` along the axis -- the replisomes at mid-cell, or at the quarters once the
        chromosome has replicated into two lobes."""
        return [(lobe.centre_um + d, 0.0) for lobe in self.lobes for d in (-FORK_HALF_SPACING_UM, FORK_HALF_SPACING_UM)]

    def lobe_of(self, u: float, v: float) -> "Lobe":
        """The lobe a point is in, else the nearest (in normalised radius)."""
        return min(self.lobes, key=lambda lobe: float(lobe.rho(u, v)))

    def area(self) -> float:
        return 2.0 * self.half_axis * self.width_um + math.pi * self.radius ** 2

    def nucleoid_area(self) -> float:
        return sum(lobe.area for lobe in self.lobes)

    def outline(self, n_cap: int = 24) -> np.ndarray:
        """The capsule's polygon in the field (µm), counter-clockwise."""
        a = np.linspace(-math.pi / 2, math.pi / 2, n_cap)
        right = np.c_[self.half_axis + self.radius * np.cos(a), self.radius * np.sin(a)]
        left = np.c_[-self.half_axis - self.radius * np.cos(a), -self.radius * np.sin(a)]
        loop = np.vstack([right, left, right[:1]])          # closed: the first point repeated
        x, y = self.to_field(loop[:, 0], loop[:, 1])
        return np.c_[x, y]

    def ribs(self, n: int = 40) -> np.ndarray:
        """Oufti's ``mesh``: ``[x_left, y_left, x_right, y_right]`` per cross-section, pole to pole (µm)."""
        s = np.linspace(-self.length_um / 2.0, self.length_um / 2.0, n)
        cap = np.maximum(0.0, np.abs(s) - self.half_axis)
        half = np.sqrt(np.clip(self.radius ** 2 - cap ** 2, 0.0, None))
        xl, yl = self.to_field(s, half)
        xr, yr = self.to_field(s, -half)
        return np.c_[xl, yl, xr, yr]


def _segment_distance(a0, a1, b0, b1) -> float:
    """Shortest distance between two 2-D segments."""
    def point_seg(p, s0, s1):
        d = s1 - s0
        t = 0.0 if not d.any() else float(np.clip(np.dot(p - s0, d) / np.dot(d, d), 0.0, 1.0))
        return float(np.hypot(*(p - (s0 + t * d))))

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    if (cross(a0, a1, b0) * cross(a0, a1, b1) < 0) and (cross(b0, b1, a0) * cross(b0, b1, a1) < 0):
        return 0.0
    return min(point_seg(a0, b0, b1), point_seg(a1, b0, b1), point_seg(b0, a0, a1), point_seg(b1, a0, a1))


def _biology(design: "SynRDesign", spec: "GroupSpec", bio_rng, cells: list) -> tuple:
    """A movie's biological noise below the day: its factor on k(S -> B), each cell's factor, and each cell's
    molecule count (None where that noise is off), drawn from ``bio_rng`` only."""
    per_cell = spec.molecules_per_cell or design.molecules_per_cell
    binding = not spec.free_only
    movie = float(math.exp(bio_rng.normal(0.0, design.movie_bound_sd))) if binding and design.movie_bound_sd else 1.0
    factors = ({c.cell_id: float(math.exp(bio_rng.normal(0.0, design.cell_bound_sd))) for c in cells}
               if binding and design.cell_bound_sd else None)
    if design.molecules_range:
        lo, hi = (int(v) for v in design.molecules_range)
        counts = {c.cell_id: int(bio_rng.integers(lo, hi + 1)) for c in cells}
    else:
        # MEAN-preserving: ``molecules_per_cell`` is the cells' average (ruled 14 Oct 2026: "a mean of 30 ...
        # with bigger variance"), the spread the measured one
        sd = float(design.molecules_sd_log)
        counts = ({c.cell_id: max(1, int(round(per_cell * math.exp(bio_rng.normal(-0.5 * sd * sd, sd)))))
                   for c in cells} if sd else None)
    return movie, factors, counts


def _bio_rng(design: "SynRDesign", name: str):
    """The biological noise's own stream, per group BY NAME (adding a group changes no other)."""
    return np.random.default_rng([design.seed, 11, zlib.crc32(name.encode("utf-8"))])


def place_cells(rng, lengths, *, width_um: float, field_um: tuple, margin_um: float = 0.3,
                gap_um: float = 0.25, tries: int = 4000, **lobe_kw) -> list:
    """Capsules of ``lengths`` placed at random positions and orientations inside ``field_um`` (w, h), none
    touching another (axis segments further apart than a width plus ``gap_um``). Fails loud when a cell cannot be
    placed in ``tries`` attempts: a crowded field is a design error, never silently a smaller field."""
    cells = []
    w, h = field_um
    for index, length in enumerate(lengths):
        for _ in range(tries):
            angle = float(rng.uniform(0.0, math.pi))
            reach = length / 2.0 + margin_um
            cx = float(rng.uniform(reach, w - reach))
            cy = float(rng.uniform(reach, h - reach))
            cell = CapsuleCell(index + 1, cx, cy, angle, float(length), float(width_um),
                               nucleoid_lobes(float(length), **lobe_kw))
            ends = np.array(cell.to_field([-cell.half_axis, cell.half_axis], [0.0, 0.0])).T
            if not np.all((ends >= cell.radius + margin_um) & (ends <= np.array([w, h]) - cell.radius - margin_um)):
                continue
            ok = all(_segment_distance(ends[0], ends[1], *other_ends) > width_um + gap_um
                     for other_ends in (np.array(o.to_field([-o.half_axis, o.half_axis], [0.0, 0.0])).T
                                        for o in cells))
            if ok:
                cells.append(cell)
                break
        else:
            raise SyntheticError(f"could not place cell {index + 1} of {len(lengths)} in a "
                                 f"{w:g} x {h:g} µm field: fewer or shorter cells, or a larger field")
    return cells


# --------------------------------------------------------------------------- #
# The groups and their rates
# --------------------------------------------------------------------------- #


#: How a molecule responds from its own onset: ``h(dt)`` rises from 0 to 1 (`Induction.response`); the shapes
#: and their arithmetic are `induction`'s, which the model fitted to the data uses too.
INDUCTION_SHAPES = RESPONSE_SHAPES


@dataclass(frozen=True)
class Induction:
    """SynK's induction at t = 0: what each state becomes, how fast, and when each molecule starts responding.

    ``D_after_um2_s`` / ``occupancy_after``: the states' D and occupancy once the response is complete (equal to
    the group's own = an uninduced control on the same clock). ``movie_start_s``: each movie's first frame in
    seconds since the induction, by the movie's place in its day (cycled; negative = the induction falls inside the
    movie). ``shape`` and ``time_s``: `INDUCTION_SHAPES` -- step (instant; ``time_s`` unused), linear (complete
    after ``time_s``), exponential (half-life ``time_s``), sigmoid (Hill, half at ``time_s``, steepness ``hill``).
    A molecule's ONSET is ``delay_s`` plus a normal offset per day, movie, cell and molecule with these SDs, floored
    at 0 (nothing responds before the induction)."""

    D_after_um2_s: tuple
    occupancy_after: tuple
    movie_start_s: tuple
    shape: str = "exponential"
    time_s: float = 5.0
    hill: float = 3.0
    delay_s: float = 0.0
    onset_sd_day_s: float = 0.0
    onset_sd_movie_s: float = 0.0
    onset_sd_cell_s: float = 0.0
    onset_sd_molecule_s: float = 0.0
    #: SESSION DRIFT (8 Nov 2026): the fraction of every moving state's D lost per hour since the session's first
    #: movie (the earliest ``movie_start_s``), the same for every group, the mock included, applied to a whole movie
    #: at its start. Measured on a lab project with no addition: each movie's mean squared jump falls over the session
    #: in both its conditions (`manuscript/kinetics_plan/real_data_null/`). 0 = none: nothing
    #: changes, draw for draw.
    session_drift_per_hour: float = 0.0

    def __post_init__(self):
        if self.shape not in INDUCTION_SHAPES:
            raise SyntheticError(f"induction shape {self.shape!r}: choose one of {INDUCTION_SHAPES}")
        if self.shape != "step" and not self.time_s > 0:
            raise SyntheticError(f"a {self.shape} response needs time_s > 0, got {self.time_s}")
        if self.shape == "sigmoid" and not self.hill > 0:
            raise SyntheticError(f"a sigmoid response needs hill > 0, got {self.hill}")
        if not self.movie_start_s:
            raise SyntheticError("movie_start_s is empty: every movie needs its time since the induction")
        if self.delay_s < 0 or min(self.onset_sd_day_s, self.onset_sd_movie_s, self.onset_sd_cell_s,
                                   self.onset_sd_molecule_s) < 0:
            raise SyntheticError("the delay and the onset SDs cannot be negative")
        if not 0.0 <= self.session_drift_per_hour < 1.0:
            raise SyntheticError(f"a session drift is a fraction lost per hour in [0, 1), got "
                                 f"{self.session_drift_per_hour}")

    def response(self, since_onset_s):
        """``h`` in [0, 1] at ``since_onset_s`` seconds after a molecule's own onset (0 before it)."""
        return response_shape(self.shape, since_onset_s, None if self.shape == "step" else self.time_s, self.hill)


@dataclass(frozen=True)
class GroupSpec:
    """One group: its diffusion coefficients (B, S, F), the occupancy its rates are built for, the bound
    residence (1 / k(B -> S)), and the S -> F rate. ``free_only`` makes a one-state free protein."""

    name: str
    D_um2_s: tuple = (0.0, 0.25, 1.8)
    occupancy: tuple = (0.30, 0.35, 0.35)
    k_bound_off_per_s: float = 2.0
    k_scan_off_per_s: float = 10.0
    free_only: bool = False
    description: str = ""
    #: This group's movies, when not the design's (a switch needs a longer window).
    n_frames: int | None = None
    molecules_per_cell: int | None = None
    #: A switch: from this frame the rates relax to ``after``'s with this half-life.
    switch_frame: int | None = None
    switch_half_life_s: float | None = None
    after: "GroupSpec | None" = None
    #: "synr" (the default: B / S / F, 2D, the nucleoid), "synn" (SynN: M / R / F in the B / S / F slots, 3D, the
    #: membrane; `_simulate_synn`) or "synk" (SynK: an induction changes D and occupancy over time, 2D, F the hub;
    #: `_simulate_synk`).
    kind: str = "synr"
    #: SynN: how close to the membrane a free molecule must be to bind there (um).
    membrane_shell_um: float = 0.05
    #: SynG: B is bound at a replication FORK (`CapsuleCell.fork_sites`), entered only within this distance of
    #: one (um) and then sitting on it; None = SynR's operators, anywhere in the nucleoid.
    fork_capture_um: float | None = None
    #: SynK: the induction (required for kind "synk", refused elsewhere).
    induction: Induction | None = None
    #: A STUCK FRACTION of a free protein (asked 25 Oct 2026: "it makes sense to include a stuck fraction for free mng
    #: as well, that was something we missed", "around 3-6 percent fluctuating", anywhere in the cell): ``(low,
    #: high)``, each replicate's share of molecules drawn uniformly in it (`stuck_level`), each molecule stuck or not
    #: at random. A stuck molecule starts anywhere in the cell and stays there, D 0 and height fixed, until it
    #: bleaches; its true state is B. DECLARED, never fitted to a real free protein. Free proteins only.
    stuck_range: tuple | None = None
    #: 3 = the molecule moves in the cell's VOLUME, a capsule of revolution, and the camera sees its projection
    #: (asked 30 Oct 2026: "in the end we should build all as more of a 3d model"). 2 = in the outline's plane,
    #: as every group built before (the height only set focus). A free protein (`_simulate_free3d`), SynR / SynG
    #: with the nucleoid's lobes as ellipsoids about the axis and the forks on it (`_simulate_synr3d`, 31 Oct 2026),
    #: and SynK in the capsule (`_simulate_synk`, 31 Oct 2026). SynN's model was always 3D: for SynN, 3 is the walk
    #: of 31 Oct 2026 (micro-steps, the instantaneous truth) and 2 keeps its walk of 21 Oct, as its sets were built.
    #: SynR's switch stays 2D.
    dims: int = 2

    def __post_init__(self):
        if self.dims not in (2, 3):
            raise SyntheticError(f"{self.name}: dims is 2 or 3, not {self.dims}")
        if self.dims == 3 and self.switch_frame is not None:
            raise SyntheticError(f"{self.name}: a 3D group has no switch (SynR's switch is 2D)")
        if self.stuck_range is not None:
            low, high = (float(v) for v in self.stuck_range)
            if not self.free_only or not 0.0 <= low <= high < 1.0:
                raise SyntheticError(f"{self.name}: a stuck range is (low, high) with 0 <= low <= high < 1, on a free "
                                     f"protein only")
        if (self.kind == "synk") != (self.induction is not None):
            raise SyntheticError(f"{self.name}: an induction belongs to a SynK group (kind 'synk') and a SynK group "
                                 f"needs one")
        if self.induction is not None:
            ind = self.induction
            if len(ind.D_after_um2_s) != len(self.D_um2_s) or len(ind.occupancy_after) != len(self.occupancy):
                raise SyntheticError(f"{self.name}: the induction's D and occupancy need one value per state "
                                     f"({len(self.D_um2_s)})")
            if len(self.D_um2_s) != 3:
                raise SyntheticError(f"{self.name}: SynK uses the three slots B / S / F (a state can be left empty "
                                     f"by an occupancy of 0)")
        if self.two_state and (self.dims != 3 or self.kind != "synr" or float(self.occupancy[0]) <= 0.0):
            raise SyntheticError(f"{self.name}: a two-state SynG (no S, F <-> B at a fork) is built in 3D, as a "
                                 f"SynR-kind group with a bound share above 0")

    @property
    def two_state(self) -> bool:
        """A SynG with NO middle state (`SYNG_2STATE_3D_GROUPS`): S's occupancy is 0, and F binds a fork directly
        (F <-> B, entered only within ``fork_capture_um`` of one)."""
        return (self.fork_capture_um is not None and not self.free_only and len(self.occupancy) == 3
                and float(self.occupancy[1]) == 0.0)

    def switch_weight(self, frame, tau_s: float):
        """How much of the BEFORE rates still applies at ``frame`` (1 before the switch, halving every
        ``switch_half_life_s`` after it); 1 for a group without a switch."""
        frame = np.asarray(frame, float)
        if self.switch_frame is None:
            return np.ones_like(frame)
        t = np.clip((frame - self.switch_frame) * tau_s, 0.0, None)
        return np.power(0.5, t / float(self.switch_half_life_s))


#: The five groups of SynR v1 (`docs/synthetic_synr.md`).
SYNR_GROUPS = (
    GroupSpec("SynR_glucose", occupancy=(0.30, 0.35, 0.35), k_bound_off_per_s=2.0,
              description="wild type on glucose: co-activator high, long operator residence"),
    GroupSpec("SynR_malate", occupancy=(0.12, 0.38, 0.50), k_bound_off_per_s=4.0,
              description="wild type on malate: co-activator low"),
    GroupSpec("SynR-coA_glucose", occupancy=(0.04, 0.36, 0.60), k_bound_off_per_s=6.0,
              description="co-activator mutant on glucose: barely binds"),
    GroupSpec("SynR-coA_malate", occupancy=(0.04, 0.36, 0.60), k_bound_off_per_s=6.0,
              description="co-activator mutant on malate: the same as on glucose (the null pair)"),
    GroupSpec("mNG_free", D_um2_s=(2.0,), occupancy=(1.0,), free_only=True,
              description="free fluorescent protein, one state: BIC must choose one"),
)
#: SynR v1, as built and measured on 13 Oct 2026.
SYNR_V1_GROUPS = SYNR_GROUPS

#: SynR v2's two groups (`docs/synthetic_synr.md`): the HMM's case and the triggered kinetics' case.
SYNR_TIGHT = GroupSpec(
    "SynR-tight_glucose", occupancy=(0.30, 0.35, 0.35), k_bound_off_per_s=0.4,
    description="operator variant on glucose: bound five times longer and five times less often, so the bound "
                "FRACTION is the wild type's and only the rates differ")
SYNR_SWITCH = GroupSpec(
    "SynR_switch", occupancy=(0.30, 0.35, 0.35), k_bound_off_per_s=2.0, n_frames=2000, molecules_per_cell=48,
    switch_frame=500, switch_half_life_s=5.0, after=SYNR_GROUPS[1],
    description="glucose, then malate from frame 500 (10 s): every rate relaxes to malate's with a 5 s half-life")
#: Free mNeonGreen at a REALISTIC cytoplasmic D (asked 14 Oct 2026: "u can do a mng with this D, on top, just
#: to see and compare"). A GFP-sized protein in bacterial cytoplasm runs 3-10 um^2/s
#: (`docs/claude/linking_and_limits.md`); at 5 most molecules never form a track at 20 ms, so this group shows the
#: detection ceiling and survivorship: what is tracked is the slow tail. `mNG_free` (D 2.0) stays as the
#: trackable free control. Appended last, so every earlier group is unchanged under the same seed.
SYNR_FREE_FAST = GroupSpec("mNG_free_D5", D_um2_s=(5.0,), occupancy=(1.0,), free_only=True,
                           description="free fluorescent protein at a realistic D of 5 um^2/s: most molecules "
                                       "never form a track")
#: Free mNG with a STUCK FRACTION (25 Oct 2026, declared by the experimenter: "around 3-6 percent fluctuating", anywhere in
#: the cell): mNG_free's D 2.0 for the rest. Appended last, so every earlier group is unchanged under the same seed.
SYNR_FREE_STUCK = GroupSpec("mNG_free_stuck", D_um2_s=(2.0,), occupancy=(1.0,), free_only=True,
                            stuck_range=(0.03, 0.06),
                            description="free fluorescent protein, 3-6 % of molecules stuck (D 0) anywhere in the "
                                        "cell, the share drawn per replicate")
SYNR_GROUPS = SYNR_V1_GROUPS + (SYNR_TIGHT, SYNR_SWITCH, SYNR_FREE_FAST, SYNR_FREE_STUCK)


def stuck_level(design, spec, day: int) -> float:
    """Replicate ``day``'s stuck share for ``spec`` (0 without a stuck range): uniform in its range, from a stream of
    its own (the seed, the group's name, the day), so drawing it moves no other molecule or group."""
    if spec.stuck_range is None:
        return 0.0
    low, high = (float(v) for v in spec.stuck_range)
    rng = np.random.default_rng([design.seed, 17, zlib.crc32(spec.name.encode("utf-8")), int(day)])
    return float(rng.uniform(low, high))

#: FREE mNG IN 3D (asked 30 Oct 2026: "free mng 3d with D 2 and 5", "3 x 1 micrometer cell"; and, after the review
#: that day, NO stuck fraction: a stuck share declared to match the measured slow population would put the answer
#: into the benchmark). One free state moving in the capsule's volume (`_simulate_free3d`); the truth holds each
#: frame's INSTANTANEOUS position beside the blurred centroid the camera records. DECLARED, never fitted.
SYNR_FREE_3D = GroupSpec("mNG_free3d", D_um2_s=(2.0,), occupancy=(1.0,), free_only=True, dims=3,
                         description="free fluorescent protein in 3D, D 2 um^2/s, one state, nothing stuck")
SYNR_FREE_3D_FAST = GroupSpec("mNG_free3d_D5", D_um2_s=(5.0,), occupancy=(1.0,), free_only=True, dims=3,
                              description="free fluorescent protein in 3D, D 5 um^2/s, one state, nothing stuck")
SYNR_GROUPS = SYNR_GROUPS + (SYNR_FREE_3D, SYNR_FREE_3D_FAST)

#: SynN (designed 21 Oct 2026 with the experimenter for a +/- rifampicin application; `docs/synthetic_synr.md`, "SynN"):
#: a synthetic RNA nuclease, inspired by a YicC-family endoribonuclease; no real protein is implied. DECLARED,
#: never fitted to lab data. States in SynR's slots: M (B; bound at the membrane / degradosome, D 0, residence
#: 1 s), R (S; RNA-bound in the cytoplasm, D 0.15, residence 0.3 s), F (free hexamer, D 0.9: free mNG's 2.0 scaled by
#: mass^-1/3 for ~360 kDa). F is the hub (F <-> M, F <-> R); a free molecule binds M only within 50 nm of the membrane
#: and then sits on it. Rifampicin removes substrate: the bound occupancies fall, the residences stay.
_SYNN = dict(kind="synn", D_um2_s=(0.0, 0.15, 0.9), k_bound_off_per_s=1.0, k_scan_off_per_s=1.0 / 0.3)
SYNN_GROUPS = (
    GroupSpec("SynN", occupancy=(0.15, 0.35, 0.50), **_SYNN,
              description="SynN untreated, exponential phase: M 15 / R 35 / F 50 %"),
    GroupSpec("SynN_Rif", occupancy=(0.10, 0.25, 0.65), **_SYNN,
              description="SynN after rifampicin, moderate (as 25 ug/ml for 30 min): 15 pp less bound"),
    GroupSpec("SynN_Rif-strong", occupancy=(0.08, 0.17, 0.75), **_SYNN,
              description="SynN after rifampicin, strong: the bound occupancies halved"),
    GroupSpec("SynN-insensitive", occupancy=(0.15, 0.35, 0.50), **_SYNN,
              description="a SynN that does not respond to rifampicin, untreated (the null pair)"),
    GroupSpec("SynN-insensitive_Rif", occupancy=(0.15, 0.35, 0.50), **_SYNN,
              description="the same, after rifampicin: the same truth (the null pair)"),
    GroupSpec("SynN-dRNA", occupancy=(0.0, 0.0, 1.0), **_SYNN,
              description="SynN that never binds: the free hexamer only"),
    GroupSpec("mNG_free_3D", kind="synn", D_um2_s=(0.0, 0.0, 2.0), occupancy=(0.0, 0.0, 1.0),
              description="free mNG (D 2) in SynN's 3D model: the SynN set's free control, untreated"),
)
#: Rifampicin also makes the CYTOPLASM faster: in E. coli the effective viscosity fell ~20 % (200 ug/ml for 1 h,
#: every protein size in the same proportion; Bellotto et al. 2022, eLife, doi:10.7554/eLife.82654), so every
#: cytoplasmic D rises by 1 / 0.8. Declared as an UPPER estimate for B. subtilis at 25 ug/ml for 30 min, where it
#: is not measured. M sits on the membrane and stays at D 0.
RIF_CYTOPLASM_SPEEDUP = 1.0 / 0.8
_SYNN_FLUID = dict(_SYNN, D_um2_s=(0.0, 0.15 * RIF_CYTOPLASM_SPEEDUP, 0.9 * RIF_CYTOPLASM_SPEEDUP))
#: Appended last (asked 21 Oct 2026), so every group above is unchanged under the same seed. The fluid null is the
#: clean test: no occupancy changes, only R and F move faster -- does a fraction difference appear anyway?
SYNN_GROUPS = SYNN_GROUPS + (
    GroupSpec("SynN_Rif-fluid", occupancy=(0.10, 0.25, 0.65), **_SYNN_FLUID,
              description="SynN_Rif's occupancies with the cytoplasm faster under rifampicin: R and F D x 1.25"),
    GroupSpec("SynN-insensitive_Rif-fluid", occupancy=(0.15, 0.35, 0.50), **_SYNN_FLUID,
              description="the null pair's rifampicin side with the cytoplasm faster: only R and F D change"),
)

#: SynG (G for a primase; named SynP until the evening it was designed) (designed 22 Oct 2026 with the experimenter for the
#: mNG-tagged primase +/- DL-serine hydroxamate application, which replaced
#: the SynN application; first labelled with another protein's name by mistake, corrected the same evening): a synthetic PRIMASE, inspired by a bacterial primase; no real
#: protein is implied. DECLARED. SynR's chain and nucleoid, with B bound AT A FORK (`CapsuleCell.fork_sites`: two per
#: nucleoid lobe), S scanning DNA, F free. (p)ppGpp, which SHX induces, arrests replication elongation within minutes
#: by inhibiting primase, and the arrested forks stay intact (Wang, Sanders & Grossman 2007, Cell,
#: doi:10.1016/j.cell.2006.12.043); it mainly blocks primer extension (Giramma et al. 2021, J Mol Biol,
#: doi:10.1016/j.jmb.2021.167189). Tracked one molecule at a time in B. subtilis (Hernandez-Tamayo, Schmitz &
#: Graumann 2021, mSphere, doi:10.1128/mSphere.00948-20), the primase reaches forks by diffusion and capture (as here), binds
#: them LESS OFTEN after SHX with a FASTER turnover, and sits MORE statically after polymerase arrest (HPUra). So SHX
#: moves the fork share down two ways built apart -- fewer bindings (SHX-down) and shorter stays (SHX-turnover), the
#: same fractions by different kinetics -- and up (SHX-up: the HPUra-like direction, to test the method both ways),
#: with a null pair and a never-fork-bound control.
#: D: B 0 (on the replisome), S 0.2, F 1.3 (the fusion, ~96 kDa: free mNG's 2.0 x (27/96)^1/3).
_SYNG = dict(D_um2_s=(0.0, 0.2, 1.3), k_bound_off_per_s=1.0, k_scan_off_per_s=5.0, fork_capture_um=0.1)
SYNG_GROUPS = (
    GroupSpec("SynG", occupancy=(0.12, 0.33, 0.55), **_SYNG,
              description="SynG untreated: fork 12 / scanning 33 / free 55 %, a second at a fork"),
    GroupSpec("SynG_SHX-down", occupancy=(0.06, 0.35, 0.59), **_SYNG,
              description="after SHX, forks recruit it half as often (a second at a fork as before): fork 6 %"),
    GroupSpec("SynG_SHX-turnover", occupancy=(0.06, 0.35, 0.59), **dict(_SYNG, k_bound_off_per_s=2.0),
              description="after SHX, it leaves a fork twice as fast, recruited about as often: fork 6 %"),
    GroupSpec("SynG_SHX-up", occupancy=(0.22, 0.30, 0.48), **dict(_SYNG, k_bound_off_per_s=0.5),
              description="the other direction: forks hold it twice as long (as after HPUra): fork 22 %"),
    GroupSpec("SynG-insensitive", occupancy=(0.12, 0.33, 0.55), **_SYNG,
              description="a SynG that does not respond to SHX, untreated (the null pair)"),
    GroupSpec("SynG-insensitive_SHX", occupancy=(0.12, 0.33, 0.55), **_SYNG,
              description="the same, after SHX: the same truth (the null pair)"),
    GroupSpec("SynG-noFork", occupancy=(0.0, 0.33, 0.67), **_SYNG,
              description="SynG that never reaches a fork (no helicase contact): scanning and free only"),
    # THE REVISION FOR THE MOVIES (26 Oct 2026, asked: "ok so lets build the synthetic dataset", after "we assumed
    # [the old data] was wrong ... the helicase primase binding domain blocked"): the two hypotheses, both DECLARED.
    # The N-terminal expectation -- (p)ppGpp blocks primer EXTENSION (Giramma 2021), so priming at a fork aborts and
    # restarts: the primase is recruited about three times as often, each stay as long (k(S -> B) x 3, fork / scanning
    # balance x 3, scanning / free unchanged). Threefold is also the size of change seen in a lab data set, so this
    # group is a POWER CHECK at that data size, not a truth taken from the data.
    GroupSpec("SynG_SHX-recruit", occupancy=(0.29, 0.27, 0.44), **_SYNG,
              description="after SHX, forks recruit it about three times as often (a second at a fork as before): "
                          "fork 29 %"),
    # The published direction (the C-terminal fusion, Hernandez-Tamayo 2021: 46 -> 12 % static), DRASTIC as asked on
    # 24 Oct ("more like 46 -> 20"): fork 12 -> 4 % and scanning 33 -> 16 %, so DNA binding falls in general; every
    # stay as long. The two slow states together 45 -> 20 %.
    GroupSpec("SynG_SHX-release", occupancy=(0.04, 0.16, 0.80), **_SYNG,
              description="after SHX, it binds DNA and forks much less often (stays as before): fork 4 / scanning 16 %"),
)

#: SYNG V2 (27 Oct 2026, asked: "I think we need to do v2", then "we need a new syng synthetic version that is a bit
#: closer to real data. what i mean that even in 2pop fit we see a strong increase"). v1's chain, stays and forks with
#: SCANNING SLOWER: D 0.1 um^2/s (v1 0.2), a declared round number near the middle state a lab data set
#: reads. Why it matters: a two-population fit of three states counts the middle state
#: with the slow population only when it sits nearer the slow one in log D. At 0.2 (read 0.145 through u-track) it sat
#: nearer the free one, so more scanning lowered the FAST D and barely raised the slow share; at 0.1 the untreated
#: movies read a slow share and D close to a lab data set's.
#: Three SHX hypotheses, each THREEFOLD -- v1's size, no new number -- at one step of the chain or at both:
#: SHX-recruit (scanning -> fork), SHX-associate (free -> scanning: forks gain only through the scanning pool; the
#: hypothesis a three-state reading raised, SHX moving the primase off free into both slower states) and SHX-both (the
#: two together). In a movie pilot (one replicate of 7 movies through u-track) associate alone raised the
#: two-population slow share, and fivefold no further, because the fork share did not move; a lab
#: SHX data set raised it more, with BOTH slower states up in three. SHX-release (the published direction) and the null
#: pair's SHX side as in v1. The names are v1's: a name alone means v1's group, so these are built from the specs.
_SYNG_V2 = dict(_SYNG, D_um2_s=(0.0, 0.1, 1.3))
SYNG_V2_GROUPS = (
    GroupSpec("SynG", occupancy=(0.12, 0.33, 0.55), **_SYNG_V2,
              description="SynG v2 untreated: fork 12 / scanning 33 / free 55 %, scanning at D 0.1"),
    GroupSpec("SynG_SHX-recruit", occupancy=(0.29, 0.27, 0.44), **_SYNG_V2,
              description="after SHX, forks recruit it about three times as often (stays as before): fork 29 %"),
    GroupSpec("SynG_SHX-associate", occupancy=(0.19, 0.52, 0.29), **_SYNG_V2,
              description="after SHX, it starts scanning DNA about three times as often (stays as before): fork 19 / "
                          "scanning 52 %"),
    GroupSpec("SynG_SHX-both", occupancy=(0.41, 0.38, 0.21), **_SYNG_V2,
              description="after SHX, it starts scanning DNA AND is recruited by forks about three times as often "
                          "(stays as before): fork 41 / scanning 38 %"),
    GroupSpec("SynG_SHX-release", occupancy=(0.04, 0.16, 0.80), **_SYNG_V2,
              description="after SHX, it binds DNA and forks much less often (stays as before): fork 4 / scanning 16 %"),
    GroupSpec("SynG-insensitive_SHX", occupancy=(0.12, 0.33, 0.55), **_SYNG_V2,
              description="a SynG that does not respond to SHX, after SHX: the untreated truth (the null)"),
)

#: SYNR AND SYNG IN 3D (asked 30 Oct 2026: "after building free mng 2 and 5. feel free to rebuild syn r and syng etc
#: with 3d projection"): the same declared groups -- chains, D's, stays, occupancies -- moving in the cell's volume
#: (`GroupSpec.dims` 3, `_simulate_synr3d`). The names are the 2D groups': a name alone means the 2D group, so a 3D set
#: is built from these specs. SynR's free controls are the 3D free protein (`SYNR_FREE_3D`, `SYNR_FREE_3D_FAST`).
SYNR_3D_GROUPS = tuple(replace(g, dims=3) for g in SYNR_V1_GROUPS[:4]) + (SYNR_FREE_3D, SYNR_FREE_3D_FAST)
SYNG_V2_3D_GROUPS = tuple(replace(g, dims=3) for g in SYNG_V2_GROUPS)

#: SYNG WITH TWO STATES (4 Nov 2026, asked: "this is high priority yes", to "a SynG with only two true states,
#: fork-bound and free, no scanning"; the same morning's ruling that the primase "is not randomly binding DNA and
#: scanning"). The question it answers: how much MIDDLE state, and how much of its rise under SHX, does the camera
#: and tracker alone make from immobile and free molecules? Last night's SynG truth against its tracked movies says
#: about 10 pp in SQD and 6 pp in the HMM, but that set HAS a middle state, which the artefact adds to.
#: DECLARED: B on a fork (D 0, `fork_capture_um` 0.1, a second there, as SynG) and F free, binding a fork directly
#: (F <-> B; `GroupSpec.two_state`, `rates_for`, `_simulate_synr3d`). The fork shares are SynG v2's own, 12 % untreated
#: and 41 % after SHX (SHX-both), so only the middle state is missing; the stay is unchanged, so SHX raises the
#: recruitment alone. Free D 1.0 um^2/s: free mNG's 2.0 x (27 / 96)^0.56 = 0.98 for the mNG-primase fusion (~96 kDa), with the
#: cytoplasm's MEASURED mass exponent (Bellotto et al. 2022, eLife, doi:10.7554/eLife.82654), not SynG's 1.3 (the
#: Stokes-Einstein 1/3); a declared round number. 3D only.
_SYNG_2STATE = dict(_SYNG, D_um2_s=(0.0, 0.1, 1.0), dims=3)
SYNG_2STATE_3D_GROUPS = (
    GroupSpec("SynG2", occupancy=(0.12, 0.0, 0.88), **_SYNG_2STATE,
              description="two-state SynG untreated: fork 12 / free 88 %, no middle state, a second at a fork"),
    GroupSpec("SynG2_SHX", occupancy=(0.41, 0.0, 0.59), **_SYNG_2STATE,
              description="two-state SynG after SHX: forks recruit it more often (stays as before): fork 41 / free "
                          "59 %, no middle state"),
)

#: SYNK THROUGH A TRACKER (26 Oct 2026, asked: "lets build synK as well. u know we need to show that we can show a
#: drop of D"). An addition at t = 0 and movies recorded at known times after it, as the lab's induction experiments
#: are (the app's "Time after the addition" clock, read from each `.stk`'s own stamps: `build_movie_project`). Three
#: groups on that ONE clock with ONE response (exponential, half-life 2 min after a 20 s delay; each replicate's
#: onset its own, SD 10 s, each cell's, SD 15 s) and the SAME mean D afterwards, 0.65 -> 0.26 um^2/s over the
#: molecules: the free D DROPS 1.0 -> 0.4 with the shares held (it moves slower, it binds no more); the bound share
#: RISES 35 -> 74 % with every D held (it binds more, no D changed); nothing changes (the null on the same clock).
#: A readout of the average D calls the first two the same; the induction model must tell a D from a share.
#: Movies 2 min before the addition (the baseline) and 0.5, 1.5, 2.5, 4, 6, 8 and 10 min after it. DECLARED.
SYNK_MOVIE_STARTS_S = (-120.0, 30.0, 90.0, 150.0, 240.0, 360.0, 480.0, 600.0)
_SYNK = dict(kind="synk", D_um2_s=(0.0, 0.2, 1.0), occupancy=(0.35, 0.0, 0.65), k_bound_off_per_s=1.0,
             k_scan_off_per_s=5.0)
_SYNK_CLOCK = dict(movie_start_s=SYNK_MOVIE_STARTS_S, shape="exponential", time_s=120.0, delay_s=20.0,
                   onset_sd_day_s=10.0, onset_sd_cell_s=15.0)
SYNK_GROUPS = (
    GroupSpec("SynK_D-drop", **_SYNK,
              induction=Induction(D_after_um2_s=(0.0, 0.2, 0.4), occupancy_after=(0.35, 0.0, 0.65), **_SYNK_CLOCK),
              description="after the addition the free D drops 1.0 -> 0.4 um^2/s; bound 35 % throughout"),
    GroupSpec("SynK_binding-up", **_SYNK,
              induction=Induction(D_after_um2_s=(0.0, 0.2, 1.0), occupancy_after=(0.74, 0.0, 0.26), **_SYNK_CLOCK),
              description="after the addition the bound share rises 35 -> 74 %; every D held"),
    GroupSpec("SynK_none", **_SYNK,
              induction=Induction(D_after_um2_s=(0.0, 0.2, 1.0), occupancy_after=(0.35, 0.0, 0.65), **_SYNK_CLOCK),
              description="the addition changes nothing (the null on the same clock)"),
)

#: SYNK AND SYNN IN 3D (31 Oct 2026, asked: "is enough space on F to rerender them with the new fixes and new 3d",
#: "It's all for one paper"): the same declared groups, SynK moving in the cell's volume (`_simulate_synk`) and SynN
#: with the 3D walk of 31 Oct (micro-steps, the instantaneous truth; `_simulate_synn`). The names are the 2D groups':
#: a name alone means the group as first built. SynN's free control (`mNG_free_3D`) is left out: the paper's free
#: mNG is v6 (`SYNR_FREE_3D`, at one user's acquisition).
SYNK_3D_GROUPS = tuple(replace(g, dims=3) for g in SYNK_GROUPS)
SYNN_3D_GROUPS = tuple(replace(g, dims=3) for g in SYNN_GROUPS if g.name != "mNG_free_3D")


def rates_for(spec: GroupSpec, *, nucleoid_share: float, bound_factor: float = 1.0,
              membrane_share: float | None = None, fork_share: float | None = None) -> np.ndarray:
    """The 3 x 3 rate matrix (s^-1, order B, S, F) that holds ``spec.occupancy`` in a chain F <-> S <-> B.

    Detailed balance on the chain: k(S->B) = k(B->S) * pB / pS and k(F->S) = k(S->F) * pS / pF. A free molecule
    can start scanning only INSIDE the nucleoid, so the F -> S rate that applies there is divided by
    ``nucleoid_share`` (the share of the cell's area the nucleoid covers) to keep the effective exchange. The
    result is approximate by design -- a free molecule's time inside is not exactly the area share -- and the
    EMPIRICAL occupancy is the truth. ``bound_factor`` is a replicate day's lognormal scaling of k(S -> B)."""
    if spec.free_only:
        raise SyntheticError(f"{spec.name} is a one-state group: it has no rates")
    if spec.kind == "synn":
        return _synn_rates(spec, bound_factor=bound_factor, membrane_share=membrane_share)
    if spec.kind == "synk":
        return _hub_rates(spec, spec.occupancy, bound_factor=bound_factor)
    if not 0.0 < nucleoid_share <= 1.0:
        raise SyntheticError(f"nucleoid_share must be in (0, 1], got {nucleoid_share}")
    p_b, p_s, p_f = spec.occupancy
    k_bs = float(spec.k_bound_off_per_s)
    if spec.two_state:
        # two-state SynG: F <-> B directly, k(F->B) = k(B->F) * pB / pF by detailed balance, divided by the capture
        # spheres' share of the CELL's volume (a free molecule roams the cell, not the nucleoid): the forks' share
        # of the nucleoid times the nucleoid's of the cell; the EMPIRICAL occupancy is the truth
        if fork_share is None or not 0.0 < fork_share <= 1.0:
            raise SyntheticError(f"{spec.name}: fork_share must be in (0, 1], got {fork_share}")
        rates = np.zeros((3, 3))
        rates[0, 2] = k_bs
        rates[2, 0] = k_bs * p_b / p_f * float(bound_factor) / (fork_share * nucleoid_share)
        return rates
    k_sf = float(spec.k_scan_off_per_s)
    k_sb = k_bs * p_b / p_s * float(bound_factor)
    if spec.fork_capture_um is not None:
        # SynG: a scanning molecule binds only near a fork, so the rate that applies there is divided by the
        # forks' share of the nucleoid, as F -> S is by the nucleoid's share of the cell; the EMPIRICAL
        # occupancy is the truth
        if fork_share is None or not 0.0 < fork_share <= 1.0:
            raise SyntheticError(f"{spec.name}: fork_share must be in (0, 1], got {fork_share}")
        k_sb /= fork_share
    k_fs = k_sf * p_s / p_f / nucleoid_share
    rates = np.zeros((3, 3))
    rates[0, 1] = k_bs
    rates[1, 0] = k_sb
    rates[1, 2] = k_sf
    rates[2, 1] = k_fs
    return rates


def _synn_rates(spec: GroupSpec, *, bound_factor: float, membrane_share: float | None) -> np.ndarray:
    """SynN's 3 x 3 rates (s^-1, order M, R, F) holding ``spec.occupancy`` with F as the hub: k(M->F) and k(R->F)
    are the residences' inverses, k(F->M) = k(M->F) pM / pF and k(F->R) = k(R->F) pR / pF by detailed balance, both
    scaled by ``bound_factor`` (a day's, movie's or cell's substrate). A free molecule binds M only within the
    membrane shell, so k(F->M) is divided by ``membrane_share`` (the shell's share of the cells' volume), as SynR
    divides F -> S by the nucleoid's share; the EMPIRICAL occupancy is the truth."""
    if membrane_share is None or not 0.0 < membrane_share <= 1.0:
        raise SyntheticError(f"{spec.name}: membrane_share must be in (0, 1], got {membrane_share}")
    return _hub_rates(spec, spec.occupancy, bound_factor=bound_factor, first_share=membrane_share)


def _hub_rates(spec: GroupSpec, occupancy, *, bound_factor: float, first_share: float = 1.0) -> np.ndarray:
    """3 x 3 rates (s^-1) holding ``occupancy`` with the third state as the hub (SynN, SynK): k(0->2) and k(1->2)
    are ``spec``'s residences' inverses, k(2->0) and k(2->1) follow by detailed balance, both scaled by
    ``bound_factor``; k(2->0) is divided by ``first_share`` (SynN's membrane shell, 1 elsewhere)."""
    p_m, p_r, p_f = (float(v) for v in occupancy)
    if p_f <= 0.0:
        raise SyntheticError(f"{spec.name}: needs a free occupancy above 0 (F is the hub)")
    k_mf, k_rf = float(spec.k_bound_off_per_s), float(spec.k_scan_off_per_s)
    rates = np.zeros((3, 3))
    rates[0, 2] = k_mf
    rates[1, 2] = k_rf
    rates[2, 0] = k_mf * p_m / p_f * float(bound_factor) / float(first_share)
    rates[2, 1] = k_rf * p_r / p_f * float(bound_factor)
    return rates


def _onset_stream(design, spec):
    """SynK's onsets come from their OWN stream (per group, by name), so they never move a molecule's draws."""
    if spec.induction is None:
        return None
    return np.random.default_rng([design.seed, 13, zlib.crc32(spec.name.encode("utf-8"))])


def _day_onset(spec, onset_rng) -> float:
    """A replicate's onset offset (s), drawn once per day from the group's onset stream."""
    ind = spec.induction
    return float(onset_rng.normal(0.0, ind.onset_sd_day_s)) if ind is not None and ind.onset_sd_day_s else 0.0


def _movie_induction(spec, onset_rng, m: int, day_offset: float, bound_factor: float):
    """One SynK movie, either stage: ``(rates after, the simulator's keywords, the movie's truth)``. The movie's
    start since the induction is ``movie_start_s`` at its place ``m`` in its day (cycled), its onsets the delay plus
    the day's and its own offset."""
    ind = spec.induction
    after = _hub_rates(spec, ind.occupancy_after, bound_factor=bound_factor)
    movie_offset = float(onset_rng.normal(0.0, ind.onset_sd_movie_s)) if ind.onset_sd_movie_s else 0.0
    start = float(ind.movie_start_s[m % len(ind.movie_start_s)])
    base = float(ind.delay_s) + day_offset + movie_offset
    return after, {"movie_start_s": start, "onset_base_s": base, "onset_rng": onset_rng}, {
        "start_s": start, "onset_base_s": base, "day_offset_s": day_offset, "movie_offset_s": movie_offset}


def membrane_share(cells: list, depth_um: float) -> float:
    """The membrane shell's share of ``cells``' total volume."""
    total = sum(c.volume() for c in cells)
    return sum(c.volume() * c.shell_share(depth_um) for c in cells) / total


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SynRDesign:
    """Everything that is the same for every group: the acquisition, the camera's effects, the cells."""

    pixel_size_nm: float = 106.0
    frame_interval_s: float = 0.020
    #: The camera's EXPOSURE within a frame (s), when shorter than the frame interval; None = the whole frame.
    #: Asked 20 Oct 2026 to compare SynR with one user's free mNG, recorded with a 20 ms exposure in a 25.0 ms frame:
    #: the first `exposed_subframes` of a frame's ``n_subframes`` sub-steps are averaged and rendered, the rest are
    #: the camera's dead time (the molecule moves and switches, nothing is recorded). Must be a whole number of
    #: sub-steps (0.8 x 5 = 4); the default changes nothing, draw for draw.
    exposure_s: float | None = None
    n_frames: int = 1000
    field_px: tuple = (128, 128)
    sigma_um: float = 0.025
    n_subframes: int = 5
    bleach_mean_frames: float = 15.0
    min_localizations: int = 5
    days: int = 3
    movies_per_day: int = 10
    cells_per_movie: int = 6
    molecules_per_cell: int = 16
    cell_length_mean_um: float = 3.6
    cell_length_sd_um: float = 0.5
    cell_length_range_um: tuple = (2.6, 5.0)
    cell_width_um: float = 0.9
    day_bound_sd: float = 0.12

    @property
    def exposed_subframes(self) -> int:
        """How many of a frame's sub-steps the camera exposes (`exposure_s`)."""
        if self.exposure_s is None:
            return int(self.n_subframes)
        exact = self.n_subframes * float(self.exposure_s) / float(self.frame_interval_s)
        n = int(round(exact))
        if abs(exact - n) > 1e-6 or not 1 <= n <= self.n_subframes:
            raise SyntheticError(
                f"an exposure of {self.exposure_s * 1000:g} ms in a {self.frame_interval_s * 1000:g} ms frame is "
                f"{exact:g} of its {self.n_subframes} sub-steps; it must be a whole number from 1 to "
                f"{self.n_subframes}")
        return n
    #: BIOLOGICAL NOISE below the day (14 Oct 2026, measured on a lab's project, 7 proteins, glucose and
    #: malate, with the tool's own nested mixed model: `docs/synthetic_synr.md`). Natural-log SDs of a factor on
    #: k(S -> B) -- which moves the bound log-odds by exactly its log -- per MOVIE and per CELL; and the SD of the
    #: log of a cell's molecule count, ``molecules_per_cell`` being the cells' MEAN. 0 = off, so SynR v1 / v2 build
    #: exactly as recorded. Drawn from their OWN stream (per group, by name), never the molecules'.
    movie_bound_sd: float = 0.0
    cell_bound_sd: float = 0.0
    molecules_sd_log: float = 0.0
    #: A DECLARED range of molecules per cell, each cell drawn evenly from it (ruled 14 Oct 2026: "Do 24 to 40,
    #: not a single value"); when set it replaces ``molecules_per_cell`` and ``molecules_sd_log``.
    molecules_range: tuple | None = None
    seed: int = 20261013

    @property
    def field_um(self) -> tuple:
        return tuple(n * self.pixel_size_nm / 1000.0 for n in self.field_px)


def _start(rng, cell: CapsuleCell, state: int) -> tuple:
    """A start position honouring the state's compartment: B and S uniform in the nucleoid, F in the cell."""
    if state in (0, 1):
        areas = np.array([lobe.area for lobe in cell.lobes])
        lobe = cell.lobes[int(rng.choice(len(cell.lobes), p=areas / areas.sum()))]
        r, a = math.sqrt(rng.uniform()), rng.uniform(0.0, 2.0 * math.pi)
        return lobe.centre_um + r * lobe.half_length_um * math.cos(a), r * lobe.half_width_um * math.sin(a)
    while True:
        u = rng.uniform(-cell.length_um / 2.0, cell.length_um / 2.0)
        v = rng.uniform(-cell.radius, cell.radius)
        if cell.inside(u, v):
            return float(u), float(v)


#: SynG: each fork sits this far either side of its lobe's centre along the axis (um).
FORK_HALF_SPACING_UM = 0.08
#: SynG: a fork-bound molecule's spread about its fork (um): the replisome's size.
FORK_SPREAD_UM = 0.02


def _on_fork(rng, fork) -> tuple:
    """A position on ``fork``, within the replisome's size."""
    d = rng.normal(0.0, FORK_SPREAD_UM, 2)
    return float(fork[0] + d[0]), float(fork[1] + d[1])


#: SynG's truth: what it is and what its states mean.
SYNG_NOTE = ("SynG is synthetic and inspired by the primase; no real protein is implied. "
             "Forks: two per nucleoid lobe at its centre +- 0.08 um along the axis; a fork-bound molecule sits within "
             "0.02 um of one and binds only within the group's capture distance")
SYNG_STATES = {"B": "bound at a replication fork (anchored on a replisome)", "S": "scanning DNA (in the nucleoid)",
               "F": "free (whole cell)"}


def _fork_share_of(spec: "GroupSpec", cells: list):
    """`fork_share` for a SynG group, None for the rest."""
    return None if spec.fork_capture_um is None else fork_share(cells, spec.fork_capture_um, dims=spec.dims)


def fork_share(cells: list, capture_um: float, *, dims: int = 2) -> float:
    """The share of the cells' nucleoid area within ``capture_um`` of a fork (`CapsuleCell.fork_sites`); with
    ``dims`` 3, of their nucleoid VOLUME within that distance (a sphere about each fork, on the axis)."""
    forks = sum(len(c.fork_sites()) for c in cells)
    if dims == 3:
        nucleoid = sum(c.nucleoid_volume() for c in cells)
        return float(min(1.0, forks * 4.0 / 3.0 * math.pi * capture_um ** 3 / max(nucleoid, 1e-12)))
    nucleoid = sum(c.nucleoid_area() for c in cells)
    return float(min(1.0, forks * math.pi * capture_um ** 2 / max(nucleoid, 1e-12)))


def nucleoid_share(cells: list, *, dims: int = 2) -> float:
    """The nucleoid's share of ``cells``: of their area, or with ``dims`` 3 of their volume (`rates_for`)."""
    if dims == 3:
        return sum(c.nucleoid_volume() for c in cells) / sum(c.volume() for c in cells)
    return sum(c.nucleoid_area() for c in cells) / sum(c.area() for c in cells)


def _cell_nucleoid_share(cell: CapsuleCell, dims: int) -> float:
    """One cell's `nucleoid_share`, for the truth's cells table."""
    return cell.nucleoid_volume() / cell.volume() if dims == 3 else cell.nucleoid_area() / cell.area()


def simulate_movie(rng, cells: list, *, design: SynRDesign, spec: GroupSpec, rates,
                   keep_short: bool = False, rates_after=None, cell_factors: dict | None = None,
                   cell_counts: dict | None = None, movie_start_s: float | None = None,
                   onset_base_s: float | None = None, onset_rng=None, stuck: float = 0.0) -> list:
    """Every track of one movie: ``[{cell, track, frames, x_um, y_um, states, in_nucleoid}]`` in FIELD
    coordinates, at most ``design.molecules_per_cell`` molecules per cell, those under the minimum length dropped
    -- unless ``keep_short`` (a rendered movie holds every molecule; the TRACKER's minimum then drops them).

    ``rates`` is `rates_for`'s matrix (None for a one-state group). Per sub-step: the state moves by the exact
    transition matrix -- except that F -> S is refused outside the nucleoid -- then the molecule moves in its
    state's compartment (B stays where it bound). A frame reports the mean of its sub-step positions plus
    localisation error; its STATE is the one held at the frame's end (`true_state`).

    ``cell_factors`` ({cell id: factor}) scales each cell's k(S -> B), its molecules starting in that cell's own
    occupancy; ``cell_counts`` ({cell id: n}) sets how many molecules each cell holds (`SynRDesign`'s biological
    noise).

    A SWITCH (``spec.switch_frame``, ``rates_after`` = the after-group's `rates_for`) uses each frame's own matrix,
    ``after + (before - after) * spec.switch_weight(frame)``, and starts a molecule in the occupancy interpolated
    the same way."""
    from .simulate import ConditionSpec, StateSpec

    if spec.kind == "synn":
        if spec.switch_frame is not None or rates_after is not None:
            raise SyntheticError(f"{spec.name}: SynN has no switch")
        return _simulate_synn(rng, cells, design=design, spec=spec, rates=rates, keep_short=keep_short,
                              cell_factors=cell_factors, cell_counts=cell_counts)
    if spec.kind == "synk":
        if movie_start_s is None or onset_base_s is None or onset_rng is None or rates_after is None:
            raise SyntheticError(f"{spec.name}: a SynK movie needs its start time, its onset base, the onset "
                                 f"stream and the after-rates")
        return _simulate_synk(rng, cells, design=design, spec=spec, rates=rates, rates_after=rates_after,
                              keep_short=keep_short, cell_factors=cell_factors, cell_counts=cell_counts,
                              movie_start_s=float(movie_start_s), onset_base_s=float(onset_base_s),
                              onset_rng=onset_rng)
    if spec.dims == 3:
        if spec.free_only:
            return _simulate_free3d(rng, cells, design=design, spec=spec, keep_short=keep_short,
                                    cell_counts=cell_counts, stuck=stuck)
        if rates_after is not None:
            raise SyntheticError(f"{spec.name}: a 3D group has no switch")
        return _simulate_synr3d(rng, cells, design=design, spec=spec, rates=rates, keep_short=keep_short,
                                cell_factors=cell_factors, cell_counts=cell_counts)
    dt = design.frame_interval_s / design.n_subframes
    n_exposed = design.exposed_subframes
    if spec.free_only:
        A = np.ones((1, 1))
        D = (float(spec.D_um2_s[0]),)
        start_p = np.array([1.0])
        offset = 2                                   # its one state is F
    else:
        condition = ConditionSpec(name=spec.name, states=tuple(StateSpec(d) for d in spec.D_um2_s),
                                  rates_per_s=rates)
        A = condition.transition_matrix(dt)
        D = tuple(float(d) for d in spec.D_um2_s)
        start_p = np.asarray(spec.occupancy, float) / sum(spec.occupancy)
        offset = 0
    switching = spec.switch_frame is not None
    if switching:
        if rates_after is None or spec.after is None:
            raise SyntheticError(f"{spec.name} switches at frame {spec.switch_frame} but has no after-rates")
        before, after = np.asarray(rates, float), np.asarray(rates_after, float)
        occ_before = np.asarray(spec.occupancy, float) / sum(spec.occupancy)
        occ_after = np.asarray(spec.after.occupancy, float) / sum(spec.after.occupancy)
        weights = spec.switch_weight(np.arange(1, design.n_frames + 1), design.frame_interval_s)
        matrices: dict = {}

        def matrix_at(frame: int):
            w = round(float(weights[frame - 1]), 4)
            if w not in matrices:
                matrices[w] = ConditionSpec(
                    name=spec.name, states=tuple(StateSpec(d) for d in spec.D_um2_s),
                    rates_per_s=after + (before - after) * w).transition_matrix(dt)
            return matrices[w]
    survive = math.exp(-1.0 / design.bleach_mean_frames)
    per_cell = spec.molecules_per_cell or design.molecules_per_cell
    base = (A, start_p, (occ_before, occ_after) if switching else None,
            matrix_at if switching else None)

    def chain_for(cell):
        """The cell's own chain: its k(S -> B) scaled by its factor, its molecules starting in its occupancy."""
        f = 1.0 if (spec.free_only or not cell_factors) else float(cell_factors.get(cell.cell_id, 1.0))
        if f == 1.0:
            return base
        weight = np.array([f, 1.0, 1.0])

        def scaled(r):
            r = np.array(r, float)
            r[1, 0] *= f
            return r

        cond = ConditionSpec(name=spec.name, states=tuple(StateSpec(d) for d in spec.D_um2_s),
                             rates_per_s=scaled(rates))
        start = np.asarray(spec.occupancy, float) * weight
        if not switching:
            return cond.transition_matrix(dt), start / start.sum(), None, None
        b, a = scaled(before), scaled(after)
        ob = occ_before * weight
        oa = occ_after * weight
        cache: dict = {}

        def m_at(frame: int):
            w = round(float(weights[frame - 1]), 4)
            if w not in cache:
                cache[w] = ConditionSpec(name=spec.name, states=tuple(StateSpec(d) for d in spec.D_um2_s),
                                         rates_per_s=a + (b - a) * w).transition_matrix(dt)
            return cache[w]
        return m_at(1), start / start.sum(), (ob / ob.sum(), oa / oa.sum()), m_at

    out = []
    for cell in cells:
        A, start_p, occs, cell_matrix_at = chain_for(cell)
        forks = cell.fork_sites() if spec.fork_capture_um is not None else None
        if switching:
            occ_before, occ_after = occs
            matrix_at = cell_matrix_at
        n_here = int(cell_counts.get(cell.cell_id, per_cell)) if cell_counts else per_cell
        for molecule in range(n_here):
            length = 1
            while rng.uniform() < survive and length < design.n_frames:
                length += 1
            if length < design.min_localizations and not keep_short:
                continue
            first = int(rng.integers(1, design.n_frames - length + 2))
            if switching:
                w = float(weights[first - 1])
                start_p = occ_after + (occ_before - occ_after) * w
            state = int(rng.choice(len(start_p), p=start_p))
            # A STUCK molecule of a free protein (`GroupSpec.stuck_range`): state B, started anywhere in the cell
            # (F's start), never moving. Drawn only when the group has a stuck share, so no other group's draws move.
            here = 0 if (stuck and rng.uniform() < stuck) else offset
            if forks is not None and state + here == 0:
                u, v = _on_fork(rng, forks[int(rng.integers(len(forks)))])
            else:
                u, v = _start(rng, cell, 2 if (here == 0 and spec.free_only) else state + here)
            xs, ys, states, inside = [], [], [], []
            true_x, true_y, sub_x, sub_y = [], [], [], []
            for k_frame in range(length):
                if switching:
                    A = matrix_at(first + k_frame)
                pos = []
                for _ in range(design.n_subframes):
                    nxt = int(rng.choice(len(A), p=A[state])) if len(A) > 1 else 0
                    if not spec.free_only and state == 2 and nxt == 1 and not bool(cell.in_nucleoid(u, v)):
                        nxt = 2                      # scanning starts only on DNA
                    if forks is not None and state != 0 and nxt == 0:
                        # SynG: binding is AT A FORK -- refused away from one, and the molecule then sits on it
                        near = min(forks, key=lambda f: (f[0] - u) ** 2 + (f[1] - v) ** 2)
                        if (near[0] - u) ** 2 + (near[1] - v) ** 2 > spec.fork_capture_um ** 2:
                            nxt = state
                        else:
                            u, v = _on_fork(rng, near)
                    state = nxt
                    s = state + here
                    if s == 1:
                        lobe = cell.lobe_of(u, v)
                        step = rng.normal(0.0, math.sqrt(2.0 * D[state] * dt), 2)
                        u, v = cell.reflect_in_lobe(u + step[0], v + step[1], lobe)
                    elif s == 2:
                        step = rng.normal(0.0, math.sqrt(2.0 * D[state] * dt), 2)
                        u, v = cell.reflect_in_cell(u + step[0], v + step[1])
                    pos.append((u, v))
                exposed = pos[:n_exposed]                # the camera's exposure; the rest is dead time
                mu, mv = np.mean(exposed, axis=0)
                x, y = cell.to_field(mu, mv)
                noise = rng.normal(0.0, design.sigma_um, 2)
                xs.append(float(x + noise[0]))
                ys.append(float(y + noise[1]))
                # The TRUE blurred centroid and the sub-step positions, for stage 2: a movie renders the
                # sub-steps and lets its photons set the localisation error (no RNG drawn here, so stage 1
                # is unchanged under the same seed).
                true_x.append(float(x))
                true_y.append(float(y))
                fx, fy = cell.to_field(np.array([p[0] for p in exposed]), np.array([p[1] for p in exposed]))
                sub_x.append([float(a) for a in np.atleast_1d(fx)])
                sub_y.append([float(a) for a in np.atleast_1d(fy)])
                states.append(STATES[state + here])
                # where the molecule IS at the frame's end, the moment its state is read: the blurred
                # average of a frame that ended in a switch can lie on the other side of the edge
                inside.append(bool(cell.in_nucleoid(u, v)))
            out.append({"cell": cell.cell_id, "track": len(out) + 1,
                        "frames": list(range(first, first + length)),
                        "x_um": xs, "y_um": ys, "states": states, "in_nucleoid": inside,
                        "x_true_um": true_x, "y_true_um": true_y, "sub_x_um": sub_x, "sub_y_um": sub_y})
    return out


def _start3(rng, cell: CapsuleCell, state: int) -> tuple:
    """A SynN start: uniform in the cell's volume, or (M) on the membrane radially outward of such a point."""
    r = cell.radius
    while True:
        u = rng.uniform(-cell.length_um / 2.0, cell.length_um / 2.0)
        v, z = rng.uniform(-r, r), rng.uniform(-r, r)
        if cell.axis_distance3(u, v, z)[0] <= r:
            break
    if state == 0:
        return cell.to_surface3(u, v, z)
    return float(u), float(v), float(z)


def _simulate_free3d(rng, cells: list, *, design: SynRDesign, spec: GroupSpec, keep_short: bool,
                     cell_counts: dict | None, stuck: float = 0.0) -> list:
    """A free protein in 3D (`GroupSpec.dims` 3), in `simulate_movie`'s shape plus ``sub_z_um`` (each exposed
    sub-step's height, which sets its focus) and ``extra``: ``x_instant_um`` / ``y_instant_um`` / ``z_instant_um``,
    the molecule's own position at the middle exposed sub-step -- where it IS, as the truth reports it -- beside
    ``x_true_um`` / ``y_true_um``, the blurred centroid the camera records.

    Each molecule starts uniform in the capsule's volume and diffuses in it at its D, reflected at the membrane in
    micro-steps (`_walk_in_cell3`; the v5 set, built 30 Oct 2026, moved one step per sub-step), so its projection
    crowds onto the centre line as a rod's does. A stuck molecule (``stuck``, the group's `stuck_range`) is state B
    and does not move; drawn only when there is a share, so a group without one draws nothing for it."""
    dt = design.frame_interval_s / design.n_subframes
    n_exposed = design.exposed_subframes
    mid = (n_exposed - 1) // 2
    D = float(spec.D_um2_s[0])
    survive = math.exp(-1.0 / design.bleach_mean_frames)
    per_cell = spec.molecules_per_cell or design.molecules_per_cell
    out = []
    for cell in cells:
        n_here = int(cell_counts.get(cell.cell_id, per_cell)) if cell_counts else per_cell
        for _molecule in range(n_here):
            length = 1
            while rng.uniform() < survive and length < design.n_frames:
                length += 1
            if length < design.min_localizations and not keep_short:
                continue
            first = int(rng.integers(1, design.n_frames - length + 2))
            is_stuck = bool(stuck) and rng.uniform() < stuck
            d = 0.0 if is_stuck else D
            u, v, z = _start3(rng, cell, 2)
            xs, ys, inside = [], [], []
            true_x, true_y, sub_x, sub_y, sub_z = [], [], [], [], []
            inst_x, inst_y, inst_z = [], [], []
            for _k in range(length):
                pos = []
                for _ in range(design.n_subframes):
                    if d > 0.0:
                        u, v, z = _walk_in_cell3(rng, cell, u, v, z, d, dt)
                    pos.append((u, v, z))
                exposed = pos[:n_exposed]                    # the camera's exposure; the rest is dead time
                mu = float(np.mean([p[0] for p in exposed]))
                mv = float(np.mean([p[1] for p in exposed]))
                x, y = cell.to_field(mu, mv)
                noise = rng.normal(0.0, design.sigma_um, 2)
                xs.append(float(x + noise[0]))
                ys.append(float(y + noise[1]))
                true_x.append(float(x))
                true_y.append(float(y))
                fx, fy = cell.to_field(np.array([p[0] for p in exposed]), np.array([p[1] for p in exposed]))
                sub_x.append([float(a) for a in np.atleast_1d(fx)])
                sub_y.append([float(a) for a in np.atleast_1d(fy)])
                sub_z.append([float(p[2]) for p in exposed])
                iu, iv, iz = exposed[mid]
                ix, iy = cell.to_field(iu, iv)
                inst_x.append(float(ix))
                inst_y.append(float(iy))
                inst_z.append(float(iz))
                inside.append(bool(cell.in_nucleoid(u, v)))
            out.append({"cell": cell.cell_id, "track": len(out) + 1,
                        "frames": list(range(first, first + length)),
                        "x_um": xs, "y_um": ys, "states": ["B" if is_stuck else "F"] * length,
                        "in_nucleoid": inside,
                        "x_true_um": true_x, "y_true_um": true_y, "sub_x_um": sub_x, "sub_y_um": sub_y,
                        "sub_z_um": sub_z,
                        "extra": {"x_instant_um": inst_x, "y_instant_um": inst_y, "z_instant_um": inst_z}})
    return out


#: A 3D GROUP MOVES IN MICRO-STEPS of at most this RMS per axis (um) between the sub-steps the camera records (31 Oct
#: 2026). A reflection at a curved wall is exact only for steps small against its curvature: with one step per
#: sub-step a free molecule at D 2 (0.14 um per axis in 5 ms, in a cell of 0.5 um radius) sat in the inner quarter of
#: the cross-section 26.4 % of the time against a uniform 25.0 %, and in its outer tenth 17.1 % against 19.0 % (D 5:
#: 29.1 and 15.9 %); at 16 micro-steps 25.2 / 18.6 % and 25.0 / 18.6 %. The v5 free mNG (built 30 Oct 2026) moved
#: one step per sub-step.
MICRO_STEP_UM = 0.02


def _micro_steps(rng, D: float, dt: float) -> np.ndarray:
    """A sub-step's Brownian increments at ``D`` over ``dt``, as ``m`` micro-steps of at most `MICRO_STEP_UM`."""
    sigma = math.sqrt(2.0 * D * dt)
    m = max(1, math.ceil((sigma / MICRO_STEP_UM) ** 2))
    return rng.normal(0.0, sigma / math.sqrt(m), (m, 3))


def _walk_in_cell3(rng, cell: CapsuleCell, u: float, v: float, z: float, D: float, dt: float) -> tuple:
    """One sub-step of free diffusion in the capsule: micro-steps (`MICRO_STEP_UM`), each reflected at the membrane
    (`CapsuleCell.reflect_in_cell3`); a path that never reaches it is taken whole."""
    steps = _micro_steps(rng, D, dt)
    path = np.cumsum(steps, axis=0) + (u, v, z)
    q = np.clip(path[:, 0], -cell.half_axis, cell.half_axis)
    if ((path[:, 0] - q) ** 2 + path[:, 1] ** 2 + path[:, 2] ** 2 <= cell.radius ** 2).all():
        return float(path[-1, 0]), float(path[-1, 1]), float(path[-1, 2])
    for du, dv, dz in steps:
        u, v, z = cell.reflect_in_cell3(u + du, v + dv, z + dz)
    return float(u), float(v), float(z)


def _walk_in_lobe3(rng, lobe: Lobe, u: float, v: float, z: float, D: float, dt: float) -> tuple:
    """One sub-step of scanning in a nucleoid lobe: micro-steps (`MICRO_STEP_UM`), a micro-step that would leave the
    lobe REFUSED (the molecule stays). Exact for the uniform density the occupancy rests on: a reflection radial in the
    lobe's normalised frame (`CapsuleCell.reflect_in_lobe3`, 2D's rule) is oblique to an elongated lobe's surface, and
    a scanning molecule then sat in the lobe's inner half (normalised) 19.7 % of the time against a uniform 12.5 %,
    in the forks' capture spheres twice their volume share, whatever the step."""
    steps = _micro_steps(rng, D, dt)
    path = np.cumsum(steps, axis=0) + (u, v, z)
    if (lobe.rho3(path[:, 0], path[:, 1], path[:, 2]) <= 1.0).all():
        return float(path[-1, 0]), float(path[-1, 1]), float(path[-1, 2])
    for du, dv, dz in steps:
        if _rho3(lobe, u + du, v + dv, z + dz) <= 1.0:
            u, v, z = u + du, v + dv, z + dz
    return float(u), float(v), float(z)


def _start_synr3(rng, cell: CapsuleCell, state: int) -> tuple:
    """A 3D SynR start: B and S uniform in the nucleoid's volume (a lobe chosen by its volume), F in the cell's."""
    if state == 2:
        return _start3(rng, cell, 2)
    volumes = np.array([lobe.volume for lobe in cell.lobes])
    lobe = cell.lobes[int(rng.choice(len(cell.lobes), p=volumes / volumes.sum()))]
    while True:
        p = rng.uniform(-1.0, 1.0, 3)
        if p @ p <= 1.0:
            break
    return (float(lobe.centre_um + p[0] * lobe.half_length_um), float(p[1] * lobe.half_width_um),
            float(p[2] * lobe.half_width_um))


def _on_fork3(rng, fork) -> tuple:
    """A position on ``fork`` in 3D, within the replisome's size (`FORK_SPREAD_UM`)."""
    d = rng.normal(0.0, FORK_SPREAD_UM, 3)
    return float(fork[0] + d[0]), float(fork[1] + d[1]), float(fork[2] + d[2])


def _in_capture3(rng, fork, capture_um: float) -> tuple:
    """A position uniform within ``capture_um`` of ``fork`` (where a released molecule starts scanning)."""
    while True:
        p = rng.uniform(-1.0, 1.0, 3)
        if p @ p <= 1.0:
            return (float(fork[0] + capture_um * p[0]), float(fork[1] + capture_um * p[1]),
                    float(fork[2] + capture_um * p[2]))


def _simulate_synr3d(rng, cells: list, *, design: SynRDesign, spec: GroupSpec, rates, keep_short: bool,
                     cell_factors: dict | None, cell_counts: dict | None) -> list:
    """SynR and SynG in 3D (`GroupSpec.dims` 3), in `_simulate_free3d`'s shape (``sub_z_um``, and the instantaneous
    position in ``extra``). `simulate_movie`'s chain F <-> S <-> B and its rules, in the cell's volume: F diffuses in
    the capsule (`_walk_in_cell3`), S in its nucleoid lobe as an ellipsoid about the axis, refused a step out of it
    (`_walk_in_lobe3`), B stays where it bound -- SynR's operators anywhere in the nucleoid, SynG's forks on the
    axis (`fork_sites3`, entered only within ``spec.fork_capture_um`` of one, in 3D, and then sitting on it; released
    uniformly within that distance of it, binding's mirror, so the declared occupancy holds).

    WHERE A WAY ONWARD IS SHUT, THE SUB-STEP'S MATRIX SHUTS IT: a free molecule outside the nucleoid moves by the
    matrix without F -> S, a scanning SynG molecule away from every fork by the one without S -> B. 2D uses the full
    matrix everywhere and refuses the move, which leaves a molecule the virtual time the full matrix spends in the
    refused state: with the 3D shares (the nucleoid 26-29 % of the volume, a fork's capture sphere 1.4-2.5 % of the
    nucleoid) the rates inside are steep enough that SHX-associate read 23 % bound against its declared 19 %. A cell's
    factor scales its k(S -> B), its molecules starting in its own occupancy, as in 2D.

    THE TWO-STATE SynG (`GroupSpec.two_state`): no S; a free molecule binds a fork directly, by the matrix without
    F -> B wherever it is not within the capture distance of one, and a cell's factor scales its k(F -> B)."""
    from .simulate import ConditionSpec, StateSpec

    dt = design.frame_interval_s / design.n_subframes
    n_exposed = design.exposed_subframes
    mid = (n_exposed - 1) // 2
    D = tuple(float(d) for d in spec.D_um2_s)
    survive = math.exp(-1.0 / design.bleach_mean_frames)
    per_cell = spec.molecules_per_cell or design.molecules_per_cell
    capture2 = None if spec.fork_capture_um is None else float(spec.fork_capture_um) ** 2

    def chain_for(cell):
        """The cell's sub-step matrices -- everywhere's, a free molecule's outside the nucleoid, a scanning one's
        away from the forks (SynG; None for SynR) -- and its start occupancy."""
        f = 1.0 if not cell_factors else float(cell_factors.get(cell.cell_id, 1.0))
        r = np.array(rates, float)
        r[1, 0] *= f
        r[2, 0] *= f                                 # the two-state SynG's binding (0 in the chain)

        def matrix(shut=None):
            rr = r.copy()
            if shut is not None:
                rr[shut] = 0.0
            return ConditionSpec(name=spec.name, states=tuple(StateSpec(d) for d in D),
                                 rates_per_s=rr).transition_matrix(dt)
        start = np.asarray(spec.occupancy, float) * np.array([f, 1.0, 1.0])
        off_fork = None if capture2 is None else matrix((2, 0) if spec.two_state else (1, 0))
        return matrix(), matrix((2, 1)), off_fork, start / start.sum()

    out = []
    for cell in cells:
        A, A_off_dna, A_off_fork, start_p = chain_for(cell)
        forks = cell.fork_sites3() if capture2 is not None else None
        n_here = int(cell_counts.get(cell.cell_id, per_cell)) if cell_counts else per_cell
        for _molecule in range(n_here):
            length = 1
            while rng.uniform() < survive and length < design.n_frames:
                length += 1
            if length < design.min_localizations and not keep_short:
                continue
            first = int(rng.integers(1, design.n_frames - length + 2))
            state = int(rng.choice(3, p=start_p))
            if forks is not None and state == 0:
                u, v, z = _on_fork3(rng, forks[int(rng.integers(len(forks)))])
            else:
                u, v, z = _start_synr3(rng, cell, state)
            xs, ys, states, inside = [], [], [], []
            true_x, true_y, sub_x, sub_y, sub_z = [], [], [], [], []
            inst_x, inst_y, inst_z = [], [], []
            for _k in range(length):
                pos = []
                for _ in range(design.n_subframes):
                    if state == 2 and spec.two_state:        # onto a fork only within its capture distance
                        row = (A if min((q[0] - u) ** 2 + (q[1] - v) ** 2 + (q[2] - z) ** 2 for q in forks)
                               <= capture2 else A_off_fork)[2]
                    elif state == 2:                         # onto DNA only inside the nucleoid
                        row = (A if cell.in_nucleoid3(u, v, z) else A_off_dna)[2]
                    elif state == 1 and forks is not None:   # onto a fork only within its capture distance
                        row = (A if min((q[0] - u) ** 2 + (q[1] - v) ** 2 + (q[2] - z) ** 2 for q in forks)
                               <= capture2 else A_off_fork)[1]
                    else:
                        row = A[state]
                    nxt = int(rng.choice(3, p=row))
                    if forks is not None and state != 0 and nxt == 0:
                        near = min(forks, key=lambda q: (q[0] - u) ** 2 + (q[1] - v) ** 2 + (q[2] - z) ** 2)
                        if (near[0] - u) ** 2 + (near[1] - v) ** 2 + (near[2] - z) ** 2 > capture2:
                            nxt = state
                        else:
                            u, v, z = _on_fork3(rng, near)
                    elif forks is not None and state == 0 and nxt != 0:
                        # RELEASED, as binding's mirror: uniform within the capture distance of its fork. Released
                        # ON the fork (2D's rule), a molecule inside a capture sphere of 1.4-2.5 % of the nucleoid
                        # rebinds before it can leave and SHX-both read 48 % bound against its declared 41 %
                        near = min(forks, key=lambda q: (q[0] - u) ** 2 + (q[1] - v) ** 2 + (q[2] - z) ** 2)
                        u, v, z = _in_capture3(rng, near, float(spec.fork_capture_um))
                    state = nxt
                    if state == 1:
                        u, v, z = _walk_in_lobe3(rng, cell.lobe_of3(u, v, z), u, v, z, D[1], dt)
                    elif state == 2:
                        u, v, z = _walk_in_cell3(rng, cell, u, v, z, D[2], dt)
                    pos.append((u, v, z))
                exposed = pos[:n_exposed]                    # the camera's exposure; the rest is dead time
                mu = float(np.mean([p[0] for p in exposed]))
                mv = float(np.mean([p[1] for p in exposed]))
                x, y = cell.to_field(mu, mv)
                noise = rng.normal(0.0, design.sigma_um, 2)
                xs.append(float(x + noise[0]))
                ys.append(float(y + noise[1]))
                true_x.append(float(x))
                true_y.append(float(y))
                fx, fy = cell.to_field(np.array([p[0] for p in exposed]), np.array([p[1] for p in exposed]))
                sub_x.append([float(a) for a in np.atleast_1d(fx)])
                sub_y.append([float(a) for a in np.atleast_1d(fy)])
                sub_z.append([float(p[2]) for p in exposed])
                iu, iv, iz = exposed[mid]
                ix, iy = cell.to_field(iu, iv)
                inst_x.append(float(ix))
                inst_y.append(float(iy))
                inst_z.append(float(iz))
                states.append(STATES[state])
                # where it IS at the frame's end, the moment its state is read (as 2D)
                inside.append(cell.in_nucleoid3(u, v, z))
            out.append({"cell": cell.cell_id, "track": len(out) + 1,
                        "frames": list(range(first, first + length)),
                        "x_um": xs, "y_um": ys, "states": states, "in_nucleoid": inside,
                        "x_true_um": true_x, "y_true_um": true_y, "sub_x_um": sub_x, "sub_y_um": sub_y,
                        "sub_z_um": sub_z,
                        "extra": {"x_instant_um": inst_x, "y_instant_um": inst_y, "z_instant_um": inst_z}})
    return out


def _simulate_synn(rng, cells: list, *, design: SynRDesign, spec: GroupSpec, rates, keep_short: bool,
                   cell_factors: dict | None, cell_counts: dict | None) -> list:
    """SynN's tracks, in `simulate_movie`'s shape plus ``sub_z_um`` (each exposed sub-step's height). In 3D: R and F
    diffuse in the capsule (reflected at the membrane), M sits on the membrane where it bound; F -> M is refused
    unless the molecule is within ``spec.membrane_shell_um`` of the membrane, and then it moves onto it (at most
    that far). A cell's factor scales both binding rates, its molecules starting in its own occupancy.

    ``spec.dims`` 3 (31 Oct 2026) walks in micro-steps (`_walk_in_cell3`) and records the instantaneous position in
    ``extra``, as every 3D group; 2 keeps the walk the SynN sets were built with (21-22 Oct 2026), one reflection
    per sub-step, which leans a molecule towards the axis ("A flaw in v5's walk", docs/synthetic_synr.md), and draws
    as it drew."""
    from .simulate import ConditionSpec, StateSpec

    dt = design.frame_interval_s / design.n_subframes
    n_exposed = design.exposed_subframes
    mid = (n_exposed - 1) // 2
    three = spec.dims == 3
    D = tuple(float(d) for d in spec.D_um2_s)
    shell = float(spec.membrane_shell_um)
    survive = math.exp(-1.0 / design.bleach_mean_frames)
    per_cell = spec.molecules_per_cell or design.molecules_per_cell

    def chain_for(cell):
        f = 1.0 if not cell_factors else float(cell_factors.get(cell.cell_id, 1.0))
        r = np.array(rates, float)
        r[2, 0] *= f
        r[2, 1] *= f
        A = ConditionSpec(name=spec.name, states=tuple(StateSpec(d) for d in D), rates_per_s=r).transition_matrix(dt)
        start = np.asarray(spec.occupancy, float) * np.array([f, f, 1.0])
        return A, start / start.sum()

    out = []
    for cell in cells:
        A, start_p = chain_for(cell)
        n_here = int(cell_counts.get(cell.cell_id, per_cell)) if cell_counts else per_cell
        for _molecule in range(n_here):
            length = 1
            while rng.uniform() < survive and length < design.n_frames:
                length += 1
            if length < design.min_localizations and not keep_short:
                continue
            first = int(rng.integers(1, design.n_frames - length + 2))
            state = int(rng.choice(3, p=start_p))
            u, v, z = _start3(rng, cell, state)
            xs, ys, states, inside = [], [], [], []
            true_x, true_y, sub_x, sub_y, sub_z = [], [], [], [], []
            inst_x, inst_y, inst_z = [], [], []
            for _k in range(length):
                pos = []
                for _ in range(design.n_subframes):
                    nxt = int(rng.choice(3, p=A[state]))
                    # ANY WAY INTO M is binding at the membrane: F -> M, and R -> M through F within one sub-step
                    # (the sub-step's exact matrix holds that path); refused away from the membrane
                    if state != 0 and nxt == 0:
                        if cell.axis_distance3(u, v, z)[0] < cell.radius - shell:
                            nxt = state
                        else:
                            u, v, z = cell.to_surface3(u, v, z)
                    state = nxt
                    if D[state] > 0.0:
                        if three:
                            u, v, z = _walk_in_cell3(rng, cell, u, v, z, D[state], dt)
                        else:
                            step = rng.normal(0.0, math.sqrt(2.0 * D[state] * dt), 3)
                            u, v, z = cell.reflect_in_cell3(u + step[0], v + step[1], z + step[2])
                    pos.append((u, v, z))
                exposed = pos[:n_exposed]
                mu = float(np.mean([p[0] for p in exposed]))
                mv = float(np.mean([p[1] for p in exposed]))
                x, y = cell.to_field(mu, mv)
                noise = rng.normal(0.0, design.sigma_um, 2)
                xs.append(float(x + noise[0]))
                ys.append(float(y + noise[1]))
                true_x.append(float(x))
                true_y.append(float(y))
                fx, fy = cell.to_field(np.array([p[0] for p in exposed]), np.array([p[1] for p in exposed]))
                sub_x.append([float(a) for a in np.atleast_1d(fx)])
                sub_y.append([float(a) for a in np.atleast_1d(fy)])
                sub_z.append([float(p[2]) for p in exposed])
                if three:
                    iu, iv, iz = exposed[mid]
                    ix, iy = cell.to_field(iu, iv)
                    inst_x.append(float(ix))
                    inst_y.append(float(iy))
                    inst_z.append(float(iz))
                states.append(STATES[state])
                inside.append(bool(cell.in_nucleoid(u, v)))
            out.append({"cell": cell.cell_id, "track": len(out) + 1,
                        "frames": list(range(first, first + length)),
                        "x_um": xs, "y_um": ys, "states": states, "in_nucleoid": inside,
                        "x_true_um": true_x, "y_true_um": true_y, "sub_x_um": sub_x, "sub_y_um": sub_y,
                        "sub_z_um": sub_z,
                        **({"extra": {"x_instant_um": inst_x, "y_instant_um": inst_y, "z_instant_um": inst_z}}
                           if three else {})})
    return out


def _simulate_synk(rng, cells: list, *, design: SynRDesign, spec: GroupSpec, rates, rates_after, keep_short: bool,
                   cell_factors: dict | None, cell_counts: dict | None, movie_start_s: float, onset_base_s: float,
                   onset_rng) -> list:
    """SynK's tracks, in `simulate_movie`'s shape plus ``extra``, per localisation: ``t_s`` (its frame's start,
    seconds since the induction), ``onset_s`` (its molecule's onset), ``h`` (the response at the frame's end) and
    ``D_true`` (the D of the state held then).

    Every state diffuses in the cell at ``D(h) = D_before + (D_after - D_before) h`` (D 0 stays put); each sub-step
    the state moves by the exact matrix of the rates ``before + (after - before) h``; a molecule starts in the
    stationary occupancy of the rates at its first frame. A cell's factor scales both binding rates, before and
    after. A molecule's onset is ``onset_base_s`` (the delay and its day's and movie's offsets) plus its cell's and
    its own offset, floored at 0; those two offsets come from ``onset_rng`` only, never the molecules' stream.

    ``spec.dims`` 3 (31 Oct 2026): every state moves in the capsule's VOLUME as a 3D free protein does (a start
    uniform in it, `_walk_in_cell3`'s micro-steps), with ``sub_z_um`` and the instantaneous position in ``extra``
    beside SynK's own columns; 2 draws as the 2D sets were drawn."""
    from .simulate import ConditionSpec, StateSpec

    ind = spec.induction
    tau = design.frame_interval_s
    dt = tau / design.n_subframes
    n_sub = design.n_subframes
    n_exposed = design.exposed_subframes
    mid = (n_exposed - 1) // 2
    three = spec.dims == 3
    D0 = np.asarray(spec.D_um2_s, float)
    dD = np.asarray(ind.D_after_um2_s, float) - D0
    states_spec = tuple(StateSpec(float(d)) for d in D0)
    survive = math.exp(-1.0 / design.bleach_mean_frames)
    per_cell = spec.molecules_per_cell or design.molecules_per_cell
    # the session's drift: one factor on every D of this movie, from its start since the session's first movie
    drift = (max(0.0, 1.0 - float(ind.session_drift_per_hour)
                 * (float(movie_start_s) - min(float(v) for v in ind.movie_start_s)) / 3600.0)
             if ind.session_drift_per_hour else 1.0)

    def chain_for(cell):
        f = 1.0 if not cell_factors else float(cell_factors.get(cell.cell_id, 1.0))
        before, after = np.array(rates, float), np.array(rates_after, float)
        for r in (before, after):
            r[2, 0] *= f
            r[2, 1] *= f
        cache: dict = {}

        def matrix_at(h: float):
            key = round(h, 3)
            if key not in cache:
                cache[key] = ConditionSpec(name=spec.name, states=states_spec,
                                           rates_per_s=before + (after - before) * key).transition_matrix(dt)
            return cache[key]

        def occupancy_at(h: float):
            r = before + (after - before) * h               # F is the hub: p_i / p_F = k(F -> i) / k(i -> F)
            p = np.array([r[2, 0] / r[0, 2], r[2, 1] / r[1, 2], 1.0])
            return p / p.sum()
        return matrix_at, occupancy_at

    out = []
    for cell in cells:
        matrix_at, occupancy_at = chain_for(cell)
        cell_onset = onset_base_s + (float(onset_rng.normal(0.0, ind.onset_sd_cell_s))
                                     if ind.onset_sd_cell_s else 0.0)
        n_here = int(cell_counts.get(cell.cell_id, per_cell)) if cell_counts else per_cell
        for _molecule in range(n_here):
            onset = max(0.0, cell_onset + (float(onset_rng.normal(0.0, ind.onset_sd_molecule_s))
                                           if ind.onset_sd_molecule_s else 0.0))
            length = 1
            while rng.uniform() < survive and length < design.n_frames:
                length += 1
            if length < design.min_localizations and not keep_short:
                continue
            first = int(rng.integers(1, design.n_frames - length + 2))
            t0 = movie_start_s + (first - 1) * tau
            # the response at the END of every sub-step of the track, at once
            h_sub = ind.response(t0 + (np.arange(length * n_sub) + 1) * dt - onset)
            state = int(rng.choice(3, p=occupancy_at(float(ind.response(t0 - onset)))))
            if three:
                u, v, z = _start3(rng, cell, 2)
            else:
                u, v = _start(rng, cell, 2)
            xs, ys, states, inside = [], [], [], []
            true_x, true_y, sub_x, sub_y, sub_z = [], [], [], [], []
            inst_x, inst_y, inst_z = [], [], []
            t_s, h_end, d_true = [], [], []
            for k_frame in range(length):
                pos = []
                for j in range(n_sub):
                    h = float(h_sub[k_frame * n_sub + j])
                    state = int(rng.choice(3, p=matrix_at(h)[state]))
                    d = float((D0[state] + dD[state] * h) * drift)
                    if three:
                        if d > 0.0:
                            u, v, z = _walk_in_cell3(rng, cell, u, v, z, d, dt)
                        pos.append((u, v, z))
                    else:
                        if d > 0.0:
                            step = rng.normal(0.0, math.sqrt(2.0 * d * dt), 2)
                            u, v = cell.reflect_in_cell(u + step[0], v + step[1])
                        pos.append((u, v))
                exposed = pos[:n_exposed]
                mu, mv = np.mean([p[:2] for p in exposed], axis=0)
                x, y = cell.to_field(mu, mv)
                noise = rng.normal(0.0, design.sigma_um, 2)
                xs.append(float(x + noise[0]))
                ys.append(float(y + noise[1]))
                true_x.append(float(x))
                true_y.append(float(y))
                fx, fy = cell.to_field(np.array([p[0] for p in exposed]), np.array([p[1] for p in exposed]))
                sub_x.append([float(a) for a in np.atleast_1d(fx)])
                sub_y.append([float(a) for a in np.atleast_1d(fy)])
                if three:
                    sub_z.append([float(p[2]) for p in exposed])
                    iu, iv, iz = exposed[mid]
                    ix, iy = cell.to_field(iu, iv)
                    inst_x.append(float(ix))
                    inst_y.append(float(iy))
                    inst_z.append(float(iz))
                states.append(STATES[state])
                inside.append(cell.in_nucleoid3(u, v, z) if three else bool(cell.in_nucleoid(u, v)))
                t_s.append(float(movie_start_s + (first - 1 + k_frame) * tau))
                h_end.append(h)
                d_true.append(d)
            extra = {"t_s": t_s, "onset_s": [onset] * length, "h": h_end, "D_true": d_true}
            if three:
                extra.update(x_instant_um=inst_x, y_instant_um=inst_y, z_instant_um=inst_z)
            out.append({"cell": cell.cell_id, "track": len(out) + 1,
                        "frames": list(range(first, first + length)),
                        "x_um": xs, "y_um": ys, "states": states, "in_nucleoid": inside,
                        "x_true_um": true_x, "y_true_um": true_y, "sub_x_um": sub_x, "sub_y_um": sub_y,
                        **({"sub_z_um": sub_z} if three else {}), "extra": extra})
    return out


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_trackmate_xml(path, tracks: list, *, design: SynRDesign, image_name: str) -> Path:
    """A minimal TrackMate XML that `trackmate.read_trackmate_xml` reads: µm coordinates, 0-based frames, one
    track per molecule with an edge between consecutive spots, every track in ``FilteredTracks``."""
    import xml.etree.ElementTree as ET

    root = ET.Element("TrackMate", version="7.11.1")
    model = ET.SubElement(root, "Model", spatialunits="micron", timeunits="sec")
    spots_el = ET.SubElement(model, "AllSpots", nspots=str(sum(len(t["frames"]) for t in tracks)))
    by_frame: dict = {}
    spot_id = 0
    track_spots = []
    for t in tracks:
        ids = []
        for frame, x, y in zip(t["frames"], t["x_um"], t["y_um"]):
            spot_id += 1
            by_frame.setdefault(frame - 1, []).append((spot_id, x, y))
            ids.append(spot_id)
        track_spots.append(ids)
    for frame in sorted(by_frame):
        holder = ET.SubElement(spots_el, "SpotsInFrame", frame=str(frame))
        for sid, x, y in by_frame[frame]:
            ET.SubElement(holder, "Spot", ID=str(sid), name=f"ID{sid}", FRAME=str(frame),
                          POSITION_X=f"{x:.6f}", POSITION_Y=f"{y:.6f}", POSITION_Z="0.0",
                          POSITION_T=f"{frame * design.frame_interval_s:.6f}", QUALITY="100.0",
                          RADIUS="0.3", VISIBILITY="1")
    tracks_el = ET.SubElement(model, "AllTracks")
    filtered = ET.SubElement(model, "FilteredTracks")
    for index, ids in enumerate(track_spots):
        tr = ET.SubElement(tracks_el, "Track", name=f"Track_{index}", TRACK_ID=str(index),
                           NUMBER_SPOTS=str(len(ids)))
        for a, b in zip(ids, ids[1:]):
            ET.SubElement(tr, "Edge", SPOT_SOURCE_ID=str(a), SPOT_TARGET_ID=str(b))
        ET.SubElement(filtered, "TrackID", TRACK_ID=str(index))
    settings = ET.SubElement(root, "Settings")
    ET.SubElement(settings, "ImageData", filename=image_name, folder="",
                  width=str(design.field_px[0]), height=str(design.field_px[1]), nslices="1",
                  nframes=str(design.n_frames), pixelwidth=f"{design.pixel_size_nm / 1000.0:g}",
                  pixelheight=f"{design.pixel_size_nm / 1000.0:g}", voxeldepth="1.0",
                  timeinterval=f"{design.frame_interval_s:g}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
    return path


#: How much wider than the cell its written outline is, in localisation errors. A segmented outline sits on the
#: halo of a real cell; drawn exactly on the capsule, the 25 nm localisation error alone pushed 1.5 % of the free
#: protein's tracks (0.3 % of SynR's) across it, and the cell filter drops a track that leaves its cell -- a
#: one-sided loss of molecules near the wall, which is where the free state is. At 3 sigma none is lost.
OUTLINE_MARGIN_SIGMA = 3.0


def _write_meshes(path, cells: list, movie: str, design: SynRDesign) -> None:
    from dataclasses import replace

    from .cellmesh import CellMesh, write_cell_meshes

    um_per_px = design.pixel_size_nm / 1000.0
    margin = OUTLINE_MARGIN_SIGMA * design.sigma_um
    drawn = [replace(c, length_um=c.length_um + 2 * margin, width_um=c.width_um + 2 * margin) for c in cells]
    meshes = [CellMesh(movie=movie, cell_id=str(cell.cell_id), polygon=cell.outline() / um_per_px,
                       mesh=cell.ribs() / um_per_px) for cell in drawn]
    # the image's own pixels (0): the loader moves them onto the tracker's (`cellmesh.to_tracker_origin`)
    write_cell_meshes(path, meshes, image_shape=(design.field_px[1], design.field_px[0]), pixel_origin=0)


def write_true_outlines(root, *, design: SynRDesign | None = None, on_line=None) -> dict:
    """Stage 2's ``<group>/cell_meshes/<movie>.mat`` from the TRUTH, drawn as stage 1 draws them (`_write_meshes`:
    each cell's capsule `OUTLINE_MARGIN_SIGMA` localisation errors wider): the outlines a perfect segmentation would
    give, so the folder opens in the analysis with no segmentation model in between.

    Asked for (14 Oct 2026): "for segmentation dont use one of our models its too bad. u know where the cells are
    can u just not create it like this?" The cells are read back from ``truth/<group>_cells.csv`` (centre, angle,
    length, width: what the build placed); the exact capsules stay in ``truth/outlines/`` to score a segmentation
    against. NEVER over an existing outline file: a folder someone segmented keeps its outlines.

    Returns ``{group: number of movies written}``."""
    import pandas as pd

    design = design or SynRDesign()
    root = Path(root)
    say = on_line or (lambda _line: None)
    truth_path = root / "truth" / "truth.json"
    if not truth_path.is_file():
        raise SyntheticError(f"{root} is not a SynR build: no truth/truth.json")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    recorded = (truth.get("design") or {}).get("pixel_size_nm")
    if recorded is not None and float(recorded) != float(design.pixel_size_nm):
        raise SyntheticError(f"the build's pixel size is {recorded} nm, the design's {design.pixel_size_nm} nm")
    out = {}
    for group in truth["groups"]:
        table = root / "truth" / f"{group}_cells.csv"
        if not table.is_file():
            raise SyntheticError(f"no {table.name} in {root / 'truth'}")
        cells = pd.read_csv(table)
        target = root / group / "cell_meshes"
        existing = sorted(p.name for p in target.glob("*.mat")) if target.is_dir() else []
        if existing:
            raise SyntheticError(f"{target} already holds {len(existing)} outline file(s); not written over")
        for movie, part in cells.groupby("movie", sort=True):
            placed = [CapsuleCell(cell_id=int(r.cell), cx_um=float(r.cx_um), cy_um=float(r.cy_um),
                                  angle=float(r.angle_rad), length_um=float(r.length_um),
                                  width_um=float(r.width_um), lobes=())
                      for r in part.itertuples(index=False)]
            _write_meshes(target / f"{movie}.mat", placed, str(movie), design)
        out[group] = int(cells["movie"].nunique())
        say(f"{group}: true outlines for {out[group]} movie(s), {len(cells)} cells, drawn "
            f"{OUTLINE_MARGIN_SIGMA:g} sigma ({OUTLINE_MARGIN_SIGMA * design.sigma_um * 1000:.0f} nm) wider")
    return out


def _transitions(tracks: list) -> list:
    """Frame-to-frame state transition COUNTS over consecutive localisations, ``[from][to]`` in `STATES`
    order: what the molecules actually did, which an HMM's per-frame matrix is scored against (the spatial rule
    on F -> S makes the effective rates differ from the matrix the chain was built with)."""
    index = {s: i for i, s in enumerate(STATES)}
    counts = np.zeros((len(STATES), len(STATES)), dtype=int)
    for t in tracks:
        for a, b in zip(t["states"], t["states"][1:]):
            counts[index[a], index[b]] += 1
    return counts.tolist()


def _switch_truth(table, spec: GroupSpec, design: SynRDesign) -> dict:
    """The switch's truth: its design, and the true share of each state among the localisations in each second
    around the switch (``t_s`` = the bin's start, seconds from the switch; negative before it)."""
    tau = design.frame_interval_s
    t = (table["frame"].to_numpy(float) - spec.switch_frame) * tau
    bins = np.floor(t).astype(int)
    by_second = []
    for b in range(int(bins.min()), int(bins.max()) + 1):
        states = table["true_state"].to_numpy()[bins == b]
        if states.size:
            by_second.append({"t_s": b, "n": int(states.size),
                              **{s: float(np.mean(states == s)) for s in STATES}})
    return {"frame": spec.switch_frame, "half_life_s": spec.switch_half_life_s,
            "before": spec.name + " (its own occupancy and rates)", "after": spec.after.name,
            "before_occupancy": dict(zip(STATES, spec.occupancy)),
            "after_occupancy": dict(zip(STATES, spec.after.occupancy)),
            "rule": "rates(t) = after + (before - after) * 2^(-t / half_life), t from the switch frame",
            "share_by_second": by_second}


def _occupancy(states) -> dict:
    counts = {s: 0 for s in STATES}
    for s in states:
        counts[s] += 1
    total = max(1, sum(counts.values()))
    return {s: counts[s] / total for s in STATES}


def build_project(root, *, design: SynRDesign | None = None, groups=SYNR_GROUPS, on_line=None,
                  name: str | None = None) -> dict:
    """Build SynR at ``root`` (which must not exist) and return the truth manifest it also writes."""
    from dataclasses import replace

    import pandas as pd

    design = design or SynRDesign()
    root = Path(root)
    if root.exists():
        raise SyntheticError(f"{root} exists: a synthetic project is built into a new folder, never over one")
    say = on_line or (lambda _line: None)
    rng = np.random.default_rng(design.seed)
    synn = all(g.kind == "synn" for g in groups)
    synk_set = all(g.kind == "synk" for g in groups)
    syng = any(g.fork_capture_um is not None for g in groups)
    name = name or ("SynN" if synn else "SynK" if synk_set else "SynG" if syng else "SynR v2" if any(
        g.name in (SYNR_TIGHT.name, SYNR_SWITCH.name) for g in groups) else "SynR v1")
    truth = {"name": name, "synthetic": True,
             "note": ("SynN is synthetic and inspired by a YicC-family RNase; no real protein is implied"
                      if synn else "SynK is synthetic: an induction changes D and occupancy over time; no real "
                                   "protein is implied" if synk_set else SYNG_NOTE if syng else
                      "SynR is synthetic and CcpA-inspired; no real protein is implied"),
             "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(design).items()},
             "states": ({"B": "M: bound at the membrane / degradosome (anchored on the membrane)",
                         "S": "R: RNA-bound (cytoplasm)", "F": "F: free (cytoplasm)"} if synn else
                        {"B": "slow (its D before and after the induction declared per group)",
                         "S": "medium", "F": "fast, the hub (whole cell)"} if synk_set else
                        SYNG_STATES if syng else
                        {"B": "bound at an operator (anchored, in the nucleoid)",
                         "S": "scanning DNA (in the nucleoid)", "F": "free (whole cell)"}),
             "groups": {}}
    (root / "truth").mkdir(parents=True)
    lo, hi = design.cell_length_range_um
    base_design = design
    for spec in groups:
        # A GROUP MAY RUN LONGER (a switch needs its window); everything else is the design's.
        design = replace(base_design, n_frames=spec.n_frames) if spec.n_frames else base_design
        group_dir = root / spec.name
        rows = []
        group_truth = {"description": spec.description, "model": spec.kind, **({"dims": 3} if spec.dims == 3 else {}),
                       "D_um2_s": dict(zip(
            STATES[-len(spec.D_um2_s):], spec.D_um2_s)), "target_occupancy": dict(zip(
                STATES[-len(spec.occupancy):], spec.occupancy)), "days": {}}
        crops = {}
        cell_rows = []
        extra_cols: dict = {}
        bio_rng = _bio_rng(design, spec.name)
        ind = spec.induction
        onset_rng = _onset_stream(design, spec)
        movies_truth = {}
        for day in range(1, design.days + 1):
            factor = float(math.exp(rng.normal(0.0, design.day_bound_sd))) if not spec.free_only else 1.0
            day_offset = _day_onset(spec, onset_rng)
            day_states, day_steps = [], []
            day_rates = None
            day_tracks = []
            for m in range(design.movies_per_day):
                movie = f"series{(day - 1) * design.movies_per_day + m + 1:03d}"
                lengths = np.clip(rng.normal(design.cell_length_mean_um, design.cell_length_sd_um,
                                             design.cells_per_movie), lo, hi)
                cells = place_cells(rng, lengths, width_um=design.cell_width_um, field_um=design.field_um)
                share = nucleoid_share(cells, dims=spec.dims)
                movie_factor, cell_factors, cell_counts = _biology(design, spec, bio_rng, cells)
                rates = None if spec.free_only else rates_for(spec, nucleoid_share=share,
                                                               membrane_share=membrane_share(
                                                                   cells, spec.membrane_shell_um),
                                                               fork_share=_fork_share_of(spec, cells),
                                                               bound_factor=factor * movie_factor)
                after = (None if spec.after is None else
                         rates_for(spec.after, nucleoid_share=share, bound_factor=factor * movie_factor))
                induction_kw = {}
                if ind is not None:
                    after, induction_kw, movies_truth[movie] = _movie_induction(
                        spec, onset_rng, m, day_offset, factor * movie_factor)
                day_rates = rates if day_rates is None else day_rates
                tracks = simulate_movie(rng, cells, design=design, spec=spec, rates=rates,
                                        rates_after=after, cell_factors=cell_factors, cell_counts=cell_counts,
                                        stuck=stuck_level(design, spec, day), **induction_kw)
                if not tracks:
                    raise SyntheticError(f"{spec.name} {movie}: no track long enough")
                day_tracks += tracks
                for c in cells:
                    cell_rows.append({
                        "movie_bound_factor": movie_factor,
                        "cell_bound_factor": (cell_factors or {}).get(c.cell_id, 1.0),
                        "molecules": (cell_counts or {}).get(c.cell_id,
                                                             spec.molecules_per_cell or design.molecules_per_cell),
                        "day": day, "movie": movie, "cell": c.cell_id, "cx_um": c.cx_um, "cy_um": c.cy_um,
                        "angle_rad": c.angle, "length_um": c.length_um, "width_um": c.width_um,
                        "nucleoid_share": _cell_nucleoid_share(c, spec.dims),
                        "lobes": json.dumps([[lb.centre_um, lb.half_length_um, lb.half_width_um]
                                             for lb in c.lobes])})
                write_trackmate_xml(group_dir / "trackmate" / f"{movie}.xml", tracks, design=design,
                                    image_name=f"{movie}.tif")
                _write_meshes(group_dir / "cell_meshes" / f"{movie}.mat", cells, movie, design)
                crops[movie] = [1, design.n_frames]
                for t in tracks:
                    day_states += t["states"]
                    day_steps += t["states"][:-1]
                    for k, frame in enumerate(t["frames"]):
                        rows.append((day, movie, t["cell"], t["track"], frame, t["x_um"][k], t["y_um"][k],
                                     t["states"][k], t["in_nucleoid"][k]))
                        for column, values in t.get("extra", {}).items():
                            extra_cols.setdefault(column, []).append(values[k])
            group_truth["days"][str(day)] = {
                "bound_rate_factor": factor,
                "rates_per_s_first_movie": None if day_rates is None else np.round(day_rates, 6).tolist(),
                "occupancy_localisations": _occupancy(day_states),
                "occupancy_steps": _occupancy(day_steps),
                "transition_counts_frames": _transitions(day_tracks),
                "n_localisations": len(day_states), "n_steps": len(day_steps),
                **({"stuck_share": stuck_level(design, spec, day)} if spec.stuck_range is not None else {})}
            say(f"{spec.name} replicate {day}: {len(day_states):,} localisations, bound "
                f"{100 * _occupancy(day_steps)['B']:.1f} % of steps")
        table = pd.DataFrame(rows, columns=["day", "movie", "cell", "track", "frame", "x_um", "y_um",
                                            "true_state", "in_nucleoid"])
        for column, values in extra_cols.items():
            if len(values) != len(table):
                raise SyntheticError(f"{spec.name}: the truth column {column!r} is missing for some tracks")
            table[column] = values
        if ind is not None:
            group_truth["induction"] = {**asdict(ind), "movies": movies_truth,
                                        "rule": "h = the shape at (t - onset); D(h) = before + (after - before) h; "
                                                "rates(h) = before + (after - before) h; onset = max(0, delay + "
                                                "replicate + movie + cell + molecule offsets)"}
        group_truth["occupancy_steps"] = _occupancy(
            [s for _key, sub in table.groupby(["movie", "track"]) for s in list(sub["true_state"])[:-1]])
        group_truth["n_frames"] = design.n_frames
        if spec.switch_frame is not None:
            group_truth["switch"] = _switch_truth(table, spec, design)
        truth["groups"][spec.name] = group_truth
        # THE DAYS, EXPLICIT: the app's door reads a condition's remembered day map
        # (`daymap.load_day_map`), so the replicate days are the simulation's whatever
        # ``movies_per_day`` is, not the default ten-series rule.
        (group_dir / "day_map.json").write_text(json.dumps({
            "movie_to_day": {movie: (int(movie[6:]) - 1) // design.movies_per_day + 1
                             for movie in crops},
            "source": "synthetic", "series_per_day": design.movies_per_day, "warnings": [],
            "written_by": "smtracker.synthetic.build_project"}, indent=2), encoding="utf-8")
        (group_dir / "bleaching").mkdir(parents=True, exist_ok=True)
        (group_dir / "bleaching" / "crops.json").write_text(json.dumps({
            "crops": crops, "pixel_size_nm": design.pixel_size_nm,
            "frame_interval_s": design.frame_interval_s, "exposure_s": design.exposure_s, "label": "mNeonGreen", "emission_nm": 517.0,
            "numerical_aperture": 1.49,
            "note": [f"SYNTHETIC ({name}): every frame of every movie is used"]}, indent=2), encoding="utf-8")
        with gzip.open(root / "truth" / f"{spec.name}_localisations.csv.gz", "wt", encoding="utf-8") as fh:
            table.to_csv(fh, index=False)
        pd.DataFrame(cell_rows).to_csv(root / "truth" / f"{spec.name}_cells.csv", index=False)
    (root / "truth" / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


# --------------------------------------------------------------------------- #
# Stage 2: movies and phase images from the same truth
# --------------------------------------------------------------------------- #

#: The camera and photon budget a SynR movie is rendered with: MEASURED on this lab's microscope, never chosen
#: (`moviesim.PhotonSpec` refuses a default, since recall is almost entirely a function of it).
@dataclass(frozen=True)
class MovieRealism:
    """The imperfections of a REALISTIC stage-2 copy (asked 14 Oct 2026, of the list of what the clean movies lack:
    autofluorescence "for QC", out-of-focus haze, the illumination falloff, blinking "for trackmate and utrack";
    crowded first frames, hot pixels and drift stay out). NO FIELD HAS A DEFAULT: each is measured on the lab's
    data and ``source`` says where (`docs/synthetic_synr.md`), as `moviesim.PhotonSpec` refuses a guessed budget.

    - ``illumination_centre_px`` / ``illumination_sigma_px``: a Gaussian beam over the field, scaled to a FIELD
      MEAN of 1 so the measured photon budgets stay field averages; it multiplies every molecule's photons, the
      background and the autofluorescence (all excited by the laser), never the camera offset.
    - ``autofluorescence_photons_per_px``: extra photons per pixel per frame inside every cell (at the field mean
      illumination), fading with ``autofluorescence_half_life_frames`` (None: constant).
    - ``defocus_depth_um`` (zd): each molecule has a height z in its cell (the circular cross-section, radius half
      the width) that diffuses with its state's D; its spot is sigma0 * sqrt(1 + (z / zd)^2) wide, the photons
      kept -- wider and dimmer away from focus, the haze of molecules above and below the focal plane.
    - ``blink_off_per_frame`` / ``blink_on_per_frame``: a two-state on/off chain per molecule, per frame; a dark
      frame emits nothing (the molecule still moves). A molecule appears ON.
    - ``haze_photons_per_px``: a diffuse glow inside every cell from the labelled protein's out-of-focus and
      unresolved molecules (measured: labelled minus WT in-cell excess) -- 16 molecules a cell cannot make it.
    - ``autofluorescence_cell_cv``: cells differ in autofluorescence (a lognormal factor per cell with this CV).
    - NO AUTOFLUORESCENT SPOTS. A spot model (emitters appearing in each cell, moving with one D or a mix, tuned
      to the WT controls' tracks) was here from 14 to 25 Oct 2026 and is DELETED, for every synthetic protein
      (ruled 25 Oct 2026: "delete it and forget these spot ideas. for all proteins"): its spots made a population
      at 0.1-0.2 um2/s that real free mNG never shows, its shape was carried over rather than measured, and a rate
      tuned on WT cells did not carry over to labelled ones. A free protein's slow floor is a declared stuck
      fraction instead (`GroupSpec.stuck_range`). Sets built before keep their spots in their pixels and truth.
    - ``emission_cv`` / ``emission_corr_frames``: a molecule's photons vary beyond shot noise (a lognormal factor,
      mean 1, this CV), correlated over frames with this time (an AR(1) in log; 0 = independent frames) -- TUNED
      to the real tracks' NOISE, not measured: defocus and blinking alone left the brightness within a track at a
      quality CV of 0.15 where real TrackMate tracks show 0.35-0.36 (two lab conditions), and independent
      per-frame flicker drops single frames the real gap pattern does not show (pilot 2, 14 Oct 2026).

    Drawn from their OWN random stream: the realistic copy holds the SAME molecules as the clean copy under the
    same seed, so the two differ only by these imperfections."""

    illumination_centre_px: tuple
    illumination_sigma_px: float
    autofluorescence_photons_per_px: float
    autofluorescence_half_life_frames: float | None
    defocus_depth_um: float
    blink_off_per_frame: float
    blink_on_per_frame: float
    haze_photons_per_px: float
    autofluorescence_cell_cv: float
    emission_cv: float
    emission_corr_frames: float
    source: str

    def illumination(self, cols: int, rows: int) -> np.ndarray:
        """The field's illumination, rows x cols, mean 1."""
        yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
        cx, cy = self.illumination_centre_px
        beam = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * float(self.illumination_sigma_px) ** 2))
        return beam / beam.mean()

    def as_dict(self) -> dict:
        from dataclasses import asdict

        out = asdict(self)
        out["illumination_centre_px"] = list(self.illumination_centre_px)
        return out


def _cell_mask(cells: list, design: "SynRDesign", weights: dict | None = None) -> np.ndarray:
    """1 inside any cell (or the cell's own ``weights[cell_id]``), 0 outside, on the field's pixel grid (pixel
    centres at integer coordinates)."""
    um = design.pixel_size_nm / 1000.0
    cols, rows = design.field_px
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    mask = np.zeros((rows, cols))
    for cell in cells:
        u, v = cell.to_local(xx * um, yy * um)
        w = 1.0 if weights is None else float(weights[cell.cell_id])
        mask = np.maximum(mask, w * cell.inside(u, v).astype(float))
    return mask


def _realistic_rows(tracks: list, cells: list, *, spec: "GroupSpec", design: "SynRDesign",
                    realism: MovieRealism, illum: np.ndarray, share: float, rng) -> tuple:
    """The sub-step rows a realistic frame renders, and per (track, frame) the molecule's mean height and whether
    it was dark. Heights diffuse with the frame's state's D inside the cell's cross-section; blinking is an on/off
    chain per frame; a dark frame renders nothing. A SynG molecule bound at a fork sits at the FORK'S height -- on
    the cell's axis, the focal plane, within the replisome's size (`FORK_SPREAD_UM`) -- for its whole stay (27 Oct
    2026: until then it kept the height it had, uniform over the cell's depth, and a quarter of fork-bound steps lay
    over 0.3 um out of focus, dim and wide, reading D ~0.08 among the scanning molecules)."""
    um = design.pixel_size_nm / 1000.0
    sigma0 = MOVIE_CALIBRATION["psf_sigma_px"]
    by_id = {c.cell_id: c for c in cells}
    dt = design.frame_interval_s / design.n_subframes
    dead = design.n_subframes - design.exposed_subframes
    states = STATES if not spec.free_only else ("F",)
    D = dict(zip(STATES, spec.D_um2_s)) if not spec.free_only else {"F": spec.D_um2_s[0]}
    forks = spec.fork_capture_um is not None
    p_off, p_on = float(realism.blink_off_per_frame), float(realism.blink_on_per_frame)
    cv = float(realism.emission_cv)
    sd_log = math.sqrt(math.log(1.0 + cv * cv))
    corr = math.exp(-1.0 / realism.emission_corr_frames) if realism.emission_corr_frames > 0 else 0.0
    rows, extra = [], {}
    rows_px, cols_px = illum.shape
    for t in tracks:
        cell = by_id[t["cell"]]
        r = cell.radius
        z, on, log_e, on_fork = None, True, None, False
        for k, (f, xs_, ys_, st) in enumerate(zip(t["frames"], t["sub_x_um"], t["sub_y_um"], t["states"])):
            at_fork = forks and st == "B"
            on_fork = on_fork and at_fork                     # left the fork: its height diffuses from there
            if k:
                on = (rng.random() >= p_off) if on else (rng.random() < p_on)
            if sd_log:
                shock = float(rng.normal(0.0, sd_log))
                log_e = shock if log_e is None else corr * log_e + math.sqrt(1.0 - corr * corr) * shock
                emission = math.exp(log_e - 0.5 * sd_log * sd_log)
            else:
                emission = 1.0
            zs = []
            known = t["sub_z_um"][k] if "sub_z_um" in t else None   # SynN: the model's own heights
            if k and dead and known is None:            # the previous frame's dead time, in which it moved too
                z += float(rng.normal(0.0, math.sqrt(2.0 * D.get(st, 0.0) * dt * dead)))
            for i_sub, (sx, sy) in enumerate(zip(xs_, ys_)):
                _u, v = cell.to_local(sx, sy)
                h = math.sqrt(max(r * r - float(v) ** 2, 0.0))
                if known is not None:
                    z = float(known[i_sub])
                elif at_fork:
                    if not on_fork:                           # captured: at the fork's height for the stay
                        z = float(rng.normal(0.0, FORK_SPREAD_UM))
                        on_fork = True
                elif z is None:
                    z = float(rng.uniform(-h, h))
                else:
                    z += float(rng.normal(0.0, math.sqrt(2.0 * D.get(st, 0.0) * dt)))
                if h > 0 and abs(z) > h:                      # reflect at the cell's top or bottom
                    z = math.copysign(2.0 * h - abs(z), z)
                    z = max(-h, min(h, z))
                elif h == 0:
                    z = 0.0
                zs.append(z)
                if on:
                    px, py = sx / um, sy / um
                    ix = min(cols_px - 1, max(0, int(round(px))))
                    iy = min(rows_px - 1, max(0, int(round(py))))
                    rows.append({"track_id": t["track"], "frame": f, "x": px, "y": py,
                                 "photons": share * float(illum[iy, ix]) * emission,
                                 "sigma_px": sigma0 * math.sqrt(1.0 + (z / realism.defocus_depth_um) ** 2)})
            extra[(t["track"], f)] = (float(np.mean(zs)), not on)
    _ = states
    return rows, extra


#: HOW A FRAME IS RENDERED (14 Oct 2026; before, a molecule was ONE sharp spot at its blurred average plus the
#: design's 25 nm error, which made fast molecules too easy to detect -- SynR's free state moves ~3.6 px a frame
#: -- and counted the localisation error twice: TrackMate read the bound floor at 0.064 um^2/s, not 0.032).
MOVIE_RENDERING = ("each frame is the molecule's n_subframes sub-step positions, each a PSF spot carrying an equal "
                   "share of the frame's photons (motion blur smears and dims a fast molecule); no localisation "
                   "error is added to the positions, the photon and camera noise set it; the truth position is "
                   "the true blurred centroid")

MOVIE_CALIBRATION = {
    "psf_sigma_px": 0.932,
    "offset_counts": 507.0,
    "em_gain": 140.0,
    "sensitivity_e_per_count": 4.24,
    "photons_per_frame": 127.0,
    "background_photons_per_px": 4.81,
    "source": ("EMCCD by photon transfer on one lab movie, 60 frames (sensitivity 4.24 e-/count at EM "
               "gain 140, rendered noise within 21 % of the real movie); photon budget from the recall benchmark "
               "(127 photons per molecule per frame, 4.81 background photons per pixel), which with the PSF sigma "
               "0.932 px predicts the 17 nm precision u-track reports: docs/claude/trackers.md, "
               "docs/claude/sigma_and_floor.md"),
}


#: Nikon 1 (MetaMorph, one user's primase movies), MEASURED 24 Oct 2026 on the wild-type movies of both strains
#: (`docs/synthetic_synr.md`, "SynG wild type on Nikon 1"): the offset and read noise from the dark frames before the
#: laser; the counts per photon from each pixel's FRAME-TO-FRAME variance outside the cells less the dark frames',
#: over its mean above the offset (EMCCD excess 2) -- 11.4-12.9 on 24 movies, where the across-pixel photon transfer
#: (`moviesim.calibrate_camera`) scattered 3.8-8.4 at this signal and read a pilot rendered at 5.7 as 2.0-7.9; the
#: background outside the cells; the photons per molecule set so the rendered spots integrate as the real ones do at
#: u-track's localisations (~1,920 counts in a 2.5 px disc; pilot 1 integrated 0.90 of what it rendered).
MOVIE_CALIBRATION_N1 = {
    "psf_sigma_px": 0.932,
    "offset_counts": 2054.5,
    "em_gain": 141.0,
    "sensitivity_e_per_count": 141.0 / 12.4,
    "read_noise_e": 31.6 * 141.0 / 12.4,
    "photons_per_frame": 171.0,
    "background_photons_per_px": 2.6,
    "source": ("Nikon 1 (MetaMorph EMCCD), one user's wild types (two strains): offset 2053.1-2055.9 "
               "counts and read noise 30.3-33.1 counts (dark frames, 24 movies); 12.4 counts per photon (11.4-12.9) "
               "from frame-to-frame variance outside cells at EM gain 141; background outside cells ~32 counts "
               "(2.6 photons/px); spots integrate 1630-2810 counts (median ~1,920) at u-track's localisations (r 2.5 "
               "px disc, annulus to 5 px), 171 photons rendered for it"),
}


# --------------------------------------------------------------------------- #
# SynR v3 (14 Oct 2026): the declared truth at two abundances, the measured noise, the realistic copy's imaging
# --------------------------------------------------------------------------- #
#: The groups of a SynR v3 movie set: the effect pair, the null pair, free mNG at D 2 and at a realistic D 5.
SYNR_V3_GROUPS = ("SynR_glucose", "SynR_malate", "SynR-coA_glucose", "SynR-coA_malate", "mNG_free", "mNG_free_D5")
#: The BIOLOGICAL NOISE, measured on a lab's project (7 proteins, glucose and malate, the nested mixed model;
#: docs/synthetic_synr.md): log-SDs of the factor on k(S -> B) per day / movie / cell, and of the molecule count.
_V3_NOISE = dict(day_bound_sd=0.15, movie_bound_sd=0.4, cell_bound_sd=0.6, molecules_sd_log=0.58)
#: The DECLARED truth at two abundances (ruled 14 Oct 2026: "one with higher one with lower abundance", named
#: "low-copy / high-copy"): the mean molecules per cell SEEN in the 20 s analysed -- the copy number the movie
#: shows after the dense first frames bleached, not the cell's total -- and a 15-frame lifetime. LOW-COPY, 35: a
#: transcription factor's sparse tracks (pilot 5: 14-17 tracks per cell per 1000 frames); HIGH-COPY, 55: an
#: abundant protein's (pilot 6: 19-22). Never fitted to the lab's tracks (docs/synthetic_synr.md).
SYNR_V3_LOWCOPY = SynRDesign(molecules_per_cell=35, bleach_mean_frames=15.0, **_V3_NOISE)
SYNR_V3_HIGHCOPY = SynRDesign(molecules_per_cell=55, bleach_mean_frames=15.0, **_V3_NOISE)
#: The realistic copy's imaging, measured on the lab's data unless marked TUNED (tuned to the real TrackMate
#: tracks' noise: the gap pattern, the brightness within a track, the autofluorescence kept), 14 Oct 2026.
SYNR_V3_REALISM = MovieRealism(
    illumination_centre_px=(53.5, 78.5), illumination_sigma_px=80.0,
    autofluorescence_photons_per_px=0.75, autofluorescence_half_life_frames=3000.0, autofluorescence_cell_cv=0.4,
    haze_photons_per_px=1.15, defocus_depth_um=0.35,
    blink_off_per_frame=0.015, blink_on_per_frame=0.4,
    emission_cv=0.9, emission_corr_frames=2.0,
    source=("illumination: pseudo-flat-field over 24 replicates (centre, sigma); autofluorescence: WT 3610 "
            "controls (+1.3 photons/px in LB, x 1/1.8 for minimal medium; cell CV 0.4); haze: labelled minus WT "
            "in-cell excess (~1.15 in minimal); defocus depth: derived from the optics (NA 1.49, 517 nm), "
            "consistent with real spot widths; blinking: TrackMate gap statistics on a real data set, then TUNED with the "
            "emission flicker to the real gap share (7-9 %), 2/1-frame ratio (0.44-0.48) and quality CV (0.35); "
            "no autofluorescent spots (the spot model was deleted on 25 Oct 2026)"))

#: Nikon 1's realistic imaging, for SynG's wild type matched to one user's primase movies (24 Oct 2026, asked: "we can build syng
#: wildtype", on Nikon 1, at a density "in between" his two strains). MEASURED there: the in-cell background over the
#: outside, 25 counts (2.0 photons/px) in the first strain (7.8 u-track tracks a cell a movie) and 70 (5.6) in the second strain
#: (25), split by a line through the two into 0.4 of autofluorescence and haze rising with the copy (3.1 at ~15
#: tracks a cell). CARRIED
#: OVER from Nikon 2 as properties of the objective and the fluorophore, not of the camera: defocus, blinking, the
#: emission flicker, the cells' autofluorescence spread and fading. NOT
#: KNOWN for Nikon 1: its beam, so the field is lit evenly (a broad beam) rather than with Nikon 2's.
SYNG_N1_REALISM = replace(
    SYNR_V3_REALISM, illumination_centre_px=(64.0, 64.0), illumination_sigma_px=5000.0,
    autofluorescence_photons_per_px=0.4, haze_photons_per_px=3.1,
    source=("Nikon 1, one user's primase movies (24 Oct 2026): in-cell background over the outside 2.0 (first strain) and 5.6 "
            "(second strain) photons/px at 12.4 counts per photon, split into autofluorescence 0.4 and haze 3.1 at ~15 "
            "tracks a cell; the field lit evenly (Nikon 1's beam not measured); defocus, blinking, emission "
            "flicker, the cell spread and fading carried over from Nikon 2's profile: " + SYNR_V3_REALISM.source))


def render_phase(cells: list, design: SynRDesign, rng, *, background: float = 2000.0,
                 interior: float = 0.62, halo: float = 0.35, halo_um: float = 0.18, edge_um: float = 0.04,
                 blur_px: float = 1.0, noise: float = 25.0) -> np.ndarray:
    """A phase-contrast-like image of ``cells``: a grey field, each cell DARK inside with a soft edge and a BRIGHT
    halo just outside it, blurred by the optics, with camera noise -- what a phase segmentation model is trained
    on. Pixel centres at integer coordinates, the same grid the fluorescence movie is rendered on."""
    from scipy.ndimage import gaussian_filter

    um = design.pixel_size_nm / 1000.0
    cols, rows = design.field_px
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    x_um, y_um = xx * um, yy * um
    dark = np.ones((rows, cols))
    bright = np.zeros((rows, cols))
    for cell in cells:
        u, v = cell.to_local(x_um, y_um)
        d, _q = cell.axis_distance(u, v)
        signed = d - cell.radius                          # < 0 inside the cell
        inside = 1.0 / (1.0 + np.exp(signed / edge_um))
        dark = np.minimum(dark, 1.0 - (1.0 - interior) * inside)
        bright += halo * np.exp(-(np.clip(signed, 0.0, None) / halo_um) ** 2) * (signed > 0)
    image = gaussian_filter(dark + bright, blur_px) * background
    image = image + rng.normal(0.0, noise, image.shape)
    return np.clip(np.rint(image), 0, 65535).astype(np.uint16)


#: STAGE 2's RECORDING CALENDAR (26 Oct 2026, for SynK's "Time after the addition"): replicate d is recorded on
#: ``MOVIE_FIRST_DATE`` + (d - 1) weeks (each its own culture, Rule 0). A SynK group's addition is at ``MOVIE_CLOCK``
#: each replicate and every movie starts at the addition plus its `Induction.movie_start_s`; any other group's movies
#: start at the clock, one after another with ``MOVIE_GAP_S`` between them. Written into each `.stk` (`write_stk`).
MOVIE_FIRST_DATE = _dt.date(2026, 1, 5)
MOVIE_CLOCK = _dt.time(10, 0, 0)
MOVIE_GAP_S = 40.0


def _metamorph_day(day: _dt.date) -> int:
    """MetaMorph's day number (the Julian day number less one, as `tifffile.julian_datetime` reads it)."""
    return day.toordinal() + 1721424


def write_stk(path, stack, *, start: _dt.datetime, tau_s: float, exposure_s: float | None = None,
              em_gain: float | None = None) -> Path:
    """``stack`` (frames, rows, columns; uint16) as a MetaMorph `.stk`: ONE image directory whose strip holds the
    first plane and the rest follow it contiguously (MetaMorph's layout, which `tifffile` reads as a series), UIC1
    (CreateTime and LastSavedTime: the first plane's), UIC2 (per plane: z 0, the creation day and millisecond, the
    same as modified), and the per-plane
    descriptions MetaMorph writes (``Exposure: 20 ms``, ``EMCCD Gain: 141``). Plane i was created at ``start`` + i
    ``tau_s``, to the whole millisecond as MetaMorph stamps, so `rawmovie.acquisition_time` reads ``start`` and
    `rawmovie.frame_interval` the frame interval (the average spacing). Before 26 Oct 2026 stage 2 wrote a plain
    TIFF stack under the `.stk` name, with no time in it. Refuses an existing file."""
    path = Path(path)
    if path.exists():
        raise SyntheticError(f"{path} exists: a camera file is never written over")
    data = np.ascontiguousarray(stack)
    if data.ndim != 3 or data.dtype != np.uint16:
        raise SyntheticError(f"a .stk holds uint16 frames (frames, rows, columns), got {data.dtype} {data.shape}")
    n, h, w = (int(v) for v in data.shape)
    plane = h * w * 2
    fields = [f"Exposure: {1000.0 * float(exposure_s if exposure_s is not None else tau_s):g} ms"]
    if em_gain is not None:
        fields.append(f"EMCCD Gain: {float(em_gain):g}")
    description = ("\x00".join(["\r\n".join(fields)] * n) + "\x00").encode("ascii")
    uic2 = np.zeros((n, 6), "<u4")
    uic2[:, 1] = 1                                          # z distance 0 / 1
    for i in range(n):
        at = start + _dt.timedelta(milliseconds=round(1000.0 * i * float(tau_s)))
        day, ms = _metamorph_day(at.date()), (at - _dt.datetime.combine(at.date(), _dt.time())) // _dt.timedelta(
            milliseconds=1)
        uic2[i, 2:] = (day, ms, day, ms)
    data_at = 8
    uic2_at = data_at + n * plane
    stamp_at = uic2_at + uic2.nbytes                        # the first plane's (day, ms): UIC1's CreateTime
    uic1_at = stamp_at + 8
    uic1 = np.array([16, stamp_at, 17, stamp_at], "<u4")    # MetaMorph's CreateTime and LastSavedTime
    text_at = uic1_at + uic1.nbytes
    ifd_at = text_at + len(description) + (len(description) % 2)
    entries = [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 1, 16), (259, 3, 1, 1), (262, 3, 1, 1),
               (270, 2, len(description), text_at), (273, 4, 1, data_at), (277, 3, 1, 1), (278, 4, 1, h),
               (279, 4, 1, plane), (284, 3, 1, 1), (339, 3, 1, 1), (33628, 4, 2, uic1_at), (33629, 5, n, uic2_at)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(b"II" + struct.pack("<HI", 42, ifd_at))
        fh.write(data.astype("<u2", copy=False).tobytes())
        fh.write(uic2.tobytes())
        fh.write(uic2[0, 2:4].tobytes())
        fh.write(uic1.tobytes())
        fh.write(description + b"\x00" * (len(description) % 2))
        fh.write(struct.pack("<H", len(entries)))
        for tag, kind, count, value in entries:
            packed = struct.pack("<H", value) + b"\x00\x00" if kind == 3 else struct.pack("<I", value)
            fh.write(struct.pack("<HHI", tag, kind, count) + packed)
        fh.write(struct.pack("<I", 0))
    return path


def _movie_starts(spec, design, day: int, n_movies: int) -> list:
    """Each movie's recording start (`MOVIE_FIRST_DATE`, `MOVIE_CLOCK`); a SynK group's at its addition plus
    `Induction.movie_start_s`. Refuses two movies of one replicate recorded at once: one camera."""
    clock = _dt.datetime.combine(MOVIE_FIRST_DATE + _dt.timedelta(weeks=day - 1), MOVIE_CLOCK)
    length = design.n_frames * design.frame_interval_s
    ind = spec.induction
    offsets = ([float(ind.movie_start_s[m % len(ind.movie_start_s)]) for m in range(n_movies)] if ind is not None
               else [m * (length + MOVIE_GAP_S) for m in range(n_movies)])
    ordered = sorted(offsets)
    clash = [(a, b) for a, b in zip(ordered, ordered[1:]) if b - a < length]
    if clash:
        raise SyntheticError(f"{spec.name}: movies starting {clash[0][0]:g} s and {clash[0][1]:g} s after the "
                             f"addition overlap ({length:g} s each): one camera records one movie at a time")
    return [clock + _dt.timedelta(seconds=s) for s in offsets]


def build_movie_project(root, *, design: SynRDesign | None = None,
                        groups=("SynR_glucose", "SynR_malate"), movies_per_day: int = 3,
                        on_line=None, realism: "MovieRealism | None" = None,
                        calibration: dict | None = None) -> dict:
    """Stage 2: SynR as a folder straight off the microscope -- ``<group>/raw/seriesNNN.stk`` (the fluorescence
    movie, uint16, every frame; a plain TIFF stack under MetaMorph's extension), ``<group>/phase/seriesNNN.tif`` (one phase image), ``<group>/bleaching/crops.json``
    (pixel size, frame interval, the whole movie as the crop) and ``day_map.json`` -- for the preparation window,
    a tracker and segmentation to work on. Its own realisation of the design, with EVERY molecule rendered (the
    short ones too: the tracker's minimum drops them). Series number blocks of ten are the days (001-00k day 1,
    011-01k day 2, ...), so the default day rule reads them.

    ``truth/``: ``truth.json`` (as stage 1, plus the rendering calibration), ``<group>_localisations.csv.gz``
    (every rendered molecule: field position in µm and pixels, true state, in-nucleoid), ``<group>_cells.csv`` and
    ``outlines/<group>/seriesNNN.mat``: the EXACT capsules, to score a segmentation against."""
    import pandas as pd
    import tifffile

    from .cellmesh import CellMesh, write_cell_meshes
    from .moviesim import CameraSpec, OpticsSpec, PhotonSpec, render_movie

    design = design or SynRDesign()
    if movies_per_day > 10:
        raise SyntheticError("at most ten movies a replicate: the series numbers' blocks of ten are the replicates")
    root = Path(root)
    if root.exists():
        raise SyntheticError(f"{root} exists: a synthetic project is built into a new folder, never over one")
    say = on_line or (lambda _line: None)
    specs = {g.name: g for g in SYNR_GROUPS + SYNN_GROUPS + SYNG_GROUPS + SYNK_GROUPS}
    # a group by its name, or declared here (a `GroupSpec`: a test's small SynK, a grid's)
    specs.update({g.name: g for g in groups if isinstance(g, GroupSpec)})
    groups = [g.name if isinstance(g, GroupSpec) else g for g in groups]
    unknown = [g for g in groups if g not in specs]
    if unknown:
        raise SyntheticError(f"no SynR, SynN, SynG or SynK group {unknown}; have {sorted(specs)}")
    family = ("SynN" if all(specs[g].kind == "synn" for g in groups) else
              "SynK" if all(specs[g].kind == "synk" for g in groups) else
              "SynG" if any(specs[g].fork_capture_um is not None for g in groups) else "SynR")
    # THE CAMERA: Nikon 2's by default (`MOVIE_CALIBRATION`), or another measured one (`MOVIE_CALIBRATION_N1`,
    # 24 Oct 2026), whose read noise is not negligible against its signal as an EMCCD's at high gain usually is
    cal = MOVIE_CALIBRATION if calibration is None else calibration
    optics = OpticsSpec(psf_sigma_px=cal["psf_sigma_px"], pixel_size_nm=design.pixel_size_nm)
    camera = CameraSpec(offset_counts=cal["offset_counts"], em_gain=cal["em_gain"],
                        sensitivity_e_per_count=cal["sensitivity_e_per_count"],
                        read_noise_e=float(cal.get("read_noise_e", 0.0)))
    photons = PhotonSpec(photons_per_frame=cal["photons_per_frame"],
                         background_photons_per_px=cal["background_photons_per_px"])
    rng = np.random.default_rng(design.seed + 2)
    um = design.pixel_size_nm / 1000.0
    lo, hi = design.cell_length_range_um
    truth = {"name": f"{family} movies" + (" (realistic)" if realism else ""), "synthetic": True, "stage": 2,
             "rendering": MOVIE_RENDERING, "realism": realism.as_dict() if realism else None,
             "note": ("SynN is synthetic and inspired by a YicC-family RNase; no real protein is implied"
                      if family == "SynN" else SYNG_NOTE if family == "SynG" else
                      "SynK is synthetic: an addition changes D and occupancy over time; no real protein is implied"
                      if family == "SynK" else
                      "SynR is synthetic and CcpA-inspired; no real protein is implied"),
             "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(design).items()},
             "movies_per_day": movies_per_day, "calibration": cal,
             "recording": {"first_date": MOVIE_FIRST_DATE.isoformat(), "clock": MOVIE_CLOCK.isoformat(),
                           "replicates_apart_days": 7, "gap_s": MOVIE_GAP_S,
                           "rule": "replicate d on first_date + (d - 1) weeks; a SynK group's addition at the "
                                   "clock, each movie at the addition + its movie_start_s; other groups' movies "
                                   "from the clock, gap_s apart; each .stk stamps its planes"},
             "groups": {}}
    for name in groups:                                    # the clocks first: nothing written on a refusal
        for day in range(1, design.days + 1):
            _movie_starts(specs[name], design, day, movies_per_day)
    (root / "truth").mkdir(parents=True)
    for name in groups:
        spec = specs[name]
        group_dir = root / name
        rows, cell_rows, crops, day_map = [], [], {}, {}
        group_truth = {"description": spec.description, "model": spec.kind, **({"dims": 3} if spec.dims == 3 else {}),
                       "D_um2_s": dict(zip(
            STATES[-len(spec.D_um2_s):], spec.D_um2_s)), "target_occupancy": dict(zip(
                STATES[-len(spec.occupancy):], spec.occupancy)), "days": {}}
        bio_rng = _bio_rng(design, name)
        ind = spec.induction
        onset_rng = _onset_stream(design, spec)
        movies_truth, recorded, extra_cols = {}, {}, {}
        for day in range(1, design.days + 1):
            factor = float(math.exp(rng.normal(0.0, design.day_bound_sd))) if not spec.free_only else 1.0
            day_offset = _day_onset(spec, onset_rng)
            starts = _movie_starts(spec, design, day, movies_per_day)
            day_states = []
            for m in range(movies_per_day):
                movie = f"series{(day - 1) * 10 + m + 1:03d}"
                lengths = np.clip(rng.normal(design.cell_length_mean_um, design.cell_length_sd_um,
                                             design.cells_per_movie), lo, hi)
                cells = place_cells(rng, lengths, width_um=design.cell_width_um, field_um=design.field_um)
                share = nucleoid_share(cells, dims=spec.dims)
                movie_factor, cell_factors, cell_counts = _biology(design, spec, bio_rng, cells)
                rates = None if spec.free_only else rates_for(spec, nucleoid_share=share,
                                                               membrane_share=membrane_share(
                                                                   cells, spec.membrane_shell_um),
                                                               fork_share=_fork_share_of(spec, cells),
                                                               bound_factor=factor * movie_factor)
                after, induction_kw = None, {}
                if ind is not None:
                    after, induction_kw, movies_truth[movie] = _movie_induction(
                        spec, onset_rng, m, day_offset, factor * movie_factor)
                tracks = simulate_movie(rng, cells, design=design, spec=spec, rates=rates, keep_short=True,
                                        rates_after=after, cell_factors=cell_factors, cell_counts=cell_counts,
                                        stuck=stuck_level(design, spec, day), **induction_kw)
                # The TRUTH is the true blurred centroid (no localisation error added: the photons set it).
                table = pd.DataFrame([
                    {"track_id": t["track"], "frame": f, "x": x / um, "y": y / um, "cell": t["cell"],
                     "true_state": STATES.index(st) + 1, "x_um": x, "y_um": y, "state": st, "in_nucleoid": nu,
                     **{column: values[k] for column, values in t.get("extra", {}).items()}}
                    for t in tracks
                    for k, (f, x, y, st, nu) in enumerate(zip(t["frames"], t["x_true_um"], t["y_true_um"],
                                                              t["states"], t["in_nucleoid"]))])
                # MOTION BLUR AS THE CAMERA RECORDS IT: each sub-step a spot with its share of the frame's
                # photons, so a fast molecule is smeared and dimmed (`MOVIE_RENDERING`).
                share = photons.photons_per_frame / design.exposed_subframes
                render_seed = int(rng.integers(2 ** 31))
                extras, backdrop = {}, {}
                if realism is None:
                    sub = pd.DataFrame([
                        {"track_id": t["track"], "frame": f, "x": sx / um, "y": sy / um, "photons": share}
                        for t in tracks
                        for f, xs_, ys_ in zip(t["frames"], t["sub_x_um"], t["sub_y_um"])
                        for sx, sy in zip(xs_, ys_)])
                else:
                    # THE IMPERFECTIONS, from their own stream: the same molecules as the clean copy
                    real_rng = np.random.default_rng([design.seed, 7, len(cell_rows), day])
                    illum = realism.illumination(design.field_px[0], design.field_px[1])
                    sub_rows, extras = _realistic_rows(tracks, cells, spec=spec, design=design,
                                                       realism=realism, illum=illum, share=share,
                                                       rng=real_rng)
                    sub = pd.DataFrame(sub_rows, columns=["track_id", "frame", "x", "y", "photons", "sigma_px"])
                    half = realism.autofluorescence_half_life_frames
                    fade = (np.ones(design.n_frames) if not half else
                            np.power(0.5, np.arange(design.n_frames) / float(half)))
                    cv = float(realism.autofluorescence_cell_cv)
                    sd_log = math.sqrt(math.log(1.0 + cv * cv))
                    af_weight = {c.cell_id: float(math.exp(real_rng.normal(-0.5 * sd_log ** 2, sd_log)))
                                 for c in cells}
                    backdrop = {"background_map": illum * (photons.background_photons_per_px
                                                           + realism.haze_photons_per_px
                                                           * _cell_mask(cells, design)),
                                "extra_background": (realism.autofluorescence_photons_per_px * illum
                                                     * _cell_mask(cells, design, af_weight), fade)}
                stack, rendered = render_movie(sub, optics=optics, camera=camera, photons=photons,
                                               field_shape=(design.field_px[1], design.field_px[0]),
                                               margin_px=0.0, seed=render_seed,
                                               centre=False, n_frames=design.n_frames, **backdrop)
                # `.stk`, MetaMorph's own layout with a stamp per plane (`write_stk`): the preparation stage takes
                # `.nd2`/`.stk` as an ACQUISITION (a plain `.tif` is ambiguous with a phase snapshot,
                # `rawmovie.ACQUISITION_SUFFIXES`), and the stamps are the recording clock
                write_stk(group_dir / "raw" / f"{movie}.stk", stack, start=starts[m], tau_s=design.frame_interval_s,
                          exposure_s=design.exposure_s, em_gain=cal.get("em_gain"))
                recorded[movie] = starts[m].isoformat(timespec="milliseconds")
                (group_dir / "phase").mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(group_dir / "phase" / f"{movie}.tif", render_phase(cells, design, rng),
                                 photometric="minisblack")
                outline_dir = root / "truth" / "outlines" / name
                outline_dir.mkdir(parents=True, exist_ok=True)
                write_cell_meshes(outline_dir / f"{movie}.mat",
                                  [CellMesh(movie=movie, cell_id=str(c.cell_id), polygon=c.outline() / um,
                                            mesh=c.ribs() / um) for c in cells],
                                  image_shape=(design.field_px[1], design.field_px[0]), pixel_origin=0)
                crops[movie] = [1, design.n_frames]
                day_map[movie] = day
                day_states += list(table["state"])
                kept = set(zip(rendered["track_id"], rendered["frame"]))
                for r in table.itertuples(index=False):
                    z, dark = extras.get((r.track_id, r.frame), (0.0, False))
                    rows.append((day, movie, r.cell, r.track_id, r.frame, r.x_um, r.y_um, r.x, r.y, r.state,
                                 r.in_nucleoid, (r.track_id, r.frame) in kept, z, dark))
                # SynK's per-localisation truth (time since the induction, onset, response, true D)
                for column in (c for t in tracks[:1] for c in t.get("extra", {})):
                    extra_cols.setdefault(column, []).extend(table[column].tolist())
                for c in cells:
                    cell_rows.append({"movie_bound_factor": movie_factor,
                                      "cell_bound_factor": (cell_factors or {}).get(c.cell_id, 1.0),
                                      "molecules": (cell_counts or {}).get(
                                          c.cell_id, spec.molecules_per_cell or design.molecules_per_cell),
                                      "day": day, "movie": movie, "cell": c.cell_id, "cx_um": c.cx_um,
                                      "cy_um": c.cy_um, "angle_rad": c.angle, "length_um": c.length_um,
                                      "width_um": c.width_um, "nucleoid_share": _cell_nucleoid_share(c, spec.dims)})
                say(f"{name} {movie} (replicate {day}): {len(tracks)} molecules, {len(table):,} localisations "
                    f"rendered into {design.n_frames} frames")
            group_truth["days"][str(day)] = {"bound_rate_factor": factor,
                                             "occupancy_localisations": _occupancy(day_states),
                                             "n_localisations": len(day_states),
                                             **({"stuck_share": stuck_level(design, spec, day)}
                                                if spec.stuck_range is not None else {})}
        if spec.stuck_range is not None:
            group_truth["stuck_range"] = [float(v) for v in spec.stuck_range]
        group_truth["recorded"] = recorded
        if ind is not None:
            group_truth["induction"] = {
                **asdict(ind), "movies": movies_truth,
                "additions": {str(day): _dt.datetime.combine(MOVIE_FIRST_DATE + _dt.timedelta(weeks=day - 1),
                                                             MOVIE_CLOCK).isoformat()
                              for day in range(1, design.days + 1)},
                "rule": "h = the shape at (t - onset); D(h) = before + (after - before) h; rates(h) = before + "
                        "(after - before) h; onset = max(0, delay + replicate + movie + cell + molecule offsets); "
                        "t = seconds since the replicate's addition"}
        truth["groups"][name] = group_truth
        frame = pd.DataFrame(rows, columns=["day", "movie", "cell", "track", "frame", "x_um", "y_um", "x_px",
                                            "y_px", "true_state", "in_nucleoid", "rendered", "z_um", "dark"])
        for column, values in extra_cols.items():
            if len(values) != len(frame):
                raise SyntheticError(f"{name}: the truth column {column!r} is missing for some tracks")
            frame[column] = values
        with gzip.open(root / "truth" / f"{name}_localisations.csv.gz", "wt", encoding="utf-8") as fh:
            frame.to_csv(fh, index=False)
        pd.DataFrame(cell_rows).to_csv(root / "truth" / f"{name}_cells.csv", index=False)
        (group_dir / "bleaching").mkdir(parents=True, exist_ok=True)
        (group_dir / "bleaching" / "crops.json").write_text(json.dumps({
            "crops": crops, "pixel_size_nm": design.pixel_size_nm,
            "frame_interval_s": design.frame_interval_s, "exposure_s": design.exposure_s, "label": "mNeonGreen", "emission_nm": 517.0,
            "numerical_aperture": 1.49,
            "note": ["SYNTHETIC (SynR movies): the whole movie is the crop"]}, indent=2), encoding="utf-8")
        (group_dir / "day_map.json").write_text(json.dumps({
            "movie_to_day": day_map, "source": "synthetic", "series_per_day": 10, "warnings": [],
            "written_by": "smtracker.synthetic.build_movie_project"}, indent=2), encoding="utf-8")
    (root / "truth" / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth
