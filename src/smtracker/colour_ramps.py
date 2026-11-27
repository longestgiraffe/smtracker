"""Custom colour ramps for the maps: built from a few colours, checked, saved and shared.

Asked for: "we add viridis as well to the colour maps, also we do a custom setting where
users can use custom colors and save and load". The built-in ramps (jet, magma, viridis)
live with the chart; this module holds everything about a USER's ramp that is not drawing:

- `CustomRamp` -- a name and 2 to 5 colour stops, validated (`RampError` names what is
  wrong, never a silent fix).
- `sample(stops)` -- the ramp the maps paint, interpolated in CIELAB between the stops,
  so lightness changes evenly between two stops instead of bunching where sRGB is
  non-linear.
- `lightness(ramp)` / `brightness_note(ramp)` -- CIE L* along the ramp, and a WARNING (never
  a refusal) when it does not rise or fall steadily. That is jet's cost, recorded in
  `spatialsqd.DEFAULT_CMAP`'s note: where brightness turns around, two different densities
  print at the same brightness and the map does not survive greyscale. Which ramp is used
  is the user's call; what it costs is said.
- `ramps_to_json` / `ramps_from_json`, `export_ramps` / `import_ramps` -- the per-user store
  (the window keeps it in `ui.json`, beside the other remembered choices) and a small file
  to hand to a colleague.

A custom ramp's key is ``"custom:<name>"`` everywhere a colormap key travels, so the
caption can say which ramp drew a map.

ONE BUILT-IN RAMP IS COMPUTED HERE, NOT SAMPLED BY HAND: v2's SDA ramp (`v2_hot_ramp`,
key `V2_HOT_KEY`), MATLAB's ``flip(hot(1024))`` from MATLAB's own formula (`matlab_hot`).
Asked for (13 Oct 2026, the SDA maps): "with same color map like v2".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CUSTOM_PREFIX", "MAX_STOPS", "MIN_STOPS", "RAMP_SAMPLES", "CustomRamp",
           "RampError", "brightness_note", "custom_key", "export_ramps", "import_ramps",
           "is_custom_key", "lightness", "name_of_key", "ramps_from_json", "ramps_to_json",
           "sample", "V2_HOT_KEY", "V2_HOT_N", "matlab_hot", "v2_hot_ramp"]

CUSTOM_PREFIX = "custom:"
MIN_STOPS, MAX_STOPS = 2, 5
#: Samples per ramp: the same 32 the built-in ramps carry, so a custom ramp paints alike.
RAMP_SAMPLES = 32
#: Lightness steps smaller than this (L* units) are not called a turn: rounding to 8-bit
#: colour alone moves L* by up to ~0.4.
_TOLERANCE = 0.5
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")
#: Names the built-in ramps already use; a custom ramp may not shadow one.
RESERVED_NAMES = frozenset({"jet", "magma", "viridis", "hot_r"})
_FILE_KIND = "smtracker-colour-ramps"


class RampError(ValueError):
    """A ramp that cannot be used as given, with what is wrong."""


def _clean_hex(value: str) -> str:
    match = _HEX.match(str(value).strip())
    if not match:
        raise RampError(f"{value!r} is not a colour (#rrggbb)")
    return "#" + match.group(1).lower()


@dataclass(frozen=True)
class CustomRamp:
    """A user's ramp: a name and 2-5 colour stops, low to high."""

    name: str
    stops: tuple

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise RampError("a ramp needs a name")
        if len(name) > 32:
            raise RampError(f"{name!r}: a name of at most 32 characters")
        if name.lower() in RESERVED_NAMES:
            raise RampError(f"{name!r} is a built-in ramp's name")
        stops = tuple(_clean_hex(s) for s in self.stops)
        if not MIN_STOPS <= len(stops) <= MAX_STOPS:
            raise RampError(f"{name!r}: {MIN_STOPS} to {MAX_STOPS} colours, got {len(stops)}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "stops", stops)

    @property
    def key(self) -> str:
        return custom_key(self.name)

    def ramp(self) -> tuple:
        return sample(self.stops)


def custom_key(name: str) -> str:
    return f"{CUSTOM_PREFIX}{name}"


def is_custom_key(key: str) -> bool:
    return str(key).startswith(CUSTOM_PREFIX)


def name_of_key(key: str) -> str:
    return str(key)[len(CUSTOM_PREFIX):] if is_custom_key(key) else str(key)


# ----------------------------------------------------------------- colour arithmetic
_WHITE = (0.95047, 1.0, 1.08883)          # D65


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _to_gamma(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _f(t: float) -> float:
    return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116


def _f_inv(t: float) -> float:
    return t ** 3 if t ** 3 > 216 / 24389 else (116 * t - 16) / (24389 / 27)


def _hex_to_lab(value: str) -> tuple:
    v = _clean_hex(value)
    r, g, b = (_to_linear(int(v[i:i + 2], 16) / 255) for i in (1, 3, 5))
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    fx, fy, fz = _f(x / _WHITE[0]), _f(y / _WHITE[1]), _f(z / _WHITE[2])
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _lab_to_hex(lab: tuple) -> str:
    lum, a, b = lab
    fy = (lum + 16) / 116
    x, y, z = (_WHITE[0] * _f_inv(fy + a / 500), _WHITE[1] * _f_inv(fy),
               _WHITE[2] * _f_inv(fy - b / 200))
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return "".join(f"{round(_to_gamma(c) * 255):02x}" for c in (r, g, bl))


def sample(stops, n: int = RAMP_SAMPLES) -> tuple:
    """The ramp as ``n`` hex colours without '#', the format the chart's ramps use,
    interpolated in CIELAB between evenly spaced stops."""
    labs = [_hex_to_lab(s) for s in stops]
    if len(labs) < MIN_STOPS:
        raise RampError(f"{MIN_STOPS} colours at least, got {len(labs)}")
    out = []
    segments = len(labs) - 1
    for i in range(n):
        pos = i / (n - 1) * segments
        k = min(int(pos), segments - 1)
        frac = pos - k
        a, b = labs[k], labs[k + 1]
        out.append(_lab_to_hex(tuple(a[j] + (b[j] - a[j]) * frac for j in range(3))))
    return tuple(out)


def lightness(ramp) -> list:
    """CIE L* (0 black .. 100 white) of every colour in ``ramp`` (hex, with or without '#')."""
    return [_hex_to_lab(c if str(c).startswith("#") else f"#{c}")[0] for c in ramp]


def brightness_note(ramp) -> str | None:
    """None when L* rises or falls steadily along the ramp; otherwise what it costs."""
    values = lightness(ramp)
    steps = [b - a for a, b in zip(values, values[1:])]
    up = sum(1 for s in steps if s > _TOLERANCE)
    down = sum(1 for s in steps if s < -_TOLERANCE)
    if not up or not down:
        return None
    against = min(up, down)
    return (f"brightness turns around along this ramp ({against} of {len(steps)} steps run "
            f"against the rest): two different values can print at the same brightness, and "
            f"it will not survive greyscale; this is jet's cost")


# ------------------------------------------------------------- v2's SDA ramp (hot)
#: v2's SDA maps (`plotHeatmap.m`: ``colormap(ax, flip(hot(1024)))``): MATLAB's ``hot``
#: REVERSED -- white at zero, through yellow and red, to black at the top. Its key
#: wherever a colormap key travels (matplotlib's name for the same direction).
V2_HOT_KEY = "hot_r"
#: The length v2 asked MATLAB for.
V2_HOT_N = 1024


def matlab_hot(m: int) -> list:
    """MATLAB's ``hot(m)`` as its own source builds it: ``n = fix(3/8*m)``; red rises
    over the first n entries (from 1/n, not from 0), green over the next n, blue over
    the remaining m - 2n. So black-red-yellow-white run over 3/8, 3/8 and 1/4 of the
    ramp, NOT thirds, and the first colour is a red of 1/n rather than black.

    matplotlib's ``hot`` is this formula at m = 64 (its first red is 1/24 = 0.0416): over
    1024 entries it differs from MATLAB's by at most 0.039 per channel (about 10 of 255),
    at the dark end, and by 0.007 on average -- measured, `tests/test_spatial_sda.py`.
    Close, but this is the function v2 called, so it is used rather than matplotlib's.
    Returns ``m`` (r, g, b) tuples in 0..1, low to high."""
    m = int(m)
    if m < 4:
        raise RampError(f"hot needs at least 4 entries, got {m}")
    n = (3 * m) // 8
    rest = m - 2 * n
    out = []
    for i in range(m):
        r = (i + 1) / n if i < n else 1.0
        g = 0.0 if i < n else ((i - n + 1) / n if i < 2 * n else 1.0)
        b = 0.0 if i < 2 * n else (i - 2 * n + 1) / rest
        out.append((r, g, b))
    return out


def v2_hot_ramp(n: int = RAMP_SAMPLES) -> tuple:
    """v2's SDA ramp, ``flip(hot(1024))``, sampled at ``n`` evenly spaced entries as hex
    without '#' (the chart's ramp format): ``"ffffff"`` (white, the bottom of the scale)
    first, ``"010000"`` (MATLAB's own first red, 1/384: black to the eye) last."""
    if n < 2:
        raise RampError(f"a ramp needs at least 2 samples, got {n}")
    full = matlab_hot(V2_HOT_N)[::-1]
    picks = [full[round(i * (V2_HOT_N - 1) / (n - 1))] for i in range(n)]
    return tuple("".join(f"{int(c * 255 + 0.5):02x}" for c in rgb) for rgb in picks)


# ---------------------------------------------------------------- store and share
def ramps_to_json(ramps) -> list:
    return [{"name": r.name, "stops": list(r.stops)} for r in ramps]


def ramps_from_json(data) -> list:
    """Ramps out of their JSON form. A malformed entry raises `RampError` naming it, and
    two ramps under one name are refused: one key would mean two pictures."""
    if not isinstance(data, list):
        raise RampError("a list of ramps was expected")
    ramps, seen = [], set()
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "name" not in item or "stops" not in item:
            raise RampError(f"entry {i + 1} is not a ramp (needs a name and stops)")
        ramp = CustomRamp(item["name"], tuple(item["stops"]))
        if ramp.name.lower() in seen:
            raise RampError(f"two ramps are called {ramp.name!r}")
        seen.add(ramp.name.lower())
        ramps.append(ramp)
    return ramps


def export_ramps(path, ramps) -> Path:
    """Write ``ramps`` to a small JSON file a colleague can import."""
    path = Path(path)
    path.write_text(json.dumps({"kind": _FILE_KIND, "ramps": ramps_to_json(ramps)},
                               indent=2), encoding="utf-8")
    return path


def import_ramps(path) -> list:
    """The ramps in a file `export_ramps` wrote. Anything else is refused by name."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RampError(f"{path.name}: not readable as a ramp file ({exc})") from exc
    if not isinstance(data, dict) or data.get("kind") != _FILE_KIND:
        raise RampError(f"{path.name}: not a colour-ramp file written by SMTracker")
    return ramps_from_json(data.get("ramps"))
