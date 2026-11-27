"""COLOUR SCIENCE FOR THE PALETTES, and the palettes a user makes (6 Nov 2026). Headless: numpy, no Qt.

Asked: "what happens if data set loads with more conditions than the palettes have colors? we should add a safeguard
for that. we should also add a color palette maker. also i noticed it would be good to expand our current palettes.
or to manually adjust some colors".

THE MEASUREMENT is the one `tests/test_palette_gates.py` anchors the shipped palettes on, written here without
scikit-image (not a dependency of the app): CIEDE2000 between CIELAB colours, and red-green vision simulated with
Machado, Oliveira & Fernandes (2009) severity-1 matrices on linear sRGB, deuteranopia AND protanopia. `gates` gives a
palette's scores against the bar every built-in palette clears (`hcl.MIN_NORMAL_DELTA_E` for normal vision and
`hcl.MIN_REDGREEN_DELTA_E` red-green, over its first three and first four colours, on the light and the dark page, and
every colour `hcl.MIN_CONTRAST` : 1 against the page once fitted); `closest` the closest pair of a whole set.

* `distinct(count, taken, dark)` -- colours as far as possible (CIEDE2000) from every colour taken and from each other,
  fitted to the page: what a condition past a palette's end gets (`condition_colours.extra_colours`), and the palette
  maker's "Add a distinct colour".
* `best_order(colours)` -- the order whose first three and four clear the gates best, the rest each as far from those
  before it as can be (the palette maker's "Order for the checks"; how the built-in palettes' orders were chosen).
* USER PALETTES -- kept with the user's settings (ui.json `USER_KEY`), registered into `hcl.PALETTES` under
  ``user-<id>`` keys so every place that lists or reads a palette has them (`ensure_registered`); `user_palettes`,
  `save_user_palette`, `delete_user_palette`.
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = ["lab", "ciede2000", "gates", "closest", "distinct", "best_order", "check_lines", "USER_KEY",
           "user_palettes", "save_user_palette", "delete_user_palette", "ensure_registered", "is_user"]

#: Machado, Oliveira & Fernandes (2009), severity 1, on linear sRGB.
MACHADO = {
    "prot": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                      [-0.003882, -0.048116, 1.051998]]),
    "deut": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                      [-0.011820, 0.042940, 0.968881]]),
}
_D65 = np.array([0.95047, 1.0, 1.08883])
_RGB_TO_XYZ = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]])
USER_KEY = "user_palettes"
USER_PREFIX = "user-"


# -- the measurement ---------------------------------------------------------------------------------------- #
def _rgb(colours) -> np.ndarray:
    return np.array([[int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for c in colours], dtype=float)


def lab(colours, sim: str | None = None) -> np.ndarray:
    """CIELAB (D65) of ``#rrggbb`` colours, seen with normal vision or simulated deuteranopia / protanopia (``sim``
    ``"deut"`` / ``"prot"``)."""
    c = _rgb(colours)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    if sim:
        lin = np.clip(lin @ MACHADO[sim].T, 0.0, 1.0)
    xyz = (lin @ _RGB_TO_XYZ.T) / _D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[:, 1] - 16.0, 500.0 * (f[:, 0] - f[:, 1]), 200.0 * (f[:, 1] - f[:, 2])], axis=1)


def ciede2000(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIEDE2000 between rows of ``a`` and ``b`` (CIELAB), broadcast (Sharma, Wu & Dalal 2005)."""
    L1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
    L2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(C1p * C2p == 0.0, 0.0, dh)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2.0))
    Lpm, Cpm = (L1 + L2) / 2.0, (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hpm = np.where(np.abs(h1p - h2p) > 180.0, np.where(hsum < 360.0, hsum + 360.0, hsum - 360.0), hsum) / 2.0
    hpm = np.where(C1p * C2p == 0.0, hsum, hpm)
    T = (1.0 - 0.17 * np.cos(np.radians(hpm - 30.0)) + 0.24 * np.cos(np.radians(2.0 * hpm))
         + 0.32 * np.cos(np.radians(3.0 * hpm + 6.0)) - 0.20 * np.cos(np.radians(4.0 * hpm - 63.0)))
    dtheta = 30.0 * np.exp(-(((hpm - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1.0 + 0.015 * (Lpm - 50.0) ** 2 / np.sqrt(20.0 + (Lpm - 50.0) ** 2)
    Sc, Sh = 1.0 + 0.045 * Cpm, 1.0 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def _worst(colours, sim=None) -> float:
    if len(colours) < 2:
        return float("inf")
    labs = lab(colours, sim)
    i, j = np.triu_indices(len(colours), 1)
    return float(ciede2000(labs[i], labs[j]).min())


def closest(colours, sim=None) -> tuple:
    """``(ΔE, i, j)``: the closest pair of ``colours`` (CIEDE2000, normal vision or ``sim``); ``(inf, -1, -1)`` for
    fewer than two."""
    if len(colours) < 2:
        return float("inf"), -1, -1
    labs = lab(colours, sim)
    i, j = np.triu_indices(len(colours), 1)
    d = ciede2000(labs[i], labs[j])
    k = int(np.argmin(d))
    return float(d[k]), int(i[k]), int(j[k])


def _fitted(colours, dark: bool) -> list:
    from .ui.theme import hcl

    ground = hcl.GROUNDS["dark" if dark else "light"]
    return [hcl.fit_to_ground(c.lower(), ground) for c in colours]


def gates(colours) -> dict:
    """A palette's scores against the bar every built-in palette clears: ``normal`` and ``red_green`` (the worst pair
    over the first three and first four colours, fitted to each page), ``contrast`` (every colour 3 : 1 on both
    pages once fitted), ``closest`` (the closest pair over ALL its colours, fitted to the light page) and ``passes``."""
    from .ui.theme import hcl

    normal, red_green = [], []
    for dark in (False, True):
        fitted = _fitted(colours, dark)
        for n in (3, 4):
            part = fitted[:n]
            if len(part) >= 2:
                normal.append(_worst(part))
                red_green.append(min(_worst(part, "deut"), _worst(part, "prot")))
    contrast = all(hcl.contrast(c, hcl.GROUNDS["dark" if dark else "light"]) >= hcl.MIN_CONTRAST
                   for dark in (False, True) for c in _fitted(colours, dark))
    out = {"normal": min(normal) if normal else float("inf"),
           "red_green": min(red_green) if red_green else float("inf"),
           "contrast": contrast, "closest": closest(_fitted(colours, False))}
    out["passes"] = bool(out["normal"] >= hcl.MIN_NORMAL_DELTA_E and out["red_green"] >= hcl.MIN_REDGREEN_DELTA_E
                         and contrast)
    return out


def check_lines(colours) -> list:
    """The palette maker's words for `gates`: one line per check, said as passing or below the bar."""
    from .ui.theme import hcl

    if len(colours) < 2:
        return ["Add at least two colours."]
    g = gates(colours)
    lines = [f"First three and four colours, normal vision: closest ΔE {g['normal']:.1f} "
             f"({'passes' if g['normal'] >= hcl.MIN_NORMAL_DELTA_E else 'below'} the built-in palettes' "
             f"{hcl.MIN_NORMAL_DELTA_E:.0f})",
             f"First three and four colours, red-green colour blindness: closest ΔE {g['red_green']:.1f} "
             f"({'passes' if g['red_green'] >= hcl.MIN_REDGREEN_DELTA_E else 'below'} "
             f"{hcl.MIN_REDGREEN_DELTA_E:.0f})",
             "Every colour 3 : 1 against the page once fitted: " + ("yes" if g["contrast"] else "no")]
    d, i, j = g["closest"]
    lines.append(f"Closest pair of all {len(colours)}: colours {i + 1} and {j + 1}, ΔE {d:.1f}"
                 + ("; hard to tell apart" if d < DISTINCT_FLOOR else ""))
    return lines


#: Below this CIEDE2000 two colours of one chart are hard to tell apart at the sizes drawn here (`condition_colours`
#: says so where a palette runs out; the palette maker says it of its closest pair). The closest pair over a whole
#: palette, fitted to the light page: Okabe-Ito 15.5, Petroff 10 13.0, Petroff 8 18.2, Tol vibrant 14.9 -- and Tol
#: muted 4.7, two of its last colours (its first four, which the gates measure, are far apart).
DISTINCT_FLOOR = 8.0


# -- choosing colours ----------------------------------------------------------------------------------------- #
def _candidates(dark: bool) -> list:
    """Page-fitted candidate colours: hues every 5 degrees at four lightnesses and two chromas, distinct."""
    from .ui.theme import hcl

    ground = hcl.GROUNDS["dark" if dark else "light"]
    out, seen = [], set()
    for lum in ((45.0, 55.0, 65.0, 75.0) if not dark else (55.0, 65.0, 75.0, 85.0)):
        for chroma in (55.0, 80.0):
            for hue in range(0, 360, 5):
                colour = hcl.fit_to_ground(hcl.to_hex(lum, chroma, float(hue)), ground)
                if colour not in seen:
                    seen.add(colour)
                    out.append(colour)
    return out


def distinct(count: int, *, taken=(), dark: bool = False) -> list:
    """``count`` colours, each in turn the candidate farthest (CIEDE2000) from every colour taken and chosen so far,
    fitted to the page; deterministic. Never repeats a colour."""
    if count <= 0:
        return []
    candidates = _candidates(dark)
    cand_lab = lab(candidates)
    chosen = [c.lower() for c in taken]
    seen = set(chosen)
    if chosen:
        nearest = np.min(ciede2000(cand_lab[:, None, :], lab(chosen)[None, :, :]), axis=1)
    else:
        nearest = np.full(len(candidates), np.inf)
    out = []
    for _ in range(int(count)):
        order = np.argsort(-nearest, kind="stable")
        pick = next((int(k) for k in order if candidates[int(k)] not in seen), None)
        if pick is None:
            break
        colour = candidates[pick]
        seen.add(colour)
        out.append(colour)
        nearest = np.minimum(nearest, ciede2000(cand_lab, cand_lab[pick][None, :]))
    return out


def _distances(colours) -> list:
    """Pairwise CIEDE2000 matrices of ``colours`` fitted to each page, normal vision and both red-green simulations:
    ``[(normal, red_green)]`` per page, each an n x n array."""
    out = []
    for dark in (False, True):
        fitted = _fitted(colours, dark)
        mats = []
        for sim in (None, "deut", "prot"):
            labs = lab(fitted, sim)
            mats.append(ciede2000(labs[:, None, :], labs[None, :, :]))
        out.append((mats[0], np.minimum(mats[1], mats[2])))
    return out


def best_order(colours) -> list:
    """``colours`` reordered: the first four the ones (in their best order) whose first three and four clear the gates
    by the widest margin, then each next colour the one farthest from those before it."""
    from .ui.theme import hcl

    colours = [c.lower() for c in colours]
    if len(colours) <= 2:
        return colours
    pool = list(range(len(colours))) if len(colours) <= 12 else [colours.index(c) for c in best_spread(colours, 12)]
    mats = _distances(colours)

    def margin(order) -> float:
        worst = np.inf
        for normal, red_green in mats:
            for n in (3, 4):
                part = order[:n]
                if len(part) < 2:
                    continue
                pairs = list(itertools.combinations(part, 2))
                worst = min(worst, min(normal[a, b] for a, b in pairs) / hcl.MIN_NORMAL_DELTA_E,
                            min(red_green[a, b] for a, b in pairs) / hcl.MIN_REDGREEN_DELTA_E)
        return worst

    head_n = min(4, len(colours))
    best = None
    for combo in itertools.combinations(pool, head_n):
        for order in itertools.permutations(combo):
            if head_n == 4 and order[0] > order[1]:
                continue                         # the first two's order does not change the first three's pairs
            score = margin(order)
            if best is None or score > best[0]:
                best = (score, list(order))
    out = list(best[1])
    rest = [k for k in range(len(colours)) if k not in out]
    normal = mats[0][0]
    while rest:
        far = max(rest, key=lambda k: min(normal[k, m] for m in out))
        out.append(far)
        rest.remove(far)
    return [colours[k] for k in out]


def _min_to(colour, others) -> float:
    return float(ciede2000(lab([colour]), lab(others)).min())


def best_spread(colours, n: int) -> list:
    """The ``n`` of ``colours`` spread widest (greedy farthest-first, from the first)."""
    out = [colours[0]]
    rest = list(colours[1:])
    while rest and len(out) < n:
        far = max(rest, key=lambda c: _min_to(c, out))
        out.append(far)
        rest.remove(far)
    return out


# -- the user's palettes ---------------------------------------------------------------------------------------- #
def is_user(key: str) -> bool:
    return str(key).startswith(USER_PREFIX)


def user_palettes() -> list:
    """The palettes the user made: ``[{"key", "label", "colors": [#rrggbb, ...]}]``, in the order they were made."""
    from .ui import theme

    value = theme.remembered(USER_KEY)
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and is_user(item.get("key", "")) and len(item.get("colors") or ()) >= 2:
            out.append({"key": item["key"], "label": str(item.get("label") or item["key"]),
                        "colors": [str(c).lower() for c in item["colors"]]})
    return out


def _palette(item):
    from .ui.theme import hcl

    names = [f"colour {i + 1}" for i in range(len(item["colors"]))]
    return hcl.Palette(item["key"], item["label"], dict(zip(names, item["colors"])), tuple(names), "made by you")


_registered: set = set()


def ensure_registered() -> None:
    """Every user palette in `hcl.PALETTES` (and none that was deleted), so every list and lookup has them."""
    from .ui.theme import hcl

    mine = user_palettes()
    keys = {item["key"] for item in mine}
    for key in list(_registered - keys):
        hcl.unregister_palette(key)
        _registered.discard(key)
    for item in mine:
        current = hcl.PALETTES.get(item["key"])
        if current is None or list(current.colors.values()) != item["colors"] or current.label != item["label"]:
            hcl.register_palette(_palette(item))
        _registered.add(item["key"])


def save_user_palette(label: str, colours, key: str | None = None) -> str:
    """Save a palette (``key``: replace that one) and register it; returns its key. Refused: fewer than two colours,
    a colour twice, a malformed colour, no name."""
    import re
    import uuid

    from .ui import theme

    label = " ".join(str(label).split())
    colours = [str(c).strip().lower() for c in colours]
    if not label:
        raise ValueError("a palette needs a name")
    if len(colours) < 2:
        raise ValueError("a palette needs at least two colours")
    bad = [c for c in colours if not re.fullmatch(r"#[0-9a-f]{6}", c)]
    if bad:
        raise ValueError(f"not a colour: {', '.join(bad)}")
    if len(set(colours)) != len(colours):
        raise ValueError("a colour is in the palette twice")
    key = key if key and is_user(key) else f"{USER_PREFIX}{uuid.uuid4().hex[:8]}"
    items = [p for p in user_palettes() if p["key"] != key]
    items.append({"key": key, "label": label, "colors": colours})
    theme.remember(USER_KEY, items)
    ensure_registered()
    return key


def delete_user_palette(key: str) -> None:
    from .ui import theme

    theme.remember(USER_KEY, [p for p in user_palettes() if p["key"] != key])
    ensure_registered()
