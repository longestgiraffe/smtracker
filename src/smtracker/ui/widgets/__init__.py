"""The widgets a page assembles: surfaces, controls, and the tooltip over them.

`primitives` is what everything else is built on — `Surface` and its `Card`,
the shadow host that paints beneath them, and the animation and layout helpers.
`controls` is what you press: `Button`, `Segmented`, `Slider`, `NumberField`.
`tooltip` is the one surface that outlives the widget it describes.

Every module here imports Qt, and every module here reads `..theme` rather than
holding a colour or a size of its own. That direction is the whole arrangement:
a widget consumes the design system, the design system knows nothing about
widgets, and a screen consumes both.

Nothing is re-exported. These are three substantial modules rather than a
namespace, and a page that wants a `Card` should say which module it came
from::

    from ..widgets.primitives import Card
    from ..widgets.controls import Button
"""
from __future__ import annotations

__all__ = ["controls", "primitives", "tooltip"]
