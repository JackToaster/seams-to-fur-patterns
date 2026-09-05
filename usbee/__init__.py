from . import (
    preferences,
    properties,
)
from .operators import appearance, distortion, export_svg, flatten, seam_curve
from .ui import operators_menu, panels

_modules = (
    properties,
    preferences,
    seam_curve,
    flatten,
    distortion,
    appearance,
    export_svg,
    panels,
    operators_menu,
)


def register():
    for mod in _modules:
        mod.register()


def unregister():
    for mod in reversed(_modules):
        mod.unregister()
