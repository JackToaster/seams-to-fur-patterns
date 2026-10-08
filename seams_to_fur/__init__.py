from . import (
    preferences,
    properties,
)
from .operators import appearance, distortion, export_dxf, export_svg, flatten, preview, seam_curve
from .ui import operators_menu, panels

_modules = (
    properties,
    preferences,
    seam_curve,
    flatten,
    distortion,
    preview,
    appearance,
    export_svg,
    export_dxf,
    panels,
    operators_menu,
)


def register():
    for mod in _modules:
        mod.register()


def unregister():
    for mod in reversed(_modules):
        mod.unregister()
