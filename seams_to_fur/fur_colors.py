"""Faux-fur color palette, sourced from a real fabric vendor's physical
color-card samples (Big Z Fabric's EcoShag faux fur line) so the swatches in
the Fur Colors panel correspond to colors actually orderable for the
finished garment, not arbitrary made-up preview colors. RGB values are
visual approximations read off photos of the physical swatches (lighting/
white-balance varied across photos), not colorimeter-measured dye codes -
close enough for an in-viewport preview, not for print/dye matching.

kept bpy-free (plain tuples) so it's importable from both operators/ and
ui/ without a circular import, and so the list itself stays easy to skim/
edit without wading through UI or operator code.
"""

FUR_COLOR_SWATCHES = (
    # Neutrals / browns
    ("White", (0.94, 0.93, 0.89)),
    ("Ivory", (0.93, 0.89, 0.80)),
    ("Platinum", (0.78, 0.76, 0.72)),
    ("Latte", (0.79, 0.66, 0.46)),
    ("Blonde", (0.85, 0.66, 0.33)),
    ("Oyster", (0.58, 0.55, 0.49)),
    ("Gray", (0.33, 0.33, 0.30)),
    ("Grey Frost", (0.66, 0.66, 0.63)),
    ("Charcoal", (0.14, 0.14, 0.14)),
    ("Pewter", (0.29, 0.25, 0.19)),
    ("Black", (0.04, 0.04, 0.04)),
    ("Brown", (0.14, 0.09, 0.06)),
    ("Café", (0.18, 0.11, 0.06)),
    ("Saddle", (0.48, 0.23, 0.08)),
    ("Camel", (0.77, 0.57, 0.24)),
    ("Cocoa", (0.42, 0.26, 0.13)),
    ("Rust", (0.54, 0.22, 0.07)),
    ("Desert Tan", (0.87, 0.72, 0.53)),
    ("Light Brown", (0.29, 0.19, 0.09)),
    # Reds / pinks
    ("Red", (0.87, 0.10, 0.10)),
    ("Scarlet Red", (0.77, 0.07, 0.16)),
    ("Maroon", (0.29, 0.05, 0.08)),
    ("Burgundy", (0.31, 0.06, 0.15)),
    ("Watermelon", (0.94, 0.25, 0.48)),
    ("Pink Lemonade", (0.94, 0.66, 0.72)),
    ("Light Pink", (0.96, 0.78, 0.82)),
    ("Blush", (0.89, 0.80, 0.75)),
    ("Dusty Rose", (0.69, 0.38, 0.46)),
    ("Bubble Gum", (0.91, 0.35, 0.63)),
    ("Fuchsia", (0.91, 0.07, 0.56)),
    # Oranges / yellows
    ("Orange", (0.94, 0.47, 0.0)),
    ("Tangerine", (0.96, 0.33, 0.04)),
    ("Amber", (0.66, 0.32, 0.11)),
    ("Mango", (0.94, 0.58, 0.12)),
    ("Sherbet Orange", (0.94, 0.69, 0.56)),
    ("Papaya", (0.94, 0.69, 0.56)),
    ("Golden Yellow", (0.96, 0.83, 0.0)),
    ("Gold", (0.85, 0.60, 0.09)),
    ("Saffron", (0.83, 0.63, 0.09)),
    ("Sunshine", (0.96, 0.94, 0.63)),
    # Greens
    ("Mint", (0.44, 0.94, 0.75)),
    ("Lime Green", (0.44, 0.90, 0.13)),
    ("Kelly Green", (0.12, 0.62, 0.24)),
    ("Spring Green", (0.30, 0.76, 0.18)),
    ("Seafoam", (0.24, 0.80, 0.55)),
    ("Hunter Green", (0.08, 0.19, 0.11)),
    ("Olive", (0.54, 0.54, 0.16)),
    ("Wasabe Green", (0.66, 0.72, 0.24)),
    # Blues
    ("Aqua", (0.31, 0.76, 0.69)),
    ("Turquoise", (0.12, 0.62, 0.55)),
    ("Teal", (0.08, 0.41, 0.41)),
    ("Electric Blue", (0.12, 0.72, 0.88)),
    ("Baby Blue", (0.66, 0.79, 0.84)),
    ("Dusty Blue", (0.43, 0.49, 0.57)),
    ("Shadow Blue", (0.43, 0.46, 0.53)),
    ("Cobalt", (0.10, 0.37, 0.66)),
    ("Royal Blue", (0.10, 0.18, 0.63)),
    ("Navy Blue", (0.04, 0.12, 0.20)),
    ("Denim", (0.14, 0.22, 0.29)),
    # Purples
    ("Lavender", (0.72, 0.66, 0.75)),
    ("Grape", (0.42, 0.10, 0.43)),
    ("Purple", (0.17, 0.06, 0.24)),
    ("Violet", (0.48, 0.31, 0.66)),
    ("Eggplant", (0.10, 0.06, 0.13)),
    ("Midnight Purple", (0.10, 0.06, 0.18)),
    ("Periwinkle", (0.42, 0.37, 0.80)),
)


def nearest_swatch_name(color):
    """Nearest FUR_COLOR_SWATCHES entry to color (an (r, g, b[, a]) tuple)
    by plain RGB Euclidean distance - used to annotate a fabrication
    export with which real, orderable fabric a piece's (possibly hand-
    picked, not necessarily an exact swatch) color is closest to."""
    r, g, b = color[0], color[1], color[2]
    best_name, best_dist = None, None
    for name, (sr, sg, sb) in FUR_COLOR_SWATCHES:
        dist = (r - sr) ** 2 + (g - sg) ** 2 + (b - sb) ** 2
        if best_dist is None or dist < best_dist:
            best_name, best_dist = name, dist
    return best_name
