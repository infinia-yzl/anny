# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The hairstyle presets of anny: their specs and the build of their guides.

Each preset is a function that returns a spec. ``python -m anny.hair.authoring.presets`` writes
every spec to ``data/hair/styles/<name>.json`` and grows its guides into
``data/hair/styles.safetensors``. A spec has four parts:

- ``groom``: the authoring of the guides (``anny.hair.authoring.groom``): whorl, comb flow, lift,
  length map, perimeter, cuts, relaxation;
- ``render``: the look of the render strands in the page (``anny.hair.styles.strands``): points
  per strand, width, clumps, curl, waves, flyaways, fade and hairline;
- ``physics``: the parameters of the page's solver (``anny.hair.dynamics``) per family, or None for
  a style that keeps its shape (the buns);
- ``controls``: the ranges of the page's sliders.

Usage::

    python -m anny.hair.authoring.presets [--only NAME ...] [--specs-only]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from anny.hair.chart import CURVE_PHI, encode_curves, smoothstep
from anny.hair.layout import load_layout
from anny.hair.styles import GUIDES_PATH, STYLE_DIR

LENGTH_PHI = np.linspace(0.0, 180.0, 37)
LENGTH_H = np.arange(-10.0, 135.0, 5.0)


def length_table(fn):
    """a length map (mm) over |azimuth| and the height above the hairline, from fn(a, h)"""
    A, H = np.meshgrid(LENGTH_PHI, LENGTH_H, indexing="ij")
    return dict(
        phi=LENGTH_PHI.tolist(),
        h=LENGTH_H.tolist(),
        values=np.round(fn(A, H), 2).tolist(),
    )


def curve(fn):
    """a style curve at CURVE_PHI from fn(|azimuth|)"""
    return np.round(fn(CURVE_PHI), 3).tolist()


DEFAULT_RENDER = dict(
    points=24,
    width_mm=0.12,
    tip_width=0.45,
    strand_jitter=[0.95, 1.03],
    thinning=dict(share=0.28, range=[0.55, 0.85]),
    clump=dict(
        coarse=[0.5, 0.82],
        coarse_tip=0.55,
        fine=[0.3, 0.65],
        fine_power=[1.0, 1.5],
        sectors=6,
    ),
    curl=dict(radius_mm=0.0, period_mm=20.0, ramp=0.15),
    frizz=dict(mm=[0.15, 0.55], cycles=[1.0, 2.6]),
    flyaways=dict(share=0.005, mm=[1.5, 4.5]),
    hairline=None,
    fade=None,
    density=0.88,
)
# the page's solver (viewer/src/hair/sim.ts, anny.hair.dynamics), per step of 1/60 s: the pull toward the groom
# from the free start of a guide to its tip, the pull toward the groom's bends, the share of the velocity lost, and the
# share of the change of gravity that acts
FAMILY_PHYSICS = dict(
    short=dict(
        global_stiffness=[0.8, 0.35], local_stiffness=0.8, damping=0.15, gravity=1.0
    ),
    medium=dict(
        global_stiffness=[0.35, 0.03], local_stiffness=0.5, damping=0.06, gravity=1.0
    ),
    long=dict(
        global_stiffness=[0.2, 0.01], local_stiffness=0.4, damping=0.05, gravity=1.0
    ),
    tied=dict(
        global_stiffness=[0.06, 0.004], local_stiffness=0.3, damping=0.05, gravity=1.0
    ),
)
DEFAULT_PHYSICS = FAMILY_PHYSICS["medium"]
# a bun keeps its shape: the solver skips it (a style's physics of None)
BUN_PHYSICS = False
DEFAULT_CONTROLS = dict(
    length=[0.7, 1.25],
    curl=[0.0, 1.5],
    volume=[0.85, 1.3],
    density=[0.5, 1.0],
)


def spec(name, label, family, groom, render=None, physics=None, controls=None):
    return dict(
        name=name,
        label=label,
        family=family,
        groom=groom,
        render=dict(DEFAULT_RENDER, **(render or {})),
        physics=None
        if physics is False
        else dict(FAMILY_PHYSICS.get(family, DEFAULT_PHYSICS), **(physics or {})),
        controls=dict(DEFAULT_CONTROLS, **(controls or {})),
    )


# ---------------------------------------------------------------------------------------------- presets
LEGACY_PERIMETER = dict(
    phi=[0, 22, 36, 48, 58, 66, 74, 82, 92, 98, 106, 116, 128, 145, 162, 180],
    y=[0.5372, 0.5375, 0.5362, 0.5320, 0.5260, 0.5200, 0.5170, 0.5165, 0.5160]
    + [0.5060, 0.4900, 0.4760, 0.4660, 0.4600, 0.4575, 0.4570],
)
LEGACY_FLOW = [
    dict(type="whorl", weight=1.0, spiral=1.9, radius=0.028, twist=0.12),
    dict(
        type="direction", vector=[0.0, -1.0, 0.0], weight=0.9, el=[55, 15], phi=[35, 70]
    ),
]
LEGACY_LIFT = dict(base=58, sd=8, top=-12, crown=10, front=-10, edge=12, min=25, max=82)


def medium_tousled():
    """the medium tousled cut of the legacy 3D Model build (build/hair2.py)"""

    def length(a, h):
        front = smoothstep(52, 30, a) * smoothstep(48, 30, h)
        side = smoothstep(40, 60, a)
        return 70.0 + 25.0 * front - 18.0 * side * smoothstep(30, 0, h)

    groom = dict(
        seed=11,
        whorl=dict(azimuth=168, elevation=60),
        flow=LEGACY_FLOW,
        tousle=0.28,
        lift=LEGACY_LIFT,
        length=length_table(length),
        length_jitter=[0.94, 1.06],
        length_extra_mm=10,
        perimeter=LEGACY_PERIMETER,
        below_perimeter_mm=[9, 13],
        lift_mm=5.0,
        sway=[[0.18, 1.6], [0.26, 1.4]],
        cuts=dict(
            layer_jitter=[0.72, 1.05],
            texture_mm=[7.0, 8.0, 7.0, 7.0, 13.0, 14.0],
            flick=dict(share=0.45, min_azimuth=50, mm=[1.5, 4.5]),
        ),
    )
    # an 8 cm cut: stiffer than the hanging medium cuts
    physics = dict(global_stiffness=[0.6, 0.15], local_stiffness=0.7, damping=0.1)
    return spec("medium_tousled", "Medium tousled", "medium", groom, physics=physics)


# ---------------------------------------------------------------------------------------------- helpers
def el_of(a, h):
    """elevation of a point at |azimuth| a and height h above the hairline"""
    from anny.hair.chart import hairline

    return h + hairline(a)


def front_of(a, h, reach=30):
    """the front of the head: next to the front hairline (the fringe)"""
    return smoothstep(55, 30, a) * smoothstep(reach, reach * 0.4, h)


def top_of(a, h):
    """the top of the head"""
    return smoothstep(40, 60, el_of(a, h))


def sides_of(a, h):
    """the sides and the back below the crown"""
    return smoothstep(45, 70, a) * smoothstep(55, 30, el_of(a, h))


def fade_curve(sides=None, back=None, front=-90.0, sideburn=None):
    """start of a fade band (degrees above the hairline) along |azimuth|: at the front, the
    sideburns (55-80), the sides above the ears (80-125) and the back"""
    from anny.hair.chart import CURVE_PHI

    sb = sides if sideburn is None else sideburn
    a = CURVE_PHI
    x = np.full(len(a), float(front))
    x = np.where(a >= 55, sb if sb is not None else front, x)
    x = np.where(a >= 80, sides if sides is not None else front, x)
    x = np.where(a >= 125, back if back is not None else front, x)
    # soften the steps between the regions
    k = np.array([0.25, 0.5, 0.25])
    for _ in range(2):
        x = np.convolve(np.pad(x, 1, mode="edge"), k, mode="valid")
    return np.round(x, 2).tolist()


SHORT_RENDER = dict(
    thinning=None,
    frizz=dict(mm=[0.05, 0.2], cycles=[0.5, 1.5]),
    flyaways=None,
    density=1.0,
)
CROWN_FLOW = dict(type="whorl", weight=1.0, spiral=1.9, radius=0.028, twist=0.12)
SIDES_DOWN = dict(
    type="direction", vector=[0.0, -1.0, 0.0], weight=0.9, el=[55, 15], phi=[35, 70]
)


def cuts(layer=(0.85, 1.05), texture=2.0, flick=None):
    return dict(layer_jitter=list(layer), texture_mm=[texture] * 6, flick=flick)


def short_groom(length, flow, lift, seed, tousle=0.2, sway=0.12, longest=1.4, **kw):
    g = dict(
        seed=seed,
        whorl=dict(azimuth=168, elevation=60),
        flow=flow,
        tousle=tousle,
        lift=lift,
        length=length_table(length),
        length_jitter=[0.9, 1.08],
        length_extra_mm=1,
        longest=longest,
        perimeter=None,
        lift_mm=1.0,
        sway=[[sway, 1.3]],
        cuts=cuts(),
        relax=dict(iters=100),
        layers=dict(base=0.4 * 0.001, per_guide=0.5 * 0.001),
        smooth=3,
    )
    g.update(kw)
    return g


def bald():
    return spec(
        "bald",
        "Bald",
        "short",
        dict(bald=True),
        render=dict(points=2, density=0.0),
        controls=dict(
            length=[1.0, 1.0], curl=[0.0, 0.0], volume=[1.0, 1.0], density=[0.0, 0.0]
        ),
    )


def buzz_cut():
    """a clipper cut of about 7 mm all over, with a tapered neckline"""
    groom = short_groom(
        lambda a, h: 7.0 * (0.55 + 0.45 * smoothstep(0, 8, h)),
        [CROWN_FLOW, dict(SIDES_DOWN, weight=0.6)],
        dict(base=42, sd=6, top=0, crown=0, front=-5, edge=10, min=25, max=60),
        seed=21,
        tousle=0.15,
        sway=0.05,
        longest=1.5,
    )
    render = dict(
        SHORT_RENDER,
        points=6,
        width_mm=0.11,
        clump=dict(
            coarse=[0.0, 0.15],
            coarse_tip=0.0,
            fine=[0.0, 0.2],
            fine_power=[1.0, 1.0],
            sectors=6,
        ),
        fade=dict(
            start=fade_curve(sides=None, back=0.0, sideburn=0.0),
            width=8.0,
            clipper=0.0015,
            top=0.007,
        ),
    )
    return spec(
        "buzz_cut",
        "Buzz cut",
        "short",
        groom,
        render,
        controls=dict(length=[0.3, 1.5], curl=[0.0, 0.0]),
    )


def crew_cut():
    """short on top, longest at the front, tapered on the sides and at the neck"""

    def length(a, h):
        return 20.0 + 10.0 * front_of(a, h, 40) - 8.0 * sides_of(a, h)

    groom = short_groom(
        length,
        [
            dict(CROWN_FLOW, weight=0.7),
            dict(
                type="direction",
                vector=[0.0, 0.4, 1.0],
                weight=1.2,
                el=[25, 45],
                phi=[70, 40],
            ),
            SIDES_DOWN,
        ],
        dict(base=50, sd=7, top=0, crown=5, front=-18, edge=8, min=15, max=75),
        seed=22,
    )
    render = dict(
        SHORT_RENDER,
        points=8,
        clump=dict(
            coarse=[0.2, 0.5],
            coarse_tip=0.2,
            fine=[0.1, 0.3],
            fine_power=[1.0, 1.3],
            sectors=6,
        ),
        fade=dict(
            start=fade_curve(sides=4.0, back=3.0, sideburn=2.0),
            width=14.0,
            clipper=0.001,
            top=0.011,
        ),
    )
    return spec("crew_cut", "Crew cut", "short", groom, render)


def taper_top(seed, top=55.0, sides=10.0, forward=0.8, tousle=0.45, lift=40):
    """a textured top over short sides: the tops of the fades and crops"""

    def length(a, h):
        return top + 6.0 * front_of(a, h, 40) - (top - sides) * sides_of(a, h)

    return short_groom(
        length,
        [
            dict(CROWN_FLOW, weight=0.8),
            dict(
                type="direction",
                vector=[0.0, 0.3, 1.0],
                weight=forward,
                el=[20, 40],
                phi=[75, 45],
            ),
            SIDES_DOWN,
        ],
        dict(base=lift, sd=9, top=-4, crown=8, front=-12, edge=12, min=12, max=80),
        seed=seed,
        tousle=tousle,
        sway=0.3,
        longest=1.35,
        cuts=cuts(layer=(0.75, 1.05), texture=4.0),
        band_mm=[3.0, 8.0, 6.0],
    )


TEXTURED = dict(
    SHORT_RENDER,
    points=12,
    clump=dict(
        coarse=[0.4, 0.75],
        coarse_tip=0.45,
        fine=[0.3, 0.6],
        fine_power=[1.0, 1.5],
        sectors=6,
    ),
    frizz=dict(mm=[0.15, 0.5], cycles=[0.8, 2.0]),
    thinning=dict(share=0.2, range=[0.6, 0.9]),
)


def low_taper_fade():
    """the most requested cut of 2026: a textured top, and a soft low fade at the sideburns and the neck"""
    render = dict(
        TEXTURED,
        fade=dict(
            start=fade_curve(sides=2.0, back=0.0, sideburn=0.0),
            width=10.0,
            clipper=0.0008,
            top=0.016,
        ),
    )
    return spec("low_taper_fade", "Low taper fade", "short", taper_top(31), render)


def mid_fade():
    """a textured top, and the sides faded to the skin from the middle of the head"""
    render = dict(
        TEXTURED,
        fade=dict(
            start=fade_curve(sides=7.0, back=10.0, sideburn=4.0),
            width=14.0,
            clipper=0.0004,
            top=0.018,
        ),
    )
    return spec("mid_fade", "Mid fade", "short", taper_top(32, sides=12.0), render)


def textured_crop():
    """a short textured top pushed forward to a short fringe, over a high fade"""
    groom = taper_top(33, top=38.0, sides=14.0, forward=1.6, tousle=0.4, lift=40)
    render = dict(
        TEXTURED,
        fade=dict(
            start=fade_curve(sides=12.0, back=16.0, sideburn=8.0),
            width=12.0,
            clipper=0.0005,
            top=0.014,
        ),
    )
    return spec("textured_crop", "Textured crop", "short", groom, render)


def french_crop():
    """a textured crop with a blunt fringe straight across the forehead"""
    groom = taper_top(34, top=42.0, sides=14.0, forward=2.0, tousle=0.25, lift=35)
    groom["perimeter"] = dict(
        phi=[0, 30, 45, 60, 180], y=[0.568, 0.568, 0.556, 0.40, 0.40]
    )
    groom["cuts"] = cuts(layer=(0.85, 1.05), texture=1.5)
    render = dict(
        TEXTURED,
        fade=dict(
            start=fade_curve(sides=10.0, back=14.0, sideburn=6.0),
            width=12.0,
            clipper=0.0006,
            top=0.014,
        ),
    )
    return spec("french_crop", "French crop", "short", groom, render)


def curly_top_taper():
    """curls on top over a low taper"""
    groom = taper_top(35, top=60.0, sides=12.0, forward=0.5, tousle=0.5, lift=30)
    render = dict(
        TEXTURED,
        curl=dict(radius_mm=2.2, period_mm=9.0, ramp=0.1),
        frizz=dict(mm=[0.3, 0.8], cycles=[1.5, 3.0]),
        clump=dict(
            coarse=[0.5, 0.8],
            coarse_tip=0.5,
            fine=[0.55, 0.85],
            fine_power=[0.8, 1.2],
            sectors=6,
        ),
        fade=dict(
            start=fade_curve(sides=2.0, back=0.0, sideburn=0.0),
            width=10.0,
            clipper=0.0008,
            top=0.016,
        ),
    )
    return spec("curly_top_taper", "Curly top with taper", "short", groom, render)


def side_part_taper():
    """longer top combed to the side from a part, with a low taper"""

    def length(a, h):
        return 60.0 - 44.0 * sides_of(a, h)

    groom = short_groom(
        length,
        [
            dict(CROWN_FLOW, weight=0.4),
            dict(type="part", x=0.034, weight=2.2, falloff=0.05, el=[20, 40]),
            dict(type="direction", vector=[-0.8, 0.1, -1.0], weight=1.5, el=[12, 32]),
            SIDES_DOWN,
        ],
        dict(base=62, sd=6, top=0, crown=4, front=-22, edge=8, min=20, max=80),
        seed=36,
        tousle=0.15,
        sway=0.12,
        longest=1.3,
        cuts=cuts(layer=(0.9, 1.05), texture=3.0),
    )
    render = dict(
        SHORT_RENDER,
        points=16,
        clump=dict(
            coarse=[0.5, 0.8],
            coarse_tip=0.5,
            fine=[0.3, 0.6],
            fine_power=[1.0, 1.5],
            sectors=6,
        ),
        frizz=dict(mm=[0.1, 0.3], cycles=[0.8, 1.5]),
        fade=dict(
            start=fade_curve(sides=2.0, back=0.0, sideburn=0.0),
            width=10.0,
            clipper=0.0008,
            top=0.016,
        ),
    )
    s = spec("side_part_taper", "Side part", "short", groom, render)
    s["mirror"] = True
    return s


def textured_quiff():
    """volume at the front, swept up and back, over a mid fade"""

    def length(a, h):
        front = smoothstep(60, 25, a) * smoothstep(55, 25, el_of(a, h))
        return 55.0 + 30.0 * front - 40.0 * sides_of(a, h)

    groom = short_groom(
        length,
        [
            dict(CROWN_FLOW, weight=0.3),
            dict(type="direction", vector=[0.0, 0.7, -1.0], weight=1.6, phi=[70, 40]),
            SIDES_DOWN,
        ],
        dict(base=50, sd=6, top=-4, crown=4, front=-42, edge=0, min=8, max=85),
        seed=37,
        tousle=0.2,
        sway=0.15,
        longest=1.3,
        relax=dict(iters=100, gravity=0.00008, stiff0=3.0),
        band_mm=[3.0, 6.0, 26.0],
        cuts=cuts(layer=(0.85, 1.05), texture=3.0),
    )
    render = dict(
        TEXTURED,
        points=16,
        fade=dict(
            start=fade_curve(sides=7.0, back=10.0, sideburn=4.0),
            width=14.0,
            clipper=0.0005,
            top=0.018,
        ),
    )
    return spec("textured_quiff", "Textured quiff", "short", groom, render)


def pixie():
    """a textured pixie: a long side-swept fringe, short sides and a tapered nape"""

    def length(a, h):
        fringe = smoothstep(70, 20, a) * smoothstep(40, 10, h)
        return 42.0 + 16.0 * fringe - 20.0 * sides_of(a, h)

    groom = short_groom(
        length,
        [
            dict(CROWN_FLOW, weight=0.6),
            dict(type="part", x=0.03, weight=1.2, falloff=0.05, el=[25, 45]),
            dict(
                type="direction",
                vector=[-1.0, -0.4, 0.4],
                weight=1.2,
                el=[45, 20],
                phi=[60, 25],
            ),
            SIDES_DOWN,
        ],
        dict(base=40, sd=8, top=-6, crown=6, front=-6, edge=8, min=20, max=70),
        seed=38,
        tousle=0.35,
        sway=0.25,
        longest=1.35,
        cuts=cuts(layer=(0.75, 1.05), texture=5.0),
    )
    render = dict(
        TEXTURED,
        points=16,
        fade=dict(start=fade_curve(back=0.0), width=12.0, clipper=0.003, top=0.02),
    )
    s = spec("pixie", "Pixie", "short", groom, render)
    s["mirror"] = True
    return s


# ---------------------------------------------------------------------------------------------- medium and long
def perimeter(front, sides, back, fringe=None, fringe_width=35.0):
    """a perimeter line (world height) along |azimuth|: an optional fringe at the front, then the
    front, the sides and the back of the cut"""
    phi = [0.0, 30.0, 50.0, 80.0, 110.0, 140.0, 180.0]
    y = [front, front, 0.5 * (front + sides), sides, 0.5 * (sides + back), back, back]
    if fringe is not None:
        phi = [0.0, fringe_width, fringe_width + 12.0] + [
            p for p in phi if p > fringe_width + 12.0
        ]
        y = [fringe, fringe, front] + [
            v
            for p, v in zip([0.0, 30.0, 50.0, 80.0, 110.0, 140.0, 180.0], y)
            if p > fringe_width + 12.0
        ]
    return dict(phi=phi, y=y)


MEDIUM_RENDER = dict(
    # blunt ends: few thinned strands, ending near the cut (loose ends over a long band look like a haze)
    thinning=dict(share=0.12, range=[0.85, 0.97]),
    clump=dict(
        coarse=[0.5, 0.8],
        coarse_tip=0.5,
        fine=[0.35, 0.65],
        fine_power=[1.0, 1.5],
        sectors=6,
    ),
    frizz=dict(mm=[0.2, 0.6], cycles=[0.8, 2.0]),
    flyaways=dict(share=0.006, mm=[2.0, 6.0]),
    density=0.95,
)


def long_groom(
    length, flow, lift, seed, perim, tousle=0.2, sway=0.15, longest=1.25, **kw
):
    g = dict(
        seed=seed,
        whorl=dict(azimuth=168, elevation=60),
        flow=flow,
        tousle=tousle,
        lift=lift,
        length=length_table(length),
        length_jitter=[0.96, 1.04],
        length_extra_mm=5,
        longest=longest,
        perimeter=perim,
        below_perimeter_mm=[9, 13],
        lift_mm=3.0,
        sway=[[sway, 1.5]],
        cuts=cuts(layer=(0.92, 1.05), texture=3.0),
        relax=dict(iters=320, band_floor=0.47, stiff_ref_mm=3.0),
        layers=dict(base=0.6 * 0.001, per_guide=1.1 * 0.001, floor=0.47, cap=0.012),
        body_collision=True,
        relax_points=32,
        ears=True,
        hang_below=0.45,
        smooth=8,
    )
    g.update(kw)
    return g


PART_MIDDLE = dict(type="part", x=0.0, weight=2.4, falloff=0.05, el=[15, 40])
DOWN_ALL = dict(type="direction", vector=[0.0, -1.0, 0.0], weight=1.2, el=[60, 20])
# long hair falls down and a little back, so that it clears the shoulders; the front pieces fall
# in front of them
DOWN_BACK = dict(
    type="direction", vector=[0.0, -1.0, -0.45], weight=1.4, el=[55, 15], phi=[50, 80]
)
DOWN_FRONT = dict(
    type="direction", vector=[0.0, -1.0, 0.15], weight=1.2, el=[55, 15], phi=[70, 45]
)
LONG_FLOW = [dict(CROWN_FLOW, weight=0.12), PART_MIDDLE, DOWN_BACK, DOWN_FRONT]
# the hair at the front hairline goes back first, so that it falls beside the face
CLEAR_FACE = dict(
    type="direction", vector=[0.0, 0.2, -1.0], weight=1.3, el=[45, 22], phi=[50, 25]
)


def curtains():
    """a middle part with curtain bangs that fall to the sides of the face"""
    groom = long_groom(
        lambda a, h: 120.0 + 0.0 * a,
        [dict(CROWN_FLOW, weight=0.4), PART_MIDDLE, DOWN_ALL],
        dict(base=55, sd=6, top=-2, crown=4, front=-10, edge=6, min=20, max=80),
        seed=41,
        perim=perimeter(front=0.488, sides=0.468, back=0.440),
        tousle=0.18,
        cuts=cuts(
            layer=(0.85, 1.05),
            texture=5.0,
            flick=dict(share=0.3, min_azimuth=20, mm=[1.0, 3.0]),
        ),
        band_mm=[3.0, 5.0, 6.0],
        body_collision=False,
    )
    render = dict(MEDIUM_RENDER, points=20)
    return spec("curtains", "Curtains", "medium", groom, render)


def modern_mullet():
    """a textured top, faded sides and a longer back"""

    def length(a, h):
        back = smoothstep(115, 145, a) * smoothstep(30, 10, el_of(a, h))
        return 55.0 + 85.0 * back - 41.0 * sides_of(a, h) * (1 - back)

    groom = long_groom(
        length,
        [
            dict(CROWN_FLOW, weight=0.8),
            dict(
                type="direction",
                vector=[0.0, 0.3, 1.0],
                weight=0.8,
                el=[20, 40],
                phi=[75, 45],
            ),
            DOWN_ALL,
        ],
        dict(base=45, sd=9, top=-4, crown=6, front=-10, edge=10, min=15, max=80),
        seed=42,
        perim=perimeter(front=0.60, sides=0.60, back=0.395),
        tousle=0.4,
        sway=0.28,
        cuts=cuts(
            layer=(0.75, 1.05),
            texture=6.0,
            flick=dict(share=0.5, min_azimuth=120, mm=[2.0, 5.0]),
        ),
        band_mm=[3.0, 8.0, 5.0],
    )
    render = dict(
        MEDIUM_RENDER,
        points=20,
        fade=dict(
            start=fade_curve(sides=4.0, back=-90.0, sideburn=2.0),
            width=12.0,
            clipper=0.0006,
            top=0.014,
        ),
    )
    return spec("modern_mullet", "Modern mullet", "medium", groom, render)


def wolf_cut():
    """a shaggy, heavily layered cut with volume at the crown and a fringe"""
    groom = long_groom(
        lambda a, h: 150.0 - 40.0 * front_of(a, h, 40),
        [
            dict(CROWN_FLOW, weight=0.7),
            dict(
                type="direction",
                vector=[0.0, -0.5, 1.0],
                weight=1.0,
                el=[40, 15],
                phi=[45, 25],
            ),
            DOWN_ALL,
        ],
        dict(base=45, sd=10, top=-8, crown=-10, front=-6, edge=8, min=15, max=80),
        seed=43,
        perim=perimeter(front=0.47, sides=0.44, back=0.37, fringe=0.528),
        tousle=0.4,
        sway=0.3,
        cuts=cuts(
            layer=(0.5, 1.05),
            texture=9.0,
            flick=dict(share=0.5, min_azimuth=40, mm=[2.0, 6.0]),
        ),
        band_mm=[4.0, 12.0, 6.0],
    )
    render = dict(
        MEDIUM_RENDER, points=24, frizz=dict(mm=[0.3, 0.9], cycles=[1.0, 2.5])
    )
    return spec("wolf_cut", "Wolf cut", "medium", groom, render)


def french_bob():
    """a blunt bob at the chin with a fringe at the brows"""
    groom = long_groom(
        lambda a, h: 200.0 + 0.0 * a,
        [
            dict(CROWN_FLOW, weight=0.5),
            dict(
                type="direction",
                vector=[0.0, -1.0, 0.7],
                weight=1.4,
                el=[45, 20],
                phi=[40, 25],
            ),
            DOWN_ALL,
        ],
        dict(base=66, sd=5, top=-2, crown=2, front=0, edge=6, min=35, max=85),
        seed=44,
        perim=perimeter(
            front=0.43, sides=0.43, back=0.432, fringe=0.532, fringe_width=38.0
        ),
        tousle=0.1,
        sway=0.06,
        longest=1.2,
        cuts=cuts(layer=(1.0, 1.05), texture=1.2),
        band_mm=[2.5, 4.0, 2.0],
    )
    render = dict(
        MEDIUM_RENDER,
        points=24,
        thinning=dict(share=0.1, range=[0.9, 0.98]),
        frizz=dict(mm=[0.1, 0.3], cycles=[0.6, 1.2]),
        flyaways=dict(share=0.003, mm=[1.0, 3.0]),
    )
    return spec("french_bob", "French bob", "medium", groom, render)


def textured_bob():
    """a soft bob at the jaw with a side part and broken-up ends"""
    groom = long_groom(
        lambda a, h: 200.0 + 0.0 * a,
        [
            dict(CROWN_FLOW, weight=0.4),
            dict(type="part", x=0.03, weight=2.2, falloff=0.05, el=[15, 40]),
            dict(
                type="direction",
                vector=[-1.0, 0.0, -0.8],
                weight=1.6,
                el=[45, 20],
                phi=[55, 30],
            ),
            DOWN_ALL,
        ],
        dict(base=60, sd=7, top=-4, crown=4, front=-8, edge=8, min=25, max=85),
        seed=45,
        perim=perimeter(front=0.445, sides=0.44, back=0.44),
        tousle=0.25,
        sway=0.15,
        cuts=cuts(
            layer=(0.8, 1.05),
            texture=7.0,
            flick=dict(share=0.4, min_azimuth=20, mm=[1.5, 4.0]),
        ),
        band_mm=[3.0, 6.0, 4.0],
    )
    render = dict(MEDIUM_RENDER, points=24)
    s = spec("textured_bob", "Textured bob", "medium", groom, render)
    s["mirror"] = True
    return s


def lob():
    """a long bob at the collarbones with a middle part"""
    groom = long_groom(
        lambda a, h: 190.0 + 0.0 * a,
        LONG_FLOW,
        dict(base=62, sd=6, top=-3, crown=3, front=-6, edge=6, min=25, max=85),
        seed=46,
        perim=perimeter(front=0.345, sides=0.345, back=0.35),
        tousle=0.15,
        cuts=cuts(
            layer=(0.85, 1.05),
            texture=6.0,
            flick=dict(share=0.3, min_azimuth=0, mm=[1.5, 4.0]),
        ),
        band_mm=[3.0, 5.0, 3.0],
    )
    render = dict(MEDIUM_RENDER, points=28)
    return spec("lob", "Lob", "medium", groom, render)


def long_layers():
    """long layered hair past the shoulders, with curtain bangs that frame the face"""

    def length(a, h):
        frame = smoothstep(70, 20, a) * smoothstep(60, 20, h)
        return 330.0 - 150.0 * frame

    groom = long_groom(
        length,
        LONG_FLOW,
        dict(base=60, sd=6, top=-3, crown=3, front=-8, edge=6, min=25, max=85),
        seed=47,
        perim=perimeter(front=0.40, sides=0.28, back=0.23),
        tousle=0.15,
        cuts=cuts(
            layer=(0.7, 1.05),
            texture=10.0,
            flick=dict(share=0.3, min_azimuth=0, mm=[2.0, 5.0]),
        ),
        band_mm=[3.0, 5.0, 4.0],
    )
    render = dict(MEDIUM_RENDER, points=32)
    return spec("long_layers", "Long layers", "long", groom, render)


def long_straight():
    """long straight hair with a middle part and a blunt end"""
    groom = long_groom(
        lambda a, h: 360.0 + 0.0 * a,
        LONG_FLOW + [CLEAR_FACE],
        dict(base=64, sd=5, top=-2, crown=2, front=-4, edge=5, min=30, max=85),
        seed=48,
        perim=perimeter(front=0.22, sides=0.21, back=0.20),
        tousle=0.1,
        sway=0.08,
        cuts=cuts(layer=(0.95, 1.05), texture=3.0),
        band_mm=[2.5, 4.0, 2.0],
    )
    render = dict(
        MEDIUM_RENDER,
        points=32,
        frizz=dict(mm=[0.1, 0.3], cycles=[0.5, 1.2]),
    )
    return spec("long_straight", "Long straight", "long", groom, render)


def long_wavy():
    """long hair in loose waves with a middle part"""
    s = long_straight()
    s["name"], s["label"] = "long_wavy", "Long wavy"
    s["groom"]["seed"] = 49
    s["groom"]["cuts"] = cuts(layer=(0.85, 1.05), texture=8.0)
    s["render"] = dict(
        s["render"],
        curl=dict(radius_mm=4.5, period_mm=95.0, ramp=0.3),
        frizz=dict(mm=[0.15, 0.4], cycles=[0.8, 1.6]),
        clump=dict(
            coarse=[0.6, 0.85],
            coarse_tip=0.5,
            fine=[0.5, 0.8],
            fine_power=[0.8, 1.2],
            sectors=6,
        ),
    )
    return s


# ---------------------------------------------------------------------------------------------- tied
TIED_RENDER = dict(
    thinning=None,
    clump=dict(
        coarse=[0.5, 0.8],
        coarse_tip=0.3,
        fine=[0.3, 0.6],
        fine_power=[1.0, 1.5],
        sectors=6,
    ),
    frizz=dict(mm=[0.1, 0.35], cycles=[0.8, 2.0]),
    flyaways=dict(share=0.004, mm=[1.5, 4.0]),
    density=1.0,
)


def tied(name, label, tie, seed, points, longest=1.3, controls=None, physics=None):
    groom = dict(seed=seed, tie=tie, longest=longest, body_collision=True)
    return spec(
        name,
        label,
        "tied",
        groom,
        dict(TIED_RENDER, points=points),
        physics=physics,
        controls=dict(
            dict(
                length=[0.6, longest],
                curl=[0.0, 1.0],
                volume=[0.9, 1.2],
                density=[0.6, 1.0],
            ),
            **(controls or {}),
        ),
    )


def low_ponytail():
    """the hair gathered at the nape into a ponytail"""
    tie = dict(
        azimuth=180,
        elevation=-14,
        offset_mm=8,
        radius_mm=10,
        bundle_mm=9,
        tail_mm=240,
        droop=1.4,
        spread=1.8,
    )
    return tied("low_ponytail", "Low ponytail", tie, 51, 40)


def high_ponytail():
    """the hair gathered high at the back of the head"""
    tie = dict(
        azimuth=180,
        elevation=32,
        offset_mm=10,
        radius_mm=10,
        bundle_mm=9,
        tail_mm=280,
        droop=0.5,
        spread=1.9,
    )
    return tied("high_ponytail", "High ponytail", tie, 52, 40)


def top_knot():
    """a bun on the crown, as a man bun"""
    tie = dict(azimuth=180, elevation=58, offset_mm=8, radius_mm=9, bundle_mm=7)
    tie.update(tail_mm=300, bun=True, bun_mm=[22, 9], spread=1.2)
    return tied(
        "top_knot",
        "Top knot",
        tie,
        53,
        48,
        longest=1.15,
        controls=dict(length=[0.85, 1.15]),
        physics=BUN_PHYSICS,
    )


def low_bun():
    """a bun at the nape"""
    tie = dict(azimuth=180, elevation=-6, offset_mm=8, radius_mm=10, bundle_mm=8)
    tie.update(tail_mm=380, bun=True, bun_mm=[26, 11], spread=1.2)
    return tied(
        "low_bun",
        "Low bun",
        tie,
        54,
        48,
        longest=1.15,
        controls=dict(length=[0.85, 1.15]),
        physics=BUN_PHYSICS,
    )


PRESETS = [
    bald,
    buzz_cut,
    crew_cut,
    low_taper_fade,
    mid_fade,
    textured_crop,
    french_crop,
    curly_top_taper,
    side_part_taper,
    textured_quiff,
    pixie,
    medium_tousled,
    curtains,
    modern_mullet,
    wolf_cut,
    french_bob,
    textured_bob,
    lob,
    long_layers,
    long_straight,
    long_wavy,
    low_ponytail,
    high_ponytail,
    top_knot,
    low_bun,
]


# ---------------------------------------------------------------------------------------------- build
def write_spec(s):
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    (STYLE_DIR / f"{s['name']}.json").write_text(json.dumps(s, indent=1) + "\n")


def encode_style(name, g):
    """the tensors of one style in styles.safetensors"""
    _, seg, codes = encode_curves(g["points"])
    return {
        f"{name}.codes": codes,
        f"{name}.segment": seg.astype(np.float32),
        f"{name}.length": g["length"].astype(np.float32),
        f"{name}.group": g["group"].astype(np.uint8),
        f"{name}.flick": g["flick"].astype(np.float32),
        f"{name}.pivot": g["pivot"].astype(np.float32),
    }


def build(names=None, body=None, verbose=True):
    from safetensors.numpy import load_file, save_file

    from anny.hair.authoring import groom
    from anny.viewer.geometry import fine_body

    body = body or fine_body()
    layout = load_layout()
    specs = [p() for p in PRESETS]
    if names:
        specs = [s for s in specs if s["name"] in names]
    tensors = load_file(str(GUIDES_PATH)) if GUIDES_PATH.exists() else {}
    need_body = any(s["groom"].get("body_collision") for s in specs)
    sdfs = groom.body_sdfs(body.V, body.N, with_body=need_body)
    for s in specs:
        write_spec(s)
        g = groom.grow(
            body.V, body.T, body.N, s, layout, s["render"]["points"], verbose, sdfs
        )
        tensors = {
            k: v for k, v in tensors.items() if not k.startswith(s["name"] + ".")
        }
        tensors.update(encode_style(s["name"], g))
    save_file(tensors, str(GUIDES_PATH))
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument(
        "--specs-only",
        action="store_true",
        help="write the specs alone, for changes to the render, physics or controls that keep the guides",
    )
    args = ap.parse_args()
    if args.specs_only:
        specs = [p() for p in PRESETS]
        specs = [s for s in specs if not args.only or s["name"] in args.only]
        for s in specs:
            write_spec(s)
        print(f"wrote {len(specs)} specs to {STYLE_DIR}")
        return
    specs = build(args.only)
    print(f"wrote {len(specs)} styles to {STYLE_DIR} and {GUIDES_PATH}")


if __name__ == "__main__":
    main()
