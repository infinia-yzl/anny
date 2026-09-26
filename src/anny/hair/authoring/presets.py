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
- ``physics``: the stiffness of the strands in the page's solver;
- ``controls``: the ranges of the page's sliders.

Usage::

    python -m anny.hair.authoring.presets [--only NAME ...]
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
DEFAULT_PHYSICS = dict(
    global_stiffness=[0.9, 0.08],
    local_stiffness=0.6,
    damping=0.08,
    gravity=0.35,
)
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
        physics=dict(DEFAULT_PHYSICS, **(physics or {})),
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
    return spec("medium_tousled", "Medium tousled", "medium", groom)


PRESETS = [medium_tousled]


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
    args = ap.parse_args()
    specs = build(args.only)
    print(f"wrote {len(specs)} styles to {STYLE_DIR} and {GUIDES_PATH}")


if __name__ == "__main__":
    main()
