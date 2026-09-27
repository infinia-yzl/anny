# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
A review grid of the hairstyles, rendered by the viewer page (``viewer/dist/anny_viewer.html``).

Each row holds one style, at the shortest, the default and the longest value of its length
slider, on a woman, a man and a child, seen from three quarters behind so that the back and the
side of the cut show. Check every change to the presets, the groom or the page's hair passes with
this grid as well as with the tests: each style must read as the cut it names, at every length and
on every body.

Usage::

    python -m anny.hair.authoring.review [--styles NAME ...] [--out PATH] [--size N]

The grid goes to ``ANNY_CACHE_DIR/hair/review.png`` by default. The page renders in software
(SwiftShader), so the full grid of 25 styles takes a long time; ``--styles`` renders a few.
Playwright is needed (``uv sync --extra faces``, or ``uv run --with playwright``).
"""

from __future__ import annotations

import argparse
import base64
import io
import pathlib

from anny.hair.styles import load_spec, style_names
from anny.paths import get_anny_cache_path

# (label, years, gender)
BODIES = [
    ("woman, 30 years", 30.0, 1.0),
    ("man, 30 years", 30.0, 0.0),
    ("girl, 8 years", 8.0, 1.0),
]
HAIR_COLOUR = "#3a2a1c"
YAW = 145  # degrees: three quarters from behind
HEADER = 20  # pixels: the labels of the bodies
# the view of the head and the shoulders about the cranium's centre (as the hair poses it), at the
# size of the head of the body
VIEW = """(yaw) => {
  const c = __HAIR.uHeadC.value, k = __BODY.anny.headScale();
  window.setView(yaw, 6, 0.95 * k, c.y - 0.1 * k, c.z, 24, c.x);
  return document.querySelector('canvas').toDataURL('image/png');
}"""


def ages(years):
    """the page's age slider for ages in years (anny's morphological age mapping)"""
    import torch

    import anny
    from anny.shape_distribution import SimpleShapeDistribution

    mapping = SimpleShapeDistribution(anny.Anny()).morphological_age_mapping
    return [
        float(mapping.morphological_to_anny_age(torch.tensor([y])).item())
        for y in years
    ]


def review(
    styles: list[str] | None = None,
    out: pathlib.Path | None = None,
    size: int = 256,
    accumulation: int = 3,
    page: pathlib.Path | None = None,
) -> pathlib.Path:
    """render the review grid and return its path"""
    from PIL import Image, ImageDraw
    from playwright.sync_api import sync_playwright

    from anny.viewer.build import REPO

    families = ["short", "medium", "long", "tied"]
    styles = styles or sorted(
        style_names(), key=lambda n: (families.index(load_spec(n)["family"]), n)
    )
    age = ages([b[1] for b in BODIES])
    page = page or REPO / "viewer" / "dist" / "anny_viewer.html"
    columns = len(BODIES) * 3
    grid = Image.new("RGB", (size * columns, HEADER + size * len(styles)), "white")
    draw = ImageDraw.Draw(grid)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/opt/pw-browsers/chromium",
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        tab = browser.new_page(viewport=dict(width=size, height=size))
        tab.goto(page.resolve().as_uri() + f"?shot=1&acc={accumulation}")
        tab.wait_for_function("window.__READY && window.__BODY.ready", timeout=900000)
        tab.add_style_tag(
            content="body * { visibility: hidden !important; } "
            "canvas { visibility: visible !important; }"
        )
        for r, name in enumerate(styles):
            lo, hi = load_spec(name)["controls"]["length"]
            for b, (_, _, gender) in enumerate(BODIES):
                for c, length in enumerate((lo, 1.0, hi)):
                    look = dict(
                        format="anny-viewer/look@3",
                        name="review",
                        phenotype=dict(age=age[b], gender=gender),
                        hair=dict(color=HAIR_COLOUR, style=name, length=length),
                    )
                    tab.evaluate("(l) => window.setLook(l)", look)
                    url = tab.evaluate(VIEW, YAW)
                    image = Image.open(
                        io.BytesIO(base64.b64decode(url.split(",", 1)[1]))
                    )
                    at = ((b * 3 + c) * size, HEADER + r * size)
                    grid.paste(image.convert("RGB"), at)
            draw.text((4, HEADER + r * size + 4), name, fill=(220, 40, 40))
        browser.close()
    for b, (label, *_) in enumerate(BODIES):
        draw.text(
            (b * 3 * size + 4, 4),
            f"{label}: shortest, default, longest",
            fill=(0, 0, 0),
        )
        if b:
            draw.line(
                [(b * 3 * size, 0), (b * 3 * size, HEADER + size * len(styles))],
                fill=(255, 255, 255),
                width=3,
            )
    out = out or get_anny_cache_path() / "hair" / "review.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="render a review grid of the hairstyles"
    )
    parser.add_argument("--styles", nargs="*", default=None)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    print(review(args.styles, args.out, args.size))


if __name__ == "__main__":
    main()
