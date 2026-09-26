# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Measures of the hair of the viewer page, and reference views of it.

The page runs in Chromium with SwiftShader (software WebGL), so the render times compare
versions of the page on one machine; they do not predict the times on a GPU. The measures:

- ``load_s``: the time until the page is ready;
- ``slider_ms``: the update of the body for a slider step, and its ``strands`` part;
- ``render_hair_ms`` and ``render_nohair_ms``: one accumulation frame of the face view, with
  and without the hair (the GPU work is waited for);
- ``hair_bytes``: the GPU buffers and textures of the hair, and ``hair_data_bytes``: the hair
  data in the page.

Usage::

    python -m anny.hair.authoring.benchmark [--page PATH] [--out DIR]

The measures go to ``DIR/benchmark.json`` and the views to ``DIR/*.png``
(``ANNY_CACHE_DIR/hair/benchmark`` by default). Playwright is needed
(``uv sync --extra faces``, or ``uv run --with playwright``).
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import time

from anny.paths import get_anny_cache_path

VIEWS = {
    "front": (0, 3),
    "side": (90, 3),
    "back": (180, 3),
    "three_quarter": (35, 8),
}
SYNC = (
    "const gl = document.querySelector('canvas').getContext('webgl2');"
    "gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));"
)
# the GPU bytes of the hair: the page's own count (window.hairStats), or the attributes of the
# legacy ribbon mesh
HAIR_BYTES = """() => {
  if (window.hairStats) return window.hairStats();
  const m = window.__hair; if (!m) return null;
  const g = m.geometry; let n = g.index ? g.index.array.byteLength : 0;
  for (const a of Object.values(g.attributes)) n += a.array.byteLength;
  return { gpu_bytes: n, strands: null };
}"""
HAIR_DATA = """() => {
  const B = window.__BUFFERS; if (!B) return null;
  let n = 0; for (const [k, b] of Object.entries(B)) if (k.startsWith('hair')) n += b.info.byteLength;
  return n;
}"""


def default_out() -> pathlib.Path:
    return get_anny_cache_path() / "hair" / "benchmark"


def run(
    page: pathlib.Path, out: pathlib.Path, size: int = 640, frames: int = 8
) -> dict:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    url = page.resolve().as_uri() + f"?shot=1&acc={frames}"
    res: dict = {"page": str(page)}
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
        t0 = time.time()
        tab.goto(url)
        tab.wait_for_function("window.__READY && window.__BODY.ready", timeout=900000)
        res["load_s"] = round(time.time() - t0, 2)
        steps = [
            tab.evaluate("(v) => window.setSliders(v)", {"age": a})
            for a in (0.3, 0.7, 0.5)
        ]
        res["slider_ms"] = [
            {"body": s["body"], "strands": s["steps"].get("strands")} for s in steps
        ]

        def render(yaw, pitch):
            tab.evaluate("() => window.setFrame('face')")
            t = time.time()
            tab.evaluate(
                f"() => {{ window.setView({yaw}, {pitch}, 0.55, undefined, undefined, 24); {SYNC} }}"
            )
            return (time.time() - t) * 1000 / frames

        def shot(path):
            url = tab.evaluate(
                "() => document.querySelector('canvas').toDataURL('image/png')"
            )
            path.write_bytes(base64.b64decode(url.split(",", 1)[1]))

        render(*VIEWS["front"])  # warm up the shaders
        for name, (yaw, pitch) in VIEWS.items():
            res[f"render_{name}_ms"] = round(render(yaw, pitch), 1)
            shot(out / f"{name}.png")
        res["render_hair_ms"] = round(render(*VIEWS["three_quarter"]), 1)
        tab.evaluate("() => window.setHair(false)")
        res["render_nohair_ms"] = round(render(*VIEWS["three_quarter"]), 1)
        tab.evaluate("() => window.setHair(true)")
        res["hair_bytes"] = tab.evaluate(HAIR_BYTES)
        res["hair_data_bytes"] = tab.evaluate(HAIR_DATA)
        browser.close()
    (out / "benchmark.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    from anny.viewer.build import REPO

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--page",
        type=pathlib.Path,
        default=REPO / "viewer" / "dist" / "anny_viewer.html",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    res = run(args.page, args.out or default_out())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
