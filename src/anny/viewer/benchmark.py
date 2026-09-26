# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The work of a moving frame of the viewer page (``viewer/dist/anny_viewer.html``).

The page stops drawing once its picture is refined, so its frame rate matters while the figure or
the camera moves. This benchmark plays the run clip with the hair's physics in the Body and Face
views, for a short and a long cut, and counts the work of each frame by wrapping the WebGL calls:
the MB of vertex and texture uploads, the draw calls and the vertices drawn (indices, times the
instances). These counts do not depend on the device. With a page that has ``window.frameStats``,
it also reports the render scale, the level of detail of the hair and the JavaScript time of a
frame.

The page runs in Chromium with SwiftShader, so it does not time the GPU. The browser has a
desktop screen (1920 x 1080, device pixel ratio 1) and a 800 x 500 viewport.

Usage::

    python -m anny.viewer.benchmark [--page PATH] [--out PATH] [--frames N]

Playwright is needed (``uv sync --extra faces``, or ``uv run --with playwright``).
"""

from __future__ import annotations

import argparse
import json
import pathlib

SCENARIOS = [
    ("body", "medium_tousled"),
    ("face", "medium_tousled"),
    ("body", "long_straight"),
    ("face", "long_straight"),
]
# counts of the WebGL work, per call site (installed before the page loads)
WRAP = """
(() => {
  const S = window.__gl = { calls: {}, bytes: {}, verts: {} };
  const P = WebGL2RenderingContext.prototype;
  const add = (k, b, v) => {
    S.calls[k] = (S.calls[k] || 0) + 1;
    if (b) S.bytes[k] = (S.bytes[k] || 0) + b;
    if (v) S.verts[k] = (S.verts[k] || 0) + v;
  };
  const size = (x) => (x && x.byteLength) || 0;
  const wrap = (name, count) => {
    const f = P[name];
    P[name] = function (...a) { const [b, v] = count(a); add(name, b, v); return f.apply(this, a); };
  };
  wrap('bufferSubData', (a) => [a[4] !== undefined ? a[4] * (a[2].BYTES_PER_ELEMENT || 1) : size(a[2]), 0]);
  wrap('bufferData', (a) => [size(a[1]), 0]);
  for (const t of ['texSubImage2D', 'texImage2D', 'texSubImage3D', 'texImage3D'])
    wrap(t, (a) => [size(a.find((x) => x && x.byteLength)), 0]);
  wrap('drawElements', (a) => [0, a[1]]);
  wrap('drawArrays', (a) => [0, a[2]]);
  wrap('drawElementsInstanced', (a) => [0, a[1] * a[4]]);
  wrap('drawArraysInstanced', (a) => [0, a[2] * a[3]]);
})();
"""


def run(page: pathlib.Path, frames: int = 6) -> dict:
    from playwright.sync_api import sync_playwright

    out: dict = {"page": str(page), "viewport": [800, 500], "scenarios": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/opt/pw-browsers/chromium",
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        ctx = browser.new_context(
            viewport=dict(width=800, height=500),
            screen=dict(width=1920, height=1080),
            device_scale_factor=1,
        )
        ctx.add_init_script(WRAP)
        tab = ctx.new_page()
        # one render pass per call of setFrame: a frame of the clip, drawn once
        tab.goto(page.resolve().as_uri() + "?shot=1&acc=1")
        tab.wait_for_function("window.__READY && window.__BODY.ready", timeout=900000)
        has = tab.evaluate(
            "() => ({ physics: typeof window.stepHair === 'function', stats: typeof window.frameStats === 'function' })"
        )
        for view, style in SCENARIOS:
            tab.evaluate(f"() => window.setHairStyle('{style}')")
            if has["physics"]:
                tab.evaluate("() => window.setHairPhysics(true)")
            tab.evaluate(f"() => window.setFrame('{view}')")
            per = []
            for k in range(frames):
                tab.evaluate(
                    f"() => {{ window.setMotion('run', {k / 30}, true);"
                    + (" window.stepHair(1 / 30);" if has["physics"] else "")
                    + " const S = __gl; S.calls = {}; S.bytes = {}; S.verts = {}; }"
                )
                tab.evaluate(f"() => window.setFrame('{view}')")
                per.append(tab.evaluate("() => JSON.parse(JSON.stringify(__gl))"))
            # the first frame of a clip has the uploads of the new pose as well
            per = per[1:]
            n = len(per)

            def mean(key, names):
                return sum(sum(f[key].get(c, 0) for c in names) for f in per) / n

            draws = [
                "drawElements",
                "drawArrays",
                "drawElementsInstanced",
                "drawArraysInstanced",
            ]
            res = {
                "vertex_upload_mb": round(
                    mean("bytes", ["bufferSubData", "bufferData"]) / 1e6, 2
                ),
                "texture_upload_mb": round(
                    mean(
                        "bytes",
                        ["texSubImage2D", "texImage2D", "texSubImage3D", "texImage3D"],
                    )
                    / 1e6,
                    2,
                ),
                "draw_calls": round(mean("calls", draws), 1),
                "mesh_vertices_m": round(
                    mean("verts", ["drawElements", "drawArrays"]) / 1e6, 2
                ),
                "hair_vertices_m": round(
                    mean("verts", ["drawElementsInstanced", "drawArraysInstanced"])
                    / 1e6,
                    2,
                ),
            }
            hs = tab.evaluate("() => window.hairStats ? window.hairStats() : null")
            if hs:
                res["hair_strands"] = hs.get("strands")
                res["hair_lod"] = hs.get("lod")
            if has["stats"]:
                res["frame"] = tab.evaluate("() => window.frameStats()")
            out["scenarios"][f"{view}/{style}"] = res
        browser.close()
    return out


def main():
    from anny.viewer.build import REPO

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--page",
        type=pathlib.Path,
        default=REPO / "viewer" / "dist" / "anny_viewer.html",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    res = run(args.page, args.frames)
    text = json.dumps(res, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
