# Corporis
# Apache License, Version 2.0
"""
Flat pictures of a posed mesh: a silhouette as an SVG path.

Every function takes the mesh already projected to the picture, as ``xy`` points (x right,
y up, any unit) with the mesh's triangles; :func:`fit` places them in a box of pixels.
"""

from __future__ import annotations

import numpy as np


def fit(xy: np.ndarray, box) -> np.ndarray:
    """``xy`` (y up) scaled into ``box`` (x, y, width, height in pixels, y down), centred"""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    x, y, w, h = box
    s = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
    ox = x + (w - (hi[0] - lo[0]) * s) / 2
    oy = y + (h - (hi[1] - lo[1]) * s) / 2
    return np.stack([ox + (xy[:, 0] - lo[0]) * s, oy + (hi[1] - xy[:, 1]) * s], axis=1)


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker simplification of a closed ring: each half keeps its own end points."""
    if len(points) < 8:
        return points
    half = len(points) // 2
    return np.concatenate(
        [
            _simplify_open(points[: half + 1], tolerance)[:-1],
            _simplify_open(points[half:], tolerance),
        ]
    )


def _simplify_open(points, tolerance):
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = points[b] - points[a]
        rel = points[a + 1 : b] - points[a]
        dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / (
            np.linalg.norm(seg) + 1e-12
        )
        i = int(np.argmax(dist))
        if dist[i] > tolerance:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return points[keep]


def silhouette_mask(points: np.ndarray, faces: np.ndarray, supersample: int = 8):
    """the filled mesh at ``supersample`` mask pixels per picture pixel, softened, and the
    picture position of the mask's corner"""
    from PIL import Image, ImageDraw, ImageFilter

    lo = points.min(axis=0) - 2
    size = ((points.max(axis=0) + 2 - lo) * supersample).astype(int) + 1
    image = Image.new("L", tuple(int(v) for v in size), 0)
    draw = ImageDraw.Draw(image)
    for tri in ((points - lo) * supersample)[faces]:
        draw.polygon([tuple(p) for p in tri], fill=255)
    image = image.filter(ImageFilter.GaussianBlur(supersample * 0.35))
    return np.asarray(image, dtype=np.float32) / 255.0, lo


def contours(mask: np.ndarray, lo, supersample: int, tolerance: float = 0.15) -> list:
    """the rings (in picture pixels) where ``mask`` crosses one half"""
    import contourpy

    gen = contourpy.contour_generator(z=mask, fill_type="OuterOffset")
    polygons, offsets = gen.filled(0.5, 2.0)
    rings = []
    for ring_points, offs in zip(polygons, offsets):
        for a, b in zip(offs[:-1], offs[1:]):
            ring = simplify(
                ring_points[a:b].astype(np.float64) / supersample + lo, tolerance
            )
            if len(ring) >= 3:
                rings.append(ring)
    return rings


def path(rings) -> str:
    return "".join(
        "M" + " L".join(f"{p[0]:.1f} {p[1]:.1f}" for p in r) + "Z" for r in rings
    )


def outline(points: np.ndarray, faces: np.ndarray, supersample: int = 8) -> str:
    """the silhouette of a mesh already in pixels (x right, y down) as an SVG path"""
    mask, lo = silhouette_mask(points, faces, supersample)
    return path(contours(mask, lo, supersample))


def svg(width, height, body, desc="", title="Corporis", background=None) -> str:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{title}</title>',
        f'  <desc id="desc">{desc}</desc>',
    ]
    if background:
        lines.append(f'  <rect width="{width}" height="{height}" fill="{background}"/>')
    return "\n".join(lines + list(body) + ["</svg>"]) + "\n"
