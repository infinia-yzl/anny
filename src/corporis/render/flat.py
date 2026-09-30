# Corporis
# Apache License, Version 2.0
"""
Flat pictures of a posed mesh: a silhouette (SVG), a line drawing (SVG) and a toon-shaded
picture (PNG).

:class:`View` sets the camera. The low-level tracer (:func:`outline`) takes the mesh already
projected into pixels (x right, y down); :func:`fit` places projected points in a box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


# ---- views, and the rasters of depth and light behind the outline and the shaded picture ----

SKIN = "#e2bca3"
INK = "#2a2320"


@dataclass
class View:
    """a camera looking at the model from the front (along +y), turned by ``yaw`` degrees
    about the vertical, tilted by ``pitch`` degrees (from below when positive), and the
    picture turned by ``roll`` degrees"""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @classmethod
    def facing(cls, forward, turn: float = 0.0, pitch: float = 0.0, roll: float = 0.0):
        """the view from in front of a figure that faces ``forward`` (the model's frame; only
        its horizontal part counts), turned ``turn`` degrees further about the vertical"""
        yaw = math.degrees(math.atan2(-float(forward[0]), -float(forward[1])))
        return cls(yaw + turn, pitch, roll)

    def basis(self):
        """the picture's right and up, and the direction toward the camera, in the model's
        frame"""
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        right = np.array([c, -s, 0.0])
        toward = np.array([-s, -c, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        up, toward = cp * up - sp * toward, sp * up + cp * toward
        cr, sr = math.cos(math.radians(self.roll)), math.sin(math.radians(self.roll))
        right, up = cr * right + sr * up, -sr * right + cr * up
        return right, up, toward

    def project(self, vertices):
        """picture points (x right, y up) and depths (larger is farther)"""
        right, up, toward = self.basis()
        v = np.asarray(vertices, dtype=np.float64)
        return np.stack([v @ right, v @ up], axis=1), -(v @ toward)


def vertex_normals(vertices, faces) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _rasters(px, depth, faces, size, supersample, shade=None):
    """the depth (inf where empty) and, when ``shade`` gives a value per triangle, the shade
    of the nearest triangle at each mask pixel, drawn far to near"""
    from PIL import Image, ImageDraw

    w, h = int(size[0] * supersample), int(size[1] * supersample)
    order = np.argsort(-depth[faces].mean(axis=1))  # far first
    far = (
        float(depth.max()) + 10.0
    )  # empty; stored as 32-bit floats, so compare with a margin
    depth_image = Image.new("F", (w, h), far)
    draw_d = ImageDraw.Draw(depth_image)
    shade_image = draw_s = None
    if shade is not None:
        shade_image = Image.new("F", (w, h), 0.0)
        draw_s = ImageDraw.Draw(shade_image)
    tri = px[faces] * supersample
    tri_depth = depth[faces].mean(axis=1)
    for i in order:
        pts = [tuple(p) for p in tri[i]]
        draw_d.polygon(pts, fill=float(tri_depth[i]))
        if draw_s is not None:
            draw_s.polygon(pts, fill=float(shade[i]))
    d = np.asarray(depth_image, dtype=np.float64).copy()
    d[d >= far - 5.0] = np.inf
    s = np.asarray(shade_image, dtype=np.float64) if shade_image is not None else None
    return d, s


def _contour_edges(vertices, faces, view, px, depth, supersample, width_px, tol=0.015):
    """a mask of the mesh's visible contour edges: the edges between a triangle that faces
    the camera and one that faces away, drawn where the depth raster shows them in front
    (within ``tol`` metres), ``width_px`` mask pixels wide"""
    from PIL import Image, ImageDraw

    v = np.asarray(vertices, dtype=np.float64)
    _, z = view.project(v)
    _, _, toward = view.basis()
    normals = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    front = normals @ toward > 0
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    key = np.sort(edges, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    pair = np.flatnonzero(np.all(key[1:] == key[:-1], axis=1))  # each interior edge
    contour = key[pair][front[owner[pair]] != front[owner[pair + 1]]]
    a, b = contour[:, 0], contour[:, 1]
    h, w = depth.shape
    seen = 0
    for t in (0.2, 0.5, 0.8):
        q = (px[a] * (1 - t) + px[b] * t) * supersample
        i = np.clip(q[:, 1].astype(int), 0, h - 1)
        j = np.clip(q[:, 0].astype(int), 0, w - 1)
        seen = seen + (z[a] * (1 - t) + z[b] * t <= depth[i, j] + tol)
    image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(width_px)))
    for e in np.flatnonzero(seen >= 2):
        draw.line(
            [tuple(px[a[e]] * supersample), tuple(px[b[e]] * supersample)],
            fill=255,
            width=width,
        )
    return np.asarray(image) > 0


def _outer_band(depth, width_px):
    """the pixels of the figure within ``width_px`` of its outer contour"""
    from PIL import Image, ImageFilter

    empty = Image.fromarray((~np.isfinite(depth) * 255).astype(np.uint8))
    grown = empty.filter(ImageFilter.MaxFilter(int(width_px) // 2 * 2 + 1))
    return np.asarray(grown) > 0


def _line_mask(edges, width_px):
    """``edges`` thickened to about ``width_px`` mask pixels, softened"""
    from PIL import Image, ImageFilter

    image = Image.fromarray((edges * 255).astype(np.uint8))
    if width_px > 1:
        image = image.filter(ImageFilter.MaxFilter(int(width_px) // 2 * 2 + 1))
    image = image.filter(ImageFilter.GaussianBlur(max(0.6, width_px * 0.25)))
    return np.asarray(image, dtype=np.float32) / 255.0


def _place(vertices, view, size, margin):
    xy, depth = view.project(vertices)
    w, h = size
    return fit(xy, (margin, margin, w - 2 * margin, h - 2 * margin)), depth


def silhouette_svg(
    vertices,
    faces,
    view=View(),
    size=(512, 512),
    margin=24,
    fill=SKIN,
    background=None,
    desc="",
) -> str:
    """the posed mesh as one filled shape"""
    px, _ = _place(vertices, view, size, margin)
    body = [f'  <path fill="{fill}" fill-rule="evenodd" d="{outline(px, faces)}"/>']
    return svg(size[0], size[1], body, desc, background=background)


def outline_svg(
    vertices,
    faces,
    view=View(),
    size=(512, 512),
    margin=24,
    ink=INK,
    fill=None,
    outer=3.0,
    inner=1.4,
    background=None,
    desc="",
    supersample=4,
) -> str:
    """the posed mesh as a line drawing: the outer contour ``outer`` pixels wide, and a line
    ``inner`` pixels wide along each visible contour inside the figure, where a limb passes
    in front of the body or another limb; ``fill`` fills the figure under the lines"""
    px, depth = _place(vertices, view, size, margin)
    d, _ = _rasters(px, depth, faces, size, supersample)
    body = []
    mask, lo = silhouette_mask(px, faces, supersample)
    rings = contours(mask, lo, supersample)
    if fill:
        body.append(f'  <path fill="{fill}" fill-rule="evenodd" d="{path(rings)}"/>')
    body.append(
        f'  <path fill="none" stroke="{ink}" stroke-width="{outer}" '
        f'stroke-linejoin="round" d="{path(rings)}"/>'
    )
    edges = _contour_edges(
        vertices, faces, view, px, d, supersample, inner * supersample
    )
    lines = _line_mask(edges & ~_outer_band(d, outer * supersample), 1)
    inner_rings = contours(lines, np.zeros(2), supersample, tolerance=0.1)
    if inner_rings:
        body.append(
            f'  <path fill="{ink}" fill-rule="nonzero" d="{path(inner_rings)}"/>'
        )
    return svg(size[0], size[1], body, desc, background=background)


def _rgb(colour):
    colour = colour.lstrip("#")
    return np.array([int(colour[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def shaded_png(
    vertices,
    faces,
    view=View(),
    size=(512, 512),
    margin=24,
    colour=SKIN,
    ink=None,
    bands=(0.62, 0.84, 1.0),
    light=(-0.45, 0.7, 0.55),
    lines=True,
    supersample=3,
):
    """the posed mesh in flat toon shading: ``bands`` of light from the light direction
    (picture right, up, toward the camera), a soft rim at the contour, and the outline's
    lines in a darker ``ink``; a PIL image with a transparent background"""
    from PIL import Image

    px, depth = _place(vertices, view, size, margin)
    right, up, toward = view.basis()
    n = vertex_normals(vertices, faces)[faces].mean(axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    L = light[0] * right + light[1] * up + light[2] * toward
    L /= np.linalg.norm(L)
    lambert = np.clip(n @ L, 0.0, 1.0)
    thresholds = np.linspace(0.0, 1.0, len(bands) + 1)[1:-1] * 0.8 + 0.1
    band = np.array(bands)[np.searchsorted(thresholds, lambert)]
    rim = np.clip(1.0 - np.abs(n @ toward), 0.0, 1.0) ** 3 * 0.12
    shade = band + rim
    d, s = _rasters(px, depth, faces, size, supersample, shade=shade)
    body = np.isfinite(d)
    base = _rgb(colour)
    rgb = np.clip(base[None, None, :] * s[..., None], 0, 255)
    alpha = body.astype(np.float64)
    if lines:
        ink_rgb = _rgb(ink) if ink else base * 0.45
        edge = _line_mask(
            _contour_edges(
                vertices, faces, view, px, d, supersample, 1.2 * supersample
            ),
            1,
        )
        outer = _line_mask(
            body & ~np.roll(body, 1, 0)
            | body & ~np.roll(body, -1, 0)
            | body & ~np.roll(body, 1, 1)
            | body & ~np.roll(body, -1, 1),
            2.0 * supersample,
        )
        k = np.clip(np.maximum(edge, outer), 0, 1)[..., None]
        rgb = rgb * (1 - k) + ink_rgb[None, None, :] * k
    image = Image.fromarray(
        np.dstack([rgb, alpha[..., None] * 255]).astype(np.uint8), mode="RGBA"
    )
    return image.resize(size, Image.LANCZOS)
