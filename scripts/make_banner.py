# OpenSculptBoy
# Apache License, Version 2.0
"""
Write the OpenSculptBoy logo and banner: a character traced from the model floats upward with the
head back, reaching for the sky with the right hand.

- docs/figures/opensculptboy_logo.svg: the mark alone, square, on a transparent background
- docs/figures/opensculptboy_banner.svg: the mark beside the name and the tagline

    uv run --extra examples python scripts/make_banner.py

The pose sets the rotation of each posed bone in the world frame; a bone's local-ref pose
parameter is its world rotation seen from its nearest posed ancestor. The posed mesh is seen
from the front turned by 40 degrees, and its silhouette is traced into an SVG path.
"""

import math
import pathlib

import contourpy
import numpy as np
import roma
import torch
from PIL import Image, ImageDraw, ImageFilter

import anny

FIGURES = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
SKIN = "#e2bca3"
PHENOTYPE = {"age": 0.84, "gender": 0.5, "weight": 0.25, "muscle": 0.5, "height": 0.6}
# The view: the front turned by YAW degrees, the picture tilted by ROLL degrees.
YAW, ROLL = -40.0, -5.0
# Directions are in the model's frame: x toward the figure's left, -y forward, z up.
# The picture's right is (0.77, 0.64, 0) at this yaw.
POSE = {
    # the torso arches back and leans to the figure's left, both spread over the spine: the
    # shoulder line falls 27 degrees to the left, as in the reference picture
    "arch": 45,
    "lean": 36,
    # the legs trail behind, the left knee bent more, the toes pointed
    "hip.R": -30,
    "knee.R": 20,
    "hip.L": -36,
    "knee.L": 28,
    "spread": 4,
    "point": 55,
    # the right arm raised: the upper arm up and out, the forearm upright
    "upper.R": (-0.54, -0.45, 0.75),
    "fore.R": (-0.04, -0.03, 1.0),
    # the left arm out to the side: the upper arm slopes down, the elbow bends softly, the
    # forearm runs nearly level
    "upper.L": (0.81, 0.29, -0.59),
    "fore.L": (1.04, 0.29, -0.17),
    # the raised hand: the fingers up, the thumb side toward the head, relaxed and spread
    "hand.R": {
        "point": (0.23, 0.19, 1.0),
        "index": (0.09, 0.99, 0.0),
        "curl": -12,
        "spread": (-10, -3, 4, 12),
        "thumb": 25,
    },
    # the other hand open, the palm to the front, the fingers along the arm, the thumb up
    "hand.L": {
        "point": (0.96, 0.41, 0.07),
        "index": (0.0, -0.2, 1.0),
        "curl": 5,
        "spread": (-4, -1, 2, 5),
        "thumb": 12,
    },
    # the neck and the head carry on the arch and turn toward the raised hand: the face
    # points at 113 degrees in the picture, just above the fingertips, and a little toward
    # the viewer
    "head": {"extension": 30, "turn": -30},
}
SPINE = ["spine05", "spine04", "spine03", "spine02", "spine01"]
NECK = ["neck01", "neck02", "neck03", "head"]
ARCH_SHARES = [
    0.10,
    0.14,
    0.16,
    0.16,
    0.14,
]  # 70% of the arch; the neck is set with the head
FINGER_PARENTS = {
    2: "metacarpal1",
    3: "metacarpal2",
    4: "metacarpal3",
    5: "metacarpal4",
}


def rot(axis, deg):
    v = torch.zeros(3)
    v["xyz".index(axis)] = math.radians(deg)
    return roma.rotvec_to_rotmat(v)


def bend(deg):
    """a backward bend of the torso, or a forward swing of a leg, in the body's side plane"""
    return rot("x", -deg)


def axis_angle(axis, deg):
    axis = torch.as_tensor(axis, dtype=torch.float32)
    return roma.rotvec_to_rotmat(axis / axis.norm() * math.radians(deg))


def align(a, b):
    """the smallest rotation that takes direction a to direction b"""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    a, b = a / a.norm(), b / b.norm()
    axis = torch.linalg.cross(a, b)
    s, c = axis.norm(), torch.dot(a, b)
    if s < 1e-8:
        return torch.eye(3)
    return roma.rotvec_to_rotmat(axis / s * torch.atan2(s, c))


def frame(f, t):
    """an orthonormal frame (as columns) from a main direction f and a second direction t"""
    f = torch.as_tensor(f, dtype=torch.float32)
    t = torch.as_tensor(t, dtype=torch.float32)
    f = f / f.norm()
    t = t - torch.dot(t, f) * f
    t = t / t.norm()
    return torch.stack([f, t, torch.linalg.cross(f, t)], dim=1)


class Poser:
    def __init__(self):
        self.model = anny.Anny().to(dtype=torch.float32)
        self.labels = self.model.bone_labels
        self.parents = self.model.bone_parents
        with torch.no_grad():
            self.rest = self.model(phenotype_kwargs=PHENOTYPE)["bone_poses"][0]
        c, s = math.cos(math.radians(YAW)), math.sin(math.radians(YAW))
        self.right = torch.tensor([c, -s, 0.0])  # the picture's right
        self.camera = torch.tensor([-s, -c, 0.0])  # toward the viewer

    def joint(self, name):
        return self.rest[self.labels.index(name), :3, 3]

    def hand(self, W, s, spec):
        """the fingers along ``point``, the index side toward ``index``, spread and curled"""
        target = frame(spec["point"], spec["index"])
        f0 = self.joint("finger3-1" + s) - self.joint("wrist" + s)
        t0 = self.joint("finger2-1" + s) - self.joint("finger5-1" + s)
        R = target @ frame(f0, t0).T
        W["wrist" + s] = R
        normal = torch.linalg.cross(target[:, 0], target[:, 1])  # across the palm
        for k, spread in zip((2, 3, 4, 5), spec["spread"]):
            W[FINGER_PARENTS[k] + s] = R
            base = axis_angle(normal, spread) @ R
            rest = self.joint(f"finger{k}-2" + s) - self.joint(f"finger{k}-1" + s)
            side = torch.linalg.cross(
                base @ rest, normal
            )  # the axis the finger curls about
            for i in (1, 2, 3):
                W[f"finger{k}-{i}" + s] = axis_angle(side, i * spec["curl"]) @ base
        thumb = axis_angle(normal, spec["thumb"]) @ R
        for i in (1, 2, 3):
            W[f"finger1-{i}" + s] = thumb

    def head(self, W, extension, turn, shares=(0.3, 0.25, 0.2, 0.25)):
        """the neck and the head carry on the arch of the spine: each bends back by its share
        of ``extension`` and turns about the neck by its share of ``turn`` (degrees, toward
        the raised right hand when negative), in its parent's frame"""
        parent = W["spine01"]
        for bone, share in zip(NECK, shares):
            parent = parent @ rot("z", turn * share) @ bend(extension * share)
            W[bone] = parent

    def world(self, p):
        """the world rotation of every posed bone"""
        W = {}
        arch = lean = 0.0
        for bone, share in zip(SPINE, ARCH_SHARES):
            arch += p["arch"] * share
            lean += p["lean"] * share
            W[bone] = rot("y", lean) @ bend(arch)
        for s in (".L", ".R"):
            hip, knee = p["hip" + s], p["knee" + s]
            spread = rot("y", -p["spread"] if s == ".L" else p["spread"])
            W["upperleg01" + s] = spread @ bend(hip)
            W["lowerleg01" + s] = spread @ bend(
                hip - knee
            )  # the knee folds the shin back
            W["foot" + s] = spread @ bend(hip - knee - p["point"])
            segments = (
                ("upperarm01", "lowerarm01", p["upper" + s]),
                ("lowerarm01", "wrist", p["fore" + s]),
            )
            for a, b, target in segments:
                rest = self.joint(b + s) - self.joint(a + s)
                W[a + s] = align(rest, target)
            self.hand(W, s, p["hand" + s])
        self.head(W, **p["head"])
        return W

    def params(self, W):
        """local-ref pose parameters: each bone's world rotation seen from its posed ancestor"""
        P = torch.eye(4)[None, None].repeat(1, self.model.bone_count, 1, 1)
        for bone, Wb in W.items():
            i = self.labels.index(bone)
            j = self.parents[i]
            while j >= 0 and self.labels[j] not in W:
                j = self.parents[j]
            Wp = W[self.labels[j]] if j >= 0 else torch.eye(3)
            P[0, i, :3, :3] = Wp.T @ Wb
        return P

    def picture(self, p):
        """the posed vertices in the picture (x right, y up), and the mesh's triangles"""
        with torch.no_grad():
            out = self.model(
                pose_parameters=self.params(self.world(p)), phenotype_kwargs=PHENOTYPE
            )
        v = out["vertices"][0].numpy()
        x = v @ self.right.numpy()
        cr, sr = math.cos(math.radians(-ROLL)), math.sin(math.radians(-ROLL))
        xy = np.stack([cr * x - sr * v[:, 2], sr * x + cr * v[:, 2]], axis=1)
        return xy, self.model.get_triangular_faces().numpy()


def simplify(points, tolerance):
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


def outline(points, faces, supersample=8):
    """the silhouette of a mesh already in pixels (x right, y down) as an SVG path"""
    lo = points.min(axis=0) - 2
    size = ((points.max(axis=0) + 2 - lo) * supersample).astype(int) + 1
    image = Image.new("L", tuple(size), 0)
    draw = ImageDraw.Draw(image)
    for tri in ((points - lo) * supersample)[faces]:
        draw.polygon([tuple(p) for p in tri], fill=255)
    image = image.filter(ImageFilter.GaussianBlur(supersample * 0.35))
    mask = np.asarray(image, dtype=np.float32) / 255.0
    gen = contourpy.contour_generator(z=mask, fill_type="OuterOffset")
    polygons, offsets = gen.filled(0.5, 2.0)
    parts = []
    for ring_points, offs in zip(polygons, offsets):
        for a, b in zip(offs[:-1], offs[1:]):
            ring = simplify(
                ring_points[a:b].astype(np.float64) / supersample + lo, 0.15
            )
            if len(ring) >= 3:
                parts.append(
                    "M" + " L".join(f"{p[0]:.1f} {p[1]:.1f}" for p in ring) + "Z"
                )
    return "".join(parts)


def fit(xy, box):
    """``xy`` (y up) scaled into ``box`` (x, y, width, height in pixels, y down), centred"""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    x, y, w, h = box
    s = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
    ox = x + (w - (hi[0] - lo[0]) * s) / 2
    oy = y + (h - (hi[1] - lo[1]) * s) / 2
    return np.stack([ox + (xy[:, 0] - lo[0]) * s, oy + (hi[1] - xy[:, 1]) * s], axis=1)


def svg(width, height, body, desc, background=None):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">OpenSculptBoy</title>',
        f'  <desc id="desc">{desc}</desc>',
    ]
    if background:
        lines.append(f'  <rect width="{width}" height="{height}" fill="{background}"/>')
    return "\n".join(lines + body + ["</svg>"]) + "\n"


def banner_text(x, centre):
    """the name, the tagline and the credit line, left of ``x``, the block centred on ``centre``

    The block runs from the cap top of the name to the baseline of the credit line. Each
    baseline follows from the one above: the descender of the line above, a gap, and the cap
    height of the line below (Georgia italic and Helvetica: caps 0.69 and 0.72 em, the "p"
    of the name 0.22 em below its baseline).
    """
    name, tagline, credit = 78, 18, 12  # font sizes
    first = name * 0.69  # the name's cap height
    second = name * 0.22 + 14 + tagline * 0.72
    third = tagline * 0.21 + 14 + credit * 0.72
    baseline = centre - (first + second + third) / 2 + first
    inset = name * 0.05  # the stem of the italic C starts this far in
    serif = 'font-family="Georgia, \'Times New Roman\', serif" font-style="italic"'
    sans = "font-family=\"'Helvetica Neue', Arial, sans-serif\""
    return [
        f'  <text x="{x:.1f}" y="{baseline:.1f}" fill="#f4efe9" {serif} '
        f'font-size="{name}" letter-spacing="-1">OpenSculptBoy</text>',
        f'  <text x="{x + inset:.1f}" y="{baseline + second:.1f}" fill="#aab2bc" {sans} '
        f'font-size="{tagline}" letter-spacing="0.3">'
        "Open humanoid characters for free, unlimited creativity</text>",
        f'  <text x="{x + inset:.1f}" y="{baseline + second + third:.1f}" fill="#8b949e" '
        f'{sans} font-size="{credit}" letter-spacing="1.8">'
        "BUILT ON THE ANNY BODY MODEL</text>",
    ]


def main():
    xy, faces = Poser().picture(POSE)
    desc = (
        "OpenSculptBoy: a humanoid character traced from the Anny body model floats upward "
        "with the head back, reaching for the sky with the right hand."
    )
    logo = outline(fit(xy, (40, 40, 432, 432)), faces)
    logo_svg = svg(
        512, 512, [f'  <path fill="{SKIN}" fill-rule="evenodd" d="{logo}"/>'], desc
    )
    (FIGURES / "opensculptboy_logo.svg").write_text(logo_svg)

    mark = fit(xy, (34, 34, 170, 172))
    body = [f'  <path fill="{SKIN}" fill-rule="evenodd" d="{outline(mark, faces)}"/>']
    body += banner_text(
        float(mark[:, 0].max()) + 40, (mark[:, 1].min() + mark[:, 1].max()) / 2
    )
    banner_svg = svg(900, 240, body, desc, background="#101317")
    (FIGURES / "opensculptboy_banner.svg").write_text(banner_svg)
    for name, text in (
        ("opensculptboy_logo.svg", logo_svg),
        ("opensculptboy_banner.svg", banner_svg),
    ):
        print(f"{FIGURES / name} {len(text) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
