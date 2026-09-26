# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Place the craniofacial landmarks on the MakeHuman base mesh and write
``data/keypoints/craniofacial.json`` (the base-mesh vertices whose mean is each landmark).

Each landmark follows its written definition (3D Facial Norms technical notes; ANSUR II
Measurer's Handbook) on anny's default body, with rules on the outer surface of the mesh:

- midline landmarks from the profile: the most protrusive or the deepest point in a height band
  (glabella, nasion, pronasale, subnasale, labiale superius and inferius, sublabiale, pogonion),
  the lowest point of the chin (gnathion), the top (vertex) and the back (opisthocranion) of the
  head;
- the eye and mouth corners from the edges of the eye and mouth cavities (face segmentation),
  where the eyelid fold continues to its tip;
- the nose landmarks among the vertices that the nose targets move;
- the ear landmarks among the vertices that ``ears/l-ear-scale-incr`` moves;
- the widest points of the head, the face and the jaw, and the forehead, as extremes in bands.

The right side mirrors the left through the symmetry of the base mesh. The landmarks are fixed
vertices (or means of vertices), so they follow every shape of the model.

Usage::

    python -m anny.faces.authoring.landmarks [--sheet review.png]
"""

from __future__ import annotations

import argparse
import functools
import json

import numpy as np

from anny.faces.authoring.base_mesh import (
    base_mesh,
    margin_vertices,
    read_target,
    texture_mask,
)
from anny.paths import get_anny_root_dir

OUTPUT = get_anny_root_dir() / "data" / "keypoints" / "craniofacial.json"


@functools.lru_cache(maxsize=1)
def _ray_mesh():
    import trimesh

    m = base_mesh()
    # the head and the neck alone, so that rays under the chin do not meet the chest
    z_neck = m.V[m.groups["joint-neck"]].mean(0)[2]
    T = m.triangles[(m.V[m.triangles, 2] > z_neck - 0.02).all(1)]
    return trimesh.Trimesh(m.V, T, process=False)


def outer(vertices: np.ndarray, direction) -> np.ndarray:
    """the vertices from which a ray along direction leaves the body without hitting it"""
    mesh = _ray_mesh()
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    origins = base_mesh().V[vertices] + 2e-4 * d
    hit = mesh.ray.intersects_any(origins, np.repeat(d[None], len(vertices), 0))
    return vertices[~hit]


@functools.lru_cache(maxsize=1)
def mirror_map() -> np.ndarray:
    """index of the mirror image (x -> -x) of each base-mesh vertex"""
    from scipy.spatial import cKDTree

    V = base_mesh().V
    dist, idx = cKDTree(V).query(V * np.array([-1.0, 1, 1]))
    assert np.median(dist) < 1e-6, "the base mesh is not symmetric"
    return idx


def _band(vertices, axis, lo, hi):
    P = base_mesh().V[vertices]
    return vertices[(P[:, axis] >= lo) & (P[:, axis] <= hi)]


def _arg(vertices, key):
    P = base_mesh().V[vertices]
    return int(vertices[np.argmin(key(P))])


def place() -> dict[str, list[int]]:
    """landmark -> base-mesh vertices (the landmark is their mean)"""
    m = base_mesh()
    V = m.V
    body = m.body_vertices
    eye = V[m.groups["helper-l-eye"]].mean(0)
    z_eye = eye[2]
    head = body[V[body, 2] > V[m.groups["joint-neck"]].mean(0)[2]]
    midline = head[np.abs(V[head, 0]) < 1e-4]
    front = outer(midline, [0, -1, 0])
    L: dict[str, list[int]] = {}

    # ---- midline profile
    L["v"] = [_arg(head, lambda P: -P[:, 2])]
    L["g"] = [_arg(_band(front, 2, z_eye, z_eye + 0.05), lambda P: P[:, 1])]
    prn = _arg(_band(front, 2, z_eye - 0.07, z_eye - 0.01), lambda P: P[:, 1])
    L["prn"] = [prn]
    z_prn, z_g = V[prn, 2], V[L["g"][0], 2]
    L["n"] = [_arg(_band(front, 2, z_prn + 0.015, z_g), lambda P: -P[:, 1])]
    # stomion: the deepest point of the profile where the lips meet (the mouth cavity edge)
    mouth = margin_vertices(m, "mouth_cavity")
    z_fissure = V[mouth[np.abs(V[mouth, 0]) < 1e-4], 2].mean()
    sto = _arg(
        _band(front, 2, z_fissure - 0.004, z_fissure + 0.004), lambda P: -P[:, 1]
    )
    L["sto"] = [sto]
    z_sto = V[sto, 2]
    sn = _arg(_band(front, 2, z_sto + 0.005, z_prn - 0.003), lambda P: -P[:, 1])
    L["sn"] = [sn]
    # the red of the lips (MakeHuman's lip mask): labiale superius and inferius lie on its border
    lips = np.nonzero(texture_mask("mpfb_lips") > 0.5)[0]
    lip_front = np.intersect1d(front, lips)
    L["ls"] = [_arg(_band(lip_front, 2, z_sto, V[sn, 2]), lambda P: -P[:, 2])]
    li = _arg(_band(lip_front, 2, z_sto - 0.02, z_sto), lambda P: P[:, 2])
    L["li"] = [li]
    pg = _arg(_band(front, 2, V[li, 2] - 0.05, V[li, 2] - 0.012), lambda P: P[:, 1])
    L["pg"] = [pg]
    L["sl"] = [_arg(_band(front, 2, V[pg, 2], V[li, 2]), lambda P: -P[:, 1])]
    below = outer(midline, [0, 0, -1])
    chin = _band(below, 1, V[pg, 1], V[pg, 1] + 0.025)
    gn = _arg(chin, lambda P: P[:, 2])
    L["gn"] = [gn]
    # the junction of the jaw and the neck: where the profile under the chin turns down the neck
    under = _band(below, 1, V[gn, 1], V[gn, 1] + 0.06)
    under = under[V[under, 2] > V[gn, 2] - 0.03]
    L["sm"] = [
        _arg(under, lambda P: -(P[:, 1] - V[gn, 1]) + 2.0 * (V[gn, 2] - P[:, 2]))
    ]
    back = outer(midline, [0, 1, 0])
    L["op"] = [_arg(_band(back, 2, z_eye - 0.02, V[L["v"][0], 2]), lambda P: -P[:, 1])]

    # ---- left side (x > 0), mirrored below
    left_head = head[V[head, 0] > 1e-4]
    lateral = outer(left_head, [1, 0, 0])

    # eyes: the eyelid margin (edge of the eye cavity); the medial fold continues to its tip
    margin = margin_vertices(m, "eye_cavity.L")
    L["ex.L"] = [_arg(margin, lambda P: -P[:, 0])]
    medial_end = margin[np.argmin(V[margin, 0])]
    near = left_head[
        (np.abs(V[left_head, 2] - V[medial_end, 2]) < 0.003)
        & (V[left_head, 0] < V[medial_end, 0])
        & (V[left_head, 0] > V[medial_end, 0] - 0.008)
        & (np.abs(V[left_head, 1] - V[medial_end, 1]) < 0.008)
    ]
    fold = _fold_vertices(near)
    L["en.L"] = [_arg(fold, lambda P: P[:, 0])] if len(fold) else [int(medial_end)]
    L["pu.L"] = [int(i) for i in m.groups["helper-l-eye"]]

    # mouth corners
    L["ch.L"] = [_arg(mouth, lambda P: -P[:, 0])]
    # crista philtri: the peak of Cupid's bow, the highest point of the upper lip's red beside
    # the midline
    upper_red = _band(
        _band(np.intersect1d(outer(left_head, [0, -1, 0]), lips), 2, z_sto, V[sn, 2]),
        0,
        0.002,
        0.010,
    )
    L["cph.L"] = [_arg(upper_red, lambda P: -P[:, 2])]

    # nose: the ala is what nose-flaring moves
    ala = _target_vertices("nose/nose-flaring-incr", 0.05, left_head)
    al = _arg(ala, lambda P: -P[:, 0])
    L["al.L"] = [al]
    # alar curvature point: the back of the alar crease, the most posterior point of the ala
    L["ac.L"] = [_arg(_band(ala, 0, 0.6 * V[al, 0], 1), lambda P: -P[:, 1])]
    # subalare: the lowest point of the alar base, where it meets the upper lip
    base = _band(outer(ala, [0, 0, -1]), 0, 0.35 * V[al, 0], 0.65 * V[al, 0])
    L["sbal.L"] = [_arg(base, lambda P: P[:, 2])]

    # ear: the pinna is what ear-wing moves
    pinna = _target_vertices("ears/l-ear-wing-incr", 0.05, body)
    L["sa.L"] = [_arg(pinna, lambda P: -P[:, 2])]
    L["sba.L"] = [_arg(pinna, lambda P: P[:, 2])]
    L["pa.L"] = [_arg(pinna, lambda P: -P[:, 1])]
    L["ela.L"] = [_arg(outer(pinna, [1, 0, 0]), lambda P: -P[:, 0])]
    z_mid = 0.5 * (V[pinna, 2].min() + V[pinna, 2].max())
    y_front = V[pinna, 1].min()
    # tragion: the notch above the tragus, the front of the ear just above its middle
    ear = _target_vertices("ears/l-ear-scale-incr", 0.05, body)
    tragus = _band(
        _band(outer(ear, [1, 0, 0]), 2, z_mid, z_mid + 0.006), 1, y_front - 0.004, 1
    )
    L["t.L"] = [_arg(tragus, lambda P: P[:, 1])]
    # preaurale: the front of the pinna where it joins the face, at the height of the tragion
    z_t = V[L["t.L"][0], 2]
    L["pra.L"] = [_arg(_band(pinna, 2, z_t - 0.006, z_t + 0.006), lambda P: P[:, 1])]
    # the head surface under the most lateral point of the ear (hidden by the ear from the side)
    ela = V[L["ela.L"][0]]
    beside = np.setdiff1d(left_head, pinna)
    under_ear = beside[
        (np.abs(V[beside, 2] - ela[2]) < 0.01) & (np.abs(V[beside, 1] - ela[1]) < 0.015)
    ]
    L["ehs.L"] = [_arg(under_ear, lambda P: -P[:, 0])]

    # widest points, away from the ears
    t = V[L["t.L"][0]]
    skull = np.setdiff1d(lateral, pinna)
    L["eu.L"] = [
        _arg(_band(skull, 2, z_eye + 0.01, V[L["v"][0], 2] - 0.02), lambda P: -P[:, 0])
    ]
    # zygion: the widest point of the zygomatic arch, 15 to 35 mm in front of the tragion (the
    # face keeps widening back to the tragus, over the jaw joint, where the arch has ended)
    cheek = _band(_band(skull, 2, z_eye - 0.04, z_eye), 1, t[1] - 0.035, t[1] - 0.015)
    L["zy.L"] = [_arg(cheek, lambda P: -P[:, 0])]
    jaw = _band(_band(skull, 2, V[gn, 2], z_sto - 0.005), 1, t[1] - 0.035, t[1] + 0.005)
    # gonion: the corner of the jaw, the widest point low and far back on the jaw line
    L["go.L"] = [
        _arg(
            jaw,
            lambda P: -(P[:, 0] + 0.5 * (P[:, 1] - t[1]) - 0.5 * (P[:, 2] - V[gn, 2])),
        )
    ]
    # frontotemporale: where the forehead turns into the temple, 2 cm above glabella
    fore = _band(outer(skull, [1, -1, 0]), 2, z_g + 0.016, z_g + 0.024)
    fore = fore[V[fore, 1] < V[L["eu.L"][0], 1]]
    L["ft.L"] = [_arg(fore, lambda P: -(P[:, 0] - P[:, 1]))]

    mirror = mirror_map()
    for name in [k for k in L if k.endswith(".L")]:
        L[name[:-2] + ".R"] = [int(mirror[i]) for i in L[name]]
    return L


def _target_vertices(target: str, fraction: float, within: np.ndarray) -> np.ndarray:
    """vertices that a target moves by more than a fraction of its largest move"""
    idx, d = read_target(target)
    mag = np.linalg.norm(d, axis=1)
    return np.intersect1d(idx[mag > fraction * mag.max()], within)


def _fold_vertices(candidates: np.ndarray) -> np.ndarray:
    """vertices of body faces that face away from the viewer (the inside of an eyelid fold)"""
    m = base_mesh()
    V = m.V
    quads = m.quads[np.isin(m.quads, candidates).any(1)]
    n = np.cross(V[quads[:, 2]] - V[quads[:, 0]], V[quads[:, 3]] - V[quads[:, 1]])
    inward = quads[n[:, 1] > 0]
    return np.intersect1d(np.unique(inward), candidates)


def review_sheet(landmarks: dict[str, list[int]], path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from anny.faces.authoring.render import draw

    m = base_mesh()
    pts = {k: m.V[v].mean(0) for k, v in landmarks.items()}
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    head = ((-0.1, 0.1), (0.58, 0.86))
    draw(axes[0, 0], m.V, m.triangles, "front", pts, box=head)
    draw(axes[0, 1], m.V, m.triangles, "left", pts, box=((-0.03, 0.2), (0.58, 0.86)))
    draw(
        axes[1, 0],
        m.V,
        m.triangles,
        "front",
        pts,
        box=((-0.05, 0.05), (0.63, 0.75)),
    )
    draw(axes[1, 1], m.V, m.triangles, "left", pts, box=((0.0, 0.12), (0.64, 0.79)))
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sheet", default=None, help="write a review sheet (PNG)")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    landmarks = place()
    with open(args.output, "w") as f:
        json.dump(landmarks, f, indent=1)
        f.write("\n")
    print(f"{len(landmarks)} landmarks -> {args.output}")
    if args.sheet:
        review_sheet(landmarks, args.sheet)


if __name__ == "__main__":
    main()
