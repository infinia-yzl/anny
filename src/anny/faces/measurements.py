# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Craniofacial landmarks and measurements of anny, as two anthropometric standards define them.

**Landmarks** (``data/keypoints/craniofacial.json``, placed by
``anny.faces.authoring.landmarks``) use the abbreviations of Farkas, with ``.L`` and ``.R`` for
the figure's left and right:

- the 24 landmarks of the 3D Facial Norms database (Weinberg et al. 2016): ``n``, ``prn``,
  ``sn``, ``ls``, ``sto``, ``li``, ``sl``, ``gn``, and ``en``, ``ex``, ``al``, ``ac``, ``sbal``,
  ``cph``, ``ch``, ``t`` on both sides;
- the landmarks of the calliper measurements and of the ANSUR II survey (Hotzman et al. 2011):
  glabella ``g``, opisthocranion ``op``, vertex ``v``, pogonion ``pg``, and eurion ``eu``,
  zygion ``zy``, gonion ``go``, frontotemporale ``ft``, pupil ``pu``, superaurale ``sa``,
  subaurale ``sba``, preaurale ``pra``, postaurale ``pa``, the most lateral point of the ear
  ``ela`` and the head surface under it ``ehs`` on both sides.

The soft-tissue nasion stands for the sellion of ANSUR II, and the gnathion for its menton.

**Measurements** are in millimetres:

- :data:`TDFN_MEASUREMENTS`: the 34 measurements of 3D Facial Norms, as distances between two
  landmarks (the names of its summary tables);
- :data:`ANSUR_LANDMARK_MEASUREMENTS`: the ANSUR II measurements that two landmarks define;
- :func:`section_measurements`: the ANSUR II circumferences and arcs, from plane sections of
  the rest mesh (the length of the convex outline, as a tape measures it).
"""

from __future__ import annotations

import functools
import json

import numpy as np
import torch

from anny.paths import get_anny_root_dir

# 3D Facial Norms: name -> landmark pair (Farkas' definitions)
TDFN_MEASUREMENTS = {
    "cranbasewidth": ("t.L", "t.R"),
    "upfacedepth_l": ("t.L", "n"),
    "upfacedepth_r": ("t.R", "n"),
    "midfacedepth_l": ("t.L", "sn"),
    "midfacedepth_r": ("t.R", "sn"),
    "lowfacedepth_l": ("t.L", "gn"),
    "lowfacedepth_r": ("t.R", "gn"),
    "morphfaceheight": ("n", "gn"),
    "upfaceheight": ("n", "sto"),
    "lowfaceheight": ("sn", "gn"),
    "incanthwidth": ("en.L", "en.R"),
    "outcanthwidth": ("ex.L", "ex.R"),
    "palpfislength_l": ("en.L", "ex.L"),
    "palpfislength_r": ("en.R", "ex.R"),
    "nasalwidth": ("al.L", "al.R"),
    "subnasalwidth": ("sbal.L", "sbal.R"),
    "nasalpro": ("sn", "prn"),
    "nasalalalength_l": ("ac.L", "prn"),
    "nasalalalength_r": ("ac.R", "prn"),
    "nasalheight": ("n", "sn"),
    "nasalbdglength": ("n", "prn"),
    "labfiswidth": ("ch.L", "ch.R"),
    "philwidth": ("cph.L", "cph.R"),
    "phillength": ("sn", "ls"),
    "uplipheight": ("sn", "sto"),
    "lowlipheight": ("sto", "sl"),
    "upvermheight": ("ls", "sto"),
    "lowvermheight": ("sto", "li"),
    "cutlowlipheight": ("li", "sl"),
    # calliper measurements
    "maxcranwidth": ("eu.L", "eu.R"),
    "minfrntwidth": ("ft.L", "ft.R"),
    "maxfacewidth": ("zy.L", "zy.R"),
    "mandwidth": ("go.L", "go.R"),
    "maxcranlength": ("g", "op"),
}

# ANSUR II (column names of its data files) -> landmark pair; ear measurements average both sides
ANSUR_LANDMARK_MEASUREMENTS = {
    "headlength": ("g", "op"),
    "headbreadth": ("eu.L", "eu.R"),
    "bizygomaticbreadth": ("zy.L", "zy.R"),
    "mentonsellionlength": ("gn", "n"),
    "interpupillarybreadth": ("pu.L", "pu.R"),
    "earlength": (("sa.L", "sba.L"), ("sa.R", "sba.R")),
    "earbreadth": (("pra.L", "pa.L"), ("pra.R", "pa.R")),
}
ANSUR_SECTION_MEASUREMENTS = (
    "headcircumference",
    "bitragionchinarc",
    "bitragionsubmandibulararc",
    "neckcircumference",
)
ANSUR_MEASUREMENTS = (
    tuple(ANSUR_LANDMARK_MEASUREMENTS)
    + ("tragiontopofhead", "earprotrusion")
    + ANSUR_SECTION_MEASUREMENTS
)

# The measurement that sizes each face-shape scale group (anny.models.face_shapes.SCALE_GROUPS):
# the geometric mean of the distances of its landmark pairs
SCALE_GROUP_MEASUREMENTS = {
    "head": (("g", "op"), ("eu.L", "eu.R")),
    "forehead": (("ft.L", "ft.R"), ("n", "sto")),
    "eyes": (("ex.L", "ex.R"),),
    "nose": (("n", "sn"), ("al.L", "al.R")),
    "mouth": (("ch.L", "ch.R"),),
    "chin": (("sn", "gn"), ("go.L", "go.R")),
    "ears": (("sa.L", "sba.L"), ("sa.R", "sba.R")),
}


@functools.lru_cache(maxsize=1)
def craniofacial_landmark_vertices() -> dict[str, list[int]]:
    """landmark -> MakeHuman base-mesh vertices whose mean is the landmark"""
    path = get_anny_root_dir() / "data" / "keypoints" / "craniofacial.json"
    with open(path) as f:
        return json.load(f)


def craniofacial_landmark_weights(vertices_count: int) -> dict[str, torch.Tensor]:
    """landmark -> regression weights over the base-mesh vertices (the format of coco.pth)"""
    out = {}
    for name, idx in craniofacial_landmark_vertices().items():
        w = torch.zeros(vertices_count, dtype=torch.float64)
        w[torch.tensor(idx)] = 1.0 / len(idx)
        out[name] = w
    return out


def _distance(landmarks: torch.Tensor, labels: list[str], a: str, b: str):
    return torch.linalg.norm(
        landmarks[..., labels.index(a), :] - landmarks[..., labels.index(b), :], dim=-1
    )


def face_shape_group_sizes(landmarks: torch.Tensor, labels: list[str]) -> torch.Tensor:
    """size (B, G) of each face-shape scale group, in metres"""
    from anny.models.face_shapes import SCALE_GROUPS

    sizes = []
    for group in SCALE_GROUPS:
        pairs = SCALE_GROUP_MEASUREMENTS[group]
        logs = [torch.log(_distance(landmarks, labels, a, b)) for a, b in pairs]
        sizes.append(torch.exp(torch.stack(logs, -1).mean(-1)))
    return torch.stack(sizes, -1)


def landmark_measurements(
    landmarks: torch.Tensor, labels: list[str]
) -> dict[str, torch.Tensor]:
    """
    The 3D Facial Norms measurements and the landmark-based ANSUR II measurements (mm) of
    landmarks (B, K, 3) of the rest body (Z up).
    """
    out = {}
    for name, (a, b) in TDFN_MEASUREMENTS.items():
        out[name] = 1000 * _distance(landmarks, labels, a, b)
    for name, pairs in ANSUR_LANDMARK_MEASUREMENTS.items():
        if isinstance(pairs[0], str):
            out[name] = 1000 * _distance(landmarks, labels, *pairs)
        else:
            out[name] = 1000 * torch.stack(
                [_distance(landmarks, labels, a, b) for a, b in pairs]
            ).mean(0)
    at = {k: landmarks[..., labels.index(k), :] for k in labels}
    out["tragiontopofhead"] = 1000 * (
        at["v"][..., 2] - 0.5 * (at["t.L"][..., 2] + at["t.R"][..., 2])
    )
    out["earprotrusion"] = (
        1000
        * 0.5
        * (
            (at["ela.L"][..., 0] - at["ehs.L"][..., 0])
            + (at["ehs.R"][..., 0] - at["ela.R"][..., 0])
        )
    )
    return out


# ------------------------------------------------------------------ plane sections
def _section_points(V: np.ndarray, T: np.ndarray, point, normal) -> np.ndarray:
    """points where the edges of triangles T cross the plane (point, normal)"""
    d = (V - point) @ normal
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[T[:, a]], d[T[:, b]]
        cross = (da * db) < 0
        t = da[cross] / (da[cross] - db[cross])
        pa, pb = V[T[cross, a]], V[T[cross, b]]
        pts.append(pa + t[:, None] * (pb - pa))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def _hull(points2: np.ndarray) -> np.ndarray | None:
    """the convex hull of a section, or None when the plane misses the mesh"""
    from scipy.spatial import ConvexHull, QhullError

    if len(points2) < 3:
        return None
    try:
        return points2[ConvexHull(points2).vertices]
    except QhullError:
        return None


def _perimeter(poly: np.ndarray | None) -> float:
    if poly is None:
        return float("nan")
    return float(np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1).sum())


def _project_on_polygon(poly: np.ndarray, p2) -> tuple[int, float, np.ndarray]:
    """closest point of a closed polygon to p2: (edge index, parameter along it, point)"""
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(((p2 - a) * ab).sum(1) / np.maximum((ab**2).sum(1), 1e-30), 0, 1)
    q = a + t[:, None] * ab
    k = int(np.argmin(np.linalg.norm(q - p2, axis=1)))
    return k, float(t[k]), q[k]


def _hull_arc(poly: np.ndarray | None, a2, b2, through2) -> float:
    """length of the closed polygon between the projections of a2 and b2, on the side of the
    projection of through2 (NaN without a section)"""
    if poly is None:
        return float("nan")
    n = len(poly)
    edge_len = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(edge_len)])
    total = cum[-1]

    def position(p2):
        k, t, _ = _project_on_polygon(poly, p2)
        return cum[k] + t * edge_len[k % n]

    pa, pb, pt = position(a2), position(b2), position(through2)
    forward = (pb - pa) % total  # from a to b along the polygon's order
    through = (pt - pa) % total
    return float(forward if through <= forward else total - forward)


def section_measurements(
    vertices: np.ndarray,
    triangles: np.ndarray,
    landmarks: np.ndarray,
    labels: list[str],
    neck_axis: tuple[np.ndarray, np.ndarray],
    head_vertices: np.ndarray,
    pinna_vertices: np.ndarray | None = None,
) -> dict[str, float]:
    """
    The circumferences and arcs of ANSUR II (mm) for one rest body: vertices (V, 3), triangles
    (T, 3), landmarks (K, 3). ``neck_axis`` is (base, top) of the neck; ``head_vertices`` selects
    the head and the neck (above the base of the neck), and the tape passes in front of the
    ``pinna_vertices``. A plane that misses the head gives NaN.
    """
    at = {k: landmarks[labels.index(k)] for k in labels}
    out = {}
    keep = np.zeros(len(vertices), bool)
    keep[head_vertices] = True
    if pinna_vertices is not None:
        keep[pinna_vertices] = False
    head_tri = triangles[keep[triangles].all(1)]

    # head circumference: the plane through glabella and opisthocranion, level across the head
    g, op = at["g"], at["op"]
    normal = np.cross(op - g, np.array([1.0, 0, 0]))
    normal /= np.linalg.norm(normal)
    pts = _section_points(vertices, head_tri, g, normal)
    e1, e2 = _plane_basis(normal)
    out["headcircumference"] = 1000 * _perimeter(
        _hull(np.stack([pts @ e1, pts @ e2], -1))
    )

    # bitragion arcs: planes through both tragions and the chin, or the jaw-neck junction
    tl, tr = at["t.L"], at["t.R"]
    for name, through in (
        ("bitragionchinarc", at["pg"]),
        ("bitragionsubmandibulararc", at["sm"]),
    ):
        normal = np.cross(tr - tl, through - tl)
        normal /= np.linalg.norm(normal)
        pts = _section_points(vertices, head_tri, tl, normal)
        e1, e2 = _plane_basis(normal)
        to2 = lambda p: np.array([p @ e1, p @ e2])  # noqa: E731
        poly = _hull(np.stack([pts @ e1, pts @ e2], -1))
        out[name] = 1000 * _hull_arc(poly, to2(tl), to2(tr), to2(through))

    # neck circumference: the plane perpendicular to the neck, halfway up it
    base, top = neck_axis
    normal = (top - base) / np.linalg.norm(top - base)
    centre = 0.5 * (base + top)
    near = triangles[
        (np.linalg.norm(vertices[triangles].mean(1) - centre, axis=1) < 0.09)
    ]
    pts = _section_points(vertices, near, centre, normal)
    e1, e2 = _plane_basis(normal)
    out["neckcircumference"] = 1000 * _perimeter(
        _hull(np.stack([pts @ e1, pts @ e2], -1))
    )
    return out


class CraniofacialMeasurements:
    """
    The measurements of 3D Facial Norms and ANSUR II (mm) of the rest bodies of an anny model
    built on the MakeHuman mesh (``anny`` or ``makehuman`` topology)::

        measure = CraniofacialMeasurements(model)
        values = measure(model(phenotype_kwargs=...))  # name -> (B,) tensor

    The landmark measurements are differentiable; the circumferences and arcs of ANSUR II come
    from plane sections (``sections=True``, NumPy).
    """

    def __init__(self, model):
        from anny.keypoints import KeypointsRegressor

        self.labels = list(craniofacial_landmark_vertices())
        self.regressor = KeypointsRegressor.craniofacial(model, self.labels)
        self.triangles = model.faces.detach().cpu().numpy()
        self.neck = [model.bone_labels.index(b) for b in ("neck01", "head")]
        self.pinna = _pinna_vertices(model)

    def landmarks(self, output: dict) -> torch.Tensor:
        return self.regressor({"vertices": output["rest_vertices"]})

    def __call__(self, output: dict, sections: bool = True) -> dict[str, torch.Tensor]:
        L = self.landmarks(output)
        out = landmark_measurements(L, self.labels)
        if sections:
            V = output["rest_vertices"].detach().cpu().numpy()
            heads = output["rest_bone_heads"].detach().cpu().numpy()
            Ln = L.detach().cpu().numpy()
            rows = []
            for b in range(len(V)):
                base, top = heads[b, self.neck[0]], heads[b, self.neck[1]]
                head = np.nonzero(V[b, :, 2] > base[2])[0]
                rows.append(
                    section_measurements(
                        V[b],
                        self.triangles,
                        Ln[b],
                        self.labels,
                        (base, top),
                        head,
                        self.pinna,
                    )
                )
            for name in ANSUR_SECTION_MEASUREMENTS:
                out[name] = torch.tensor([r[name] for r in rows], dtype=L.dtype)
        return out


def _pinna_vertices(model) -> np.ndarray:
    """vertices of the model that the MakeHuman ear-wing targets move (the pinnae)"""
    from anny.faces.authoring.base_mesh import read_target

    base = model.base_mesh_vertex_indices.detach().cpu().numpy()
    pinna = set()
    for side in ("l", "r"):
        idx, d = read_target(f"ears/{side}-ear-wing-incr")
        mag = np.linalg.norm(d, axis=1)
        pinna.update(idx[mag > 0.05 * mag.max()].tolist())
    return np.nonzero(np.isin(base, list(pinna)))[0]
