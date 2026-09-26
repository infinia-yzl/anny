# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Build the data of the viewer page (``viewer/build``): raw buffers and a manifest that
``viewer/build.mjs`` encodes and puts into ``viewer/dist/anny_viewer.html``.

The page draws anny in the legacy frame of the authoring rig (``anny.poses.authoring.rig``):
Y up, X toward the figure's left, Z forward, with anny's default body scaled uniformly so that
its eyes and its floor sit where the legacy figure had them. This keeps the tuning of the legacy
shaders, lights and cameras. Every shape the page shows comes from anny:

- the phenotype sliders use anny's coefficients (:mod:`anny.viewer.export`); the shape space of
  anny's default sliders is exact with 128 components;
- the face-shape sliders add their sparse offsets on the coarse body, scaled with the size of the
  head as in anny (:mod:`anny.models.face_shapes`), and "Random face" samples the calibrated
  distribution of :mod:`anny.faces.distribution`;
- the skeleton is anny's rig, its joints follow the coefficients;
- the fine body is the mixed Catmull-Clark subdivision of anny's body, with the detail layers of
  :mod:`anny.viewer.geometry` scaled with the local size of the body;
- the correctives, the hair and the poses come from :mod:`anny.correctives`, :mod:`anny.hair`
  and :mod:`anny.poses`.

Slow stages (the groom, the bakes) are cached under ``ANNY_CACHE_DIR/viewer``.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch

from anny.paths import get_anny_cache_path
from anny.poses.authoring import posing as P
from anny.poses.authoring.rig import ANNY_TO_LEGACY, top_weights

from . import export
from .geometry import fine_body, vertex_normals

REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "viewer" / "build"
# anny's phenotype sliders of the page: the default ones and the three race phenotypes, whose
# values mix by their shares (anny's default, 0.5 each, is an equal mix)
SLIDERS = [
    "gender",
    "age",
    "muscle",
    "weight",
    "height",
    "proportions",
    "african",
    "asian",
    "caucasian",
]
M = ANNY_TO_LEGACY


def cache_dir() -> pathlib.Path:
    d = get_anny_cache_path() / "viewer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached(name, make, version=1):
    path = cache_dir() / f"{name}_v{version}.npz"
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    t0 = time.time()
    out = make()
    np.savez(path, **out)
    print(f"{name}: {time.time() - t0:.0f} s")
    return out


class Packer:
    """raw buffers and a manifest, as the legacy pack_data.py wrote them"""

    def __init__(self, out_dir):
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        for f in self.out.glob("*.raw"):
            f.unlink()
        self.man = {"buffers": []}

    def add(self, name, arr, kind, count, stride, **meta):
        raw = np.ascontiguousarray(arr).tobytes()
        pad = (-len(raw)) % 4 if kind == "raw" else 0
        raw += b"\0" * pad
        assert len(raw) == count * stride + pad, (name, len(raw), count, stride)
        (self.out / f"{name}.raw").write_bytes(raw)
        self.man["buffers"].append(
            dict(name=name, kind=kind, count=int(count), stride=int(stride), **meta)
        )

    def write(self):
        with open(self.out / "manifest.json", "w") as f:
            json.dump(self.man, f)


def q16(P_, lo, hi):
    return np.clip(np.round((P_ - lo) / (hi - lo) * 65535), 0, 65535).astype(np.uint16)


def u8(x, lo=0.0, hi=1.0):
    return np.clip(np.round((np.asarray(x) - lo) / (hi - lo) * 255), 0, 255).astype(
        np.uint8
    )


def s8(x):
    return np.clip(np.round(np.asarray(x) * 127), -127, 127).astype(np.int8)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def oct_enc(n):
    n = n / (np.abs(n).sum(1, keepdims=True) + 1e-12)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    ox = np.where(z >= 0, x, (1 - np.abs(y)) * np.sign(x + 1e-12))
    oy = np.where(z >= 0, y, (1 - np.abs(x)) * np.sign(y + 1e-12))
    return np.stack([ox, oy], 1)


def skin_record(si, sw):
    w8 = np.clip(np.round(sw * 255), 0, 255).astype(np.int32)
    w8[:, 0] += 255 - w8.sum(1)
    return np.concatenate([si.astype(np.uint8), w8.astype(np.uint8)], 1)


# ------------------------------------------------------------------ stages
def grow_brows_lashes(body):
    from anny.hair.authoring import brows_lashes

    bl = brows_lashes.grow(body.V, body.T, body.N, body.eye_centers)
    return dict(brows=bl["brows"], lashes=bl["lashes"])


def attributes(body, hair):
    from .attributes import eye_occlusion, skin_attributes

    attrs, skin = skin_attributes(body, hair)
    eye_ao, eye_info = eye_occlusion(body)
    return dict(
        attrs,
        skin=json.dumps(skin),
        eye_ao_l=eye_ao["l"],
        eye_ao_r=eye_ao["r"],
        eye_info=json.dumps(eye_info),
    )


def fine_skin_weights(body):
    """anny's weights through the subdivision, the eight largest per fine vertex"""
    W = body.subdivision(body.dense_weights)
    return dict(zip(("si", "sw"), top_weights(W, 8)))


def skin_lut():
    from .bake import make_lut

    lut = make_lut()
    img = (np.sqrt(np.clip(lut, 0, 1)) * 255 + 0.5).astype(np.uint8)[::-1]
    return dict(lut=img)


# ------------------------------------------------------------------ shape space
def shape_space(model, compact):
    """anny's phenotype space on the compact coarse vertices, in the legacy frame"""
    rig = P.RIG
    tables = export.phenotype_tables(model)
    comp = export.compress_shapes(model, compact, max_error=1e-4, sliders=SLIDERS)
    components = rig.scale * comp["components"] @ M.T  # rotation and scale only
    template = rig.to_legacy(model.template_vertices.detach().numpy()[compact])
    joints = export.joint_tables(model)
    joint_template = rig.to_legacy(joints["template"])
    joint_blend = rig.scale * joints["blendshapes"] @ M.T
    return dict(
        tables=tables,
        report=comp["report"],
        components=components,
        projection=comp["projection"],
        template=template,
        bones=joints["bones"],
        parents=joints["parents"],
        joint_template=joint_template,
        joint_blend=joint_blend,
    )


def corrective_shapes(body, compact_of, weights):
    """
    The corrective shapes of anny on the fine body, in the legacy frame, with normal changes.

    The normal change of a shape makes the skinned normal match the corrected surface at the
    key pose of the shape: the normals of the posed, corrected surface turn back into the rest
    pose through the blended bone rotation of each vertex. (On the rest body alone, a large
    shape can fold the surface, and its normals there would point the wrong way.)
    """
    from anny.correctives import DATA_DIR
    from anny.correctives.authoring import train
    from safetensors.torch import load_file

    spec = json.load(open(DATA_DIR / "soft_tissue.json"))
    tensors = load_file(str(DATA_DIR / "soft_tissue.safetensors"))
    base_to_anny = {
        int(b): i for i, b in enumerate(P.preview_mesh()["coarse"]["base_index"])
    }
    keys = {}
    for j, J in train.JOINTS.items():
        names = J["keys"] if J["kind"] == "hinge" else list(J["targets"])
        for k, nm in zip(names, train.shape_names(j)):
            for side in "LR":
                keys[f"{nm}.{side}"] = (j, k, side)
    V0, T = body.V, body.T
    N0 = vertex_normals(V0, T)
    si, sw = weights["si"], weights["sw"]
    n_coarse = len(P.preview_mesh()["coarse"]["V"])
    shapes = []
    for info in spec["shapes"]:
        name = info["name"]
        base = tensors[name + ".indices"].numpy()
        offsets = tensors[name + ".offsets"].numpy().astype(np.float64)
        anny_idx = np.array([base_to_anny[int(b)] for b in base])
        D = np.zeros((n_coarse, 3))
        D[anny_idx] = P.RIG.scale * offsets @ M.T
        Df = body.subdivision(D)
        moved = np.linalg.norm(Df, axis=1) > 2e-5
        # region of the normals: the moved vertices and their neighbours
        region = moved.copy()
        region[T[region[T].any(1)].ravel()] = True
        # every triangle around the region, skinned at the key pose with the shape on
        tri = T[region[T].any(1)]
        verts = np.unique(tri)
        F = P.fk(train.key_pose(*keys[name]))
        posed = np.zeros_like(V0)
        posed[verts] = P.skin(F, V0[verts] + Df[verts], si[verts], sw[verts])
        face = np.cross(
            posed[tri[:, 1]] - posed[tri[:, 0]], posed[tri[:, 2]] - posed[tri[:, 0]]
        )
        acc = np.zeros_like(V0)
        for c in range(3):
            np.add.at(acc, tri[:, c], face)
        ids = np.nonzero(region)[0]
        # the blended bone rotation of each vertex, and the posed normal turned back through it
        A = np.einsum("nk,nkij->nij", sw[ids], F[0][si[ids]])
        n_rest = np.linalg.solve(A, acc[ids][:, :, None])[:, :, 0]
        n_rest /= np.maximum(np.linalg.norm(n_rest, axis=1, keepdims=True), 1e-12)
        dN = np.zeros_like(V0)
        dN[ids] = n_rest - N0[ids]
        keep = np.nonzero(region & (moved | (np.linalg.norm(dN, axis=1) > 4e-3)))[0]
        shapes.append(
            dict(
                name=name,
                idx=keep,
                D=Df[keep],
                N=dN[keep],
                support=np.array([compact_of[i] for i in anny_idx]),
                radius=info["reference_radius"] * P.RIG.scale,
            )
        )
    drivers = legacy_drivers(spec)
    return shapes, drivers


def legacy_drivers(spec):
    """the drivers with directions in the legacy axes"""
    out = json.loads(json.dumps(spec))
    for j in out["joints"]:
        if j["type"] == "cone":
            for t in j["targets"]:
                t["dir"] = [round(float(x), 6) for x in M @ np.array(t["dir"])]
    out.pop("shapes", None)
    out.pop("fit", None)
    return out


def motion_data():
    """the pose library in the legacy axes; root offsets and bounds relative to the hip height"""
    import anny.poses

    lib = anny.poses.library()
    q = lib.rotations.numpy().astype(np.float64)
    q_leg = np.concatenate([q[..., :3] @ M.T, q[..., 3:]], -1)
    root = lib.root_offsets.numpy().astype(np.float64) @ M.T
    clips = [dict(e) for e in lib.meta["entries"]]
    # bounds of each entry on the authoring body (lowest and highest joint above the floor,
    # widest joint), relative to the hip height, for the framing of the Body view
    hip = P.RIG.hip_height * P.RIG.scale
    bone_index = [lib.bones.index(n) for n in P.NAMES]
    for c in clips:
        lo, hi, wx = 1e9, -1e9, 0.0
        frames = range(c["start"], c["start"] + c["count"])
        for f in list(frames)[:: max(1, c["count"] // 12)]:
            pose = P.Pose()
            pose.q = q_leg[f][bone_index]
            pose.root = root[f] * hip
            Rw, Pw = P.fk(pose)
            tails = np.einsum("nij,nj->ni", Rw, P.TAILS - P.HEADS) + Pw
            pts = np.concatenate([Pw, tails])
            lo = min(lo, pts[:, 1].min())
            hi = max(hi, pts[:, 1].max())
            wx = max(wx, np.abs(pts[:, 0]).max())
        c["bounds"] = [
            round(float(lo - P.FLOOR) / hip, 4),
            round(float(hi + 0.05 - P.FLOOR) / hip, 4),
            round(float(wx + 0.03) / hip, 4),
        ]
    return q_leg, root, clips, lib.bones


# ------------------------------------------------------------------ build
def build(out_dir=DEFAULT_OUT, verbose=True):
    import anny
    from anny.hair import closest_triangles

    t0 = time.time()
    rig = P.RIG
    body = fine_body()
    hair = cached("brows_lashes", lambda: grow_brows_lashes(body))
    A = cached("attributes", lambda: attributes(body, hair), version=2)
    weights = cached("weights", lambda: fine_skin_weights(body))
    lut = cached("lut", skin_lut)
    if verbose:
        print(f"stages ready: {time.time() - t0:.0f} s")

    model = anny.Anny(topology="anny-quads", face_shapes="all", phenotypes="all").to(
        dtype=torch.float64
    )
    coarse = rig.preview["coarse"]
    base_index = coarse["base_index"]
    body_quads = coarse[
        "quads"
    ]  # the body faces of anny, in the order of the fine body
    body_used = np.unique(body_quads)
    eye_ids = np.nonzero((base_index >= 14598) & (base_index < 14742))[0]
    compact = np.concatenate([body_used, eye_ids])
    compact_of = -np.ones(len(base_index), np.int64)
    compact_of[compact] = np.arange(len(compact))
    space = shape_space(model, compact)

    pk = Packer(out_dir)
    man = pk.man
    V, T = body.V, body.T.astype(np.uint32)
    nV = len(V)
    lo = V.min(0) - 0.05
    hi = V.max(0) + 0.05

    # ---------------- fine body: the legacy vertex record, positions of anny's default body
    rec = np.zeros((nV, 32), np.uint8)
    rec[:, 0:6] = q16(V, lo, hi).view(np.uint8).reshape(nV, 6)
    rec[:, 6:9] = u8(lin2srgb(A["albedo"]))
    aA, aB, aC, aD, aE = A["attrA"], A["attrB"], A["attrC"], A["attrD"], A["attrE"]
    rec[:, 9] = u8(aA[:, 0])
    rec[:, 10] = u8(aB[:, 0])
    rec[:, 11] = u8(aB[:, 2])
    rec[:, 12] = u8(aB[:, 3])
    rec[:, 13] = u8(aC[:, 0])
    rec[:, 14] = u8(aC[:, 2])
    rec[:, 15] = u8(np.clip(aA[:, 1], 0, 0.4), 0, 0.4)
    rec[:, 16:18] = s8(oct_enc(A["nsmooth"])).view(np.uint8)
    rec[:, 18] = u8(np.sqrt(np.clip(aD[:, 0] / 0.03, 0, 1)))
    rec[:, 19] = u8(aD[:, 1])
    rec[:, 20] = u8(aD[:, 2])
    rec[:, 21] = u8(aD[:, 3])
    rec[:, 22] = u8(aE[:, 0])
    rec[:, 23] = u8(aE[:, 1])
    rec[:, 24:28] = u8(A["wear"])
    rec[:, 28:32] = u8(A["cover"])
    pk.add("head_v", rec, "vertex", nV, 32, lo=lo.tolist(), hi=hi.tolist())
    pk.add("head_i", T.reshape(-1), "index", T.size, 4)
    # the row of each fine vertex in the last level of the subdivision, for the page's
    # numbering of the coarse body (the body vertices alone, see coarse_quads)
    body_of = -np.ones(len(base_index), np.int64)
    body_of[body_used] = np.arange(len(body_used))
    pk.add(
        "head_row",
        body.subdivision.renumbered_rows(body_of, len(body_used)).astype(np.uint32),
        "vertex",
        nV,
        4,
    )
    # detail layers in the (t1, t2, n) frames of the smooth surface, steps of 0.01 mm
    # (t1, t2, n) without the relief, then the height of the relief along n, which the page fades with the
    # sliders (the record of 8 bytes suits meshopt, which encodes strides that are multiples of 4)
    base = body.detail_local.copy()
    base[:, 2] -= body.relief_height
    dl = np.round(np.concatenate([base, body.relief_height[:, None]], axis=1) / 1e-5)
    assert np.abs(dl).max() < 32767
    pk.add("head_detail", dl.astype(np.int16), "vertex", nV, 8, step=1e-5)
    pk.add(
        "head_skin",
        skin_record(weights["si"], weights["sw"]),
        "vertex",
        nV,
        16,
        influences=8,
    )

    man["skin"] = json.loads(str(A["skin"]))
    man["skin"]["wearables"] = []

    # ---------------- coarse body and anny's shape space
    quads = compact_of[body_quads].astype(np.uint32)
    from anny.utils.subdivision import faces_weighted_to_bones

    region = faces_weighted_to_bones(model, body_quads).astype(np.uint8)
    pk.add("coarse_quads", quads.reshape(-1), "raw", quads.size * 4, 1)
    pk.add("coarse_region", region, "raw", len(region), 1)
    comp = space["components"]
    comp_scale = np.abs(comp).reshape(len(comp), -1).max(1) / 32767
    cq = np.round(comp / comp_scale[:, None, None]).astype(np.int16)
    proj = space["projection"].astype(np.float32)
    # the rounding of the components, centred on anny's default body: the template takes the rounding error of the
    # default body, so each body keeps only the rounding error of its difference to the default body (the error is
    # linear in the coefficients)
    a0 = proj.astype(np.float64) @ export.coefficients(model, {})
    rounding = comp - cq * comp_scale[:, None, None]
    template = space["template"] + np.einsum("k, knd -> nd", a0, rounding)
    pk.add("coarse_template", template.astype(np.float32), "raw", template.size * 4, 1)
    # one record of 4 values (x, y, z and a zero pad) per component and vertex: meshopt encodes them as
    # vertices, and the neighbouring vertices of a component move alike
    cq = np.concatenate([cq, np.zeros(cq.shape[:2] + (1,), np.int16)], 2).reshape(-1, 4)
    pk.add("shape_components", cq, "vertex", len(cq), 8)
    pk.add("shape_projection", proj, "raw", proj.size * 4, 1)
    jt = space["joint_template"].astype(np.float32)
    jb = space["joint_blend"].astype(np.float32)
    pk.add("joint_template", jt, "raw", jt.size * 4, 1)
    pk.add("joint_blend", jb, "raw", jb.size * 4, 1)
    # anny's skinning weights on the coarse body (for grounding the poses in the page)
    csi, csw = top_weights(coarse["W"][compact], 8)
    pk.add("coarse_skin", skin_record(csi, csw), "raw", len(compact) * 16, 1)
    man["shape"] = dict(
        tables=space["tables"],
        sliders=SLIDERS,
        components=int(len(comp)),
        component_scale=comp_scale.tolist(),
        blend_shapes=int(proj.shape[1]),
        coarse_vertices=int(len(compact)),
        body_vertices=int(len(body_used)),
        eye_vertices={
            "l": [int(len(body_used)), int(len(body_used) + 72)],
            "r": [int(len(body_used) + 72), int(len(compact))],
        },
        report=space["report"],
        frame=dict(
            scale=rig.scale,
            offset=rig.offset.tolist(),
            floor=P.FLOOR,
            axes="x left, y up, z forward",
        ),
        base_level=body.subdivision.base_level,
        fine_vertices=int(nV),
        default_eye_sphere={
            k: [v[0].tolist(), float(v[1])] for k, v in body.eye_sphere.items()
        },
        eye_offset=[0.0, -0.0005, 0.0012],
    )

    # ---------------- face shapes: sparse offsets on the coarse body and bone-head deltas (legacy
    # frame), the landmarks that size the scale groups (anny's frame: only their ratios matter),
    # and the face-shape distribution
    ft = export.face_tables(model)
    fo = export.face_offsets(model, compact)
    off = rig.scale * fo["offsets"] @ M.T
    qo = np.round(off / 1e-5)
    assert np.abs(qo).max() < 32767
    rec = np.zeros((len(qo), 4), np.int16)
    rec[:, :3] = qo
    pk.add("face_ids", fo["ids"], "raw", fo["ids"].size * 4, 1)
    pk.add("face_offsets", rec, "vertex", len(rec), 8)
    fb = (rig.scale * fo["bone_deltas"] @ M.T).astype(np.float32)
    pk.add("face_bones", fb, "raw", fb.size * 4, 1)
    lm = export.face_landmark_tables(model, ft["landmarks"])
    lt = lm["template"].astype(np.float32)
    lb = lm["blendshapes"].astype(np.float32)
    pk.add("face_lm_template", lt, "raw", lt.size * 4, 1)
    pk.add("face_lm_blend", lb, "raw", lb.size * 4, 1)
    from anny.faces.distribution import DEFAULT_PATH as FACE_PRIOR

    man["shape"]["face"] = dict(
        ft, starts=fo["starts"], counts=fo["counts"], bones=fo["bones"], step=1e-5
    )
    if FACE_PRIOR.exists():
        prior_meta, factors = export.face_prior_tables(model)
        pk.add("face_prior", factors, "raw", factors.size * 4, 1)
        man["shape"]["face"]["prior"] = prior_meta

    # ---------------- rig and motion
    q, root, clips, bones = motion_data()
    assert bones == space["bones"], (
        "the pose library and the rig name the bones differently"
    )
    mq = np.clip(np.round(q * 32767), -32767, 32767).astype(np.int16)
    pk.add(
        "motion_q",
        mq.reshape(-1).view(np.uint8),
        "raw",
        mq.size * 2,
        1,
        frames=int(mq.shape[0]),
        bones=int(mq.shape[1]),
    )
    pk.add(
        "motion_root",
        root.astype(np.float32).reshape(-1).view(np.uint8),
        "raw",
        root.size * 4,
        1,
    )
    man["rig"] = dict(names=space["bones"], parents=space["parents"])
    man["motion"] = clips
    import anny.poses

    seat = [
        int(compact_of[i])
        for i in anny.poses.library().seat_vertices
        if compact_of[i] >= 0
    ]
    man["stool"] = dict(
        seat_vertices=seat, forward=anny.poses.library().meta["stool_forward"]
    )

    # ---------------- correctives (fine level, both sides)
    shapes, drivers = corrective_shapes(body, compact_of, weights)
    recs, entries, off = [], [], 0
    for s in shapes:
        qd = np.round(s["D"] / 1e-5)
        assert np.abs(qd).max() < 32767
        r = np.zeros((len(s["idx"]), 16), np.uint8)
        r[:, 0:4] = s["idx"].astype("<u4").view(np.uint8).reshape(-1, 4)
        r[:, 4:10] = qd.astype("<i2").view(np.uint8).reshape(-1, 6)
        r[:, 10:13] = (
            np.clip(np.round(s["N"] / 0.01), -127, 127)
            .astype(np.int8)
            .view(np.uint8)
            .reshape(-1, 3)
        )
        recs.append(r)
        entries.append(
            dict(
                name=s["name"],
                start=off,
                count=int(len(s["idx"])),
                radius=s["radius"],
                support_start=0,
                support_count=int(len(s["support"])),
            )
        )
        off += len(s["idx"])
    support = np.concatenate([s["support"] for s in shapes]).astype(np.uint32)
    so = 0
    for e, s in zip(entries, shapes):
        e["support_start"] = so
        so += len(s["support"])
    pk.add(
        "corr",
        np.concatenate(recs),
        "sparse",
        off,
        16,
        body="head",
        fields=1,
        sort="shapes",
    )
    pk.add("corr_support", support, "raw", support.size * 4, 1)
    man["correctives"] = dict(step=1e-5, nstep=0.01, shapes=entries, spec=drivers)

    # ---------------- hair: the scalp layout and the styles (anny.hair.styles), bound to the skin
    man["hair"] = hair_buffers(pk, body, weights, len(space["bones"]))

    # ---------------- brows and lashes: strands of anny's default body, bound to the skin
    allpts = np.concatenate(
        [hair["brows"].reshape(-1, 3), hair["lashes"].reshape(-1, 3)]
    )
    slo, shi = allpts.min(0) - 1e-4, allpts.max(0) + 1e-4
    for name, S, ao_, step, mx in (
        ("brows", hair["brows"], np.full(hair["brows"].shape[:2], 0.75), 0.0012, 7),
        ("lashes", hair["lashes"], np.full(hair["lashes"].shape[:2], 0.85), 0.0012, 8),
    ):
        pack_strands(pk, name, S, ao_, slo, shi, step, mx)
        pk.add(
            name + "_bind",
            bind_record(*closest_triangles(S[:, 0], V, body.T)[:2], body.T),
            "sparse",
            len(S),
            20,
            body="head",
            fields=3,
            sort="none",
        )

    # ---------------- eyes and the skin lookup table
    info = json.loads(str(A["eye_info"]))
    for s in "lr":
        e = u8(A[f"eye_ao_{s}"])
        pk.add(f"eye_ao_{s}", e, "raw", len(e), 1)
    man["eyes"] = info
    man["eye_mesh"] = dict(nlat=64, nlon=96)
    img = lut["lut"]
    man["lut"] = dict(w=int(img.shape[1]), h=int(img.shape[0]))
    rgba = np.concatenate(
        [img.reshape(-1, 3), np.zeros((img.shape[0] * img.shape[1], 1), np.uint8)], 1
    )
    pk.add("lut", rgba, "raw", rgba.size, 1)
    man["authoring"] = dict(
        hip_height=rig.hip_height * rig.scale,
        eye=[0.0, 0.5116, 0.1169],
        hair_center=[0.0, 0.515, 0.035],
    )
    pk.write()
    if verbose:
        tot = sum(b["count"] * b["stride"] for b in man["buffers"])
        print(
            f"viewer data: {tot / 1e6:.1f} MB raw -> {out_dir} ({time.time() - t0:.0f} s)"
        )
    return man


def chart_record(phi, el):
    """azimuth and elevation (degrees) as two int16 in steps of 0.01 degree"""
    return np.round(np.stack([phi, el], 1) * 100).astype(np.int16)


def bind_record(tri, bary, T):
    """the 3 fine vertices (uint32) and 2 barycentric coordinates (float32) of each binding"""
    rb = np.zeros((len(tri), 20), np.uint8)
    rb[:, 0:12] = T[tri].astype("<u4").view(np.uint8).reshape(-1, 12)
    rb[:, 12:20] = bary[:, :2].astype("<f4").view(np.uint8).reshape(-1, 8)
    return rb


def hair_buffers(pk, body, weights, n_bones):
    """
    The scalp layout and every style (anny.hair.layout, anny.hair.styles) for the page:

    - ``hair_guide_root`` (float32 x 3): the guide roots on anny's default body;
    - ``hair_guide_bind``, ``hair_root_bind``: the triangles under the guide and render roots;
    - ``hair_guide_info``: per guide the mirror, the three simulated guides (uint16) and their
      weights (uint8), and a pad byte;
    - ``hair_guide_skin``: the four bones of the skin under each guide root, as ``head_skin``;
    - ``hair_root_data``: per render root its four guides (uint16), their weights (uint8) and its
      chart (int16, 0.01 degree);
    - per style ``hair_<name>_codes`` (the octahedral codes of the guide segments),
      ``hair_<name>_guide`` (float32: segment, default length, flick, pivot; uint8: group, tip
      bound)
      and ``hair_<name>_tip`` (the triangles under the guide tips).

    Returns the manifest entry: the layout meta and the specs without their groom part.
    """
    from safetensors.numpy import load_file

    from anny.hair import styles as H
    from anny.hair.layout import load_layout

    V, T = body.V, body.T
    layout = load_layout()
    binding = H.Binding.build(layout, V, T)
    G, R = layout.guides, layout.roots
    pk.add("hair_guide_root", layout.guide_position.astype(np.float32), "raw", G * 12, 1)
    pk.add("hair_guide_bind", bind_record(binding.guide_tri, binding.guide_bary, T),
           "sparse", G, 20, body="head", fields=3, sort="none")
    pk.add("hair_root_bind", bind_record(binding.root_tri, binding.root_bary, T),
           "sparse", R, 20, body="head", fields=3, sort="none")
    info = np.zeros((G, 12), np.uint8)
    u16 = np.concatenate([layout.guide_mirror[:, None], layout.guide_sim], 1)
    info[:, 0:8] = u16.astype("<u2").view(np.uint8).reshape(G, 8)
    info[:, 8:11] = layout.guide_sim_weights
    pk.add("hair_guide_info", info, "raw", G * 12, 1)
    # the skin under each guide root: the weights of its triangle's corners, four bones
    W = np.zeros((G, n_bones))
    corners = T[binding.guide_tri]
    for k in range(3):
        vi = corners[:, k]
        for j in range(weights["si"].shape[1]):
            np.add.at(W, (np.arange(G), weights["si"][vi, j]), binding.guide_bary[:, k] * weights["sw"][vi, j])
    si, sw = top_weights(W, 4)
    pk.add("hair_guide_skin", skin_record(si, sw), "raw", G * 8, 1)
    data = np.zeros((R, 16), np.uint8)
    data[:, 0:8] = layout.root_guides.astype("<u2").view(np.uint8).reshape(R, 8)
    data[:, 8:12] = layout.root_weights
    data[:, 12:16] = chart_record(*layout.root_chart.T).view(np.uint8).reshape(R, 4)
    pk.add("hair_root_data", data, "raw", R * 16, 1)
    tensors = load_file(str(H.GUIDES_PATH))
    styles = []
    for name in H.style_names():
        style = H.load_style(name, layout)
        codes = tensors[f"{name}.codes"]
        P = codes.shape[1] + 1
        pk.add(f"hair_{name}_codes", codes, "raw", codes.size, 1)
        rec = np.zeros((G, 20), np.uint8)
        f32 = np.stack([tensors[f"{name}.segment"], style.length, style.flick, style.pivot], 1)
        rec[:, 0:16] = f32.astype("<f4").view(np.uint8).reshape(G, 16)
        tri, bary, on = H.tip_binding(style, V, T)
        rec[:, 16] = style.group
        rec[:, 17] = on
        pk.add(f"hair_{name}_guide", rec, "raw", G * 20, 1)
        pk.add(f"hair_{name}_tip", bind_record(tri, bary, T),
               "sparse", G, 20, body="head", fields=3, sort="none")
        spec = {k: v for k, v in style.spec.items() if k != "groom"}
        styles.append(dict(spec, points=int(P)))
    from anny.hair import chart as C

    return dict(
        layout=dict(layout.meta),
        styles=styles,
        centre=C.CRANIUM_CENTRE.tolist(),
        hairline=dict(phi=C.HAIRLINE_PHI.tolist(), el=C.HAIRLINE_EL.tolist()),
        curve_phi=C.CURVE_PHI.tolist(),
        similarity=H.SIMILARITY,
        sector_radius=H.SECTOR_RADIUS,
        volume=dict(
            step=H.VOLUME_STEP,
            sample=H.VOLUME_SAMPLE,
            ray=H.VOLUME_RAY,
            k=H.VOLUME_K,
            near=H.VOLUME_NEAR,
        ),
    )


def pack_strands(pk, name, P_, ao, lo, hi, step, max_pts, dq=0.00004):
    """strands as roots (16-bit) and 8-bit steps, resampled to about `step` between points"""
    nS, Mp, _ = P_.shape
    seg = np.linalg.norm(np.diff(P_, axis=1), axis=2)
    cum = np.concatenate([np.zeros((nS, 1)), np.cumsum(seg, 1)], 1)
    L = cum[:, -1]
    npts = np.clip(
        np.round(L / step).astype(int) + 1, 3 if name != "hair" else 4, max_pts
    )
    rq = q16(P_[:, 0], lo, hi)
    rroot = lo + rq.astype(np.float64) / 65535 * (hi - lo)
    deltas = []
    for i in range(nS):
        u = np.linspace(0, L[i], npts[i])
        pts = np.stack([np.interp(u, cum[i], P_[i, :, k]) for k in range(3)], 1)
        cur = rroot[i].copy()
        for j in range(1, npts[i]):
            d = np.clip(np.round((pts[j] - cur) / dq), -127, 127).astype(np.int8)
            deltas.append(d)
            cur = cur + d.astype(np.float64) * dq
    deltas = np.array(deltas, np.int8).reshape(-1, 3)
    pk.add(
        name + "_r",
        rq.view(np.uint8).reshape(-1),
        "raw",
        nS * 6,
        1,
        lo=lo.tolist(),
        hi=hi.tolist(),
    )
    pk.add(name + "_d", deltas.view(np.uint8).reshape(-1), "raw", deltas.size, 1, dq=dq)
    pk.add(name + "_n", npts.astype(np.uint8), "raw", nS, 1, strands=int(nS))
    # occlusion along each strand, resampled like the points
    aos = np.concatenate(
        [np.interp(np.linspace(0, L[i], npts[i]), cum[i], ao[i]) for i in range(nS)]
    )
    pk.add(name + "_ao", u8(aos), "raw", len(aos), 1)
