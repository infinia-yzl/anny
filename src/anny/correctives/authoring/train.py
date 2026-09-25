# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Corrective shapes for the arms and legs, fitted to the soft-tissue simulation (sim.py) on
anny's default body. Ported from the legacy 3D Model build (build/correctives.py).

Each joint has key poses. A corrective shape is the rest-space change that makes plain
skinning reach the simulated body at a key pose, so it adds to the rest shape before skinning,
as morph targets do in every engine. Drivers set the weights of the shapes from the skeleton:

- hinge joints (elbow, knee): the bend angle between the two bones; the weights interpolate
  linearly between the keys.
- ball joints (shoulder, hip): the direction of the upper bone in the frame of the torso bone
  above it; the key directions form triangles on the sphere, and the weights are the
  barycentric coordinates of the current direction in its triangle.

At the rest pose every weight of a shape is 0, so the rest shape stays exactly as it is. Shapes
are built for the left side and mirrored to the right side. The shapes live on anny's own
vertices and use anny's skinning weights.

    python -m anny.correctives.authoring.train train   # simulate the key poses (cached)
    python -m anny.correctives.authoring.train build   # fit the shapes -> data/correctives

Training needs SciPy and TetGen; each key pose takes between a few seconds and two minutes.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file

from anny.paths import get_anny_cache_path
from anny.poses.authoring import poselib as PL
from anny.poses.authoring import posing as P
from anny.poses.authoring.rig import ANNY_TO_LEGACY

IX, H = P.IDX, P.HEADS
DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "correctives"


def key_cache_dir() -> pathlib.Path:
    return get_anny_cache_path() / "correctives" / "keys"


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def cone_dir(elev, plane, side="L"):
    """direction from elevation (0 = hanging straight down) and plane (0 = out to the side,
    90 = forward, -90 = back, 180 = across the body)"""
    e, p = np.radians(elev), np.radians(plane)
    sg = 1 if side == "L" else -1
    return np.array([sg * np.sin(e) * np.cos(p), -np.cos(e), np.sin(e) * np.sin(p)])


def rest_dir(bone, end):
    return unit(H[IX[end]] - H[IX[bone]])


# ------------------------------------------------------------------ joints (left side; the right side mirrors them)
JOINTS = {
    "elbow": dict(
        kind="hinge",
        bones=("upperarm01", "lowerarm01", "wrist"),
        region="arm",
        keys=[0, 90, 130],
        radius=(0.09, 0.15),
    ),
    "knee": dict(
        kind="hinge",
        bones=("upperleg01", "lowerleg01", "foot"),
        region="leg",
        keys=[0, 50, 90, 130],
        radius=(0.12, 0.19),
    ),
    "shoulder": dict(
        kind="cone",
        bones=("upperarm01", "lowerarm01"),
        frame="spine01",
        region="arm",
        radius=(0.15, 0.24),
        smooth=0.1,
        ridge=0.003,
        targets={
            "down": (5, 20),
            "side": (90, 0),
            "up": (180, 0),
            "forward": (90, 90),
            "forward_up": (135, 90),
            "side_forward": (90, 45),
            "back": (45, -90),
            "across": (90, 150),
            "across_low": (40, 150),
            "forward_low": (45, 90),
            "side_up": (130, 20),
            "back_high": (75, -90),
            "back_side": (60, -40),
            "up_in": (155, 175),
            "back_up": (150, -90),
        },
        # overhead and across or behind the head, the simulation folds the armpit in on itself,
        # so the correction fades out toward these keys
        empty=["up_in", "back_up"],
        # straight up, a smoother shape leaves a dent along the side of the chest where the
        # skin weights change from the torso to the arm; a lighter smoothness reaches the
        # simulation there
        smooth_keys={"up": 0.01},
    ),
    "hip": dict(
        kind="cone",
        bones=("upperleg01", "lowerleg01"),
        frame="root",
        region="leg",
        radius=(0.17, 0.27),
        targets={
            "flex45": (45, 90),
            "flex90": (90, 90),
            "flex125": (125, 90),
            "flex90_out": (95, 55),
            "flex125_out": (125, 60),
            "back": (25, -90),
            "out": (45, 0),
            "in": (18, 180),
            "flex45_out": (45, 45),
            "back_out": (30, -35),
        },
    ),
}


def rest_bend(j):
    a, b, c = JOINTS[j]["bones"]
    u = unit(H[IX[b + ".L"]] - H[IX[a + ".L"]])
    f = unit(H[IX[c + ".L"]] - H[IX[b + ".L"]])
    return float(np.degrees(np.arccos(np.clip(u @ f, -1, 1))))


def cone_targets(j, side="L"):
    """names and directions of the targets of a ball joint, with the rest direction first"""
    J = JOINTS[j]
    names = ["rest"] + list(J["targets"])
    d0 = rest_dir(J["bones"][0] + "." + side, J["bones"][1] + "." + side)
    dirs = [d0] + [cone_dir(*J["targets"][k], side) for k in J["targets"]]
    return names, np.array(dirs)


def cone_triangles(j):
    """triangles of the key directions on the sphere: the faces of their convex hull that face
    away from the centre (when the keys fill less than a hemisphere, the faces toward the
    centre are left out)"""
    from scipy.spatial import ConvexHull

    names, D = cone_targets(j, "L")
    hull = ConvexHull(D)
    tris = []
    for f, eq in zip(hull.simplices, hull.equations):
        if eq[3] > -1e-9:  # the centre lies outside this face's half-space
            continue
        f = [int(x) for x in f]
        if np.linalg.det(D[f]) < 0:
            f = [f[0], f[2], f[1]]
        tris.append(f)
    return tris


def swing(u, d):
    """minimal rotation matrix taking unit u to unit d"""
    u, d = unit(u), unit(d)
    ax = np.cross(u, d)
    s, c = np.linalg.norm(ax), float(u @ d)
    if s < 1e-9:
        return np.eye(3)
    return P.qmat(P.qaxis(ax / s, np.degrees(np.arctan2(s, c))))


def key_pose(j, key, side="L"):
    """the pose of one key: only this joint moves from the rest pose (and the shoulder girdle
    for the shoulder)"""
    J = JOINTS[j]
    if J["kind"] == "hinge":
        return value_pose(j, float(key), side)
    return value_pose(j, cone_dir(*J["targets"][key], "L"), side)


def value_pose(j, value, side="L"):
    """the pose for a bend angle (hinge) or a direction given for the left side (cone)"""
    J = JOINTS[j]
    p = P.Pose()
    s = side
    if J["kind"] == "hinge":
        a, b, c = [x + "." + s for x in J["bones"]]
        u, f = unit(H[IX[b]] - H[IX[a]]), unit(H[IX[c]] - H[IX[b]])
        h = unit(np.cross(u, f))
        p.rot(b, h, float(value) - rest_bend(j))
        return p
    d = unit(value)
    if s == "R":
        d = d * np.array([-1.0, 1.0, 1.0])
    bone, end = J["bones"][0] + "." + s, J["bones"][1] + "." + s
    if j == "shoulder":
        # the chest is at rest here, so the rotation relative to the chest is the world
        # rotation; the girdle follows
        PL.set_arm(p, s, P.mat2q(swing(rest_dir(bone, end), d)))
        return p
    F = P.fk(p)
    P.set_world_rot(p, bone, swing(rest_dir(bone, end), d), F)
    return p


# ------------------------------------------------------------------ drivers (reference implementation)
def measure(j, F, side):
    J = JOINTS[j]
    Rw, Pw = F
    if J["kind"] == "hinge":
        a, b, c = [IX[x + "." + side] for x in J["bones"]]
        u, f = unit(Pw[b] - Pw[a]), unit(Pw[c] - Pw[b])
        return float(np.degrees(np.arccos(np.clip(u @ f, -1, 1))))
    a, b = IX[J["bones"][0] + "." + side], IX[J["bones"][1] + "." + side]
    return unit(Rw[IX[J["frame"]]].T @ (Pw[b] - Pw[a]))


def hinge_weights(j, angle):
    """weights of the keys (the rest angle is an extra key with no shape)"""
    keys = JOINTS[j]["keys"]
    xs = sorted([(rest_bend(j), None)] + [(float(k), i) for i, k in enumerate(keys)])
    w = np.zeros(len(keys))
    a = float(np.clip(angle, xs[0][0], xs[-1][0]))
    for (x0, i0), (x1, i1) in zip(xs[:-1], xs[1:]):
        if x0 <= a <= x1:
            t = (a - x0) / max(x1 - x0, 1e-9)
            if i0 is not None:
                w[i0] += 1 - t
            if i1 is not None:
                w[i1] += t
            break
    return w


def cone_weights(j, d, side="L"):
    """barycentric weights of direction d in the key triangles (rest first); outside every
    triangle, the nearest triangle with its negative weights cut to zero"""
    names, D = cone_targets(j, side)
    best, best_w = None, None
    for t in TRIS[j]:
        M = D[t].T
        try:
            b = np.linalg.solve(M, d)
        except np.linalg.LinAlgError:
            continue
        if b.sum() <= 0:
            continue
        b = b / b.sum()
        if best is None or b.min() > best:
            best, best_w = b.min(), (t, b)
    w = np.zeros(len(names))
    t, b = best_w
    b = np.clip(b, 0, None)
    b /= b.sum()
    w[t] = b
    return w[1:]


def weights(F, side):
    """all corrective weights of one side for a posed skeleton F: {joint: array}"""
    out = {}
    for j, J in JOINTS.items():
        m = measure(j, F, side)
        out[j] = (
            hinge_weights(j, m) if J["kind"] == "hinge" else cone_weights(j, m, side)
        )
    return out


def shape_names(j):
    J = JOINTS[j]
    if J["kind"] == "hinge":
        return [f"{j}_{int(k):03d}" for k in J["keys"]]
    return [f"{j}_{k}" for k in J["targets"]]


TRIS = {j: cone_triangles(j) for j, J in JOINTS.items() if J["kind"] == "cone"}


# ------------------------------------------------------------------ training
def body_hash() -> str:
    import hashlib

    coarse = P.preview_mesh()["coarse"]
    return hashlib.sha1(np.round(coarse["V"], 6).tobytes()).hexdigest()[:16]


def train(force=False, only=None):
    from . import sim

    cache = key_cache_dir() / body_hash()
    cache.mkdir(parents=True, exist_ok=True)
    S = sim.Sim()
    for j, J in JOINTS.items():
        keys = J["keys"] if J["kind"] == "hinge" else list(J["targets"])
        for k, nm in zip(keys, shape_names(j)):
            if k in J.get("empty", []) or (only and nm not in only):
                continue
            fn = cache / f"{nm}.npz"
            if fn.exists() and not force:
                continue
            p = key_pose(j, k)
            X = S.solve(p, region=J["region"] + ".L")
            np.savez(fn, X=X[: S.ns].astype(np.float64), q=p.q, root=p.root)
            print(
                f"{nm:>24}: {S.last['iters']} iterations, {S.last['time']:.1f} s",
                flush=True,
            )


def falloff(V, c, r0, r1):
    d = np.linalg.norm(V - c, axis=1)
    t = np.clip((d - r0) / (r1 - r0), 0, 1)
    return 1 - t * t * (3 - 2 * t)


def mirror_map(V, tol=2e-5):
    """index of the mirror partner (x -> -x) of every vertex; -1 when there is none"""
    from scipy.spatial import cKDTree

    d, j = cKDTree(V).query(V * np.array([-1.0, 1.0, 1.0]))
    j[d > tol] = -1
    return j


def edge_laplacian(T, reg, n):
    """graph Laplacian of the mesh edges between the vertices of a region (uniform weights)"""
    import scipy.sparse as sp

    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = np.unique(np.sort(E, 1), axis=0)
    pos = -np.ones(n, int)
    pos[reg] = np.arange(len(reg))
    a, b = pos[E[:, 0]], pos[E[:, 1]]
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    m = len(reg)
    W = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
    W = (W + W.T).tocsr()
    return sp.diags(np.asarray(W.sum(1)).ravel()) - W


SMOOTH = 0.01  # weight of the smoothness term: it decides the shape where the blend of bones nearly cancels
NECK_Y = 0.39  # nothing above the base of the neck (legacy frame)


def fit_shapes():
    """
    Shapes on anny's vertices with anny's skinning weights. For each key the shape solves

        min sum_v |A_v d_v - D_v|^2 + SMOOTH sum_edges |d_u - d_v|^2

    where A_v is the blended skin matrix of vertex v at the key and D_v the distance from plain
    skinning to the simulation. Where the blend of bones is well conditioned the shape reaches
    the simulation; where the bones nearly cancel each other the smoothness term decides.

    Returns a list of dicts (name, idx, D) in the legacy frame, and a report per key.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    coarse = P.preview_mesh()["coarse"]
    used = coarse["used"]
    remap = -np.ones(len(coarse["V"]), np.int64)
    remap[used] = np.arange(len(used))
    Vs = coarse["V"][used]
    T = remap[coarse["quads"]]
    T = np.concatenate([T[:, [0, 1, 2]], T[:, [0, 2, 3]]])
    si, sw = coarse["si"][used], coarse["sw"][used]
    mirror = mirror_map(Vs)
    assert (mirror[Vs[:, 1] < NECK_Y] >= 0).all(), (
        "the body below the neck is not symmetric"
    )
    cache = key_cache_dir() / body_hash()
    neck = np.clip((NECK_Y - Vs[:, 1]) / 0.03, 0, 1)
    shapes, report = [], []
    for j, J in JOINTS.items():
        keys = J["keys"] if J["kind"] == "hinge" else list(J["targets"])
        c = H[IX[J["bones"][1 if J["kind"] == "hinge" else 0] + ".L"]]
        mask = falloff(Vs, c, *J["radius"]) * neck * (Vs[:, 0] > -0.03)
        reg = np.where(mask > 0)[0]
        n_r = len(reg)
        Lap = sp.kron(edge_laplacian(T, reg, len(Vs)), sp.eye(3)).tocsr()
        rows = np.repeat(np.arange(n_r * 3).reshape(n_r, 3), 3, axis=1).ravel()
        cols = np.tile(np.arange(n_r * 3).reshape(n_r, 3), (1, 3)).ravel()
        for k, nm in zip(keys, shape_names(j)):
            if k in J.get("empty", []):
                continue
            d = np.load(cache / f"{nm}.npz")
            p = P.Pose()
            p.q = d["q"]
            p.root = d["root"]
            F = P.fk(p)
            Xt = d["X"][reg]
            Xl = P.skin(F, Vs[reg], si[reg], sw[reg])
            A = np.einsum("nk,nkij->nij", sw[reg], F[0][si[reg]])
            AtA = np.einsum("nki,nkj->nij", A, A)
            b = np.einsum("nki,nk->ni", A, Xt - Xl).ravel()
            mu = J.get("smooth_keys", {}).get(k, J.get("smooth", SMOOTH))
            Mtx = (
                sp.coo_matrix(
                    (AtA.ravel(), (rows, cols)), shape=(3 * n_r, 3 * n_r)
                ).tocsr()
                + mu * Lap
                + J.get("ridge", 1e-8) * sp.eye(3 * n_r)
            )
            delta = spla.spsolve(Mtx.tocsc(), b).reshape(n_r, 3)
            D = np.zeros_like(Vs)
            D[reg] = delta * mask[reg][:, None]
            keep = np.linalg.norm(D, axis=1) > 2e-5
            left = np.where(keep)[0]
            shapes.append(dict(name=f"{nm}.L", idx=left, D=D[left]))
            right = mirror[left]
            order = np.argsort(right)
            shapes.append(
                dict(
                    name=f"{nm}.R",
                    idx=right[order],
                    D=(D[left] * np.array([-1.0, 1.0, 1.0]))[order],
                )
            )
            fit = np.linalg.norm(np.einsum("nij,nj->ni", A, delta) - (Xt - Xl), axis=1)
            plain = np.linalg.norm(Xt - Xl, axis=1)
            report.append(
                dict(
                    key=nm,
                    vertices=int(keep.sum()),
                    largest_move_mm=float(np.linalg.norm(D, axis=1).max() * 1000),
                    plain_mean_mm=float(plain.mean() * 1000),
                    fit_mean_mm=float(fit.mean() * 1000),
                    fit_max_mm=float(fit.max() * 1000),
                )
            )
            r = report[-1]
            print(
                f"{nm:>24}: {r['vertices']} vertices, largest move {r['largest_move_mm']:.1f} mm, "
                f"plain skinning {r['plain_mean_mm']:.2f} mm, fit {r['fit_mean_mm']:.2f} mm",
                flush=True,
            )
    return shapes, report


def driver_spec():
    """the drivers in anny's axes; the bone names are those of anny's rig"""
    M = ANNY_TO_LEGACY
    out = dict(
        format="anny-correctives@1",
        conventions=dict(
            units="metres and degrees",
            axes="anny's axes: X to the left of the figure, Y backward, Z up",
            shapes=(
                "offsets of anny's rest vertices (MakeHuman base mesh indices), added before "
                "skinning; they were fitted on anny's default body and scale with the size of "
                "the body around them"
            ),
            hinge=(
                "angle between head(bones[1]) - head(bones[0]) and head(bones[2]) - "
                "head(bones[1]) from posed positions; the weights interpolate linearly between "
                "neighbouring keys and stay at the first or last key outside them; the key "
                "marked rest takes the angle of the rest skeleton and has no shape"
            ),
            cone=(
                "d = normalize(inverse(rotation(frame)) * (head(end) - head(bone))), where "
                "rotation(frame) is the rotation of the frame bone from its rest pose; for each "
                "triangle solve d = a*t0 + b*t1 + c*t2 and divide by a + b + c; the triangle with "
                "the largest smallest coordinate wins; negative coordinates become 0 and the "
                "rest are divided by their sum; the target marked rest takes the direction of the "
                "rest skeleton"
            ),
            girdle=(
                "the key poses of the shoulder move clavicle and shoulder01 with the arm by the "
                "rule of anny.poses.authoring.poselib.girdle_for, as the pose library does"
            ),
        ),
        joints=[],
    )
    for side in "LR":
        for j, J in JOINTS.items():
            b = [x + "." + side for x in J["bones"]]
            if J["kind"] == "hinge":
                keys = [dict(angle=round(rest_bend(j), 3), shape=None, rest=True)] + [
                    dict(angle=float(k), shape=f"{n}.{side}")
                    for k, n in zip(J["keys"], shape_names(j))
                ]
                keys.sort(key=lambda k: k["angle"])
                out["joints"].append(
                    dict(name=f"{j}.{side}", type="hinge", bones=b, keys=keys)
                )
            else:
                names, D = cone_targets(j, side)
                shp = [None] + [
                    None if k in J.get("empty", []) else f"{n}.{side}"
                    for k, n in zip(J["targets"], shape_names(j))
                ]
                tris = TRIS[j] if side == "L" else [[t[0], t[2], t[1]] for t in TRIS[j]]
                targets = []
                for i, (n, d, s) in enumerate(zip(names, D, shp)):
                    t = dict(name=n, dir=[round(float(x), 6) for x in M.T @ d], shape=s)
                    if i == 0:
                        t["rest"] = True
                    targets.append(t)
                out["joints"].append(
                    dict(
                        name=f"{j}.{side}",
                        type="cone",
                        bone=b[0],
                        end=b[1],
                        frame=J["frame"],
                        targets=targets,
                        triangles=tris,
                    )
                )
    return out


def build():
    shapes, report = fit_shapes()
    coarse = P.preview_mesh()["coarse"]
    used = coarse["used"]
    base_index = coarse["base_index"]
    M = ANNY_TO_LEGACY
    rest_anny = P.RIG.to_anny(coarse["V"][used])
    tensors, info = {}, []
    for s in shapes:
        D_anny = s["D"] @ M / P.RIG.scale
        idx = used[s["idx"]]
        support = rest_anny[s["idx"]]
        radius = float(np.sqrt(((support - support.mean(0)) ** 2).sum(1).mean()))
        tensors[s["name"] + ".indices"] = torch.tensor(
            base_index[idx], dtype=torch.int64
        )
        tensors[s["name"] + ".offsets"] = torch.tensor(D_anny, dtype=torch.float32)
        info.append(dict(name=s["name"], vertices=len(idx), reference_radius=radius))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(DATA_DIR / "soft_tissue.safetensors"))
    spec = driver_spec()
    spec["shapes"] = info
    spec["fit"] = report
    with open(DATA_DIR / "soft_tissue.json", "w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(shapes)} shapes -> {DATA_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1] == "train":
        train(
            force="--force" in sys.argv,
            only=[a for a in sys.argv[2:] if a != "--force"],
        )
    elif sys.argv[1] == "build":
        build()
    print(f"done in {time.time() - t0:.0f} s")
