# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Soft-tissue simulation of the body for corrective shapes, on anny's default body in the frame
of the authoring rig (``anny.poses.authoring.rig``). Ported from the legacy 3D Model build
(build/sim.py). It needs SciPy and TetGen (``pip install scipy tetgen``).

The base body (MakeHuman resolution) is filled with tetrahedra. Rigid cores follow the skeleton: the long bones of the
limbs (split between the twist bones), the joint ends of the bones (the humerus and femur condyles, the olecranon, the
tibial plateau and the kneecap), the collarbone, the ribcage, the spine and the pelvis. The hands, the feet and the
neck and head follow standard skinning. The rest of the volume is soft tissue, solved to rest with projective dynamics:
every tetrahedron tries to keep its shape (as rigid as possible) and, more strongly, its volume. The solver works on a
region of the body at a time; nodes outside the region follow standard skinning.

    from anny.correctives.authoring import sim
    S = sim.Sim()
    X = S.solve(pose, region="arm.L")   # all nodes; X[:S.ns] is the surface

The thresholds below (heights of the neck, the ribcage and the regions) are in the legacy frame
of the authoring rig, where anny's default body has the same eye height and floor as the legacy
body; its joints sit within a few millimetres of the legacy joints there.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from anny.paths import get_anny_cache_path
from anny.poses.authoring import posing as P
from anny.poses.authoring.rig import authoring_rig
from anny.utils.subdivision import quads_to_triangles


def cache_file():
    """the tetrahedra of the authoring body (one file per body)"""
    import hashlib

    V = P.preview_mesh()["coarse"]["V"]
    key = hashlib.sha1(np.round(V, 6).tobytes()).hexdigest()[:16]
    return get_anny_cache_path() / "correctives" / f"sim_mesh_{key}.npz"


def crossing_vertices(V, T):
    """vertices of the edges and triangles where the surface passes through itself"""
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    m = trimesh.Trimesh(V, T, process=False)
    E = m.edges_unique
    o = V[E[:, 0]]
    d = V[E[:, 1]] - o
    length = np.linalg.norm(d, axis=1)
    d = d / length[:, None]
    locs, ray, tri = RayMeshIntersector(m).intersects_location(o, d, multiple_hits=True)
    if len(ray) == 0:
        return np.zeros(0, np.int64)
    dist = np.einsum("ij,ij->i", locs - o[ray], d[ray])
    e = E[ray]
    adjacent = (T[tri] == e[:, :1]).any(1) | (T[tri] == e[:, 1:]).any(1)
    hit = (dist > 1e-7) & (dist < length[ray] - 1e-7) & ~adjacent
    return np.unique(np.concatenate([e[hit].ravel(), T[tri[hit]].ravel()]))


def separate_crossings(V, T, rounds=20):
    """
    Relax the surface where it passes through itself (on anny's default body, the small toes
    touch their neighbours). The vertices around each crossing move toward the mean of their
    neighbours until TetGen accepts the surface. Only the simulation sees this surface.
    """
    V = V.copy()
    n = len(V)
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    for _ in range(rounds):
        bad = crossing_vertices(V, T)
        if len(bad) == 0:
            break
        near = np.zeros(n, bool)
        near[bad] = True
        for _ in range(2):  # two rings around the crossing
            near[E[near[E[:, 0]], 1]] = True
        for _ in range(5):
            acc = np.zeros_like(V)
            cnt = np.zeros(n)
            np.add.at(acc, E[:, 0], V[E[:, 1]])
            np.add.at(cnt, E[:, 0], 1)
            avg = acc / np.maximum(cnt, 1)[:, None]
            V[near] = 0.5 * V[near] + 0.5 * avg[near]
    return V


# TetGen: p = surface mesh, q1.8/12 = quality (radius-edge ratio, min dihedral),
# Y = keep the surface vertices, a = max volume
TET_SWITCHES = "pq1.8/12YQa2e-7"
IX, H, TL = P.IDX, P.HEADS, P.TAILS


def ring_size(V, quads):
    """mean length of the edges around each vertex of a quad mesh"""
    E = np.concatenate(
        [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
    )
    E = np.unique(np.sort(E, 1), axis=0)
    length = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    total = np.bincount(E.ravel(), np.repeat(length, 2), minlength=len(V))
    count = np.bincount(E.ravel(), minlength=len(V))
    return total / np.maximum(count, 1)


def seg_param(p, a, b):
    ab = b - a
    t = ((p - a) @ ab) / (ab @ ab)
    d = np.linalg.norm(p - (a + np.clip(t, 0, 1)[:, None] * ab), axis=1)
    return t, d


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def unit_volume(S, iters=60, tol=1e-9):
    """closest singular values with product 1 (per row): s_i = (S_i + sqrt(S_i^2 + 4 lam)) / 2, with lam from a
    safeguarded Newton search on sum(log s_i) = 0 (only the rows that have not converged are updated)"""
    n = len(S)
    lo = -(S**2).min(1) / 4.0
    hi = np.maximum(1.0, np.abs(S).max(1)) ** 2 * 4.0
    lam = np.zeros(n)
    lam[(S <= 0).any(1)] = 1e-3
    act = np.arange(n)
    for _ in range(iters):
        Sa, la = S[act], lam[act]
        r = np.sqrt(np.maximum(Sa**2 + 4 * la[:, None], 1e-30))
        s = 0.5 * (Sa + r)
        f = np.log(np.maximum(s, 1e-30)).sum(1)
        big = f > 0
        hi[act] = np.where(big, la, hi[act])
        lo[act] = np.where(big, lo[act], la)
        df = (1.0 / (np.maximum(s, 1e-30) * r)).sum(1)
        step = la - f / np.maximum(df, 1e-30)
        inside = (step >= lo[act]) & (step <= hi[act])
        keep = np.abs(f) > tol
        lam[act] = np.where(
            ~keep, la, np.where(inside, step, 0.5 * (lo[act] + hi[act]))
        )
        act = act[keep]
        if len(act) == 0:
            break
    r = np.sqrt(np.maximum(S**2 + 4 * lam[:, None], 0.0))
    out = 0.5 * (S + r)
    # strongly stretched rows have no root on this branch: the smallest value takes the other root,
    # s_3 = (S_3 - sqrt(S_3^2 + 4 lam)) / 2 with lam in [-S_3^2 / 4, 0), found by bisection
    miss = np.where(np.abs(np.log(np.maximum(out, 1e-30)).sum(1)) > 1e-6)[0]
    if len(miss):
        Sm = S[miss]
        k = np.argmin(Sm, 1)
        a = -(Sm[np.arange(len(Sm)), k] ** 2) / 4.0
        b = np.zeros(len(Sm))
        onehot = np.eye(3, dtype=bool)[k]
        for _ in range(60):
            lm = 0.5 * (a + b)
            rr = np.sqrt(np.maximum(Sm**2 + 4 * lm[:, None], 0.0))
            sv = np.where(onehot, 0.5 * (Sm - rr), 0.5 * (Sm + rr))
            f = np.log(np.maximum(sv, 1e-30)).sum(1)
            a = np.where(f > 0, lm, a)
            b = np.where(f > 0, b, lm)
        lm = 0.5 * (a + b)
        rr = np.sqrt(np.maximum(Sm**2 + 4 * lm[:, None], 0.0))
        sv = np.where(onehot, 0.5 * (Sm - rr), 0.5 * (Sm + rr))
        bad = np.abs(np.log(np.maximum(sv, 1e-30)).sum(1)) > 1e-6
        if bad.any():
            Sa = np.abs(Sm[bad])
            sv[bad] = Sa / np.cbrt(np.maximum(Sa.prod(1), 1e-30))[:, None]
        out[miss] = sv
    return out


def hinge_axis(upper, lower, end, heads=None):
    """rest hinge axis of a limb joint (normal of the plane of the two bones), and the two bone directions"""
    heads = H if heads is None else heads
    a, b, c = heads[IX[upper]], heads[IX[lower]], heads[IX[end]]
    u, f = unit(b - a), unit(c - b)
    return unit(np.cross(u, f)), u, f


# helper frames: a node attached to a helper follows part of the rotation between a bone and its parent (the kneecap)
HELPERS = {}
for _s in "LR":
    HELPERS[f"patella.{_s}"] = dict(
        parent=f"upperleg02.{_s}", child=f"lowerleg01.{_s}", share=0.5
    )


# creases: where a joint folds, the tissue on the two sides meets. The surface on the inner side of each joint
# keeps to its own side of the plane that halves the angle between the two segments.
CREASES = {}
for _s in "LR":
    CREASES[f"knee.{_s}"] = dict(
        upper=(f"upperleg01.{_s}", f"lowerleg01.{_s}"),
        lower=(f"lowerleg01.{_s}", f"foot.{_s}"),
        lower_bones=(f"lowerleg01.{_s}", f"lowerleg02.{_s}", f"foot.{_s}"),
        radius=0.15,
    )
    CREASES[f"elbow.{_s}"] = dict(
        upper=(f"upperarm01.{_s}", f"lowerarm01.{_s}"),
        lower=(f"lowerarm01.{_s}", f"wrist.{_s}"),
        lower_bones=(f"lowerarm01.{_s}", f"lowerarm02.{_s}", f"wrist.{_s}"),
        radius=0.13,
    )
# The hip has no crease plane: at deep flexion the plane pushes the belly into lumps, and the fold stays hidden.
CONTACT_W = 0.5  # stiffness of the contact term (the elastic terms of a node are about 0.01 to 0.1)
CONTACT_GAP = -0.012  # the two sides may press 12 mm past the plane: the fold takes in some tissue, as skin does


class Sim:
    def __init__(self, stiff_shape=1.0, stiff_volume=10.0, verbose=True, opts=None):
        t0 = time.time()
        self.opts = dict(dict(acromion=False), **(opts or {}))
        coarse = P.preview_mesh()["coarse"]
        used = coarse["used"]
        remap = -np.ones(len(coarse["V"]), np.int64)
        remap[used] = np.arange(len(used))
        Vb, Fq = coarse["V"][used], remap[coarse["quads"]]
        self.Ts = quads_to_triangles(Fq)
        Vb = separate_crossings(Vb, self.Ts)
        self.Vs, self.Fq, self.used = Vb, Fq, used
        self.ns = len(Vb)
        CACHE = cache_file()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            d = np.load(CACHE)
            nodes, tets = d["nodes"], d["tets"]
            assert str(d["switches"]) == TET_SWITCHES and np.allclose(
                nodes[: self.ns], Vb
            )
        except Exception:
            import tetgen

            tg = tetgen.TetGen(Vb.astype(np.float64), self.Ts.astype(np.int32))
            nodes, tets = tg.tetrahedralize(switches=TET_SWITCHES)[:2]
            assert np.allclose(nodes[: self.ns], Vb), (
                "TetGen moved the surface vertices"
            )
            np.savez(CACHE, nodes=nodes, tets=tets, switches=TET_SWITCHES)
        self.X0 = nodes.astype(np.float64)
        self.tets_all = tets.astype(np.int64)
        # skin weights: surface from the base weights, interior from the nearest surface vertex
        Wb = coarse["W"][used]
        si, sw = coarse["si"][used], coarse["sw"][used]
        _, near = cKDTree(Vb).query(self.X0)
        self.si, self.sw = si[near], sw[near]
        self.si[: self.ns], self.sw[: self.ns] = si, sw
        # The cores, the creases and the regions are measured on anny's default body. For another
        # body (rig.authoring_phenotype) each node goes to the default body through its nearest
        # surface vertex, with its offset scaled by the size of the body around that vertex, and
        # the masks look at these default positions (Xd) and the default joints (H0, TL0).
        default = authoring_rig()
        self.H0, self.TL0 = default.heads, default.tails
        self.size_ratio = P.RIG.hip_height / default.hip_height
        scale = np.ones(len(Vb))
        if P.RIG is default:
            self.Xd = self.X0
        else:
            c0 = default.preview["coarse"]
            V0 = c0["V"][c0["used"]]
            scale = ring_size(Vb, Fq) / ring_size(V0, Fq)
            self.Xd = V0[near] + (self.X0 - Vb[near]) / scale[near][:, None]
            self.Xd[: self.ns] = V0
        self.depth = self._depth()
        self.depth_default = self.depth / scale[near]
        self._cores()
        self._creases(Wb)
        # per-element operators on the whole mesh
        X, T = self.X0, self.tets_all
        Dm = np.stack(
            [X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]],
            2,
        )
        self.vol_all = np.abs(np.linalg.det(Dm)) / 6.0
        B = np.linalg.inv(Dm)
        c = np.zeros((len(T), 4, 3))
        c[:, 1:, :] = B
        c[:, 0, :] = -B.sum(1)
        self.c_all = c
        self.ks, self.kv = stiff_shape, stiff_volume
        self.kmix = stiff_volume / (stiff_shape + stiff_volume)
        self.regions = {}
        self.muscles = []
        if verbose:
            print(
                f"sim: {len(X)} nodes ({self.ns} on the surface), {len(T)} tets, soft {int((self.fix == -1).sum())}, "
                f"cores {int((self.fix >= 0).sum() + (self.fix <= -3).sum())}, skinned {int((self.fix == -2).sum())}; "
                f"{time.time() - t0:.1f} s"
            )

    # ------------------------------------------------------------------ geometry
    def _depth(self):
        """distance of every node below the surface (from dense surface samples)"""
        tri = self.Vs[self.Ts]
        bary = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1 / 3, 1 / 3, 1 / 3],
                [0.5, 0.5, 0],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
            ]
        )
        pts = np.einsum("kj,tjd->tkd", bary, tri).reshape(-1, 3)
        d, _ = cKDTree(pts).query(self.X0)
        d[: self.ns] = 0.0
        return d

    def _cores(self):
        """fix[i]: bone index for nodes in a rigid core, -3 - k for helper k, -2 for skinned nodes,
        -1 for soft tissue"""
        X, H, TL, depth = self.Xd, self.H0, self.TL0, self.depth_default
        fix = -np.ones(len(X), int)
        self.helper_names = list(HELPERS)
        y = X[:, 1]
        # neck and head, hands, feet: standard skinning
        skinned = y > 0.405
        for s, sg in (("L", 1), ("R", -1)):
            t, _ = seg_param(X, H[IX[f"lowerarm01.{s}"]], H[IX[f"wrist.{s}"]])
            skinned |= (t > 0.97) & (X[:, 0] * sg > 0.25)
        skinned |= y < H[IX["foot.L"]][1] + 0.012
        fix[skinned] = -2
        counts = {}

        def take(mask, bone, label):
            m = mask & (fix == -1)
            fix[m] = IX[bone] if isinstance(bone, str) else bone
            counts[label] = counts.get(label, 0) + int(m.sum())

        def capsule(a, b, r, t0=0.0, t1=1.0):
            t, d = seg_param(X, a, b)
            return (d < r) & (t >= t0) & (t <= t1)

        def sphere(c, r):
            return np.linalg.norm(X - c, axis=1) < r

        for s in "LR":
            sg = 1 if s == "L" else -1
            # ---------------- arm
            sh, el, wr = (
                H[IX[f"upperarm01.{s}"]],
                H[IX[f"lowerarm01.{s}"]],
                H[IX[f"wrist.{s}"]],
            )
            h, u, f = hinge_axis(
                f"upperarm01.{s}", f"lowerarm01.{s}", f"wrist.{s}", heads=H
            )
            # humerus: head, shaft, and the condyles along the hinge axis at the elbow
            split = seg_param(H[IX[f"upperarm02.{s}"]][None], sh, el)[0][0]
            take(sphere(sh, 0.013), f"upperarm01.{s}", f"humerus.{s}")
            t, _ = seg_param(X, sh, el)
            m = capsule(sh, el, 0.011, 0.0, 0.93)
            take(m & (t < split), f"upperarm01.{s}", f"humerus.{s}")
            take(m & (t >= split), f"upperarm02.{s}", f"humerus.{s}")
            take(
                capsule(el - h * 0.017, el + h * 0.017, 0.007),
                f"upperarm02.{s}",
                f"humerus.{s}",
            )
            # forearm (radius and ulna): shaft, and the olecranon behind the elbow
            split = seg_param(H[IX[f"lowerarm02.{s}"]][None], el, wr)[0][0]
            t, _ = seg_param(X, el, wr)
            m = capsule(el, wr, 0.009, 0.1, 0.95)
            take(m & (t < split), f"lowerarm01.{s}", f"forearm.{s}")
            take(m & (t >= split), f"lowerarm02.{s}", f"forearm.{s}")
            take(
                capsule(el - f * 0.022, el + f * 0.02, 0.0065),
                f"lowerarm01.{s}",
                f"olecranon.{s}",
            )
            # ---------------- leg
            hp, kn, an = (
                H[IX[f"upperleg01.{s}"]],
                H[IX[f"lowerleg01.{s}"]],
                H[IX[f"foot.{s}"]],
            )
            h, u, f = hinge_axis(
                f"upperleg01.{s}", f"lowerleg01.{s}", f"foot.{s}", heads=H
            )
            fwd = unit(np.cross(np.array([1.0, 0, 0]), u))
            fwd = fwd if fwd[2] > 0 else -fwd
            side = unit(np.cross(u, fwd))
            # femur: head, neck to the greater trochanter, shaft, condyles
            troch = hp + np.array([sg * 0.03, -0.018, -0.004])
            take(sphere(hp, 0.017), f"upperleg01.{s}", f"femur.{s}")
            take(capsule(hp, troch, 0.011), f"upperleg01.{s}", f"femur.{s}")
            split = seg_param(H[IX[f"upperleg02.{s}"]][None], troch, kn)[0][0]
            t, _ = seg_param(X, troch, kn)
            m = capsule(troch, kn, 0.014, 0.0, 0.92)
            take(m & (t < split), f"upperleg01.{s}", f"femur.{s}")
            take(m & (t >= split), f"upperleg02.{s}", f"femur.{s}")
            cond = kn + u * 0.004 - fwd * 0.003
            take(
                capsule(cond - side * 0.022, cond + side * 0.022, 0.012),
                f"upperleg02.{s}",
                f"femur.{s}",
            )
            # kneecap: in front of the knee, on a helper that turns half as far as the knee
            pat = kn + fwd * 0.024 + u * 0.004
            take(
                sphere(pat, 0.013),
                -3 - self.helper_names.index(f"patella.{s}"),
                f"patella.{s}",
            )
            # tibia: plateau, then the shaft toward the front of the shin
            if self.opts.get("plateau", True):
                plat = kn + f * 0.018
                take(
                    capsule(plat - side * 0.02, plat + side * 0.02, 0.011),
                    f"lowerleg01.{s}",
                    f"tibia.{s}",
                )
            split = seg_param(H[IX[f"lowerleg02.{s}"]][None], kn, an)[0][0]
            t, _ = seg_param(X, kn, an)
            m = capsule(kn + fwd * 0.004, an + fwd * 0.004, 0.012, 0.08, 0.95)
            take(m & (t < split), f"lowerleg01.{s}", f"tibia.{s}")
            take(m & (t >= split), f"lowerleg02.{s}", f"tibia.{s}")
            # ---------------- collarbone and the top of the shoulder blade
            for bn, r in ((f"clavicle.{s}", 0.007), (f"shoulder01.{s}", 0.009)):
                if bn.startswith("shoulder01") and not self.opts["acromion"]:
                    continue
                take(capsule(H[IX[bn]], TL[IX[bn]], r, 0.0, 0.75), bn, f"girdle.{s}")
        # ribcage: the deep part of the chest follows the upper spine
        rib = (y > 0.1) & (y < 0.335) & (np.abs(X[:, 0]) < 0.09) & (depth > 0.022)
        take(rib & (y >= 0.21), "spine01", "ribcage")
        take(rib & (y < 0.21), "spine02", "ribcage")
        for bn in ("spine05", "spine04", "spine03", "spine02", "spine01"):
            take(capsule(H[IX[bn]], TL[IX[bn]], 0.014), bn, "spine")
        # pelvis: deep tissue between the iliac crests and the sit bones, clear of the hip joints
        hipL, hipR = H[IX["upperleg01.L"]], H[IX["upperleg01.R"]]
        away = (np.linalg.norm(X - hipL, axis=1) > 0.024) & (
            np.linalg.norm(X - hipR, axis=1) > 0.024
        )
        low = y < hipL[1] - 0.01
        pel = (y > hipL[1] - 0.06) & (y < hipL[1] + 0.055) & (depth > 0.02) & away
        pel &= np.where(low, np.abs(X[:, 0]) < 0.045, np.abs(X[:, 0]) < hipL[0] + 0.005)
        take(pel, "root", "pelvis")
        self.fix = fix
        self.core_counts = counts

    def _creases(self, Wb):
        """surface nodes on the inner side of each crease, with the side of the halving plane they belong to"""
        X, H = self.Xd[: self.ns], self.H0
        self.creases = {}
        for name, cr in CREASES.items():
            c0 = H[IX[cr["lower"][0]]]
            f0 = unit(H[IX[cr["lower"][1]]] - c0)
            u0 = (
                np.array([0.0, -1.0, 0.0])
                if cr["upper"] is None
                else unit(c0 - H[IX[cr["upper"][0]]])
            )
            n0 = unit(u0 + f0)
            if "inner" in cr:
                b0 = np.array(cr["inner"])
            else:
                h = unit(np.cross(u0, f0))
                b0 = unit(np.cross(h, f0))
            wl = Wb[:, [IX[b] for b in cr["lower_bones"]]].sum(1)
            side = np.where(wl > 0.5, 1.0, -1.0)
            rel = X - c0
            ok = (
                (np.linalg.norm(rel, axis=1) < cr["radius"])
                & (rel @ b0 > 0.005)
                & (side * (rel @ n0) > 0.004)
            )
            ok &= self.fix[: self.ns] == -1
            self.creases[name] = dict(idx=np.where(ok)[0], side=side[ok])

    def contacts(self, R, F):
        """the contact terms of a pose for the free nodes of a region: free positions, planes and sides"""
        Rw, Pw = F
        pos = -np.ones(len(self.X0), int)
        pos[R["free"]] = np.arange(len(R["free"]))
        rows, cs, ns, ss = [], [], [], []
        for name, cr in CREASES.items():
            cd = self.creases[name]
            k = pos[cd["idx"]]
            keep = k >= 0
            if not keep.any():
                continue
            c = Pw[IX[cr["lower"][0]]]
            f = unit(Pw[IX[cr["lower"][1]]] - c)
            u = (
                Rw[IX["root"]] @ np.array([0.0, -1.0, 0.0])
                if cr["upper"] is None
                else unit(c - Pw[IX[cr["upper"][0]]])
            )
            n = unit(u + f)
            rows.append(k[keep])
            ss.append(cd["side"][keep])
            cs.append(np.tile(c, (keep.sum(), 1)))
            ns.append(np.tile(n, (keep.sum(), 1)))
        if not rows:
            return None
        return dict(
            rows=np.concatenate(rows),
            c=np.concatenate(cs),
            n=np.concatenate(ns),
            s=np.concatenate(ss),
            gap=CONTACT_GAP * self.size_ratio,
        )

    # ------------------------------------------------------------------ regions
    REGION_BOXES = {
        "arm.L": lambda X: (X[:, 0] > 0.02) & (X[:, 1] > -0.02),
        "arm.R": lambda X: (X[:, 0] < -0.02) & (X[:, 1] > -0.02),
        "leg.L": lambda X: (X[:, 0] > -0.03) & (X[:, 1] < 0.06),
        "leg.R": lambda X: (X[:, 0] < 0.03) & (X[:, 1] < 0.06),
        "all": lambda X: np.ones(len(X), bool),
    }

    def region(self, name):
        if name in self.regions:
            return self.regions[name]
        t0 = time.time()
        inside = self.REGION_BOXES[name](self.Xd)
        free = np.where((self.fix == -1) & inside)[0]
        isfree = np.zeros(len(self.X0), bool)
        isfree[free] = True
        tet_ids = np.where(isfree[self.tets_all].any(1))[0]
        T = self.tets_all[tet_ids]
        c = self.c_all[tet_ids]
        w = self.vol_all[tet_ids] * (self.ks + self.kv)
        touched = np.unique(T)
        cons = touched[~isfree[touched]]
        rows = np.repeat(T, 4, axis=1).ravel()
        cols = np.tile(T, (1, 4)).ravel()
        vals = (np.einsum("mkd,mld->mkl", c, c) * w[:, None, None]).ravel()
        n = len(self.X0)
        L = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        R = dict(
            name=name,
            free=free,
            cons=cons,
            T=T,
            c=c,
            w=w,
            tet_ids=tet_ids,
            L_fc=L[free][:, cons].tocsr(),
            L_ff=L[free][:, free].tocsc(),
        )
        R["solve_ff"] = spla.factorized(R["L_ff"])
        R["t_setup"] = time.time() - t0
        self.regions[name] = R
        return R

    # ------------------------------------------------------------------ pose targets
    def helper_frames(self, F):
        Rw, Pw = F
        out = []
        for name in self.helper_names:
            hd = HELPERS[name]
            ip, ic = IX[hd["parent"]], IX[hd["child"]]
            q = P.mat2q(Rw[ip].T @ Rw[ic])
            out.append(
                (Rw[ip] @ P.qmat(P.qpow(q, hd["share"])), Pw[ic], H[ic])
            )  # a turn about the child joint
        return out

    def targets(self, F, idx=None):
        """positions of nodes idx in pose F: rigid for cores and helpers, standard skinning for the rest"""
        idx = np.arange(len(self.X0)) if idx is None else idx
        Rw, Pw = F
        b = self.fix[idx]
        X0 = self.X0[idx]
        out = P.skin(F, X0, self.si[idx], self.sw[idx])
        rigid = b >= 0
        bb = b[rigid]
        out[rigid] = np.einsum("nij,nj->ni", Rw[bb], X0[rigid] - H[bb]) + Pw[bb]
        for k, (Rh, ph, h0) in enumerate(self.helper_frames(F)):
            m = b == -3 - k
            if m.any():
                out[m] = (X0[m] - h0) @ Rh.T + ph
        return out

    # ------------------------------------------------------------------ muscles
    def add_muscle(self, name, sel, fiber, drive):
        """sel: node mask; fiber: (3,) rest direction; drive(pose) -> length ratio (1 = rest, 0.85 = 15 % shorter)"""
        tm = sel[self.tets_all].mean(1) > 0.5
        self.muscles.append(
            dict(name=name, tets=np.where(tm)[0], f=unit(fiber), drive=drive)
        )

    def muscle_rest(self, R, pose):
        """rest shape change of the muscle elements in this pose: shorter along the fibre, thicker across it"""
        if not self.muscles:
            return None, None
        lookup = -np.ones(len(self.tets_all), int)
        lookup[R["tet_ids"]] = np.arange(len(R["tet_ids"]))
        A = None
        for mu_ in self.muscles:
            ratio = float(mu_["drive"](pose))
            if abs(ratio - 1) < 1e-4:
                continue
            ids = lookup[mu_["tets"]]
            ids = ids[ids >= 0]
            if A is None:
                A = np.tile(np.eye(3), (len(R["T"]), 1, 1))
            f = mu_["f"]
            ff = np.outer(f, f)
            A[ids] = (
                np.eye(3)
                + (ratio - 1) * ff
                + (1 / np.sqrt(ratio) - 1) * (np.eye(3) - ff)
            )
        if A is None:
            return None, None
        return A, np.linalg.inv(A)

    # ------------------------------------------------------------------ solve
    def _local(self, R, X, A=None, Ainv=None):
        Fm = np.einsum("mkd,mke->mde", X[R["T"]], R["c"])
        G = Fm @ Ainv if A is not None else Fm
        U, S, Vt = np.linalg.svd(G)
        neg = np.linalg.det(U) * np.linalg.det(Vt) < 0
        U[neg, :, 2] *= -1
        S[neg, 2] *= -1
        Rot = U @ Vt
        Pv = (U * unit_volume(S)[:, None, :]) @ Vt
        if A is not None:
            Rot, Pv = Rot @ A, Pv @ A
        k = self.kmix
        Pt = (1 - k) * Rot + k * Pv
        E = 0.5 * float(
            (
                R["w"]
                * (
                    (1 - k) * ((Fm - Rot) ** 2).sum((1, 2))
                    + k * ((Fm - Pv) ** 2).sum((1, 2))
                )
            ).sum()
        )
        return Pt, E

    def _global(self, R, Pt, rhs_c, C=None, Cp=None, solver=None):
        contrib = np.einsum("mkd,med->mke", R["c"], Pt) * R["w"][:, None, None]
        n = len(self.X0)
        flat = R["T"].ravel()
        cf = contrib.reshape(-1, 3)
        rhs = np.stack([np.bincount(flat, cf[:, j], minlength=n) for j in range(3)], 1)
        b = rhs[R["free"]] - rhs_c
        if C is not None:
            np.add.at(b, C["rows"], CONTACT_W * Cp)
        sv = solver or R["solve_ff"]
        return np.stack([sv(b[:, j]) for j in range(3)], 1)

    @staticmethod
    def _contact_local(C, xf):
        """projection of the contact nodes to their side of the plane, and the contact energy"""
        x = xf[C["rows"]]
        g = C["s"] * np.einsum("ij,ij->i", x - C["c"], C["n"]) - C["gap"]
        push = np.minimum(g, 0.0)
        p = x - (push * C["s"])[:, None] * C["n"]
        return p, 0.5 * CONTACT_W * float((push**2).sum())

    def solve(
        self,
        pose,
        region="all",
        iters=400,
        X_init=None,
        tol=2e-6,
        m_aa=6,
        verbose=False,
        etol=-1.0,
        contacts=True,
    ):
        """quasi-static rest of the tissue for a pose: local-global iterations with Anderson acceleration"""
        t0 = time.time()
        R = self.region(region)
        F = P.fk(pose)
        X = self.targets(F)
        if X_init is not None:
            X[R["free"]] = X_init[R["free"]]
        rhs_c = R["L_fc"] @ X[R["cons"]]
        A, Ainv = self.muscle_rest(R, pose)
        fr = R["free"]
        x = X[fr].copy()
        C = self.contacts(R, F) if contacts else None
        solver = None
        if C is not None:
            d = np.zeros(len(fr))
            np.add.at(d, C["rows"], CONTACT_W)
            solver = spla.factorized((R["L_ff"] + sp.diags(d)).tocsc())

        def local(X):
            Pt, E = self._local(R, X, A, Ainv)
            if C is None:
                return (Pt, None), E
            Cp, Ec = self._contact_local(C, X[fr])
            return (Pt, Cp), E + Ec

        Pt, E = local(X)
        E0 = E
        dG, dF = [], []
        g_prev = f_prev = None
        resets = 0
        for it in range(iters):
            g = self._global(R, Pt[0], rhs_c, C, Pt[1], solver)
            f = g - x
            if g_prev is not None:
                dG.append(g - g_prev)
                dF.append(f - f_prev)
                if len(dG) > m_aa:
                    dG.pop(0)
                    dF.pop(0)
            g_prev, f_prev = g, f
            if dF and m_aa > 0:
                theta = np.linalg.lstsq(
                    np.stack([d.ravel() for d in dF], 1), f.ravel(), rcond=None
                )[0]
                x_new = g - (np.stack([d.ravel() for d in dG], 1) @ theta).reshape(
                    g.shape
                )
            else:
                x_new = g
            X[fr] = x_new
            Pt_new, E_new = local(X)
            if dF and E_new > E:
                x_new = g
                X[fr] = g
                Pt_new, E_new = local(X)
                dG, dF = [], []
                g_prev = f_prev = None
                resets += 1
            step = np.abs(
                f
            ).max()  # size of the plain local-global step: zero at the rest state
            x, Pt, E_old, E = x_new, Pt_new, E, E_new
            if verbose and it % 10 == 0:
                print(f"  iter {it:3d}  energy {E:.6e}  plain step {step:.2e}")
            if step < tol or (E_old - E) < etol * E0:
                break
        X[fr] = x
        self.last = dict(iters=it + 1, energy=E, resets=resets, time=time.time() - t0)
        return X

    def lbs(self, pose):
        """standard skinning of the surface (8 influences), for comparison"""
        return P.skin(
            P.fk(pose), self.X0[: self.ns], self.si[: self.ns], self.sw[: self.ns]
        )
