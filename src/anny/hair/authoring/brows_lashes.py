# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Eyebrow and eyelash strand generation. Ported from the legacy 3D Model build
(build/brows_lashes.py); it works in the legacy frame of the authoring rig.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.ray.ray_pyembree import RayMeshIntersector

from anny.viewer import eyes

MM = 0.001


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Surface:
    """nearest-point queries on the head mesh (dense vertex samples)"""

    def __init__(self, V, N):
        self.V = V
        self.N = N
        self.tree = cKDTree(V)

    def sdf(self, P, k=4):
        d, i = self.tree.query(P, k=k)
        # plane distance to nearest samples, weighted
        w = 1.0 / (d + 1e-5)
        s = ((P[:, None, :] - self.V[i]) * self.N[i]).sum(2)
        sd = (s * w).sum(1) / w.sum(1)
        n = (self.N[i] * w[..., None]).sum(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return sd, n


# ---------------------------------------------------------------- eyebrows
BROW_U = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 1.0])
BROW_X = np.array([14.5, 20.5, 28.5, 36.0, 42.5, 49.0]) * MM
BROW_Y = np.array([524.5, 527.2, 530.0, 531.0, 529.6, 526.3]) * MM
BROW_H = np.array([4.6, 4.2, 3.6, 3.1, 2.4, 1.2]) * MM  # half height

EYE_REF = np.array([0.0266, 0.5111])


def brow_roots(rng, n=900, eye=None):
    u = rng.random(n * 3)
    v = rng.uniform(-1.25, 1.25, n * 3)
    # density: fuller in the head/body, feathered edges
    dens = (
        smoothstep(1.25, 0.55, np.abs(v))
        * smoothstep(1.02, 0.85, u)
        * smoothstep(-0.02, 0.06, u)
    )
    dens *= np.interp(u, [0, 0.2, 0.7, 1.0], [0.75, 1.0, 0.9, 0.5])
    keep = rng.random(len(u)) < dens
    u, v = u[keep][:n], v[keep][:n]
    x = np.interp(u, BROW_U, BROW_X)
    y = np.interp(u, BROW_U, BROW_Y) + v * np.interp(u, BROW_U, BROW_H)
    if eye is not None:
        x = x + (abs(eye[0]) - EYE_REF[0])
        y = y + (eye[1] - EYE_REF[1])
    return u, v, x, y


def project_front(inter, x, y, zdir=None):
    origins = np.stack([x, y, np.full_like(x, 0.3)], 1)
    D = np.tile([0, 0, -1.0], (len(x), 1))
    locs, ri, ti = inter.intersects_location(origins, D, multiple_hits=False)
    P = np.full((len(x), 3), np.nan)
    P[ri] = locs
    return P


def grow_fine(
    surf,
    roots,
    dirs,
    lengths,
    nseg,
    lift_deg,
    bend,
    min_off=0.12 * MM,
    curl_axis=None,
    curl=0.0,
):
    nS = len(roots)
    _, n0 = surf.sdf(roots)
    t0 = dirs - (dirs * n0).sum(1, keepdims=True) * n0
    t0 /= np.linalg.norm(t0, axis=1, keepdims=True) + 1e-12
    lift = np.radians(lift_deg)
    d = t0 * np.cos(lift)[:, None] + n0 * np.sin(lift)[:, None]
    P = np.zeros((nS, nseg + 1, 3))
    P[:, 0] = roots
    p = roots.copy()
    seg = lengths / nseg
    for i in range(1, nseg + 1):
        sd, n = surf.sdf(p)
        if curl_axis is not None:
            # rotate direction about the given axis (lash curl)
            ang = curl * seg / np.maximum(lengths, 1e-6)
            k = curl_axis
            c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
            d = d * c + np.cross(k, d) * s + k * (k * d).sum(1, keepdims=True) * (1 - c)
        else:
            # bend down toward the skin tangent
            tt = d - (d * n).sum(1, keepdims=True) * n
            tt /= np.linalg.norm(tt, axis=1, keepdims=True) + 1e-12
            d = d + (tt - d) * bend
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = p + d * seg[:, None]
        sd, n = surf.sdf(p)
        low = sd < min_off
        p[low] += n[low] * (min_off - sd[low])[:, None]
        P[:, i] = p
    return P


def make_brows(surf, inter, rng, eye=None):
    out = []
    for sgn in (1, -1):
        u, v, x, y = brow_roots(rng, eye=eye)
        P0 = project_front(inter, x * sgn, y)
        ok = ~np.isnan(P0[:, 0])
        u, v, P0 = u[ok], v[ok], P0[ok]
        # growth direction in the front plane: head points up, body lateral with herringbone, tail lateral-down
        ang_head = np.radians(58 + 10 * rng.normal(0, 1, len(u)))
        ang_body = np.radians(
            np.where(v < 0, 22, -14) * np.abs(v) ** 0.7
            + 4
            + 7 * rng.normal(0, 1, len(u))
        )
        ang_tail = np.radians(-22 + 6 * rng.normal(0, 1, len(u)))
        a = np.where(u < 0.12, ang_head, np.where(u < 0.8, ang_body, ang_tail))
        mix1 = smoothstep(0.05, 0.2, u)
        a = ang_head * (1 - mix1) + a * mix1
        mix2 = smoothstep(0.7, 0.92, u)
        a = a * (1 - mix2) + ang_tail * mix2
        dx = np.cos(a) * sgn
        dy = np.sin(a)
        dirs = np.stack([dx, dy, np.zeros_like(dx)], 1)
        L = (
            np.interp(u, [0, 0.12, 0.3, 0.6, 1.0], [4.2, 5.5, 7.5, 7.5, 5.0])
            * MM
            * rng.uniform(0.75, 1.12, len(u))
        )
        P = grow_fine(
            surf,
            P0,
            dirs,
            L,
            6,
            lift_deg=rng.uniform(8, 18, len(u)),
            bend=0.45,
            min_off=0.12 * MM,
        )
        out.append(P)
    return np.concatenate(out)


# ---------------------------------------------------------------- lashes
def lid_margin(inter_head, inter_eye, c, Rm, upper=True, n=90):
    """find the fissure boundary as seen along the eye axis; returns points on lid margin + params"""
    fwd = Rm[:, 2]
    ex = Rm[:, 0]
    ey = Rm[:, 1]
    th = (
        np.linspace(np.radians(8), np.radians(172), n)
        if upper
        else np.linspace(np.radians(-172), np.radians(-8), n)
    )
    pts = []
    tvals = []
    for k, t in enumerate(th):
        rho = np.linspace(0.0, 0.02, 400)
        dirp = ex * np.cos(t) + ey * np.sin(t)
        origins = c[None] + dirp[None] * rho[:, None] + fwd[None] * 0.03
        D = np.tile(-fwd, (len(rho), 1))
        lh, rh, _ = inter_head.intersects_location(origins, D, multiple_hits=False)
        le, re_, _ = inter_eye.intersects_location(origins, D, multiple_hits=False)
        zh = np.full(len(rho), -np.inf)
        zh[rh] = (lh - c) @ fwd
        ze = np.full(len(rho), -np.inf)
        ze[re_] = (le - c) @ fwd
        eye_first = ze > zh
        # first rho where head is in front
        idx = np.where(~eye_first & (np.arange(len(rho)) > 5))[0]
        if len(idx) == 0:
            continue
        j = idx[0]
        jj = min(j + 3, len(rho) - 1)  # slightly onto the lid
        hp = lh[np.where(rh == jj)[0]] if np.any(rh == jj) else None
        if hp is None or len(hp) == 0:
            continue
        pts.append(hp[0])
        tvals.append(t)
    return np.array(pts), np.array(tvals)


def make_lashes(surf, inter_head, eye_meshes, centers, rots, rng):
    out = []
    for side in ("l", "r"):
        c = centers[side]
        Rm = rots[side]
        inter_eye = RayMeshIntersector(eye_meshes[side])
        fwd = Rm[:, 2]
        up = np.array([0, 1.0, 0])
        for upper in (True, False):
            M, th = lid_margin(inter_head, inter_eye, c, Rm, upper)
            if len(M) < 5:
                continue
            # arc-length parametrisation
            seglen = np.linalg.norm(np.diff(M, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seglen)])
            s /= s[-1]
            nL = 120 if upper else 34
            us = np.sort(rng.random(nL))
            us = 0.03 + 0.94 * us
            roots = np.stack([np.interp(us, s, M[:, k]) for k in range(3)], 1)
            # lateral = away from nose
            lat = np.sign(c[0]) * np.array([1.0, 0, 0])
            # u measured from medial to lateral
            med_to_lat = np.sign(((M[-1] - M[0]) @ lat))
            ul = us if med_to_lat > 0 else 1 - us
            # tangent of the lid margin
            tang = np.stack(
                [
                    np.interp(us + 0.01, s, M[:, k]) - np.interp(us - 0.01, s, M[:, k])
                    for k in range(3)
                ],
                1,
            )
            tang /= np.linalg.norm(tang, axis=1, keepdims=True)
            vdir = up if upper else -up
            fan = lat * (ul - 0.35)[:, None] * (0.9 if upper else 0.6)
            d0 = fwd[None] * 0.85 + vdir[None] * (-0.02 if upper else 0.30) + fan
            d0 += rng.normal(0, 0.08, d0.shape)
            d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
            if upper:
                L = (
                    np.interp(ul, [0, 0.2, 0.6, 0.85, 1], [5.0, 7.0, 8.8, 8.0, 6.0])
                    * MM
                )
            else:
                L = np.interp(ul, [0, 0.3, 0.7, 1], [1.2, 2.2, 2.8, 2.0]) * MM
            L *= rng.uniform(0.75, 1.1, len(L))
            # push roots slightly out of the skin & forward
            _, nr = surf.sdf(roots)
            roots = roots + nr * 0.1 * MM
            # curl axis = lid tangent oriented so rotation lifts the lash upward (upper) / downward (lower)
            axis = np.cross(d0, vdir[None])
            axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
            curl = (
                np.radians(rng.uniform(50, 75, len(L)))
                if upper
                else np.radians(rng.uniform(10, 25, len(L)))
            )
            P = grow_fine(
                surf,
                roots,
                d0,
                L,
                7,
                lift_deg=np.full(len(L), 0.0),
                bend=0.0,
                min_off=-1.0,
                curl_axis=axis,
                curl=curl,
            )
            out.append(P)
    return out


def grow(V, T, N, eye_centers, seed=11):
    """brows and lashes on a head mesh with eyeballs at eye_centers ('l', 'r'; legacy frame)"""
    rng = np.random.default_rng(seed)
    surf = Surface(V, N)
    inter = RayMeshIntersector(trimesh.Trimesh(V, T, process=False))
    brows = make_brows(surf, inter, rng, eye=eye_centers["l"][:2])
    Pe, Te, _ = eyes.eyeball_mesh(64, 96)
    rots, meshes = {}, {}
    for s in "lr":
        c = eye_centers[s]
        Rm = eyes.rot_to(np.array([0, c[1], c[2] + 1.2]) - c)
        rots[s] = Rm
        meshes[s] = trimesh.Trimesh((Rm @ Pe.T).T + c, Te, process=False)
    lashes = make_lashes(surf, inter, meshes, eye_centers, rots, rng)
    return dict(brows=brows, lashes=np.concatenate(lashes))
