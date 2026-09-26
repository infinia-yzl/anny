# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The physics of anny's hair: the NumPy reference (float64) of the page's solver
(``viewer/src/hair/sim.ts``), which ``test/test_hair_dynamics.py`` compares with it.

The solver moves the simulated guides of the scalp layout (the first ``Layout.simulated`` guides
of its progressive order); the other guides take their motion through the layout's weights
(:func:`anny.hair.styles.sim_offsets`). Each step of :data:`STEP` seconds runs
:data:`SUBSTEPS` substeps:

1. Verlet integration with damping, under the gravity that the groom does not already hold. The
   groom rests in balance with gravity as it acted in the head's rest frame, so only the change of
   gravity in the head's frame acts (:func:`gravity_change`; a sag-free rest, after Hsu et al.
   2023), and the groom keeps its shape while the head is still.
2. The global shape constraints toward the posed groom (Han and Harada 2012), with a stiffness
   that falls from the free start of the guide to its tip.
3. From the root to the tip, point by point: the local shape constraint (the bend of the groom at
   the previous point, applied without a change of velocity), the colliders (capsules on the
   bones), and the length of the segment with the velocity correction of the dynamic
   follow-the-leader method (Mueller, Kim and Chentanez 2012).

The points before the free start of a guide follow the groom: the tie of a ponytail, or the part
of long hair that lies on the head (the pivot of the groom). A guide without a pivot keeps its
root only.

Example::

    sim = HairSim(rest, style.pivot[:S], style.spec["physics"], capsules_at_rest)
    for targets, g in frames:            # the posed groom (S, P, 3) and gravity_change(R_head)
        sim.step(targets, g, capsules)   # the capsules in the pose (n, 7): two ends and a radius
    motion = sim.x - targets
"""

from __future__ import annotations

import numpy as np

STEP = 1.0 / 60.0
SUBSTEPS = 1
NO_PIVOT = 100.0  # m: a pivot beyond this means that the guide lies on the head along its length
GRAVITY = 9.81
DEFAULTS = dict(
    global_stiffness=[0.5, 0.05],
    local_stiffness=0.5,
    damping=0.08,
    gravity=1.0,
    ftl_damping=0.9,
)


def gravity_change(head_rotation, share=1.0):
    """
    The acceleration that the posed groom does not hold: g - R g for the head's rotation R (3, 3),
    times ``share``; the rest groom holds g = (0, -9.81, 0).
    """
    R = np.asarray(head_rotation, np.float64)
    k = share * GRAVITY
    return np.array([k * R[0, 1], k * (R[1, 1] - 1.0), k * R[2, 1]])


def capsule_offsets(caps, X):
    """the offsets of the points X (n, 3) from the nearest points on the axis of each capsule (7,)"""
    c0 = caps[:3]
    e = caps[3:6] - c0
    ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
    if ee > 0:
        rel = X - c0
        t = np.minimum(
            1.0,
            np.maximum(
                0.0, (rel[:, 0] * e[0] + rel[:, 1] * e[1] + rel[:, 2] * e[2]) / ee
            ),
        )
    else:
        t = np.zeros(len(X))
    return X - (c0 + t[:, None] * e)


class HairSim:
    """
    The solver on S guides of P points: ``rest`` (S, P, 3) sets the segment lengths, ``pivot``
    (S,) the free start of each guide, ``params`` the physics of the style spec, and ``capsules``
    (n, 7) the colliders at rest. Each point has its own radius for each collider, at most its
    distance from the collider at rest, so the groom at rest touches no collider.
    """

    margin = 0.0015  # m: the thickness of the hair over a collider

    def __init__(self, rest, pivot, params=None, capsules=()):
        rest = np.asarray(rest, np.float64)
        self.params = dict(DEFAULTS, **(params or {}))
        S, P, _ = rest.shape
        self.S, self.P = S, P
        # the segment lengths, summed from the root as the page sums them
        L = np.zeros(S)
        for j in range(1, P):
            d = rest[:, j] - rest[:, j - 1]
            L = L + np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        self.seg = L / (P - 1)
        pivot = np.asarray(pivot, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            at_pivot = np.clip(np.ceil(pivot / self.seg - 1e-6), 1, P)
        self.free = np.where(
            self.seg < 1e-5, P, np.where(pivot < NO_PIVOT, at_pivot, 1)
        ).astype(np.int64)
        k0, k1 = self.params["global_stiffness"]
        j = np.arange(P)[None].astype(np.float64)
        f = self.free[:, None].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(f >= P - 1, 1.0, np.clip((j - f) / (P - 1 - f), 0.0, 1.0))
        self.kg = np.where(j < f, 1.0, k0 + (k1 - k0) * t)
        # the radius of each collider for each point (S, P, n)
        caps = np.asarray(capsules, np.float64).reshape(-1, 7)
        X = rest.reshape(-1, 3)
        rad = []
        for cap in caps:
            d = capsule_offsets(cap, X)
            dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
            rad.append(np.maximum(0.0, np.minimum(cap[6] + self.margin, dist - 0.0005)))
        self.rad = (
            np.stack(rad, 1).reshape(S, P, -1) if len(caps) else np.zeros((S, P, 0))
        )
        self.reset(rest)

    def reset(self, targets):
        self.x = np.array(targets, np.float64)
        self.xp = self.x.copy()

    def step(self, targets, g, capsules):
        """
        One step toward the posed groom ``targets`` (S, P, 3) under the acceleration ``g`` (3,)
        with the capsules in the pose (n, 7; their ends move, the radii of the rest hold); returns
        the largest speed of a free point in the step (m/s): its move over the step (the
        constraints change the previous positions, so ``x - xp`` stays above zero for hair that
        rests against a collider).
        """
        T = np.asarray(targets, np.float64)
        caps = np.asarray(capsules, np.float64).reshape(-1, 7)[: self.rad.shape[2]]
        x, xp, P = self.x, self.xp, self.P
        prm = self.params
        h = STEP / SUBSTEPS
        gv = np.asarray(g, np.float64) * h * h
        keep = 1.0 - prm["damping"]
        kl = prm["local_stiffness"]
        sd = prm.get("ftl_damping", 0.9)
        f = self.free
        pinned = np.arange(P)[None] < f[:, None]
        moved = 0.0
        for _ in range(SUBSTEPS):
            x[pinned] = T[pinned]
            xp[pinned] = T[pinned]
            for j in range(1, P):
                on = j >= f
                if not on.any():
                    continue
                # 1-2. integration and the global shape (the page runs them in the same pass)
                xj = x[:, j]
                v = (xj - xp[:, j]) * keep
                xp[:, j] = np.where(on[:, None], xj, xp[:, j])
                q0 = xj + v + gv
                p = np.where(on[:, None], q0 + (T[:, j] - q0) * self.kg[:, j, None], xj)
                # 3. the groom's segment, turned by the turn of the previous segment from the groom
                r = T[:, j] - T[:, j - 1]
                if j >= 2:
                    u = T[:, j - 1] - T[:, j - 2]
                    w = x[:, j - 1] - x[:, j - 2]
                    uu = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1] + u[:, 2] * u[:, 2]
                    ww = w[:, 0] * w[:, 0] + w[:, 1] * w[:, 1] + w[:, 2] * w[:, 2]
                    uw = np.sqrt(uu * ww)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        inv = 1 / uw
                        c = (
                            u[:, 0] * w[:, 0] + u[:, 1] * w[:, 1] + u[:, 2] * w[:, 2]
                        ) * inv
                        a = np.stack(
                            [
                                (u[:, 1] * w[:, 2] - u[:, 2] * w[:, 1]) * inv,
                                (u[:, 2] * w[:, 0] - u[:, 0] * w[:, 2]) * inv,
                                (u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]) * inv,
                            ],
                            1,
                        )
                        d = (
                            a[:, 0] * r[:, 0] + a[:, 1] * r[:, 1] + a[:, 2] * r[:, 2]
                        ) / (1 + c)
                    q = np.stack(
                        [
                            r[:, 0] * c
                            + (a[:, 1] * r[:, 2] - a[:, 2] * r[:, 1])
                            + a[:, 0] * d,
                            r[:, 1] * c
                            + (a[:, 2] * r[:, 0] - a[:, 0] * r[:, 2])
                            + a[:, 1] * d,
                            r[:, 2] * c
                            + (a[:, 0] * r[:, 1] - a[:, 1] * r[:, 0])
                            + a[:, 2] * d,
                        ],
                        1,
                    )
                    ok = (uw > 1e-18) & (c > -0.99)
                    r = np.where(ok[:, None], q, r)
                # the pull moves the point without changing its velocity: its target turns with the
                # strand, so as a force (a follower force) it would pump energy into the strand
                dl = np.where(on[:, None], (x[:, j - 1] + r - p) * kl, 0.0)
                p = p + dl
                xp[:, j] = xp[:, j] + dl
                # the colliders
                for k, cap in enumerate(caps):
                    dv = capsule_offsets(cap, p)
                    dd = dv[:, 0] * dv[:, 0] + dv[:, 1] * dv[:, 1] + dv[:, 2] * dv[:, 2]
                    rad = self.rad[:, j, k]
                    hit = on & (dd < rad * rad) & (dd > 1e-18)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fct = rad / np.sqrt(dd) - 1
                    p = np.where(hit[:, None], p + dv * fct[:, None], p)
                # the length, and the velocity correction of the previous point
                dv = p - x[:, j - 1]
                ln = np.sqrt(
                    dv[:, 0] * dv[:, 0] + dv[:, 1] * dv[:, 1] + dv[:, 2] * dv[:, 2]
                )
                go = on & (ln >= 1e-12)
                with np.errstate(divide="ignore", invalid="ignore"):
                    e = self.seg / ln
                nx = x[:, j - 1] + dv * e[:, None]
                corr = go & (j - 1 >= f)
                xp[:, j - 1] = np.where(
                    corr[:, None], xp[:, j - 1] + sd * (nx - p), xp[:, j - 1]
                )
                new = np.where(go[:, None], nx, p)
                mv = new - x[:, j]
                m2 = mv[:, 0] * mv[:, 0] + mv[:, 1] * mv[:, 1] + mv[:, 2] * mv[:, 2]
                moved = max(moved, float(np.where(on, m2, 0.0).max()))
                x[:, j] = np.where(on[:, None], new, x[:, j])
        return float(np.sqrt(moved) / h)
