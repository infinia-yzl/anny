# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Per-vertex skin attributes of the fine body for the viewer's shaders. Ported from the legacy 3D
Model build (build/run_bake.py, build/bake_open.py and build/export_dev.py), without the
wearables: their contact shade stays at 1 and their cover at 0, so the shaders keep their
layout.

All values ride on the vertex identity of the fine body, so they follow every phenotype.
"""

from __future__ import annotations

import numpy as np
import trimesh

from anny.hair.chart import chart, hairline

from . import bake, eyes, regions


def eyeball_meshes(eye_centers, nlat=48, nlon=64):
    Pe, Te, _ = eyes.eyeball_mesh(nlat, nlon)
    out = {}
    for s, c in eye_centers.items():
        Rm = eyes.rot_to(np.array([0, c[1], c[2] + 1.2]) - c)
        out[s] = trimesh.Trimesh((Rm @ Pe.T).T + c, Te, process=False)
    return out


def skin_attributes(body, hair=None, nrays=48, verbose=True):
    """
    Attributes of the fine body (``anny.viewer.geometry.FineBody``); ``hair`` holds the
    strands of the brows (legacy frame).

    The head is baked without its hair: the scalp tint (``attrB.w``) and the occlusion by the
    hair (``attrB.z``) come from the style that the page draws (``anny.hair.styles``), so
    ``attrB.z`` is 1 and ``attrB.w`` is 0 here.
    """
    import time

    t0 = time.time()
    V, T = body.V, body.T
    N = bake.vnormals(V, T)
    E = bake.edges(T)
    if body.mouth["xs"] and len(body.mouth["xs"]) > 3:
        regions.set_mouth(body.mouth)
    eye_meshes = eyeball_meshes(body.eye_centers)
    scene = trimesh.util.concatenate(
        [trimesh.Trimesh(V, T, process=False)] + list(eye_meshes.values())
    )
    ao = bake.bake_ao(V, N, scene, nrays=nrays)
    body_mesh = trimesh.Trimesh(V, T, process=False)
    th = bake.bake_thickness(V, N, body_mesh, nrays=8)
    th_open = bake.bake_thickness_open(V, N, body_mesh, nrays=12)
    if verbose:
        print(f"bakes: {time.time() - t0:.0f} s")
    # smoothed normal field (diffusion scale) and curvature
    Ns = bake.smooth_values(N, E, len(V), iters=40, lam=0.6)
    Ns /= np.linalg.norm(Ns, axis=1, keepdims=True)
    curv = bake.curvature(V, T, Ns, E)
    curv = bake.smooth_values(curv, E, len(V), 6)
    phi, el, _ = chart(V)
    hl = hairline(phi)
    scalp = regions.smoothstep(hl + 1.0, hl + 9.0, el) * (V[:, 1] > 0.43)
    hocc = np.ones(len(V))
    brow_pre = None
    brow = np.zeros(len(V))
    if hair is not None:
        from scipy.spatial import cKDTree

        br = hair["brows"][:, 0, :]
        dd, _ = cKDTree(br).query(V, k=12, distance_upper_bound=0.004)
        w = np.where(np.isfinite(dd), np.exp(-((dd / 0.0015) ** 2)), 0).sum(1)
        brow_pre = np.clip(w / 6.0, 0, 1) * 0.5
        cnt = np.array([len(x) for x in cKDTree(br).query_ball_point(V, 0.0016)])
        brow = np.clip(cnt / 6.0, 0, 1)
    centers = [body.eye_centers["l"], body.eye_centers["r"]]
    alb, masks = regions.compute_albedo(V, N, ao, centers, eyes.R_SCLERA, brow=brow_pre)
    J = regions.joint_positions()
    alb, bmask = regions.body_albedo(V, N, alb, J)
    alb, lm = regions.body_tone(V, N, alb, J)
    oil = regions.rough_mask(V)
    oil = np.maximum(oil, 0.8 * bmask["nail"])
    vein = regions.vein_mask(V, N, J, lm)
    fuzz = regions.fuzz_mask(V, N, J, lm, masks["lip"], scalp)
    facew = regions.face_weight(V)
    moles = regions.body_moles(V, N, lm)
    wet = np.zeros(len(V))
    for c in centers:
        dist = np.linalg.norm(V - c, axis=1)
        wet = np.maximum(
            wet,
            regions.smoothstep(eyes.R_SCLERA + 0.0016, eyes.R_SCLERA + 0.0003, dist),
        )
    zeros = np.zeros(len(V))
    earm = regions.ear_mask(V)
    pore = regions.pore_mask(V, masks["lip"], earm)
    # thickness for light through thin parts: rays that end in a closed cavity (mouth, eye
    # sockets, the folds under the arms) count as solid; the ears, hands and feet keep the plain
    # thickness because light reaches them from open space on the far side
    shell = np.clip(np.maximum(earm, np.maximum(lm["hand"], lm["foot"])), 0, 1)
    thick = th_open * (1 - shell) + th * shell
    mouth = regions.blob(
        V, regions.LM["stomion"], 0.016, mirror=False, scale=(1.5, 1.2, 1.0)
    ) * (V[:, 2] > 0.1)
    thick = np.maximum(thick, 0.03 * regions.smoothstep(0.25, 0.6, mouth))
    thick = bake.smooth_values(thick, E, len(V), 4)
    attrs = dict(
        normal=N,
        nsmooth=Ns,
        albedo=alb,
        attrA=np.stack([ao, np.clip(curv / 1000.0, -1, 1), th * 1000.0, oil], 1),
        attrB=np.stack([masks["lip"], np.clip(masks["red"], 0, 1), hocc, zeros], 1),
        attrC=np.stack([wet, brow, earm, zeros], 1),
        attrD=np.stack([thick, oil, vein, facew], 1),
        attrE=np.stack([fuzz, pore, zeros, zeros], 1),
        wear=np.ones((len(V), 4)),
        cover=np.zeros((len(V), 4)),
    )
    skin = dict(moles=moles, knuckles=regions.knuckles(J), wearables=[])
    if verbose:
        print(f"attributes: {time.time() - t0:.0f} s")
    return attrs, skin


def eye_occlusion(body, nrays=64):
    """occlusion of each eyeball by the head, on the eyeball mesh of the page (64 x 96)"""
    Pe, Te, De = eyes.eyeball_mesh(64, 96)
    head = trimesh.Trimesh(body.V, body.T, process=False)
    out, info = {}, {}
    for s, c in body.eye_centers.items():
        Rm = eyes.rot_to(np.array([0, c[1], c[2] + 1.2]) - c)
        W = (Rm @ Pe.T).T + c
        Nw = bake.vnormals(W, Te)
        bad = np.linalg.norm(Nw, axis=1) < 0.5
        Nw[bad] = (Rm @ De.T).T[bad]
        out[s] = bake.bake_ao(W, Nw, head, nrays=nrays, maxdist=0.02, eps=0.00005)
        info[s] = dict(center=c.tolist(), rot=Rm.tolist())
    zL, zcc = eyes.eye_params()
    return out, dict(
        eyes=info,
        zL=zL,
        zcc=zcc,
        R=eyes.R_SCLERA,
        Rc=eyes.R_CORNEA,
        RL=eyes.R_LIMBUS,
    )
