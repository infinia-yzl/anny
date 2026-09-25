# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The eyeball of the viewer: a sclera sphere and a cornea cap blended at the limbus. Ported from
the legacy 3D Model build (build/eyes.py). Sizes are in metres for anny's default body; the
viewer scales the eyeball with the sphere it fits to anny's own eye vertices.
"""

import numpy as np

R_SCLERA = 0.0123
R_CORNEA = 0.0078
R_LIMBUS = 0.0059


def eye_params():
    zL = np.sqrt(R_SCLERA**2 - R_LIMBUS**2)
    zcc = zL - np.sqrt(R_CORNEA**2 - R_LIMBUS**2)
    return zL, zcc


def eyeball_mesh(nlat=96, nlon=128):
    """Unit eyeball in local frame: +z is optical axis (forward). Returns positions (m) and triangles.
    Surface = max(sclera sphere, cornea sphere) along each direction, with smooth blend at limbus."""
    zL, zcc = eye_params()
    # latitude from front pole (theta=0 at +z)
    th = np.linspace(0, np.pi, nlat + 1)
    # concentrate rows near the front: remap
    u = np.linspace(0, 1, nlat + 1)
    th = np.pi * (u**1.6)
    ph = np.linspace(0, 2 * np.pi, nlon + 1)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
    # ray from center along d: sclera radius R; cornea: solve |t d - (0,0,zcc)| = Rc
    b = d[..., 2] * zcc
    disc = b * b - (zcc**2 - R_CORNEA**2)
    tc = b + np.sqrt(np.maximum(disc, 0))
    tc[disc < 0] = 0
    # smooth max for blend
    k = 0.00025
    ts = R_SCLERA
    h = np.clip(0.5 + 0.5 * (tc - ts) / k, 0, 1)
    t = tc * h + ts * (1 - h) + k * h * (1 - h)
    P = d * t[..., None]
    idx = np.arange((nlat + 1) * (nlon + 1)).reshape(nlat + 1, nlon + 1)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(
        -1, 4
    )
    T = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    # remove degenerate at poles
    A, B, C = (
        P.reshape(-1, 3)[T[:, 0]],
        P.reshape(-1, 3)[T[:, 1]],
        P.reshape(-1, 3)[T[:, 2]],
    )
    area = np.linalg.norm(np.cross(B - A, C - A), axis=1)
    T = T[area > 1e-14]
    return P.reshape(-1, 3), T, d.reshape(-1, 3)


def rot_to(forward, up=np.array([0, 1, 0.0])):
    f = forward / np.linalg.norm(forward)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], 1)  # columns: local x,y,z -> world
