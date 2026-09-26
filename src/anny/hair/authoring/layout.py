# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The scalp layout that every hairstyle shares (:class:`anny.hair.layout.Layout`).

- **Guide roots:** a Poisson-disk set on the scalp of anny's default body (4.2 mm apart),
  symmetric in x, so that the mirror of a style is a permutation of its guides. They are in
  progressive order, with each root next to its mirror, and the first ``simulated`` of them carry
  the physics.
- **Render roots:** a Poisson-disk set (about 64,000) in progressive order, so that a prefix of
  any length covers the scalp evenly. Each one takes its four nearest guides, with weights.
- **Simulation table:** each guide takes its three nearest simulated guides, with weights.

The scalp is the skin above anny's default hairline (``anny.hair.chart``), less 1.5 degrees,
without the ears. The layout stores positions in the legacy frame of the authoring rig and chart
coordinates; the viewer build binds the roots to the fine body.

Usage::

    python -m anny.hair.authoring.layout [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.spatial import cKDTree

from anny.hair import closest_triangles
from anny.hair.chart import chart, hairline
from anny.hair.layout import DEFAULT_PATH

from .sampling import maximal_independent_set, progressive_order, sample_on_mesh

GUIDE_SPACING = 0.0042
ROOT_COUNT = 64000
SIMULATED = 384
# interpolation: inverse square distances, softened (as the legacy groom's)
ROOT_SOFTENING = 0.0012
SIM_SOFTENING = 0.002


def on_ear(P):
    """points on the part of the ear that stands out from the head (anny's default body)"""
    return (
        (np.abs(P[:, 0]) > 0.0712)
        & (P[:, 1] < 0.535)
        & (P[:, 2] > -0.01)
        & (P[:, 2] < 0.075)
    )


def scalp_weights(V, T, margin=1.5):
    """area of each triangle that belongs to the scalp (0 elsewhere)"""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    cen = (A + B + C) / 3
    phi, el, _ = chart(cen)
    ok = (el > hairline(phi) - margin) & (cen[:, 1] > 0.43) & ~on_ear(cen)
    return area * ok


def inverse_distance_weights(d, softening, total=255):
    """weights 1 / (d + softening)^2, rounded to integers that sum to ``total``"""
    w = 1.0 / (d + softening) ** 2
    w = w / w.sum(1, keepdims=True) * total
    q = np.floor(w).astype(np.int64)
    # the remainder goes to the largest fractions, so every row sums to ``total``
    rest = total - q.sum(1)
    frac_order = np.argsort(-(w - q), axis=1)
    for k in range(w.shape[1]):
        add = rest > k
        q[np.nonzero(add)[0], frac_order[add, k]] += 1
    return q.astype(np.uint8)


def guide_roots(V, T, weights, rng, spacing=GUIDE_SPACING, candidates=80000):
    """symmetric Poisson-disk roots: (positions, mirror index, progressive order)"""
    P, _, _ = sample_on_mesh(V, T, weights, candidates, rng)
    left = P[P[:, 0] >= spacing / 2]
    mid = P[np.abs(P[:, 0]) < spacing / 2] * np.array([0.0, 1.0, 1.0])
    half = np.concatenate([left, mid])
    keep = maximal_independent_set(half, spacing, rng)
    half, is_left = half[keep], np.arange(len(half))[keep] < len(left)
    # the order of the half set, each left root followed by its mirror
    order, _ = progressive_order(half, spacing, rng)
    pos, mirror = [], []
    for i in order:
        p = half[i]
        if is_left[i]:
            k = len(pos)
            pos += [p, p * np.array([-1.0, 1.0, 1.0])]
            mirror += [k + 1, k]
        else:
            mirror.append(len(pos))
            pos.append(p)
    pos = np.array(pos)
    # onto the surface (the mid-line roots moved to x = 0; the body is symmetric to a few µm)
    tri, bary, _ = closest_triangles(pos, V, T)
    pos = np.einsum("nk,nkd->nd", bary, V[T[tri]])
    return pos, np.array(mirror)


def render_roots(V, T, weights, rng, count=ROOT_COUNT):
    """about ``count`` Poisson-disk roots in progressive order, and the level bounds"""
    area = weights.sum()
    # the density of a maximal Poisson set is about 0.7 / r^2; start there and tighten
    r = np.sqrt(0.7 * area / count)
    P, _, _ = sample_on_mesh(V, T, weights, int(count * 6), rng)
    for _ in range(8):
        keep = np.nonzero(maximal_independent_set(P, r, rng))[0]
        if abs(len(keep) - count) < 0.02 * count:
            break
        r *= np.sqrt(len(keep) / count)
    P = P[keep]
    order, bounds = progressive_order(P, r, rng)
    return P[order], bounds, r


def build(V, T, seed=5, simulated=SIMULATED, root_count=ROOT_COUNT, verbose=True):
    rng = np.random.default_rng(seed)
    w = scalp_weights(V, T)
    G, mirror = guide_roots(V, T, w, rng)
    R, bounds, r_root = render_roots(V, T, w, rng, root_count)
    d, idx = cKDTree(G).query(R, k=4)
    roots_w = inverse_distance_weights(d, ROOT_SOFTENING)
    S = min(simulated, len(G))
    ds, si = cKDTree(G[:S]).query(G, k=3)
    sim_w = inverse_distance_weights(ds, SIM_SOFTENING)
    # a simulated guide follows itself alone
    own = np.arange(S)
    si[own] = np.stack([own, own, own], 1)
    sim_w[own] = np.array([255, 0, 0], np.uint8)
    out = {
        "guides.position": G.astype(np.float32),
        "guides.chart": np.stack(chart(G)[:2], 1).astype(np.float32),
        "guides.mirror": mirror.astype(np.int32),
        "guides.sim": si.astype(np.int32),
        "guides.sim_weights": sim_w,
        "roots.position": R.astype(np.float32),
        "roots.chart": np.stack(chart(R)[:2], 1).astype(np.float32),
        "roots.guides": idx.astype(np.int32),
        "roots.weights": roots_w,
    }
    meta = dict(
        version=1,
        guides=int(len(G)),
        roots=int(len(R)),
        simulated=int(S),
        guide_spacing=GUIDE_SPACING,
        root_spacing=float(r_root),
        root_levels=[int(b) for b in bounds],
        seed=seed,
    )
    if verbose:
        print(
            f"layout: {len(G)} guides ({S} simulated), {len(R)} roots "
            f"{r_root * 1000:.2f} mm apart, {len(bounds)} levels"
        )
    return out, meta


def save(path, arrays, meta):
    from safetensors.numpy import save_file

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: np.ascontiguousarray(v) for k, v in arrays.items()},
        str(path),
        metadata={"layout": json.dumps(meta)},
    )


def main():
    from anny.viewer.geometry import fine_body

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_PATH)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    body = fine_body()
    arrays, meta = build(body.V, body.T, seed=args.seed)
    save(args.out, arrays, meta)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
