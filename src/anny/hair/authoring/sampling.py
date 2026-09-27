# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Point sets on the skin for the hair layouts: area samples on triangles, Poisson-disk subsets
(a maximal independent set, found in vectorised rounds), and a progressive order in which every
prefix covers the surface evenly.

The progressive order is hierarchical: the Poisson set at the radius ``r`` holds a Poisson set at
``r * sqrt(2)``, which holds one at ``2 r``, and so on. The order lists the coarsest set first,
then the points that each finer set adds, each group in random order. A prefix that ends at a
level is a Poisson set; a prefix inside a level adds a random share of the next level to it. This
reaches the goal of Yuksel's sample elimination (2015) with a few vectorised passes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def sample_on_mesh(V, T, weights, n, rng):
    """``n`` points on the triangles (V, T), with probability proportional to ``weights``;
    returns the points, the triangles and the barycentric coordinates"""
    p = weights / weights.sum()
    tri = rng.choice(len(T), n, p=p)
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    bary = np.stack([1 - u - v, u, v], 1)
    P = np.einsum("nk,nkd->nd", bary, V[T[tri]])
    return P, tri, bary


def maximal_independent_set(P, r, rng, blocked=None):
    """
    A maximal subset of the points with no two closer than ``r`` (Luby's rounds: a candidate
    joins when its random priority beats every candidate neighbour). ``blocked`` points never
    join. Returns a boolean mask.
    """
    n = len(P)
    pairs = cKDTree(P).query_pairs(r, output_type="ndarray")
    a, b = (pairs[:, 0], pairs[:, 1]) if len(pairs) else (np.zeros(0, int),) * 2
    cand = np.ones(n, bool) if blocked is None else ~blocked
    chosen = np.zeros(n, bool)
    while cand.any():
        pri = rng.random(n)
        live = cand[a] & cand[b]
        aa, bb = a[live], b[live]
        lose = np.zeros(n, bool)
        lose[aa[pri[aa] < pri[bb]]] = True
        lose[bb[pri[bb] < pri[aa]]] = True
        win = cand & ~lose
        chosen |= win
        cand &= ~win
        near = np.zeros(n, bool)
        near[b[win[a]]] = True
        near[a[win[b]]] = True
        cand &= ~near
    return chosen


def poisson_subset(P, r, rng):
    """indices of a Poisson-disk subset of the points at the radius ``r``"""
    return np.nonzero(maximal_independent_set(P, r, rng))[0]


def progressive_order(P, r, rng, levels=None):
    """
    An order of the points (a Poisson set at the radius ``r``) in which every prefix covers their
    surface evenly. Returns the order and the prefix size at each level (coarsest first).
    """
    n = len(P)
    idx = np.arange(n)
    sets = [idx]
    radius = r
    while len(sets[-1]) > 8 and (levels is None or len(sets) <= levels):
        radius *= np.sqrt(2.0)
        cur = sets[-1]
        sub = cur[maximal_independent_set(P[cur], radius, rng)]
        if len(sub) == len(cur):
            break
        sets.append(sub)
    order, bounds, placed = [], [], np.zeros(n, bool)
    for s in reversed(sets):
        new = s[~placed[s]]
        new = new[rng.permutation(len(new))]
        placed[new] = True
        order.append(new)
        bounds.append(int(placed.sum()))
    return np.concatenate(order), bounds


def coverage_radius(P, subset, probes):
    """the largest distance from a probe point to the nearest point of the subset"""
    d, _ = cKDTree(P[subset]).query(probes)
    return float(d.max())
