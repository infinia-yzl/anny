# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Orthographic, flat-shaded pictures of a mesh region with labelled points, for review sheets."""

from __future__ import annotations

import numpy as np

VIEWS = {
    # name: (right axis, up axis, view direction toward the viewer)
    "front": (np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])),
    "left": (np.array([0, -1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])),
    "right": (np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([-1.0, 0, 0])),
    "below": (np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0])),
}


def draw(
    ax,
    V,
    T,
    view="front",
    points=None,
    box=None,
    labels=True,
    color=(0.85, 0.72, 0.62),
):
    """draw triangles T of vertices V seen from a view, with points {name: (3,)} on top"""
    from matplotlib.collections import PolyCollection

    right, up, toward = VIEWS[view]
    P2 = np.stack([V @ right, V @ up], -1)
    depth = V @ toward
    tri = T
    if box is not None:
        (x0, x1), (y0, y1) = box
        c = P2[tri].mean(1)
        tri = tri[(c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)]
    n = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    facing = n @ toward
    tri, facing = tri[facing > 0], facing[facing > 0]
    order = np.argsort(depth[tri].mean(1))
    shade = 0.25 + 0.75 * facing[order]
    colors = np.clip(np.asarray(color)[None] * shade[:, None], 0, 1)
    ax.add_collection(
        PolyCollection(
            P2[tri[order]], facecolors=colors, edgecolors=colors * 0.9, linewidths=0.2
        )
    )
    if points:
        for name, p in points.items():
            q = np.array([p @ right, p @ up])
            ax.plot(*q, "o", color="crimson", ms=3)
            if labels:
                ax.annotate(
                    name,
                    q,
                    fontsize=6,
                    color="navy",
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    if box is not None:
        ax.set_xlim(*box[0])
        ax.set_ylim(*box[1])
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(view, fontsize=8)
