# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""BVH reader for MakeHuman poses (poses and clips exported from MakeHuman or MPFB2).
Ported from the legacy 3D Model build (build/bvh.py).

    import bvh
    B = bvh.load(path)            # B.names, B.parents, B.offsets, B.channels, B.frames (n_frames x n_values)
    R, T = B.local(frame)         # per joint: 3x3 local rotation in the file's axes, and the root translation

A BVH file has no rest rotations: every joint rests with the file's axes, and a joint's rotation turns its children
about its head. The rotation channels apply in the listed order as intrinsic rotations, so 'Zrotation Xrotation
Yrotation' gives R = Rz * Rx * Ry.
"""

import numpy as np


class BVH:
    pass


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load(path=None, text=None):
    """read a BVH file from `path`, or from its `text`"""
    tok = (text if text is not None else open(path).read()).split()
    B = BVH()
    B.names, B.parents, B.offsets, B.channels = [], [], [], []
    stack, i, last = [], 0, -1
    while i < len(tok):
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            B.names.append(tok[i + 1])
            B.parents.append(stack[-1] if stack else -1)
            B.offsets.append(np.zeros(3))
            B.channels.append([])
            last = len(B.names) - 1
            i += 2
        elif t == "End":
            # an end site: skip its block
            depth = 0
            i += 2
            while True:
                if tok[i] == "{":
                    depth += 1
                elif tok[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        elif t == "{":
            stack.append(last)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            B.offsets[last] = np.array([float(x) for x in tok[i + 1 : i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1])
            B.channels[last] = tok[i + 2 : i + 2 + n]
            i += 2 + n
        elif t == "MOTION":
            i += 1
            break
        else:
            i += 1
    nf = int(tok[i + 1])
    ft = float(tok[i + 4])
    i += 5
    nvals = sum(len(c) for c in B.channels)
    vals = np.array([float(x) for x in tok[i : i + nf * nvals]])
    B.frames = vals.reshape(nf, nvals)
    B.frame_time = ft
    B.offsets = np.array(B.offsets)
    B.parents = np.array(B.parents)
    B.index = {n: k for k, n in enumerate(B.names)}
    starts = np.cumsum([0] + [len(c) for c in B.channels])[:-1]
    B.starts = starts

    def local(frame=0):
        v = B.frames[frame]
        R = np.zeros((len(B.names), 3, 3))
        T = np.zeros(3)
        for j, ch in enumerate(B.channels):
            M = np.eye(3)
            for k, c in enumerate(ch):
                x = v[B.starts[j] + k]
                if c.endswith("rotation"):
                    M = M @ _rot(c[0], x)
                elif j == 0 and c.endswith("position"):
                    T["XYZ".index(c[0])] = x
            R[j] = M
        return R, T

    B.local = local
    return B
