# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Detail face shapes: the symmetric principal components of what anny's MakeHuman face shapes miss
of the ICT-FaceKit identities.

1. **Residuals.** For the training fits of :mod:`anny.faces.authoring.fit_3d`, the corresponding
   ICT points minus anny's fitted head, in anny's rest frame, on the head vertices whose
   correspondence is valid.
2. **Components.** The residuals are made symmetric (the mean of each residual and its mirror
   image), centred, and reduced to their first principal components. A value of 1 of a
   component is ``UNIT_SD`` standard deviations of the ICT identities along it.
3. **Extension.** Each component is extended over the rest of the head (the ears, the eye and
   mouth cavities, the vertices without a valid correspondence) by harmonic interpolation on the
   base mesh, falling to zero down the neck. The eyes follow their eye cavities and the tongue
   follows the mouth cavity.
4. **Values.** The value of each component for each fit is the projection of its residual, and
   the fits then carry the detail values next to the MakeHuman ones. The coverage report of
   :mod:`anny.faces.authoring.fit_3d` runs again with them.

Usage::

    python -m anny.faces.authoring.detail [--components 10]

It writes ``data/faces/detail_shapes.safetensors`` and the ``detail`` entries of
``data/faces/face_shapes.json``, and it updates ``ANNY_CACHE_DIR/faces/ict_fits.npz``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch
from safetensors.torch import save_file

import anny
from anny.faces.authoring import fit_3d, ict
from anny.faces.authoring.base_mesh import base_mesh
from anny.faces.authoring.landmarks import mirror_map
from anny.faces.authoring.sources import cache_dir
from anny.models.face_shapes import DETAIL_PATH, FACES_DIR, face_shape_spec

COMPONENTS = 10
UNIT_SD = 3.0
# the offsets fall to zero between these heights below the lowest fitted vertex (m)
NECK_FALLOFF = 0.06
MIRROR = np.array([-1.0, 1.0, 1.0])


def residuals(fits: dict, head, ict_model, idx: np.ndarray, batch: int = 500):
    """residuals (n, H, 3) of fits in anny's rest frame: ICT points minus anny's head"""
    reg = dict(
        rotation=fits["align_rotation"],
        translation=fits["align_translation"],
        triangles=fits["corr_triangles"],
        bary=fits["corr_bary"],
    )
    out = []
    for start in range(0, len(idx), batch):
        ii = idx[start : start + batch]
        targets = fit_3d.targets_for(ict_model, reg, fits["beta"][ii]).numpy()
        with torch.no_grad():
            X = head(
                torch.tensor(fits["phenotype"][ii]), torch.tensor(fits["face"][ii])
            ).numpy()
        R, t = fits["rotation"][ii], fits["translation"][ii]
        out.append(np.einsum("bji, bhj -> bhi", R, targets - t[:, None]) - X)
    return np.concatenate(out)


def head_mirror(model, vertex_ids: np.ndarray) -> np.ndarray:
    """position in vertex_ids of the mirror image of each head vertex"""
    base = model.base_mesh_vertex_indices.numpy()
    to_model = -np.ones(len(base_mesh().V), int)
    to_model[base] = np.arange(len(base))
    position = -np.ones(len(base), int)
    position[vertex_ids] = np.arange(len(vertex_ids))
    mirror = position[to_model[mirror_map()[base[vertex_ids]]]]
    assert (mirror >= 0).all(), "the head vertices are not symmetric"
    return mirror


def components(R: np.ndarray, valid: np.ndarray, mirror: np.ndarray, k: int):
    """the first k symmetric components (k, H, 3), unit norm on the valid vertices, and the SD
    of the fits along each"""
    S = 0.5 * (R + R[:, mirror] * MIRROR)
    S = S - S.mean(0)
    A = S[:, valid].reshape(len(S), -1)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    fields = np.zeros((k, R.shape[1], 3))
    fields[:, valid] = Vt[:k].reshape(k, -1, 3)
    # a fixed sign: the largest offset points along its axis
    for i in range(k):
        flat = fields[i].ravel()
        if flat[np.argmax(np.abs(flat))] < 0:
            fields[i] *= -1
    return fields, sv[:k] / np.sqrt(len(A))


def neck_and_head(model) -> np.ndarray:
    """base-mesh mask of the vertices with any skinning weight on the neck, the head or their
    descendants (where the MakeHuman face targets act)"""
    subtree = {model.bone_labels.index("neck01")}
    for i, parent in enumerate(model.bone_parents):
        if parent in subtree:
            subtree.add(i)
    in_subtree = torch.zeros(len(model.bone_labels), dtype=torch.bool)
    in_subtree[list(subtree)] = True
    weights = model.vertex_bone_weights * in_subtree[model.vertex_bone_indices]
    mask = np.zeros(len(base_mesh().V), bool)
    mask[model.base_mesh_vertex_indices.numpy()] = (weights.sum(-1) > 0).numpy()
    return mask


def extend(
    fields_base: dict[int, np.ndarray], k: int, region: np.ndarray
) -> np.ndarray:
    """
    offsets (k, V, 3) on every base-mesh vertex: the given values on their vertices, harmonic
    over the rest of the body surface within region and above the neck, zero elsewhere, and the
    eyes and the tongue moved with the means of their cavities
    """
    mesh = base_mesh()
    V = mesh.V
    n = len(V)
    q = mesh.quads
    edges = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    A = scipy.sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
    ).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    L = scipy.sparse.diags(np.asarray(A.sum(1)).ravel()) - A
    fixed_ids = np.array(sorted(fields_base))
    lowest = V[fixed_ids, 2].min()
    body = np.zeros(n, bool)
    body[mesh.body_vertices] = True
    known = np.zeros(n, bool)
    known[fixed_ids] = True
    # zero below the neck band, outside the region and off the body surface
    assert region[fixed_ids].all()
    zero = ~body | ~region | (V[:, 2] < lowest - NECK_FALLOFF)
    zero &= ~known
    free = ~known & ~zero
    values = np.zeros((k, n, 3))
    values[:, fixed_ids] = np.stack([fields_base[i] for i in fixed_ids], 1)
    Lff = L[free][:, free].tocsc()
    Lfk = L[free][:, known]
    solve = scipy.sparse.linalg.factorized(Lff)
    for c in range(k):
        for d in range(3):
            values[c, free, d] = solve(-(Lfk @ values[c, known, d]))
    # the helpers: the eyes with their cavities and the tongue with the mouth cavity
    for helper, label in (
        ("helper-l-eye", "eye_cavity.L"),
        ("helper-r-eye", "eye_cavity.R"),
        ("helper-tongue", "mouth_cavity"),
    ):
        ids = np.unique(np.asarray(mesh.groups[helper]))
        cavity = np.unique(q[mesh.face_labels == label])
        values[:, ids] = values[:, cavity].mean(1, keepdims=True)
    # exact symmetry
    mirror = mirror_map()
    return 0.5 * (values + values[:, mirror] * MIRROR)


def write_spec(names: list[str]):
    """the detail entries of face_shapes.json (replacing any previous ones)"""
    path = FACES_DIR / "face_shapes.json"
    with open(path) as f:
        spec = json.load(f)
    spec["parameters"] = [
        p for p in spec["parameters"] if p.get("source", "makehuman") != "ict"
    ] + [
        dict(
            name=name,
            group="detail",
            positive=[],
            negative=[],
            range=[-1.0, 1.0],
            source="ict",
        )
        for name in names
    ]
    if "detail" not in spec["groups"]:
        spec["groups"].append("detail")
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    face_shape_spec.cache_clear()


def run(k: int = COMPONENTS, samples: int = 5000):
    fits = dict(np.load(cache_dir() / "ict_fits.npz"))
    makehuman = [
        str(x) for x in fits["face_labels"] if not str(x).startswith("detail-")
    ]
    fits["face"] = fits["face"][:, : len(makehuman)]
    fits["face_labels"] = np.array(makehuman)
    model = fit_3d.fitting_model()
    assert list(model.face_shape_labels) == makehuman
    ids = fits["vertex_ids"]
    head = fit_3d.HeadModel(model, ids)
    ict_model = ict.load()
    valid = fits["corr_valid"].astype(bool)
    mirror = head_mirror(model, ids)
    assert (valid == valid[mirror]).all()

    train = np.nonzero(~fits["held_out"])[0]
    R = residuals(fits, head, ict_model, train[:samples])
    fields, sd = components(R, valid, mirror, k)
    # the RMS offset of a value of 1 over the valid vertices
    rms = UNIT_SD * sd / np.sqrt(valid.sum())
    print("RMS offset of a value of 1 (mm):", np.round(1000 * rms, 2).tolist())

    # offsets of a value of 1, on the base mesh
    base = model.base_mesh_vertex_indices.numpy()[ids]
    unit = UNIT_SD * sd[:, None, None] * fields
    offsets = extend(
        {int(b): unit[:, i] for i, b in enumerate(base) if valid[i]},
        k,
        neck_and_head(model),
    )
    moved = np.nonzero(np.abs(offsets).max((0, 2)) > 1e-7)[0]
    names = [f"detail-{i + 1}" for i in range(k)]
    save_file(
        dict(
            vertex_indices=torch.tensor(moved, dtype=torch.int32),
            # rounded to 0.1 micrometre, so that runs give the same file
            offsets=torch.tensor(
                np.ascontiguousarray(np.round(offsets[:, moved], 7)),
                dtype=torch.float32,
            ),
        ),
        str(DETAIL_PATH),
        # one metadata key: safetensors writes several keys in no fixed order
        metadata=dict(
            detail=json.dumps(
                dict(
                    names=names,
                    unit_sd=UNIT_SD,
                    rms_mm=np.round(1000 * rms, 3).tolist(),
                    source="ICT-FaceKit (MIT licence), fits of anny.faces.authoring.fit_3d",
                )
            )
        ),
    )
    write_spec(names)
    print(f"{k} components on {len(moved)} vertices -> {DETAIL_PATH}")

    # the detail values of every fit: the projections of its residual (valid vertices)
    R_all = residuals(fits, head, ict_model, np.arange(len(fits["face"])))
    centre = (0.5 * (R + R[:, mirror] * MIRROR)).mean(0)
    A = (R_all - centre)[:, valid].reshape(len(R_all), -1)
    values = (A @ fields[:, valid].reshape(k, -1).T) / (UNIT_SD * sd)
    fits["face"] = np.concatenate([fits["face"], values], 1)
    fits["face_labels"] = np.array(makehuman + names)
    np.savez(cache_dir() / "ict_fits.npz", **fits)

    # the coverage report with the detail shapes
    full = anny.Anny(face_shapes="all").to(dtype=torch.float64)
    assert list(full.face_shape_labels) == makehuman + names
    fit_3d.report(fits, full, ict_model, fit_3d.HeadModel(full, ids))


def main():
    parser = argparse.ArgumentParser(description="detail face shapes from the ICT fits")
    parser.add_argument("--components", type=int, default=COMPONENTS)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    run(args.components, args.samples)


if __name__ == "__main__":
    main()
