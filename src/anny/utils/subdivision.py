# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Catmull-Clark subdivision of quad meshes as sparse linear operators.

Catmull-Clark subdivision is linear in the vertex positions: each new vertex is a fixed
weighted sum of the vertices of the previous level. This module builds these weights once
from the quad faces, so that any per-vertex quantity (rest vertices, blend shapes, skinning
weights, corrective shapes) can be subdivided with the same map, in NumPy or in PyTorch.

:class:`MixedSubdivision` builds a mesh that uses one more level of subdivision in a region
(typically the head) than elsewhere, joined without cracks. It follows the full-body build of
the legacy 3D Model experiment (``legacy/3d_model``).

Example::

    model = anny.Anny(topology="anny-quads")
    subdivision = MixedSubdivision.for_model(model)
    output = model(phenotype_kwargs={"age": 0.2})
    fine_vertices = subdivision(output["rest_vertices"][0])
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import torch

HEAD_REGION_BONES = ("head", "neck03", "neck02", "neck01", "eye.L", "eye.R")


class SparseOperator:
    """
    A sparse linear map ``y = S x`` from ``shape[1]`` input rows to ``shape[0]`` output rows.

    Entries are stored sorted by row with no duplicates, and every output row has at least
    one entry. The operator applies to NumPy arrays or PyTorch tensors of shape ``(n, ...)``,
    and the PyTorch path is differentiable with respect to the input.
    """

    def __init__(self, rows, cols, values, shape):
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        # Sum duplicate entries and sort them by row, then by column.
        key = rows * shape[1] + cols
        order = np.argsort(key, kind="stable")
        key, values = key[order], values[order]
        first = np.ones(len(key), dtype=bool)
        first[1:] = key[1:] != key[:-1]
        starts = np.nonzero(first)[0]
        summed = np.add.reduceat(values, starts) if len(starts) else values
        key = key[starts]
        self.rows = key // shape[1]
        self.cols = key % shape[1]
        self.values = summed
        self.shape = (int(shape[0]), int(shape[1]))
        counts = np.bincount(self.rows, minlength=self.shape[0])
        if (counts == 0).any():
            raise ValueError("Every output row of a SparseOperator needs an entry.")
        self.row_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._torch_cache = {}

    @property
    def nnz(self) -> int:
        return len(self.values)

    def select_rows(self, row_indices) -> "SparseOperator":
        """Operator restricted to the given output rows, in the given order."""
        row_indices = np.asarray(row_indices, dtype=np.int64)
        counts = np.bincount(self.rows, minlength=self.shape[0])
        starts = np.concatenate([[0], np.cumsum(counts)])
        lengths = counts[row_indices]
        new_rows = np.repeat(np.arange(len(row_indices)), lengths)
        offsets = np.arange(lengths.sum()) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        entries = np.repeat(starts[row_indices], lengths) + offsets
        return SparseOperator(
            new_rows,
            self.cols[entries],
            self.values[entries],
            (len(row_indices), self.shape[1]),
        )

    def __matmul__(self, other: "SparseOperator") -> "SparseOperator":
        """Composition ``self @ other`` of two sparse operators."""
        if not isinstance(other, SparseOperator):
            return self.apply(other)
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        counts = np.bincount(other.rows, minlength=other.shape[0])
        starts = np.concatenate([[0], np.cumsum(counts)])
        lengths = counts[self.cols]
        offsets = np.arange(lengths.sum()) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        entries = np.repeat(starts[self.cols], lengths) + offsets
        return SparseOperator(
            np.repeat(self.rows, lengths),
            other.cols[entries],
            np.repeat(self.values, lengths) * other.values[entries],
            (self.shape[0], other.shape[1]),
        )

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.shape)
        dense[self.rows, self.cols] = self.values
        return dense

    def _torch_matrix(self, dtype, device):
        key = (dtype, str(device))
        if key not in self._torch_cache:
            crow = np.concatenate([self.row_starts, [self.nnz]])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support")
                self._torch_cache[key] = torch.sparse_csr_tensor(
                    torch.as_tensor(crow, device=device),
                    torch.as_tensor(self.cols, device=device),
                    torch.as_tensor(self.values, dtype=dtype, device=device),
                    size=self.shape,
                    check_invariants=False,
                )
        return self._torch_cache[key]

    def apply(self, x):
        """Apply the operator to ``x`` of shape ``(shape[1], ...)``."""
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"Expected {self.shape[1]} input rows, got {x.shape[0]}.")
        if isinstance(x, torch.Tensor):
            flat = x.reshape(x.shape[0], -1)
            matrix = self._torch_matrix(flat.dtype, flat.device)
            return (matrix @ flat).reshape((self.shape[0],) + tuple(x.shape[1:]))
        x = np.asarray(x)
        flat = x.reshape(x.shape[0], -1).astype(np.float64, copy=False)
        products = flat[self.cols] * self.values[:, None]
        out = np.add.reduceat(products, self.row_starts, axis=0)
        return out.reshape((self.shape[0],) + x.shape[1:])

    __call__ = apply


def build_edges(quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unique edges of a quad mesh.

    Returns:
        edges: (E, 2) sorted vertex pairs.
        face_edges: (F, 4) edge index of the edge from corner ``k`` to corner ``k + 1``.
    """
    quads = np.asarray(quads, dtype=np.int64)
    pairs = np.stack([quads, np.roll(quads, -1, axis=1)], -1).reshape(-1, 2)
    edges, inverse = np.unique(np.sort(pairs, 1), axis=0, return_inverse=True)
    return edges, inverse.reshape(len(quads), 4)


@dataclasses.dataclass
class CatmullClarkLevel:
    """One level of Catmull-Clark subdivision of a quad mesh."""

    operator: SparseOperator  # (V + E + F, V)
    quads: np.ndarray  # (4F, 4) faces of the subdivided mesh
    parent_faces: np.ndarray  # (4F,) index of the parent face of each new face
    edges: np.ndarray  # (E, 2) edges of the input mesh
    face_edges: np.ndarray  # (F, 4) edges of each input face
    num_vertices: int  # V, the number of input vertices


def catmull_clark(num_vertices: int, quads: np.ndarray) -> CatmullClarkLevel:
    """
    One level of Catmull-Clark subdivision of an all-quad mesh, as a linear operator.

    The new vertices come in the order: vertex points (one per input vertex), edge points
    (one per edge of :func:`build_edges`), then face points (one per face). Boundary edges
    use their midpoint, boundary vertices with two boundary edges use the (1, 6, 1) / 8 rule,
    and other boundary vertices stay in place.
    """
    quads = np.asarray(quads, dtype=np.int64)
    n_vertices, n_faces = int(num_vertices), len(quads)
    edges, face_edges = build_edges(quads)
    n_edges = len(edges)
    rows, cols, vals = [], [], []

    def add(r, c, v):
        r, c, v = np.broadcast_arrays(r, c, v)
        rows.append(r.ravel())
        cols.append(c.ravel())
        vals.append(v.astype(np.float64).ravel())

    face_rows = n_vertices + n_edges + np.arange(n_faces)
    edge_rows = n_vertices + np.arange(n_edges)

    # Face points: the mean of the four corners.
    add(face_rows[:, None], quads, 0.25)

    # Edge points: the midpoint for boundary edges; otherwise the mean of the midpoint and
    # of the two adjacent face points.
    faces_per_edge = np.bincount(face_edges.ravel(), minlength=n_edges)
    boundary_edge = faces_per_edge == 1
    add(edge_rows[boundary_edge, None], edges[boundary_edge], 0.5)
    inner = ~boundary_edge
    add(edge_rows[inner, None], edges[inner], 0.25)
    inner_face_edges = inner[face_edges]
    face_ids, corner = np.nonzero(inner_face_edges)
    add(
        edge_rows[face_edges[face_ids, corner]][:, None],
        quads[face_ids],
        0.5 * 0.5 * 0.25,
    )

    # Vertex points: (F + 2 R + (n - 3) P) / n, where F is the mean of the adjacent face
    # points, R the mean of the incident edge midpoints and n the valence.
    valence = np.bincount(edges.ravel(), minlength=n_vertices).astype(np.float64)
    faces_per_vertex = np.bincount(quads.ravel(), minlength=n_vertices).astype(
        np.float64
    )
    boundary_edges = edges[boundary_edge]
    boundary_count = np.bincount(boundary_edges.ravel(), minlength=n_vertices)
    is_boundary = boundary_count > 0
    smooth_boundary = is_boundary & (boundary_count == 2)
    corner_vertex = is_boundary & (boundary_count != 2)
    interior = ~is_boundary & (faces_per_vertex > 0)
    n = np.maximum(valence, 1)

    # Interior vertices.
    iv = np.nonzero(interior)[0]
    add(iv, iv, (n[iv] - 3) / n[iv])
    f_ids, k = np.nonzero(interior[quads])
    v = quads[f_ids, k]
    add(
        v[:, None],
        quads[f_ids],
        (1.0 / n[v] / np.maximum(faces_per_vertex[v], 1) * 0.25)[:, None],
    )
    for side in (0, 1):
        e_ids = np.nonzero(interior[edges[:, side]])[0]
        v = edges[e_ids, side]
        add(v[:, None], edges[e_ids], (2.0 / n[v] / n[v] * 0.5)[:, None])

    # Boundary vertices.
    sb = np.nonzero(smooth_boundary)[0]
    add(sb, sb, 6.0 / 8.0)
    for side in (0, 1):
        v = boundary_edges[:, side]
        keep = smooth_boundary[v]
        add(v[keep], boundary_edges[keep, 1 - side], 1.0 / 8.0)
    cv = np.nonzero(corner_vertex)[0]
    add(cv, cv, 1.0)

    # Vertices with no face stay in place.
    lonely = np.nonzero(faces_per_vertex == 0)[0]
    add(lonely, lonely, 1.0)

    operator = SparseOperator(
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(vals),
        (n_vertices + n_edges + n_faces, n_vertices),
    )
    e = n_vertices + face_edges
    f = n_vertices + n_edges + np.arange(n_faces)
    v0, v1, v2, v3 = quads.T
    new_quads = np.concatenate(
        [
            np.stack([v0, e[:, 0], f, e[:, 3]], 1),
            np.stack([v1, e[:, 1], f, e[:, 0]], 1),
            np.stack([v2, e[:, 2], f, e[:, 1]], 1),
            np.stack([v3, e[:, 3], f, e[:, 2]], 1),
        ],
        0,
    )
    parent_faces = np.concatenate([np.arange(n_faces)] * 4)
    return CatmullClarkLevel(
        operator, new_quads, parent_faces, edges, face_edges, n_vertices
    )


def quads_to_triangles(quads: np.ndarray) -> np.ndarray:
    quads = np.asarray(quads)
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], 0)


def faces_weighted_to_bones(model, faces, bone_labels=HEAD_REGION_BONES, threshold=0.5):
    """
    Mask of the faces whose corners all have at least ``threshold`` of their skinning weight
    on the given bones. The mask depends only on the topology and the rig, so it holds for
    every phenotype.
    """
    labels = list(model.bone_labels)
    ids = [labels.index(b) for b in bone_labels if b in labels]
    indices = model.vertex_bone_indices.detach().cpu().numpy()
    weights = model.vertex_bone_weights.detach().cpu().numpy()
    vertex_weight = (weights * np.isin(indices, ids)).sum(1)
    return (vertex_weight[np.asarray(faces)] >= threshold).all(1)


class MixedSubdivision:
    """
    Catmull-Clark subdivision with one extra level inside a region, joined without cracks.

    Faces of the region use ``base_level + 1`` levels, and the other faces use
    ``base_level`` levels. The corners of the coarser faces are vertex points of the finer
    level, and a coarser face that borders the region gets the finer edge points of its
    shared edges, triangulated as a fan. The output is a triangle mesh.

    Args:
        num_vertices: number of vertices of the input mesh.
        quads: (F, 4) quad faces of the input mesh. Vertices that no face uses are dropped.
        region: (F,) boolean mask of the input faces that get the extra level.
        base_level: number of levels outside the region (at least 1).
    """

    def __init__(self, num_vertices, quads, region, base_level: int = 2):
        if base_level < 1:
            raise ValueError("base_level must be at least 1.")
        quads = np.asarray(quads, dtype=np.int64)
        region = np.asarray(region, dtype=bool)
        self.input_vertex_count = int(num_vertices)
        self.input_quads = quads
        self.region = region
        self.base_level = base_level

        levels = []
        n, q, face_region = int(num_vertices), quads, region
        for _ in range(base_level):
            level = catmull_clark(n, q)
            levels.append(level)
            face_region = face_region[level.parent_faces]
            n, q = level.operator.shape[0], level.quads
        # The extra level, on the whole base-level mesh; only the rows in use are kept.
        top = catmull_clark(n, q)
        # (input, output) vertex counts of every level, the top level included
        self.level_sizes = [
            (level.num_vertices, level.operator.shape[0]) for level in levels + [top]
        ]
        base_count = n
        base_quads, base_region = q, face_region

        fine_quads = top.quads[base_region[top.parent_faces]]
        triangles = [quads_to_triangles(fine_quads)]
        edges, face_edges = top.edges, top.face_edges
        region_side = np.zeros(len(edges), int)
        other_side = np.zeros(len(edges), int)
        np.add.at(region_side, face_edges[base_region].ravel(), 1)
        np.add.at(other_side, face_edges[~base_region].ravel(), 1)
        split = (region_side > 0) & (other_side > 0)

        outer_quads = base_quads[~base_region]
        outer_edges = face_edges[~base_region]
        has_split = split[outer_edges]
        triangles.append(quads_to_triangles(outer_quads[~has_split.any(1)]))
        fan = []
        for quad, quad_edges, splits in zip(
            outer_quads[has_split.any(1)],
            outer_edges[has_split.any(1)],
            has_split[has_split.any(1)],
        ):
            ring = []
            for k in range(4):
                ring.append(quad[k])
                if splits[k]:
                    ring.append(base_count + quad_edges[k])
            # A fan from the first inserted edge point keeps the triangles well shaped.
            k0 = next(i for i, v in enumerate(ring) if v >= base_count)
            ring = ring[k0:] + ring[:k0]
            for i in range(1, len(ring) - 1):
                fan.append([ring[0], ring[i], ring[i + 1]])
        triangles.append(np.asarray(fan, dtype=np.int64).reshape(-1, 3))
        triangles = np.concatenate(triangles)

        used = np.unique(triangles)
        remap = -np.ones(top.operator.shape[0], np.int64)
        remap[used] = np.arange(len(used))
        self.triangles = remap[triangles]
        # Level of each output vertex: True where the vertex belongs to the finer level.
        fine = np.zeros(len(used), bool)
        fine[remap[np.unique(fine_quads)]] = True
        self.fine_level_vertices = fine
        self.used_top_vertices = used
        self.operators = [level.operator for level in levels] + [
            top.operator.select_rows(used)
        ]
        self.vertex_count = len(used)
        self._composed = None

    @classmethod
    def for_model(
        cls,
        model,
        region_bones=HEAD_REGION_BONES,
        threshold: float = 0.5,
        base_level: int = 2,
        faces_mask=None,
    ) -> "MixedSubdivision":
        """
        Subdivision of an Anny model built with quad faces (e.g. ``topology="anny-quads"``).

        The region with the extra level is made of the faces weighted to ``region_bones``
        (by default the head and the neck). ``faces_mask`` selects the input faces to keep
        (by default all of them).
        """
        faces = model.faces.detach().cpu().numpy()
        if faces.ndim != 2 or faces.shape[1] != 4:
            raise ValueError(
                "MixedSubdivision needs quad faces; build the model with a '-quads' topology."
            )
        if faces_mask is not None:
            faces = faces[np.asarray(faces_mask, dtype=bool)]
        region = faces_weighted_to_bones(model, faces, region_bones, threshold)
        return cls(model.template_vertices.shape[0], faces, region, base_level)

    def renumbered_rows(self, vertex_map, num_vertices: int) -> np.ndarray:
        """
        The rows of ``used_top_vertices`` for the same faces on renumbered input vertices.

        The viewer page subdivides the body vertices alone, in a compact numbering. Each level
        numbers its vertex points first, then its edge points and then its face points, so
        a different count of input vertices shifts the rows of the edge and face points.

        Args:
            vertex_map: (input vertices,) new index of each input vertex. It must keep the
                order of the vertices that the faces use, so that the edges keep their order.
            num_vertices: number of vertices of the renumbered input mesh.

        Returns:
            (output vertices,) the top-level row of each output vertex in the new numbering.
        """
        vertex_map = np.asarray(vertex_map, dtype=np.int64)
        mapped = vertex_map[np.unique(self.input_quads)]
        if (
            np.any(np.diff(mapped) <= 0)
            or mapped.min() < 0
            or mapped.max() >= num_vertices
        ):
            raise ValueError(
                "The map must keep the order of the vertices of the faces."
            )
        row_map, n_new = vertex_map, int(num_vertices)
        for n_in, n_out in self.level_sizes:
            added = n_out - n_in
            row_map = np.concatenate([row_map, n_new + np.arange(added)])
            n_new += added
        return row_map[self.used_top_vertices]

    @property
    def operator(self) -> SparseOperator:
        """The composed map from the input vertices to the output vertices."""
        if self._composed is None:
            composed = self.operators[-1]
            for op in reversed(self.operators[:-1]):
                composed = composed @ op
            self._composed = composed
        return self._composed

    def apply(self, x):
        """Subdivide a per-vertex quantity of shape ``(num_vertices, ...)``."""
        for op in self.operators:
            x = op.apply(x)
        return x

    __call__ = apply
