# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Parity of the viewer page's maths (viewer/src/anny_shape.ts, viewer/src/subdivision.ts) with anny.
Node runs the TypeScript modules directly; the test skips when node is missing.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

import anny
from anny.utils.subdivision import MixedSubdivision
from anny.viewer import export

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "parity.mjs"


def node_available():
    if shutil.which("node") is None:
        return False
    out = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout
    major = int(out.strip().lstrip("v").split(".")[0])
    return major >= 22


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestViewerParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny(topology="anny-quads").to(dtype=torch.float64)
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        model = cls.model
        rng = np.random.default_rng(3)
        labels = model.phenotype_labels
        settings = [
            {k: float(v) for k, v in zip(labels, rng.random(len(labels)))}
            for _ in range(200)
        ]
        settings[0] = {}
        settings[1] = {k: 0.0 for k in labels}
        settings[2] = {k: 1.0 for k in labels}
        settings[3] = {"age": 1 / 3, "gender": 0.5}
        cls.settings = settings
        # an exact shape space on a few vertices: every blend shape's own direction
        cls.vertex_ids = np.sort(
            rng.choice(model.template_vertices.shape[0], 300, replace=False)
        )
        n = model.stacked_phenotype_blend_shapes_mask.shape[0]
        B = model.blendshapes[:n, cls.vertex_ids].numpy().reshape(n, -1)
        _, S, Vt = np.linalg.svd(B, full_matrices=False)
        K = int((S > 1e-9 * S[0]).sum())
        comp = Vt[:K]
        scale = np.abs(comp).max(1) / 32767
        cls.components = np.round(comp / scale[:, None]).astype(np.int16)
        cls.component_scale = scale
        cls.projection = (comp @ B.T).astype(np.float32)
        joints = export.joint_tables(model)
        template = model.template_vertices.numpy()[cls.vertex_ids]
        for name, arr in (
            ("template.bin", template.astype(np.float32)),
            ("components.bin", cls.components),
            ("projection.bin", cls.projection),
            ("joint_template.bin", joints["template"].astype(np.float32)),
            ("joint_blend.bin", joints["blendshapes"].astype(np.float32)),
        ):
            arr.tofile(cls.dir / name)
        # the fine body: anny's body quads through the mixed subdivision
        base_index = model.base_mesh_vertex_indices.numpy()
        faces = model.faces.numpy()
        body = (base_index[faces] < 13380).all(1)
        cls.sub = MixedSubdivision.for_model(model, faces_mask=body)
        cls.rest = model(phenotype_kwargs={"age": 0.2})["rest_vertices"][0].numpy()
        # the page subdivides the body vertices alone, in a compact numbering
        used = np.unique(faces[body])
        compact = -np.ones(len(cls.rest), np.int64)
        compact[used] = np.arange(len(used))
        compact[faces[body]].astype(np.uint32).tofile(cls.dir / "quads.bin")
        rows = cls.sub.renumbered_rows(compact, len(used))
        rows.astype(np.uint32).tofile(cls.dir / "rows.bin")
        cls.rest[used].astype(np.float32).tofile(cls.dir / "coarse_in.bin")
        inp = dict(
            tables=export.phenotype_tables(model),
            settings=settings,
            n_shapes=n,
            component_scale=scale.tolist(),
            shape_settings=20,
            subdivision=dict(n=int(len(used)), levels=2),
        )
        with open(cls.dir / "input.json", "w") as f:
            json.dump(inp, f)
        subprocess.run(["node", str(SCRIPT), str(cls.dir)], check=True, cwd=REPO)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_coefficients(self):
        n = self.model.stacked_phenotype_blend_shapes_mask.shape[0]
        C = np.fromfile(self.dir / "coefficients.bin", dtype=np.float64).reshape(-1, n)
        for i, s in enumerate(self.settings):
            expected = export.coefficients(self.model, s)
            self.assertLess(np.abs(C[i] - expected).max(), 1e-9, s)

    def test_coarse_body_and_joints(self):
        V = np.fromfile(self.dir / "coarse.bin", dtype=np.float32).reshape(20, -1, 3)
        J = np.fromfile(self.dir / "joints.bin", dtype=np.float32).reshape(20, -1, 3)
        for i in range(20):
            with torch.no_grad():
                out = self.model(phenotype_kwargs=self.settings[i])
            rest = out["rest_vertices"][0].numpy()[self.vertex_ids]
            heads = out["rest_bone_heads"][0].numpy()
            # float32 in the page and 16-bit components: well under a millimetre
            self.assertLess(np.abs(V[i] - rest).max(), 2e-4)
            self.assertLess(np.abs(J[i] - heads).max(), 1e-5)

    def test_subdivision(self):
        F = np.fromfile(self.dir / "fine.bin", dtype=np.float32).reshape(-1, 3)
        expected = self.sub(self.rest)
        self.assertEqual(F.shape, expected.shape)
        self.assertLess(np.abs(F - expected).max(), 1e-5)


def _quantised(offsets, step):
    q = np.round(offsets / step)
    assert np.abs(q).max() < 32767
    return q.astype(np.int16)


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestViewerFaceParity(unittest.TestCase):
    """the page's face shapes (anny_shape.ts) against anny's"""

    @classmethod
    def setUpClass(cls):
        from anny.faces.distribution import DEFAULT_PATH

        model = anny.Anny(topology="anny-quads", face_shapes="all").to(
            dtype=torch.float64
        )
        cls.model = model
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        rng = np.random.default_rng(5)
        labels = model.phenotype_labels
        names = model.face_shape_labels
        settings = []
        for i in range(12):
            phen = {k: float(v) for k, v in zip(labels, rng.random(len(labels)))}
            face = {}
            for k in rng.choice(len(names), 20, replace=False):
                lo, hi = model.face_shape_ranges[names[k]]
                face[names[k]] = float(rng.uniform(lo - 0.3, hi))
            settings.append(dict(phenotype=phen, face=face))
        settings[0] = dict(phenotype={}, face={names[0]: 1.0})
        cls.settings = settings
        cls.vertex_ids = np.sort(
            rng.choice(model.template_vertices.shape[0], 3000, replace=False)
        )
        prior = None
        if DEFAULT_PATH.exists():
            prior, factors = export.face_prior_tables(model, rank=len(names))
            factors.astype(np.float32).tofile(cls.dir / "face_prior.bin")
        tables = export.face_tables(model, prior)
        fo = export.face_offsets(model, cls.vertex_ids)
        step = 1e-5  # as the page's build: offsets up to about 4 cm fit in 16 bits
        tables.update(
            starts=fo["starts"], counts=fo["counts"], bones=fo["bones"], step=step
        )
        lm = export.face_landmark_tables(model, tables["landmarks"])
        for name, arr in (
            ("face_lm_template.bin", lm["template"].astype(np.float32)),
            ("face_lm_blend.bin", lm["blendshapes"].astype(np.float32)),
            ("face_ids.bin", fo["ids"].astype(np.uint32)),
            ("face_offsets.bin", _quantised(fo["offsets"], step)),
            ("face_bones.bin", fo["bone_deltas"].astype(np.float32)),
        ):
            arr.tofile(cls.dir / name)
        cls.tables = tables
        # values in the order of the page (missing names take 0)
        face_vectors = [[s["face"].get(n, 0.0) for n in names] for s in settings]
        inp = dict(
            tables=export.phenotype_tables(model),
            settings=[],
            n_shapes=int(model.stacked_phenotype_blend_shapes_mask.shape[0]),
            face=dict(
                tables=tables,
                settings=[
                    dict(phenotype=s["phenotype"], face=v)
                    for s, v in zip(settings, face_vectors)
                ],
                vertices=int(len(cls.vertex_ids)),
                bones=int(len(model.bone_labels)),
            ),
        )
        with open(cls.dir / "input.json", "w") as f:
            json.dump(inp, f)
        # the shape space inputs of parity.mjs, empty
        for name in ("template.bin", "components.bin", "projection.bin"):
            np.zeros(0, np.float32).tofile(cls.dir / name)
        joints = export.joint_tables(model)
        joints["template"].astype(np.float32).tofile(cls.dir / "joint_template.bin")
        joints["blendshapes"].astype(np.float32).tofile(cls.dir / "joint_blend.bin")
        inp["component_scale"] = []
        inp["shape_settings"] = 0
        with open(cls.dir / "input.json", "w") as f:
            json.dump(inp, f)
        subprocess.run(["node", str(SCRIPT), str(cls.dir)], check=True, cwd=REPO)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _params(self, s):
        return self.model._parse_parameter_kwargs(
            s["phenotype"], self.model.phenotype_labels, 0.5, "phenotype_kwargs"
        )

    def test_scales_and_weights(self):
        G = len(self.tables["scale_groups"])
        R = len(self.tables["row_param"])
        S = np.fromfile(self.dir / "face_scales.bin", dtype=np.float64).reshape(-1, G)
        W = np.fromfile(self.dir / "face_weights.bin", dtype=np.float64).reshape(-1, R)
        for i, s in enumerate(self.settings):
            params = self._params(s)
            face = self.model._parse_parameter_kwargs(
                s["face"], self.model.face_shape_labels, 0.0, "face"
            )
            with torch.no_grad():
                scales = self.model._face_shape_scales(params)[0].numpy()
                weights = self.model._face_shape_coefficients(params, face)[0].numpy()
            # float32 landmarks in the page
            self.assertLess(np.abs(S[i] - scales).max(), 1e-5)
            self.assertLess(np.abs(W[i] - weights).max(), 1e-5)

    def test_offsets_and_joints(self):
        n = len(self.settings)
        V = np.fromfile(self.dir / "face_coarse.bin", dtype=np.float32).reshape(
            n, -1, 3
        )
        J = np.fromfile(self.dir / "face_joints.bin", dtype=np.float32).reshape(
            n, -1, 3
        )
        for i, s in enumerate(self.settings):
            with torch.no_grad():
                a = self.model(
                    phenotype_kwargs=s["phenotype"], face_shape_kwargs=s["face"]
                )
                b = self.model(phenotype_kwargs=s["phenotype"])
            dv = (a["rest_vertices"] - b["rest_vertices"])[0].numpy()[self.vertex_ids]
            dj = (a["rest_bone_heads"] - b["rest_bone_heads"])[0].numpy()
            # offsets in steps of 10 micrometres, summed over a few rows
            self.assertLess(np.abs(V[i] - dv).max(), 1e-4)
            self.assertLess(np.abs(J[i] - dj).max(), 1e-6)

    def test_prior_mean(self):
        from anny.faces.distribution import DEFAULT_PATH, FaceShapeDistribution

        if not DEFAULT_PATH.exists():
            self.skipTest("no face prior")
        F = len(self.model.face_shape_labels)
        M = np.fromfile(self.dir / "face_prior_mean.bin", dtype=np.float64).reshape(
            -1, F
        )
        dist = FaceShapeDistribution(self.model)
        for i, s in enumerate(self.settings):
            with torch.no_grad():
                mean = dist.mean(s["phenotype"])[0].numpy()
            self.assertLess(np.abs(M[i] - mean).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
