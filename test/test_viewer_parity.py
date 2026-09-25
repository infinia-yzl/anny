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


if __name__ == "__main__":
    unittest.main()
