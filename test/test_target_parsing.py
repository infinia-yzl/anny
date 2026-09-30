# Corporis
# Apache License, Version 2.0
"""
The vectorised parse of the MakeHuman target files (``load_blend_shape``) gives exactly the
values of the parse line by line that it replaced, so the model caches stay valid.
"""

import gzip
import pathlib
import tempfile
import unittest

import roma
import torch

from anny.models.full_model import load_blend_shape

TARGETS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "anny"
    / "data"
    / "mpfb2"
    / "targets"
)


def reference_load_blend_shape(filename, vertices_count, world_transformation, dtype):
    """The former parser, line by line."""
    blend_shape = torch.zeros((vertices_count, 3), dtype=dtype)
    with gzip.open(filename, "rt") as archive:
        for line in archive.readlines():
            data = line.strip().split()
            index = int(data[0])
            assert 0 <= index < vertices_count
            offset = [float(x) for x in data[1:]]
            assert len(offset) == 3
            blend_shape[index, :] = torch.as_tensor(offset, dtype=dtype)
    return world_transformation.apply(blend_shape)


class TestTargetParsing(unittest.TestCase):
    VERTICES = 19158  # the MakeHuman base mesh

    def transform(self, dtype):
        # the transformation of full_model.load_data: decimetres to metres, Y up to Z up
        return roma.Linear(
            0.1 * roma.euler_to_rotmat("X", [90], degrees=True, dtype=dtype)
        )[None]

    def test_same_values_as_the_line_by_line_parse(self):
        files = sorted(TARGETS.rglob("*.target.gz"))
        self.assertGreater(len(files), 1000)
        for dtype in (torch.float64, torch.float32):
            transform = self.transform(dtype)
            for filename in files[::7]:  # about 180 files of every kind of target
                with self.subTest(file=filename.name, dtype=str(dtype)):
                    new = load_blend_shape(filename, self.VERTICES, transform, dtype)
                    old = reference_load_blend_shape(
                        filename, self.VERTICES, transform, dtype
                    )
                    self.assertTrue(torch.equal(new, old))

    def test_index_out_of_range_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.target.gz"
            with gzip.open(path, "wt") as f:
                f.write("0 0.1 0.2 0.3\n5 0.1 0.2 0.3\n")
            with self.assertRaises(AssertionError):
                load_blend_shape(path, 5, self.transform(torch.float64), torch.float64)


if __name__ == "__main__":
    unittest.main()
