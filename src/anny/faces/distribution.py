# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
A distribution of anny's face-shape values, calibrated against anthropometric data
(``python -m anny.faces.authoring.calibrate``; sources in ``data/faces/SOURCES.md``).

For a body of given phenotypes, the face values follow a Gaussian:

- its mean and covariance are stored at anchor ages (anny's age scale) for a male body
  (gender 0) and a female body (gender 1), and interpolate linearly in age and gender;
- the mean shifts linearly with weight and muscle (the regression of the ICT-FaceKit fits);
- with anny's race phenotypes (``phenotypes="all"``), the mean adds the race offsets that
  ANSUR II calibrates, weighted by the race values.

Example::

    model = anny.Anny(face_shapes="all")
    faces = FaceShapeDistribution(model)
    phenotypes = {"age": torch.full((8,), 0.8), "gender": torch.rand(8)}
    values = faces.sample(phenotypes)  # (8, F) in the order of model.face_shape_labels
    output = model(phenotype_kwargs=phenotypes, face_shape_kwargs=values)
"""

from __future__ import annotations

import json

import torch
from safetensors import safe_open

import anny.utils.interpolation
from anny.paths import get_anny_root_dir

DEFAULT_PATH = (
    get_anny_root_dir() / "data" / "shape_calibration" / "face_prior.safetensors"
)
RACES = ("african", "asian", "caucasian")


def load_prior(path=DEFAULT_PATH) -> tuple[dict[str, torch.Tensor], dict]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["prior"])
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors, meta


class FaceShapeDistribution(torch.nn.Module):
    """Calibrated distribution of the face-shape values of a model (see the module)."""

    def __init__(self, model, path=DEFAULT_PATH):
        super().__init__()
        if not model.face_shape_labels:
            raise ValueError(
                "The model has no face shapes; build it with face_shapes=..."
            )
        tensors, meta = load_prior(path)
        labels = meta["face_labels"]
        missing = [k for k in model.face_shape_labels if k not in labels]
        if missing:
            raise ValueError(f"The face prior does not cover {missing}.")
        idx = torch.tensor([labels.index(k) for k in model.face_shape_labels])
        dtype, device = model.dtype, model.device

        def buf(name, value):
            self.register_buffer(name, value.to(dtype=dtype, device=device))

        buf("age_anchors", tensors["age_anchors"])
        buf("gender_anchors", tensors["gender_anchors"])
        buf("means", tensors["mean"][..., idx])
        # the rows of the Cholesky factor keep the marginal distribution of a subset of labels
        buf("scale_trils", tensors["scale_tril"][..., idx, :])
        buf("weight_muscle_regression", tensors["weight_muscle_regression"][idx])
        buf("weight_muscle_centre", tensors["weight_muscle_centre"])
        buf("race_offsets", tensors["race_offsets"][..., idx])
        lo = [model.face_shape_ranges[k][0] for k in model.face_shape_labels]
        hi = [model.face_shape_ranges[k][1] for k in model.face_shape_labels]
        buf("low", torch.tensor(lo))
        buf("high", torch.tensor(hi))
        self.phenotype_labels = list(model.phenotype_labels)
        self.face_shape_labels = list(model.face_shape_labels)
        self.meta = meta
        self._parse = model._parse_parameter_kwargs

    def _phenotypes(self, phenotype_kwargs, batch_size):
        params = self._parse(
            phenotype_kwargs, self.phenotype_labels, 0.5, "phenotype_kwargs"
        )
        if batch_size is not None and params.shape[0] == 1:
            params = params.expand(batch_size, -1)
        return params

    def _value(self, params, label):
        if label in self.phenotype_labels:
            return params[:, self.phenotype_labels.index(label)]
        return params.new_full((params.shape[0],), 0.5)

    def parameters_for(
        self, phenotype_kwargs=None, batch_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mean (B, F) and Cholesky factor (B, F, K) of the face values for these phenotypes"""
        params = self._phenotypes(phenotype_kwargs, batch_size)
        age = self._value(params, "age")
        gender = self._value(params, "gender")
        wa = anny.utils.interpolation.linear_interpolation_coefficients(
            age, self.age_anchors, extrapolate=False
        )
        g0, g1 = self.gender_anchors
        t = ((gender - g0) / (g1 - g0)).clamp(0, 1)
        wg = torch.stack([1 - t, t], -1)
        w = wa[:, :, None] * wg[:, None, :]  # (B, A, 2)
        mean = torch.einsum("bag, agf -> bf", w, self.means)
        tril = torch.einsum("bag, agfk -> bfk", w, self.scale_trils)
        wm = torch.stack(
            [self._value(params, "weight"), self._value(params, "muscle")], -1
        )
        mean = mean + (wm - self.weight_muscle_centre) @ self.weight_muscle_regression.T
        if all(r in self.phenotype_labels for r in RACES):
            race = torch.stack([self._value(params, r) for r in RACES], -1)
            race = torch.nan_to_num(
                race / race.sum(-1, keepdim=True), 1 / 3, 1 / 3, 1 / 3
            )
            offsets = torch.einsum("bg, gRf -> bRf", wg, self.race_offsets)
            mean = mean + torch.einsum("bR, bRf -> bf", race, offsets)
        return mean, tril

    def mean(
        self, phenotype_kwargs=None, batch_size: int | None = None
    ) -> torch.Tensor:
        return self.parameters_for(phenotype_kwargs, batch_size)[0]

    def sample(
        self,
        phenotype_kwargs=None,
        batch_size: int | None = None,
        generator: torch.Generator | None = None,
        spread: float = 1.0,
    ) -> torch.Tensor:
        """face values (B, F) for these phenotypes, clipped to the slider ranges"""
        mean, tril = self.parameters_for(phenotype_kwargs, batch_size)
        z = torch.randn(
            tril.shape[0],
            tril.shape[2],
            generator=generator,
            dtype=tril.dtype,
            device="cpu" if generator is None else generator.device,
        ).to(tril.device)
        values = mean + spread * torch.einsum("bfk, bk -> bf", tril, z)
        return torch.maximum(torch.minimum(values, self.high), self.low)
