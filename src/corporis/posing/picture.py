# Corporis
# Apache License, Version 2.0
"""
Pose from a picture: a character posed as the figure in a picture, and the flat views of it.

:func:`pose_from_image` reads the picture's landmarks with MediaPipe
(:func:`corporis.posing.landmarks.detect`), turns them into a world rotation for each bone
(:class:`corporis.posing.retarget.Retargeter`), optionally refines the rotations against the
landmarks (:func:`corporis.posing.refine.refine`), and stores the result in the character's
``pose`` and ``facial_actions``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from corporis.character import Character
from corporis.posing.landmarks import Landmarks, detect
from corporis.posing.refine import refine as refine_pose
from corporis.posing.retarget import Retargeter
from corporis.posing.skeleton import Skeleton
from corporis.render.flat import View

# the flat views: the picture's own camera, the figure from the front, and a three-quarter
# view turned 40 degrees toward the figure's right
VIEWS = ("image", "front", "three-quarter")


def pose_from_landmarks(
    landmarks: Landmarks,
    character: Character | None = None,
    model=None,
    refine: bool = True,
    face_threshold: float = 0.01,
) -> Character:
    """``character`` (the default character when None) posed by ``landmarks``: the body and
    the hands in ``pose``, and the face's scores above ``face_threshold`` in
    ``facial_actions``"""
    character = character or Character(name="pose")
    model = model if model is not None else character.build_model()
    skeleton = Skeleton(model, character.phenotype)
    retargeter = Retargeter(skeleton)
    W, face = retargeter(landmarks)
    if refine:
        W = refine_pose(retargeter, W, landmarks)
    pose = Character.pose_from_parameters(model, skeleton.params(W))
    actions = dict(character.facial_actions)
    actions.update({n: round(v, 3) for n, v in face.items() if v >= face_threshold})
    return dataclasses.replace(character, pose=pose, facial_actions=actions)


def pose_from_image(
    image,
    character: Character | None = None,
    model=None,
    refine: bool = True,
    hands: bool = True,
    face: bool = True,
) -> Character:
    """``character`` (the default character when None) posed as the most prominent figure in
    ``image`` (a path, a PIL image or an RGB array). ``hands`` and ``face`` read the hands and
    the facial expression as well as the body. Needs the ``pose`` extra (MediaPipe)."""
    landmarks = detect(image, hands=hands, face=face)
    return pose_from_landmarks(landmarks, character, model, refine)


def posed_mesh(character: Character, model=None) -> tuple[np.ndarray, np.ndarray]:
    """the vertices (V, 3) of the posed character and the triangles (F, 3) of the mesh"""
    model = model if model is not None else character.build_model()
    with torch.no_grad():
        out = model(
            pose_parameters=character.pose_parameters(model),
            pose_parameterization="local-ref",
            **character.model_kwargs(),
        )
    vertices = out["vertices"][0].double().cpu().numpy()
    return vertices, model.get_triangular_faces().cpu().numpy()


def flat_view(character: Character, name: str = "image") -> View:
    """one of :data:`VIEWS` for the posed character: ``image`` keeps the picture's camera (the
    retarget keeps the figure's turn toward it), ``front`` faces the figure's hips, and
    ``three-quarter`` turns 40 degrees from there"""
    if name not in VIEWS:
        raise ValueError(f"Unknown view {name!r}; the views are {list(VIEWS)}.")
    if name == "image":
        return View()
    forward = np.array([0.0, -1.0, 0.0])
    root = character.pose.get("bones", {}).get("root")
    if root is not None:
        import roma

        R = roma.unitquat_to_rotmat(torch.as_tensor(root, dtype=torch.float64))
        forward = R.numpy() @ forward
    return View.facing(forward, turn=-40.0 if name == "three-quarter" else 0.0)
