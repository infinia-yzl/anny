# OpenSculptBoy
# Apache License, Version 2.0
"""
The T-pose rebind of a VRM file.

VRM files rest in a T-pose: the arms along X with the palms down, the fingers straight, the
thumbs at 45 degrees toward the front and the feet forward. Anny rests in an A-pose.
:func:`vrm_t_pose` builds the world matrices T_j of an exact T-pose from Anny's rest bone poses
B_j, and :func:`rebind` turns Anny's rest mesh and morph targets into a bind mesh and targets in
that T-pose (all in Anny's frame: metres, Z up, the figure facing -Y).

CONTRACT (PR 1, stream A2 implements): the current bodies are placeholders that keep the rest
pose (T_j = B_j), so that the VRM writer runs end to end before the T-pose lands.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class Rebind:
    """A bind mesh, its morph targets and its skeleton, in Anny's frame."""

    vertices: np.ndarray  # (V, 3) float64: the bind mesh v'
    targets: np.ndarray  # (R, V, 3) float64: the target offsets, delta' = A_i delta
    bone_poses: np.ndarray  # (J, 4, 4) float64: T_j, the world matrices of the T-pose
    rest_bone_poses: np.ndarray  # (J, 4, 4) float64: B_j
    # (J, 3) float64: the joint of each bone in the bind pose
    joint_positions: np.ndarray
    method: str  # "forward" or "inverse"


def vrm_t_pose(
    model, rest_bone_poses: torch.Tensor, keep_leg_spread: bool = False
) -> torch.Tensor:
    """
    World matrices T_j (J, 4, 4) of the VRM T-pose for rest bone poses B_j (J, 4, 4).

    Root, pelvis, spine, neck, head, eyes, clavicles and shoulder01 keep T_j = B_j. The arm,
    finger and foot chains turn to the VRM T-pose; the upper legs turn so that the line from
    hip to ankle is vertical, unless ``keep_leg_spread``. The result is consistent with forward
    kinematics (``pose_parameterization="world-orient"``).
    """
    return rest_bone_poses.clone()  # placeholder: stream A2


def rebind(
    model,
    rest_vertices: np.ndarray,
    target_offsets: np.ndarray,
    rest_bone_poses: np.ndarray,
    file_weights: np.ndarray,
    file_indices: np.ndarray,
    method: str = "forward",
    keep_leg_spread: bool = False,
) -> Rebind:
    """
    The bind mesh and targets of a VRM file.

    Args:
        model: the Anny model (its full skin weights drive the forward bind).
        rest_vertices: (V, 3) rest vertices of row 0 (no facial action, no face-shape target).
        target_offsets: (R, V, 3) rest offsets of the morph targets, relative to row 0.
        rest_bone_poses: (J, 4, 4) B_j of row 0.
        file_weights, file_indices: (V, 4) skin weights that the file will carry (the inverse
            bind uses them).
        method: ``"forward"`` (v' = sum_j w_ij T_j B_j^-1 v_i with Anny's weights) or
            ``"inverse"`` (v' = (sum_j w4_ij X_j^-1)^-1 v_i with the file's weights).
        keep_leg_spread: passed to :func:`vrm_t_pose`.
    """
    if method not in ("forward", "inverse"):
        raise ValueError(f"Unknown bind method {method!r}.")
    B = np.asarray(rest_bone_poses, dtype=np.float64)
    # placeholder: stream A2
    return Rebind(
        vertices=np.asarray(rest_vertices, dtype=np.float64).copy(),
        targets=np.asarray(target_offsets, dtype=np.float64).copy(),
        bone_poses=B.copy(),
        rest_bone_poses=B.copy(),
        joint_positions=B[:, :3, 3].copy(),
        method=method,
    )
