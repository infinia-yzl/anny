# OpenSculptBoy
# Apache License, Version 2.0
"""
The T-pose rebind of a VRM file.

VRM files rest in a T-pose: the arms along X with the palms down, the fingers straight, the
thumbs at 45 degrees toward the front and the feet forward. Anny rests in an A-pose.
:func:`vrm_t_pose` builds the world matrices T_j of an exact T-pose from Anny's rest bone poses
B_j, and :func:`rebind` turns Anny's rest mesh and morph targets into a bind mesh and targets in
that T-pose (all in Anny's frame: metres, Z up, the figure facing -Y).

The T-pose, in Anny's frame (s = +1 for the left side, -1 for the right side):

- the root, the pelvis bones, the spine, the neck, the head, the eyes, the clavicles and
  shoulder01 keep their rest pose (T_j = B_j);
- the upper arm, the forearm and the hand point along (s, 0, 0); the direction of a bone runs
  from its head to the head of the next bone of its humanoid chain (upper arm to forearm,
  forearm to wrist), and the twist bones (upperarm02, lowerarm02) turn with the bone they
  twist;
- the palm (the plane through the heads of the wrist, the index finger and the little finger)
  faces down (-Z), the back of the hand up (+Z), and the hand points along (s, 0, 0) from the
  wrist toward the middle finger, within the palm (see :func:`palm`); the metacarpals turn
  with the hand, so the knuckles keep their spread;
- the bones of the four fingers point along (s, 0, 0) with the nails up, and the last bone of
  each of them turns with the bone before it, as Unity's humanoid does for the distal bones;
- the thumb points along (s, -1, 0) / sqrt(2), level and at 45 degrees toward the front, and
  its last bone (finger1-3, which carries the nail) turns about the thumb's axis so that the
  nail faces (-s, -1, 0) / sqrt(2), level and a quarter turn from the nails of the other
  fingers (VRM T-pose definition 1.8: for the left thumb, between -X and +Z of the file);
- the legs turn as one piece about the hip, so that the line from the hip (upperleg01) to the
  ankle (foot) is vertical, unless ``keep_leg_spread``;
- the feet turn about the vertical axis so that they point along -Y (the line from the ankle
  to the middle toe, seen from above), and keep their rest pitch and roll so that the soles
  stay level; the toes turn with the feet.

The arm and finger bones first turn with their parent and then by the smallest change that
aims them at their target, so the joints straighten without twisting; the finger bones make
that change about the normal of the palm and then about the knuckle axis (yaw, then pitch), so
that their nails stay perpendicular to the palm. The last bone of the thumb then rolls about
the thumb's axis until its nail, carried from the rest pose (:func:`thumb_nail`), faces its
target; the two bones at the base of the thumb keep their aim without a roll, so that the web
between the thumb and the index finger keeps its shape (:data:`THUMB_ROLL`). The hand, the
legs and the feet take the turns above from their rest pose. The positions then follow from
forward kinematics with absolute orientations (``pose_parameterization="world-orient"``): the
head of every child bone is T_p B_p^-1 h, where p is its parent and h its rest head.

The library's ``t_pose`` is not used: it turns the clavicles, and its ``local-ref`` identity is
the reference pose of another body shape.

:func:`rebind` offers two bind meshes. The forward bind is Anny's own skinning into the
T-pose; the inverse bind is the mesh that the file's 4 weights skin back onto Anny's rest pose
exactly, positions and normals alike. :func:`bind_error` measures both against Anny's posed
meshes with the file's weights: on the arms-down poses (``relaxed``, ``walk``) the inverse
bind lands closer, by a third at the 99th percentile of the vertex distances (8.5 mm against
12.5 mm on the default body) and by a third to a half at the largest distance (12 to 14 mm
against 21 mm), and the 99th percentile of its normal angles stays at 14 to 18 degrees,
against 28 to 29 for the forward bind. :func:`humanoid_error` measures the file as a VRM app
drives it: the humanoid bones alone, and the roll constraints of the twist bones.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

# The bones that keep their rest pose. Any other bone without a rule below turns with its
# parent, so these stay at rest because their parents do.
KEPT_BONES = (
    "root",
    "pelvis.L",
    "pelvis.R",
    "spine05",
    "spine04",
    "spine03",
    "spine02",
    "spine01",
    "neck01",
    "neck02",
    "neck03",
    "head",
    "eye.L",
    "eye.R",
    "clavicle.L",
    "clavicle.R",
    "shoulder01.L",
    "shoulder01.R",
)
# The four fingers (index to little) and their metacarpals in the anny rig: finger k hangs from
# metacarpal k - 1.
FINGERS = (2, 3, 4, 5)
_UP = np.array([0.0, 0.0, 1.0])
# The normal of the left thumb nail in the rest frame of the last thumb bone (finger1-3.L):
# the area-weighted normal of the faces that MakeHuman's fingernail mask
# (``anny/data/mpfb2/textures/mpfb_fingernails.jpg``) covers and that finger1-3.L carries with
# a weight above 0.9, measured on Anny's default body. The rest frames of the anny rig follow
# the vertices of each bone (``"cached"`` orientations), so this direction follows the nail of
# other bodies closely, though not exactly. Over the 64 corners of the six default phenotypes,
# the nail of the rest mesh lies up to 6.1 degrees from it (3.3 at the median), and the nail of
# the T-posed mesh rolls up to 4.05 degrees off its target (on the corners at age 0; 1.2 at
# age 1, 0.6 on the default body). The right thumb mirrors it (x -> -x).
THUMB_NAIL = (-0.177, 0.022, -0.984)
# The share of the thumb's roll (34 degrees on the default body, 30 to 44 over the corners of
# the phenotype space) that each thumb bone takes about the thumb's axis. The last bone
# (finger1-3) carries the nail and takes the whole roll; the two bones at the base of the thumb
# keep their aim. The web between the thumb and the index finger blends these two bones with
# the index metacarpal (metacarpal1), which turns with the hand, so a roll of either of them
# creases the web. Measured on the default body, for the inverse and then the forward bind,
# over the 1712 edges of the skin of each hand that the thumb bones, metacarpal1 and the wrist
# carry (the largest dihedral angle between the two triangles of an edge; 96.6 degrees on the
# rest mesh), and over the 247 triangles of the web (those that finger1-1 carries: their area
# against the rest mesh, at the 1st percentile and at least):
#
# - with the whole roll on finger1-2 and finger1-3 and half of it on finger1-1, the web folds
#   back onto itself (179.6 and 179.5 degrees) and keeps 0.37 and 0.34 of its area (0.21 and
#   0.16 at least);
# - with the roll on finger1-3 alone, as without any roll, the hand bends by 96.6 and 103.0
#   degrees at most, and the web keeps 0.49 and 0.39 of its area (0.40 and 0.28 at least);
# - each degree of roll of finger1-2 adds about 1.5 degrees to the sharpest crease. A share of
#   0.2 keeps the default body and the fixture body of the tests below 120 degrees (118.6 at
#   most), and on the 64 corners of the six default phenotypes every share above 0.05 folds
#   edges that the bind without a roll keeps open (over both hands and both binds, 8 edges at
#   0.1 and 72 at 0.2). A share of 0.05 turns finger1-2 by less than 2 degrees, too little to
#   matter, so it takes none of the roll; a share of finger1-1 adds to the crease as well.
#
# The roll of finger1-3 twists the joint between the phalanges: the triangles that blend
# finger1-2 and finger1-3 keep 0.87 of their rest area or more in the forward bind, and 0.91 or
# more in the inverse bind.
THUMB_ROLL = {"finger1-1": 0.0, "finger1-2": 0.0, "finger1-3": 1.0}


@dataclasses.dataclass(frozen=True)
class _Rule:
    """How one bone turns: ``kind`` is "aim", "fan", "hand", "leg" or "foot"."""

    kind: str
    start: str  # the bone whose head starts the direction
    end: str  # the bone whose head ends the direction
    target: tuple[float, float, float]


def _sign(side: str) -> float:
    return 1.0 if side == "L" else -1.0


def thumb_direction(side: str) -> np.ndarray:
    """The thumb's direction in the T-pose: (s, -1, 0) / sqrt(2), level and 45 degrees ahead."""
    return np.array([_sign(side), -1.0, 0.0]) / np.sqrt(2.0)


def thumb_nail_target(side: str) -> np.ndarray:
    """
    The direction that the thumb nail faces in the T-pose: (-s, -1, 0) / sqrt(2), level and
    perpendicular to the thumb (in the file, between -X and +Z for the left thumb).
    """
    return np.array([-_sign(side), -1.0, 0.0]) / np.sqrt(2.0)


def thumb_nail(rest_bone_poses: np.ndarray, labels, side: str) -> np.ndarray:
    """
    The unit normal of the thumb nail of one side at rest (Anny's frame), for rest bone poses
    (J, 4, 4): :data:`THUMB_NAIL` in the rest frame of finger1-3.
    """
    local = np.array(THUMB_NAIL) * [_sign(side), 1.0, 1.0]
    rotation = np.asarray(rest_bone_poses, dtype=np.float64)[
        list(labels).index(f"finger1-3.{side}"), :3, :3
    ]
    return _unit(rotation @ local)


def _side_rules(side: str, keep_leg_spread: bool) -> dict[str, _Rule]:
    s = _sign(side)
    along = (s, 0.0, 0.0)
    thumb = tuple(thumb_direction(side))

    def b(name):
        return f"{name}.{side}"

    rules = {
        b("upperarm01"): _Rule("aim", b("upperarm01"), b("lowerarm01"), along),
        b("lowerarm01"): _Rule("aim", b("lowerarm01"), b("wrist"), along),
        b("wrist"): _Rule("hand", b("wrist"), b("finger3-1"), along),
        b("finger1-1"): _Rule("fan", b("finger1-1"), b("finger1-2"), thumb),
        b("finger1-2"): _Rule("fan", b("finger1-2"), b("finger1-3"), thumb),
        b("foot"): _Rule("foot", b("foot"), b("toe3-1"), (0.0, -1.0, 0.0)),
    }
    for k in FINGERS:
        rules[b(f"finger{k}-1")] = _Rule(
            "fan", b(f"finger{k}-1"), b(f"finger{k}-2"), along
        )
        rules[b(f"finger{k}-2")] = _Rule(
            "fan", b(f"finger{k}-2"), b(f"finger{k}-3"), along
        )
    if not keep_leg_spread:
        rules[b("upperleg01")] = _Rule(
            "leg", b("upperleg01"), b("foot"), (0.0, 0.0, -1.0)
        )
    return rules


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _unit_rows(v: np.ndarray) -> np.ndarray:
    """The rows of ``v`` (..., 3) at unit length; zero rows stay zero."""
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(length > 0, length, 1.0)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """The rotation matrix of ``angle`` radians about the unit ``axis``."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation that turns the direction ``a`` into the direction ``b``."""
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    sin, cos = np.linalg.norm(axis), float(np.dot(a, b))
    if sin < 1e-12:
        if cos > 0.0:
            return np.eye(3)
        # Opposite directions: half a turn about any axis perpendicular to a.
        other = np.eye(3)[int(np.argmin(np.abs(a)))]
        return _axis_angle(_unit(np.cross(a, other)), np.pi)
    return _axis_angle(axis / sin, float(np.arctan2(sin, cos)))


def _yaw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The turn about the vertical axis that brings ``a``, seen from above, onto ``b``."""
    yaw = np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
    return _axis_angle(_UP, float(yaw))


def _yaw_pitch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    The rotation that turns the direction ``a`` into the direction ``b`` about the vertical axis
    and then about a level axis: the turn of a finger about the normal of a level palm, then
    about its knuckle axis.
    """
    turn = _yaw(a, b)
    return _rotation_between(turn @ a, b) @ turn


def palm(heads: np.ndarray, labels, side: str) -> tuple[np.ndarray, np.ndarray]:
    """
    The direction of a hand and the normal of the back of the hand, for bone heads (J, 3).

    The palm is the plane through the heads of the wrist, the index finger (finger2-1) and the
    little finger (finger5-1); the normal of the back points away from the palm. The direction
    runs from the wrist to the middle finger (finger3-1) within that plane: the middle knuckle
    stands about 2.5 degrees above the plane, which would otherwise tilt the palm.
    """
    labels = list(labels)
    s = 1.0 if side == "L" else -1.0

    def head(name):
        return heads[labels.index(f"{name}.{side}")]

    wrist = head("wrist")
    back = _unit(s * np.cross(head("finger2-1") - wrist, head("finger5-1") - wrist))
    middle = head("finger3-1") - wrist
    return _unit(middle - (middle @ back) * back), back


def t_pose_turns(
    labels, parents, rest_bone_poses: np.ndarray, keep_leg_spread: bool = False
) -> np.ndarray:
    """
    The world rotation that turns each bone from its rest pose into the VRM T-pose: (J, 3, 3)
    matrices U_j, so that the T-pose orientation of bone j is U_j R_j, with R_j its rest
    orientation.
    """
    labels = list(labels)
    parents = [int(p) for p in parents]
    index = {label: j for j, label in enumerate(labels)}
    rules = {}
    for side in ("L", "R"):
        rules.update(_side_rules(side, keep_leg_spread))
    missing = sorted(
        {name for r in rules.values() for name in (r.start, r.end)} - set(index)
    )
    if missing:
        raise ValueError(
            f"The VRM T-pose needs the bones of the anny rig; missing {missing}."
        )
    B = np.asarray(rest_bone_poses, dtype=np.float64)
    heads = B[:, :3, 3]
    turns = np.tile(np.eye(3), (len(labels), 1, 1))
    for j in _parents_first(parents):
        p = parents[j]
        carry = turns[p] if p >= 0 else np.eye(3)
        rule = rules.get(labels[j])
        if rule is None:
            turns[j] = carry
        else:
            direction = heads[index[rule.end]] - heads[index[rule.start]]
            target = np.asarray(rule.target)
            if rule.kind == "aim":
                turns[j] = _rotation_between(carry @ direction, target) @ carry
            elif rule.kind == "fan":
                turns[j] = _yaw_pitch(carry @ direction, target) @ carry
            elif rule.kind == "leg":
                turns[j] = _rotation_between(direction, target)
            elif rule.kind == "foot":
                # A turn about the vertical axis alone keeps the sole level and the pitch.
                turns[j] = _yaw(direction, target)
            elif rule.kind == "hand":
                along, back = palm(heads, labels, labels[j][-1])
                turns[j] = _frame(target, _UP) @ _frame(along, back).T
    for side in ("L", "R"):
        # The thumb lies along its axis through the head of finger1-1, so a roll of its bones
        # about that axis keeps its direction and its joints. Each thumb bone takes its share
        # of the roll that brings the nail of finger1-3 onto its target.
        axis = thumb_direction(side)
        angle = _thumb_roll(turns[index[f"finger1-3.{side}"]], B, labels, side)
        for name, share in THUMB_ROLL.items():
            j = index[f"{name}.{side}"]
            turns[j] = _axis_angle(axis, share * angle) @ turns[j]
    return turns


def _thumb_roll(turn: np.ndarray, rest_bone_poses, labels, side: str) -> float:
    """
    The roll about the thumb axis (radians) that brings the nail of the last thumb bone,
    turned by ``turn`` from its rest pose, onto :func:`thumb_nail_target`. The last bone keeps
    its rest bend, so the nail tilts a little toward the tip; the roll aligns the part of the
    nail normal across the axis.
    """
    axis = thumb_direction(side)
    nail = turn @ thumb_nail(rest_bone_poses, labels, side)
    nail = nail - (nail @ axis) * axis
    target = thumb_nail_target(side)
    return float(np.arctan2(axis @ np.cross(nail, target), nail @ target))


def _frame(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The orthonormal frame (columns x, y, z) with x along ``x`` and z nearest to ``z``."""
    x = _unit(x)
    z = _unit(z - (z @ x) * x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def _parents_first(parents: list[int]) -> list[int]:
    """The bone indices in an order where every parent comes before its children."""
    order, placed = [], set()
    pending = list(range(len(parents)))
    while pending:
        rest = []
        for j in pending:
            if parents[j] < 0 or parents[j] in placed:
                order.append(j)
                placed.add(j)
            else:
                rest.append(j)
        if len(rest) == len(pending):
            raise ValueError("The bone hierarchy has a cycle.")
        pending = rest
    return order


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
    from anny.utils.kinematics import parallel_forward_kinematic_absolute_orientations

    B = torch.as_tensor(rest_bone_poses)
    batched = B.dim() == 4
    if batched and B.shape[0] != 1:
        raise ValueError("vrm_t_pose takes the rest bone poses of one body.")
    B = B.reshape(-1, 4, 4)
    turns = t_pose_turns(
        model.bone_labels,
        model.bone_parents,
        B.detach().cpu().double().numpy(),
        keep_leg_spread=keep_leg_spread,
    )
    orientations = torch.as_tensor(turns, dtype=B.dtype, device=B.device) @ B[:, :3, :3]
    poses, _ = parallel_forward_kinematic_absolute_orientations(
        model.kinematic_propagation_fronts,
        rest_bone_poses=B[None],
        absolute_orientations=orientations[None],
    )
    return poses if batched else poses[0]


def rigid_inverse(matrices: np.ndarray) -> np.ndarray:
    """The inverses of rigid 4x4 matrices (..., 4, 4)."""
    matrices = np.asarray(matrices, dtype=np.float64)
    inverse = np.zeros_like(matrices)
    rotation = np.swapaxes(matrices[..., :3, :3], -1, -2)
    inverse[..., :3, :3] = rotation
    inverse[..., :3, 3] = -(rotation @ matrices[..., :3, 3, None])[..., 0]
    inverse[..., 3, 3] = 1.0
    return inverse


def blend(weights: np.ndarray, indices: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """
    Per-vertex blends sum_k w_ik M[j_ik] (V, 4, 4) of bone matrices M (J, 4, 4), for skin
    weights and bone indices (V, K).
    """
    weights = np.asarray(weights, dtype=np.float64)
    return np.einsum("vk,vkab->vab", weights, matrices[np.asarray(indices)])


def skin(
    vertices: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    """
    Linear blend skinning sum_k w_ik M[j_ik] v_i of vertices (V, 3) with bone transforms
    (J, 4, 4) or (F, J, 4, 4); returns (V, 3) or (F, V, 3).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    indices = np.asarray(indices)
    result = np.zeros(transforms.shape[:-3] + vertices.shape)
    for k in range(weights.shape[1]):
        m = transforms[..., indices[:, k], :, :]  # (..., V, 4, 4)
        moved = (m[..., :3, :3] @ vertices[..., None])[..., 0] + m[..., :3, 3]
        result += weights[:, k, None] * moved
    return result


@dataclasses.dataclass
class Rebind:
    """A bind mesh, its normals, its morph targets and its skeleton, in Anny's frame."""

    vertices: np.ndarray  # (V, 3) float64: the bind mesh v'
    # (V, 3) float64, unit length: the normals of the file (see :func:`rebind`)
    normals: np.ndarray
    targets: np.ndarray  # (R, V, 3) float64: the target offsets, delta' = A_i delta
    bone_poses: np.ndarray  # (J, 4, 4) float64: T_j, the world matrices of the T-pose
    rest_bone_poses: np.ndarray  # (J, 4, 4) float64: B_j
    # (J, 3) float64: the joint of each bone in the bind pose
    joint_positions: np.ndarray
    method: str  # "forward" or "inverse"
    # (3,) float64: the translation, in Anny's frame, that centres the figure: the hips (the
    # head of root) at x = y = 0 and the lowest bind vertex at z = 0. The file adds it to the
    # vertices and the joints (mapped by its frame); ``vertices``, ``bone_poses`` and
    # ``joint_positions`` stay in Anny's frame without it.
    offset: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))

    @property
    def transforms(self) -> np.ndarray:
        """X_j = T_j B_j^-1 (J, 4, 4): the turn of each bone from Anny's rest pose."""
        return self.bone_poses @ rigid_inverse(self.rest_bone_poses)


def centring_offset(
    vertices: np.ndarray, joint_positions: np.ndarray, labels
) -> np.ndarray:
    """
    The translation (3,) in Anny's frame that puts the hips (the head of ``root``) at
    x = y = 0 and the lowest of ``vertices`` at z = 0.
    """
    hips = np.asarray(joint_positions, dtype=np.float64)[list(labels).index("root")]
    low = float(np.asarray(vertices, dtype=np.float64)[:, 2].min())
    return np.array([-hips[0], -hips[1], -low])


def rebind(
    model,
    rest_vertices: np.ndarray,
    target_offsets: np.ndarray,
    rest_bone_poses: np.ndarray,
    file_weights: np.ndarray,
    file_indices: np.ndarray,
    method: str = "forward",
    keep_leg_spread: bool = False,
    max_influences: int = 4,
) -> Rebind:
    """
    The bind mesh and targets of a VRM file.

    Both methods turn each vertex by a blend of the bone transforms X_j = T_j B_j^-1:

    - ``"forward"`` skins Anny's rest mesh into the T-pose with Anny's full skin weights:
      v'_i = sum_j w_ij X_j v_i, and delta'_i = A_i delta_i with A_i = sum_j w_ij Q_j R_j^T
      (Q_j and R_j the rotations of T_j and B_j). The bind mesh is exactly Anny's own forward
      pass with ``pose_parameterization="world"`` and the pose parameters T, and every target
      added to it is exactly that pass for the target's row.
    - ``"inverse"`` solves for the bind mesh that the file's own 4 weights skin back onto
      Anny's rest mesh: with N_i = sum_j w4_ij X_j^-1, v'_i = N_i^-1 v_i and delta'_i =
      L_i^-1 delta_i, with L_i the linear part of N_i. Posed back to Anny's rest pose, the
      file then shows Anny's rest mesh and targets exactly.

    The normals of the forward bind are those of the welded bind mesh. Those of the inverse
    bind are Anny's rest normals n_i (area-weighted, of the welded rest mesh) carried by the
    map of the targets, n'_i = L_i^-1 n_i, normalised. Engines skin a normal with the same
    blended matrix as its vertex and normalise it, so the file posed at Anny's rest pose
    shows Anny's rest normals exactly, and the arms-down poses stay close to them. Skinned
    that way, the normals of the welded bind mesh would land more than 45 degrees off at the
    armpits, the sides of the chest, the clavicles and the fingers, even at Anny's rest pose,
    and the inverse transpose of the map (the geometric normal of the bind mesh, L_i^T n_i)
    up to 5 degrees off.

    Args:
        model: the Anny model (its full skin weights drive the forward bind).
        rest_vertices: (V, 3) rest vertices of row 0 (no facial action, no face-shape target).
        target_offsets: (R, V, 3) rest offsets of the morph targets, relative to row 0.
        rest_bone_poses: (J, 4, 4) B_j of row 0.
        file_weights, file_indices: (V, K) skin weights of the file (the inverse bind uses
            them). Like :func:`~opensculptboy.export.body.build_body`, the bind keeps the
            strongest ``max_influences`` of each vertex, renormalised, so the untruncated
            weights of ``vrm_tables.vrm_skin_weights`` may be passed as they are.
        method: ``"forward"`` (v' = sum_j w_ij T_j B_j^-1 v_i with Anny's weights) or
            ``"inverse"`` (v' = (sum_j w4_ij X_j^-1)^-1 v_i with the file's weights).
        keep_leg_spread: passed to :func:`vrm_t_pose`.
        max_influences: the bones per vertex that the file keeps (4 in a VRM file).

    Returns:
        A :class:`Rebind`; ``joint_positions`` are the heads of T_j, and ``offset`` centres
        the bind mesh (see :func:`centring_offset`).
    """
    from opensculptboy.export.body import (
        top_skin_weights,
        triangulated_faces,
        vertex_normals,
    )

    if method not in ("forward", "inverse"):
        raise ValueError(f"Unknown bind method {method!r}.")
    B = np.asarray(rest_bone_poses, dtype=np.float64)
    vertices = np.asarray(rest_vertices, dtype=np.float64)
    offsets = np.asarray(target_offsets, dtype=np.float64).reshape(-1, *vertices.shape)
    triangles, _ = triangulated_faces(model)
    T = (
        vrm_t_pose(model, torch.from_numpy(B), keep_leg_spread=keep_leg_spread)
        .cpu()
        .numpy()
    )
    X = T @ rigid_inverse(B)
    if method == "forward":
        weights = model.vertex_bone_weights.detach().cpu().double().numpy()
        indices = model.vertex_bone_indices.cpu().numpy()
        M = blend(weights, indices, X)
        linear = M[:, :3, :3]
        bind = (linear @ vertices[..., None])[..., 0] + M[:, :3, 3]
        normals = vertex_normals(bind, triangles)
    else:
        weights, indices, _ = top_skin_weights(
            file_weights, file_indices, max_influences
        )
        N = blend(weights, indices, rigid_inverse(X))
        linear = np.linalg.inv(N[:, :3, :3])
        bind = (linear @ (vertices - N[:, :3, 3])[..., None])[..., 0]
        normals = _unit_rows(
            (linear @ vertex_normals(vertices, triangles)[..., None])[..., 0]
        )
    targets = np.einsum("vab,rvb->rva", linear, offsets)
    joints = T[:, :3, 3].copy()
    return Rebind(
        vertices=bind,
        normals=normals,
        targets=targets,
        bone_poses=T,
        rest_bone_poses=B.copy(),
        joint_positions=joints,
        method=method,
        offset=centring_offset(bind, joints, model.bone_labels),
    )


def file_pose(
    bone_poses: np.ndarray, rebind: Rebind, frame: np.ndarray | None = None
) -> np.ndarray:
    """
    World matrices of the joints of a VRM file (normalised joints, centred) that pose it like
    Anny's world bone poses M_j (..., J, 4, 4).

    With G the rotation from Anny's frame to the file's, o = G ``rebind.offset`` and
    P_j = G t_j + o the joint of bone j in the file, the joint matrix is
    W_j = translate(o) G M_j T_j^-1 G^-1 translate(P_j - o). Each inverse bind matrix of the
    file being translate(-P_j), the file then skins a vertex G v' + o of the bind mesh to
    G sum_j w_ij M_j T_j^-1 v' + o. ``frame`` defaults to glTF's (VRM 1.0).
    """
    if frame is None:
        from opensculptboy.export.body import ANNY_TO_GLTF as frame
    G = np.eye(4)
    G[:3, :3] = np.asarray(frame, dtype=np.float64)
    o = np.eye(4)
    o[:3, 3] = G[:3, :3] @ rebind.offset
    joints = np.tile(np.eye(4), (len(rebind.joint_positions), 1, 1))
    joints[:, :3, 3] = rebind.joint_positions @ G[:3, :3].T
    M = np.asarray(bone_poses, dtype=np.float64)
    return o @ G @ M @ rigid_inverse(rebind.bone_poses) @ G.T @ joints


def default_twist(version: str) -> str:
    """The twist mode of a VRM version by default: "constraint" for 1.0, "merge" for 0.x."""
    from opensculptboy.export import vrm_tables

    if version not in vrm_tables.VERSIONS:
        raise ValueError(f"Unknown VRM version {version!r}; use '1.0' or '0.x'.")
    return "constraint" if version == "1.0" else "merge"


def _strongest_four(weights: np.ndarray, indices: np.ndarray):
    """The strongest 4 bones of each vertex, renormalised in float64, as the VRM writer has."""
    from opensculptboy.export.body import top_skin_weights

    weights, indices, _ = top_skin_weights(weights, indices, 4)
    weights = weights.astype(np.float64)
    return weights / weights.sum(axis=1, keepdims=True), indices


def file_skin_weights(
    model, version: str = "1.0", twist: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    The (V, 4) skin weights (float64, rows summing to 1) and bone indices of a VRM file: those
    of ``vrm_tables.vrm_skin_weights`` for ``twist`` (by default :func:`default_twist` of
    ``version``), truncated to the strongest 4 bones and renormalised, as the VRM writer does.
    """
    from opensculptboy.export import vrm_tables

    twist = twist or default_twist(version)
    return _strongest_four(*vrm_tables.vrm_skin_weights(model, twist))


def skin_normals(
    normals: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    """
    Normals (V, 3) skinned as engines skin them: the linear part of the blended matrix
    sum_k w_ik M[j_ik] applied to n_i, then normalised. ``transforms`` are bone transforms
    (J, 4, 4) or (F, J, 4, 4); returns (V, 3) or (F, V, 3).
    """
    normals = np.asarray(normals, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    indices = np.asarray(indices)
    linear = np.zeros(transforms.shape[:-3] + (len(normals), 3, 3))
    for k in range(weights.shape[1]):
        linear += weights[:, k, None, None] * transforms[..., indices[:, k], :3, :3]
    return _unit_rows((linear @ normals[..., None])[..., 0])


def angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The angles between the directions of two arrays (..., 3), in degrees."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.degrees(
        np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))
    )


def roll_constraint(rotation: np.ndarray, axis, weight: float) -> np.ndarray:
    """
    The local rotation that a roll constraint (``VRMC_node_constraint``) gives its node, for
    the local rotation (3, 3) of its source, with the rest rotations of both at the identity
    (the normalised nodes of a VRM 1.0 file): the turn of the source about the unit ``axis``
    left after the smallest turn that brings ``axis`` back onto itself, taken at ``weight``
    (a slerp from the identity).
    """
    axis = _unit(np.asarray(axis, dtype=np.float64))
    rotation = np.asarray(rotation, dtype=np.float64)
    twist = _rotation_between(axis, rotation @ axis).T @ rotation
    skew = twist - twist.T
    sin = axis @ np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) / 2.0
    cos = (np.trace(twist) - 1.0) / 2.0
    return _axis_angle(axis, weight * float(np.arctan2(sin, cos)))


def _body_kwargs(phenotype_kwargs, local_changes_kwargs, face_shape_kwargs) -> dict:
    kwargs = dict(
        phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs
    )
    if face_shape_kwargs:
        kwargs["face_shape_kwargs"] = face_shape_kwargs
    return kwargs


def _rest_body(model, kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    """The rest vertices (V, 3) and rest bone poses (J, 4, 4) of a body, float64."""
    with torch.no_grad():
        rest = model(pose_parameterization="local-ref", **kwargs)
    return (
        rest["rest_vertices"][0].double().cpu().numpy(),
        rest["rest_bone_poses"][0].double().cpu().numpy(),
    )


def _posed_body(
    model, pose, kwargs: dict, frame_step: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Anny's posed vertices (F, V, 3) and world bone poses (F, J, 4, 4) for ``pose``: the name of
    an ``anny.poses`` entry (``local-ref``, without grounding), every ``frame_step``-th frame of
    a clip, or world bone poses (F, J, 4, 4) of the body.
    """
    import anny.poses

    if isinstance(pose, str):
        params = anny.poses.pose_parameters(
            model,
            pose,
            phenotype_kwargs=kwargs.get("phenotype_kwargs"),
            local_changes_kwargs=kwargs.get("local_changes_kwargs"),
            grounded=False,
        )["pose_parameters"][::frame_step]
        parameterization = "local-ref"
    else:
        params = torch.as_tensor(
            np.asarray(pose).reshape(-1, *np.shape(pose)[-3:]),
            dtype=model.template_vertices.dtype,
        )
        parameterization = "world"
    with torch.no_grad():
        posed = model(
            pose_parameters=params, pose_parameterization=parameterization, **kwargs
        )
    return (
        posed["vertices"].double().cpu().numpy(),
        posed["bone_poses"].double().cpu().numpy(),
    )


def _distance_report(moved: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(moved - reference, axis=-1) * 1000.0
    return dict(
        max_mm=float(distance.max()),
        p99_mm=float(np.percentile(distance, 99.0)),
        mean_mm=float(distance.mean()),
    )


def bind_error(
    model,
    method: str = "forward",
    pose_names=("relaxed", "walk"),
    phenotype_kwargs=None,
    local_changes_kwargs=None,
    face_shape_kwargs=None,
    file_weights: np.ndarray | None = None,
    file_indices: np.ndarray | None = None,
    keep_leg_spread: bool = False,
    frame_step: int = 1,
    version: str = "1.0",
    twist: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    How far a VRM bind mesh, posed by the file's skin, lands from Anny's own posed mesh.

    For each pose or clip of ``anny.poses`` (every ``frame_step``-th frame of a clip), the bind
    mesh of ``method`` is skinned with the file's 4 weights through W_j T_j^-1, where W_j are
    Anny's posed world bone poses (``local-ref``, without grounding), and compared with Anny's
    posed vertices (all of Anny's weights, from the rest pose). The normals of the file,
    skinned as engines skin them (:func:`skin_normals`), are compared with the area-weighted
    normals of Anny's posed mesh.

    Args:
        model: the Anny model.
        method: the bind method of :func:`rebind`.
        pose_names: names of ``anny.poses`` entries.
        phenotype_kwargs, local_changes_kwargs, face_shape_kwargs: the body.
        file_weights, file_indices: (V, K) weights of the file, of which the strongest 4 of
            each vertex count; by default those of a VRM file of ``version`` and ``twist``
            (:func:`file_skin_weights`: the eyelids moved to the head, the twist bones merged).
        keep_leg_spread: passed to :func:`vrm_t_pose`.
        frame_step: the step between the frames of a clip that are compared.
        version, twist: the VRM version and twist mode of the default file weights.

    Returns:
        ``{name: {"max_mm": ..., "p99_mm": ..., "mean_mm": ..., "normal_p99_deg": ...,
        "normal_mean_deg": ..., "frames": ...}}``: the vertex distances in millimetres and the
        normal angles in degrees, over all compared frames.
    """
    from opensculptboy.export.body import triangulated_faces, vertex_normals

    kwargs = _body_kwargs(phenotype_kwargs, local_changes_kwargs, face_shape_kwargs)
    vertices, B = _rest_body(model, kwargs)
    if file_weights is None or file_indices is None:
        weights, indices = file_skin_weights(model, version, twist)
    else:
        weights, indices = _strongest_four(file_weights, file_indices)
    rb = rebind(
        model,
        vertices,
        np.zeros((0, *vertices.shape)),
        B,
        weights,
        indices,
        method=method,
        keep_leg_spread=keep_leg_spread,
    )
    triangles, _ = triangulated_faces(model)
    inverse_bind = rigid_inverse(rb.bone_poses)
    report = {}
    for name in pose_names:
        reference, W = _posed_body(model, name, kwargs, frame_step)
        transforms = W @ inverse_bind
        moved = skin(rb.vertices, weights, indices, transforms)
        normals = skin_normals(rb.normals, weights, indices, transforms)
        normal_angles = np.stack(
            [
                angles(n, vertex_normals(v, triangles))
                for n, v in zip(normals, reference)
            ]
        )
        report[name] = dict(
            **_distance_report(moved, reference),
            normal_p99_deg=float(np.percentile(normal_angles, 99.0)),
            normal_mean_deg=float(normal_angles.mean()),
            frames=int(len(reference)),
        )
    return report


def humanoid_error(
    model,
    version: str = "1.0",
    twist: str | None = None,
    method: str = "inverse",
    pose_names=("relaxed", "walk"),
    phenotype_kwargs=None,
    local_changes_kwargs=None,
    face_shape_kwargs=None,
    keep_leg_spread: bool = False,
    frame_step: int = 1,
    poses=None,
    constraints=None,
    region: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    How far a VRM file lands from Anny's posed mesh when a VRM app turns its humanoid bones
    alone.

    The file (the bind of ``method``, the weights of :func:`file_skin_weights` and the node
    hierarchy of ``vrm_tables.file_parents``, every node at rest at the identity rotation) is
    posed in three ways for each frame of each ``anny.poses`` entry of ``pose_names`` and of
    each entry of ``poses``, with W_j Anny's posed world bone poses:

    - ``"every_node"``: every node turns as Anny's bone does, through W_j T_j^-1 (the error
      of :func:`bind_error`);
    - ``"humanoid"``: the humanoid bones of ``version`` take their local rotations in the
      file's hierarchy from Anny's pose, the hips keep Anny's posed position, every other node
      keeps its rest local rotation, and the twist bones follow the roll constraints of
      ``constraints`` (:func:`roll_constraint`). The non-humanoid bones between two humanoid
      bones (spine05, spine03, neck02 and neck03, shoulder01, the pelvis bones and the
      metacarpals) then stay at rest, and so do the leaves (the toes, and the twist bones
      without constraints, whose weights the file merges);
    - ``"humanoid_folded"``: as ``"humanoid"``, but each humanoid bone takes its rotation
      relative to its nearest humanoid ancestor, so that the humanoid bones keep Anny's world
      rotations, as an app that retargets a motion onto the humanoid bones does: the turn of
      each non-humanoid bone between two humanoid bones folds into the humanoid bone below
      it. (Folding it into the humanoid bone above would turn the other branches of that
      bone: spine05 hangs from the hips beside the legs.)

    Args:
        pose_names: names of ``anny.poses`` entries (``local-ref``, without grounding).
        poses: other poses, ``{name: W}`` with W Anny's world bone poses (F, J, 4, 4) of the
            body (``pose_parameterization="world"``), such as a library pose with a turn added.
        constraints: the roll constraints that the app applies, as
            ``vrm_tables.twist_constraints`` gives them (constrained bone, source bone, roll
            axis of the VRM 1.0 file, weight): by default those of the file, and ``[]`` for an
            app that ignores ``VRMC_node_constraint``.
        region: a (V,) boolean mask of the vertices whose distances count; all by default.
        The other arguments are those of :func:`bind_error`.

    Returns:
        ``{name: {variant: {"max_mm": ..., "p99_mm": ..., "mean_mm": ..., "frames": ...}}}``,
        the vertex distances over all compared frames, in millimetres.
    """
    from opensculptboy.export import vrm_tables
    from opensculptboy.export.body import ANNY_TO_GLTF

    twist = twist or default_twist(version)
    kwargs = _body_kwargs(phenotype_kwargs, local_changes_kwargs, face_shape_kwargs)
    vertices, B = _rest_body(model, kwargs)
    weights, indices = file_skin_weights(model, version, twist)
    rb = rebind(
        model,
        vertices,
        np.zeros((0, *vertices.shape)),
        B,
        weights,
        indices,
        method=method,
        keep_leg_spread=keep_leg_spread,
    )
    labels = list(model.bone_labels)
    parents = [int(p) for p in vrm_tables.file_parents(model, version, twist)]
    order = _parents_first(parents)
    humanoid = sorted(
        labels.index(b) for b in vrm_tables.humanoid_bones(version).values()
    )
    ancestors = {}
    for j in humanoid:
        a = parents[j]
        while a >= 0 and a not in humanoid:
            a = parents[a]
        ancestors[j] = a
    references = {
        "humanoid": {j: parents[j] for j in humanoid},
        "humanoid_folded": ancestors,
    }
    if constraints is None:
        constraints = vrm_tables.twist_constraints(version, twist)
    # The roll axes of the file in Anny's frame; the sign of an axis does not change a roll.
    constraints = [
        (
            labels.index(bone),
            labels.index(source),
            ANNY_TO_GLTF.T[:, "XYZ".index(axis)],
            w,
        )
        for bone, source, axis, w in constraints
    ]
    joints = rb.joint_positions
    inverse_bind = rigid_inverse(rb.bone_poses)
    keep = slice(None) if region is None else np.asarray(region, dtype=bool)
    bind, weights, indices = rb.vertices[keep], weights[keep], indices[keep]
    report = {}
    entries = {name: name for name in pose_names}
    entries.update(poses or {})
    for name, pose in entries.items():
        reference, W = _posed_body(model, pose, kwargs, frame_step)
        reference = reference[:, keep]
        transforms = W @ inverse_bind
        posed = {"every_node": transforms}
        for variant, refs in references.items():
            posed[variant] = np.stack(
                [
                    _humanoid_drive(
                        t[:, :3, :3],
                        w[:, :3, 3],
                        joints,
                        parents,
                        order,
                        refs,
                        constraints,
                    )
                    for t, w in zip(transforms, W)
                ]
            )
        report[name] = {
            variant: dict(
                **_distance_report(skin(bind, weights, indices, m), reference),
                frames=int(len(reference)),
            )
            for variant, m in posed.items()
        }
    return report


def _humanoid_drive(
    rotations: np.ndarray,
    heads: np.ndarray,
    joints: np.ndarray,
    parents: list[int],
    order: list[int],
    references: dict[int, int],
    constraints,
) -> np.ndarray:
    """
    The skin matrices (J, 4, 4) of a file whose humanoid bones turn alone (Anny's frame).

    ``rotations`` (J, 3, 3) are the world rotations of the file's nodes in the full pose and
    ``heads`` (J, 3) their posed positions; ``joints`` (J, 3) are the joints of the bind
    pose. Each humanoid bone j (the keys of ``references``) takes the rotation of j relative
    to the node ``references[j]`` (-1: relative to the world); the constraints set the twist
    bones; every other node keeps the identity. The nodes without a parent keep their posed
    position.
    """
    count = len(parents)
    local = np.tile(np.eye(3), (count, 1, 1))
    for j, r in references.items():
        local[j] = rotations[j] if r < 0 else rotations[r].T @ rotations[j]
    for bone, source, axis, weight in constraints:
        local[bone] = roll_constraint(local[source], axis, weight)
    world = np.empty((count, 3, 3))
    position = np.empty((count, 3))
    for j in order:
        p = parents[j]
        if p < 0:
            world[j], position[j] = local[j], heads[j]
        else:
            world[j] = world[p] @ local[j]
            position[j] = position[p] + world[p] @ (joints[j] - joints[p])
    matrices = np.tile(np.eye(4), (count, 1, 1))
    matrices[:, :3, :3] = world
    matrices[:, :3, 3] = position - (world @ joints[..., None])[..., 0]
    return matrices
