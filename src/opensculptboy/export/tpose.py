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
- the bones of the four fingers point along (s, 0, 0) with the nails up; the thumb points
  along (s, -1, 0) / sqrt(2), level and at 45 degrees toward the front; the last bone of each
  finger turns with the bone before it, as Unity's humanoid does for the distal bones;
- the legs turn as one piece about the hip, so that the line from the hip (upperleg01) to the
  ankle (foot) is vertical, unless ``keep_leg_spread``;
- the feet turn about the vertical axis so that they point along -Y (the line from the ankle
  to the middle toe, seen from above), and keep their rest pitch and roll so that the soles
  stay level; the toes turn with the feet.

The arm and finger bones first turn with their parent and then by the smallest change that
aims them at their target, so the joints straighten without twisting; the finger bones make
that change about the normal of the palm and then about the knuckle axis (yaw, then pitch), so
that their nails stay perpendicular to the palm. The hand, the legs and the feet take the turns
above from their rest pose. The positions then follow from forward kinematics with absolute
orientations (``pose_parameterization="world-orient"``): the head of every child bone is
T_p B_p^-1 h, where p is its parent and h its rest head.

The library's ``t_pose`` is not used: it turns the clavicles, and its ``local-ref`` identity is
the reference pose of another body shape.

:func:`rebind` offers two bind meshes. The forward bind is Anny's own skinning into the
T-pose; the inverse bind is the mesh that the file's 4 weights skin back onto Anny's rest pose
exactly. :func:`bind_error` measures both against Anny's posed meshes: on the arms-down poses
(``relaxed``, ``walk``) the inverse bind lands about twice as close.
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


@dataclasses.dataclass(frozen=True)
class _Rule:
    """How one bone turns: ``kind`` is "aim", "fan", "hand", "leg" or "foot"."""

    kind: str
    start: str  # the bone whose head starts the direction
    end: str  # the bone whose head ends the direction
    target: tuple[float, float, float]


def _side_rules(side: str, keep_leg_spread: bool) -> dict[str, _Rule]:
    s = 1.0 if side == "L" else -1.0
    along = (s, 0.0, 0.0)
    thumb = (s / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0)

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
    return turns


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
    """A bind mesh, its morph targets and its skeleton, in Anny's frame."""

    vertices: np.ndarray  # (V, 3) float64: the bind mesh v'
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
    from opensculptboy.export.body import top_skin_weights

    if method not in ("forward", "inverse"):
        raise ValueError(f"Unknown bind method {method!r}.")
    B = np.asarray(rest_bone_poses, dtype=np.float64)
    vertices = np.asarray(rest_vertices, dtype=np.float64)
    offsets = np.asarray(target_offsets, dtype=np.float64).reshape(-1, *vertices.shape)
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
    else:
        weights, indices, _ = top_skin_weights(
            file_weights, file_indices, max_influences
        )
        N = blend(weights, indices, rigid_inverse(X))
        linear = np.linalg.inv(N[:, :3, :3])
        bind = (linear @ (vertices - N[:, :3, 3])[..., None])[..., 0]
    targets = np.einsum("vab,rvb->rva", linear, offsets)
    joints = T[:, :3, 3].copy()
    return Rebind(
        vertices=bind,
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
) -> dict[str, dict[str, float]]:
    """
    How far a VRM bind mesh, posed by the file's skin, lands from Anny's own posed mesh.

    For each pose or clip of ``anny.poses`` (every ``frame_step``-th frame of a clip), the bind
    mesh of ``method`` is skinned with the file's 4 weights through W_j T_j^-1, where W_j are
    Anny's posed world bone poses (``local-ref``, not grounded), and compared with Anny's posed
    vertices (all of Anny's weights, from the rest pose).

    Args:
        model: the Anny model.
        method: the bind method of :func:`rebind`.
        pose_names: names of ``anny.poses`` entries.
        phenotype_kwargs, local_changes_kwargs, face_shape_kwargs: the body.
        file_weights, file_indices: (V, 4) weights of the file; by default Anny's strongest 4
            bones of each vertex, renormalised.
        keep_leg_spread: passed to :func:`vrm_t_pose`.
        frame_step: the step between the frames of a clip that are compared.

    Returns:
        ``{name: {"max_mm": ..., "p99_mm": ..., "mean_mm": ..., "frames": ...}}``, the vertex
        distances over all compared frames, in millimetres.
    """
    import anny.poses

    from opensculptboy.export.body import top_skin_weights

    kwargs = dict(
        phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs
    )
    if face_shape_kwargs:
        kwargs["face_shape_kwargs"] = face_shape_kwargs
    with torch.no_grad():
        rest = model(pose_parameterization="local-ref", **kwargs)
    vertices = rest["rest_vertices"][0].double().cpu().numpy()
    B = rest["rest_bone_poses"][0].double().cpu().numpy()
    if file_weights is None or file_indices is None:
        file_weights, file_indices, _ = top_skin_weights(
            model.vertex_bone_weights.detach().cpu().double().numpy(),
            model.vertex_bone_indices.cpu().numpy(),
            4,
        )
    rb = rebind(
        model,
        vertices,
        np.zeros((0, *vertices.shape)),
        B,
        file_weights,
        file_indices,
        method=method,
        keep_leg_spread=keep_leg_spread,
    )
    inverse_bind = rigid_inverse(rb.bone_poses)
    report = {}
    for name in pose_names:
        params = anny.poses.pose_parameters(
            model,
            name,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
            grounded=False,
        )["pose_parameters"][::frame_step]
        with torch.no_grad():
            posed = model(
                pose_parameters=params, pose_parameterization="local-ref", **kwargs
            )
        reference = posed["vertices"].double().cpu().numpy()
        W = posed["bone_poses"].double().cpu().numpy()
        moved = skin(rb.vertices, file_weights, file_indices, W @ inverse_bind)
        distance = np.linalg.norm(moved - reference, axis=-1) * 1000.0
        report[name] = dict(
            max_mm=float(distance.max()),
            p99_mm=float(np.percentile(distance, 99.0)),
            mean_mm=float(distance.mean()),
            frames=int(len(params)),
        )
    return report
