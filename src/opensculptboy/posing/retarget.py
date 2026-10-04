# OpenSculptBoy
# Apache License, Version 2.0
"""
From landmarks to a pose: the world rotation of each of Anny's bones.

The torso takes its frame from the hips and the shoulders, the head from the ears and the
nose, and each limb aims along its landmarks with a hinge at the elbow or the knee
(:meth:`opensculptboy.posing.skeleton.Skeleton.hinge`). The hands take their frames from the wrist
and the knuckles, and each finger joint bends about the hand's width. The face's blend shape
scores become Anny's facial actions, which carry the same ARKit names.

:class:`AnnyLandmarks` places the same landmarks on Anny's mesh. The retarget reads its rest
frames from them, and they turn any pose of the model into the landmarks MediaPipe would give,
which the round-trip test uses.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import roma
import torch

from opensculptboy.posing.head import (
    HEAD_POINTS,
    face_mesh_rest,
    plausible,
    robust_fit,
    robust_fit_image,
)
from opensculptboy.posing.landmarks import BODY, HAND, Landmarks
from opensculptboy.posing.skeleton import (
    Rotations,
    Skeleton,
    axis_angle,
    bend,
    frame,
    rot,
    slerp,
)

SIDES = {".L": "left", ".R": "right"}
FINGERS = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}
KNUCKLES = ("mcp", "pip", "dip", "tip")
THUMB = ("cmc", "mcp", "ip", "tip")
# the spine's share of the turn from the pelvis to the chest, from the hips up, and the neck's
# share of the turn from the chest to the head
SPINE = [
    ("spine05", 0.15),
    ("spine04", 0.35),
    ("spine03", 0.55),
    ("spine02", 0.75),
    ("spine01", 1.0),
]
NECK = [("neck01", 0.3), ("neck02", 0.55), ("neck03", 0.75)]
# the 3D head points lead only when this many of the 11 agree
STRONG_AGREEMENT = 7
# the head turns at most this far from the chest (degrees): a larger turn comes from a bad fit
HEAD_LIMIT = 100.0
# a hand's fingers, from the knuckle to the tip, against the width of the palm across the
# knuckles, and the palm's length against its width: outside these, a hand is not believed
FINGER_RANGE = (0.8, 4.0)
PALM_RANGE = (0.8, 3.5)


# the neck's shares of a head adjustment, from the base of the neck to the head
HEAD_SHARES = [("neck01", 0.3), ("neck02", 0.55), ("neck03", 0.75), ("head", 1.0)]


def adjust_head(
    W: Rotations, turn: float = 0.0, up: float = 0.0, tilt: float = 0.0
) -> Rotations:
    """``W`` with the head turned further in its own frame, in degrees: ``turn`` toward the
    figure's right (negative: its left), ``up`` to raise the face (negative: lower it) and
    ``tilt`` to lean the head toward the right shoulder (negative: the left). The neck carries
    a growing share of the turn up to the head, as a neck does.

    A picture made by an AI model can leave the head's turn unclear to MediaPipe (see
    :meth:`Retargeter.head_fit`); these three angles let a person correct it.
    """
    A = rot("z", -turn) @ bend(up) @ rot("y", -tilt)
    v = roma.rotmat_to_rotvec(A)
    W = dict(W)
    for bone, share in HEAD_SHARES:
        W[bone] = W[bone] @ roma.rotvec_to_rotmat(v * share).to(W[bone].dtype)
    return W


@dataclasses.dataclass
class HeadChoice:
    """the head's world rotation (None: the head follows the chest), its source, and the
    trust (1 or 0) in each of the 11 head points in 3D and in the picture"""

    rotation: np.ndarray | None
    source: str
    in_3d: np.ndarray
    in_image: np.ndarray


def plausible_hand(h: np.ndarray) -> bool:
    """whether 21 hand points (:data:`HAND`) have a hand's proportions"""
    h = np.asarray(h, np.float64)
    at = lambda n: h[HAND.index(n)]  # noqa: E731
    width = np.linalg.norm(at("index_mcp") - at("pinky_mcp"))
    if not np.isfinite(h).all() or width < 1e-9:
        return False
    length = np.linalg.norm(at("middle_mcp") - at("wrist")) / width
    if not PALM_RANGE[0] <= length <= PALM_RANGE[1]:
        return False
    for finger in FINGERS.values():
        names = [f"{finger}_{n}" for n in (THUMB if finger == "thumb" else KNUCKLES)]
        chain = sum(
            np.linalg.norm(at(b) - at(a)) for a, b in zip(names[:-1], names[1:])
        )
        if not FINGER_RANGE[0] <= chain / width <= FINGER_RANGE[1]:
            return False
    return True


class AnnyLandmarks:
    """MediaPipe's landmarks on Anny's mesh: each is a joint or a vertex of the rest pose."""

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        model = skeleton.model
        rest = skeleton.output({})
        v = rest["vertices"][0].float()
        labels = skeleton.labels
        top = model.vertex_bone_indices[
            torch.arange(len(v)), model.vertex_bone_weights.argmax(dim=1)
        ]
        bone_of = np.array([labels[int(i)] for i in top])
        joint = skeleton.joint

        def nearest(point, among=None):
            d = (v - torch.as_tensor(point, dtype=v.dtype)).norm(dim=1)
            if among is not None:
                d = torch.where(torch.as_tensor(among), d, torch.full_like(d, 1e9))
            return ("vertex", int(d.argmin()))

        def extreme(mask, key):
            idx = np.flatnonzero(mask)
            return ("vertex", int(idx[np.argmax(key(v[idx].numpy()))]))

        head = bone_of == "head"
        sources = {}
        nose = extreme(head & (np.abs(v[:, 0].numpy()) < 0.01), lambda p: -p[:, 1])
        sources["nose"] = nose
        n = v[nose[1]]
        for s, side in SIDES.items():
            sign = 1.0 if s == ".L" else -1.0
            eye = joint("eye" + s)
            sources[f"{side}_eye"] = ("joint", "eye" + s)
            sources[f"{side}_eye_inner"] = nearest(
                eye + torch.tensor([-sign * 0.016, -0.01, 0.0])
            )
            sources[f"{side}_eye_outer"] = nearest(
                eye + torch.tensor([sign * 0.016, -0.005, 0.0])
            )
            level = head & (np.abs(v[:, 2].numpy() - float(eye[2]) + 0.02) < 0.02)
            level &= v[:, 1].numpy() > float(eye[1]) + 0.04
            sources[f"{side}_ear"] = extreme(level, lambda p, k=sign: k * p[:, 0])
            sources[f"mouth_{side}"] = nearest(
                torch.stack([torch.tensor(sign * 0.024), n[1] + 0.012, n[2] - 0.045])
            )
            for name, bone in (
                ("shoulder", "upperarm01"),
                ("elbow", "lowerarm01"),
                ("wrist", "wrist"),
                ("pinky", "finger5-1"),
                ("index", "finger2-1"),
                ("thumb", "finger1-3"),
                ("hip", "upperleg01"),
                ("knee", "lowerleg01"),
                ("ankle", "foot"),
            ):
                sources[f"{side}_{name}"] = ("joint", bone + s)
            foot = np.isin(
                bone_of,
                ["foot" + s]
                + [f"toe{k}-{i}" + s for k in range(1, 6) for i in (1, 2, 3)],
            )
            low = v[:, 2].numpy() < float(joint("foot" + s)[2]) - 0.03
            sources[f"{side}_heel"] = extreme(foot & low, lambda p: p[:, 1])
            sources[f"{side}_foot_index"] = extreme(foot, lambda p: -p[:, 1])
        self.body = [sources[name] for name in BODY]
        self.hands = {}
        for s in SIDES:
            hand = {"wrist": ("joint", "wrist" + s)}
            for k, finger in FINGERS.items():
                names = THUMB if k == 1 else KNUCKLES
                for i, knuckle in enumerate(names[:3]):
                    hand[f"{finger}_{knuckle}"] = ("joint", f"finger{k}-{i + 1}" + s)
                last = f"finger{k}-3" + s
                d = joint(last) - joint(f"finger{k}-2" + s)
                hand[f"{finger}_tip"] = extreme(
                    bone_of == last, lambda p, d=d: p @ d.numpy()
                )
            self.hands[s] = [hand[name] for name in HAND]

    def points(self, output: dict, sources) -> np.ndarray:
        v = output["vertices"][0].float()
        bp = output["bone_poses"][0].float()
        labels = self.skeleton.labels
        out = [
            bp[labels.index(ref), :3, 3] if kind == "joint" else v[ref]
            for kind, ref in sources
        ]
        return torch.stack(out).numpy().astype(np.float64)

    def landmarks(self, output: dict, face: dict | None = None) -> Landmarks:
        """the landmarks MediaPipe would give for a posed model's output"""
        return Landmarks(
            body=self.points(output, self.body),
            hands={s: self.points(output, src) for s, src in self.hands.items()},
            face=dict(face or {}),
        )


def _unit(v) -> torch.Tensor:
    v = torch.as_tensor(np.asarray(v, dtype=np.float32))
    return v / v.norm()


class Retargeter:
    """Poses a :class:`Skeleton` from :class:`Landmarks`."""

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.anny = AnnyLandmarks(skeleton)
        self.rest = self.anny.landmarks(skeleton.output({}))
        R = self.rest
        self.rest_pelvis = self._pelvis(R)
        self.rest_chest = self._chest(R)
        self.rest_hands = {s: self._palm(R, s) for s in SIDES}
        self.rest_face = face_mesh_rest(
            skeleton.model, skeleton.output({})["vertices"][0].double().numpy()
        )

    # the frames of the torso and the head, as (main direction, second direction)
    @staticmethod
    def _pelvis(L: Landmarks):
        up = (L.point("left_shoulder") + L.point("right_shoulder")) / 2 - (
            L.point("left_hip") + L.point("right_hip")
        ) / 2
        return frame(_unit(L.point("left_hip") - L.point("right_hip")), _unit(up))

    @staticmethod
    def _chest(L: Landmarks):
        up = (L.point("left_shoulder") + L.point("right_shoulder")) / 2 - (
            L.point("left_hip") + L.point("right_hip")
        ) / 2
        return frame(
            _unit(L.point("left_shoulder") - L.point("right_shoulder")), _unit(up)
        )

    def head_fit(self, L: Landmarks, chest=None) -> HeadChoice:
        """the head's world rotation and the head points to trust, from the first source that
        gives a believable head (:mod:`opensculptboy.posing.head`):

        1. "face": the face landmarker's mesh, when it found a face;
        2. "points": the pose landmarker's 11 head points in 3D, when most of them agree;
        3. "picture": the same points' positions in the picture alone, whose depth MediaPipe
           only guesses (on a drawing, the guess can put the hidden ear on the cheek);
        4. "neck": no believable fit; the head follows the chest.
        """
        if chest is None:
            chest = self._chest(L) @ self.rest_chest.T
        chest = np.asarray(chest, np.float64)
        n = HEAD_POINTS
        vis = np.ones(n) if L.visibility is None else np.asarray(L.visibility[:n])
        points = robust_fit(self.rest.body[:n], L.body[:n], vis)
        if points is not None and not plausible(points, self._body_scale(L), chest):
            points = None
        strong = points is not None and points.inliers.sum() >= STRONG_AGREEMENT
        picture = None
        if not strong and L.image is not None:
            image = np.stack([L.image[:n, 0], -L.image[:n, 1]], axis=1)
            picture = robust_fit_image(
                self.rest.body[:n], image, vis, chest, self._pixel_scale(L)
            )
            if picture is not None and not plausible(
                picture, self._pixel_scale(L), chest
            ):
                picture = None
        in_3d = points.inliers if strong else np.zeros(n)
        in_image = (
            picture.inliers if picture is not None else in_3d if strong else np.zeros(n)
        )
        if L.face_points is not None and self.rest_face is not None:
            # the face mesh is in pixels: its scale is the body's in pixels, from the picture
            face = robust_fit(self.rest_face, L.face_points)
            if face is not None and plausible(face, self._pixel_scale(L), chest):
                return HeadChoice(face.rotation, "face", in_3d, in_image)
        if strong:
            return HeadChoice(points.rotation, "points", in_3d, in_image)
        if picture is not None:
            return HeadChoice(picture.rotation, "picture", in_3d, in_image)
        return HeadChoice(None, "neck", in_3d, in_image)

    def _body_scale(self, L: Landmarks) -> float:
        """the body's size in the landmarks against Anny's: the shoulders' width"""
        width = lambda M: np.linalg.norm(  # noqa: E731
            M.point("left_shoulder") - M.point("right_shoulder")
        )
        return float(width(L) / width(self.rest))

    def _pixel_scale(self, L: Landmarks) -> float:
        """pixels per metre of Anny's body: the pixel length of the torso against Anny's
        (the longest side of the shoulders and hips, which a turn of the body shortens
        least)"""
        if L.image is None:
            return float("nan")
        i = lambda n: L.image[BODY.index(n)]  # noqa: E731
        r = lambda n: self.rest.point(n)[[0, 2]]  # noqa: E731
        pairs = [
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_shoulder", "right_shoulder"),
        ]
        return float(
            max(
                np.linalg.norm(i(a) - i(b)) / np.linalg.norm(r(a) - r(b))
                for a, b in pairs
            )
        )

    @staticmethod
    def _palm(L: Landmarks, s: str):
        """the hand's frame: along the middle knuckle, then toward the index side"""
        if s in L.hands and plausible_hand(L.hands[s]):
            h = lambda n: L.hand(s, n)  # noqa: E731
            return frame(
                _unit(h("middle_mcp") - h("wrist")),
                _unit(h("index_mcp") - h("pinky_mcp")),
            )
        side = SIDES[s]
        p = lambda n: L.point(f"{side}_{n}")  # noqa: E731
        knuckles = (p("index") + p("pinky")) / 2
        return frame(_unit(knuckles - p("wrist")), _unit(p("index") - p("pinky")))

    def __call__(self, L: Landmarks) -> tuple[Rotations, dict[str, float]]:
        """the world rotation of each posed bone, and the facial actions"""
        sk = self.skeleton
        W: Rotations = {}
        pelvis = self._pelvis(L) @ self.rest_pelvis.T
        chest = self._chest(L) @ self.rest_chest.T
        rotation = self.head_fit(L, chest).rotation
        head = (
            chest
            if rotation is None
            else torch.as_tensor(rotation, dtype=torch.float32)
        )
        # a turn from the chest beyond HEAD_LIMIT is cut back to it
        turn = float(torch.rad2deg(roma.rotmat_to_rotvec(chest.T @ head).norm()))
        if turn > HEAD_LIMIT:
            head = slerp(chest, head, HEAD_LIMIT / turn)
        W["root"] = pelvis
        for bone, share in SPINE:
            W[bone] = slerp(pelvis, chest, share)
        for bone, share in NECK:
            W[bone] = slerp(chest, head, share)
        W["head"] = head
        for s, side in SIDES.items():
            p = lambda n, side=side: L.point(f"{side}_{n}")  # noqa: E731
            # the shoulder rises with the arm: a third of the arm's lift above 60 degrees
            # (from hanging), two thirds of it at the collarbone and one third at shoulder01
            arm = _unit(p("elbow") - p("shoulder"))
            down = -(chest @ torch.tensor([0.0, 0.0, 1.0]))
            lift = torch.rad2deg(
                torch.arccos(torch.clamp(torch.dot(arm, down), -1.0, 1.0))
            )
            rise = max(0.0, float(lift) - 60.0) / 3.0
            forward = chest @ torch.tensor([0.0, 1.0, 0.0])
            sign = -1.0 if s == ".L" else 1.0
            W["clavicle" + s] = axis_angle(forward, sign * rise * 2 / 3) @ chest
            W["shoulder01" + s] = axis_angle(forward, sign * rise) @ chest
            # the arm: the elbow's axis at rest is the one the rest pose bends about
            u0 = sk.rest_direction("upperarm01" + s, "lowerarm01" + s)
            f0 = sk.rest_direction("lowerarm01" + s, "wrist" + s)
            elbow = torch.linalg.cross(u0, f0)
            sk.hinge(
                W,
                ("upperarm01" + s, "lowerarm01" + s, "wrist" + s),
                p("elbow") - p("shoulder"),
                p("wrist") - p("elbow"),
                elbow / elbow.norm(),
            )
            # the leg bends at the knee about the body's width
            axis, rest_axis = sk.hinge(
                W,
                ("upperleg01" + s, "lowerleg01" + s, "foot" + s),
                p("knee") - p("hip"),
                p("ankle") - p("knee"),
                torch.tensor([1.0, 0.0, 0.0]),
            )
            rest_foot = _unit(
                self.rest.point(f"{side}_foot_index") - self.rest.point(f"{side}_ankle")
            )
            foot = _unit(p("foot_index") - p("ankle"))
            W["foot" + s] = frame(foot, axis) @ frame(rest_foot, rest_axis).T
            self._hand(W, L, s)
        face = (
            {n: float(L.face[n]) for n in sk.model.facial_action_labels if n in L.face}
            if hasattr(sk.model, "facial_action_labels")
            else dict(L.face)
        )
        return W, face

    def _hand(self, W: Rotations, L: Landmarks, s: str) -> None:
        R = self._palm(L, s) @ self.rest_hands[s].T
        W["wrist" + s] = R
        for k in (2, 3, 4, 5):
            W[f"metacarpal{k - 1}" + s] = R
        if s not in L.hands or not plausible_hand(L.hands[s]):
            return
        h = lambda n: L.hand(s, n)  # noqa: E731
        rest = lambda n: self.rest.hand(s, n)  # noqa: E731
        # the fingers bend about the width of the hand, across the knuckles
        rest_width = self.rest_hands[s][:, 1]
        width = R @ rest_width
        for k, finger in FINGERS.items():
            names = [f"{finger}_{n}" for n in (THUMB if k == 1 else KNUCKLES)]
            for i in (1, 2, 3):
                target = _unit(h(names[i]) - h(names[i - 1]))
                at_rest = _unit(rest(names[i]) - rest(names[i - 1]))
                W[f"finger{k}-{i}" + s] = (
                    frame(target, width) @ frame(at_rest, rest_width).T
                )


def retarget(landmarks: Landmarks, skeleton: Skeleton | None = None):
    """the world rotations and the facial actions for ``landmarks`` (a new default skeleton
    when none is given)"""
    return Retargeter(skeleton or Skeleton())(landmarks)
