# Corporis
# Apache License, Version 2.0
"""
The landmarks of a picture: MediaPipe's 33 body points, 21 points per hand and the face's 52
ARKit blend shape scores.

Points are held in the model's frame, seen from the front: x to the picture's right (the
figure's left when it faces the camera), y away from the camera, z up, in metres. MediaPipe's
world points (x right, y down, z away from the camera) turn into it by (x, y, z) -> (x, z, -y).
"""

from __future__ import annotations

import dataclasses

import numpy as np

BODY = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]
HAND = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


def from_mediapipe(points) -> np.ndarray:
    """MediaPipe world points (x right, y down, z away) in the model's frame"""
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


@dataclasses.dataclass
class Landmarks:
    """The landmarks of one figure in a picture.

    ``body`` holds the 33 points of :data:`BODY` (m, the model's frame), ``visibility`` their
    MediaPipe visibility (0 to 1) and ``image`` their pixel positions. ``hands`` maps a side
    (".L", ".R", the figure's left and right) to the 21 points of :data:`HAND`. ``face`` holds
    the ARKit blend shape scores, which name Anny's facial actions.
    """

    body: np.ndarray
    visibility: np.ndarray | None = None
    image: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    hands: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    face: dict[str, float] = dataclasses.field(default_factory=dict)

    def point(self, name: str) -> np.ndarray:
        return self.body[BODY.index(name)]

    def hand(self, side: str, name: str) -> np.ndarray:
        return self.hands[side][HAND.index(name)]
