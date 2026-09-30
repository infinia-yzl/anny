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
    face_points: np.ndarray | None = None

    def point(self, name: str) -> np.ndarray:
        return self.body[BODY.index(name)]

    def hand(self, side: str, name: str) -> np.ndarray:
        return self.hands[side][HAND.index(name)]


# MediaPipe's models, downloaded on first use into ANNY_CACHE_DIR/corporis/models
MODELS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task",
}


def model_path(name: str):
    """the cached file of one of MediaPipe's models, downloaded if needed"""
    import requests

    from anny.paths import get_anny_cache_path

    folder = get_anny_cache_path() / "corporis" / "models"
    folder.mkdir(parents=True, exist_ok=True)
    url = MODELS[name]
    path = folder / url.rsplit("/", 1)[1]
    if not path.exists():
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"could not download MediaPipe's {name} model from {url} ({error})"
            ) from error
        tmp = path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for block in response.iter_content(1 << 20):
                f.write(block)
        tmp.rename(path)
    return path


def detect(image, hands: bool = True, face: bool = True) -> Landmarks:
    """the landmarks of the most prominent figure in ``image`` (a path, a PIL image or an RGB
    array), with MediaPipe's pose, hand and face landmarkers"""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image

    if not isinstance(image, np.ndarray):
        image = np.asarray(
            Image.open(image).convert("RGB")
            if not hasattr(image, "convert")
            else image.convert("RGB")
        )
    height, width = image.shape[:2]
    picture = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)
    )

    def options(kind, cls, **kwargs):
        return cls(
            base_options=BaseOptions(model_asset_path=str(model_path(kind))), **kwargs
        )

    with vision.PoseLandmarker.create_from_options(
        options("pose", vision.PoseLandmarkerOptions, num_poses=1)
    ) as landmarker:
        result = landmarker.detect(picture)
    if not result.pose_world_landmarks:
        raise ValueError("MediaPipe found no figure in the picture")
    world = result.pose_world_landmarks[0]
    normalised = result.pose_landmarks[0]
    body = from_mediapipe([[p.x, p.y, p.z] for p in world])
    visibility = np.array([p.visibility for p in normalised], dtype=np.float64)
    pixels = np.array(
        [[p.x * width, p.y * height] for p in normalised], dtype=np.float64
    )
    L = Landmarks(body, visibility, pixels, (width, height))
    if hands:
        with vision.HandLandmarker.create_from_options(
            options("hand", vision.HandLandmarkerOptions, num_hands=2)
        ) as landmarker:
            found = landmarker.detect(picture)
        p = lambda n: pixels[BODY.index(n)]  # noqa: E731
        wrists = {".L": p("left_wrist"), ".R": p("right_wrist")}
        # a hand belongs to a wrist only within most of a forearm's length of it
        reach = {
            ".L": 0.6 * np.linalg.norm(p("left_wrist") - p("left_elbow")),
            ".R": 0.6 * np.linalg.norm(p("right_wrist") - p("right_elbow")),
        }
        taken = set()
        for points, world_points in zip(
            found.hand_landmarks, found.hand_world_landmarks
        ):
            at = np.array([points[0].x * width, points[0].y * height])
            # MediaPipe names a hand's side as seen in a mirror; the nearest wrist is surer
            side = min(
                (s for s in wrists if s not in taken),
                key=lambda s: np.linalg.norm(wrists[s] - at),
                default=None,
            )
            if side is None or np.linalg.norm(wrists[side] - at) > reach[side]:
                continue
            taken.add(side)
            L.hands[side] = from_mediapipe([[p.x, p.y, p.z] for p in world_points])
    if face:
        _detect_face(L, image, options, vision, mp)
    return L


def _detect_face(L: Landmarks, image: np.ndarray, options, vision, mp) -> None:
    """the face landmarker on a crop around the head, which finds a small face far more
    often than the whole picture does, then on the whole picture"""
    height, width = image.shape[:2]
    head = L.image[:11]
    centre = head.mean(0)
    half = max(48.0, 1.6 * float(np.abs(head - centre).max()))
    x0, y0 = (int(max(0, c - half)) for c in centre)
    x1, y1 = int(min(width, centre[0] + half)), int(min(height, centre[1] + half))
    crops = [(x0, y0, image[y0:y1, x0:x1])] if x1 - x0 > 16 and y1 - y0 > 16 else []
    crops.append((0, 0, image))
    with vision.FaceLandmarker.create_from_options(
        options(
            "face",
            vision.FaceLandmarkerOptions,
            output_face_blendshapes=True,
            num_faces=1,
        )
    ) as landmarker:
        for ox, oy, crop in crops:
            h, w = crop.shape[:2]
            found = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)
                )
            )
            if not found.face_landmarks:
                continue
            if found.face_blendshapes:
                L.face = {
                    c.category_name: float(c.score) for c in found.face_blendshapes[0]
                }
            # pixels of the whole picture; the depth is in the same units as x
            mesh = [
                [ox + q.x * w, oy + q.y * h, q.z * w] for q in found.face_landmarks[0]
            ]
            L.face_points = from_mediapipe(mesh[:468])
            return
