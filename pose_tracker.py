"""Real-time full-body pose estimation (MediaPipe Pose Landmarker task API)."""

from __future__ import annotations

import dataclasses
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import pose_landmarker as pl_module
from mediapipe.tasks.python.vision.core import image as mp_image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as vtrm

_MODEL_NAME = "pose_landmarker_lite.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# BlazePose full-body topology (33 landmarks)
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]


class LandmarkIdx:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclasses.dataclass
class PoseResult:
    landmarks: np.ndarray
    world_landmarks: Optional[np.ndarray]
    visibility: np.ndarray


def ensure_pose_model() -> Path:
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / _MODEL_NAME
    if not path.is_file():
        print(f"Downloading pose model to {path} …")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


class PoseTracker:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        del static_image_mode, smooth_landmarks, model_complexity
        model_path = str(ensure_pose_model())
        opts = pl_module.PoseLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=model_path),
            running_mode=vtrm.VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = pl_module.PoseLandmarker.create_from_options(opts)

    def close(self) -> None:
        self._landmarker.close()

    def process_bgr(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[PoseResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp_image_module.Image(mp_image_module.ImageFormat.SRGB, rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        lm_list = result.pose_landmarks[0]
        arr = np.array(
            [[float(p.x or 0), float(p.y or 0), float(p.z or 0)] for p in lm_list],
            dtype=np.float64,
        )
        vis = np.array(
            [float(p.visibility if p.visibility is not None else 1.0) for p in lm_list],
            dtype=np.float64,
        )

        world: Optional[np.ndarray] = None
        if result.pose_world_landmarks:
            wl = result.pose_world_landmarks[0]
            world = np.array(
                [[float(p.x or 0), float(p.y or 0), float(p.z or 0)] for p in wl],
                dtype=np.float64,
            )

        return PoseResult(landmarks=arr, world_landmarks=world, visibility=vis)

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        highlight: Optional[List[int]] = None,
    ) -> None:
        h, w = frame_bgr.shape[:2]
        highlight_set = set(highlight or [])

        def pt(i: int) -> Tuple[int, int]:
            return int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)

        for a, b in POSE_CONNECTIONS:
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            color = (0, 255, 0)
            if a in highlight_set or b in highlight_set:
                color = (0, 165, 255)
            cv2.line(frame_bgr, pt(a), pt(b), color, 2, cv2.LINE_AA)

        for i in range(len(landmarks)):
            rad = 5 if i in highlight_set else 3
            col = (0, 165, 255) if i in highlight_set else (255, 255, 0)
            cv2.circle(frame_bgr, pt(i), rad, col, -1, cv2.LINE_AA)


def mid_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def landmark_xyz(
    pose: PoseResult, idx: int, prefer_world: bool = True
) -> np.ndarray:
    if prefer_world and pose.world_landmarks is not None:
        return pose.world_landmarks[idx].copy()
    return pose.landmarks[idx].copy()


def angle_degrees_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return float("nan")
    cos_v = float(np.dot(ba, bc) / (nba * nbc))
    cos_v = max(-1.0, min(1.0, cos_v))
    return float(np.degrees(np.arccos(cos_v)))
