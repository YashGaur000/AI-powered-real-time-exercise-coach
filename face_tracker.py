"""Real-time face landmarks + blendshapes (MediaPipe Face Landmarker task)."""

from __future__ import annotations

import dataclasses
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import face_landmarker as fl_module
from mediapipe.tasks.python.vision.core import image as mp_image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as vtrm

_MODEL_NAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class FaceLandmarkIdx:
    """Common MediaPipe face mesh indices for guidance overlays."""

    LIP_TOP_INNER = 13
    LIP_BOTTOM_INNER = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 152
    NOSE_TIP = 1
    LEFT_CHEEK = 123
    RIGHT_CHEEK = 352
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374


@dataclasses.dataclass
class FaceResult:
    landmarks: np.ndarray  # (N, 3) normalized x, y, z
    blendshape_scores: Dict[str, float]  # Blendshapes enum name -> score


def ensure_face_model() -> Path:
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / _MODEL_NAME
    if not path.is_file():
        print(f"Downloading face model to {path} …")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def _blendshape_dict(blendshape_categories) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not blendshape_categories:
        return out
    for c in blendshape_categories[0]:
        idx = c.index
        if idx is None:
            continue
        try:
            name = fl_module.Blendshapes(idx).name
        except ValueError:
            name = f"BS_{idx}"
        out[name] = float(c.score if c.score is not None else 0.0)
    return out


class FaceTracker:
    def __init__(
        self,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = str(ensure_face_model())
        opts = fl_module.FaceLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=model_path),
            running_mode=vtrm.VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._face = fl_module.FaceLandmarker.create_from_options(opts)
        self._fl_connections = fl_module.FaceLandmarksConnections

    def close(self) -> None:
        self._face.close()

    def process_bgr(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[FaceResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp_image_module.Image(mp_image_module.ImageFormat.SRGB, rgb)
        result = self._face.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        lm_list = result.face_landmarks[0]
        arr = np.array(
            [[float(p.x or 0), float(p.y or 0), float(p.z or 0)] for p in lm_list],
            dtype=np.float64,
        )
        bs = _blendshape_dict(result.face_blendshapes)
        return FaceResult(landmarks=arr, blendshape_scores=bs)

    def draw_face_mesh(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        region: str = "all",
    ) -> None:
        h, w = frame_bgr.shape[:2]

        def pt(i: int) -> tuple[int, int]:
            return int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)

        def draw_conns(conns: List, color: tuple[int, int, int], t: int = 1) -> None:
            for c in conns:
                a, b = c.start, c.end
                if a < len(landmarks) and b < len(landmarks):
                    cv2.line(frame_bgr, pt(a), pt(b), color, t, cv2.LINE_AA)

        fc = self._fl_connections
        if region in ("all", "lips"):
            draw_conns(fc.FACE_LANDMARKS_LIPS, (180, 200, 255), 1)
        if region in ("all", "eyes"):
            draw_conns(fc.FACE_LANDMARKS_LEFT_EYE, (120, 220, 255), 1)
            draw_conns(fc.FACE_LANDMARKS_RIGHT_EYE, (120, 220, 255), 1)
        if region in ("all", "brows"):
            draw_conns(fc.FACE_LANDMARKS_LEFT_EYEBROW, (140, 180, 200), 1)
            draw_conns(fc.FACE_LANDMARKS_RIGHT_EYEBROW, (140, 180, 200), 1)
        if region == "all":
            draw_conns(fc.FACE_LANDMARKS_FACE_OVAL, (80, 140, 100), 1)


def face_width_norm(landmarks: np.ndarray) -> float:
    return float(np.clip(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]), 1e-6, 1.0))


def lip_width_norm(landmarks: np.ndarray) -> float:
    fw = face_width_norm(landmarks)
    a = landmarks[FaceLandmarkIdx.MOUTH_LEFT]
    b = landmarks[FaceLandmarkIdx.MOUTH_RIGHT]
    return float(np.linalg.norm(a[:2] - b[:2]) / fw)


def mouth_opening_norm(landmarks: np.ndarray) -> float:
    fw = face_width_norm(landmarks)
    u = landmarks[FaceLandmarkIdx.LIP_TOP_INNER]
    lo = landmarks[FaceLandmarkIdx.LIP_BOTTOM_INNER]
    return float(abs(lo[1] - u[1]) / fw)


def eye_openness_norm(landmarks: np.ndarray) -> float:
    fw = face_width_norm(landmarks)
    le = abs(landmarks[FaceLandmarkIdx.LEFT_EYE_TOP, 1] - landmarks[FaceLandmarkIdx.LEFT_EYE_BOTTOM, 1])
    re = abs(landmarks[FaceLandmarkIdx.RIGHT_EYE_TOP, 1] - landmarks[FaceLandmarkIdx.RIGHT_EYE_BOTTOM, 1])
    return float(((le + re) * 0.5) / fw)
