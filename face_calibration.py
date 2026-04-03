"""Neutral-face calibration: baselines for blendshapes and geometry (normalized by face width)."""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from mediapipe.tasks.python.vision import face_landmarker as flm

from face_tracker import (
    FaceResult,
    eye_openness_norm,
    face_width_norm,
    lip_width_norm,
    mouth_opening_norm,
)


@dataclasses.dataclass
class FaceCalibrationData:
    face_width_norm: float
    lip_width_norm: float
    mouth_open_norm: float
    eye_open_norm: float
    blendshape_baseline: np.ndarray  # length 52, order matches mediapipe Blendshapes enum

    @staticmethod
    def from_face(face: FaceResult) -> Optional["FaceCalibrationData"]:
        lm = face.landmarks
        if len(lm) < 400:
            return None
        fw = face_width_norm(lm)
        vec = np.zeros(52, dtype=np.float64)
        for i in range(52):
            key = flm.Blendshapes(i).name
            vec[i] = face.blendshape_scores.get(key, 0.0)
        return FaceCalibrationData(
            face_width_norm=fw,
            lip_width_norm=lip_width_norm(lm),
            mouth_open_norm=mouth_opening_norm(lm),
            eye_open_norm=eye_openness_norm(lm),
            blendshape_baseline=vec,
        )


class FaceCalibrationStore:
    def __init__(self) -> None:
        self.data: Optional[FaceCalibrationData] = None

    def capture(self, face: FaceResult) -> bool:
        c = FaceCalibrationData.from_face(face)
        if c is None:
            return False
        self.data = c
        return True

    @property
    def is_ready(self) -> bool:
        return self.data is not None
