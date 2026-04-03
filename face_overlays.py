"""Facial exercise HUD: mesh accents, region highlights, directional arrows."""

from __future__ import annotations

from typing import Set, Tuple

import cv2
import numpy as np

from face_exercises import FaceExerciseAnalysis, FaceExerciseDefinition, FaceExerciseId
from face_tracker import FaceLandmarkIdx, FaceResult
from hud_layout import FONT, bottom_status_band_h, content_right_bound, line_height, wrap_text


def _pt(lm: np.ndarray, idx: int, w: int, h: int) -> Tuple[int, int]:
    return int(lm[idx, 0] * w), int(lm[idx, 1] * h)


def draw_face_feedback_overlay(
    frame_bgr: np.ndarray,
    face: FaceResult,
    analysis: FaceExerciseAnalysis,
    exercise: FaceExerciseDefinition,
    exercise_id: FaceExerciseId,
) -> None:
    h, w = frame_bgr.shape[:2]
    lm = face.landmarks
    max_x = content_right_bound(w)
    y_max = h - bottom_status_band_h() - 8
    x_left = 12
    max_text_w = max_x - x_left

    highlight: Set[int] = set()
    for fb in analysis.feedback:
        highlight.update(fb.highlight_indices)

    for idx in highlight:
        if idx < len(lm):
            x, y = _pt(lm, idx, w, h)
            cv2.circle(frame_bgr, (x, y), 7, (100, 200, 255), 2, cv2.LINE_AA)

    def arrow(x0: int, y0: int, dx: float, dy: float) -> None:
        x1 = int(np.clip(x0 + dx, 0, w - 1))
        y1 = int(np.clip(y0 + dy, 0, h - 1))
        cv2.arrowedLine(frame_bgr, (x0, y0), (x1, y1), (80, 180, 255), 2, tipLength=0.22)

    for fb in analysis.feedback:
        if fb.severity != "warn":
            continue
        if fb.arrow_from_idx is not None and fb.arrow_delta_xy is not None:
            x0, y0 = _pt(lm, fb.arrow_from_idx, w, h)
            dx, dy = fb.arrow_delta_xy
            arrow(x0, y0, dx, dy)
            if exercise_id == FaceExerciseId.SMILE and fb.arrow_from_idx == FaceLandmarkIdx.MOUTH_LEFT:
                xr, yr = _pt(lm, FaceLandmarkIdx.MOUTH_RIGHT, w, h)
                arrow(xr, yr, -dx, dy)
            if exercise_id == FaceExerciseId.CHEEK_LIFT and fb.arrow_from_idx == FaceLandmarkIdx.LEFT_CHEEK:
                xr, yr = _pt(lm, FaceLandmarkIdx.RIGHT_CHEEK, w, h)
                arrow(xr, yr, dx, dy)

    y = 36
    cv2.putText(
        frame_bgr,
        f"Face: {exercise.display_name}",
        (x_left, y),
        FONT,
        0.75,
        (255, 240, 220),
        2,
        cv2.LINE_AA,
    )
    y += line_height(0.75, 2, gap=10)

    cv2.putText(
        frame_bgr,
        f"Reps: {analysis.rep_count}  Hold: {int(analysis.hold_progress * 100)}%",
        (x_left, y),
        FONT,
        0.55,
        (200, 255, 220),
        2,
        cv2.LINE_AA,
    )
    y += line_height(0.55, 2, gap=8)

    bar_w = min(200, max_text_w)
    bx, by = x_left, y
    cv2.rectangle(frame_bgr, (bx, by - 12), (bx + bar_w, by + 4), (50, 50, 55), -1)
    fill = int(bar_w * float(np.clip(analysis.quality_score, 0.0, 1.0)))
    col = (80, 220, 120) if analysis.quality_score > 0.65 else (60, 160, 255)
    cv2.rectangle(frame_bgr, (bx, by - 12), (bx + fill, by + 4), col, -1)

    y += line_height(0.55, 2, gap=12)
    msg_scale = 0.42
    msg_th = 1
    msg_lh = line_height(msg_scale, msg_th, gap=5)

    for fb in analysis.feedback[:5]:
        if y > y_max:
            break
        color = (200, 200, 200)
        if fb.severity == "warn":
            color = (160, 200, 255)
        elif fb.severity == "good":
            color = (180, 255, 180)
        for line in wrap_text(fb.message, max_text_w, msg_scale, msg_th):
            if y > y_max:
                break
            cv2.putText(
                frame_bgr,
                line,
                (x_left, y),
                FONT,
                msg_scale,
                color,
                msg_th,
                cv2.LINE_AA,
            )
            y += msg_lh
