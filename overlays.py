"""Visual guidance: skeleton highlights, arrows, and HUD text."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from exercises import ExerciseAnalysis, ExerciseDefinition
from hud_layout import FONT, bottom_status_band_h, content_right_bound, line_height, wrap_text
from pose_tracker import PoseResult


def _pt(lm: np.ndarray, idx: int, w: int, h: int) -> Tuple[int, int]:
    return int(lm[idx, 0] * w), int(lm[idx, 1] * h)


def draw_feedback_overlay(
    frame_bgr: np.ndarray,
    pose: PoseResult,
    analysis: ExerciseAnalysis,
    exercise: ExerciseDefinition,
) -> None:
    h, w = frame_bgr.shape[:2]
    lm = pose.landmarks
    max_x = content_right_bound(w)
    y_max = h - bottom_status_band_h() - 8

    highlight: List[int] = []
    for fb in analysis.feedback:
        highlight.extend(fb.highlight_landmarks)

    for fb in analysis.feedback:
        if fb.severity != "warn":
            continue
        if fb.arrow_from_idx is not None and fb.arrow_to_pixel_offset is not None:
            x0, y0 = _pt(lm, fb.arrow_from_idx, w, h)
            dx, dy = fb.arrow_to_pixel_offset
            x1 = int(np.clip(x0 + dx, 0, w - 1))
            y1 = int(np.clip(y0 + dy, 0, h - 1))
            cv2.arrowedLine(frame_bgr, (x0, y0), (x1, y1), (0, 200, 255), 3, tipLength=0.25)

    x_left = 12
    max_text_w = max_x - x_left

    y = 36
    cv2.putText(
        frame_bgr,
        exercise.display_name,
        (x_left, y),
        FONT,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += line_height(0.8, 2, gap=10)

    cv2.putText(
        frame_bgr,
        f"Reps: {analysis.rep_count}  Phase: {analysis.rep_phase}",
        (x_left, y),
        FONT,
        0.6,
        (200, 255, 200),
        2,
        cv2.LINE_AA,
    )
    y += line_height(0.6, 2, gap=8)

    bar_w = min(200, max_text_w)
    bx, by = x_left, y
    cv2.rectangle(frame_bgr, (bx, by - 14), (bx + bar_w, by + 4), (60, 60, 60), -1)
    fill = int(bar_w * float(np.clip(analysis.quality_score, 0.0, 1.0)))
    col = (0, 220, 0) if analysis.quality_score > 0.75 else (0, 180, 255)
    cv2.rectangle(frame_bgr, (bx, by - 14), (bx + fill, by + 4), col, -1)

    y += line_height(0.6, 2, gap=12)
    msg_scale = 0.45
    msg_th = 1
    msg_lh = line_height(msg_scale, msg_th, gap=5)

    for fb in analysis.feedback[:4]:
        if y > y_max:
            break
        color = (180, 180, 180)
        if fb.severity == "warn":
            color = (80, 180, 255)
        elif fb.severity == "good":
            color = (120, 255, 120)
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
