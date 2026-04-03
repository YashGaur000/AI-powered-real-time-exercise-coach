"""Joint angles, alignment metrics, and exercise-specific analysis dispatch."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from calibration import CalibrationData
from exercises import (
    ExerciseAnalysis,
    ExerciseId,
    AnalysisContext,
    analyze_arm_raise,
    analyze_lunge,
    analyze_posture,
    analyze_pushup,
    analyze_side_stretch,
    analyze_squat,
)
from pose_tracker import LandmarkIdx, PoseResult, angle_degrees_3d, landmark_xyz, mid_point


def _angle_safe(pose: PoseResult, a: int, b: int, c: int) -> float:
    pa = landmark_xyz(pose, a)
    pb = landmark_xyz(pose, b)
    pc = landmark_xyz(pose, c)
    return angle_degrees_3d(pa, pb, pc)


def compute_angles(pose: PoseResult) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["left_knee"] = _angle_safe(pose, LandmarkIdx.LEFT_HIP, LandmarkIdx.LEFT_KNEE, LandmarkIdx.LEFT_ANKLE)
    out["right_knee"] = _angle_safe(pose, LandmarkIdx.RIGHT_HIP, LandmarkIdx.RIGHT_KNEE, LandmarkIdx.RIGHT_ANKLE)
    out["left_elbow"] = _angle_safe(
        pose, LandmarkIdx.LEFT_SHOULDER, LandmarkIdx.LEFT_ELBOW, LandmarkIdx.LEFT_WRIST
    )
    out["right_elbow"] = _angle_safe(
        pose, LandmarkIdx.RIGHT_SHOULDER, LandmarkIdx.RIGHT_ELBOW, LandmarkIdx.RIGHT_WRIST
    )

    ls = landmark_xyz(pose, LandmarkIdx.LEFT_SHOULDER)
    rs = landmark_xyz(pose, LandmarkIdx.RIGHT_SHOULDER)
    lh = landmark_xyz(pose, LandmarkIdx.LEFT_HIP)
    rh = landmark_xyz(pose, LandmarkIdx.RIGHT_HIP)

    m_sh = mid_point(ls, rs)
    m_hip = mid_point(lh, rh)
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    trunk = m_sh - m_hip
    nt = np.linalg.norm(trunk)
    if nt > 1e-9:
        t = trunk / nt
        dotv = float(np.clip(np.dot(t, vertical), -1.0, 1.0))
        out["trunk_lean_deg"] = float(np.degrees(np.arccos(dotv)))
    else:
        out["trunk_lean_deg"] = float("nan")

    nose = landmark_xyz(pose, LandmarkIdx.NOSE)
    dx = float(nose[0] - m_sh[0])
    dz = float(nose[2] - m_sh[2])
    forward_dist = float(np.sqrt(dx * dx + dz * dz))
    vertical_neck = float(abs(nose[1] - m_sh[1]))
    if vertical_neck + forward_dist > 1e-9:
        out["forward_head_deg"] = float(np.degrees(np.arctan2(forward_dist, vertical_neck + 1e-9)))
    else:
        out["forward_head_deg"] = 0.0

    sh_vec = rs - ls
    out["shoulder_tilt_deg"] = float(
        np.degrees(np.arctan2(abs(sh_vec[1]), max(np.linalg.norm(sh_vec[[0, 2]]), 1e-9)))
    )

    lm_early = pose.landmarks
    ls_im = lm_early[LandmarkIdx.LEFT_SHOULDER]
    rs_im = lm_early[LandmarkIdx.RIGHT_SHOULDER]
    lw_im = lm_early[LandmarkIdx.LEFT_WRIST]
    rw_im = lm_early[LandmarkIdx.RIGHT_WRIST]
    # Normalized image: y grows downward → “above” is smaller y.
    out["left_wrist_above_shoulder"] = bool(lw_im[1] < ls_im[1])
    out["right_wrist_above_shoulder"] = bool(rw_im[1] < rs_im[1])

    hip_vec = rh - lh
    lateral = np.array([hip_vec[0], 0.0, hip_vec[2]], dtype=np.float64)
    nl = np.linalg.norm(lateral)
    trunk_h = np.array([trunk[0], 0.0, trunk[2]], dtype=np.float64)
    ntr = np.linalg.norm(trunk_h)
    if nl > 1e-9 and ntr > 1e-9:
        lateral /= nl
        trunk_h /= ntr
        out["shoulder_hip_lateral_asym_deg"] = float(
            np.degrees(np.arccos(float(np.clip(abs(np.dot(lateral, trunk_h)), 0.0, 1.0))))
        )
    else:
        out["shoulder_hip_lateral_asym_deg"] = float("nan")

    la = landmark_xyz(pose, LandmarkIdx.LEFT_ANKLE)
    ra = landmark_xyz(pose, LandmarkIdx.RIGHT_ANKLE)
    shoulder_mid = m_sh
    hip_mid = m_hip
    ankle_mid = mid_point(la, ra)
    vec1 = shoulder_mid - hip_mid
    vec2 = hip_mid - ankle_mid
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 > 1e-9 and n2 > 1e-9:
        v1 = vec1 / n1
        v2 = vec2 / n2
        dev = float(np.degrees(np.arccos(float(np.clip(np.dot(v1, v2), -1.0, 1.0)))))
        out["body_line_deviation_deg"] = abs(dev - 180.0)
    else:
        out["body_line_deviation_deg"] = float("nan")

    lm = pose.landmarks
    mid_hip_y = float((lm[LandmarkIdx.LEFT_HIP, 1] + lm[LandmarkIdx.RIGHT_HIP, 1]) * 0.5)
    out["mid_hip_y_norm"] = mid_hip_y
    ms_x = float((lm[LandmarkIdx.LEFT_SHOULDER, 0] + lm[LandmarkIdx.RIGHT_SHOULDER, 0]) * 0.5)
    mh_x = float((lm[LandmarkIdx.LEFT_HIP, 0] + lm[LandmarkIdx.RIGHT_HIP, 0]) * 0.5)
    out["lateral_shift_norm"] = abs(ms_x - mh_x)

    return out


class MotionAnalyzer:
    def __init__(self) -> None:
        self._rep_counters: Dict[str, Dict[str, Any]] = {}

    def _rep_state(self, ex_id: ExerciseId) -> Dict[str, Any]:
        key = ex_id.value
        if key not in self._rep_counters:
            self._rep_counters[key] = {}
        return self._rep_counters[key]

    def reset_reps(self, ex_id: Optional[ExerciseId] = None) -> None:
        if ex_id is None:
            self._rep_counters.clear()
        else:
            self._rep_counters.pop(ex_id.value, None)

    def analyze(
        self,
        pose: PoseResult,
        ex_id: ExerciseId,
        cal: Optional[CalibrationData],
    ) -> ExerciseAnalysis:
        angles = compute_angles(pose)
        ctx = AnalysisContext(
            shoulder_width=cal.shoulder_width if cal else 1.0,
            torso_length=cal.torso_length if cal else 1.0,
            torso_height_norm=cal.torso_height_norm if cal else 0.2,
        )
        rep = self._rep_state(ex_id)
        standing_mid_hip = cal.mid_hip_y_norm if cal else None

        if ex_id == ExerciseId.SQUAT:
            return analyze_squat(angles, ctx, angles["mid_hip_y_norm"], standing_mid_hip, rep)
        if ex_id == ExerciseId.PUSHUP:
            return analyze_pushup(angles, ctx, rep)
        if ex_id == ExerciseId.LUNGE:
            return analyze_lunge(angles, ctx, rep)
        if ex_id == ExerciseId.ARM_RAISE:
            return analyze_arm_raise(angles, ctx, rep)
        if ex_id == ExerciseId.SIDE_STRETCH:
            return analyze_side_stretch(angles, ctx, rep)
        if ex_id == ExerciseId.POSTURE:
            return analyze_posture(angles, ctx, rep)
        raise ValueError(f"Unknown exercise: {ex_id}")
