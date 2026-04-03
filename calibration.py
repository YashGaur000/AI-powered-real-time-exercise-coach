"""Neutral-stance calibration for body proportions and standing reference."""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from pose_tracker import LandmarkIdx, PoseResult, angle_degrees_3d, landmark_xyz, mid_point


@dataclasses.dataclass
class CalibrationData:
    """Captured while user stands in a relaxed neutral pose facing the camera."""

    shoulder_width: float
    hip_width: float
    torso_length: float
    standing_left_knee_angle: float
    standing_right_knee_angle: float
    standing_left_elbow_angle: float
    standing_right_elbow_angle: float
    mid_hip_y_norm: float
    mid_shoulder_y_norm: float
    torso_height_norm: float

    @staticmethod
    def from_pose(pose: PoseResult) -> Optional["CalibrationData"]:
        lm = pose.landmarks
        wl = pose.world_landmarks
        if wl is None:
            return None

        ls = landmark_xyz(pose, LandmarkIdx.LEFT_SHOULDER)
        rs = landmark_xyz(pose, LandmarkIdx.RIGHT_SHOULDER)
        lh = landmark_xyz(pose, LandmarkIdx.LEFT_HIP)
        rh = landmark_xyz(pose, LandmarkIdx.RIGHT_HIP)
        lk = landmark_xyz(pose, LandmarkIdx.LEFT_KNEE)
        rk = landmark_xyz(pose, LandmarkIdx.RIGHT_KNEE)
        la = landmark_xyz(pose, LandmarkIdx.LEFT_ANKLE)
        ra = landmark_xyz(pose, LandmarkIdx.RIGHT_ANKLE)
        le = landmark_xyz(pose, LandmarkIdx.LEFT_ELBOW)
        re = landmark_xyz(pose, LandmarkIdx.RIGHT_ELBOW)
        lw = landmark_xyz(pose, LandmarkIdx.LEFT_WRIST)
        rw = landmark_xyz(pose, LandmarkIdx.RIGHT_WRIST)

        sw = float(np.linalg.norm(ls - rs))
        hw = float(np.linalg.norm(lh - rh))
        msh = mid_point(ls, rs)
        mhip = mid_point(lh, rh)
        torso = float(np.linalg.norm(msh - mhip))

        lk_ang = angle_degrees_3d(lh, lk, la)
        rk_ang = angle_degrees_3d(rh, rk, ra)
        le_ang = angle_degrees_3d(ls, le, lw)
        re_ang = angle_degrees_3d(rs, re, rw)

        if any(np.isnan(x) for x in (lk_ang, rk_ang, le_ang, re_ang)):
            return None

        mid_hip_y = float((lm[LandmarkIdx.LEFT_HIP, 1] + lm[LandmarkIdx.RIGHT_HIP, 1]) * 0.5)
        mid_sh_y = float((lm[LandmarkIdx.LEFT_SHOULDER, 1] + lm[LandmarkIdx.RIGHT_SHOULDER, 1]) * 0.5)
        torso_h = abs(mid_hip_y - mid_sh_y)

        return CalibrationData(
            shoulder_width=max(sw, 1e-6),
            hip_width=max(hw, 1e-6),
            torso_length=max(torso, 1e-6),
            standing_left_knee_angle=lk_ang,
            standing_right_knee_angle=rk_ang,
            standing_left_elbow_angle=le_ang,
            standing_right_elbow_angle=re_ang,
            mid_hip_y_norm=mid_hip_y,
            mid_shoulder_y_norm=mid_sh_y,
            torso_height_norm=max(torso_h, 0.05),
        )


class CalibrationStore:
    def __init__(self) -> None:
        self.data: Optional[CalibrationData] = None

    def capture(self, pose: PoseResult) -> bool:
        c = CalibrationData.from_pose(pose)
        if c is None:
            return False
        self.data = c
        return True

    @property
    def is_ready(self) -> bool:
        return self.data is not None
