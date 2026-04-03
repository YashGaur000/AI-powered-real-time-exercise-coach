"""Predefined exercise motion patterns and coaching thresholds."""

from __future__ import annotations

import dataclasses
import enum
from typing import List, Optional, Tuple

import numpy as np

from pose_tracker import LandmarkIdx


class ExerciseId(enum.Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    LUNGE = "lunge"
    ARM_RAISE = "arm_raise"
    SIDE_STRETCH = "side_stretch"
    POSTURE = "posture"


@dataclasses.dataclass
class ExerciseDefinition:
    id: ExerciseId
    display_name: str
    description: str
    primary_landmarks: List[int]
    coaching_hints: List[str]


EXERCISES: dict[ExerciseId, ExerciseDefinition] = {
    ExerciseId.SQUAT: ExerciseDefinition(
        id=ExerciseId.SQUAT,
        display_name="Squat",
        description="Hips back and down, knees track over toes, chest up.",
        primary_landmarks=[
            LandmarkIdx.LEFT_HIP,
            LandmarkIdx.RIGHT_HIP,
            LandmarkIdx.LEFT_KNEE,
            LandmarkIdx.RIGHT_KNEE,
            LandmarkIdx.LEFT_ANKLE,
            LandmarkIdx.RIGHT_ANKLE,
        ],
        coaching_hints=[
            "Sit back as if into a chair",
            "Keep knees aligned with feet",
            "Lower until thighs are near parallel",
        ],
    ),
    ExerciseId.PUSHUP: ExerciseDefinition(
        id=ExerciseId.PUSHUP,
        display_name="Push-up",
        description="Straight line from head to heels; elbows ~45° from body.",
        primary_landmarks=[
            LandmarkIdx.LEFT_SHOULDER,
            LandmarkIdx.RIGHT_SHOULDER,
            LandmarkIdx.LEFT_ELBOW,
            LandmarkIdx.RIGHT_ELBOW,
            LandmarkIdx.LEFT_WRIST,
            LandmarkIdx.RIGHT_WRIST,
            LandmarkIdx.LEFT_HIP,
            LandmarkIdx.RIGHT_HIP,
        ],
        coaching_hints=[
            "Keep core tight",
            "Do not let hips sag",
            "Control the descent",
        ],
    ),
    ExerciseId.LUNGE: ExerciseDefinition(
        id=ExerciseId.LUNGE,
        display_name="Forward lunge",
        description="Step forward, drop back knee toward floor, torso upright.",
        primary_landmarks=[
            LandmarkIdx.LEFT_HIP,
            LandmarkIdx.RIGHT_HIP,
            LandmarkIdx.LEFT_KNEE,
            LandmarkIdx.RIGHT_KNEE,
            LandmarkIdx.LEFT_ANKLE,
            LandmarkIdx.RIGHT_ANKLE,
        ],
        coaching_hints=[
            "Front knee over ankle",
            "Keep torso vertical",
            "Lower back knee gently",
        ],
    ),
    ExerciseId.ARM_RAISE: ExerciseDefinition(
        id=ExerciseId.ARM_RAISE,
        display_name="Arm raise",
        description="Raise both arms overhead with controlled tempo.",
        primary_landmarks=[
            LandmarkIdx.LEFT_SHOULDER,
            LandmarkIdx.RIGHT_SHOULDER,
            LandmarkIdx.LEFT_ELBOW,
            LandmarkIdx.RIGHT_ELBOW,
            LandmarkIdx.LEFT_WRIST,
            LandmarkIdx.RIGHT_WRIST,
        ],
        coaching_hints=[
            "Straighten arms without locking elbows",
            "Reach tall through the fingertips",
            "Keep shoulders relaxed",
        ],
    ),
    ExerciseId.SIDE_STRETCH: ExerciseDefinition(
        id=ExerciseId.SIDE_STRETCH,
        display_name="Side stretch",
        description="Lateral flexion: reach one arm overhead, lengthen the side body.",
        primary_landmarks=[
            LandmarkIdx.LEFT_SHOULDER,
            LandmarkIdx.RIGHT_SHOULDER,
            LandmarkIdx.LEFT_HIP,
            LandmarkIdx.RIGHT_HIP,
        ],
        coaching_hints=[
            "Breathe into the stretch",
            "Avoid collapsing forward",
        ],
    ),
    ExerciseId.POSTURE: ExerciseDefinition(
        id=ExerciseId.POSTURE,
        display_name="Posture check",
        description="Stand tall: ears over shoulders, shoulders over hips.",
        primary_landmarks=[
            LandmarkIdx.NOSE,
            LandmarkIdx.LEFT_SHOULDER,
            LandmarkIdx.RIGHT_SHOULDER,
            LandmarkIdx.LEFT_HIP,
            LandmarkIdx.RIGHT_HIP,
        ],
        coaching_hints=[
            "Gently draw shoulders back",
            "Level your hips",
        ],
    ),
}


def list_exercises() -> List[ExerciseDefinition]:
    return list(EXERCISES.values())


def get_exercise(eid: ExerciseId) -> ExerciseDefinition:
    return EXERCISES[eid]


@dataclasses.dataclass
class AnalysisContext:
    """Optional calibration-aware scaling."""

    shoulder_width: float = 1.0
    torso_length: float = 1.0
    torso_height_norm: float = 0.2


@dataclasses.dataclass
class FeedbackItem:
    message: str
    severity: str  # "info" | "warn" | "good"
    highlight_landmarks: Tuple[int, ...] = ()
    arrow_from_idx: Optional[int] = None
    arrow_to_pixel_offset: Optional[Tuple[float, float]] = None


@dataclasses.dataclass
class ExerciseAnalysis:
    feedback: List[FeedbackItem]
    rep_phase: str
    rep_count: int
    quality_score: float


def _hip_vertical_drop_ratio(
    mid_hip_y: float, standing_mid_hip_y: float, standing_torso: float
) -> float:
    """Positive when hips move down in image (y increases). Rough depth proxy."""
    if standing_torso <= 1e-9:
        return 0.0
    return float((mid_hip_y - standing_mid_hip_y) / standing_torso)


def analyze_squat(
    angles: dict,
    ctx: AnalysisContext,
    mid_hip_y_norm: float,
    standing_mid_hip_y: Optional[float],
    rep_state: dict,
) -> ExerciseAnalysis:
    feedback: List[FeedbackItem] = []
    lk = angles.get("left_knee")
    rk = angles.get("right_knee")
    hip_line = angles.get("trunk_lean_deg")

    min_k = min(lk, rk) if lk is not None and rk is not None else float("nan")
    depth_ok = False
    if standing_mid_hip_y is not None:
        drop = _hip_vertical_drop_ratio(mid_hip_y_norm, standing_mid_hip_y, ctx.torso_height_norm)
        depth_ok = drop > 0.12
        if drop < 0.05:
            feedback.append(
                FeedbackItem(
                    "Lower your hips — squat a bit deeper",
                    "warn",
                    (LandmarkIdx.LEFT_HIP, LandmarkIdx.RIGHT_HIP),
                    LandmarkIdx.LEFT_HIP,
                    (0, 40),
                )
            )

    if not np.isnan(min_k) and min_k > 100:
        feedback.append(
            FeedbackItem(
                "Bend your knees more to reach depth",
                "warn",
                (LandmarkIdx.LEFT_KNEE, LandmarkIdx.RIGHT_KNEE),
            )
        )

    if hip_line is not None and hip_line > 25:
        feedback.append(
            FeedbackItem(
                "Straighten your torso — chest a bit higher",
                "warn",
                (LandmarkIdx.LEFT_SHOULDER, LandmarkIdx.RIGHT_SHOULDER),
            )
        )

    if depth_ok and not np.isnan(min_k) and min_k < 100 and (hip_line is None or hip_line < 30):
        feedback.append(FeedbackItem("Good squat depth and alignment", "good", ()))

    phase: str = str(rep_state.get("phase", "up"))
    rep_count = int(rep_state.get("count", 0))
    if not np.isnan(min_k):
        if min_k < 95 and phase == "up":
            phase = "down"
        elif min_k > 150 and phase == "down":
            rep_count += 1
            phase = "up"
    rep_state["phase"] = phase
    rep_state["count"] = rep_count

    score = 1.0
    if any(f.severity == "warn" for f in feedback):
        score = 0.5
    return ExerciseAnalysis(feedback=feedback, rep_phase=phase, rep_count=rep_count, quality_score=score)


def analyze_pushup(
    angles: dict,
    ctx: AnalysisContext,
    rep_state: dict,
) -> ExerciseAnalysis:
    feedback: List[FeedbackItem] = []
    le = angles.get("left_elbow")
    re = angles.get("right_elbow")
    body_line = angles.get("body_line_deviation_deg")

    if body_line is not None and body_line > 20:
        feedback.append(
            FeedbackItem(
                "Keep a straight line from head to hips",
                "warn",
                (LandmarkIdx.LEFT_HIP, LandmarkIdx.RIGHT_HIP, LandmarkIdx.LEFT_SHOULDER),
            )
        )

    avg_elb = (le + re) * 0.5 if le is not None and re is not None else float("nan")
    if not np.isnan(avg_elb) and avg_elb > 160:
        feedback.append(
            FeedbackItem(
                "Bend elbows more on the way down",
                "warn",
                (LandmarkIdx.LEFT_ELBOW, LandmarkIdx.RIGHT_ELBOW),
            )
        )

    phase = rep_state.get("phase", "up")
    rep_count = int(rep_state.get("count", 0))
    if not np.isnan(avg_elb):
        if avg_elb < 100 and phase == "up":
            phase = "down"
        elif avg_elb > 155 and phase == "down":
            rep_count += 1
            phase = "up"
    rep_state["phase"] = phase
    rep_state["count"] = rep_count

    if not feedback:
        feedback.append(FeedbackItem("Maintain steady push-up rhythm", "info", ()))

    score = 0.5 if any(f.severity == "warn" for f in feedback) else 1.0
    return ExerciseAnalysis(feedback=feedback, rep_phase=phase, rep_count=rep_count, quality_score=score)


def analyze_lunge(angles: dict, ctx: AnalysisContext, rep_state: dict) -> ExerciseAnalysis:
    feedback: List[FeedbackItem] = []
    lk = angles.get("left_knee")
    rk = angles.get("right_knee")
    trunk = angles.get("trunk_lean_deg")

    if trunk is not None and trunk > 22:
        feedback.append(
            FeedbackItem("Stack shoulders over hips — stay more upright", "warn", (LandmarkIdx.LEFT_HIP, LandmarkIdx.RIGHT_HIP))
        )

    min_k = min(lk, rk) if lk is not None and rk is not None else float("nan")
    if not np.isnan(min_k) and min_k > 130:
        feedback.append(
            FeedbackItem("Lower your back knee closer to the floor", "warn", (LandmarkIdx.LEFT_KNEE, LandmarkIdx.RIGHT_KNEE))
        )

    phase = rep_state.get("phase", "stand")
    rep_count = int(rep_state.get("count", 0))
    if not np.isnan(min_k):
        if min_k < 110 and phase == "stand":
            phase = "deep"
        elif min_k > 145 and phase == "deep":
            rep_count += 1
            phase = "stand"
    rep_state["phase"] = phase
    rep_state["count"] = rep_count

    if not feedback:
        feedback.append(FeedbackItem("Good lunge control", "good", ()))

    score = 0.5 if any(f.severity == "warn" for f in feedback) else 1.0
    return ExerciseAnalysis(feedback=feedback, rep_phase=phase, rep_count=rep_count, quality_score=score)


def analyze_arm_raise(angles: dict, ctx: AnalysisContext, rep_state: dict) -> ExerciseAnalysis:
    feedback: List[FeedbackItem] = []
    le = angles.get("left_elbow")
    re = angles.get("right_elbow")
    lwrist_h = angles.get("left_wrist_above_shoulder")
    rwrist_h = angles.get("right_wrist_above_shoulder")

    raised = bool(lwrist_h and rwrist_h)
    if not raised:
        feedback.append(
            FeedbackItem(
                "Raise your arms higher overhead",
                "warn",
                (LandmarkIdx.LEFT_WRIST, LandmarkIdx.RIGHT_WRIST),
                LandmarkIdx.LEFT_WRIST,
                (0, -50),
            )
        )

    if le is not None and re is not None:
        if le < 150 or re < 150:
            feedback.append(
                FeedbackItem("Straighten your arms more as you lift", "info", (LandmarkIdx.LEFT_ELBOW, LandmarkIdx.RIGHT_ELBOW))
            )

    rep_count = int(rep_state.get("count", 0))
    was_raised = bool(rep_state.get("was_raised", False))
    if raised and not was_raised:
        rep_count += 1
    rep_state["was_raised"] = raised
    rep_state["count"] = rep_count
    phase = "up" if raised else "down"
    rep_state["phase"] = phase

    score = 0.6 if any(f.severity == "warn" for f in feedback) else 1.0
    return ExerciseAnalysis(feedback=feedback, rep_phase=phase, rep_count=rep_count, quality_score=score)


def analyze_side_stretch(angles: dict, ctx: AnalysisContext, rep_state: dict) -> ExerciseAnalysis:
    feedback: List[FeedbackItem] = []
    lat = angles.get("lateral_shift_norm")
    if lat is not None and lat < 0.04:
        feedback.append(
            FeedbackItem(
                "Reach up and lean slightly to one side to open the torso",
                "warn",
                (LandmarkIdx.LEFT_SHOULDER, LandmarkIdx.RIGHT_SHOULDER),
            )
        )
    else:
        feedback.append(FeedbackItem("Nice side lengthening", "good", ()))

    return ExerciseAnalysis(feedback=feedback, rep_phase="hold", rep_count=0, quality_score=1.0)


def analyze_posture(angles: dict, ctx: AnalysisContext, rep_state: dict) -> ExerciseAnalysis:
    feedback: List[FeedbackItem] = []
    fwd = angles.get("forward_head_deg")
    sh_tilt = angles.get("shoulder_tilt_deg")

    if fwd is not None and fwd > 12:
        feedback.append(
            FeedbackItem(
                "Ease your head back — stack ears over shoulders",
                "warn",
                (LandmarkIdx.NOSE, LandmarkIdx.LEFT_SHOULDER),
            )
        )
    if sh_tilt is not None and sh_tilt > 10:
        feedback.append(
            FeedbackItem("Level your shoulders", "warn", (LandmarkIdx.LEFT_SHOULDER, LandmarkIdx.RIGHT_SHOULDER))
        )
    if not any(f.severity == "warn" for f in feedback):
        feedback.append(FeedbackItem("Posture looks balanced", "good", ()))

    score = 0.5 if any(f.severity == "warn" for f in feedback) else 1.0
    return ExerciseAnalysis(feedback=feedback, rep_phase="neutral", rep_count=0, quality_score=score)


