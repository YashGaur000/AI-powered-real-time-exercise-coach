"""Facial exercise definitions, normalized metrics, and coaching logic."""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_calibration import FaceCalibrationData
from mediapipe.tasks.python.vision import face_landmarker as _flm

from face_tracker import FaceLandmarkIdx, FaceResult, mouth_opening_norm


class FaceExerciseId(enum.Enum):
    SMILE = "smile"
    MOUTH_OPEN = "mouth_open"
    EYE_SQUEEZE = "eye_squeeze"
    CHEEK_LIFT = "cheek_lift"
    JAW_STRETCH = "jaw_stretch"


@dataclasses.dataclass
class FaceExerciseDefinition:
    id: FaceExerciseId
    display_name: str
    description: str
    highlight_indices: Tuple[int, ...]
    voice_hints: Tuple[str, ...]


FACE_EXERCISES: Dict[FaceExerciseId, FaceExerciseDefinition] = {
    FaceExerciseId.SMILE: FaceExerciseDefinition(
        id=FaceExerciseId.SMILE,
        display_name="Smile",
        description="Lift cheeks, widen mouth, slight eye smile.",
        highlight_indices=(
            FaceLandmarkIdx.MOUTH_LEFT,
            FaceLandmarkIdx.MOUTH_RIGHT,
            FaceLandmarkIdx.LEFT_CHEEK,
            FaceLandmarkIdx.RIGHT_CHEEK,
        ),
        voice_hints=("Smile wider", "Lift your cheeks slightly", "Hold the smile"),
    ),
    FaceExerciseId.MOUTH_OPEN: FaceExerciseDefinition(
        id=FaceExerciseId.MOUTH_OPEN,
        display_name="Mouth opening",
        description="Lower the jaw, open the mouth vertically.",
        highlight_indices=(FaceLandmarkIdx.LIP_TOP_INNER, FaceLandmarkIdx.LIP_BOTTOM_INNER, FaceLandmarkIdx.CHIN),
        voice_hints=("Open your mouth wider", "Lower your jaw slowly", "Relax and repeat"),
    ),
    FaceExerciseId.EYE_SQUEEZE: FaceExerciseDefinition(
        id=FaceExerciseId.EYE_SQUEEZE,
        display_name="Eye squeeze",
        description="Gently close eyelids without furrowing.",
        highlight_indices=(
            FaceLandmarkIdx.LEFT_EYE_TOP,
            FaceLandmarkIdx.LEFT_EYE_BOTTOM,
            FaceLandmarkIdx.RIGHT_EYE_TOP,
            FaceLandmarkIdx.RIGHT_EYE_BOTTOM,
        ),
        voice_hints=("Squeeze your eyes gently", "Hold for two seconds", "Relax your eyes"),
    ),
    FaceExerciseId.CHEEK_LIFT: FaceExerciseDefinition(
        id=FaceExerciseId.CHEEK_LIFT,
        display_name="Cheek lift",
        description="Lift cheeks and corners of the mouth upward.",
        highlight_indices=(FaceLandmarkIdx.LEFT_CHEEK, FaceLandmarkIdx.RIGHT_CHEEK, FaceLandmarkIdx.MOUTH_LEFT, FaceLandmarkIdx.MOUTH_RIGHT),
        voice_hints=("Lift your cheeks upward", "Hold the position", "Relax your cheeks"),
    ),
    FaceExerciseId.JAW_STRETCH: FaceExerciseDefinition(
        id=FaceExerciseId.JAW_STRETCH,
        display_name="Jaw stretch",
        description="Open and move the jaw slightly forward in a controlled stretch.",
        highlight_indices=(FaceLandmarkIdx.CHIN, FaceLandmarkIdx.LIP_BOTTOM_INNER),
        voice_hints=("Stretch your jaw downward", "Move slightly forward", "Relax and repeat"),
    ),
}


def get_face_exercise(eid: FaceExerciseId) -> FaceExerciseDefinition:
    return FACE_EXERCISES[eid]


def _bs(face: FaceResult, name: str) -> float:
    return float(face.blendshape_scores.get(name, 0.0))


def _bs_delta(cal: Optional[FaceCalibrationData], idx: int, current: float) -> float:
    if cal is None or idx < 0 or idx >= len(cal.blendshape_baseline):
        return current
    return float(current - cal.blendshape_baseline[idx])


def _idx(name: str) -> int:
    return int(_flm.Blendshapes[name])


@dataclasses.dataclass
class FaceFeedbackItem:
    message: str
    severity: str
    highlight_indices: Tuple[int, ...] = ()
    arrow_from_idx: Optional[int] = None
    arrow_delta_xy: Optional[Tuple[float, float]] = None
    region_glow: str = ""


@dataclasses.dataclass
class FaceExerciseAnalysis:
    feedback: List[FaceFeedbackItem]
    rep_count: int
    rep_phase: str
    quality_score: float
    hold_progress: float


class FaceMotionAnalyzer:
    def __init__(self) -> None:
        self._rep: Dict[str, Dict[str, float]] = {}
        self._hold_since: Optional[float] = None

    def reset_reps(self, eid: Optional[FaceExerciseId] = None) -> None:
        if eid is None:
            self._rep.clear()
        else:
            self._rep.pop(eid.value, None)
        self._hold_since = None

    def clear_hold_timer(self) -> None:
        self._hold_since = None

    def _state(self, eid: FaceExerciseId) -> Dict[str, float]:
        if eid.value not in self._rep:
            self._rep[eid.value] = {"count": 0.0, "phase": 0.0}
        return self._rep[eid.value]

    def analyze(
        self,
        face: FaceResult,
        eid: FaceExerciseId,
        cal: Optional[FaceCalibrationData],
    ) -> FaceExerciseAnalysis:
        lm = face.landmarks
        mo = mouth_opening_norm(lm)
        st = self._state(eid)
        now = time.monotonic()
        hold_prog = 0.0

        feedback: List[FaceFeedbackItem] = []
        quality = 0.85

        def add(msg: str, sev: str, **kwargs) -> None:
            nonlocal quality
            feedback.append(FaceFeedbackItem(message=msg, severity=sev, **kwargs))
            if sev == "warn":
                quality = min(quality, 0.45)

        if eid == FaceExerciseId.SMILE:
            sl = _bs(face, "MOUTH_SMILE_LEFT")
            sr = _bs(face, "MOUTH_SMILE_RIGHT")
            smile = (sl + sr) * 0.5
            d_sl = _bs_delta(cal, _idx("MOUTH_SMILE_LEFT"), sl)
            d_sr = _bs_delta(cal, _idx("MOUTH_SMILE_RIGHT"), sr)
            delta_smile = (d_sl + d_sr) * 0.5
            cheek = (_bs(face, "CHEEK_SQUINT_LEFT") + _bs(face, "CHEEK_SQUINT_RIGHT")) * 0.5

            if smile < 0.2 and delta_smile < 0.12:
                add(
                    "Smile a little wider — lift the corners of your mouth",
                    "warn",
                    highlight_indices=(
                        FaceLandmarkIdx.MOUTH_LEFT,
                        FaceLandmarkIdx.MOUTH_RIGHT,
                        FaceLandmarkIdx.LEFT_CHEEK,
                        FaceLandmarkIdx.RIGHT_CHEEK,
                    ),
                    arrow_from_idx=FaceLandmarkIdx.MOUTH_LEFT,
                    arrow_delta_xy=(-35.0, -8.0),
                )
            elif cheek < 0.08 and smile > 0.15:
                add("Lift your cheeks slightly", "warn", highlight_indices=(FaceLandmarkIdx.LEFT_CHEEK, FaceLandmarkIdx.RIGHT_CHEEK))
            else:
                add("Good — hold the smile", "good", highlight_indices=(FaceLandmarkIdx.LEFT_CHEEK, FaceLandmarkIdx.RIGHT_CHEEK))
                if self._hold_since is None:
                    self._hold_since = now
                hold_prog = min(1.0, (now - self._hold_since) / 2.0)
            if smile < 0.12:
                self._hold_since = None

            ph = int(st["phase"])
            cnt = int(st["count"])
            if smile > 0.28 and ph == 0:
                st["phase"] = 1
            elif smile < 0.1 and ph == 1:
                cnt += 1
                st["phase"] = 0
            st["count"] = float(cnt)

        elif eid == FaceExerciseId.MOUTH_OPEN:
            jaw = _bs(face, "JAW_OPEN")
            d_jaw = _bs_delta(cal, _idx("JAW_OPEN"), jaw)
            mo_n = mo
            baseline_mo = cal.mouth_open_norm if cal else 0.05
            if jaw < 0.25 and d_jaw < 0.15 and mo_n < max(baseline_mo * 1.8, 0.04):
                add(
                    "Open your mouth wider — lower your jaw slowly",
                    "warn",
                    highlight_indices=(FaceLandmarkIdx.CHIN,),
                    arrow_from_idx=FaceLandmarkIdx.CHIN,
                    arrow_delta_xy=(0.0, 45.0),
                )
            else:
                add("Nice controlled opening — relax and repeat", "good")
            ph = int(st["phase"])
            cnt = int(st["count"])
            if jaw > 0.35 and ph == 0:
                st["phase"] = 1
            elif jaw < 0.12 and ph == 1:
                cnt += 1
                st["phase"] = 0
            st["count"] = float(cnt)

        elif eid == FaceExerciseId.EYE_SQUEEZE:
            sq = (_bs(face, "EYE_SQUINT_LEFT") + _bs(face, "EYE_SQUINT_RIGHT")) * 0.5
            bl = (_bs(face, "EYE_BLINK_LEFT") + _bs(face, "EYE_BLINK_RIGHT")) * 0.5
            squeeze = max(sq, bl * 0.9)
            if squeeze < 0.35:
                add(
                    "Squeeze your eyes gently",
                    "warn",
                    highlight_indices=(
                        FaceLandmarkIdx.LEFT_EYE_TOP,
                        FaceLandmarkIdx.RIGHT_EYE_TOP,
                    ),
                    arrow_from_idx=FaceLandmarkIdx.LEFT_EYE_TOP,
                    arrow_delta_xy=(0.0, 12.0),
                )
            else:
                add("Good — hold for two seconds", "good")
                if self._hold_since is None:
                    self._hold_since = now
                hold_prog = min(1.0, (now - self._hold_since) / 2.0)
            if squeeze < 0.2:
                self._hold_since = None

            ph = int(st["phase"])
            cnt = int(st["count"])
            if squeeze > 0.45 and ph == 0:
                st["phase"] = 1
            elif squeeze < 0.15 and ph == 1:
                cnt += 1
                st["phase"] = 0
            st["count"] = float(cnt)

        elif eid == FaceExerciseId.CHEEK_LIFT:
            ul = (_bs(face, "MOUTH_UPPER_UP_LEFT") + _bs(face, "MOUTH_UPPER_UP_RIGHT")) * 0.5
            sm = (_bs(face, "MOUTH_SMILE_LEFT") + _bs(face, "MOUTH_SMILE_RIGHT")) * 0.5
            lift = ul * 0.6 + sm * 0.4
            d_ul = (
                _bs_delta(cal, _idx("MOUTH_UPPER_UP_LEFT"), _bs(face, "MOUTH_UPPER_UP_LEFT"))
                + _bs_delta(cal, _idx("MOUTH_UPPER_UP_RIGHT"), _bs(face, "MOUTH_UPPER_UP_RIGHT"))
            ) * 0.5
            if lift < 0.18 and d_ul < 0.1:
                add(
                    "Lift your cheeks higher",
                    "warn",
                    highlight_indices=(FaceLandmarkIdx.LEFT_CHEEK, FaceLandmarkIdx.RIGHT_CHEEK),
                    arrow_from_idx=FaceLandmarkIdx.LEFT_CHEEK,
                    arrow_delta_xy=(0.0, -40.0),
                )
            else:
                add("Good cheek lift — hold the position", "good")

            ph = int(st["phase"])
            cnt = int(st["count"])
            if lift > 0.22 and ph == 0:
                st["phase"] = 1
            elif lift < 0.1 and ph == 1:
                cnt += 1
                st["phase"] = 0
            st["count"] = float(cnt)

        elif eid == FaceExerciseId.JAW_STRETCH:
            jaw_o = _bs(face, "JAW_OPEN")
            jaw_f = _bs(face, "JAW_FORWARD")
            stretch = jaw_o * 0.65 + jaw_f * 0.35
            if stretch < 0.3:
                add(
                    "Stretch your jaw downward, then slightly forward",
                    "warn",
                    highlight_indices=(FaceLandmarkIdx.CHIN,),
                    arrow_from_idx=FaceLandmarkIdx.CHIN,
                    arrow_delta_xy=(8.0, 50.0),
                )
            else:
                add("Controlled stretch — relax and repeat", "good")

            ph = int(st["phase"])
            cnt = int(st["count"])
            if stretch > 0.4 and ph == 0:
                st["phase"] = 1
            elif stretch < 0.15 and ph == 1:
                cnt += 1
                st["phase"] = 0
            st["count"] = float(cnt)

        return FaceExerciseAnalysis(
            feedback=feedback,
            rep_count=int(st["count"]),
            rep_phase=str(int(st["phase"])),
            quality_score=quality,
            hold_progress=hold_prog,
        )
