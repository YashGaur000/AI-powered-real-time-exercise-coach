"""
AI-powered exercise coach: full-body pose + facial exercise guidance.

Run: .venv\\Scripts\\python.exe main.py
Tab: switch body / face mode.
Body: 1–6 exercises, C calibrate stance.
Face: 1–5 exercises, C calibrate neutral face.
R reset reps, V voice, Q quit.
"""

from __future__ import annotations

import sys
import time

import cv2

from calibration import CalibrationStore
from exercises import ExerciseId, get_exercise
from face_calibration import FaceCalibrationStore
from face_exercises import FaceExerciseId, FaceMotionAnalyzer, get_face_exercise
from face_overlays import draw_face_feedback_overlay
from face_tracker import FaceTracker
from hud_layout import draw_bottom_status_bar, draw_help_panel
from motion_analyzer import MotionAnalyzer
from overlays import draw_feedback_overlay
from pose_tracker import PoseTracker
from voice_guide import VoiceGuide

EXERCISE_KEYS_BODY = {
    ord("1"): ExerciseId.SQUAT,
    ord("2"): ExerciseId.PUSHUP,
    ord("3"): ExerciseId.LUNGE,
    ord("4"): ExerciseId.ARM_RAISE,
    ord("5"): ExerciseId.SIDE_STRETCH,
    ord("6"): ExerciseId.POSTURE,
}

EXERCISE_KEYS_FACE = {
    ord("1"): FaceExerciseId.SMILE,
    ord("2"): FaceExerciseId.MOUTH_OPEN,
    ord("3"): FaceExerciseId.EYE_SQUEEZE,
    ord("4"): FaceExerciseId.CHEEK_LIFT,
    ord("5"): FaceExerciseId.JAW_STRETCH,
}


def _draw_notice(frame_bgr, message: str, *, face_mode: bool) -> None:
    """Top-left banner when pose/face is missing; width-limited so it clears the help panel."""
    from hud_layout import FONT, content_right_bound, line_height, wrap_text

    h, w = frame_bgr.shape[:2]
    x0 = 12
    max_w = max(80, content_right_bound(w) - x0)
    scale = 0.58
    thickness = 2
    lines = wrap_text(message, max_w, scale, thickness)
    if not lines:
        return
    lh = line_height(scale, thickness, gap=6)
    pad = 8
    band_h = pad + len(lines) * lh + 6
    x1 = min(x0 + max_w + 16, content_right_bound(w) + 4)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (4, 6), (x1, 6 + band_h), (35, 32, 28), -1)
    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (4, 6), (x1, 6 + band_h), (90, 85, 75), 1)
    color = (140, 220, 255) if face_mode else (120, 210, 255)
    y = 6 + pad + lh - 4
    for line in lines:
        cv2.putText(frame_bgr, line, (x0, y), FONT, scale, color, thickness, cv2.LINE_AA)
        y += lh


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0.", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    pose_tracker = PoseTracker(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    )
    face_tracker = FaceTracker(
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    body_cal = CalibrationStore()
    face_cal = FaceCalibrationStore()
    body_analyzer = MotionAnalyzer()
    face_analyzer = FaceMotionAnalyzer()
    voice = VoiceGuide(enabled=True)

    body_exercise = ExerciseId.SQUAT
    face_exercise = FaceExerciseId.SMILE
    face_mode = False
    window = "Exercise Coach"

    t_start = time.perf_counter()

    def help_lines() -> list[str]:
        if face_mode:
            return [
                "FACE MODE",
                "1 Smile  2 Mouth open  3 Eye squeeze",
                "4 Cheek lift  5 Jaw stretch",
                "C Calibrate neutral face",
                "Tab Body mode  R Reps  V Voice  Q Quit",
            ]
        return [
            "BODY MODE",
            "1 Squat  2 Push-up  3 Lunge",
            "4 Arm raise  5 Side stretch  6 Posture",
            "C Calibrate neutral stance",
            "Tab Face mode  R Reps  V Voice  Q Quit",
        ]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.perf_counter() - t_start) * 1000)

            if face_mode:
                face = face_tracker.process_bgr(frame, timestamp_ms)
                fdef = get_face_exercise(face_exercise)
                if face is not None:
                    face_tracker.draw_face_mesh(frame, face.landmarks, region="all")
                    analysis_f = face_analyzer.analyze(face, face_exercise, face_cal.data)
                    draw_face_feedback_overlay(
                        frame, face, analysis_f, fdef, face_exercise
                    )
                    warns = [f.message for f in analysis_f.feedback if f.severity == "warn"]
                    if warns:
                        voice.speak(warns[0])
                else:
                    _draw_notice(
                        frame,
                        "Center your face in the frame",
                        face_mode=True,
                    )

                fstat = "Face calibrated" if face_cal.is_ready else "Press C — neutral face (relaxed)"
                draw_bottom_status_bar(
                    frame,
                    fstat,
                    ok=face_cal.is_ready,
                )
            else:
                pose = pose_tracker.process_bgr(frame, timestamp_ms)
                ex = get_exercise(body_exercise)
                if pose is not None:
                    pose_tracker.draw_skeleton(
                        frame, pose.landmarks, highlight=ex.primary_landmarks
                    )
                    analysis_b = body_analyzer.analyze(pose, body_exercise, body_cal.data)
                    draw_feedback_overlay(frame, pose, analysis_b, ex)
                    warns = [f.message for f in analysis_b.feedback if f.severity == "warn"]
                    if warns:
                        voice.speak(warns[0])
                else:
                    _draw_notice(
                        frame,
                        "Step into frame — full body visible",
                        face_mode=False,
                    )

                bstat = "Body calibrated" if body_cal.is_ready else "Press C — neutral stance"
                draw_bottom_status_bar(
                    frame,
                    bstat,
                    ok=body_cal.is_ready,
                )

            draw_help_panel(frame, help_lines())
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if key == 9:
                face_mode = not face_mode
                face_analyzer.clear_hold_timer()
                voice.speak_priority("Face exercises" if face_mode else "Body exercises")

            if face_mode:
                if key in EXERCISE_KEYS_FACE:
                    face_exercise = EXERCISE_KEYS_FACE[key]
                    face_analyzer.clear_hold_timer()
                    voice.speak_priority(get_face_exercise(face_exercise).display_name + " selected")
                if key == ord("c"):
                    fg = face_tracker.process_bgr(frame, timestamp_ms)
                    if fg is not None and face_cal.capture(fg):
                        voice.speak_priority("Face calibration saved.")
                    else:
                        voice.speak_priority("Face not detected — look at the camera in neutral pose.")
                if key == ord("r"):
                    face_analyzer.reset_reps(face_exercise)
                    voice.speak_priority("Rep count reset")
            else:
                if key in EXERCISE_KEYS_BODY:
                    body_exercise = EXERCISE_KEYS_BODY[key]
                    voice.speak_priority(get_exercise(body_exercise).display_name + " selected")
                if key == ord("c"):
                    pose = pose_tracker.process_bgr(frame, timestamp_ms)
                    if pose is not None and body_cal.capture(pose):
                        voice.speak_priority("Body calibration saved.")
                    else:
                        voice.speak_priority("Could not calibrate body. Stand in a clear neutral pose.")
                if key == ord("r"):
                    body_analyzer.reset_reps(body_exercise)
                    voice.speak_priority("Rep count reset")

            if key == ord("v"):
                voice.enabled = not voice.enabled
                voice.speak_priority("Voice on" if voice.enabled else "Voice off")

    finally:
        voice.close()
        pose_tracker.close()
        face_tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
