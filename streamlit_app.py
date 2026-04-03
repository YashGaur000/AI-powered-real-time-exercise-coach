from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from calibration import CalibrationStore
from exercises import EXERCISES, ExerciseId, get_exercise
from face_calibration import FaceCalibrationStore
from face_exercises import FACE_EXERCISES, FaceExerciseId, FaceMotionAnalyzer, get_face_exercise
from face_overlays import draw_face_feedback_overlay
from face_tracker import FaceTracker
from hud_layout import draw_bottom_status_bar, draw_help_panel
from main import _draw_notice
from motion_analyzer import MotionAnalyzer
from overlays import draw_feedback_overlay
from pose_tracker import PoseTracker


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    face_mode: bool = False
    body_exercise: ExerciseId = ExerciseId.SQUAT
    face_exercise: FaceExerciseId = FaceExerciseId.SMILE
    calibrate_body_requested: bool = False
    calibrate_face_requested: bool = False
    reset_reps_requested: bool = False
    last_feedback: str = "Open camera to start"
    last_mode_label: str = "BODY MODE"


class CoachVideoProcessor(VideoProcessorBase):
    def __init__(self, state: SharedState) -> None:
        self.state = state
        self.pose_tracker = PoseTracker(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self.face_tracker = FaceTracker(
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.body_cal = CalibrationStore()
        self.face_cal = FaceCalibrationStore()
        self.body_analyzer = MotionAnalyzer()
        self.face_analyzer = FaceMotionAnalyzer()
        self.t_start = time.perf_counter()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        ts = int((time.perf_counter() - self.t_start) * 1000)

        with self.state.lock:
            face_mode = self.state.face_mode
            body_exercise = self.state.body_exercise
            face_exercise = self.state.face_exercise
            calibrate_body = self.state.calibrate_body_requested
            calibrate_face = self.state.calibrate_face_requested
            reset_reps = self.state.reset_reps_requested
            self.state.calibrate_body_requested = False
            self.state.calibrate_face_requested = False
            self.state.reset_reps_requested = False

        if reset_reps:
            if face_mode:
                self.face_analyzer.reset_reps(face_exercise)
            else:
                self.body_analyzer.reset_reps(body_exercise)

        if face_mode:
            feedback_line = "Center your face in the frame"
            face = self.face_tracker.process_bgr(img, ts)
            fdef = get_face_exercise(face_exercise)
            if face is not None:
                self.face_tracker.draw_face_mesh(img, face.landmarks, region="all")
                analysis = self.face_analyzer.analyze(face, face_exercise, self.face_cal.data)
                draw_face_feedback_overlay(img, face, analysis, fdef, face_exercise)
                warns = [f.message for f in analysis.feedback if f.severity == "warn"]
                if warns:
                    feedback_line = warns[0]
                elif analysis.feedback:
                    feedback_line = analysis.feedback[0].message

                if calibrate_face:
                    if self.face_cal.capture(face):
                        feedback_line = "Face calibration saved"
                    else:
                        feedback_line = "Face calibration failed. Look straight with neutral expression."
            else:
                _draw_notice(img, "Center your face in the frame", face_mode=True)
                if calibrate_face:
                    feedback_line = "Face not detected for calibration"

            fstat = "Face calibrated" if self.face_cal.is_ready else "Press Calibrate in sidebar (neutral face)"
            draw_bottom_status_bar(img, fstat, ok=self.face_cal.is_ready)
            help_lines = [
                "FACE MODE",
                "Sidebar: select exercise + calibrate",
                "Smile / Mouth open / Eye squeeze",
                "Cheek lift / Jaw stretch",
                "Use Reset reps when needed",
            ]
            mode_label = "FACE MODE"
        else:
            feedback_line = "Step into frame with full body visible"
            pose = self.pose_tracker.process_bgr(img, ts)
            ex = get_exercise(body_exercise)
            if pose is not None:
                self.pose_tracker.draw_skeleton(img, pose.landmarks, highlight=ex.primary_landmarks)
                analysis = self.body_analyzer.analyze(pose, body_exercise, self.body_cal.data)
                draw_feedback_overlay(img, pose, analysis, ex)
                warns = [f.message for f in analysis.feedback if f.severity == "warn"]
                if warns:
                    feedback_line = warns[0]
                elif analysis.feedback:
                    feedback_line = analysis.feedback[0].message

                if calibrate_body:
                    if self.body_cal.capture(pose):
                        feedback_line = "Body calibration saved"
                    else:
                        feedback_line = "Body calibration failed. Stand in neutral stance."
            else:
                _draw_notice(img, "Step into frame — full body visible", face_mode=False)
                if calibrate_body:
                    feedback_line = "Body not detected for calibration"

            bstat = "Body calibrated" if self.body_cal.is_ready else "Press Calibrate in sidebar (neutral stance)"
            draw_bottom_status_bar(img, bstat, ok=self.body_cal.is_ready)
            help_lines = [
                "BODY MODE",
                "Sidebar: select exercise + calibrate",
                "Squat / Push-up / Lunge",
                "Arm raise / Side stretch / Posture",
                "Use Reset reps when needed",
            ]
            mode_label = "BODY MODE"

        draw_help_panel(img, help_lines)

        with self.state.lock:
            self.state.last_feedback = feedback_line
            self.state.last_mode_label = mode_label

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self) -> None:
        try:
            self.pose_tracker.close()
        except Exception:
            pass
        try:
            self.face_tracker.close()
        except Exception:
            pass


def _get_shared_state() -> SharedState:
    if "shared_state" not in st.session_state:
        st.session_state.shared_state = SharedState()
    return st.session_state.shared_state


def main() -> None:
    st.set_page_config(page_title="AI Exercise Coach", layout="wide")
    st.markdown(
        """
        <style>
        /* Make webcam feed occupy available main panel width */
        [data-testid="stAppViewContainer"] video,
        [data-testid="stAppViewContainer"] canvas {
            width: 100% !important;
            height: auto !important;
            max-height: 72vh !important;
            object-fit: contain !important;
            background: #000;
        }
        [data-testid="stAppViewContainer"] .stVideo,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] > div:has(video),
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] > div:has(canvas) {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Real-Time Exercise Coach")
    st.caption("Body + facial guidance in Streamlit camera view")

    state = _get_shared_state()

    with st.sidebar:
        st.header("Controls")
        mode_str = st.radio("Mode", ["Body", "Face"], index=1 if state.face_mode else 0)
        face_mode = mode_str == "Face"

        if face_mode:
            face_options = {v.display_name: k for k, v in FACE_EXERCISES.items()}
            selected_face_name = st.selectbox(
                "Facial exercise",
                list(face_options.keys()),
                index=list(face_options.values()).index(state.face_exercise),
            )
            selected_face = face_options[selected_face_name]
            selected_body = state.body_exercise
        else:
            body_options = {v.display_name: k for k, v in EXERCISES.items()}
            selected_body_name = st.selectbox(
                "Body exercise",
                list(body_options.keys()),
                index=list(body_options.values()).index(state.body_exercise),
            )
            selected_body = body_options[selected_body_name]
            selected_face = state.face_exercise

        col1, col2 = st.columns(2)
        calibrate_clicked = col1.button("Calibrate", use_container_width=True)
        reset_clicked = col2.button("Reset reps", use_container_width=True)

        st.markdown("---")
        st.write("Then click **START** under camera to open webcam.")

    with state.lock:
        state.face_mode = face_mode
        state.body_exercise = selected_body
        state.face_exercise = selected_face
        if calibrate_clicked:
            if face_mode:
                state.calibrate_face_requested = True
            else:
                state.calibrate_body_requested = True
        if reset_clicked:
            state.reset_reps_requested = True

    ctx = webrtc_streamer(
        key="coach-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False,
        },
        video_processor_factory=lambda: CoachVideoProcessor(state),
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "controls": True,
            "style": {
                "width": "100%",
                "height": "auto",
                "maxHeight": "72vh",
                "objectFit": "contain",
                "background": "black",
            },
        },
    )

    with state.lock:
        st.info(f"Mode: {state.last_mode_label}")
        st.success(f"Feedback: {state.last_feedback}")

    if not ctx.state.playing:
        st.warning("Camera is off. Click START to open camera.")


if __name__ == "__main__":
    main()

