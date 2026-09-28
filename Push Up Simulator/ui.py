import sys
import time
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

import main as tracker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Push-Up Tracker")
        self.resize(1350, 800)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, tracker.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, tracker.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.reps = 0
        self.stage = "UP"
        self.beep_enabled = True
        self.start_time = time.monotonic()

        self.down_confirm_start = None
        self.up_confirm_start = None
        self.locked_side = None
        self.bad_side_frames = 0
        self.alignment_side = None
        self.alignment_switch_frames = 0
        self.rep_form_valid = True
        self.tracking_lost_start = None
        self.rep_start_time = None

        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        title = QLabel("PUSH-UP TRACKER")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: white;")
        main_layout.addWidget(title)

        content = QHBoxLayout()
        content.setSpacing(16)

        self.camera_label = QLabel("Opening camera...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(850, 560)
        self.camera_label.setStyleSheet("background: #05070a; border: 1px solid #30363d; border-radius: 12px; color: #8b949e;")
        content.addWidget(self.camera_label, 3)

        side = QFrame()
        side.setStyleSheet("QFrame { background: #161b22; border: 1px solid #30363d; border-radius: 12px; } QLabel { color: white; border: none; }")

        stats = QVBoxLayout(side)
        stats.setContentsMargins(18, 18, 18, 18)
        stats.setSpacing(18)

        self.reps_label = QLabel("REPS\n0")
        self.stage_label = QLabel("POSITION\nUP")
        self.form_label = QLabel("FORM\nWAITING")
        self.time_label = QLabel("TIME\n00:00")

        for label in (self.reps_label, self.stage_label, self.form_label, self.time_label):
            label.setStyleSheet("font-size: 22px; font-weight: 700; padding: 14px;")
            stats.addWidget(label)

        stats.addStretch()

        self.reset_button = QPushButton("Reset")
        self.beep_button = QPushButton("Beep: ON")

        for button in (self.reset_button, self.beep_button):
            button.setMinimumHeight(42)
            button.setStyleSheet("QPushButton { background: #21262d; color: white; border: 1px solid #30363d; border-radius: 8px; font-weight: 700; } QPushButton:hover { background: #30363d; }")
            stats.addWidget(button)

        content.addWidget(side, 1)
        main_layout.addLayout(content)

        self.reset_button.clicked.connect(self.reset_tracker)
        self.beep_button.clicked.connect(self.toggle_beep)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def clear_histories(self):
        tracker.elbow_history.clear()
        tracker.hip_history.clear()
        tracker.knee_history.clear()

    def reset_rep_state(self):
        self.stage = "UP"
        self.down_confirm_start = None
        self.up_confirm_start = None
        self.rep_form_valid = True
        self.rep_start_time = None

    def handle_tracking_loss(self, current_time):
        self.clear_histories()
        if self.tracking_lost_start is None:
            self.tracking_lost_start = current_time
            return
        if current_time - self.tracking_lost_start < tracker.TRACKING_LOSS_GRACE:
            return
        self.reset_rep_state()
        self.alignment_side = None
        self.alignment_switch_frames = 0

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            return

        current_time = time.monotonic()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = tracker.pose.process(rgb)
        rgb.flags.writeable = True

        form_text = "WAITING"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.locked_side, self.bad_side_frames, switched = tracker.update_side(landmarks, self.locked_side, self.bad_side_frames)
            self.alignment_side, self.alignment_switch_frames = tracker.update_alignment_side(landmarks, self.alignment_side, self.alignment_switch_frames)

            if switched:
                self.clear_histories()
                self.reset_rep_state()

            if tracker.arm_visible(landmarks, self.locked_side):
                self.tracking_lost_start = None
                points, alignment_correct = tracker.draw_active_side(frame, landmarks, self.locked_side, self.alignment_side)

                raw_elbow_angle = tracker.get_elbow_angle(landmarks, self.locked_side)
                filtered_elbow_angle = tracker.filter_angle(tracker.elbow_history, raw_elbow_angle)

                raw_hip_angle, raw_knee_angle = tracker.get_body_angles(landmarks, self.alignment_side)
                tracker.filter_angle(tracker.hip_history, raw_hip_angle)
                tracker.filter_angle(tracker.knee_history, raw_knee_angle)

                form_text = "GOOD" if alignment_correct else "BAD"

                if filtered_elbow_angle is not None:
                    elbow_x, elbow_y = points["elbow"]
                    tracker.draw_text(frame, f"{filtered_elbow_angle:.1f}", (elbow_x + 20, elbow_y - 15), tracker.YELLOW, 0.9, 2)

                    if self.stage == "UP":
                        self.rep_form_valid = True
                        if filtered_elbow_angle < tracker.UP_ANGLE:
                            self.stage = "DESCENDING"
                            self.rep_start_time = current_time

                    elif self.stage == "DESCENDING":
                        if not alignment_correct:
                            self.rep_form_valid = False

                        if filtered_elbow_angle <= tracker.DOWN_ANGLE:
                            if self.down_confirm_start is None:
                                self.down_confirm_start = current_time
                            elif current_time - self.down_confirm_start >= tracker.DOWN_CONFIRM_TIME:
                                self.stage = "DOWN"
                                self.down_confirm_start = None
                        else:
                            self.down_confirm_start = None

                        if filtered_elbow_angle >= tracker.UP_ANGLE:
                            self.reset_rep_state()

                    elif self.stage == "DOWN":
                        if not alignment_correct:
                            self.rep_form_valid = False
                        if filtered_elbow_angle > tracker.DOWN_ANGLE:
                            self.stage = "ASCENDING"

                    elif self.stage == "ASCENDING":
                        if not alignment_correct:
                            self.rep_form_valid = False

                        if filtered_elbow_angle >= tracker.UP_ANGLE:
                            if self.up_confirm_start is None:
                                self.up_confirm_start = current_time
                            elif current_time - self.up_confirm_start >= tracker.UP_CONFIRM_TIME:
                                rep_duration = current_time - self.rep_start_time if self.rep_start_time is not None else None

                                if rep_duration is None:
                                    form_text = "INVALID"
                                elif rep_duration < tracker.MIN_REP_DURATION:
                                    form_text = "TOO FAST"
                                elif not self.rep_form_valid:
                                    form_text = "BAD FORM"
                                else:
                                    self.reps += 1
                                    form_text = "GOOD"
                                    if self.beep_enabled:
                                        tracker.play_beep()
                                    tracker.check_and_trigger(self.reps)
                                    tracker.check_milestones(self.reps)

                                self.reset_rep_state()
                        else:
                            self.up_confirm_start = None
            else:
                form_text = "TRACKING LOST"
                self.handle_tracking_loss(current_time)

        else:
            form_text = "NO PERSON"
            self.handle_tracking_loss(current_time)

        tracker.draw_text(frame, f"REPS: {self.reps}", (40, 90), tracker.GREEN, 4.5, 3)
        tracker.draw_text(frame, f"POSITION: {self.stage}", (40, 180), tracker.GREEN if self.stage == "UP" else tracker.ORANGE, 4, 2)

        elapsed = int(current_time - self.start_time)
        minutes, seconds = divmod(elapsed, 60)
        elapsed_text = f"{minutes:02d}:{seconds:02d}"

        self.reps_label.setText(f"REPS\n{self.reps}")
        self.stage_label.setText(f"POSITION\n{self.stage}")
        self.form_label.setText(f"FORM\n{form_text}")
        self.time_label.setText(f"TIME\n{elapsed_text}")

        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_display.shape
        image = QImage(rgb_display.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_label.setPixmap(pixmap)

    def reset_tracker(self):
        self.reps = 0
        self.reset_rep_state()
        self.locked_side = None
        self.bad_side_frames = 0
        self.alignment_side = None
        self.alignment_switch_frames = 0
        self.tracking_lost_start = None
        self.clear_histories()
        self.start_time = time.monotonic()

    def toggle_beep(self):
        self.beep_enabled = not self.beep_enabled
        self.beep_button.setText(f"Beep: {'ON' if self.beep_enabled else 'OFF'}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        tracker.pose.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
