import sys
import time
import cv2
import pygame
import os

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation
from PySide6.QtGui import QImage, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QSlider, QSpinBox, QSizePolicy, QStackedWidget, QProgressBar, QGraphicsOpacityEffect

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import Main as tracker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Push-Up Tracker")
        self.resize(1450, 750)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, tracker.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, tracker.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.reps = 0
        self.stage = "UP"
        self.target_reps = None
        self.last_frame = None

        self.beep_enabled = True
        self.music_enabled = True
        self.music_value = 10

        self.start_time = time.monotonic() #Set universal time clock

        self.down_confirm_start = None 
        self.up_confirm_start = None
        self.rep_start_time = None

        self.locked_side = None
        self.bad_side_frames = 0
        self.alignment_side = None
        self.alignment_switch_frames = 0
        self.rep_form_valid = True
        self.tracking_lost_start = None


        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.tracker_page = QWidget()
        self.summary_page = QWidget()
        self.stack.addWidget(self.tracker_page)
        self.stack.addWidget(self.summary_page)

        main_layout = QVBoxLayout(self.tracker_page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        title = QLabel("PUSH-UP TRACKER")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: white;")
        main_layout.addWidget(title)

        content = QHBoxLayout()
        content.setSpacing(16)

        analysis_panel = QFrame()
        analysis_panel.setMinimumWidth(210)
        analysis_panel.setMaximumWidth(240)

        analysis_panel.setStyleSheet(
    """
    QFrame {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
    }

    QLabel {
        border: none;
        color: white;
    }
    """)
        
        analysis_layout = QVBoxLayout(analysis_panel)
        analysis_layout.setContentsMargins(16, 18, 16, 18)
        analysis_layout.setSpacing(12)

        #Declare analysis_title
        analysis_title = QLabel("REP ANALYSER")
        analysis_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        analysis_title.setStyleSheet(
    """
    font-size: 19px;
    font-weight: 900;
    color: #00d9ff;
    """
)

        self.analysis_rep_label = QLabel("WAITING\nFOR REP")
        self.analysis_rep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_rep_label.setStyleSheet(
    """
    font-size: 16px;
    font-weight: 800;
    color: #8b949e;
    padding: 10px;
    """)

        self.analysis_score_label = QLabel(
    "SCORE\n--")
        self.analysis_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_score_label.setStyleSheet(
    """
    font-size: 28px;
    font-weight: 900;
    color: white;
    """)

        self.analysis_depth_label = QLabel("DEPTH\n--")
        self.analysis_tempo_label = QLabel("TEMPO\n--")
        self.analysis_form_label = QLabel("FORM\n--")
        self.analysis_duration_label = QLabel("DURATION\n--")
        self.analysis_min_angle_label = QLabel("MIN ANGLE\n--")

        for label in (self.analysis_depth_label, self.analysis_tempo_label, self.analysis_form_label, self.analysis_duration_label, self.analysis_min_angle_label):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
        """
        font-size: 15px;
        font-weight: 700;
        padding: 5px;
        """)


        analysis_layout.addWidget(analysis_title)
        analysis_layout.addWidget(self.analysis_rep_label)
        analysis_layout.addWidget(self.analysis_score_label)
        analysis_layout.addWidget(self.analysis_depth_label)
        analysis_layout.addWidget(self.analysis_tempo_label)
        analysis_layout.addWidget(self.analysis_form_label)
        analysis_layout.addWidget(self.analysis_duration_label)
        analysis_layout.addWidget(self.analysis_min_angle_label)
        analysis_layout.addStretch()


        self.camera_label = QLabel("Opening camera...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(0, 0) #Automatically set size
        self.camera_label.setStyleSheet("background: #05070a; border: 1px solid #30363d; border-radius: 12px; color: #8b949e;")
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) 

        side = QFrame()
        side.setStyleSheet("QFrame { background: #161b22; border: 1px solid #30363d; border-radius: 12px; } QLabel { color: white; border: none; }")
        side.setMinimumWidth(260)
        side.setMaximumWidth(320)

        stats = QVBoxLayout(side)
        stats.setContentsMargins(12, 10, 12, 10)
        stats.setSpacing(4) 


        self.rank_label = QLabel("IRON")
        self.rank_reps_label = QLabel("0 PUSHUPS")
        self.streak_label = QLabel("🔥 STREAK 0")
        self.rank_progress = QProgressBar()
        self.stage_label = QLabel("POSITION\nUP")
        self.form_label = QLabel("FORM\nWAITING")
        self.angle_label = QLabel("ELBOW ANGLE\n0.0")
        self.time_label = QLabel("TIME\n00:00")

        #Rank labels
        self.rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rank_reps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.streak_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rank_label.setStyleSheet("font-size: 24px; font-weight: 900; color: #9ca3af;")
        self.rank_reps_label.setStyleSheet("font-size: 16px; font-weight: 800; color: white;")
        self.streak_label.setStyleSheet("font-size: 15px; font-weight: 800; color: #ff8c32;")
        self.rank_progress.setRange(0, 100)
        self.rank_progress.setValue(0)
        self.rank_progress.setTextVisible(True)

        
        stats.addWidget(self.rank_label)
        stats.addWidget(self.rank_reps_label)
        stats.addWidget(self.streak_label)
        stats.addWidget(self.rank_progress)


        self.streak_effect = QGraphicsOpacityEffect(self.streak_label)
        self.streak_label.setGraphicsEffect(self.streak_effect)

        self.streak_animation = QPropertyAnimation(self.streak_effect, b"opacity")
        self.streak_animation.setDuration(450)
        self.streak_animation.setStartValue(0.35)
        self.streak_animation.setEndValue(1.0)

        #Initialise music
        pygame.mixer.init()
        script_directory = os.path.dirname(os.path.abspath(__file__))
        song_path = os.path.join(script_directory, "stargirl.mp3")
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play(-1)
        tracker.change_music_vol(self.music_value)

        for label in (self.stage_label, self.form_label, self.angle_label, self.time_label):
            label.setStyleSheet("font-size: 15px; font-weight: 700; padding: 2px;")
            stats.addWidget(label)


        self.reset_button = QPushButton("Reset")
        self.beep_button = QPushButton("Beep: ON")
        self.music_button = QPushButton("Music: ON")
        self.end_button = QPushButton("End Session")
    
        for button in (self.reset_button, self.beep_button, self.music_button, self.end_button):
            button.setMinimumHeight(32)
            button.setStyleSheet("QPushButton { background: #21262d; color: white; border: 1px solid #30363d; border-radius: 8px; font-weight: 700; } QPushButton:hover { background: #30363d; }")
            stats.addWidget(button)

        content.addWidget(analysis_panel, 5)
        content.addWidget(self.camera_label, 14)
        content.addWidget(side, 5)
        main_layout.addLayout(content)
     

        self.volume_label = QLabel("VOLUME: 10%")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(10)
        stats.addWidget(self.volume_label)
        stats.addWidget(self.volume_slider)


        self.goal_label = QLabel("GOAL\nNone")
        self.goal_input = QSpinBox()
        self.goal_input.setRange(1, 200)            
        self.goal_input.setValue(20)
        self.goal_button = QPushButton("Set Goal")

        self.rep_history = []
        self.current_rep_min_angle = None
        self.current_rep_alignment_good = True

        stats.addWidget(self.goal_label)
        stats.addWidget(self.goal_input)
        stats.addWidget(self.goal_button)

        stats.addStretch()

        self.reset_button.clicked.connect(self.reset_tracker)
        self.beep_button.clicked.connect(self.toggle_beep)
        self.music_button.clicked.connect(self.toggle_music)
        self.volume_slider.valueChanged.connect(self.change_volume)
        self.goal_button.clicked.connect(self.set_goal)
        self.end_button.clicked.connect(self.end_session)



        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        summary_layout = QVBoxLayout(self.summary_page)
        summary_layout.setContentsMargins(80, 80, 80, 80)
        summary_layout.setSpacing(30)
        summary_title = QLabel("WORKOUT SUMMARY")
        summary_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary_title.setStyleSheet("font-size: 36px; font-weight: 800; color: white;")

        self.summary_reps = QLabel("TOTAL REPS\n0")
        self.summary_average = QLabel("AVERAGE SCORE\n --")
        self.summary_calories = QLabel("CALORIES\n0.0 kcal")
        self.summary_time = QLabel("TIME\n00:00")

        #Set CSS

        for label in self.summary_reps, self.summary_calories, self.summary_time, self.summary_average:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 26px; font-weight: 700; color: white; padding: 20px;")

        summary_stats = QHBoxLayout()
        summary_stats.addWidget(self.summary_reps)
        summary_stats.addWidget(self.summary_average)
        summary_stats.addWidget(self.summary_calories)
        summary_stats.addWidget(self.summary_time)

        self.summary_figure = Figure(figsize=(7, 3))
        self.summary_canvas = FigureCanvas(self.summary_figure)
        self.summary_canvas.setStyleSheet("background: transparent; border: none;")
        self.summary_ax = self.summary_figure.add_subplot(1,1,1) #1 row, 1 column, plot number 1
        self.summary_figure.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.22)

        self.restart_button = QPushButton("Restart Workout")
        self.quit_button = QPushButton("Quit")
        self.restart_button.clicked.connect(self.restart_session)
        self.quit_button.clicked.connect(self.close)

        for button in (self.restart_button, self.quit_button):
            button.setMinimumHeight(50)
            button.setStyleSheet(
                "QPushButton {"
                "background: #21262d;"
                "color: white;"
                "border: 1px solid #30363d;"
                "border-radius: 8px;"
                "font-size: 18px;"
                "font-weight: 700;"
                "}"
            "QPushButton:hover { background: #30363d; }")

        summary_layout.addStretch()
        summary_layout.addWidget(summary_title)

        summary_layout.addLayout(summary_stats)

        summary_layout.addWidget(self.summary_canvas, 1)
        summary_layout.addWidget(self.restart_button)
        summary_layout.addWidget(self.quit_button)
        summary_layout.addStretch()
        self.summary_page.setStyleSheet("background: #0d1117;")

        #tracker.show_instructions()

        #Shortcuts
        reset_shortcut = QShortcut(self)
        reset_shortcut.setKey(Qt.Key.Key_R)
        reset_shortcut.activated.connect(self.reset_tracker)

        quit_shortcut = QShortcut(self)
        quit_shortcut.setKey(Qt.Key.Key_Q)
        quit_shortcut.activated.connect(self.close)

        end_session_shortcut = QShortcut(self)
        end_session_shortcut.setKey(Qt.Key.Key_E)
        end_session_shortcut.activated.connect(self.end_session)

        beep_toggle_shortcut = QShortcut(self)
        beep_toggle_shortcut.setKey(Qt.Key.Key_B)
        beep_toggle_shortcut.activated.connect(self.toggle_beep)

        music_toggle_shortcut = QShortcut(self)
        music_toggle_shortcut.setKey(Qt.Key.Key_M)
        music_toggle_shortcut.activated.connect(self.toggle_music)

        #instructions_shortcut = QShortcut(self)
        #instructions_shortcut.setKey(Qt.Key.Key_I)
        #instructions_shortcut.activated.connect(tracker.show_instructions)



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
        self.current_rep_min_angle = None
        self.current_rep_alignment_good = True

    def handle_tracking_loss(self, current_time): #afk detection
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
        self.last_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        #Convert tracker to rgb
        results = tracker.pose.process(rgb)
        rgb.flags.writeable = True
    

        form_text = "WAITING"
        filtered_elbow_angle = None

        if results.pose_landmarks:
            #Retrive landmarks from results
            landmarks = results.pose_landmarks.landmark
            self.locked_side, self.bad_side_frames, switched = tracker.update_side(landmarks, self.locked_side, self.bad_side_frames)
            self.alignment_side, self.alignment_switch_frames = tracker.update_alignment_side(landmarks, self.alignment_side, self.alignment_switch_frames)

            #If arm switched, then clear history of deques containing the body angles to get specific average for specific side
            if switched:
                self.clear_histories()
                self.reset_rep_state()

            if tracker.arm_visible(landmarks, self.locked_side):
                self.tracking_lost_start = None
                _, alignment_correct = tracker.draw_active_side(frame, landmarks, self.locked_side, self.alignment_side, show_text = False)

                raw_elbow_angle = tracker.get_elbow_angle(landmarks, self.locked_side)
                filtered_elbow_angle = tracker.filter_angle(tracker.elbow_history, raw_elbow_angle)

                if self.stage in ("DESCENDING", "DOWN", "ASCENDING"):   #Retrieve lowest angle for quality check
                    if self.current_rep_min_angle is None:
                        self.current_rep_min_angle = filtered_elbow_angle
                    else:
                        self.current_rep_min_angle = min(self.current_rep_min_angle, filtered_elbow_angle)


                raw_hip_angle, raw_knee_angle = tracker.get_body_angles(landmarks, self.alignment_side)
                tracker.filter_angle(tracker.hip_history, raw_hip_angle)
                tracker.filter_angle(tracker.knee_history, raw_knee_angle)

                form_text = "GOOD" if alignment_correct else "BAD"

                if filtered_elbow_angle is not None:

                    if self.stage == "UP":
                        self.rep_form_valid = True
                        if filtered_elbow_angle < tracker.UP_ANGLE:
                            self.stage = "DESCENDING"
                            self.rep_start_time = current_time

                            self.current_rep_min_angle = filtered_elbow_angle
                            self.current_rep_alignment_good = True

                    elif self.stage == "DESCENDING":
                        if not alignment_correct:
                            self.rep_form_valid = False
                            self.current_rep_alignment_good = False

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
                            self.current_rep_alignment_good = False
                        if filtered_elbow_angle > tracker.DOWN_ANGLE:
                            self.stage = "ASCENDING"

                    elif self.stage == "ASCENDING":
                        if not alignment_correct:
                            self.rep_form_valid = False
                            self.current_rep_alignment_good = False

                        if filtered_elbow_angle >= tracker.UP_ANGLE:
                            if self.up_confirm_start is None:
                                self.up_confirm_start = current_time
                            elif current_time - self.up_confirm_start >= tracker.UP_CONFIRM_TIME:
                                rep_duration = current_time - self.rep_start_time if self.rep_start_time is not None else None
                                if rep_duration is None:
                                    form_text = "INVALID FORM"
                                elif rep_duration < tracker.MIN_REP_DURATION:
                                    form_text = "TOO FAST"
                                elif not self.rep_form_valid:
                                    form_text = "BAD FORM"
                                else:
                                    self.reps += 1
                                    form_text = "GOOD"

                                    analyse = self.analyse_rep(rep_duration)
                                    self.rep_history.append(analyse)

                                    self.analysis_rep_label.setText(f"REP {self.reps}")
                                    self.analysis_score_label.setText(f"SCORE\n{analyse['score']}/100")
                                    self.analysis_depth_label.setText(f"DEPTH\n{analyse['depth']}")
                                    self.analysis_tempo_label.setText(f"TEMPO\n{analyse['tempo']}")
                                    self.analysis_form_label.setText(f"FORM\n{analyse['form']}")
                                    self.analysis_duration_label.setText(f"DURATION\n{analyse['duration']:.2f}s")
                                    self.analysis_min_angle_label.setText(f"MIN ANGLE\n{analyse['min_angle']:.1f}°")

                                    score = analyse["score"]
                                    if score >= 90:
                                         score_colour = "#00ff78"
                                    elif score >= 75:
                                         score_colour = "#facc15"
                                    else:
                                         score_colour = "#ff4d4d"

                                    self.analysis_score_label.setStyleSheet(f"""font-size: 28px; font-weight: 900;color: {score_colour}; """) #Set colour

                                    #Update rank, play animation
                                    self.update_rank_display()
                                    self.animate_streak()

                                    #Play beep sound
                                    if self.beep_enabled:
                                        tracker.play_beep()

                                    tracker.check_and_trigger(self.reps)
                                    tracker.check_milestones(self.reps)
                                    if self.target_reps is not None:
                                        tracker.check_target(self.reps, self.target_reps)

                                self.reset_rep_state()
                        else:
                            self.up_confirm_start = None
            else:
                form_text = "TRACKING LOST"
                self.handle_tracking_loss(current_time)

        else:
            form_text = "NO PERSON"
            self.handle_tracking_loss(current_time)


        elapsed = int(current_time - self.start_time)
        minutes, seconds = divmod(elapsed, 60)
        elapsed_text = f"{minutes:02d}:{seconds:02d}"

        self.update_rank_display()
        self.stage_label.setText(f"POSITION\n{self.stage}")
        self.form_label.setText(f"FORM\n{form_text}")
        self.angle_label.setText(f"ELBOW ANGLE\n{filtered_elbow_angle:.1f}" if filtered_elbow_angle is not None else "ELBOW ANGLE\nN/A")
        self.time_label.setText(f"TIME\n{elapsed_text}")


        if form_text == "GOOD":
            form_colour = "00ff78"
        elif form_text in ("BAD", "BAD FORM", "INVALID FORM", "TOO FAST"):
            form_colour = "#ff4d4d"
        elif form_text in ("TRACKING LOST", "NO PERSON"):
            form_colour = "#ffa500"
        else:
            form_colour = "#ffffff"
        self.form_label.setStyleSheet(f"font-size: 18px; font-weight: 700; padding: 6px; color: {form_colour};")

        if self.stage == "UP":
            stage_colour = "#00ff78"
        elif self.stage == "DOWN":
            stage_colour = "#ff4d4d"
        else:
            stage_colour = "#ffa500"

        self.stage_label.setStyleSheet(f"font-size: 18px; font-weight: 700; padding: 6px; color: {stage_colour};")

        self.angle_label.setStyleSheet("font-size: 18px; font-weight: 700; padding: 6px; color: #00d9ff;")

        self.time_label.setStyleSheet("font-size: 18px; font-weight: 700; padding: 6px; color: #ffffff;")


        if self.target_reps is not None:
            if self.reps >= self.target_reps:
                self.goal_label.setText(f"GOAL\n{self.reps}/{self.target_reps} ✓")
            else:
                self.goal_label.setText(f"GOAL\n{self.reps}/{self.target_reps}")

        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_display.shape

        image = QImage(rgb_display.data, width, height, channels * width, QImage.Format.Format_RGB888).copy() #Convert to QImage

        #Actual pixel data, image width, image height, bytes per row
        pixmap = QPixmap.fromImage(image).scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_label.setPixmap(pixmap)

    def set_goal(self):
        self.target_reps = self.goal_input.value()
        self.goal_label.setText(f"GOAL\n0/{self.target_reps}")

    def reset_tracker(self):
        self.reps = 0
        self.reset_rep_state()
        self.locked_side = None
        self.bad_side_frames = 0
        self.alignment_side = None
        self.rep_history.clear()
        self.alignment_switch_frames = 0
        self.tracking_lost_start = None
        self.clear_histories()
        self.start_time = time.monotonic()

    def analyse_rep(self, rep_duration):
        min_angle = self.current_rep_min_angle
        if min_angle is None:
            depth_score = 0
        elif min_angle <= tracker.DOWN_ANGLE:
            depth_score = 100
        else:
            depth_score = max(0, int(100 - (min_angle - tracker.DOWN_ANGLE) * 4))
        ideal_duration = 1.5
        tempo_score = max(0, int(100 - abs(rep_duration - ideal_duration) * 40))

        if self.current_rep_alignment_good:
            form_score = 100
        else:
            form_score = 50

        overall_score = int(depth_score * 0.40 + tempo_score * 0.25 + form_score * 0.35)
        return {"score": overall_score, "depth": depth_score, "tempo": tempo_score, "form": form_score, "duration": rep_duration, "min_angle": min_angle}


    def toggle_beep(self):
        self.beep_enabled = not self.beep_enabled
        self.beep_button.setText(f"Beep: {'ON' if self.beep_enabled else 'OFF'}")

    def animate_streak(self):
        self.streak_animation.stop()
        self.streak_animation.start()


    def toggle_music(self):
        self.music_enabled = not self.music_enabled
        if self.music_enabled:
            pygame.mixer.music.unpause()
        else:
            pygame.mixer.music.pause()
        self.music_button.setText(f"Music: {'ON' if self.music_enabled else 'OFF'}")

    def change_volume(self, value):
        self.music_value = value
        tracker.change_music_vol(value)
        self.volume_label.setText(f"VOLUME: {value}%")

    def get_rank_info(self):
        ranks = [
        (0, "IRON", "#9ca3af"),
        (5, "BRONZE", "#cd7f32"),
        (9, "SILVER", "#cbd5e1"),
        (17, "GOLD", "#facc15"),
        (29, "PLATINUM", "#2dd4bf"),
        (37, "DIAMOND", "#60a5fa"),
        (49, "ASCENDANT", "#4ade80"),
        (58, "IMMORTAL", "#f43f5e"),
        (70, "RADIANT", "#fde68a") ]

        for index in range(len(ranks) - 1, -1, -1):
            threshold, rank, colour = ranks[index]
            if self.reps >= threshold:
                if index == len(ranks) - 1:
                    return rank, colour, 100, None

                next_threshold = ranks[index + 1][0]
                next_rank = ranks[index + 1][1]

                progress = int(((self.reps - threshold) /(next_threshold - threshold)) * 100)

                return rank, colour, progress, next_rank

    def update_rank_display(self):
        rank, colour, progress, next_rank = self.get_rank_info()
        self.rank_label.setText(rank)
        self.rank_label.setStyleSheet(f"font-size: 30px; font-weight: 900; color: {colour};")

        self.rank_reps_label.setText(f"{self.reps} PUSH-UPS")
        self.streak_label.setText(f"🔥 STREAK {self.reps} 🔥" if self.reps > 0 else "STREAK 0")
        self.rank_progress.setValue(progress)

        if next_rank is None:
            self.rank_progress.setFormat("MAX RANK!")
        else:
            self.rank_progress.setFormat(f"{progress}% TO {next_rank}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        tracker.pose.close()
        event.accept()

    def end_session(self):
        self.timer.stop()

        passed_time = time.monotonic() - self.start_time
        calories = self.reps * tracker.CALORIES_PER_REP

        minutes = int(passed_time // 60)
        seconds = int(passed_time % 60)

        if self.rep_history:
            average_score = sum(rep["score"] for rep in self.rep_history) / len(self.rep_history)
        else:
            average_score = 0

        self.summary_reps.setText(f"TOTAL REPS\n{self.reps}")
        self.summary_calories.setText(f"CALORIES\n{calories:.1f} kcal")
        self.summary_time.setText(f"TIME\n{minutes:02d}:{seconds:02d}")
        self.summary_average.setText(f"AVERAGE SCORE\n{average_score:.0f}/100")

        self.update_summary_graph()

        self.stack.setCurrentWidget(self.summary_page)


    def update_summary_graph(self):
        ax = self.summary_ax
        fig = self.summary_figure

        ax.clear()
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        if not self.rep_history:
            ax.text(
            0.5, 0.5,
            "No Reps Recorded!",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.summary_canvas.draw()
            return

        rep_numbers = list(range(1, len(self.rep_history) + 1))
        scores = [rep["score"] for rep in self.rep_history]
        ax.plot(rep_numbers,
        scores,
        linewidth=8,
        color="#00d9ff",
        alpha=0.12)
        ax.plot(
        rep_numbers,
        scores,
        marker="o",
        linewidth=2.8,
        markersize=8,
        color="#00d9ff",
        markerfacecolor="#00d9ff",
        markeredgecolor="white",
        markeredgewidth=1.2)

    
        ax.fill_between(
        rep_numbers,
        scores,
        0,
        color="#00d9ff",
        alpha=0.08)


        ax.set_title(
        "REP SCORE",
        color="white",
        fontsize=18,
        fontweight="bold",
        pad=10)

        ax.set_xlabel("Rep", color="#c9d1d9", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", color="#c9d1d9", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlim(1, len(rep_numbers))
        ax.tick_params(axis="x", colors="#c9d1d9", labelsize=11)
        ax.tick_params(axis="y", colors="#c9d1d9", labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("#30363d")
            spine.set_linewidth(1.2)

        ax.grid(True, color="#30363d", linestyle="--", linewidth=0.8, alpha=0.7)

  
        ax.scatter(
        rep_numbers[-1],
        scores[-1],
        s=140,
        color="#7ee787",
        edgecolors="white",
        linewidths=1.5,
        zorder=5)

        ax.annotate(
        f"{scores[-1]}",
        (rep_numbers[-1], scores[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="#7ee787",
        fontsize=10,
        fontweight="bold")

        self.summary_canvas.draw()
        
    def restart_session(self):
        self.reset_tracker()
        self.stack.setCurrentWidget(self.tracker_page)
        self.timer.start(30)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
