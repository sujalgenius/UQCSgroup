import cv2
import time
import random
import threading
import pyttsx3

MILESTONE_STEP = 10 # trigger every 10 reps: 10, 20, 30... infinity
BANNER_DURATION = 2.5 # the number of seconds the text stays on the screen

MILESTONE_PHRASES = [
    #"Keep going sweetie!",
    "Keep going!",
    "Nice work human!",
    "Stay strong!",
    "You're the best!",
    #"You're tung tung tung tung sahooring!",
    "That's the spirit!",
    #"Push sweetheart!",
    "Push!",
    "You are a machine!",
    "Why did the egg cross the road? Because the chicken did!",
    "Your arms look so good!",
]

TARGET_PHRASE = "You reached the target, stand proud, you are strong!"

_last_milestone = 0 # the highest milestone already announced this session
_target_announced = False # whether the target-reached message has already fired or not
_banner_text = None
_banner_started_at = None


def check_milestones(reps):
    """
    Call this once per completed rep.
    Fires every MILESTONE_STEP reps.
    """
    global _last_milestone
    milestone = (reps // MILESTONE_STEP) * MILESTONE_STEP
    if milestone > 0 and milestone > _last_milestone:
        _last_milestone = milestone
        phrase = random.choice(MILESTONE_PHRASES)
        _show_banner(f"{milestone} reps! {phrase}")
        _speak(phrase)


def check_target(reps, target_reps):
    """
    Call this once per completed rep, only if a target has been set.
    """
    global _target_announced
    if target_reps is None:
        return
    if reps >= target_reps and not _target_announced:
        _target_announced = True
        _show_banner(TARGET_PHRASE)
        _speak(TARGET_PHRASE)


def reset():
    """
    Call this whenever the user resets their rep count (aka pressing 'r').
    """
    global _last_milestone, _target_announced
    _last_milestone = 0
    _target_announced = False


def _show_banner(text):
    global _banner_text, _banner_started_at
    _banner_text = text
    _banner_started_at = time.time()


def draw_if_active(frame):
    """
    This calls every frame.
    And draws the motivational banner if one is currently active.
    """
    global _banner_text, _banner_started_at
    if _banner_text is None:
        return

    elapsed = time.time() - _banner_started_at
    if elapsed > BANNER_DURATION:
        _banner_text = None
        return

    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thickness = 3

    (text_w, text_h), _ = cv2.getTextSize(_banner_text, font, scale, thickness)
    x = (width - text_w) // 2
    y = int(height * 0.45)

    cv2.putText(frame, _banner_text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, _banner_text, (x, y), font, scale, (0, 255, 120), thickness, cv2.LINE_AA)


def _speak(phrase):
    threading.Thread(target=_speak_now, args=(phrase,), daemon=True).start()


def _speak_now(phrase):
    engine = pyttsx3.init()
    engine.say(phrase)
    engine.runAndWait()