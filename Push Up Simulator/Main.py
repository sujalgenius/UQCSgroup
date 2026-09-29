
#Import modules
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import pygame
import os

#Import specialised modules
from countdown import run_countdown
from audio_beep import play_beep
from stopwatch import start_timer, get_total_time
from six_seven import check_and_trigger, draw_if_active
from Summary import summary_screen
from motivation import check_milestones, check_target, draw_if_active as draw_motivation, reset as reset_motivation
from instructions import show_instructions

#Some parameters
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CALORIES_PER_REP = 0.4
BIG_MACS_PER_REP = 0.048
WEIGHT_KG = 70
ACTIVITY_MET_MOD = 3.8
DOWN_ANGLE = 95.5
UP_ANGLE = 145 
ARM_VISIBILITY = 0.30 
BODY_VISIBILITY = 0.20   
DOWN_CONFIRM_TIME = 0.10 
UP_CONFIRM_TIME = 0.10
ANGLE_HISTORY_SIZE = 3
PEACE_CONFIRM_FRAMES = 8
BODY_ALIGNMENT_THRESHOLD_ANGLE = 150
ALIGNMENT_VISIBILITY = 0.20
TRACKING_LOSS_GRACE = 0.30
MIN_REP_DURATION = 0.35
ALIGNMENT_SWITCH_FRAMES = 5
ALIGNMENT_SWITCH_MARGIN = 0.08


#Variables for better side detection
SIDE_SWITCH_FRAMES = 20
SIDE_SWITCH_MARGIN = 0.15

#Target rep key dictionary
TARGET_KEYS = {
    ord("1"): 5,
    ord("2"): 20,
    ord("3"): 30,
    ord("4"): 40,
    ord("5"): 50,
    ord("6"): 60,
    ord("7"): 70,
    ord("8"): 80,
    ord("9"): 90
}


#COLOUR HEX CODES FOR EASIER REFERENCES
WHITE = (255, 255, 255)
GREEN = (0, 255, 120)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
TORQOISE_BLUE = (0, 255, 239)

ARM_COLOR = (0, 255, 255)
TORSO_COLOR = (0, 255, 120)
LEG_COLOR = (255, 180, 0)

#Mediapipe setting
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

#Pose setting
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.45, min_tracking_confidence=0.45)

#Landmark indices for left and right body parts. THESE ARE THE POINTS TO BE DRAWN
BODY = {"left":  {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27, "heel": 29}, "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28, "heel": 30}
        }

#Deque has "max" feature which could drop item, whereas list could not
elbow_history = deque(maxlen=ANGLE_HISTORY_SIZE)
hip_history = deque(maxlen=ANGLE_HISTORY_SIZE)
knee_history = deque(maxlen=ANGLE_HISTORY_SIZE)

def calculate_angle(point_a, point_b, point_c):
    ba_x = point_a[0] - point_b[0]
    ba_y = point_a[1] - point_b[1]
    bc_x = point_c[0] - point_b[0]
    bc_y = point_c[1] - point_b[1]

    mag_ba = np.sqrt(ba_x ** 2 + ba_y ** 2)
    mag_bc = np.sqrt(bc_x ** 2 + bc_y ** 2)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    dot_product = ba_x * bc_x + ba_y * bc_y #Dot product of vectors BA and BC
    cosine = dot_product / (mag_ba * mag_bc) #Cosine of vectors BA and BC
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine))) #Angle in degrees between vectors BA and BC


def check_alignment(frame, landmarks, alignment_side):

    ids = BODY[alignment_side]
    shoulder_raw_coord = landmarks[ids["shoulder"]]
    hip_raw_coord = landmarks[ids["hip"]]
    ankle_raw_coord = landmarks[ids["ankle"]]
    
    if (shoulder_raw_coord.visibility < ALIGNMENT_VISIBILITY or hip_raw_coord.visibility < ALIGNMENT_VISIBILITY or  ankle_raw_coord.visibility < ALIGNMENT_VISIBILITY):
        draw_text(frame, "Alignment not visible!", (25, 415), ORANGE, 2.5, 2)
        return False 
    body_angle = calculate_angle(get_point(shoulder_raw_coord), get_point(hip_raw_coord), get_point(ankle_raw_coord))

    alignment_correct = (body_angle >= BODY_ALIGNMENT_THRESHOLD_ANGLE)
    height, width = frame.shape[:2]

    shoulder_pixel_coord = get_pixel(shoulder_raw_coord, width, height) #Convert from normalised coords to pixel coords
    ankle_pixel_coord = get_pixel(ankle_raw_coord, width, height)
    
    if alignment_correct:
        cv2.line(frame, shoulder_pixel_coord, ankle_pixel_coord, GREEN, 6, cv2.LINE_AA)
        draw_text(frame, "Alignment straight!", (25, 415), GREEN, 2.5, 2)
    else:
        draw_text(frame, "Alignment isnt straight!", (25, 415), RED, 2.5, 2)
        
    shoulder_x, shoulder_y = shoulder_pixel_coord
    draw_text(frame, f"{body_angle:.1f}", (shoulder_x - 80, shoulder_y - 10), WHITE, 0.7)
    return alignment_correct

def get_point(landmark):
    return [landmark.x, landmark.y] #Return the x and y coordinates of a LANDMARK AS A LIST

def get_pixel(landmark, width, height):
    return (int(landmark.x * width), int(landmark.y * height)) #Return the PIXEL COORDINATES of a LANDMARK as tuple

def filter_angle(history, value): 
    if value is None:
        return None
    history.append(value)
    return float(np.median(history)) #Return median of angles in deque out of 3

def arm_visibility_score(landmarks, side):
    indice_list = BODY[side] #Retrieve shoulder, elbow, wrist coords depending on body orientation
    shoulder_visibility_score = landmarks[indice_list["shoulder"]].visibility
    elbow_visibility_score = landmarks[indice_list["elbow"]].visibility          
    wrist_visibility_score = landmarks[indice_list["wrist"]].visibility                              
    return min(shoulder_visibility_score, elbow_visibility_score, wrist_visibility_score) #Return the minimum visibility score of the 3 joints

def update_side(landmarks, locked_side, bad_side_frames):
    left_score = arm_visibility_score(landmarks, "left")
    right_score = arm_visibility_score(landmarks, "right")
    switched = False

    if locked_side is None:
        if right_score >= left_score:
            locked_side = "right"
        else:
            locked_side = "left"
    locked_score = right_score if locked_side == "right" else left_score
    other_score = left_score if locked_side == "right" else right_score

    looked_bad = locked_score < ARM_VISIBILITY
    other_looks_plausible = other_score >= ARM_VISIBILITY or other_score > locked_score + SIDE_SWITCH_MARGIN

    if looked_bad and other_looks_plausible:
        bad_side_frames += 1
    else:
        bad_side_frames = 0

    if bad_side_frames >= SIDE_SWITCH_FRAMES:
        locked_side = "left" if locked_side == "right" else "right"
        bad_side_frames = 0
        switched = True
    
    return locked_side, bad_side_frames, switched 

    
def arm_visible(landmarks, side):
    ids = BODY[side]
    shoulder = landmarks[ids["shoulder"]]
    elbow = landmarks[ids["elbow"]]
    wrist = landmarks[ids["wrist"]]
    return (shoulder.visibility >= ARM_VISIBILITY and elbow.visibility >= ARM_VISIBILITY and wrist.visibility >= ARM_VISIBILITY)

def get_elbow_angle(landmarks, side):
    ids = BODY[side]
    shoulder = get_point(landmarks[ids["shoulder"]])
    elbow = get_point(landmarks[ids["elbow"]])
    wrist = get_point(landmarks[ids["wrist"]])
    return calculate_angle(shoulder, elbow, wrist)

def get_body_angles(landmarks, side):
    ids = BODY[side]
    shoulder_lm = landmarks[ids["shoulder"]]
    hip_lm = landmarks[ids["hip"]]
    knee_lm = landmarks[ids["knee"]]
    ankle_lm = landmarks[ids["ankle"]]
    hip_angle = None
    knee_angle = None

    if (shoulder_lm.visibility >= BODY_VISIBILITY and hip_lm.visibility >= BODY_VISIBILITY and knee_lm.visibility >= BODY_VISIBILITY):
        hip_angle = calculate_angle(get_point(shoulder_lm), get_point(hip_lm), get_point(knee_lm))   #CALCULATE KNEE ANGLE BASED ON HIP, KNEE AND ANKLE IF ABOVE BODY_VISIBILITY

    if (hip_lm.visibility >= BODY_VISIBILITY and knee_lm.visibility >= BODY_VISIBILITY and ankle_lm.visibility >= BODY_VISIBILITY):
        knee_angle = calculate_angle(get_point(hip_lm), get_point(knee_lm), get_point(ankle_lm))
    return (hip_angle, knee_angle)

def body_visibility_score(landmarks, side):
    ids = BODY[side]

    shoulder = landmarks[ids["shoulder"]].visibility
    hip = landmarks[ids["hip"]].visibility
    ankle = landmarks[ids["ankle"]].visibility

    return min(shoulder, hip, ankle)

def update_alignment_side(landmarks, alignment_side, better_side_frames):

    left_score = body_visibility_score(landmarks, "left")
    right_score = body_visibility_score(landmarks, "right")

    # First detection
    if alignment_side is None:
        alignment_side = "left" if left_score >= right_score else "right"
        return alignment_side, 0

    current_score = (left_score if alignment_side == "left" else right_score)
    other_side = ("right" if alignment_side == "left" else "left")
    other_score = (right_score if alignment_side == "left" else left_score)
 
    should_switch = (current_score < ALIGNMENT_VISIBILITY and other_score >= ALIGNMENT_VISIBILITY) or other_score > current_score + ALIGNMENT_SWITCH_MARGIN

    if should_switch:
        better_side_frames += 1

        if better_side_frames >= ALIGNMENT_SWITCH_FRAMES:
            alignment_side = other_side
            better_side_frames = 0
    else:
        better_side_frames = 0

    return alignment_side, better_side_frames

def draw_text(frame, text, position, color=WHITE, scale=0.7, thickness=2):
    x, y = position
    # Black outline
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA) 
    # Main text
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)


def draw_active_side(frame, landmarks, side, alignment_side):
    height, width = frame.shape[:2] #Get the height and width of the frame
    landmarks_dict = BODY[side] #Retrieve all the landmarks based on the side (left or right) and store them in dict
    points = {}
    for name, index in landmarks_dict.items(): 
        actual_mediapipe_landmark = landmarks[index]
        points[name] = get_pixel(actual_mediapipe_landmark, width, height) #Get the pixel coordinates of the landmark

    # ARM
    cv2.line(frame, points["wrist"], points["elbow"], (0, 255, 255), 5, cv2.LINE_AA)
    cv2.line(frame, points["elbow"], points["shoulder"], (0, 255, 255), 5, cv2.LINE_AA)

    #Check body alignment:
    alignment_correct = check_alignment(frame, landmarks, alignment_side)


    # TORSO
    if landmarks[landmarks_dict["hip"]].visibility >= BODY_VISIBILITY and landmarks[landmarks_dict["shoulder"]].visibility >= BODY_VISIBILITY:
        cv2.line(frame, points["shoulder"], points["hip"], (0, 255, 255), 5, cv2.LINE_AA) 
    #Line aa is smoothed jagged line

    # UPPER LEG
    if (landmarks[landmarks_dict["hip"]].visibility >= BODY_VISIBILITY and landmarks[landmarks_dict["knee"]].visibility >= BODY_VISIBILITY):
        cv2.line(frame, points["hip"], points["knee"], LEG_COLOR, 5, cv2.LINE_AA)

    # LOWER LEG
    if (landmarks[landmarks_dict["knee"]].visibility >= BODY_VISIBILITY and landmarks[landmarks_dict["ankle"]].visibility >= BODY_VISIBILITY):
        cv2.line(frame, points["knee"], points["ankle"], LEG_COLOR, 5, cv2.LINE_AA)

    # DRAW JOINTS INDIVIDUALLY IF VISIBLE
    for name, index in landmarks_dict.items():
        visibility = landmarks[index].visibility
        if visibility >= BODY_VISIBILITY:
            cv2.circle(frame, points[name], 7, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, points[name], 11, WHITE, 2, cv2.LINE_AA)

    return points, alignment_correct

def calorie_tracker(frame, reps):
    height, width = frame.shape[:2]
    calories_burned = reps * CALORIES_PER_REP
    perc_of_big_mac = reps * BIG_MACS_PER_REP
    big_mac_num = 0
    calorie_text = f"Calories: {calories_burned:.2f}kcal"
    big_mac_perc = f"% of Big Mac: {perc_of_big_mac:.2f}%"
    num_big_mac = f"# of Big Macs: {big_mac_num} Big Macs"
    if perc_of_big_mac == 100:
        big_mac_num += 1

    draw_text(frame, calorie_text, (width - 220, 30), YELLOW, 1.2, 2)
    draw_text(frame, big_mac_perc, (width - 270,50 ), YELLOW, 1.2, 2)
    draw_text(frame, num_big_mac, (width - 280,75 ), YELLOW, 1.2, 2)

    return calories_burned


def change_music_vol(music_value):
    set_volume_music = music_value/100
    pygame.mixer.music.set_volume(set_volume_music) #Set volume

def main():
    music_value = 10
    cap = cv2.VideoCapture(0)   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH) #Defne the width and height of camera frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT) #Define the width and height of camera frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) #Define the buffer size of camera frame

    if not cap.isOpened(): #If camera is not opened...
        print("Sorry, camera couldnt be opened!")
        return

    pygame.mixer.init()
    script_directory = os.path.dirname(os.path.abspath(__file__)) #Find absolute location of music file
    song_path = os.path.join(script_directory, "workout_song.mp3")
    pygame.mixer.music.load(song_path) #Load Music
    pygame.mixer.music.play(-1) #Make sure song loops infinitely!
    change_music_vol(music_value)

    if not show_instructions("instructions.png"):
        cap.release()
        cv2.destroyAllWindows()
        return


    #cv2.createTrackbar("Volume", "Side Push-Up Tracker",10, 100, change_music_vol)

    #Default values for reps, stage, down_frames and up_frames
    reps = 0
    stage = "UP"
    down_confirm_start = None
    up_confirm_start = None
    beep_enabled = True
    music_enabled = True
    target_reps = None

    #changes in 28/09/2026

    locked_side = None
    bad_side_frames = 0
    alignment_side = None
    alignment_switch_frames = 0
    rep_form_valid = True
    tracking_lost_start = None
    rep_start_time = None

    if not run_countdown(cap, seconds=5): #Run countdown before starting the pushup tracker
        cap.release()
        cv2.destroyAllWindows()
        return
    
    start_time = start_timer()

    try:
        while cap.isOpened(): #While camera is opened, read the camera frame and process it

            current_time = time.monotonic() #current measured time
            passed_time_sec = current_time - start_time #elapsed time

            camera_read_successfully, frame = cap.read() 
            if not camera_read_successfully:
                print("Sorry, camera frame couldnt be read!")
                break

            # MIRROR CAMERA
            frame = cv2.flip(frame, 1) #Flip the frame horizontally
            h, w = frame.shape[:2]

            # MEDIAPIPE
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Convert
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            # Initialize variables safely to prevent crashes when people leave the frame
            filtered_elbow_angle = None
            hip_angle = None
            knee_angle = None
            landmarks = None
        
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                locked_side, bad_side_frames, switched = update_side(landmarks, locked_side, bad_side_frames)
                alignment_side, alignment_switch_frames = update_alignment_side(landmarks, alignment_side, alignment_switch_frames)

                if switched:
                    elbow_history.clear()
                    hip_history.clear()
                    knee_history.clear()
                    down_confirm_start = None
                    up_confirm_start = None        
                    stage = "UP"
                    rep_form_valid = True
                    rep_start_time = None


                # CHECK IF COMPONENTS OF ARMS ARE VISIBLE
                if arm_visible(landmarks, locked_side):
                    tracking_lost_start = None

                    points, alignment_correct = draw_active_side(frame, landmarks, locked_side, alignment_side) 

                    # ELBOW ANGLE
                    raw_elbow_angle = get_elbow_angle(landmarks, locked_side)                
                    filtered_elbow_angle = filter_angle(elbow_history, raw_elbow_angle)

                    # HIP AND KNEE ANGLES
                    (raw_hip_angle, raw_knee_angle) = get_body_angles(landmarks, alignment_side)
                    hip_angle = filter_angle(hip_history, raw_hip_angle)
                    knee_angle = filter_angle(knee_history, raw_knee_angle)
            
                    # SHOW ELBOW ANGLE
                    if filtered_elbow_angle is not None:
                        elbow_x_coord, elbow_y_coord = points["elbow"]
                        draw_text(frame, f"{filtered_elbow_angle:.1f}", (elbow_x_coord + 20, elbow_y_coord - 15), YELLOW, 0.9, 2) #Display the angle SLIGHTLY ABOVE AND RIGHT OF THE ELBOW JOINT

                    #PUSHUP REP LOGIC
                    if filtered_elbow_angle is not None: 

                        #Up stage

                        if stage == "UP":
                            rep_form_valid = True

                            if filtered_elbow_angle < UP_ANGLE:
                                stage = "DESCENDING"
                                rep_start_time = current_time 

                        #Descending stage
                        elif stage == "DESCENDING":
                            if not alignment_correct:
                                rep_form_valid = False
                            if filtered_elbow_angle <= DOWN_ANGLE:

                                if down_confirm_start is None:
                                  down_confirm_start = current_time

                                elif current_time - down_confirm_start >= DOWN_CONFIRM_TIME:
                                  stage = "DOWN"
                                  down_confirm_start = None
                            else:
                                down_confirm_start = None
                            if filtered_elbow_angle >= UP_ANGLE:
                              stage = "UP"
                              down_confirm_start = None
                              rep_form_valid = True
                              rep_start_time = None

                        #DOWN STAGE
                        elif stage == "DOWN":
                            if not alignment_correct:
                                rep_form_valid = False
                            if filtered_elbow_angle > DOWN_ANGLE:
                                stage = "ASCENDING"

                        #ASCENDING STAGE
                        elif stage == "ASCENDING":
                            if not alignment_correct:
                                rep_form_valid = False

                            if filtered_elbow_angle >= UP_ANGLE:
                                if up_confirm_start is None:
                                    up_confirm_start = current_time

                                elif current_time - up_confirm_start >= UP_CONFIRM_TIME:
                                    if rep_start_time is not None:
                                        rep_duration = current_time - rep_start_time 
                                    else:
                                        rep_duration = None

                                    if rep_duration is None:
                                        print("Invalid Rep Duration!")
                                    elif rep_duration < MIN_REP_DURATION:
                                        print(f"Rep too fast! Duration: {rep_duration:.2f}s. Rep not counted.")
                                    elif not rep_form_valid:
                                        print("Rep form invalid! Rep not counted.")
                                    else:
                                        reps += 1
                                        print(f"Push-up completed! Total: {reps}")
                                        if beep_enabled:
                                             play_beep()

                                        check_and_trigger(reps)
                                        check_milestones(reps)
                                        check_target(reps, target_reps)
                                    #Reset variables for next rep
                                    rep_form_valid = True
                                    stage = "UP"
                                    up_confirm_start = None
                                    rep_start_time = None
                            else:
                                up_confirm_start = None
                else:
                    elbow_history.clear()
                    hip_history.clear()
                    knee_history.clear()
                    if tracking_lost_start is None:
                        tracking_lost_start = current_time
                    elif current_time - tracking_lost_start >= TRACKING_LOSS_GRACE:
                        #Abort current rep, reset stage to UP, and reset rep form validity
                        stage = "UP"
                        rep_form_valid = True          
                        down_confirm_start = None  
                        up_confirm_start = None
                        rep_start_time = None
                        alignment_side = None
                        alignment_switch_frames = 0

            else:
                # NO PERSON DETECTED
                elbow_history.clear()
                hip_history.clear()
                knee_history.clear()
                if tracking_lost_start is None:
                    tracking_lost_start = current_time
                elif current_time - tracking_lost_start >= TRACKING_LOSS_GRACE:
                    stage = "UP"
                    rep_form_valid = True          
                    down_confirm_start = None  
                    up_confirm_start = None
                    rep_start_time = None
                    alignment_side = None
                    alignment_switch_frames = 0
                    draw_text(frame, "ERROR! OUT OF FOCUS!", (w // 2 - 350, h // 2), RED, 4.0, 4)
            
            draw_text(frame, f"REPS: {reps}", (40, 90), GREEN, 4.5, 3)

            if target_reps is not None:
                if reps >= target_reps:
                    draw_text(frame, f"TARGET REP OF {target_reps} REACHED!",(w - 350, 150), GREEN, 1.5, 2)
                else:
                    draw_text(frame, f"TARGET: {target_reps}",(w - 225, 150), WHITE, 1.5, 2)

            beep_status_colour = GREEN if beep_enabled else RED
            draw_text(frame, f"Beep: {'ON' if beep_enabled else 'OFF'} (B to toggle)", (40, 650), beep_status_colour, 2.5, 1)

            music_status_colour = GREEN if music_enabled else RED
            draw_text(frame, f"Music: {'ON' if music_enabled else 'OFF'} (M to toggle)", (40, 700), music_status_colour, 2.5, 1)

            stage_color = GREEN if stage == "UP" else ORANGE
            draw_text(frame, f"POSITION: {stage}", (40, 180), stage_color, 4, 2)

            elapsed_str = get_total_time(start_time)
            draw_text(frame, f"Time: {elapsed_str}", (w - 220, 100), WHITE, 1.2, 2)


            # DISPLAY CALORIES
            calorie_tracker(frame, reps)

            # to draw the call so it actually renders
            draw_if_active(frame) ## for 67 call
            draw_motivation(frame) ## for milestones and target call (10 reps increments)

            # DISPLAY
            cv2.imshow("Side Push-Up Tracker", frame)

            # KEYBOARD
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Q = quit
                break

            elif key == ord("r"):  
                #r key for reset. Reset reps, assume user is in the "up" position, clear all elbow history to reset weighting
                reps = 0
                stage = "UP" 
                down_confirm_start = None
                up_confirm_start = None
                elbow_history.clear()
                hip_history.clear()
                knee_history.clear() 
                start_time = start_timer()
                locked_side = None
                alignment_side = None
                alignment_switch_frames = 0
                bad_side_frames = 0
                rep_start_time = None
                rep_form_valid = True
                tracking_lost_start = None



                reset_motivation()
                print("[INFO] Counter reset")

            elif key == ord("b"): # B = toggle beep sound on/off
                beep_enabled = not beep_enabled
                status = "enabled" if beep_enabled else "disabled"
                print(f"[INFO] Beep sound is: {status}")

            elif key == ord("m"): # M = toggle music on/off
                music_enabled = not music_enabled
                if music_enabled:
                    pygame.mixer.music.unpause()
                    print("[INFO] Music is: enabled")
                else:
                    pygame.mixer.music.pause()
                    status = "enabled" if music_enabled else "disabled"
                    print(f"[INFO] Music is: {status}")
                
            elif key == ord("e"): #E  = end session
                final_calories_burned = reps * CALORIES_PER_REP
                choice = summary_screen(frame, reps, final_calories_burned, passed_time_sec)
                if choice == "restart":
                    reps = 0
                    stage = "UP"
                    locked_side = None
                    bad_side_frames = 0
                    rep_start_time = None
                    rep_form_valid = True
                    down_confirm_start = None
                    up_confirm_start = None
                    elbow_history.clear()
                    hip_history.clear()
                    knee_history.clear() 
                    start_time = start_timer()
                    tracking_lost_start = None
                    alignment_side = None
                    alignment_switch_frames = 0
                    reset_motivation()
                else:
                    break # quit and exit the main loop same as before
            elif key in TARGET_KEYS:
                target_reps = TARGET_KEYS[key]
                _target_announced = False 
                print(f"[INFO] Target reps set to: {target_reps}") #For debugging 

            elif key == ord("d"):
                    music_value += 10
                    change_music_vol(music_value)
            elif key == ord("a"):
                    music_value -= 10
                    change_music_vol(music_value)
        
    finally:
        pygame.mixer.music.stop()
        cap.release()
        pose.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()