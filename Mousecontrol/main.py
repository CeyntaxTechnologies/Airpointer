import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import threading
import speech_recognition as sr

# Lock to synchronize mouse commands between voice and gesture threads
lock = threading.Lock()

#__________________________________________________________Voice system
'''''
def voice_control():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                print("Listening for voice command...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print("Heard:", command)

                if "right click" in command:
                    with lock:
                        time.sleep(0.1)
                        pyautogui.rightClick()
                        time.sleep(0.1)
                    print("Performed: Right Click")

                elif "double click" in command:
                    with lock:
                        time.sleep(0.1)
                        pyautogui.doubleClick()
                        time.sleep(0.1)
                    print("Performed: Double Click")

                elif "click" in command:
                    with lock:
                        time.sleep(0.1)
                        pyautogui.click()
                        time.sleep(0.1)
                    print("Performed: Left Click")

                elif "scroll up" in command:
                    with lock:
                        pyautogui.scroll(300)
                    print("Performed: Scroll Up")

                elif "scroll down" in command:
                    with lock:
                        pyautogui.scroll(-300)
                    print("Performed: Scroll Down")

        except sr.UnknownValueError:
            print("Didn't catch that.")
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            print("Error:", e)

# Start voice thread
voice_thread = threading.Thread(target=voice_control, daemon=True)
voice_thread.start()

#__________________________________________________________ end of voice system
'''''
# Setup screen
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Smoothing
prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0
base_alpha = 0.25  # Smoothing base

# Click state
click_cooldown = 0.3
last_left_click = 0
last_right_click = 0

# Dead zone to filter jitter
dead_zone_px = 10

# Helper functions
def get_finger_pos(landmarks, index):
    return landmarks.landmark[index].x, landmarks.landmark[index].y

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Display instructions
    #cv2.putText(img, "Use index finger to move cursor", (10, 30),
               # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #cv2.putText(img, "Pinch thumb + index = Left Click", (10, 55),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    #cv2.putText(img, "Pinch thumb + middle = Right Click", (10, 80),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get fingertips
        x_index, y_index = get_finger_pos(hand_landmarks, 8)
        x_thumb, y_thumb = get_finger_pos(hand_landmarks, 4)
        x_middle, y_middle = get_finger_pos(hand_landmarks, 12)

        # Convert to screen coordinates
        target_x = int(x_index * screen_w)
        target_y = int(y_index * screen_h)

        # Calculate distance moved
        delta = np.linalg.norm([target_x - prev_x, target_y - prev_y])
        prev_x, prev_y = target_x, target_y

        # Dynamic smoothing (faster motion = less smoothing)
        alpha = base_alpha if delta < 30 else 0.4

        # Exponential moving average smoothing
        smooth_x = (1 - alpha) * smooth_x + alpha * target_x
        smooth_y = (1 - alpha) * smooth_y + alpha * target_y

        # Dead zone filtering
        if np.linalg.norm([smooth_x - target_x, smooth_y - target_y]) > dead_zone_px:
            with lock:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)

        # Show pointer for feedback
        cv2.circle(img, (int(x_index * w), int(y_index * h)), 10, (0, 255, 255), -1)

        # Gesture detection
        dist_thumb_index = distance((x_index, y_index), (x_thumb, y_thumb))
        dist_thumb_middle = distance((x_middle, y_middle), (x_thumb, y_thumb))
        current_time = time.time()

        # LEFT CLICK (thumb + index pinch)
        if dist_thumb_index < 0.04 and current_time - last_left_click > click_cooldown:
            with lock:
                pyautogui.click()
            last_left_click = current_time
            time.sleep(0.2)

        # RIGHT CLICK (thumb + middle pinch)
        if dist_thumb_middle < 0.04 and current_time - last_right_click > click_cooldown:
            with lock:
                pyautogui.rightClick()
            last_right_click = current_time
            time.sleep(0.2)

    # Show camera feed
    cv2.imshow("Hand Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
