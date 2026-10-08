# inference_mediapipe.py

# ViT가 아닌 MediaPipe를 활용하여 동영상 인식해보기

import os
import time
import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

CLASS_COLORS = {
    'ROCK': (0, 0, 255),      # Red
    'SCISSORS': (0, 255, 0),  # Green
    'PAPER': (255, 0, 0)      # Blue
}

def classify_rps(landmarks, handedness_label):
    fingers_open = []

    # 1. Thumb (x-coord based)
    if handedness_label == 'Right':
        thumb_is_open = landmarks[4].x < landmarks[3].x
    else:
        thumb_is_open = landmarks[4].x > landmarks[3].x
    fingers_open.append(thumb_is_open)

    # 2. Index, Middle, Ring, Pinky (y-coord based)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        is_open = landmarks[tip].y < landmarks[pip].y
        fingers_open.append(is_open)

    index_open, middle_open, ring_open, pinky_open = fingers_open[1:]

    # 3. Gesture Classification
    # PAPER: all 4 fingers open
    if index_open and middle_open and ring_open and pinky_open:
        return 'PAPER', fingers_open

    # SCISSORS: index and middle open, ring and pinky closed
    if index_open and middle_open and (not ring_open) and (not pinky_open):
        return 'SCISSORS', fingers_open

    # ROCK: all 4 fingers closed
    if (not index_open) and (not middle_open) and (not ring_open) and (not pinky_open):
        return 'ROCK', fingers_open

    # Fallback by count
    open_count = sum(fingers_open)
    if open_count >= 4:
        return 'PAPER', fingers_open
    elif open_count <= 1:
        return 'ROCK', fingers_open
    else:
        return 'SCISSORS', fingers_open

def draw_hud(img, current_gesture, fingers_open, fps):
    overlay = img.copy()
    panel_x1, panel_y1, panel_x2, panel_y2 = 10, 10, 360, 150
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
    alpha = 0.70
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (panel_x1, panel_y1), (panel_x2, panel_y2), (100, 100, 100), 1)

    # FPS
    cv2.putText(img, f"FPS: {fps:.1f}", (panel_x1 + 15, panel_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Detected Gesture
    color = CLASS_COLORS.get(current_gesture, (200, 200, 200))
    cv2.putText(img, f"DETECTED: {current_gesture}", (panel_x1 + 15, panel_y1 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Finger Status (T: Thumb, I: Index, M: Middle, R: Ring, P: Pinky)
    names = ['T', 'I', 'M', 'R', 'P']
    status_str = " ".join([f"{n}:{'O' if f else 'X'}" for n, f in zip(names, fingers_open)]) if fingers_open else "No Hand"
    cv2.putText(img, f"Fingers: {status_str}", (panel_x1 + 15, panel_y1 + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    video_path = 'tests/test_video.mp4'
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps is None:
        video_fps = 30.0
    frame_delay = max(1, int(1000 / video_fps))

    prev_time = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Video playback completed.")
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_gesture = "NONE"
            fingers_open = []

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    handedness_label = handedness.classification[0].label
                    current_gesture, fingers_open = classify_rps(hand_landmarks.landmark, handedness_label)
                    active_color = CLASS_COLORS.get(current_gesture, (255, 255, 255))

                    # 1. 21 Landmarks skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 2. Label above wrist
                    wrist_pt = hand_landmarks.landmark[0]
                    cx, cy = int(wrist_pt.x * w), int(wrist_pt.y * h)
                    cv2.putText(frame, current_gesture, (cx - 40, cy + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, active_color, 2, cv2.LINE_AA)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # HUD rendering
            draw_hud(frame, current_gesture, fingers_open, fps)

            cv2.imshow("MediaPipe Gesture Tracking", frame)

            elapsed_ms = int((time.time() - loop_start) * 1000)
            wait_time = max(1, frame_delay - elapsed_ms)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()