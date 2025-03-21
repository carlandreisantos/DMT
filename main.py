import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque, Counter
from feature_extraction import process_keypoints
from static_detection import check_gesture_match

GESTURE_COMBINATIONS = {
    ("mabuti", "kamusta"): "magandang umaga",
    ("salamat1", "resting"): "salamat",
    ("mabuti", "c"): "magandang gabi"
}

EXCLUDED_WORDS = {"resting", "mabuti", "salamat1"}
FILTERED_HISTORY_LIMIT = 10
GESTURE_HOLD_THRESHOLD = 5  # Number of frames the gesture must be held

def process_filtered_history(raw_history):
    filtered_history = deque(maxlen=FILTERED_HISTORY_LIMIT)
    temp_sequence = deque(maxlen=2)
    for gesture in raw_history:
        temp_sequence.append(gesture)
        gesture_tuple = tuple(temp_sequence)
        if gesture_tuple in GESTURE_COMBINATIONS:
            combined_word = GESTURE_COMBINATIONS[gesture_tuple]
            filtered_history.append(combined_word)
            temp_sequence.clear()
        else:
            if gesture not in EXCLUDED_WORDS:
                filtered_history.append(gesture)
    return filtered_history

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition UI")

        self.label = tk.Label(root, text="Sign to Text", font=("Arial", 14))
        self.label.pack(pady=5)

        self.textbox = tk.Text(root, height=2, width=50, font=("Arial", 12))
        self.textbox.tag_configure("center", justify="center")
        self.textbox.pack(pady=5)

        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

        self.speech_label = tk.Label(root, text="Speech to Text", font=("Arial", 14))
        self.speech_label.pack(pady=5)

        self.speech_textbox = tk.Text(root, height=2, width=50, font=("Arial", 12))
        self.speech_textbox.pack(pady=5)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        self.speech_button = tk.Button(button_frame, text="Start Speech to Text", font=("Arial", 12))
        self.speech_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(button_frame, text="Reset History", font=("Arial", 12), command=self.reset_history)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.cap = cv2.VideoCapture(0)
        self.raw_gesture_history = deque(maxlen=10)
        self.filtered_gesture_history = deque(maxlen=FILTERED_HISTORY_LIMIT)
        self.gesture_counter = Counter()
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.current_sign = ""
        self.update_frame()

    def reset_history(self):
        self.raw_gesture_history.clear()
        self.filtered_gesture_history.clear()
        self.textbox.delete("1.0", tk.END)
        self.gesture_counter.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)

        left_hand_keypoints = [(0, 0, 0)] * 21
        right_hand_keypoints = [(0, 0, 0)] * 21
        pose_keypoints = [(0, 0, 0)] * 33

        if results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            pose_keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in results_pose.pose_landmarks.landmark]

        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
                if handedness.classification[0].label == "Left":
                    left_hand_keypoints = hand_keypoints
                elif handedness.classification[0].label == "Right":
                    right_hand_keypoints = hand_keypoints

        flexion, position, rotation = process_keypoints(left_hand_keypoints, right_hand_keypoints, pose_keypoints)
        gesture_match = check_gesture_match(flexion, position, rotation)

        if gesture_match and gesture_match != "No Match":
            self.gesture_counter[gesture_match] += 1
            if self.gesture_counter[gesture_match] >= GESTURE_HOLD_THRESHOLD:
                if not self.raw_gesture_history or self.raw_gesture_history[-1] != gesture_match:
                    self.raw_gesture_history.append(gesture_match)
                    self.filtered_gesture_history = process_filtered_history(self.raw_gesture_history)
                    self.current_sign = gesture_match
                self.gesture_counter.clear()
        else:
            self.gesture_counter.clear()

        filtered_history_text = "  ".join(self.filtered_gesture_history)
        self.textbox.delete("1.0", tk.END)
        self.textbox.insert(tk.END, filtered_history_text)
        self.textbox.tag_add("center", "1.0", "end")

        cv2.putText(frame_rgb, f'{self.current_sign}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
