import cv2
import mediapipe as mp
import json
import os
from feature_extraction import process_keypoints

def delete_gesture():
    if not os.path.exists("gestures.json"):
        print("No gesture data found.")
        return
    
    with open("gestures.json", "r") as f:
        gestures_data = json.load(f)
    
    if not gestures_data:
        print("No gestures to delete.")
        return
    
    print("Existing gestures:")
    for gesture in gestures_data.keys():
        print(f"- {gesture}")
    
    gesture_name = input("Enter the name of the gesture to delete: ").strip()
    
    if gesture_name in gestures_data:
        del gestures_data[gesture_name]
        with open("gestures.json", "w") as f:
            json.dump(gestures_data, f, indent=4)
        print(f"Gesture '{gesture_name}' deleted successfully.")
    else:
        print("Gesture not found.")

def collect_gesture_data():
    gesture_name = input("Enter gesture name: ")
    samples = []
    sample_count = 0
    total_samples = 50

    collect_flexion = input("Collect flexion features? (y/n): ").strip().lower() == 'y'
    collect_position = input("Collect position features? (y/n): ").strip().lower() == 'y'
    collect_rotation = input("Collect rotation features? (y/n): ").strip().lower() == 'y'

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened() and sample_count < total_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_pose = pose.process(frame_rgb)
            results_hands = hands.process(frame_rgb)

            left_hand_keypoints = [(0, 0, 0)] * 21
            right_hand_keypoints = [(0, 0, 0)] * 21
            pose_keypoints = [(0, 0, 0)] * 33

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                pose_keypoints = [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark]

            if results_hands.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    hand_keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    if handedness.classification[0].label == "Left":
                        left_hand_keypoints = hand_keypoints
                    elif handedness.classification[0].label == "Right":
                        right_hand_keypoints = hand_keypoints

            flexion, position, rotation = process_keypoints(left_hand_keypoints, right_hand_keypoints, pose_keypoints)
            selected_features = {}
            if collect_flexion:
                selected_features["flexion"] = flexion
            if collect_position:
                selected_features["position"] = position
            if collect_rotation:
                selected_features["rotation"] = rotation
            
            cv2.imshow('MediaPipe Pose & Hands', frame)
            key = cv2.waitKey(5) & 0xFF
            
            if key == 32:
                samples.append(selected_features)
                sample_count += 1
                print(f"Sample {sample_count}/{total_samples} collected")
            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    if os.path.exists("gestures.json"):
        with open("gestures.json", "r") as f:
            gestures_data = json.load(f)
    else:
        gestures_data = {}

    gestures_data[gesture_name] = samples

    with open("gestures.json", "w") as f:
        json.dump(gestures_data, f, indent=4)
    
    print(f"Gesture data saved to gestures.json")

if __name__ == "__main__":
    action = input("Do you want to add a new gesture or delete one? (add/delete): ").strip().lower()
    if action == "add":
        collect_gesture_data()
    elif action == "delete":
        delete_gesture()
    else:
        print("Invalid option. Exiting.")