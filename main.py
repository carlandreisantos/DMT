import cv2
import mediapipe as mp
from feature_extraction import process_keypoints

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
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
                
                pose_keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in results_pose.pose_landmarks.landmark]
            
            if results_hands.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    
                    hand_keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
                    
                    if handedness.classification[0].label == "Left":
                        left_hand_keypoints = hand_keypoints
                    elif handedness.classification[0].label == "Right":
                        right_hand_keypoints = hand_keypoints
            

            flexion, position, rotation, roll = process_keypoints(left_hand_keypoints, right_hand_keypoints, pose_keypoints)
            print(flexion)
            cv2.imshow('MediaPipe Pose & Hands', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
