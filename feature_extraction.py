import numpy as np

def calculate_position(wrist, shoulder):

    if wrist == (0, 0, 0) or shoulder == (0, 0, 0):
        return (0, 0, 0)  # Default if no detection
    
    x_rel = wrist[0] - shoulder[0]
    y_rel = wrist[1] - shoulder[1]
    z_rel = wrist[2] - shoulder[2]

    return (x_rel, y_rel, z_rel)  

def process_keypoints(left_hand_keypoints, right_hand_keypoints, pose_keypoints):
    flexion = []
    rotation = []
    roll = []
    position = []

    
    left_shoulder = pose_keypoints[11] if len(pose_keypoints) > 11 else (0, 0, 0)
    right_shoulder = pose_keypoints[12] if len(pose_keypoints) > 12 else (0, 0, 0)

    
    if left_hand_keypoints:
        flexion.extend(get_flexions(left_hand_keypoints))
        rotation.append(calculate_rotation(left_hand_keypoints[0], left_hand_keypoints[5]))
        roll.append(calculate_roll(left_hand_keypoints[0], left_hand_keypoints[17]))
        position.append(calculate_position(left_hand_keypoints[0], left_shoulder))
    else:
        flexion.extend([0] * 5)
        rotation.append(0)
        roll.append(0)
        position.append((0, 0, 0))

    
    if right_hand_keypoints:
        flexion.extend(get_flexions(right_hand_keypoints))
        rotation.append(calculate_rotation(right_hand_keypoints[0], right_hand_keypoints[5]))
        roll.append(calculate_roll(right_hand_keypoints[0], right_hand_keypoints[17]))
        position.append(calculate_position(right_hand_keypoints[0], right_shoulder))
    else:
        flexion.extend([0] * 5)
        rotation.append(0)
        roll.append(0)
        position.append((0, 0, 0))

    return flexion, position, rotation, roll

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)


    ab = a - b
    cb = c - b

    cos_theta = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) 
    return np.degrees(angle)

def get_flexions(keypoints):

    finger_indices = [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    flexions = []

    for i, (a, b, c) in enumerate(finger_indices):
        if keypoints[a] != (0, 0, 0) and keypoints[b] != (0, 0, 0) and keypoints[c] != (0, 0, 0):
            angle = calculate_angle(keypoints[a], keypoints[b], keypoints[c])
        else:
            angle = 0  
        flexions.append(angle)

    return flexions

def calculate_rotation(wrist, index_base):
    if wrist == (0, 0, 0) or index_base == (0, 0, 0):  
        return 0  

    dx = index_base[0] - wrist[0]
    dy = index_base[1] - wrist[1]

    angle = np.degrees(np.arctan2(dy, dx))

    return angle 


def calculate_roll(wrist, pinky_base):
    if wrist == (0, 0, 0) or pinky_base == (0, 0, 0):
        return 0  

    dx = pinky_base[0] - wrist[0]
    dz = pinky_base[2] - wrist[2]  
    
    angle = np.degrees(np.arctan2(dz, dx))

    return angle  