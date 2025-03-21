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
    position = []

    pose_keypoints = normalize_pose_keypoints(pose_keypoints)  # Normalize pose

    left_shoulder = pose_keypoints[11] if len(pose_keypoints) > 11 else (0, 0, 0)
    right_shoulder = pose_keypoints[12] if len(pose_keypoints) > 12 else (0, 0, 0)

    # Process left hand if available
    if left_hand_keypoints:
        flexion.extend(get_flexions(left_hand_keypoints))
        rotation.append(calculate_rotation(left_hand_keypoints[0], left_hand_keypoints[5]))
        position.append(calculate_position(left_hand_keypoints[0], left_shoulder))
    else:
        flexion.extend([0] * 9)
        rotation.append(0)
        position.append((0, 0, 0))

    # Process right hand if available (add to the same features)
    if right_hand_keypoints:
        flexion.extend(get_flexions(right_hand_keypoints))
        rotation.append(calculate_rotation(right_hand_keypoints[0], right_hand_keypoints[5]))
        position.append(calculate_position(right_hand_keypoints[0], right_shoulder))
    else:
        flexion.extend([0] * 9)
        rotation.append(0)
        position.append((0, 0, 0))

    # Flatten position (since it contains tuples)
    position = [coord for pos in position for coord in pos]

    return flexion, position, rotation

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def normalize_keypoints(keypoints):
    wrist = keypoints[0]
    index_base = keypoints[5]
    
    if wrist == (0, 0, 0) or index_base == (0, 0, 0):
        return keypoints  # Return unmodified if normalization is not possible
    
    scale_factor = euclidean_distance(wrist, index_base)
    
    if scale_factor == 0:
        return keypoints  # Avoid division by zero
    
    return [(x / scale_factor, y / scale_factor, z / scale_factor) if (x, y, z) != (0, 0, 0) else (0, 0, 0) for x, y, z in keypoints]

def normalize_pose_keypoints(pose_keypoints):
    if len(pose_keypoints) < 13:
        return pose_keypoints  # Not enough keypoints to normalize
    
    left_shoulder = pose_keypoints[11]
    right_shoulder = pose_keypoints[12]
    
    if left_shoulder == (0, 0, 0) or right_shoulder == (0, 0, 0):
        return pose_keypoints  # Avoid normalization if shoulders are missing
    
    scale_factor = euclidean_distance(left_shoulder, right_shoulder)
    
    if scale_factor == 0:
        return pose_keypoints  # Avoid division by zero
    
    return [(x / scale_factor, y / scale_factor, z / scale_factor) if (x, y, z) != (0, 0, 0) else (0, 0, 0) for x, y, z in pose_keypoints]

def get_flexions(keypoints):
    keypoints = normalize_keypoints(keypoints)  # Normalize before computing distances
    distance_pairs = [(4, 0), (8, 0), (12, 0), (16, 0), (20, 0), (4, 8), (8, 12), (12, 16), (16, 20)]
    flexions = []
    
    for a, b in distance_pairs:
        if keypoints[a] != (0, 0, 0) and keypoints[b] != (0, 0, 0):
            distance = euclidean_distance(keypoints[a], keypoints[b])
        else:
            distance = 0  
        flexions.append(distance)
    
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
