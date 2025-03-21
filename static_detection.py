import numpy as np
import json
from collections import deque

# Threshold values for comparison
thresholds = {
    "Flexion": .05,
    "Position": 0.5,
    "Rotation": 15,
}

# Load recorded gestures
with open("gestures.json", "r") as f:
    recorded_gestures = json.load(f)

# Rolling window for smoothing
window_size = 5
match_history = deque(maxlen=window_size)

# Precompute feature ranges for each gesture
gesture_ranges = {}
for gesture_name, samples in recorded_gestures.items():
    feature_ranges = {}

    for feature in ["flexion", "position", "rotation"]:
        if all(feature in sample for sample in samples):
            values = np.array([sample[feature] for sample in samples])
            feature_ranges[feature] = {
                "min": np.min(values, axis=0),
                "max": np.max(values, axis=0),
            }

    gesture_ranges[gesture_name] = feature_ranges

def is_within_range(current, feature_range, threshold):
    """Check if the current feature values are within the range plus/minus threshold."""
    if current is None or feature_range is None:
        return False
    
    current = np.array(current)
    min_range = np.array(feature_range["min"]) - threshold
    max_range = np.array(feature_range["max"]) + threshold
    
    return np.all((current >= min_range) & (current <= max_range))

def check_gesture_match(flexion, position, rotation):
    """Check if the current gesture falls within the precomputed gesture ranges."""
    for gesture_name, feature_ranges in gesture_ranges.items():
        all_match = True
        
        if "flexion" in feature_ranges and not is_within_range(flexion, feature_ranges["flexion"], thresholds["Flexion"]):
            all_match = False
        if "position" in feature_ranges and not is_within_range(position, feature_ranges["position"], thresholds["Position"]):
            all_match = False
        if "rotation" in feature_ranges and not is_within_range(rotation, feature_ranges["rotation"], thresholds["Rotation"]):
            all_match = False
        
        if all_match:
            match_history.append(gesture_name)
            return get_smooth_match()
    
    match_history.append("No Match")
    return get_smooth_match()

def get_smooth_match():
    """Return the most frequent match in the rolling window to smooth transitions."""
    if not match_history:
        return "No Match"
    
    return max(set(match_history), key=match_history.count)
