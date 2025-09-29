import numpy as np
from filters.overlay_utils import overlay_filter 

def apply_sunglasses_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    
    # Landmarks for sizing and positioning
    nose = landmarks.landmark[1]
    left_eye = landmarks.landmark[133]
    right_eye = landmarks.landmark[362]
    
    # Convert normalized coordinates to pixel values
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

    # Scaling
    FILTER_WIDTH_MULTIPLIER = 5.5 
    eye_distance = abs(right_eye_x - left_eye_x)
    
    filter_w = int(FILTER_WIDTH_MULTIPLIER * eye_distance)
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    
    # Calculates the angle of the eye line
    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    angle = -np.degrees(np.arctan2(dy, dx))

    #Positioning 
    anchor_x = nose_x
    anchor_y = int((left_eye_y + right_eye_y) / 2)
    
    # Define the bounding box
    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h
    
    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)