import numpy as np
from filters.overlay_utils import overlay_filter 

def apply_shark_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    
    shark_scale = 3.2
    
    # Landmarks for face center/orientation
    nose = landmarks.landmark[1]
    left = landmarks.landmark[234]     
    right = landmarks.landmark[454]    
    
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)

    # Scaling
    face_width = abs(right_x - left_x)
    
    filter_w = int(face_width * shark_scale) 
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])

    # Calculating Angle
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))

    anchor_x = nose_x
    
    vertical_offset = int(face_width * .2) 
    anchor_y = nose_y - vertical_offset

    # Calculate final bounding box coordinates
    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h

    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)