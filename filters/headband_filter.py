import numpy as np
from filters.overlay_utils import overlay_filter 

def apply_headband_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    
    # Landmarks for headband placement
    top_head = landmarks.landmark[10] 
    left = landmarks.landmark[234]    
    right = landmarks.landmark[454]    
    
    top_head_x, top_head_y = int(top_head.x * w), int(top_head.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)

    # Scaling
    face_width = abs(right_x - left_x)
    filter_w = int(2.2 * face_width) 
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    
    # Angle
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))

    # Positioning
    anchor_x = top_head_x
    vertical_offset = int(filter_h * 0.0) 
    anchor_y = top_head_y + vertical_offset

    # Bounding box coordinates
    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h

    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)