import numpy as np
from filters.overlay_utils import overlay_filter 

def apply_cat_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    
    # Key Landmarks for a stable position
    nose = landmarks.landmark[1]
    left_face_boundary = landmarks.landmark[234] 
    right_face_boundary = landmarks.landmark[454]
    TOP_HEAD = landmarks.landmark[10] 

    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_x, left_y = int(left_face_boundary.x * w), int(left_face_boundary.y * h)
    right_x, right_y = int(right_face_boundary.x * w), int(right_face_boundary.y * h)
    top_head_x, top_head_y = int(TOP_HEAD.x * w), int(TOP_HEAD.y * h)

#scaling
    FACE_WIDTH_MULTIPLIER = 1.1 
    filter_w = int(FACE_WIDTH_MULTIPLIER * abs(right_x - left_x))
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    
    # Rotation angle
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))

    anchor_x = nose_x
    
    CAT_Y_OFFSET = 30 # Moves the anchor point 30 pixels down from the top of the head
    anchor_y = top_head_y + CAT_Y_OFFSET 

    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h

    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)