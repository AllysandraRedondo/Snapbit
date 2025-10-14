import numpy as np
from filters.overlay_utils import overlay_filter 

def apply_dog_filter(frame, landmarks, dog_ears_img, dog_nose_img, tongue_img):
    h, w, _ = frame.shape
    
    # Landmarks for positioning and rotation
    nose = landmarks.landmark[1]
    upper_lip = landmarks.landmark[13]
    lower_lip = landmarks.landmark[14]
    left = landmarks.landmark[234]
    right = landmarks.landmark[454]
    top_head = landmarks.landmark[10]

    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)
    upper_lip_y = int(upper_lip.y * h)
    lower_lip_y = int(lower_lip.y * h)
    top_head_x, top_head_y = int(top_head.x * w), int(top_head.y * h)

    # Calculate face-based angle for rotation
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))
    face_width = abs(right_x - left_x)

    if dog_ears_img is not None:
        ears_w = int(1.5 * face_width)
        ears_h = int(ears_w * dog_ears_img.shape[0] / dog_ears_img.shape[1])
        
        anchor_x = top_head_x 
        anchor_y = top_head_y - int(ears_h * 0.1)

        x1_ears = anchor_x - ears_w // 2
        y1_ears = anchor_y - ears_h
        x2_ears = x1_ears + ears_w
        y2_ears = y1_ears + ears_h
        
        frame = overlay_filter(frame, dog_ears_img, x1_ears, y1_ears, x2_ears, y2_ears, angle)

    if dog_nose_img is not None:
        nose_w = int(1.5 * face_width)
        nose_h = int(nose_w * dog_nose_img.shape[0] / dog_nose_img.shape[1])
        
        anchor_x = nose_x
        anchor_y = nose_y
        
        x1_nose = anchor_x - nose_w // 2
        y1_nose = anchor_y - nose_h // 2
        x2_nose = x1_nose + nose_w
        y2_nose = y1_nose + nose_h

        frame = overlay_filter(frame, dog_nose_img, x1_nose, y1_nose, x2_nose, y2_nose, angle)

    lip_dist = abs(lower_lip_y - upper_lip_y)
    
    # Calculate mouth openness ratio 
    mouth_openness_ratio = lip_dist / face_width if face_width > 0 else 0
    
    if mouth_openness_ratio > 0.05 and tongue_img is not None: 
        tongue_w = int(0.6 * face_width)
        tongue_h = int(tongue_w * tongue_img.shape[0] / tongue_img.shape[1])
        
        mouth_x = nose_x
        mouth_y = lower_lip_y - 10
        
        tx1 = mouth_x - tongue_w // 2
        ty1 = mouth_y
        tx2 = tx1 + tongue_w
        ty2 = ty1 + tongue_h
        
        frame = overlay_filter(frame, tongue_img, tx1, ty1, tx2, ty2, angle)
    
    return frame