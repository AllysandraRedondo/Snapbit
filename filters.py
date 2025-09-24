import cv2
import numpy as np

def _rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def overlay_filter(frame, filter_img, x1, y1, x2, y2, angle):
    overlay_w = x2 - x1
    overlay_h = y2 - y1
    if overlay_w <= 0 or overlay_h <= 0 or filter_img is None:
        return frame

    try:
        resized = cv2.resize(filter_img, (overlay_w, overlay_h))
        rotated = _rotate_image(resized, angle)
    except cv2.error as e:
        print(f"Resize/Rotate Error: {e}")
        return frame

    frame_h, frame_w = frame.shape[:2]
    x1_clipped, y1_clipped = max(0, x1), max(0, y1)
    x2_clipped, y2_clipped = min(frame_w, x2), min(frame_h, y2)
    cropped_w = x2_clipped - x1_clipped
    cropped_h = y2_clipped - y1_clipped

    if cropped_w <= 0 or cropped_h <= 0:
        return frame

    filter_cropped = rotated[0:cropped_h, 0:cropped_w]

    if filter_cropped.shape[2] == 4:
        alpha = filter_cropped[:, :, 3] / 255.0
        for c in range(3):
            frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped, c] = (
                alpha * filter_cropped[:, :, c] +
                (1 - alpha) * frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped, c]
            )
    return frame

def apply_dog_filter(frame, landmarks, filter_img, tongue_img):
    h, w, _ = frame.shape
    nose = landmarks.landmark[1]
    upper_lip = landmarks.landmark[13]
    lower_lip = landmarks.landmark[14]
    left = landmarks.landmark[234]
    right = landmarks.landmark[454]

    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)
    upper_lip_y = int(upper_lip.y * h)
    lower_lip_y = int(lower_lip.y * h)

    filter_w = int(1.2 * abs(right_x - left_x))
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))

    x1 = nose_x - filter_w // 2
    y1 = nose_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h
    frame = overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)

    lip_dist = abs(lower_lip_y - upper_lip_y)
    if lip_dist > 15 and tongue_img is not None:
        tongue_w = int(0.6 * filter_w)
        tongue_h = int(tongue_w * tongue_img.shape[0] / tongue_img.shape[1])
        mouth_x = nose_x
        mouth_y = lower_lip_y
        tx1 = mouth_x - tongue_w // 2
        ty1 = mouth_y
        tx2 = tx1 + tongue_w
        ty2 = ty1 + tongue_h
        frame = overlay_filter(frame, tongue_img, tx1, ty1, tx2, ty2, angle)
    
    return frame

def apply_sunglasses_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    nose = landmarks.landmark[1]
    left_eye = landmarks.landmark[133]
    right_eye = landmarks.landmark[362]
    
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

    filter_w = int(1.4 * abs(right_eye_x - left_eye_x))
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    angle = -np.degrees(np.arctan2(dy, dx))

    anchor_x = nose_x
    anchor_y = int((left_eye_y + right_eye_y) / 2)
    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h
    
    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)

def apply_beard_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    chin = landmarks.landmark[152]
    upper_lip = landmarks.landmark[13]
    left = landmarks.landmark[234]
    right = landmarks.landmark[454]

    chin_x, chin_y = int(chin.x * w), int(chin.y * h)
    upper_lip_y = int(upper_lip.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)

    filter_w = int(1.1 * abs(right_x - left_x))
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))

    anchor_x = chin_x
    anchor_y = (chin_y + upper_lip_y) // 2
    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h

    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)

def apply_cat_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    nose = landmarks.landmark[1]
    left = landmarks.landmark[234]
    right = landmarks.landmark[454]

    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)

    filter_w = int(1.1 * abs(right_x - left_x))
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))

    anchor_x = nose_x
    anchor_y = nose_y
    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h

    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)
    
def apply_mask_filter(frame, landmarks, filter_img):
    h, w, _ = frame.shape
    chin = landmarks.landmark[152]
    upper_lip = landmarks.landmark[13]
    left_cheek = landmarks.landmark[119]
    right_cheek = landmarks.landmark[348]

    chin_x, chin_y = int(chin.x * w), int(chin.y * h)
    upper_lip_y = int(upper_lip.y * h)
    left_x, left_y = int(left_cheek.x * w), int(left_cheek.y * h)
    right_x, right_y = int(right_cheek.x * w), int(right_cheek.y * h)

    filter_w = int(1.1 * abs(right_x - left_x))
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])
    dx = right_x - left_x
    dy = right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))

    anchor_x = (right_x + left_x) // 2
    anchor_y = (chin_y + upper_lip_y) // 2
    x1 = anchor_x - filter_w // 2
    y1 = anchor_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h

    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)