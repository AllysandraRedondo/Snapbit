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
    # Clip coordinates to be within the frame
    x1_clipped, y1_clipped = max(0, x1), max(0, y1)
    x2_clipped, y2_clipped = min(frame_w, x2), min(frame_h, y2)
    cropped_w = x2_clipped - x1_clipped
    cropped_h = y2_clipped - y1_clipped

    if cropped_w <= 0 or cropped_h <= 0:
        return frame

    # Calculate offset for filter image cropping 
    filter_crop_x_start = x1_clipped - x1
    filter_crop_y_start = y1_clipped - y1
    
    filter_cropped = rotated[
        filter_crop_y_start:filter_crop_y_start + cropped_h, 
        filter_crop_x_start:filter_crop_x_start + cropped_w
    ]

    # Ensure the cropped filter has 4 channels for alpha blending
    if filter_cropped.shape[2] == 4:
        alpha = filter_cropped[:, :, 3] / 255.0
        for c in range(3):
            fg = filter_cropped[:, :, c]
            bg = frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped, c]
            
            frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped, c] = (
                alpha * fg + (1 - alpha) * bg
            )
    return frame