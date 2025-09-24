import cv2
import mediapipe as mp
import numpy as np
import time

from filters import overlay_filter, apply_dog_filter, apply_sunglasses_filter, apply_beard_filter, apply_cat_filter, apply_mask_filter

def load_image_or_none(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load image from '{path}'. Skipping this filter.")
    return img

# all filters 
dog_img = load_image_or_none("filters/lab.png")
tongue_img = load_image_or_none("filters/tongue.png")
beard_img = load_image_or_none("filters/beard.png")
cat_img = load_image_or_none("filters/cat.png")
sunglasses_img = load_image_or_none("filters/sunglasses.png")
mask_img = load_image_or_none("filters/mask.png")

ui_icons = [
    load_image_or_none("filters/lab.png"),
    load_image_or_none("filters/beard.png"),
    load_image_or_none("filters/cat.png"),
    load_image_or_none("filters/sunglasses.png"),
    load_image_or_none("filters/mask.png")
]

# Dictionary of filters
filters = {
    0: ("Dog", dog_img, apply_dog_filter, tongue_img),
    1: ("Beard", beard_img, apply_beard_filter),
    2: ("Cat", cat_img, apply_cat_filter),
    3: ("Sunglasses", sunglasses_img, apply_sunglasses_filter),
    4: ("Mask", mask_img, apply_mask_filter),
}
current_filter_index = 0
last_click_time = 0
CLICK_DEBOUNCE_TIME = 250  

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Exiting.")
    exit()

def draw_ui(frame):
    global current_filter_index
    ui_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], ui_height), (0, 0, 0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    icon_size = 50
    gap = 20
    start_x = 20
    for i, icon in enumerate(ui_icons):
        if icon is None:
            continue
        
        x1 = start_x + i * (icon_size + gap)
        y1 = 15
        
        if i == current_filter_index:
            cv2.rectangle(frame, (x1 - 5, y1 - 5), (x1 + icon_size + 5, y1 + icon_size + 5), (0, 255, 0), 2)
        
        frame = overlay_filter(frame, icon, x1, y1, x1 + icon_size, y1 + icon_size, 0)
    
    return frame

def mouse_callback(event, x, y, flags, param):
    global current_filter_index, last_click_time
    current_time = int(time.time() * 1000)
    
    if event == cv2.EVENT_LBUTTONDOWN and (current_time - last_click_time > CLICK_DEBOUNCE_TIME):
        last_click_time = current_time
        icon_size = 50
        gap = 20
        start_x = 20
        ui_height = 80
        
        if 0 < y < ui_height:
            for i, icon in enumerate(ui_icons):
                if icon is None:
                    continue
                x1 = start_x + i * (icon_size + gap)
                x2 = x1 + icon_size
                if x1 <= x <= x2:
                    current_filter_index = i
                    print(f"Filter changed to: {filters[current_filter_index][0]}")
                    break

cv2.namedWindow("AR Face Filters")
cv2.setMouseCallback("AR Face Filters", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame from the webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            filter_data = filters[current_filter_index]
            filter_name = filter_data[0]
            filter_image = filter_data[1]
            filter_function = filter_data[2]
            
            if filter_function and filter_image is not None:
                if filter_name == "Dog" and len(filter_data) > 3:
                    tongue_image = filter_data[3]
                    frame = filter_function(frame, face_landmarks, filter_image, tongue_image)
                else:
                    frame = filter_function(frame, face_landmarks, filter_image)

    frame = draw_ui(frame)
    cv2.imshow("AR Face Filters", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()