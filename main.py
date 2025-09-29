import cv2
import mediapipe as mp
import numpy as np
import time

from filters.overlay_utils import overlay_filter 

from filters.dog_filter import apply_dog_filter
from filters.sunglasses_filter import apply_sunglasses_filter
from filters.mustache_filter import apply_mustache_filter
from filters.cat_filter import apply_cat_filter
from filters.headband_filter import apply_headband_filter
from filters.shark_filter import apply_shark_filter



def load_image_or_none(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load image from '{path}'. Skipping this filter.")
    return img

# Load Filter Images
dog_ears_img = load_image_or_none("filter_images/dog_ears.png")
dog_nose_img = load_image_or_none("filter_images/dog_nose.png")
tongue_img = load_image_or_none("filter_images/dog_tongue.png")
mustache_img = load_image_or_none("filter_images/mustache.png")
cat_img = load_image_or_none("filter_images/cat.png")
sunglasses_img = load_image_or_none("filter_images/sunglasses.png")
headband_img = load_image_or_none("filter_images/headband.png")
shark_img = load_image_or_none("filter_images/shark.png")

# List of all filters 
filters_list = [
    ("Dog", dog_ears_img, apply_dog_filter, dog_nose_img, tongue_img),         
    ("Mustache", mustache_img, apply_mustache_filter),                         
    ("Cat", cat_img, apply_cat_filter),                                        
    ("Sunglasses", sunglasses_img, apply_sunglasses_filter),                   
    ("Headband", headband_img, apply_headband_filter),                         
    ("Shark", shark_img, apply_shark_filter),                                  
]

# mapping of UI position to filter index 
active_filters = {} 
active_ui_icons = [] 

filter_index_counter = 0
ui_position_counter = 0

for filter_data in filters_list:
    icon_img = filter_data[1]
    
    # Only adds filters that successfully loaded an icon image
    if icon_img is not None:
        active_ui_icons.append(icon_img)
        active_filters[ui_position_counter] = filter_index_counter
        ui_position_counter += 1
        
    filter_index_counter += 1

current_filter_index = active_filters.get(0, 0) 
current_ui_position = 0

last_click_time = 0
CLICK_DEBOUNCE_TIME = 250 

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Hand Gesture Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Exiting.")
    exit()


def draw_ui(frame):
    global current_ui_position
    ui_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], ui_height), (0, 0, 0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    icon_size = 50
    gap = 20
    start_x = 20
    
    for i, icon in enumerate(active_ui_icons): 
        
        x1 = start_x + i * (icon_size + gap) 
        y1 = 15
        
        if i == current_ui_position: 
            cv2.rectangle(frame, (x1 - 5, y1 - 5), (x1 + icon_size + 5, y1 + icon_size + 5), (0, 255, 0), 2)
        
        frame = overlay_filter(frame, icon, x1, y1, x1 + icon_size, y1 + icon_size, 0)
    
    return frame


def mouse_callback(event, x, y, flags, param):
    global current_filter_index, current_ui_position, last_click_time
    current_time = int(time.time() * 1000)
    
    if event == cv2.EVENT_LBUTTONDOWN and (current_time - last_click_time > CLICK_DEBOUNCE_TIME):
        last_click_time = current_time
        icon_size = 50
        gap = 20
        start_x = 20
        ui_height = 80
        
        if 0 < y < ui_height:
            for i, icon in enumerate(active_ui_icons):
                x1 = start_x + i * (icon_size + gap)
                x2 = x1 + icon_size
                if x1 <= x <= x2:
                    current_ui_position = i
                    current_filter_index = active_filters[current_ui_position]
                    print(f"Filter changed to: {filters_list[current_filter_index][0]} (UI Pos: {i}, Filter Index: {current_filter_index})")
                    break

cv2.namedWindow("AR Face Filters")
cv2.setMouseCallback("AR Face Filters", mouse_callback)

# Main Loop 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame from the webcam.")
        break

    frame = cv2.flip(frame, 1)
    original_frame = frame.copy() 
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb) 
      

    # Apply Face Filter
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            filter_data = filters_list[current_filter_index]
            filter_name = filter_data[0]
            filter_function = filter_data[2]
            
            if filter_name == "Dog":
                dog_ears_img = filter_data[1]
                dog_nose_img = filter_data[3]
                tongue_image = filter_data[4]
                
                if dog_ears_img is not None or dog_nose_img is not None:
                    frame = filter_function(frame, face_landmarks, dog_ears_img, dog_nose_img, tongue_image)
            
            else:
                # All other generic filters 
                filter_image = filter_data[1]
                
                if filter_image is not None and filter_function is not None:
                    frame = filter_function(frame, face_landmarks, filter_image)

    # Hand Masking
    if hand_results.multi_hand_landmarks:
        mask_layer = np.zeros((h, w, 4), dtype=np.uint8)
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            points = []
            for i in [0, 5, 9, 13, 17, 8, 12, 16, 20]:
                point = hand_landmarks.landmark[i]
                points.append((int(point.x * w), int(point.y * h)))
            
            hull_points = cv2.convexHull(np.array(points, dtype=np.int32))
            cv2.fillPoly(mask_layer, [hull_points], (255, 255, 255, 255))
            
        raw_mask = mask_layer[:, :, 3]
        blur_mask = cv2.GaussianBlur(raw_mask, (21, 21), 0)
        alpha_mask = blur_mask / 255.0

        filtered_frame_float = frame.astype(float)
        original_frame_float = original_frame.astype(float)
        
        filter_part = filtered_frame_float * (1 - alpha_mask[:, :, np.newaxis])
        hand_part = original_frame_float * alpha_mask[:, :, np.newaxis]
        
        frame = cv2.add(filter_part, hand_part).astype(np.uint8)


    frame = draw_ui(frame)
    cv2.imshow("AR Face Filters", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()