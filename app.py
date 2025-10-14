from flask import Flask, render_template, Response, request, url_for, redirect, jsonify
import cv2
import mediapipe as mp
import numpy as np
from flask_mail import Mail, Message
import os
import time
import base64
import mysql.connector 
import threading
import webview

app = Flask(__name__)

# --------------------- ROUTES -------------------------------
@app.route("/")
def welcome():
    global bg_index, current_filter
    bg_index = None       # reset background
    current_filter = None # reset filter
    return render_template("welcome.html")

@app.route("/instructions")
def instruction():
    return render_template("instructions.html")

@app.route("/template")
def template():
    return render_template("template.html")

@app.route("/camera")
def camera():
    template = request.args.get("template")
    return render_template("camera.html", template=template)

@app.route("/design")
def design():
    return render_template("design.html")

@app.route("/waiting")
def waiting():
    return render_template("waiting.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/email")
def email():
    return render_template("email.html")

@app.route("/thankyou")
def thankyou():
    return render_template("thankyou.html")

# --------------------- BACKGROUND -------------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

background_colors = [
    ("Remove", None),
    ("Light Pink", (203, 192, 255)),
    ("Light Blue", (255, 230, 200)),
    ("Light Violet", (238, 130, 238)),
    ("Mint Green", (189, 252, 201)),
    ("Soft Yellow", (255, 255, 200))
]

bg_index = None   # default: no background replacement

@app.route('/set_background/<int:index>', methods=['POST'])
def set_background(index):
    global bg_index
    if 0 <= index < len(background_colors):
        bg_index = index
        return "Background updated"
    return "Invalid background", 400

# --------------------- FILTERS -------------------------------
from filters.dog_filter import apply_dog_filter
from filters.sunglasses_filter import apply_sunglasses_filter
from filters.mustache_filter import apply_mustache_filter
from filters.cat_filter import apply_cat_filter
from filters.headband_filter import apply_headband_filter
from filters.shark_filter import apply_shark_filter

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

dog_ears_img = load_image("static/Images/filters/dog_ears.png")
dog_nose_img = load_image("static/Images/filters/dog_nose.png")
tongue_img = load_image("static/Images/filters/dog_tongue.png")
mustache_img = load_image("static/Images/filters/mustache.png")
cat_img = load_image("static/Images/filters/cat.png")
sunglasses_img = load_image("static/Images/filters/sunglasses.png")
headband_img = load_image("static/Images/filters/headband.png")
shark_img = load_image("static/Images/filters/shark.png")

filters_list = {
    "dog": lambda frame, face: apply_dog_filter(frame, face, dog_ears_img, dog_nose_img, tongue_img),
    "mustache": lambda frame, face: apply_mustache_filter(frame, face, mustache_img),
    "cat": lambda frame, face: apply_cat_filter(frame, face, cat_img),
    "sunglasses": lambda frame, face: apply_sunglasses_filter(frame, face, sunglasses_img),
    "headband": lambda frame, face: apply_headband_filter(frame, face, headband_img),
    "shark": lambda frame, face: apply_shark_filter(frame, face, shark_img),
}

current_filter = None  # no filter by default

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------- MERGED CAMERA FEED -------------------------------
cap = cv2.VideoCapture(0)

def generate_frames():
    global bg_index, current_filter


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- Background Replacement ----------
        results_bg = segment.process(rgb)
        if bg_index is not None:
            bg_color = background_colors[bg_index][1]
            if bg_color is not None:
                mask = results_bg.segmentation_mask
                mask = cv2.GaussianBlur(mask, (11, 11), 0)
                mask = np.clip(mask, 0, 1)
                background = np.full_like(frame, bg_color, dtype=np.uint8)
                mask_3ch = cv2.merge([mask, mask, mask])
                frame = (frame * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

        # ---------- Face Filters ----------
        results_face = face_mesh.process(rgb)
        if results_face.multi_face_landmarks and current_filter in filters_list:
            for face_landmarks in results_face.multi_face_landmarks:
                frame = filters_list[current_filter](frame, face_landmarks)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_filter/<name>')
def set_filter(name):
    global current_filter
    if name in filters_list:
        current_filter = name
    else:
        current_filter = None
    return "OK"

# --------------------- EMAIL -------------------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = "snapbitspicture@gmail.com"
app.config['MAIL_PASSWORD'] = "tlcuzdiuzomkibat"
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

@app.route("/submit", methods=["POST"])
def submit():
    email = request.form["email"]
    photo_path = os.path.join("static", "photos", "last_photo.png")

    msg = Message(
        subject="SnapBits Photo",
        sender=app.config['MAIL_USERNAME'],
        recipients=[email]
    )
    msg.body = "Hi! This is your photo. Thank you for trying Snapbits."

    with app.open_resource(photo_path) as fp:
        msg.attach("photo.png", "image/png", fp.read())

    try:
        mail.send(msg)
        print("✅ Email sent successfully with attachment!")
    except Exception as e:
        print("❌ Email Error:", e)

    return redirect(url_for("thankyou"))

@app.route("/save_photo", methods=["POST"])
def save_photo():
    data = request.get_json()
    image_data = data["image"].split(",")[1]

    if not os.path.exists("static/photos"):
        os.makedirs("static/photos")

    filename = "last_photo.png"
    filepath = os.path.join("static/photos", filename)

    with open(filepath, "wb") as f:
        f.write(base64.b64decode(image_data))

    return {"filename": filename}

# ----------------------------- FEEDBACK ---------------------------------
def get_db(): 
    return mysql.connector.connect(
        host='localhost', 
        user='root',
        password='',
        database='snapbits_db'
    )

@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json() 
    rating = data['rating']

    try: 
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback_tbl (rating) VALUES (%s)", 
            (rating,)
        )
        conn.commit()
        cursor.close()
        conn.close()

        print(f"Rating is saved: {rating} stars")
        return jsonify({"message": "Thank you for your feedback!"})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to save the rating"})

# --------------------- START -------------------------------
def start_flask():
    """Run Flask in a background thread"""
    app.run(debug=False, port=5000, use_reloader=False)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()

    import time
    time.sleep(3)  # Wait for Flask to boot completely

    print("🚀 Launching SnapBits Photo Booth window...")
    webview.create_window("SnapBits Photo Booth", "http://127.0.0.1:5000")
    webview.start()

