# server.py
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import threading
import numpy as np
from utils.heatmap import generate_heatmap
from utils.alert import check_crowd_alert
from ultralytics import YOLO

# -----------------------------
# --- YOLO Model ---
# -----------------------------
YOLO_WEIGHTS = "models/yolov8n.pt"
model = YOLO(YOLO_WEIGHTS)

# -----------------------------
# --- Flask App ---
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# --- Global Variables ---
# -----------------------------
latest_count_lock = threading.Lock()
latest_count = 0
CROWD_THRESHOLD = 4  # alert threshold
processing = False   # controls start/stop
video_source = 0     # default camera

# -----------------------------
# --- Detection + Streaming ---
# -----------------------------
def detect_frame(frame):
    """
    Run YOLO on frame, draw boxes + heatmap, return annotated frame and count
    """
    global latest_count
    results = model(frame)[0]

    person_count = 0
    person_positions = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:  # person
            person_count += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, "Person", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            person_positions.append((int((x1+x2)/2), int((y1+y2)/2)))

    # alert beep
    check_crowd_alert(0, person_count, CROWD_THRESHOLD)  # frame_count not needed here

    # overlay heatmap
    frame = generate_heatmap(frame, person_positions)

    # update latest_count
    with latest_count_lock:
        latest_count = person_count

    return frame, person_count

def video_generator(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open video source:", source)
        return

    global processing
    while True:
        if not processing:
            # if not processing, send blank frame
            blank_frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            ret2, jpeg = cv2.imencode('.jpg', blank_frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = detect_frame(frame)
        ret2, jpeg = cv2.imencode('.jpg', annotated)
        if not ret2:
            continue
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# -----------------------------
# --- Routes ---
# -----------------------------
@app.route('/video_feed')
def video_feed():
    return Response(video_generator(video_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count')
def get_count():
    with latest_count_lock:
        c = latest_count
    return jsonify({"peopleCount": int(c)})

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global processing
    processing = True
    return jsonify({"status": "started"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global processing
    processing = False
    return jsonify({"status": "stopped"})

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# -----------------------------
# --- Main ---
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
