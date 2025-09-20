from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO
from utils.heatmap import generate_heatmap
from utils.alert import check_crowd_alert

# -----------------------------
# --- Flask app ---
# -----------------------------
app = Flask(__name__)

# -----------------------------
# --- YOLO & Video ---
# -----------------------------
YOLO_WEIGHTS = "models/yolov8n.pt"
model = YOLO(YOLO_WEIGHTS)
cap = cv2.VideoCapture(0)  # webcam

# -----------------------------
# --- Counting & State ---
# -----------------------------
max_people = 0
CROWD_THRESHOLD = 4
frame_count = 0
current_people = 0

# -----------------------------
# --- Frame generator ---
# -----------------------------
def gen_frames():
    global max_people, frame_count, current_people
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)[0]

        person_count = 0
        person_positions = []

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 0:  # person class
                person_count += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Person", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                person_positions.append((int((x1+x2)/2), int((y1+y2)/2)))

        if person_count > max_people:
            max_people = person_count
        current_people = person_count

        # Crowd alert
        check_crowd_alert(frame_count, person_count, CROWD_THRESHOLD)

        # Heatmap overlay
        frame = generate_heatmap(frame, person_positions)

        # Counts overlay
        cv2.putText(frame, f"People: {person_count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        cv2.putText(frame, f"Max: {max_people}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -----------------------------
# --- Routes ---
# -----------------------------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        "frame": frame_count,
        "people": current_people,
        "max_people": max_people
    })

# -----------------------------
# --- Run server ---
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
