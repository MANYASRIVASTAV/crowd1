
# utils/video.py
import cv2
from ultralytics import YOLO
from .heatmap import generate_heatmap
from .alert import check_crowd_alert

YOLO_WEIGHTS = "models/yolov8n.pt"
model = YOLO(YOLO_WEIGHTS)

def process_video(input_path, output_path="output_video.mp4", crowd_threshold=2):
    """
    Process an input video file and write annotated output video.
    Returns a dict with stats.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    max_people = 0
    frame_count = 0
    total_people = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)[0]

        person_count = 0
        person_positions = []

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                person_positions.append(((x1 + x2) // 2, (y1 + y2) // 2))

        total_people += person_count
        max_people = max(max_people, person_count)

        # call alert (server side alert should not play sound; your alert.py probably just logs/flags)
        check_crowd_alert(frame_count, person_count, crowd_threshold)

        # overlay heatmap if you have logic in generate_heatmap (should accept frame + positions)
        frame = generate_heatmap(frame, person_positions)

        # overlay counts
        cv2.putText(frame, f"People in frame: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Max so far: {max_people}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    return {
        "max_people": int(max_people),
        "frames": int(frame_count),
        "total_people": int(total_people),
        "output_file": output_path
    }

if __name__ == "__main__":
    # keep a convenient CLI for local testing
    import sys
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else "output_video.mp4"
        stats = process_video(inp, out)
        print("Done:", stats)
    else:
        print("Usage: python utils/video.py <input_video_path> [output_path]")
