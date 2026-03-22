# main.py
import cv2
import json
import os
import argparse
import numpy as np
import config
from modules.detector import ObjectDetector
from modules.tracker import PersonTracker
from modules.behaviour import BehaviorAnalyzer
from modules.ai_reasoning import explain_event

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Colors (BGR)
COLOR_PERSON = (0, 255, 0)       # green
COLOR_OBJECT = (0, 165, 255)     # orange
COLOR_ALERT  = (0, 0, 255)       # red
COLOR_TEXT   = (255, 255, 255)   # white

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

def draw_alert_banner(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"ALERT: {text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def run(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_num = 0

    detector  = ObjectDetector()
    tracker   = PersonTracker()
    analyzer  = BehaviorAnalyzer(fps=fps)

    all_events   = []
    active_alerts = []   # list of (alert_text, expiry_frame)

    print(f"[INFO] Processing: {video_path} at {fps:.1f} FPS")
    print(f"[INFO] Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"[INFO] Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"[INFO] Press 'q' to quit\n")

    # Process every Nth frame for speed (1 = every frame, 2 = every other, etc.)
    SKIP_FRAMES = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for speed
        if frame_num % SKIP_FRAMES != 0:
            frame_num += 1
            continue

        # Resize for display + faster inference
        frame = cv2.resize(frame, (1280, 720))

        # 1. Detect
        detections = detector.detect(frame)
        persons, objects = detector.split(detections)

        # 2. Track persons
        tracked_persons = tracker.update(persons)

        # 3. Analyze behavior
        new_alerts = analyzer.update(frame_num, tracked_persons, objects)

# 4. Draw person boxes with tracker ID
        if tracked_persons.tracker_id is not None:
            for i, tid in enumerate(tracked_persons.tracker_id):
                box = tracked_persons.xyxy[i]
                conf = tracked_persons.confidence[i]
                if conf < 0.4:      # skip low confidence detections
                    continue
                draw_box(frame, box, f"P{tid}", COLOR_PERSON)

        # 5. Draw object boxes — skip static/environmental objects
        SKIP_LABELS = {
            "traffic light", "stop sign", "parking meter",
            "bench", "chair", "dining table", "tv", "laptop",
            "building", "wall", "floor", "ceiling", "tree",
            "pole", "sign"
        }
        for i, box in enumerate(objects.xyxy):
            cls_id = int(objects.class_id[i])
            label = detector.model.names[cls_id]
            conf = objects.confidence[i]
            if label in SKIP_LABELS:    # skip environmental objects
                continue
            if conf < 0.5:              # skip low confidence objects
                continue
            draw_box(frame, box, label, COLOR_OBJECT)
        # 6. Handle new alerts
        for alert in new_alerts:
            log_text = analyzer.build_behavior_log(alert)
            print(f"\n[ALERT] {log_text}")

            explanation = explain_event(log_text)
            alert["ai_explanation"] = explanation
            print(f"[AI]    {explanation}")

            all_events.append(alert)
            active_alerts.append((explanation[:80], frame_num + int(fps * 10)))

            # Draw red box on abandoned object
            bag_center = alert["bag_center"]
            cx, cy = int(bag_center[0]), int(bag_center[1])
            cv2.rectangle(frame, (cx-40, cy-40), (cx+40, cy+40), COLOR_ALERT, 3)
            cv2.putText(frame, "ABANDONED!", (cx-40, cy-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ALERT, 2)

            # Save alert frame
            cv2.imwrite(f"{config.OUTPUT_DIR}/alert_{frame_num}.jpg", frame)

        # 7. Show active alert banners (stay on screen for 10 seconds)
        active_alerts = [(txt, exp) for txt, exp in active_alerts if frame_num < exp]
        for idx, (txt, _) in enumerate(active_alerts):
            draw_alert_banner(frame, txt)

        # 8. Show frame info overlay
        cv2.putText(frame, f"Frame: {frame_num} | FPS: {fps:.0f}",
                    (10, 720 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        # 9. Display
        cv2.imshow("Cognitive Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

    with open(config.LOG_FILE, "w") as f:
        json.dump(all_events, f, indent=2)
    print(f"\n[INFO] Done. {frame_num} frames processed.")
    print(f"[INFO] Events saved to {config.LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()
    run(args.video) 