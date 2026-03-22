# main.py
import cv2
import json
import os
import argparse
import threading
import numpy as np
import config
from modules.detector import ObjectDetector
from modules.tracker import PersonTracker
from modules.behaviour import BehaviorAnalyzer
from modules.ai_reasoning import explain_event, describe_scene

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

COLOR_PERSON  = (0, 255, 0)
COLOR_OBJECT  = (0, 165, 255)
COLOR_ALERT   = (0, 0, 255)
COLOR_TEXT    = (255, 255, 255)

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

def wrap_text(text, max_chars=38):
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def draw_sidebar(frame, frame_num, fps, person_count, object_count,
                 scene_lines, alert_lines, ai_status):
    h, w = frame.shape[:2]
    panel_x = w - 320

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    total_sec = int(frame_num / fps)
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    timestamp = f"{hh:02}:{mm:02}:{ss:02}"

    # Header
    cv2.putText(frame, "COGNITIVE SURVEILLANCE", (panel_x + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)
    cv2.line(frame, (panel_x + 8, 36), (w - 8, 36), (60, 60, 60), 1)

    # Stats
    cv2.putText(frame, f"Time   : {timestamp}", (panel_x + 10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1)
    cv2.putText(frame, f"Frame  : {frame_num}", (panel_x + 10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1)
    cv2.putText(frame, f"Persons: {person_count}", (panel_x + 10, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
    cv2.putText(frame, f"Objects: {object_count}", (panel_x + 10, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1)
    cv2.line(frame, (panel_x + 8, 130), (w - 8, 130), (60, 60, 60), 1)

    # Scene description
    cv2.putText(frame, "SCENE ANALYSIS", (panel_x + 10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 180), 1)
    cv2.line(frame, (panel_x + 8, 160), (w - 8, 160), (60, 60, 60), 1)

    # AI thinking indicator
    if ai_status == "thinking":
        cv2.putText(frame, "AI analyzing...", (panel_x + 10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 0), 1)
        y_pos = 198
    else:
        y_pos = 176

    for line in scene_lines[-6:]:
        cv2.putText(frame, line, (panel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 200), 1)
        y_pos += 17

    # Divider
    cv2.line(frame, (panel_x + 8, y_pos + 6), (w - 8, y_pos + 6), (60, 60, 60), 1)

    # Alert section
    alert_y = y_pos + 26
    cv2.putText(frame, "ALERT LOG", (panel_x + 10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
    cv2.line(frame, (panel_x + 8, alert_y + 10), (w - 8, alert_y + 10), (60, 60, 60), 1)

    if alert_lines:
        ay = alert_y + 28
        for line in alert_lines[-10:]:
            cv2.putText(frame, line, (panel_x + 10, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 180, 180), 1)
            ay += 16
    else:
        cv2.putText(frame, "No alerts yet.", (panel_x + 10, alert_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)


def run(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_num = 0

    detector = ObjectDetector()
    tracker  = PersonTracker()
    analyzer = BehaviorAnalyzer(fps=fps)

    all_events       = []
    alert_lines      = []
    scene_lines      = ["Waiting for scene data..."]
    ai_status        = "idle"       # "idle" | "thinking"
    SKIP_FRAMES      = 2
    SCENE_INTERVAL   = int(fps * 8) # describe scene every 8 seconds

    print(f"[INFO] Processing: {video_path} at {fps:.1f} FPS")
    print(f"[INFO] Scene AI description every 8 seconds")
    print(f"[INFO] Press 'q' to quit\n")

    def fetch_scene_description(p_count, o_count, ts, p_ids):
        """Runs in background thread so it doesn't freeze the video."""
        nonlocal scene_lines, ai_status
        ai_status = "thinking"
        result = describe_scene(p_count, o_count, ts, p_ids)
        scene_lines = wrap_text(result, max_chars=38)
        ai_status = "idle"
        print(f"[SCENE] {result}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % SKIP_FRAMES != 0:
            frame_num += 1
            continue

        frame = cv2.resize(frame, (960, 720))

        # Detect + Track
        detections = detector.detect(frame)
        persons, objects = detector.split(detections)
        tracked_persons = tracker.update(persons)
        new_alerts = analyzer.update(frame_num, tracked_persons, objects)

        # Draw persons
        person_count = 0
        active_pids  = []
        if tracked_persons.tracker_id is not None:
            for i, tid in enumerate(tracked_persons.tracker_id):
                if tracked_persons.confidence[i] < 0.4:
                    continue
                draw_box(frame, tracked_persons.xyxy[i], f"P{tid}", COLOR_PERSON)
                person_count += 1
                active_pids.append(tid)

        # Draw objects
        object_count = 0
        for i, box in enumerate(objects.xyxy):
            cls_id = int(objects.class_id[i])
            label  = detector.model.names[cls_id]
            if label in config.SKIP_OBJECT_LABELS or objects.confidence[i] < 0.5:
                continue
            draw_box(frame, box, label, COLOR_OBJECT)
            object_count += 1

        # Trigger scene description every N frames in background
        total_sec = int(frame_num / fps)
        hh = total_sec // 3600
        mm = (total_sec % 3600) // 60
        ss = total_sec % 60
        ts_str = f"{hh:02}:{mm:02}:{ss:02}"

        if frame_num % SCENE_INTERVAL == 0 and ai_status == "idle":
            t = threading.Thread(
                target=fetch_scene_description,
                args=(person_count, object_count, ts_str, active_pids),
                daemon=True
            )
            t.start()

# Handle abandoned object alerts
        for alert in new_alerts:
            # Resolve actual object class name
            cls_id = alert.get("class_id", -1)
            object_label = detector.model.names.get(cls_id, "object")

            log_text = analyzer.build_behavior_log(alert, object_label=object_label)
            print(f"\n[ALERT] {log_text}")

            explanation = explain_event(log_text, object_label=object_label)
            alert["ai_explanation"] = explanation
            print(f"[AI]    {explanation}")
            all_events.append(alert)

            alert_lines.append(f"@ {alert['alert_time']}")
            alert_lines += wrap_text(explanation, max_chars=38)
            alert_lines.append("")

            cx, cy = int(alert["obj_center"][0]), int(alert["obj_center"][1])
            cv2.rectangle(frame, (cx-40, cy-40), (cx+40, cy+40), COLOR_ALERT, 3)
            cv2.putText(frame,
                        f"ABANDONED {object_label.upper()}!",
                        (cx-40, cy-52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ALERT, 2)
            cv2.imwrite(f"{config.OUTPUT_DIR}/alert_{frame_num}.jpg", frame)
        # Draw sidebar
        draw_sidebar(frame, frame_num, fps, person_count, object_count,
                     scene_lines, alert_lines, ai_status)

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