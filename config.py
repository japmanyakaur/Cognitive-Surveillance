# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Model ---
YOLO_MODEL = "yolov8m.pt"
YOLO_CONFIDENCE = 0.5
PERSON_CLASS_ID = 0

# --- Tracking ---
DISTANCE_THRESHOLD = 150
TIME_THRESHOLD_SECONDS = 15

# --- AI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Output ---
OUTPUT_DIR = "outputs"
LOG_FILE = "outputs/events.json"

# --- Objects to skip ---
SKIP_OBJECT_LABELS = {
    "traffic light", "stop sign", "parking meter",
    "bench", "chair", "car", "truck", "bus",
    "bicycle", "motorcycle"
}
SCENE_LOG_FILE = "outputs/scene_log.txt"
