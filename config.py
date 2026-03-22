# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Model ---
YOLO_MODEL = "yolov8n.pt"          # nano = fast; swap to yolov8s.pt for better accuracy
YOLO_CONFIDENCE = 0.3
PERSON_CLASS_ID = 0  

# --- Tracking ---
TRACKER_CONFIDENCE = 0.4

# --- Abandoned object logic ---
DISTANCE_THRESHOLD = 150           # pixels — how far person must move from bag
TIME_THRESHOLD_SECONDS = 15        # seconds — how long bag must be unattended
                                   # lower for testing, raise to 30-60 for real use

# --- AI ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Video ---
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# --- Output ---
OUTPUT_DIR = "outputs"
LOG_FILE = "outputs/events.json"