# modules/ai_reasoning.py
import google.generativeai as genai
import config
import cv2
import base64
import numpy as np

genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 for Gemini vision."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def explain_event(behavior_log: str, object_label: str = "object") -> str:
    """Called when abandoned object alert fires."""
    try:
        response = model.generate_content([
            f"You are a surveillance analyst AI. "
            f"Based on this CCTV behavior log involving an abandoned {object_label}, "
            f"write a concise 2-3 sentence security alert in natural language "
            f"describing what happened and why it may be suspicious.\n\n"
            f"Log: {behavior_log}"
        ])
        return response.text.strip()
    except Exception as e:
        return f"[AI reasoning unavailable: {e}]"


def describe_scene(frame, person_count: int, object_count: int,
                   timestamp: str, person_ids: list) -> str:
    """Send actual frame to Gemini for real visual scene understanding."""
    try:
        ids_str = ", ".join([f"P{i}" for i in person_ids]) if person_ids else "none"
        prompt = (
            "You are a CCTV surveillance AI assistant. "
            "Describe what is currently happening in the scene in 2 sentences. "
            "Be concise, factual, and use natural language like a security analyst would use.\n\n"
            f"Current scene data at {timestamp}:\n"
            f"- People visible: {person_count} (IDs: {ids_str})\n"
            f"- Unattended objects detected: {object_count}\n"
            f"- This is a CCTV night vision / outdoor scene.\n\n"
            "Describe the scene naturally."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Scene description unavailable: {e}]"
