# modules/ai_reasoning.py
import google.generativeai as genai
import config

genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

def explain_event(behavior_log: str) -> str:
    """Called when abandoned object alert fires."""
    try:
        prompt = (
            "You are a surveillance analyst AI. "
            "Based on the following behavior log from a CCTV system, "
            "write a concise 2-3 sentence security alert explanation "
            "describing what happened and why it is suspicious.\n\n"
            f"Log: {behavior_log}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[AI reasoning unavailable: {e}]"


def describe_scene(person_count: int, object_count: int, timestamp: str, person_ids: list) -> str:
    """Called every N seconds to describe what is currently happening in the scene."""
    try:
        ids_str = ", ".join([f"P{i}" for i in person_ids]) if person_ids else "none"
        prompt = (
            "You are a CCTV surveillance AI assistant. "
            "Describe what is currently happening in the scene in 2 sentences. "
            "Be concise, factual, and use natural language like a security analyst would.\n\n"
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