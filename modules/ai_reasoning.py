# modules/ai_reasoning.py
import google.generativeai as genai
import config

genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # fast + free tier

def explain_event(behavior_log: str) -> str:
    """Send behavior log to Gemini and get a natural language explanation."""
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