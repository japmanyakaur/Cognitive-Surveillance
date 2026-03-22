# modules/ai_reasoning.py
import anthropic
import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

def explain_event(behavior_log: str) -> str:
    """Send behavior log to Claude and get a natural language explanation."""
    try:
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a surveillance analyst AI. "
                        "Based on the following behavior log from a CCTV system, "
                        "write a concise 2-3 sentence security alert explanation "
                        "describing what happened and why it is suspicious.\n\n"
                        f"Log: {behavior_log}"
                    )
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"[AI reasoning unavailable: {e}]"