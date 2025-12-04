from flask import Blueprint, jsonify, request
import json
import os
import base64
from collections import defaultdict, deque

from openai import OpenAI
from dotenv import load_dotenv

from helper.normalize_location import normalize_event_location

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ask_bp = Blueprint('ask', __name__)


def extract_json(raw: str) -> dict:
    """
    Clean up model output and return the first {...} JSON object inside.
    Raises ValueError if no valid JSON is found.
    """
    # Strip fenced code blocks if present
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.strip("`").strip()

    # Find first '{'
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    # Match braces to find the end of the first JSON object
    depth = 0
    end = None
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError("Unbalanced braces in model output")

    json_str = raw[start: end + 1]

    event_json = json.loads(json_str)
    # add new field location_at = event_json["location"] will keep original raw location
    print(event_json)  # This prints successfully because it is a valid dict

    # Make sure we got a dict
    if not isinstance(event_json, dict):
        raise ValueError(
            f"Expected JSON object but got {type(event_json).__name__}")

    # Normalize the location field if present
    # CORRECT: Using .get() and bracket notation
    loc = event_json.get("location")
    event_json["location_at"] = loc
    print(loc)
    if loc is not None:
        normalized = normalize_event_location(loc)
        # if normalize_event_location returns None, fall back to the original string
        event_json["location"] = normalized if normalized is not None else loc

    # DELETED: The line causing the crash (event_json.location) was here.

    return event_json


# ─────────────────────────────
# Events memory / config
# ─────────────────────────────
# In-memory store for user message history (for /ask/events)
message_history = defaultdict(lambda: deque(maxlen=10))

# Constants to prevent abuse
MAX_CONTEXT_TOKENS = 3000  # rough input limit
MAX_COMPLETION_TOKENS = 800  # output limit


def approximate_token_count(messages):
    # Very rough estimate: ~1 token ≈ 4 characters
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        # If content isn't a string, coerce to string conservatively
        if not isinstance(content, str):
            try:
                content = json.dumps(content)
            except Exception:
                content = str(content)
        total += len(content) // 4
    return total


# Load events.json (safe fallback to empty list if missing)
EVENTS = []
EVENTS_PATH = os.path.join(os.getcwd(), "events.json")
if os.path.exists(EVENTS_PATH):
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            EVENTS = json.load(f)
    except Exception:
        EVENTS = []


# ─────────────────────────────
# Routes
# ─────────────────────────────
@ask_bp.route("/endpoints", methods=["GET"])
def root():
    return jsonify({"ok": True, "endpoints": ["/ask/vision (POST)", "/ask/events (POST)"]})


@ask_bp.route("/ask", methods=["POST"])
def ask_vision():
    """
    Multipart form-data with a file field named 'file'.
    Returns strict JSON extracted from the image:
      {
        "date": "",
        "time": "",
        "location": "",
        "names": [],
        "event_name": "",
        "description": ""
      }
    """
    if "file" not in request.files:
        return jsonify({"error": "No image file provided (field name should be 'file')"}), 400

    uploaded = request.files["file"]
    img_bytes = uploaded.read()
    if not img_bytes:
        return jsonify({"error": "Empty file"}), 400

    mime = uploaded.mimetype or "image/png"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    system_message = {
        "role": "system",
        "content": (
            "You are a vision-enabled assistant. "
            "Extract from the image: date, time, location, names, event name, and a short description of the event. "
            "Respond *only* with valid JSON matching this schema:\n\n"
            "{\n"
            "  \"date\": \"\",\n"
            "  \"time\": \"\",\n"
            "  \"location\": \"\",\n"
            "  \"names\": [],\n"
            "  \"event_name\": \"\",\n"
            "  \"description\": \"\"\n"
            "}\n"
            "Rules:\n"
            "- If a field is unknown, use an empty string (or empty array for names).\n"
            "- Do not add extra keys. Do not include explanations."
        ),
    }

    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            temperature=0.0,
            max_tokens=400,
        )

        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return jsonify({"error": "Empty model response"}), 502

        try:
            result = extract_json(raw)
        except ValueError:
            # Return the raw text to help debugging the prompt/formatting
            return jsonify({"error": "Failed to extract JSON", "raw_response": raw}), 500

        # Success
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Model did not return valid JSON"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ask_bp.route("/ask/events", methods=["POST"])
def ask_events():
    """
    JSON body:
      {
        "user_id": "abc123",       # optional, for per-user short memory
        "question": "What should I attend?",
        "tags": ["freshmen","engineering"]  # optional tag filtering
      }
    Returns:
      {
        "response": "<assistant text>",
        "matched_events": [ ...events whose IDs were referenced... ]
      }
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "default")
    question = data.get("question", "")
    tags = data.get("tags", [])

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "No question provided"}), 400

    # Filter events by tags (if provided)
    filtered_events = [
        event for event in EVENTS
        if not tags or any(tag in event.get("tags", []) for tag in tags)
    ]

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant for UC Merced's Bobcat Day. "
            "You help students find relevant events based on their interests. "
            "When recommending events, include their IDs at the end in a JSON array like [\"event002\", \"event004\"]."
        ),
    }

    # Maintain short per-user history
    history = message_history[user_id]
    history.append({"role": "user", "content": question})

    # Create a compact context to keep token usage sane
    context_prompt = (
        f"User asked: \"{question}\"\n\n"
        f"Here is a list of events:\n{json.dumps(filtered_events, ensure_ascii=False)}"
    )

    messages = [system_message] + \
        list(history) + [{"role": "user", "content": context_prompt}]

    # Truncate if too long
    while approximate_token_count(messages) > MAX_CONTEXT_TOKENS and len(history) > 0:
        history.popleft()
        messages = [system_message] + \
            list(history) + [{"role": "user", "content": context_prompt}]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4,
            max_tokens=MAX_COMPLETION_TOKENS,
        )

        reply = (response.choices[0].message.content or "").strip()

        # Save assistant reply to history
        history.append({"role": "assistant", "content": reply})

        # Extract event IDs like event001, event123, etc.
        event_ids = re.findall(r"event\d{3}", reply)
        matched_events = [
            event for event in EVENTS if event.get("id") in set(event_ids)]

        return jsonify({"response": reply, "matched_events": matched_events})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
