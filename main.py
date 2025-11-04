from openai import OpenAI
import os
import re
import base64
import json
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ─────────────────────────────
# New: Mongo
# ─────────────────────────────
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

# ─────────────────────────────
# Setup
# ─────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)


MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is required")

mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
try:
    # trigger connection check early
    mongo_client.admin.command("ping")
except Exception as e:
    raise RuntimeError(f"Failed to connect to MongoDB: {e}")

# If the URI includes '/campusmap', this returns that DB; else fallback to 'campusmap'
db = mongo_client.get_default_database()
events_col = db["events"]
meta_col = db["meta"]  # optional for storing last refresh time, etc.

# Ensure helpful indexes (id unique; time & duplicate-detection helpers)
events_col.create_index([("id", ASCENDING)], unique=True, name="uniq_id")
events_col.create_index([("start_dt", ASCENDING)], name="start_dt_idx")
events_col.create_index(
    [("title_norm", ASCENDING), ("location_norm", ASCENDING), ("start_dt", ASCENDING)],
    name="dupe_probe_idx",
)


# ─────────────────────────────
# Shared helpers
# ─────────────────────────────


def extract_json(raw: str) -> dict:
    """
    Clean up model output and return the first {...} JSON object inside.
    Raises ValueError if no valid JSON is found.
    """
    # Strip fenced code blocks if present
    if raw.startswith("```") and raw.endswith("```"):
        # remove the fences, keep internal content
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
    return json.loads(json_str)


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
@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "endpoints": ["/ask/vision (POST)", "/ask/events (POST)"]})


@app.route("/ask", methods=["POST"])
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


@app.route("/ask/events", methods=["POST"])
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


@app.route("/ask/events", methods=["POST"])
def get_events():
    x = 10


# ─────────────────────────────
# Entrypoint
# ─────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "6050"))
    app.run(host="0.0.0.0", port=port)
