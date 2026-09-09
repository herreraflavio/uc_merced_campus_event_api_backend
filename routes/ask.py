from flask import Flask, request, jsonify, Blueprint
from flask import Blueprint, request, jsonify
from flask import Blueprint, jsonify, request
import logging
import json
import os
import re
import base64
import math
from collections import Counter
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict, deque

from openai import OpenAI
from dotenv import load_dotenv

from helper.normalize_location import normalize_event_location
from .ai_retrieval import (
    MAX_CONTEXT_ITEMS as AI_MAX_CONTEXT_ITEMS,
    MIN_CONTEXT_ITEMS as AI_MIN_CONTEXT_ITEMS,
    rank_items,
    serialize_retrieval_debug,
)
from .events import build_content_api_payload

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ask_bp = Blueprint('ask', __name__)

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------------------------------
# This will create or append to "ai_debug_log.txt" in your app's working directory.
logging.basicConfig(
    filename="ai_debug_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="a"
)
logger = logging.getLogger(__name__)


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


# ─────────────────────────────
# Presence Pages Cache / Config
# ─────────────────────────────
PRESENCE_PAGES = []
PRESENCE_CACHE_PATH = os.path.join(os.getcwd(), "presence_pages_cache.json")

if os.path.exists(PRESENCE_CACHE_PATH):
    try:
        with open(PRESENCE_CACHE_PATH, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            if isinstance(cache_data, dict) and "pages" in cache_data:
                PRESENCE_PAGES = cache_data["pages"]
            elif isinstance(cache_data, list):
                PRESENCE_PAGES = cache_data
    except Exception as e:
        print(f"Failed to load presence_pages_cache.json: {e}")

# ─────────────────────────────
# New AI Smart Search Route
# ─────────────────────────────

# ─────────────────────────────
# Presence Pages Cache / Config
# ─────────────────────────────
PRESENCE_PAGES = []
PRESENCE_CACHE_PATH = os.path.join(os.getcwd(), "presence_pages_cache.json")

if os.path.exists(PRESENCE_CACHE_PATH):
    try:
        with open(PRESENCE_CACHE_PATH, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            if isinstance(cache_data, dict) and "pages" in cache_data:
                PRESENCE_PAGES = cache_data["pages"]
            elif isinstance(cache_data, list):
                PRESENCE_PAGES = cache_data
    except Exception as e:
        print(f"Failed to load presence_pages_cache.json: {e}")

# Make sure you have your blueprint and client defined above this like:
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ask_bp = Blueprint('ask', __name__)

# ─────────────────────────────
# Presence Pages Cache / Config
# ─────────────────────────────
PRESENCE_PAGES = []
PRESENCE_CACHE_PATH = os.path.join(os.getcwd(), "presence_pages_cache.json")

if os.path.exists(PRESENCE_CACHE_PATH):
    try:
        with open(PRESENCE_CACHE_PATH, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            if isinstance(cache_data, dict):
                # We check for "events" first, as that matches your JSON schema
                if "events" in cache_data:
                    PRESENCE_PAGES = cache_data["events"]
                elif "pages" in cache_data:
                    PRESENCE_PAGES = cache_data["pages"]
            elif isinstance(cache_data, list):
                PRESENCE_PAGES = cache_data
    except Exception as e:
        print(f"Failed to load presence_pages_cache.json: {e}")

# ─────────────────────────────
# New AI Smart Search Route
# ─────────────────────────────

# Ensure you have your Flask blueprint and client setup
# ask_bp = Blueprint('ask_bp', __name__)
# client = ... (OpenAI client)

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------------------------------
logging.basicConfig(
    filename="ai_debug_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="a"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"

MAX_DESC_CHARS = 220

# Increased max context items to give the LLM enough options to actually return 10+
MAX_CONTEXT_ITEMS = 25
MIN_CONTEXT_ITEMS = 4

MAX_NESTED_SEARCH_CHARS = 16000
MAX_NESTED_CONTEXT_CHARS = 1200
MAX_BLOB_FOR_FUZZY = 1800
MAX_SEGMENT_FOR_FUZZY = 500

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

SKIP_NESTED_KEYS = {
    "image_urls", "pin_url", "source_url", "url", "urls", "href", "link", "links"
}

# ------------------------------------------------------------------------------
# STOP WORDS & DYNAMIC WORD BANK
# ------------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "were", "it", "this", "that", "these", "those",
    "then", "just", "so", "than", "such", "both", "through", "about", "while",
    "during", "what", "they", "we", "he", "she", "if", "because", "as", "when",
    "where", "how", "who", "which", "be", "has", "have", "had", "do", "does", "did"
}

DYNAMIC_EXPANSIONS = {}

# ------------------------------------------------------------------------------
# STATIC WORD BANK / ALIASES
# ------------------------------------------------------------------------------
DAY_ALIASES = {
    "monday": {"monday", "mon"},
    "tuesday": {"tuesday", "tue", "tues"},
    "wednesday": {"wednesday", "wed"},
    "thursday": {"thursday", "thu", "thur", "thurs"},
    "friday": {"friday", "fri"},
    "saturday": {"saturday", "sat"},
    "sunday": {"sunday", "sun"},
}

MEAL_ALIASES = {
    "breakfast": {"breakfast", "bf", "brkfst", "morning"},
    "lunch": {"lunch", "midday", "noon"},
    "dinner": {"dinner", "supper", "evening"},
    "bakery": {"bakery", "dessert", "pastry", "pastries", "cake", "cookies", "muffin", "croissant", "strudel"},
}

LOCATION_ALIASES = {
    "pavilion": {"pavilion", "pav"},
}

TOKEN_WORD_BANK = {
    "pav": {"pavilion"}, "pavilion": {"pav"},
    "fri": {"friday"}, "friday": {"fri"},
    "thu": {"thursday"}, "thurs": {"thursday"}, "thursday": {"thu", "thurs"},
    "wed": {"wednesday"}, "wednesday": {"wed"},
    "tue": {"tuesday"}, "tues": {"tuesday"}, "tuesday": {"tue", "tues"},
    "mon": {"monday"}, "monday": {"mon"},
    "sat": {"saturday"}, "saturday": {"sat"},
    "sun": {"sunday"}, "sunday": {"sun"},
    "veggie": {"vegetarian", "vegan", "plant", "plantbased"},
    "veg": {"vegetarian", "vegan", "plant", "plantbased"},
    "vegan": {"vegetarian", "plant", "plantbased"},
    "vegetarian": {"vegan", "plant", "plantbased"},
    "plant": {"plantbased", "vegan", "vegetarian"},
    "plantbased": {"plant", "vegan", "vegetarian"},
    "gf": {"gluten", "free", "glutenfree"},
    "glutenfree": {"gluten", "free", "gf"},
    "coffee": {"decaf", "drinks", "tea"},
    "tea": {"drinks", "coffee"},
    "burger": {"burgers"},
    "taco": {"tacos"},
    "ramen": {"noodle", "pho"},
    "pho": {"ramen", "noodle"},
    "pizza": {"pies"},
    "salad": {"greens"},
    "park": {"parking", "lot"},
    "parking": {"park", "lot"},
    "lot": {"park", "parking"},
}

PHRASE_WORD_BANK = {
    "gluten free": {"glutenfree", "gf"},
    "plant based": {"plantbased", "vegan", "vegetarian"},
}


# ------------------------------------------------------------------------------
# TEXT HELPERS
# ------------------------------------------------------------------------------
def make_singular(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('es') and not word.endswith('ss'):
        return word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s:/=\-|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    tokens = set()
    for tok in normalize_text(text).split():
        if len(tok) > 1 and tok not in STOP_WORDS:
            tokens.add(tok)
            tokens.add(make_singular(tok))
    return tokens


def strip_urls(text: str) -> str:
    if not text:
        return ""
    text = URL_RE.sub("", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_description(text: str, limit: int = MAX_DESC_CHARS) -> str:
    text = strip_urls(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def compact_nested_value(value: str) -> str:
    text = strip_urls(value)
    if not text:
        return ""
    m = re.match(r"^\s*station\s*:\s*(.+)$", text, re.IGNORECASE)
    if m:
        return f"station={m.group(1).strip()}"
    m = re.match(r"^\s*description\s*:\s*(.+)$", text, re.IGNORECASE)
    if m:
        return f"desc={m.group(1).strip()}"
    m = re.match(r"^\s*calories\s*:\s*([0-9]+)", text, re.IGNORECASE)
    if m:
        return f"cal={m.group(1).strip()}"
    text = text.strip(" |;,-")
    return text


def join_with_limit(parts, max_chars):
    out = []
    total = 0
    for part in parts:
        if not part:
            continue
        add_len = len(part) if not out else len(part) + 5
        if total + add_len > max_chars:
            break
        out.append(part)
        total += add_len
    return " ### ".join(out)


# ------------------------------------------------------------------------------
# DYNAMIC GENERATION HELPERS
# ------------------------------------------------------------------------------
def generate_dynamic_word_bank(pages: list):
    global DYNAMIC_EXPANSIONS
    DYNAMIC_EXPANSIONS.clear()

    all_words = []
    for p in pages:
        text_blob = f"{p.get('title', '')} {p.get('description', '')} {json.dumps(p.get('nested_content', ''))}"
        norm = normalize_text(text_blob)
        words = [make_singular(w) for w in norm.split()
                 if len(w) > 2 and w not in STOP_WORDS]
        all_words.extend(words)

    if not all_words:
        return

    counts = Counter(all_words)
    freqs = list(counts.values())

    mean_freq = sum(freqs) / len(freqs)
    variance = sum((f - mean_freq) ** 2 for f in freqs) / len(freqs)
    std_dev = math.sqrt(variance) if variance > 0 else 1

    valid_words = [w for w, f in counts.items() if (
        mean_freq - std_dev) <= f <= (mean_freq + std_dev)]

    for w in valid_words:
        sing = make_singular(w)
        if sing not in DYNAMIC_EXPANSIONS:
            DYNAMIC_EXPANSIONS[sing] = set()
        DYNAMIC_EXPANSIONS[sing].add(w)
        if w != sing:
            if w not in DYNAMIC_EXPANSIONS:
                DYNAMIC_EXPANSIONS[w] = set()
            DYNAMIC_EXPANSIONS[w].add(sing)


# ------------------------------------------------------------------------------
# QUERY / WORD BANK HELPERS
# ------------------------------------------------------------------------------
def detect_canonical_matches(text: str, alias_map: dict) -> set:
    norm_text = normalize_text(text)
    tokens = set(norm_text.split())
    found = set()

    for canonical, aliases in alias_map.items():
        variants = set(aliases) | {canonical}
        for alias in variants:
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue

            if " " in alias_norm:
                if alias_norm in norm_text:
                    found.add(canonical)
                    break
            else:
                if alias_norm in tokens or make_singular(alias_norm) in tokens:
                    found.add(canonical)
                    break
    return found


def build_query_hints(query: str) -> dict:
    query_norm = normalize_text(query)
    raw_tokens = set(query_norm.split())
    expanded_tokens = set(raw_tokens)

    for tok in list(raw_tokens):
        expanded_tokens.add(make_singular(tok))

    for phrase, expansions in PHRASE_WORD_BANK.items():
        phrase_norm = normalize_text(phrase)
        if phrase_norm in query_norm:
            expanded_tokens.update(expansions)

    for tok in list(expanded_tokens):
        expanded_tokens.update(TOKEN_WORD_BANK.get(tok, set()))
        expanded_tokens.update(DYNAMIC_EXPANSIONS.get(tok, set()))

    days = detect_canonical_matches(query, DAY_ALIASES)
    meals = detect_canonical_matches(query, MEAL_ALIASES)
    locations = detect_canonical_matches(query, LOCATION_ALIASES)

    for d in days:
        expanded_tokens.add(d)
        expanded_tokens.update(DAY_ALIASES.get(d, set()))

    for m in meals:
        expanded_tokens.add(m)
        expanded_tokens.update(MEAL_ALIASES.get(m, set()))

    for loc in locations:
        expanded_tokens.add(loc)
        expanded_tokens.update(LOCATION_ALIASES.get(loc, set()))

    return {
        "query": query,
        "query_norm": query_norm,
        "raw_tokens": raw_tokens,
        "expanded_tokens": expanded_tokens,
        "days": days,
        "meals": meals,
        "locations": locations,
    }


def canonicalize_from_aliases(value: str, alias_map: dict) -> str:
    matches = detect_canonical_matches(value or "", alias_map)
    if matches:
        return sorted(matches)[0]
    return normalize_text(value or "")


# ------------------------------------------------------------------------------
# NESTED CONTENT EXTRACTION
# ------------------------------------------------------------------------------
def gather_generic_strings(obj, fragments):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in SKIP_NESTED_KEYS:
                continue
            if isinstance(value, str):
                compact = compact_nested_value(value)
                if compact:
                    fragments.append(compact)
            elif isinstance(value, (dict, list)):
                gather_generic_strings(value, fragments)
    elif isinstance(obj, list):
        for item in obj:
            gather_generic_strings(item, fragments)
    elif isinstance(obj, str):
        compact = compact_nested_value(obj)
        if compact:
            fragments.append(compact)


def make_segment_text(day_title: str, tab_title: str, header: str, extra_parts: list) -> str:
    parts = []
    if day_title:
        parts.append(f"day={day_title}")
    if tab_title:
        parts.append(f"tab={tab_title}")
    if header:
        parts.append(f"item={header}")

    for part in extra_parts:
        part = compact_nested_value(part)
        if part:
            parts.append(part)

    deduped = []
    seen = set()
    for part in parts:
        norm = normalize_text(part)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(part)

    return " || ".join(deduped).strip()


def extract_nested_segments(nested_content):
    segments = []
    seq = 0

    if not nested_content:
        return segments

    if isinstance(nested_content, list):
        for day_obj in nested_content:
            if not isinstance(day_obj, dict):
                continue

            day_title = strip_urls(day_obj.get("title", "") or "")
            tabs = day_obj.get("tabs", [])

            if isinstance(tabs, list) and tabs:
                for tab_obj in tabs:
                    if not isinstance(tab_obj, dict):
                        continue

                    tab_title = strip_urls(tab_obj.get("title", "") or "")
                    sections = tab_obj.get("sections", [])

                    if isinstance(sections, list) and sections:
                        for section in sections:
                            if not isinstance(section, dict):
                                continue

                            header = strip_urls(
                                section.get("header", "") or "")
                            extra_parts = []

                            bullets = section.get("bullets", [])
                            if isinstance(bullets, list):
                                extra_parts.extend(bullets)

                            for key, value in section.items():
                                if key in {"header", "bullets"} or key in SKIP_NESTED_KEYS:
                                    continue
                                if isinstance(value, str):
                                    extra_parts.append(f"{key}={value}")

                            text = make_segment_text(
                                day_title, tab_title, header, extra_parts)
                            if not text:
                                continue

                            segments.append({
                                "idx": seq,
                                "day": day_title,
                                "tab": tab_title,
                                "header": header,
                                "text": text,
                                "normalized": normalize_text(text),
                                "token_set": tokenize(text),
                                "day_canonical": canonicalize_from_aliases(day_title, DAY_ALIASES),
                                "tab_canonical": canonicalize_from_aliases(tab_title, MEAL_ALIASES),
                            })
                            seq += 1
                    else:
                        fallback_parts = []
                        gather_generic_strings(tab_obj, fallback_parts)
                        text = make_segment_text(
                            day_title, tab_title, "", fallback_parts)
                        if text:
                            segments.append({
                                "idx": seq,
                                "day": day_title,
                                "tab": tab_title,
                                "header": "",
                                "text": text,
                                "normalized": normalize_text(text),
                                "token_set": tokenize(text),
                                "day_canonical": canonicalize_from_aliases(day_title, DAY_ALIASES),
                                "tab_canonical": canonicalize_from_aliases(tab_title, MEAL_ALIASES),
                            })
                            seq += 1
            else:
                fallback_parts = []
                gather_generic_strings(day_obj, fallback_parts)
                text = make_segment_text(day_title, "", "", fallback_parts)
                if text:
                    segments.append({
                        "idx": seq,
                        "day": day_title,
                        "tab": "",
                        "header": "",
                        "text": text,
                        "normalized": normalize_text(text),
                        "token_set": tokenize(text),
                        "day_canonical": canonicalize_from_aliases(day_title, DAY_ALIASES),
                        "tab_canonical": "",
                    })
                    seq += 1
    else:
        fallback_parts = []
        gather_generic_strings(nested_content, fallback_parts)
        text = make_segment_text("", "", "", fallback_parts)
        if text:
            segments.append({
                "idx": seq,
                "day": "",
                "tab": "",
                "header": "",
                "text": text,
                "normalized": normalize_text(text),
                "token_set": tokenize(text),
                "day_canonical": "",
                "tab_canonical": "",
            })

    return segments


def collapse_structured_segments(segments, max_chars=MAX_NESTED_SEARCH_CHARS) -> str:
    return join_with_limit([seg["text"] for seg in segments], max_chars)


# ------------------------------------------------------------------------------
# SEGMENT SCORING / QUERY-AWARE EXCERPT
# ------------------------------------------------------------------------------
def score_segment(query_hints: dict, segment: dict) -> float:
    expanded_tokens = query_hints["expanded_tokens"]
    query_norm = query_hints["query_norm"]

    if not query_norm:
        return 0.0

    seg_norm = segment["normalized"][:MAX_SEGMENT_FOR_FUZZY]
    seg_tokens = segment["token_set"]

    overlap = len(expanded_tokens & seg_tokens)
    overlap_ratio = overlap / max(len(expanded_tokens), 1)

    seq_ratio = SequenceMatcher(None, query_norm, seg_norm).ratio()
    contains_boost = 1.0 if query_norm in seg_norm else 0.0

    day_bonus = 0.0
    meal_bonus = 0.0

    if query_hints["days"] and segment["day_canonical"] in query_hints["days"]:
        day_bonus += 0.60
    if query_hints["meals"] and segment["tab_canonical"] in query_hints["meals"]:
        meal_bonus += 0.50

    header_hits = 0
    header_norm = normalize_text(segment.get("header", ""))
    if header_norm:
        header_hits = sum(1 for tok in expanded_tokens if tok in header_norm)

    score = (
        (overlap_ratio * 0.35) +
        (seq_ratio * 0.25) +
        (contains_boost * 0.15) +
        day_bonus +
        meal_bonus +
        (header_hits * 0.08)
    )

    return round(score, 6)


def build_query_aware_nested_excerpt(segments, query_hints, max_chars=MAX_NESTED_CONTEXT_CHARS) -> str:
    if not segments:
        return ""

    scored = []
    for seg in segments:
        seg_score = score_segment(query_hints, seg)
        scored.append((seg_score, seg["idx"], seg))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    selected_chars = 0
    group_counts = {}

    for seg_score, _, seg in scored:
        if seg_score <= 0 and selected:
            continue

        group_key = (seg.get("day_canonical", ""),
                     seg.get("tab_canonical", ""))
        if group_counts.get(group_key, 0) >= 4:
            continue

        text = seg["text"]
        add_len = len(text) if not selected else len(text) + 5

        if selected_chars + add_len > max_chars:
            continue

        selected.append(seg)
        selected_chars += add_len
        group_counts[group_key] = group_counts.get(group_key, 0) + 1

        if len(selected) >= 8:
            break

    if not selected:
        for seg in sorted(segments, key=lambda s: s["idx"]):
            text = seg["text"]
            add_len = len(text) if not selected else len(text) + 5
            if selected_chars + add_len > max_chars:
                break
            selected.append(seg)
            selected_chars += add_len

    selected.sort(key=lambda s: s["idx"])
    return " ### ".join(seg["text"] for seg in selected)


# ------------------------------------------------------------------------------
# ITEM ENCODING / SCORING
# ------------------------------------------------------------------------------
def encode_item(item: dict) -> dict:
    title = item.get("title", "") or ""
    subtitle = item.get("subtitle", "") or ""
    host = item.get("host", "") or ""
    # description = compact_description(item.get("description", "") or "")
    description = item.get("description", "") or ""
    tags = item.get("tags", []) or []
    item_type = item.get("type", "") or ""
    start = item.get("start", "") or ""
    end = item.get("end", "") or ""

    nested_segments = extract_nested_segments(item.get("nested_content", []))
    nested_structured_text = collapse_structured_segments(
        nested_segments, max_chars=MAX_NESTED_SEARCH_CHARS)

    tags_text = " ".join(str(t) for t in tags)

    search_blob = " | ".join(
        part for part in [title, subtitle, host, tags_text, description, nested_structured_text, item_type] if part
    )

    compact_item = {
        "id": item.get("id"),
        "title": title,
        "subtitle": subtitle,
        "host": host,
        "description": description,
        "tags": tags,
        "type": item_type,
        "start": start,
        "end": end,
    }

    return {
        "raw": item,
        "compact": compact_item,
        "search_blob": search_blob,
        "normalized_blob": normalize_text(search_blob),
        "token_set": tokenize(search_blob),
        "nested_segments": nested_segments,
        "nested_structured_text": nested_structured_text,
    }


def score_encoded_item(query_hints: dict, encoded: dict) -> float:
    query_norm = query_hints["query_norm"]
    expanded_tokens = query_hints["expanded_tokens"]

    if not query_norm:
        return 0.0

    compact = encoded["compact"]
    blob = encoded["normalized_blob"][:MAX_BLOB_FOR_FUZZY]
    token_set = encoded["token_set"]

    title_norm = normalize_text(compact.get("title", ""))
    subtitle_norm = normalize_text(compact.get("subtitle", ""))
    host_norm = normalize_text(compact.get("host", ""))
    desc_norm = normalize_text(compact.get("description", ""))
    tags_norm = normalize_text(" ".join(compact.get("tags", [])))

    overlap = len(expanded_tokens & token_set)
    overlap_ratio = overlap / max(len(expanded_tokens), 1)

    seq_ratio = SequenceMatcher(None, query_norm, blob).ratio()
    contains_boost = 1.0 if query_norm in blob else 0.0

    title_hits = sum(1 for tok in expanded_tokens if tok in title_norm)
    subtitle_hits = sum(1 for tok in expanded_tokens if tok in subtitle_norm)
    host_hits = sum(1 for tok in expanded_tokens if tok in host_norm)
    desc_hits = sum(1 for tok in expanded_tokens if tok in desc_norm)
    tag_hits = sum(1 for tok in expanded_tokens if tok in tags_norm)

    field_boost = (
        (title_hits * 0.16) +
        (subtitle_hits * 0.08) +
        (host_hits * 0.06) +
        (desc_hits * 0.06) +
        (tag_hits * 0.05)
    )

    nested_scores = [score_segment(query_hints, seg)
                     for seg in encoded["nested_segments"]]
    best_nested_score = max(nested_scores) if nested_scores else 0.0

    nested_match_count = sum(1 for s in nested_scores if s >= 0.40)

    score = (
        (overlap_ratio * 0.30) +
        (seq_ratio * 0.20) +
        (contains_boost * 0.12) +
        field_boost +
        (best_nested_score * 0.40) +
        (min(nested_match_count, 3) * 0.07)
    )

    return round(score, 6)


# ------------------------------------------------------------------------------
# ROUTE
# ------------------------------------------------------------------------------
@ask_bp.route("/ai", methods=["POST"])
def ask_ai():
    logger.info("=" * 80)
    logger.info("📥 /ai REQUEST START")
    logger.info("=" * 80)

    try:
        logger.debug(
            f"Request Method: {request.method} | Path: {request.path} | Content-Type: {request.content_type}")

        raw_body = request.get_data(cache=True, as_text=True)
        data = request.get_json(silent=True) or {}

        logger.debug(f"Raw Request Body: {raw_body}")
        logger.debug(f"Parsed JSON Data: {data}")

        if data.get("ai_data_sharing_consent") is not True:
            return jsonify({
                "error": {
                    "code": "ai_data_sharing_consent_required",
                    "message": "AI Search data sharing consent is required.",
                }
            }), 400

        query = str(data.get("query", "")).strip()
        item_ids = data.get("item_ids", [])
        context_token_budget = parse_context_token_budget(
            data.get("max_tokens")
        )

        logger.info(f"Extracted Query: '{query}'")
        logger.info(f"Extracted Item IDs: {item_ids}")

        if not query:
            logger.warning("Aborting: No query provided")
            return jsonify({"error": "No query provided"}), 400
        if not isinstance(item_ids, list):
            logger.warning("Aborting: item_ids is not a list")
            return jsonify({"error": "item_ids must be an array"}), 400
        if not item_ids:
            logger.warning("Aborting: Empty item_ids list provided")
            return jsonify({
                "ai_overview": "No item_ids were provided in the request.",
                "citations": [],
                "ranked_item_ids": []
            }), 200

        # ------------------------------------------------------------------
        # 1) LOAD CURRENT CONTENT FROM THE SAME SOURCE AS /contentAPIURL
        # ------------------------------------------------------------------
        content_json = build_content_api_payload()
        pages = content_json.get("pages", [])
        logger.debug(
            f"Loaded {len(pages)} pages from current Content API sources.")

        if not isinstance(pages, list):
            logger.error("Invalid content API response: 'pages' is not a list")
            return jsonify({"error": "Invalid content API response"}), 500

        # ------------------------------------------------------------------
        # 2) FILTER TO USER item_ids
        # ------------------------------------------------------------------
        item_id_set = {str(item_id).strip() for item_id in item_ids if str(item_id).strip()}
        valid_items = [p for p in pages if str(p.get("id", "")).strip() in item_id_set]
        logger.info(
            f"Found {len(valid_items)} valid items matching provided item_ids.")

        if not valid_items:
            logger.warning("No matching items found for provided item_ids.")
            return jsonify({
                "ai_overview": "I could not find any matching items for the item_ids you sent.",
                "citations": [],
                "ranked_item_ids": []
            }), 200

        # ------------------------------------------------------------------
        # 3) QUERY UNDERSTANDING + LOCAL RETRIEVAL
        # ------------------------------------------------------------------
        retrieval_result = rank_items(
            query,
            valid_items,
            user_location=data.get("user_location") or data.get("coordinates"),
            max_context_items=AI_MAX_CONTEXT_ITEMS,
            min_context_items=AI_MIN_CONTEXT_ITEMS,
            max_context_tokens=context_token_budget,
        )
        top_scored = retrieval_result["context_rows"]
        llm_candidates = retrieval_result["llm_candidates"]
        debug_scores = [
            {
                "rank": row["rank"],
                "id": row["id"],
                "score": row["score"],
                "signals": row["signals"],
                "matched_concepts": sorted(row["matched_concepts"]),
                "protected_recall": row["protected_recall"],
            }
            for row in retrieval_result["ranked_rows"][:40]
        ]
        logger.debug(f"Local Pre-Rank Scores: {debug_scores}")

        logger.info(
            f"Selected {len(llm_candidates)} candidates for LLM processing; "
            f"context token estimate={retrieval_result['context_token_estimate']}.")
        llm_candidate_ids = [
            str(candidate.get("id", "")).strip()
            for candidate in llm_candidates
            if str(candidate.get("id", "")).strip()
        ]
        llm_candidate_id_set = set(llm_candidate_ids)
        candidate_by_id = {
            str(candidate.get("id", "")).strip(): candidate
            for candidate in llm_candidates
            if str(candidate.get("id", "")).strip()
        }

        # ------------------------------------------------------------------
        # 4) BUILD PROMPT
        # ------------------------------------------------------------------
        system_prompt = build_ai_system_prompt()
        user_content = build_ai_user_content(query, llm_candidates)

        if data.get("debug_retrieval") is True:
            logger.debug(
                f"=== SYSTEM PROMPT ===\n{system_prompt}\n=====================")
            logger.debug(
                f"=== USER CONTENT (LLM CANDIDATES) ===\n{user_content}\n=====================================")

        # ------------------------------------------------------------------
        # 5) CALL MODEL
        # ------------------------------------------------------------------
        logger.info(f"Calling OpenAI model: {MODEL_NAME}")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=800
        )

        raw = (resp.choices[0].message.content or "").strip()
        if data.get("debug_retrieval") is True:
            logger.debug(
                f"=== RAW LLM RESPONSE ===\n{raw}\n========================")

        # ------------------------------------------------------------------
        # 6) EXTRACT AND VALIDATE RANKED IDS
        # ------------------------------------------------------------------
        model_selected_item_ids = parse_model_ranked_ids(raw, llm_candidate_id_set)
        direct_model_selected_item_ids = dedupe_ranked_ids(
            model_selected_item_ids,
            llm_candidate_id_set,
        )
        ranked_item_ids = list(direct_model_selected_item_ids)
        ai_overview = clean_ai_overview(raw)
        model_answer_unavailable = answer_claims_unavailable(ai_overview)
        used_local_fallback = False
        pre_backfill_item_ids = list(ranked_item_ids)

        if not ranked_item_ids and not model_answer_unavailable:
            logger.warning(
                "Regex extraction failed or empty. Falling back to local top_scored order.")
            used_local_fallback = True
            ranked_item_ids = [row["id"] for row in top_scored]

        ranked_item_ids = dedupe_ranked_ids(ranked_item_ids, llm_candidate_id_set)
        if not model_answer_unavailable:
            pre_backfill_item_ids = list(ranked_item_ids)
            ranked_item_ids = append_local_backfill_ids(
                ranked_item_ids,
                top_scored,
                max_ids=10,
            )
        backfilled_item_ids = [
            item_id
            for item_id in ranked_item_ids
            if item_id not in set(pre_backfill_item_ids)
        ]

        logger.info(f"Final Ranked Item IDs: {ranked_item_ids}")

        if not ai_overview:
            ai_overview = "Here are the top matches based on your search."

        logger.debug(f"Cleaned AI Overview: '{ai_overview}'")

        # ------------------------------------------------------------------
        # 7) BUILD CITATIONS
        # ------------------------------------------------------------------
        citations = build_ai_citations(
            ranked_item_ids,
            valid_items,
            candidate_by_id,
        )

        final_response = {
            "ai_overview": ai_overview,
            "citations": citations,
            "ranked_item_ids": ranked_item_ids
        }

        if data.get("debug_retrieval") is True:
            final_response["retrieval_debug"] = serialize_retrieval_debug(
                retrieval_result
            )
            final_response["stage2_debug"] = {
                "llm_request": {
                    "api": "chat.completions",
                    "model": MODEL_NAME,
                    "temperature": 0.3,
                    "max_tokens": 800,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                },
                "llm_candidate_ids": llm_candidate_ids,
                "llm_candidates": llm_candidates,
                "raw_model_output": raw,
                "raw_model_selected_ids": model_selected_item_ids,
                "model_selected_ids": direct_model_selected_item_ids,
                "ids_before_backfill": pre_backfill_item_ids,
                "used_local_fallback": used_local_fallback,
                "answer_claims_unavailable": model_answer_unavailable,
                "backfilled_ids": backfilled_item_ids,
                "final_ranked_item_ids": ranked_item_ids,
                "citations": citations,
            }

        logger.debug(
            f"=== FINAL JSON RESPONSE ===\n{json.dumps(final_response, indent=2)}\n===========================")

        logger.info("📤 /ai REQUEST END - SUCCESS")
        logger.info("=" * 80 + "\n")

        return jsonify(final_response), 200

    except Exception as e:
        logger.error(
            f"Failed to generate AI response: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate AI response", "details": str(e)}), 500


def dedupe_ranked_ids(
    ranked_item_ids: list[str],
    allowed_item_ids: set[str],
) -> list[str]:
    seen = set()
    deduped = []

    for item_id in ranked_item_ids:
        if item_id not in allowed_item_ids or item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item_id)

    return deduped


def append_local_backfill_ids(
    ranked_item_ids: list[str],
    context_rows: list[dict],
    *,
    max_ids: int,
) -> list[str]:
    seen = set(ranked_item_ids)
    backfilled = list(ranked_item_ids)

    for row in context_rows:
        item_id = row.get("id")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        backfilled.append(item_id)
        if len(backfilled) >= max_ids:
            break

    return backfilled[:max_ids]


def build_ai_system_prompt() -> str:
    return (
        "You are answering questions about the UC Merced campus using only the "
        "supplied current campus records. Read each candidate's title, type, "
        "tags, description, evidence excerpts, timing, and relevant nested "
        "sections before answering. Use explicit information from the records, "
        "including conditions such as times, dates, restrictions, availability, "
        "and qualifications. Do not invent facts, but do not say information is "
        "unavailable when a supplied record explicitly contains it. If a record "
        "directly answers the question, use it even if its page type is not the "
        "obvious category. Treat the local rank as a useful prior, not final "
        "truth. Write a concise grounded answer in 1-3 sentences, preserving any "
        "important conditions, then include up to 10 supporting item IDs ranked "
        "by how directly they answer the user's question in this exact format:\n"
        "[IDS: id1, id2, id3, id4, ...]\n"
        "Do not output JSON."
    )


def build_ai_user_content(query: str, llm_candidates: list[dict]) -> str:
    return json.dumps({
        "query": query,
        "available_items": llm_candidates,
    }, indent=2, ensure_ascii=False)


def parse_model_ranked_ids(raw: str, allowed_item_ids: set[str]) -> list[str]:
    tag_match = re.search(r"\[IDS:\s*(.*?)\]", raw or "", re.IGNORECASE | re.DOTALL)
    if tag_match:
        found_ids = [
            item_id.strip()
            for item_id in tag_match.group(1).split(",")
            if item_id.strip()
        ]
        return [item_id for item_id in found_ids if item_id in allowed_item_ids]

    fallback_match = re.search(r"IDs?:\s*(.*)", raw or "", re.IGNORECASE)
    if fallback_match:
        found_ids = fallback_match.group(1).replace(",", " ").split()
        return [
            item_id.strip()
            for item_id in found_ids
            if item_id.strip() in allowed_item_ids
        ]

    return []


def clean_ai_overview(raw: str) -> str:
    overview = re.sub(r"\[IDS:.*?\]", "", raw or "", flags=re.IGNORECASE | re.DOTALL).strip()
    return re.sub(r"(?im)^IDs?:.*$", "", overview).strip()


def answer_claims_unavailable(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not normalized:
        return False
    negative_patterns = [
        r"\bdoes not specify\b",
        r"\bdo not specify\b",
        r"\bdoesn't specify\b",
        r"\bnot specified\b",
        r"\bnot available\b",
        r"\bno (?:relevant )?(?:information|records?|locations?|items?) (?:is |are )?(?:available|found|provided|specified)\b",
        r"\bi (?:could not|can't|cannot) find\b",
        r"\bunable to find\b",
    ]
    return any(re.search(pattern, normalized) for pattern in negative_patterns)


def build_ai_citations(
    ranked_item_ids: list[str],
    valid_items: list[dict],
    candidate_by_id: dict[str, dict],
) -> list[dict]:
    valid_item_by_id = {
        str(item.get("id", "")).strip(): item
        for item in valid_items
        if str(item.get("id", "")).strip()
    }
    citations = []

    for pid in ranked_item_ids:
        matched_item = valid_item_by_id.get(str(pid).strip())
        if not matched_item:
            continue

        candidate = candidate_by_id.get(str(pid).strip(), {})
        snippet = choose_citation_snippet(matched_item, candidate)
        citations.append({
            "page_id": pid,
            "title": matched_item.get("title", ""),
            "snippet": snippet,
        })

    return citations


def choose_citation_snippet(
    matched_item: dict,
    candidate: dict,
) -> str:
    for key in ("evidence_excerpt", "description", "nested_content_compact"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return trim_citation_snippet(value)

    if isinstance(matched_item.get("label"), dict) and matched_item["label"].get("name"):
        return trim_citation_snippet(str(matched_item["label"].get("name")))
    for key in ("location", "location_at", "host", "subtitle", "title"):
        value = matched_item.get(key)
        if isinstance(value, str) and value.strip():
            return trim_citation_snippet(value)

    return "Location not specified"


def trim_citation_snippet(value: str, limit: int = 280) -> str:
    snippet = re.sub(r"\s+", " ", value).strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def parse_context_token_budget(value) -> int:
    try:
        token_budget = int(value)
    except (TypeError, ValueError):
        token_budget = 4000

    return max(1500, min(token_budget, 8000))
