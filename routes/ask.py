from flask import Blueprint, request, jsonify
from flask import Blueprint, jsonify, request
import json
import os
import re
import base64
import requests
import unicodedata
from difflib import SequenceMatcher
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


# from openai import OpenAI

# Make sure these exist in your app:
# ask_bp = Blueprint("ask", __name__)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CONTENT_API_URL = "https://uc-merced-campus-event-api-backend.onrender.com/contentAPIURL"

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
MODEL_NAME = "gpt-4o"

MAX_DESC_CHARS = 220
MAX_CONTEXT_ITEMS = 8

# Keep full-ish structured nested content for local search only
MAX_NESTED_SEARCH_CHARS = 16000

# Small query-aware compact nested text for the LLM
MAX_NESTED_CONTEXT_CHARS = 1200

# Truncate fuzzy comparisons so they stay fast
MAX_BLOB_FOR_FUZZY = 1800
MAX_SEGMENT_FOR_FUZZY = 500

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

SKIP_NESTED_KEYS = {
    "image_urls", "pin_url", "source_url", "url", "urls", "href", "link", "links"
}

# ------------------------------------------------------------------------------
# WORD BANK / ALIASES
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
    "pav": {"pavilion"},
    "pavilion": {"pav"},

    "fri": {"friday"},
    "friday": {"fri"},
    "thu": {"thursday"},
    "thurs": {"thursday"},
    "thursday": {"thu", "thurs"},
    "wed": {"wednesday"},
    "wednesday": {"wed"},
    "tue": {"tuesday"},
    "tues": {"tuesday"},
    "tuesday": {"tue", "tues"},
    "mon": {"monday"},
    "monday": {"mon"},
    "sat": {"saturday"},
    "saturday": {"sat"},
    "sun": {"sunday"},
    "sunday": {"sun"},

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
}

PHRASE_WORD_BANK = {
    "gluten free": {"glutenfree", "gf"},
    "plant based": {"plantbased", "vegan", "vegetarian"},
}


# ------------------------------------------------------------------------------
# TEXT HELPERS
# ------------------------------------------------------------------------------
def normalize_text(value: str) -> str:
    """Lowercase, ASCII-normalize, remove noise, collapse whitespace."""
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
    return {tok for tok in normalize_text(text).split() if len(tok) > 1}


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
    """
    Compress nested text while preserving meaning.
    Examples:
      'Station: Lake Wok' -> 'station=Lake Wok'
      'Description: Choice of ...' -> 'desc=Choice of ...'
      'Calories: 940 Cal.' -> 'cal=940'
    """
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

        add_len = len(part) if not out else len(part) + 5  # " ### "
        if total + add_len > max_chars:
            break

        out.append(part)
        total += add_len

    return " ### ".join(out)


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
                if alias_norm in tokens:
                    found.add(canonical)
                    break

    return found


def build_query_hints(query: str) -> dict:
    query_norm = normalize_text(query)
    raw_tokens = set(query_norm.split())
    expanded_tokens = set(raw_tokens)

    # phrase expansion
    for phrase, expansions in PHRASE_WORD_BANK.items():
        phrase_norm = normalize_text(phrase)
        if phrase_norm in query_norm:
            expanded_tokens.update(expansions)

    # token expansion
    for tok in list(raw_tokens):
        expanded_tokens.update(TOKEN_WORD_BANK.get(tok, set()))

    days = detect_canonical_matches(query, DAY_ALIASES)
    meals = detect_canonical_matches(query, MEAL_ALIASES)
    locations = detect_canonical_matches(query, LOCATION_ALIASES)

    # include canonical forms and aliases in expanded token bag
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
    """
    Fallback walker for unexpected nested structures.
    Pulls string values while skipping known URL/image/link keys.
    """
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
    """
    Preserve nested content as compact, structured, delimited segments.
    Example segment:
      day=Friday || tab=Lunch || item=Blackened Salmon || station=Rufus || desc=Protein Choice ...
    """
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

                            # include any other string fields except skipped structural/media keys
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
                        # fallback for a tab with no sections
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
                # fallback for a day object with no tabs
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
        (overlap_ratio * 0.45) +
        (seq_ratio * 0.20) +
        (contains_boost * 0.15) +
        day_bonus +
        meal_bonus +
        (header_hits * 0.08)
    )

    return round(score, 6)


def build_query_aware_nested_excerpt(segments, query_hints, max_chars=MAX_NESTED_CONTEXT_CHARS) -> str:
    """
    Selects the most relevant structured nested segments for the LLM.
    Keeps delimiters and structure.
    """
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
        add_len = len(text) if not selected else len(text) + 5  # " ### "

        if selected_chars + add_len > max_chars:
            continue

        selected.append(seg)
        selected_chars += add_len
        group_counts[group_key] = group_counts.get(group_key, 0) + 1

        if len(selected) >= 8:
            break

    # fallback: if nothing scored, use the first segments in original order
    if not selected:
        for seg in sorted(segments, key=lambda s: s["idx"]):
            text = seg["text"]
            add_len = len(text) if not selected else len(text) + 5
            if selected_chars + add_len > max_chars:
                break
            selected.append(seg)
            selected_chars += add_len

    # restore original order for readability
    selected.sort(key=lambda s: s["idx"])

    return " ### ".join(seg["text"] for seg in selected)


# ------------------------------------------------------------------------------
# ITEM ENCODING / SCORING
# ------------------------------------------------------------------------------
def encode_item(item: dict) -> dict:
    title = item.get("title", "") or ""
    subtitle = item.get("subtitle", "") or ""
    host = item.get("host", "") or ""
    description = compact_description(item.get("description", "") or "")
    tags = item.get("tags", []) or []
    item_type = item.get("type", "") or ""
    start = item.get("start", "") or ""
    end = item.get("end", "") or ""

    nested_segments = extract_nested_segments(item.get("nested_content", []))
    nested_structured_text = collapse_structured_segments(
        nested_segments,
        max_chars=MAX_NESTED_SEARCH_CHARS
    )

    tags_text = " ".join(str(t) for t in tags)

    # this larger search blob is only for local ranking
    search_blob = " | ".join(
        part for part in [
            title,
            subtitle,
            host,
            tags_text,
            description,
            nested_structured_text,
            item_type
        ] if part
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
    """
    Local pre-ranker.
    Combines top-level field relevance + best nested segment relevance.
    """
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
    nested_match_count = sum(1 for s in nested_scores if s >= 0.60)

    score = (
        (overlap_ratio * 0.35) +
        (seq_ratio * 0.15) +
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
    print("\n" + "=" * 100)
    print("📥 /ai REQUEST START")
    print("=" * 100)

    try:
        # ------------------------------------------------------------------
        # 1) LOG REQUEST
        # ------------------------------------------------------------------
        print(f"Method: {request.method}")
        print(f"Path: {request.path}")
        print(f"Content-Type: {request.content_type}")

        raw_body = request.get_data(cache=True, as_text=True)
        print("\n--- RAW REQUEST BODY ---")
        print(raw_body if raw_body else "(empty body)")

        data = request.get_json(silent=True) or {}
        print("\n--- PARSED JSON BODY ---")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        query = str(data.get("query", "")).strip()
        item_ids = data.get("item_ids", [])

        print("\n--- EXTRACTED FIELDS ---")
        print(f"query: {query!r}")
        print(f"item_ids type: {type(item_ids).__name__}")
        print(
            f"item_ids count: {len(item_ids) if isinstance(item_ids, list) else 'N/A'}")
        print(f"item_ids value: {item_ids}")

        if not query:
            print("❌ No query provided")
            return jsonify({"error": "No query provided"}), 400

        if not isinstance(item_ids, list):
            print("❌ item_ids must be an array")
            return jsonify({"error": "item_ids must be an array"}), 400

        if not item_ids:
            empty_response = {
                "ai_overview": "No item_ids were provided in the request.",
                "citations": [],
                "ranked_item_ids": []
            }
            print("\n--- FINAL RESPONSE (NO item_ids) ---")
            print(json.dumps(empty_response, indent=2, ensure_ascii=False))
            print("=" * 100)
            print("📤 /ai REQUEST END")
            print("=" * 100 + "\n")
            return jsonify(empty_response), 200

        # ------------------------------------------------------------------
        # 2) QUERY HINTS / WORD BANK EXPANSION
        # ------------------------------------------------------------------
        query_hints = build_query_hints(query)

        print("\n--- QUERY HINTS / WORD BANK EXPANSION ---")
        print(json.dumps({
            "query_norm": query_hints["query_norm"],
            "raw_tokens": sorted(query_hints["raw_tokens"]),
            "expanded_tokens": sorted(query_hints["expanded_tokens"]),
            "days": sorted(query_hints["days"]),
            "meals": sorted(query_hints["meals"]),
            "locations": sorted(query_hints["locations"]),
        }, indent=2, ensure_ascii=False))

        # ------------------------------------------------------------------
        # 3) FETCH LIVE CONTENT
        # ------------------------------------------------------------------
        print("\n--- FETCHING CONTENT API ---")
        print(f"GET {CONTENT_API_URL}")

        content_resp = requests.get(CONTENT_API_URL, timeout=15)
        print(f"Content API status: {content_resp.status_code}")

        print("\n--- CONTENT API RAW TEXT (truncated) ---")
        print(content_resp.text[:5000])

        content_resp.raise_for_status()
        content_json = content_resp.json()

        print("\n--- CONTENT API PARSED JSON (truncated) ---")
        print(json.dumps(content_json, indent=2, ensure_ascii=False)[:5000])

        pages = content_json.get("pages", [])
        if not isinstance(pages, list):
            print("❌ Invalid content API response: 'pages' is not a list")
            return jsonify({
                "error": "Invalid content API response",
                "details": "'pages' was not a list"
            }), 500

        print("\n--- CONTENT API SUMMARY ---")
        print(f"Total pages returned: {len(pages)}")

        available_ids = [p.get("id") for p in pages if p.get("id")]
        print(f"Available page IDs ({len(available_ids)}):")
        print(json.dumps(available_ids, indent=2))

        # ------------------------------------------------------------------
        # 4) FILTER TO USER item_ids
        # ------------------------------------------------------------------
        valid_items = [p for p in pages if p.get("id") in item_ids]

        print("\n--- MATCHING ITEMS AFTER item_ids FILTER ---")
        print(f"Matched valid_items count: {len(valid_items)}")
        print("Matched IDs:")
        print(json.dumps([item.get("id") for item in valid_items], indent=2))

        if not valid_items:
            no_match_response = {
                "ai_overview": "I could not find any matching items for the item_ids you sent.",
                "citations": [],
                "ranked_item_ids": []
            }
            print("\n--- FINAL RESPONSE (NO MATCHING ITEMS) ---")
            print(json.dumps(no_match_response, indent=2, ensure_ascii=False))
            print("=" * 100)
            print("📤 /ai REQUEST END")
            print("=" * 100 + "\n")
            return jsonify(no_match_response), 200

        # ------------------------------------------------------------------
        # 5) ENCODE ITEMS WITH FULL STRUCTURED NESTED CONTENT
        # ------------------------------------------------------------------
        encoded_items = [encode_item(item) for item in valid_items]

        original_payload_size = len(
            json.dumps(valid_items, ensure_ascii=False))
        compact_payload_size = len(json.dumps(
            [e["compact"] for e in encoded_items], ensure_ascii=False))
        nested_structured_size = len(json.dumps(
            [e["nested_structured_text"] for e in encoded_items], ensure_ascii=False))

        print("\n--- CONTEXT SIZE BEFORE / AFTER COMPACTION ---")
        print(f"Original valid_items JSON chars:  {original_payload_size}")
        print(f"Compact encoded JSON chars:      {compact_payload_size}")
        print(f"Structured nested JSON chars:    {nested_structured_size}")

        print("\n--- STRUCTURED NESTED CONTENT PREVIEW ---")
        for enc in encoded_items[:5]:
            print(json.dumps({
                "id": enc["compact"]["id"],
                "title": enc["compact"]["title"],
                "nested_segment_count": len(enc["nested_segments"]),
                "nested_structured_preview": enc["nested_structured_text"][:700]
            }, indent=2, ensure_ascii=False))

        # ------------------------------------------------------------------
        # 6) LOCAL PRE-RANK
        # ------------------------------------------------------------------
        scored = []
        for enc in encoded_items:
            local_score = score_encoded_item(query_hints, enc)
            scored.append({
                "score": local_score,
                "encoded": enc
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        print("\n--- LOCAL PRE-RANK RESULTS ---")
        for row in scored:
            c = row["encoded"]["compact"]
            best_excerpt = build_query_aware_nested_excerpt(
                row["encoded"]["nested_segments"],
                query_hints,
                max_chars=350
            )

            print(json.dumps({
                "score": row["score"],
                "id": c["id"],
                "title": c["title"],
                "subtitle": c.get("subtitle", ""),
                "tags": c.get("tags", []),
                "nested_segment_count": len(row["encoded"]["nested_segments"]),
                "query_aware_nested_preview": best_excerpt
            }, indent=2, ensure_ascii=False))

        # ------------------------------------------------------------------
        # 7) BUILD LLM CANDIDATES
        # ------------------------------------------------------------------
        top_scored = scored[:MAX_CONTEXT_ITEMS]
        llm_candidates = []

        for row in top_scored:
            enc = row["encoded"]
            compact = dict(enc["compact"])  # copy

            nested_compact = build_query_aware_nested_excerpt(
                enc["nested_segments"],
                query_hints,
                max_chars=MAX_NESTED_CONTEXT_CHARS
            )

            if nested_compact:
                compact["nested_content_compact"] = nested_compact

            llm_candidates.append(compact)

        print("\n--- TOP ITEMS SENT TO LLM ---")
        print(json.dumps(llm_candidates, indent=2, ensure_ascii=False))

        print("\n--- TOP ITEM IDS SENT TO LLM ---")
        print([item["id"] for item in llm_candidates])

        reduced_payload_size = len(json.dumps(
            llm_candidates, ensure_ascii=False))
        print(f"\nReduced LLM context JSON chars: {reduced_payload_size}")

        # ------------------------------------------------------------------
        # 8) BUILD PROMPT
        # ------------------------------------------------------------------
        system_prompt = (
            "You are an intelligent campus content assistant. "
            "Answer the user's query using ONLY the provided campus items. "
            "Some items include nested_content_compact, which is a structured compact collapse "
            "of deeper nested content such as menus or schedules using delimiters like "
            "'day=Friday || tab=Lunch || item=Blackened Salmon || desc=...'. "
            "Use that compact field when it is relevant to the query. "
            "Be concise and helpful in 2-3 sentences.\n\n"
            "At the end, output the relevant item IDs ranked best to worst in this exact format:\n"
            "[IDS: id1, id2, id3]\n"
            "Do not output JSON."
        )

        user_content = json.dumps({
            "query": query,
            "available_items": llm_candidates
        }, indent=2, ensure_ascii=False)

        print("\n" + "-" * 100)
        print("🧠 SYSTEM PROMPT")
        print("-" * 100)
        print(system_prompt)

        print("\n" + "-" * 100)
        print("🧠 USER CONTENT TO MODEL")
        print("-" * 100)
        print(user_content)

        # ------------------------------------------------------------------
        # 9) CALL MODEL
        # ------------------------------------------------------------------
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

        print("\n" + "-" * 100)
        print("🤖 RAW LLM RESPONSE")
        print("-" * 100)
        print(raw if raw else "(empty model response)")

        # ------------------------------------------------------------------
        # 10) EXTRACT RANKED IDS
        # ------------------------------------------------------------------
        ranked_item_ids = []

        tag_match = re.search(r"\[IDS:\s*(.*?)\]", raw, re.IGNORECASE)
        if tag_match:
            found_ids = [x.strip()
                         for x in tag_match.group(1).split(",") if x.strip()]
            ranked_item_ids = [x for x in found_ids if x in item_ids]

            print("\n--- ID EXTRACTION: TAG MATCH ---")
            print(f"Found IDs: {found_ids}")
            print(f"Validated ranked_item_ids: {ranked_item_ids}")

        elif re.search(r"IDs?:\s*(.*)", raw, re.IGNORECASE):
            fallback_match = re.search(r"IDs?:\s*(.*)", raw, re.IGNORECASE)
            found_ids = fallback_match.group(1).replace(",", " ").split()
            ranked_item_ids = [x.strip()
                               for x in found_ids if x.strip() in item_ids]

            print("\n--- ID EXTRACTION: FALLBACK MATCH ---")
            print(f"Found IDs: {found_ids}")
            print(f"Validated ranked_item_ids: {ranked_item_ids}")

        if not ranked_item_ids:
            ranked_item_ids = [row["encoded"]["compact"]["id"]
                               for row in top_scored]
            print("\n--- ID EXTRACTION FALLBACK: USE LOCAL PRE-RANK ---")
            print(f"ranked_item_ids: {ranked_item_ids}")

        ai_overview = re.sub(r"\[IDS:.*?\]", "", raw,
                             flags=re.IGNORECASE).strip()
        ai_overview = re.sub(r"(?i)IDs?:.*", "", ai_overview).strip()

        if not ai_overview:
            ai_overview = "Here are the top matches based on your search."

        # ------------------------------------------------------------------
        # 11) BUILD CITATIONS
        # ------------------------------------------------------------------
        citations = []
        for pid in ranked_item_ids:
            matched_item = next(
                (item for item in valid_items if item.get("id") == pid), None)
            if not matched_item:
                continue

            # Return the location name of the building.
            # Falls back to "host" if the specific "location" field is absent or empty.
            snippet = matched_item.get("location", matched_item.get(
                "host", "Location not specified"))

            citations.append({
                "page_id": pid,
                "title": matched_item.get("title", ""),
                "snippet": snippet
            })

        final_response = {
            "ai_overview": ai_overview,
            "citations": citations,
            "ranked_item_ids": ranked_item_ids
        }

        print("\n--- FINAL RESPONSE PAYLOAD ---")
        print(json.dumps(final_response, indent=2, ensure_ascii=False))

        print("=" * 100)
        print("📤 /ai REQUEST END")
        print("=" * 100 + "\n")

        return jsonify(final_response), 200

    except requests.exceptions.RequestException as e:
        print("\n" + "!" * 100)
        print("💥 CONTENT API REQUEST ERROR")
        print("!" * 100)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("!" * 100 + "\n")

        return jsonify({
            "error": "Failed to fetch content API data",
            "details": str(e)
        }), 502

    except Exception as e:
        print("\n" + "!" * 100)
        print("💥 /ai ERROR")
        print("!" * 100)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("!" * 100 + "\n")

        return jsonify({
            "error": "Failed to generate AI response",
            "details": str(e)
        }), 500
