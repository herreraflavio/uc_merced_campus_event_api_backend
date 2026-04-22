import logging
import json
import os
import re
import base64
import requests
import math
import atexit
import unicodedata
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher

from flask import Blueprint, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

try:
    import tiktoken
except ImportError:
    tiktoken = None

# --- NLP AND SCHEDULING ---
from apscheduler.schedulers.background import BackgroundScheduler
import nltk
from nltk.stem import WordNetLemmatizer

from helper.normalize_location import normalize_event_location

# Ensure nltk wordnet is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ask_bp = Blueprint('ask', __name__)

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

CONTENT_API_URL = "https://uc-merced-campus-event-api-backend.onrender.com/contentAPIURL"

# ------------------------------------------------------------------------------
# CONFIG & STATE
# ------------------------------------------------------------------------------
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "gpt-5.4")
AI_REASONING_EFFORT = os.getenv("AI_REASONING_EFFORT", "low")
AI_MAX_TOTAL_COST_USD = float(os.getenv("AI_MAX_TOTAL_COST_USD", "0.04"))
AI_TARGET_TOTAL_COST_USD = float(
    os.getenv("AI_TARGET_TOTAL_COST_USD", "0.032"))
AI_INPUT_COST_PER_1M = 2.50
AI_OUTPUT_COST_PER_1M = 15.00
AI_MAX_RANKED_IDS = 10
AI_DEFAULT_MAX_COMPLETION_TOKENS = 450
AI_MIN_COMPLETION_TOKENS = 180
AI_MAX_ITEMS_PER_REQUEST = 250
AI_COMPRESSION_STEPS = [
    {
        "name": "balanced",
        "title": 96,
        "subtitle": 72,
        "location": 60,
        "host": 48,
        "desc": 120,
        "nested": 120,
        "tags": 6,
        "include_subtitle": True,
        "include_host": True,
        "include_desc": True,
        "include_nested": True,
    },
    {
        "name": "tight",
        "title": 90,
        "subtitle": 56,
        "location": 56,
        "host": 0,
        "desc": 80,
        "nested": 70,
        "tags": 5,
        "include_subtitle": True,
        "include_host": False,
        "include_desc": True,
        "include_nested": True,
    },
    {
        "name": "lean",
        "title": 84,
        "subtitle": 0,
        "location": 52,
        "host": 0,
        "desc": 55,
        "nested": 0,
        "tags": 4,
        "include_subtitle": False,
        "include_host": False,
        "include_desc": True,
        "include_nested": False,
    },
    {
        "name": "bare",
        "title": 78,
        "subtitle": 0,
        "location": 46,
        "host": 0,
        "desc": 0,
        "nested": 0,
        "tags": 3,
        "include_subtitle": False,
        "include_host": False,
        "include_desc": False,
        "include_nested": False,
    },
]

MAX_DESC_CHARS = 220
MAX_CONTEXT_ITEMS = 25
MIN_CONTEXT_ITEMS = 4
MAX_NESTED_SEARCH_CHARS = 16000
MAX_NESTED_CONTEXT_CHARS = 1200
MAX_BLOB_FOR_FUZZY = 1800
MAX_SEGMENT_FOR_FUZZY = 500

MAX_CONTEXT_TOKENS = 3000
MAX_COMPLETION_TOKENS = 800

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SKIP_NESTED_KEYS = {"image_urls", "pin_url",
                    "source_url", "url", "urls", "href", "link", "links"}

# In-memory stores
GLOBAL_CONTENT_CACHE = []
DYNAMIC_EXPANSIONS = {}
message_history = defaultdict(lambda: deque(maxlen=10))

# Load static events for the /ask/events route
EVENTS = []
EVENTS_PATH = os.path.join(os.getcwd(), "events.json")
if os.path.exists(EVENTS_PATH):
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            EVENTS = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load events.json: {e}")
        EVENTS = []

# ------------------------------------------------------------------------------
# STOP WORDS & STATIC WORD BANK
# ------------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "were", "it", "this", "that", "these", "those",
    "then", "just", "so", "than", "such", "both", "through", "about", "while",
    "during", "what", "they", "we", "he", "she", "if", "because", "as", "when",
    "where", "how", "who", "which", "be", "has", "have", "had", "do", "does", "did"
}

DAY_ALIASES = {
    "monday": {"monday", "mon"}, "tuesday": {"tuesday", "tue", "tues"},
    "wednesday": {"wednesday", "wed"}, "thursday": {"thursday", "thu", "thur", "thurs"},
    "friday": {"friday", "fri"}, "saturday": {"saturday", "sat"}, "sunday": {"sunday", "sun"},
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
# HELPERS (Vision & Chat)
# ------------------------------------------------------------------------------
def extract_json(raw: str) -> dict:
    """Clean up model output and return the first {...} JSON object inside."""
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.strip("`").strip()

    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

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

    if not isinstance(event_json, dict):
        raise ValueError(
            f"Expected JSON object but got {type(event_json).__name__}")

    loc = event_json.get("location")
    event_json["location_at"] = loc
    if loc is not None:
        normalized = normalize_event_location(loc)
        event_json["location"] = normalized if normalized is not None else loc

    return event_json


def approximate_token_count(messages):
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(content)
            except Exception:
                content = str(content)
        total += len(content) // 4
    return total


# ------------------------------------------------------------------------------
# TEXT & NLP HELPERS
# ------------------------------------------------------------------------------
def make_singular(word: str) -> str:
    return lemmatizer.lemmatize(word)


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
    return re.sub(r"\s+", " ", text).strip()


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
    return text.strip(" |;,-")


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
# BACKGROUND CACHING & WORD BANK GENERATION
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


def fetch_and_cache_content():
    global GLOBAL_CONTENT_CACHE
    try:
        logger.info("Fetching fresh content from API in background...")
        resp = requests.get(CONTENT_API_URL, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("pages", [])
        if pages:
            generate_dynamic_word_bank(pages)
            GLOBAL_CONTENT_CACHE = pages
            logger.info(f"Successfully cached {len(pages)} pages.")
    except Exception as e:
        logger.error(f"Background fetch failed: {e}")


# Start the background job
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_and_cache_content, trigger="interval", minutes=5)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# Fire once on startup to populate
fetch_and_cache_content()


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
        if normalize_text(phrase) in query_norm:
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
        "query": query, "query_norm": query_norm, "raw_tokens": raw_tokens,
        "expanded_tokens": expanded_tokens, "days": days, "meals": meals, "locations": locations,
    }


def canonicalize_from_aliases(value: str, alias_map: dict) -> str:
    matches = detect_canonical_matches(value or "", alias_map)
    return sorted(matches)[0] if matches else normalize_text(value or "")


# ------------------------------------------------------------------------------
# NESTED CONTENT & ENCODING
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


def make_segment_text(day_title, tab_title, header, extra_parts) -> str:
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
                            extra_parts = section.get("bullets", [])[:]
                            for key, value in section.items():
                                if key not in {"header", "bullets"} and key not in SKIP_NESTED_KEYS and isinstance(value, str):
                                    extra_parts.append(f"{key}={value}")

                            text = make_segment_text(
                                day_title, tab_title, header, extra_parts)
                            if text:
                                segments.append({
                                    "idx": seq, "day": day_title, "tab": tab_title, "header": header,
                                    "text": text, "normalized": normalize_text(text), "token_set": tokenize(text),
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
                                "idx": seq, "day": day_title, "tab": tab_title, "header": "",
                                "text": text, "normalized": normalize_text(text), "token_set": tokenize(text),
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
                        "idx": seq, "day": day_title, "tab": "", "header": "", "text": text,
                        "normalized": normalize_text(text), "token_set": tokenize(text),
                        "day_canonical": canonicalize_from_aliases(day_title, DAY_ALIASES), "tab_canonical": "",
                    })
                    seq += 1
    else:
        fallback_parts = []
        gather_generic_strings(nested_content, fallback_parts)
        text = make_segment_text("", "", "", fallback_parts)
        if text:
            segments.append({
                "idx": seq, "day": "", "tab": "", "header": "", "text": text,
                "normalized": normalize_text(text), "token_set": tokenize(text),
                "day_canonical": "", "tab_canonical": "",
            })
    return segments


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

    day_bonus = 0.60 if query_hints["days"] and segment["day_canonical"] in query_hints["days"] else 0.0
    meal_bonus = 0.50 if query_hints["meals"] and segment["tab_canonical"] in query_hints["meals"] else 0.0

    header_hits = sum(1 for tok in expanded_tokens if tok in normalize_text(
        segment.get("header", "")))

    return round((overlap_ratio * 0.35) + (seq_ratio * 0.25) + (contains_boost * 0.15) + day_bonus + meal_bonus + (header_hits * 0.08), 6)


def build_query_aware_nested_excerpt(segments, query_hints, max_chars=MAX_NESTED_CONTEXT_CHARS) -> str:
    if not segments:
        return ""
    scored = sorted([(score_segment(query_hints, seg), seg["idx"], seg)
                    for seg in segments], key=lambda x: (x[0], -x[1]), reverse=True)

    selected, selected_chars, group_counts = [], 0, {}
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

    return " ### ".join(seg["text"] for seg in sorted(selected, key=lambda s: s["idx"]))


def encode_item(item: dict) -> dict:
    title, subtitle, host = item.get("title", "") or "", item.get(
        "subtitle", "") or "", item.get("host", "") or ""
    description, tags = item.get(
        "description", "") or "", item.get("tags", []) or []
    item_type, start, end = item.get("type", "") or "", item.get(
        "start", "") or "", item.get("end", "") or ""

    nested_segments = extract_nested_segments(item.get("nested_content", []))
    nested_structured_text = join_with_limit(
        [s["text"] for s in nested_segments], MAX_NESTED_SEARCH_CHARS)

    search_blob = " | ".join(part for part in [title, subtitle, host, " ".join(
        str(t) for t in tags), description, nested_structured_text, item_type] if part)

    return {
        "raw": item,
        "compact": {"id": item.get("id"), "title": title, "subtitle": subtitle, "host": host, "description": description, "tags": tags, "type": item_type, "start": start, "end": end},
        "search_blob": search_blob, "normalized_blob": normalize_text(search_blob), "token_set": tokenize(search_blob),
        "nested_segments": nested_segments, "nested_structured_text": nested_structured_text,
    }


def score_encoded_item(query_hints: dict, encoded: dict) -> float:
    query_norm = query_hints["query_norm"]
    if not query_norm:
        return 0.0

    compact = encoded["compact"]
    expanded_tokens = query_hints["expanded_tokens"]
    blob = encoded["normalized_blob"][:MAX_BLOB_FOR_FUZZY]

    overlap = len(expanded_tokens & encoded["token_set"])
    overlap_ratio = overlap / max(len(expanded_tokens), 1)

    field_boost = (
        (sum(1 for tok in expanded_tokens if tok in normalize_text(compact.get("title", ""))) * 0.16) +
        (sum(1 for tok in expanded_tokens if tok in normalize_text(compact.get("subtitle", ""))) * 0.08) +
        (sum(1 for tok in expanded_tokens if tok in normalize_text(compact.get("host", ""))) * 0.06) +
        (sum(1 for tok in expanded_tokens if tok in normalize_text(compact.get("description", ""))) * 0.06) +
        (sum(1 for tok in expanded_tokens if tok in normalize_text(
            " ".join(compact.get("tags", [])))) * 0.05)
    )

    nested_scores = [score_segment(query_hints, seg)
                     for seg in encoded["nested_segments"]]
    best_nested_score = max(nested_scores) if nested_scores else 0.0

    return round((overlap_ratio * 0.30) + (SequenceMatcher(None, query_norm, blob).ratio() * 0.20) +
                 ((1.0 if query_norm in blob else 0.0) * 0.12) + field_boost + (best_nested_score * 0.40) +
                 (min(sum(1 for s in nested_scores if s >= 0.40), 3) * 0.07), 6)


# ------------------------------------------------------------------------------
# LIGHTWEIGHT /ai HELPERS
# ------------------------------------------------------------------------------
def extract_first_json_object(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty model response")

    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

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

    parsed = json.loads(raw[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


_TOKENIZER_CACHE = {}


def count_text_tokens(text: str, model: str = AI_MODEL_NAME) -> int:
    if not text:
        return 0

    if tiktoken is not None:
        try:
            enc = _TOKENIZER_CACHE.get(model)
            if enc is None:
                try:
                    enc = tiktoken.encoding_for_model(model)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
                _TOKENIZER_CACHE[model] = enc
            return len(enc.encode(text))
        except Exception:
            pass

    return max(1, len(text) // 4)


def estimate_chat_prompt_tokens(messages, model: str = AI_MODEL_NAME) -> int:
    total = 3
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(
                    content, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                content = str(content)
        total += count_text_tokens(content, model=model) + 6
    return total


def usd_for_input_tokens(token_count: int) -> float:
    return (token_count / 1_000_000) * AI_INPUT_COST_PER_1M


def usd_for_output_tokens(token_count: int) -> float:
    return (token_count / 1_000_000) * AI_OUTPUT_COST_PER_1M


def estimate_total_cost_usd(prompt_tokens: int, max_completion_tokens: int) -> float:
    return usd_for_input_tokens(prompt_tokens) + usd_for_output_tokens(max_completion_tokens)


def compact_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = strip_urls(value or "")
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip(" ,;:-") + "…"


def get_item_location_text(item: dict) -> str:
    if isinstance(item.get("label"), dict) and item["label"].get("name"):
        return str(item["label"].get("name") or "").strip()
    if item.get("location"):
        return str(item.get("location") or "").strip()
    if item.get("subtitle"):
        return str(item.get("subtitle") or "").strip()
    return ""


def get_item_time_text(item: dict) -> str:
    start = str(item.get("start") or "").strip()
    end = str(item.get("end") or "").strip()
    if start and end:
        return f"{start} | {end}"
    return start or end


def build_nested_preview(item: dict, max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    segments = extract_nested_segments(item.get("nested_content", []))
    if segments:
        preview = " ### ".join(seg.get("text", "")
                               for seg in segments[:3] if seg.get("text"))
        return compact_text(preview, max_chars)

    fragments = []
    gather_generic_strings(item.get("nested_content", []), fragments)
    preview = " | ".join(fragments[:8])
    return compact_text(preview, max_chars)


def build_compact_ai_item(item: dict, tier: dict) -> dict:
    compact = {
        "id": item.get("id"),
        "title": compact_text(item.get("title", ""), tier["title"]),
        "type": compact_text(item.get("type", ""), 24),
        "location": compact_text(get_item_location_text(item), tier["location"]),
        "time": compact_text(get_item_time_text(item), 72),
    }

    if tier.get("include_subtitle"):
        compact["subtitle"] = compact_text(
            item.get("subtitle", ""), tier["subtitle"])

    if tier.get("include_host"):
        compact["host"] = compact_text(item.get("host", ""), tier["host"])

    raw_tags = item.get("tags") or []
    tags = []
    for tag in raw_tags[: tier["tags"]]:
        t = compact_text(str(tag), 24)
        if t:
            tags.append(t)
    if tags:
        compact["tags"] = tags

    if tier.get("include_desc"):
        compact["description"] = compact_text(
            item.get("description", ""), tier["desc"])

    if tier.get("include_nested"):
        nested_preview = build_nested_preview(item, tier["nested"])
        if nested_preview:
            compact["nested"] = nested_preview

    return {k: v for k, v in compact.items() if v not in (None, "", [], {})}


def build_ai_messages(query: str, valid_items: list):
    system_prompt = (
        "You are a campus informant. Use only the provided compact campus items. "
        "Answer the user's query with a short helpful summary and rank the best matching item IDs. "
        "Return valid JSON only with this schema: "
        '{"summary":"string","ranked_ids":["id1","id2"]}. '
        "summary must be 1-3 short sentences. ranked_ids must contain at most 10 IDs, ordered best to worst. "
        "Do not invent IDs. Ignore irrelevant items."
    )

    def make_messages(items_for_prompt):
        user_content = json.dumps(
            {"query": query, "items": items_for_prompt},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def try_budget(max_budget_usd: float):
        for tier in AI_COMPRESSION_STEPS:
            compact_items = [
                build_compact_ai_item(item, tier)
                for item in valid_items[:AI_MAX_ITEMS_PER_REQUEST]
            ]
            messages = make_messages(compact_items)
            prompt_tokens = estimate_chat_prompt_tokens(
                messages, model=AI_MODEL_NAME)

            remaining_usd = max_budget_usd - \
                usd_for_input_tokens(prompt_tokens)
            if remaining_usd <= 0:
                continue

            max_completion_tokens = min(
                AI_DEFAULT_MAX_COMPLETION_TOKENS,
                int((remaining_usd / AI_OUTPUT_COST_PER_1M) * 1_000_000),
            )
            if max_completion_tokens < AI_MIN_COMPLETION_TOKENS:
                continue

            return {
                "messages": messages,
                "compact_items": compact_items,
                "tier": tier["name"],
                "prompt_tokens": prompt_tokens,
                "max_completion_tokens": max_completion_tokens,
                "estimated_cost_usd": estimate_total_cost_usd(prompt_tokens, max_completion_tokens),
            }
        return None

    plan = try_budget(AI_TARGET_TOTAL_COST_USD)
    if plan is not None:
        return plan

    plan = try_budget(AI_MAX_TOTAL_COST_USD)
    if plan is not None:
        return plan

    return None


def fallback_rank_ids(query: str, compact_items: list, allowed_ids: list) -> list:
    query_tokens = tokenize(query)
    scored = []
    for item in compact_items:
        blob = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        score = len(query_tokens & tokenize(blob))
        scored.append((score, item.get("id")))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [item_id for score, item_id in scored if item_id in allowed_ids]
    if not ranked:
        ranked = [item.get("id")
                  for item in compact_items if item.get("id") in allowed_ids]
    return ranked[:AI_MAX_RANKED_IDS]


def build_citations(ranked_item_ids: list, valid_items: list) -> list:
    citations = []
    for pid in ranked_item_ids:
        matched_item = next(
            (item for item in valid_items if item.get("id") == pid), None)
        if not matched_item:
            continue

        snippet = ""
        if isinstance(matched_item.get("label"), dict) and matched_item["label"].get("name"):
            snippet = str(matched_item["label"].get("name") or "").strip()
        elif matched_item.get("subtitle"):
            snippet = str(matched_item.get("subtitle") or "").strip()
        elif matched_item.get("location"):
            snippet = str(matched_item.get("location") or "").strip()
        elif matched_item.get("host"):
            snippet = str(matched_item.get("host") or "").strip()

        if not snippet:
            snippet = "Location not specified"

        citations.append({
            "page_id": pid,
            "title": matched_item.get("title", ""),
            "snippet": snippet,
        })

    return citations


# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@ask_bp.route("/endpoints", methods=["GET"])
def root():
    return jsonify({"ok": True, "endpoints": ["/ask/vision (POST)", "/ask/events (POST)", "/ai (POST)"]})


@ask_bp.route("/ask", methods=["POST"])
def ask_vision():
    """
    Multipart form-data with a file field named 'file'.
    Returns strict JSON extracted from the image.
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
            return jsonify({"error": "Failed to extract JSON", "raw_response": raw}), 500

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Model did not return valid JSON"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ask_bp.route("/ask/events", methods=["POST"])
def ask_events():
    """
    Chatbot endpoint for events with memory.
    """
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "default")
    question = data.get("question", "")
    tags = data.get("tags", [])

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "No question provided"}), 400

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

    history = message_history[user_id]
    history.append({"role": "user", "content": question})

    context_prompt = (
        f"User asked: \"{question}\"\n\n"
        f"Here is a list of events:\n{json.dumps(filtered_events, ensure_ascii=False)}"
    )

    messages = [system_message] + \
        list(history) + [{"role": "user", "content": context_prompt}]

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
        history.append({"role": "assistant", "content": reply})

        event_ids = re.findall(r"event\d{3}", reply)
        matched_events = [
            event for event in EVENTS if event.get("id") in set(event_ids)]

        return jsonify({"response": reply, "matched_events": matched_events})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ask_bp.route("/ai", methods=["POST"])
def ask_ai():
    """
    Cost-capped /ai route that sends compact campus items directly to GPT-5.4.
    """
    logger.info("=" * 80)
    logger.info("📥 /ai REQUEST START")
    logger.info("=" * 80)

    try:
        data = request.get_json(silent=True) or {}
        query = str(data.get("query", "")).strip()
        item_ids = data.get("item_ids", [])

        if not query:
            return jsonify({"error": "No query provided"}), 400
        if not isinstance(item_ids, list):
            return jsonify({"error": "item_ids must be an array"}), 400
        if not item_ids:
            return jsonify({
                "ai_overview": "No item_ids were provided in the request.",
                "citations": [],
                "ranked_item_ids": [],
            }), 200

        pages = GLOBAL_CONTENT_CACHE
        if not pages:
            logger.warning(
                "Cache is empty. Running synchronous fallback fetch.")
            fetch_and_cache_content()
            pages = GLOBAL_CONTENT_CACHE

        valid_items = [p for p in pages if p.get("id") in item_ids]
        if not valid_items:
            return jsonify({
                "ai_overview": "I could not find any matching items for the item_ids you sent.",
                "citations": [],
                "ranked_item_ids": [],
            }), 200

        ai_plan = build_ai_messages(query, valid_items)
        if ai_plan is None:
            logger.warning(
                "Unable to compress request enough to stay under the configured /ai budget.")
            return jsonify({
                "ai_overview": "This request is too large to process under the configured cost cap without dropping item detail.",
                "citations": [],
                "ranked_item_ids": [],
            }), 413

        logger.info(
            "AI plan | items=%s | tier=%s | prompt_tokens_est=%s | max_completion_tokens=%s | est_cost_usd=%.6f",
            len(valid_items),
            ai_plan["tier"],
            ai_plan["prompt_tokens"],
            ai_plan["max_completion_tokens"],
            ai_plan["estimated_cost_usd"],
        )

        resp = client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=ai_plan["messages"],
            response_format={"type": "json_object"},
            reasoning_effort=AI_REASONING_EFFORT,
            max_completion_tokens=ai_plan["max_completion_tokens"],
        )

        raw_content = (resp.choices[0].message.content or "").strip()
        ranked_item_ids = []
        ai_overview = "Here are the top matches based on your search."

        try:
            llm_output = extract_first_json_object(raw_content)
            ai_overview = str(llm_output.get("summary") or ai_overview).strip()
            ranked_item_ids = [
                item_id
                for item_id in llm_output.get("ranked_ids", [])
                if item_id in item_ids
            ][:AI_MAX_RANKED_IDS]
        except Exception:
            logger.warning(
                "Failed to parse model JSON cleanly. Falling back to lexical ranking.")

        if not ranked_item_ids:
            ranked_item_ids = fallback_rank_ids(
                query, ai_plan["compact_items"], item_ids)

        citations = build_citations(ranked_item_ids, valid_items)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens_used = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens_used = int(
                getattr(usage, "completion_tokens", 0) or 0)
            actual_cost_usd = estimate_total_cost_usd(
                prompt_tokens_used, completion_tokens_used)
            logger.info(
                "AI usage | prompt_tokens=%s | completion_tokens=%s | actual_cost_usd=%.6f",
                prompt_tokens_used,
                completion_tokens_used,
                actual_cost_usd,
            )
            if actual_cost_usd > AI_MAX_TOTAL_COST_USD:
                logger.warning(
                    "AI request exceeded configured budget | actual_cost_usd=%.6f | cap_usd=%.6f",
                    actual_cost_usd,
                    AI_MAX_TOTAL_COST_USD,
                )

        final_response = {
            "ai_overview": ai_overview,
            "citations": citations,
            "ranked_item_ids": ranked_item_ids,
        }

        logger.info("📤 /ai REQUEST END - SUCCESS")
        return jsonify(final_response), 200

    except Exception as e:
        logger.error(
            f"Failed to generate AI response: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate AI response", "details": str(e)}), 500
