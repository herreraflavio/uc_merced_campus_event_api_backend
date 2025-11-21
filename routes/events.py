# events.py
import os
import json
import re
from datetime import datetime, timezone, timedelta

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import uuid

import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from html import unescape
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Blueprint & constants
# ─────────────────────────────────────────

events_bp = Blueprint("events", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

PACIFIC = ZoneInfo("America/Los_Angeles")

# LiveWhale RSS
FEED_URL = "https://events.ucmerced.edu/live/rss/events/header/All%20Events"

# Presence API endpoints
PRESENCE_EVENTS_URL = "https://api.presence.io/ucmerced/v1/events"
PRESENCE_CAMPUS_URL = "https://api.presence.io/ucmerced/v1/app/campus"

# Location data (your locations.json)
LOCATIONS_JSON_PATH = os.path.join(os.path.dirname(__file__), "locations.json")

# Whitelist of allowed location tokens (from first names only)
LOCATION_KEYWORDS_LOWER: set[str] = set()
# For pretty output, map lowercase token -> canonical-cased token from first names
LOCATION_TOKEN_CANONICAL: dict[str, str] = {}

# Full-name variant map:
#   normalized("KL") -> "Kolligian Library"
#   normalized("Library (KL)") -> "Kolligian Library"
LOCATION_VARIANT_TO_CANONICAL: dict[str, str] = {}

# Presence cache config
PRESENCE_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "presence_events_cache.json"
)
PRESENCE_CACHE_TTL = timedelta(hours=6)

# In-process cache
PRESENCE_CACHE: list[dict] | None = None
PRESENCE_CACHE_EXPIRES_AT: datetime | None = None


# ─────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_html_to_text(html_text: str) -> str:
    """Roughly convert Presence HTML descriptions into plain text."""
    text = unescape(html_text or "")
    text = re.sub(r"<[^>]+>", "", text)  # strip tags
    return " ".join(text.split())        # collapse whitespace


def _tokenize_words(s: str) -> list[str]:
    """
    Split a string into alphanumeric tokens.
    Example: "OSI Granite Pass 163" -> ["OSI", "Granite", "Pass", "163"]
    """
    if not s:
        return []
    return re.findall(r"[A-Za-z0-9]+", s)


# Words we want to ignore when matching (you can tweak this list)
_LOCATION_STOPWORDS = {
    "room", "rm", "suite", "building", "bldg", "hall", "floor", "level"
}


def _normalize_location_key(s: str) -> str:
    """
    Normalize a location string for matching against locations.json variants.

    - lowercase
    - remove punctuation / parentheses
    - drop pure numbers (room numbers, e.g. 184)
    - drop generic words like 'room', 'rm', 'building', etc.
    - sort tokens so 'Library (KL)' and 'KL Library' normalize the same

    Example:
      "KL 184"          -> "kl"
      "Library (KL)"    -> "kl library"
      "KL 184 (Library)"-> "kl library"
    """
    if not s:
        return ""

    s = s.lower()
    # Remove parentheses and replace non-alphanumeric with spaces
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)

    tokens: list[str] = []
    for tok in s.split():
        if not tok:
            continue
        if tok.isdigit():         # drop room numbers like "184"
            continue
        if tok in _LOCATION_STOPWORDS:
            continue
        tokens.append(tok)

    if not tokens:
        return ""

    # Sort to make word order irrelevant: "library kl" == "kl library"
    tokens.sort()
    return " ".join(tokens)


# ─────────────────────────────────────────
# Location keyword loading + simple filtering
# ─────────────────────────────────────────

def load_location_keywords() -> None:
    """
    Load locations.json and build:

      - LOCATION_KEYWORDS_LOWER: set of lowercase tokens
        from the FIRST name only (for partial-token cleanup).
      - LOCATION_TOKEN_CANONICAL: lowercase token -> canonical-cased token
      - LOCATION_VARIANT_TO_CANONICAL: normalized full variant name
        (using ALL names in each entry) -> canonical first name

    Example for a KL entry:
      "Kolligian Library"      -> canonical
      "Leo & Dottie Kolligian Library"
      "Library (KL)"
      "KL"
    all normalize to keys that map to "Kolligian Library".
    """
    global LOCATION_KEYWORDS_LOWER, LOCATION_TOKEN_CANONICAL, LOCATION_VARIANT_TO_CANONICAL

    try:
        with open(LOCATIONS_JSON_PATH, "r", encoding="utf-8") as f:
            locations = json.load(f)
    except OSError as e:
        logger.warning("Could not load locations.json from %s: %s",
                       LOCATIONS_JSON_PATH, e)
        LOCATION_KEYWORDS_LOWER = set()
        LOCATION_TOKEN_CANONICAL = {}
        LOCATION_VARIANT_TO_CANONICAL = {}
        return

    keywords: set[str] = set()
    canonical_map: dict[str, str] = {}
    variant_map: dict[str, str] = {}

    for loc in locations:
        names = loc.get("name") or []
        if not names:
            continue

        canonical_name = names[0]  # first name is the canonical one

        # --- token whitelist from canonical name (what you had before) ---
        tokens = _tokenize_words(canonical_name)
        for tok in tokens:
            lower = tok.lower()
            keywords.add(lower)
            canonical_map.setdefault(lower, tok)

        # --- full-name variant map using ALL names ---
        for variant in names:
            key = _normalize_location_key(variant)
            if not key:
                continue
            # First writer wins for a given key (good enough for campus data)
            variant_map.setdefault(key, canonical_name)

    LOCATION_KEYWORDS_LOWER = keywords
    LOCATION_TOKEN_CANONICAL = canonical_map
    LOCATION_VARIANT_TO_CANONICAL = variant_map

    logger.info(
        "[locations] loaded %d canonical tokens and %d variant keys",
        len(LOCATION_KEYWORDS_LOWER),
        len(LOCATION_VARIANT_TO_CANONICAL),
    )


# Build keyword whitelist at import time
load_location_keywords()


def clean_location_from_whitelist(raw_location: str | None) -> str | None:
    """
    1) Try to match the entire location (after normalization) against any of the
       names in locations.json (variants). If found, return the canonical first name.

       Example:
         raw_location = "KL 184"
         _normalize_location_key("KL 184") -> "kl"
         LOCATION_VARIANT_TO_CANONICAL["kl"] == "Kolligian Library"
         => return "Kolligian Library"

    2) If no full-name match, fall back to token whitelist from canonical names,
       which keeps only tokens that appear in first names.

       Example:
         raw_location = "OSI Granite Pass 163"
         kept tokens from canonical "Granite Pass" = ["Granite", "Pass"]
         => "Granite Pass"
    """
    if not raw_location:
        return raw_location

    # --- 1) Try exact name / variant match (using all names from locations.json) ---
    norm_key = _normalize_location_key(raw_location)
    if norm_key and LOCATION_VARIANT_TO_CANONICAL:
        canonical = LOCATION_VARIANT_TO_CANONICAL.get(norm_key)
        if canonical:
            return canonical  # e.g. "Kolligian Library"

    # --- 2) Fallback: keep only tokens that appear in canonical first names ---
    tokens = _tokenize_words(raw_location)
    if not tokens or not LOCATION_KEYWORDS_LOWER:
        return raw_location.strip()

    kept: list[str] = []
    for tok in tokens:
        lower = tok.lower()
        if lower in LOCATION_KEYWORDS_LOWER:
            kept.append(LOCATION_TOKEN_CANONICAL.get(lower, tok))

    if not kept:
        # nothing matched our whitelist; safer to return the original
        return raw_location.strip()

    return " ".join(kept)


def extract_slug_from_guid(guid_url: str) -> str | None:
    """
    Example:
      https://events.ucmerced.edu/event/7571-fsc-general-meeting#7571-1763452800
    -> "7571-fsc-general-meeting"
    """
    parsed = urlparse(guid_url)
    path = parsed.path  # "/event/7571-fsc-general-meeting"
    parts = [p for p in path.split("/") if p]

    try:
        idx = parts.index("event")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        return None

    return None


# ─────────────────────────────────────────
# Presence events builder + cache
# ─────────────────────────────────────────

def build_presence_events() -> list[dict]:
    """
    Fetch Presence events, normalize to /get/events shape,
    and return a list of event dicts (no caching here).
    """

    # 1) Campus info (for poster URLs)
    cdn_base = None
    campus_api_id = None
    try:
        campus_resp = requests.get(PRESENCE_CAMPUS_URL, timeout=10)
        campus_resp.raise_for_status()
        campus_info = campus_resp.json()
        cdn_base = campus_info.get("cdn")
        campus_api_id = campus_info.get("apiId")
    except requests.RequestException as e:
        logger.warning("Failed to fetch Presence campus info: %s", e)
        # Non-fatal: fallback to placeholder posters

    # 2) Events
    resp = requests.get(PRESENCE_EVENTS_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        raise RuntimeError("Presence API did not return a list")

    now_pacific = datetime.now(PACIFIC)
    normalized_events: list[dict] = []

    for ev in data:
        if not isinstance(ev, dict):
            continue

        has_ended = ev.get("hasEventEnded", False)
        start_utc_str = ev.get("startDateTimeUtc")
        end_utc_str = ev.get("endDateTimeUtc")
        if not start_utc_str or not end_utc_str:
            continue

        try:
            # Example incoming: "2025-11-05T19:00:00Z"
            start_utc = datetime.fromisoformat(
                start_utc_str.replace("Z", "+00:00"))
            end_utc = datetime.fromisoformat(
                end_utc_str.replace("Z", "+00:00"))
            start_local = start_utc.astimezone(PACIFIC)
            end_local = end_utc.astimezone(PACIFIC)
        except ValueError:
            continue

        # Double-check end time vs now
        if has_ended or end_local <= now_pacific:
            continue

        date_str = start_local.date().isoformat()
        start_at_str = start_local.strftime("%H:%M")
        end_at_str = end_local.strftime("%H:%M")

        # Poster URL
        poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"
        if cdn_base and campus_api_id and ev.get("hasCoverImage") and ev.get("photoUri"):
            poster_url = f"{cdn_base}/event-photos/{campus_api_id}/{ev['photoUri']}"

        description_html = ev.get("description") or ""
        description_text = clean_html_to_text(description_html)

        raw_loc = ev.get("location") or ""
        cleaned_loc = clean_location_from_whitelist(raw_loc)

        normalized_events.append(
            {
                # Prefix ID to avoid collisions with your Mongo events
                "id": f"presence-{ev.get('eventNoSqlId') or ev.get('uri')}",
                "_id": ev.get("eventNoSqlId"),  # not a Mongo ObjectId

                # cleaned name for matching, raw for display if needed
                "location_at": raw_loc,
                "location": cleaned_loc,

                "date": date_str,
                "start_at": start_at_str,
                "end_at": end_at_str,
                "host": ev.get("organizationName") or ev.get("campusName"),
                "title": ev.get("eventName"),
                "description": description_text,
                "poster_path": None,
                "poster_url": poster_url,
                "start_dt": start_local.replace(tzinfo=None).isoformat(),
                "created_at": None,
            }
        )

    return normalized_events


def _save_presence_cache_to_file(events: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }
    with open(PRESENCE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_presence_cache_from_file() -> tuple[list[dict] | None, datetime | None]:
    try:
        with open(PRESENCE_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    events = payload.get("events")
    gen_str = payload.get("generated_at")

    if not isinstance(events, list):
        return None, None

    gen_at = None
    if isinstance(gen_str, str):
        try:
            gen_at = datetime.fromisoformat(gen_str.replace("Z", "+00:00"))
        except ValueError:
            gen_at = None

    return events, gen_at


def refresh_presence_cache() -> list[dict]:
    """
    Rebuild Presence cache from upstream API,
    store in memory + disk, and return it.
    """
    global PRESENCE_CACHE, PRESENCE_CACHE_EXPIRES_AT

    events = build_presence_events()
    PRESENCE_CACHE = events

    now_utc = datetime.now(timezone.utc)
    PRESENCE_CACHE_EXPIRES_AT = now_utc + PRESENCE_CACHE_TTL

    _save_presence_cache_to_file(events)
    logger.info(
        "[presence_cache] refreshed %d events, expires at %s",
        len(events),
        PRESENCE_CACHE_EXPIRES_AT,
    )
    return events


def get_presence_events_cached() -> list[dict]:
    """
    Fast path for /presence_events:

    1) In-memory cache (if not expired).
    2) Disk cache JSON (if not expired).
    3) As a last resort, rebuild now.
       If rebuild fails, fall back to *any* local data we have.
    """
    global PRESENCE_CACHE, PRESENCE_CACHE_EXPIRES_AT

    now_utc = datetime.now(timezone.utc)

    # 1) in-memory
    if PRESENCE_CACHE is not None and PRESENCE_CACHE_EXPIRES_AT is not None:
        if now_utc < PRESENCE_CACHE_EXPIRES_AT:
            return PRESENCE_CACHE

    # 2) disk cache (fresh enough)
    events_file, gen_at = _load_presence_cache_from_file()
    if events_file is not None and gen_at is not None:
        if now_utc - gen_at < PRESENCE_CACHE_TTL:
            PRESENCE_CACHE = events_file
            PRESENCE_CACHE_EXPIRES_AT = gen_at + PRESENCE_CACHE_TTL
            return events_file

    # 3) rebuild now
    try:
        events = refresh_presence_cache()
        return events
    except Exception as e:
        logger.error("Failed to refresh Presence cache: %s", e)
        # fall back to any local data, even if "stale"
        if events_file is not None:
            return events_file
        if PRESENCE_CACHE is not None:
            return PRESENCE_CACHE
        # nothing available
        raise


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@events_bp.route("/presence_events", methods=["GET"])
def presence_events():
    """
    Return cached Presence events (normalized to /get/events shape).
    Cache TTL: 6 hours. Backed by disk JSON and remote Presence API.
    """
    try:
        events = get_presence_events_cached()
        return jsonify(events)
    except Exception as e:
        return jsonify(
            {"error": "Failed to load Presence events", "details": str(e)}
        ), 502


@events_bp.route("/rss_events", methods=["GET"])
def rss_events():
    """
    Fetch LiveWhale RSS and return all slugs extracted from <guid> URLs.
    Example response:
      {
        "slugs": [
          "7571-fsc-general-meeting",
          "15203-gift-of-life",
          ...
        ]
      }
    """
    try:
        resp = requests.get(FEED_URL, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": "Failed to fetch RSS feed", "details": str(e)}), 502

    xml_data = resp.content

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        return jsonify({"error": "Failed to parse RSS XML", "details": str(e)}), 500

    channel = root.find("channel")
    if channel is None:
        return jsonify({"error": "No <channel> element in RSS"}), 500

    slugs = []

    for item in channel.findall("item"):
        guid_el = item.find("guid")
        if guid_el is None or not guid_el.text:
            continue

        guid_url = guid_el.text.strip()
        slug = extract_slug_from_guid(guid_url)
        if slug:
            slugs.append(slug)

    return jsonify({"slugs": slugs})


@events_bp.route("/get/events", methods=["GET"])
def get_events():
    events_col = current_app.config["EVENTS_COL"]

    # "Now" in server local time (assumed PST for your use case)
    now = datetime.now()

    docs = list(events_col.find().sort("start_dt", 1))

    events = []
    for d in docs:
        date_str = d.get("date")      # e.g., "2025-11-30"
        end_at_str = d.get("end_at")  # e.g., "21:00"

        keep = False
        if date_str and end_at_str:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                end_time = datetime.strptime(end_at_str, "%H:%M").time()
                end_dt = datetime.combine(event_date, end_time)

                if end_dt > now:
                    keep = True
            except ValueError:
                keep = False

        if not keep:
            continue

        events.append({
            "id": d.get("id"),
            "_id": str(d["_id"]),
            "location_at": d.get("location_at"),
            "location": d.get("location"),
            "date": d.get("date"),
            "start_at": d.get("start_at"),
            "end_at": d.get("end_at"),
            "host": d.get("host"),
            "title": d.get("title"),
            "description": d.get("description"),
            "poster_path": d.get("poster_path"),
            "poster_url": d.get("poster_url"),
            "start_dt": d.get("start_dt").isoformat() if d.get("start_dt") else None,
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
        })

    return jsonify(events)


@events_bp.route("/add/events", methods=["POST"])
def add_events():
    # Text fields come from request.form
    location_at = request.form.get("location_at")
    location = request.form.get("location")
    date_str = request.form.get("date")
    start_at = request.form.get("start_at")
    end_at = request.form.get("end_at")
    host = request.form.get("host")
    title = request.form.get("title")
    description = request.form.get("description")

    poster_file = request.files.get("poster")

    # Basic validation
    missing = [k for k, v in {
        "location_at": location_at,
        "location": location,
        "date": date_str,
        "start_at": start_at,
        "end_at": end_at,
        "host": host,
        "title": title,
        "description": description,
        "poster": poster_file.filename if poster_file else None
    }.items() if not v]
    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400

    # Parse date/time (for validation + start_dt)
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    try:
        start_time = datetime.strptime(start_at, "%H:%M").time()
    except ValueError:
        return jsonify({"error": "Invalid start_at format. Use HH:MM (24h)."}), 400

    if poster_file and not allowed_file(poster_file.filename):
        return jsonify({"error": "Invalid poster type. Allowed: png, jpg, jpeg, gif, webp"}), 400

    poster_path = None
    poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"

    start_dt = datetime.combine(date, start_time)

    event_doc = {
        "id": str(uuid.uuid4()),
        "location_at": location_at,
        "location": location,
        "date": date.isoformat(),
        "start_at": start_at,
        "end_at": end_at,
        "host": host,
        "title": title,
        "description": description,
        "poster_path": poster_path,
        "poster_url": poster_url,
        "start_dt": start_dt,
        "created_at": datetime.utcnow(),
    }

    events_col = current_app.config["EVENTS_COL"]
    result = events_col.insert_one(event_doc)

    response_event = event_doc.copy()
    response_event["_id"] = str(result.inserted_id)
    response_event["start_dt"] = start_dt.isoformat()
    response_event["created_at"] = event_doc["created_at"].isoformat() + "Z"

    return jsonify({"message": "Event created", "event": response_event}), 201

# working version bellow
# # events.py
# import os
# import json
# import re
# from datetime import datetime, timezone, timedelta

# from flask import Blueprint, request, jsonify, current_app
# from werkzeug.utils import secure_filename
# import uuid

# import requests
# import xml.etree.ElementTree as ET
# from urllib.parse import urlparse
# from zoneinfo import ZoneInfo
# from html import unescape
# import logging

# logger = logging.getLogger(__name__)

# # ─────────────────────────────────────────
# # Blueprint & constants
# # ─────────────────────────────────────────

# events_bp = Blueprint("events", __name__)

# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# PACIFIC = ZoneInfo("America/Los_Angeles")

# # LiveWhale RSS
# FEED_URL = "https://events.ucmerced.edu/live/rss/events/header/All%20Events"

# # Presence API endpoints
# PRESENCE_EVENTS_URL = "https://api.presence.io/ucmerced/v1/events"
# PRESENCE_CAMPUS_URL = "https://api.presence.io/ucmerced/v1/app/campus"

# # Location data (your locations.json)
# LOCATIONS_JSON_PATH = os.path.join(os.path.dirname(__file__), "locations.json")

# # Whitelist of allowed location tokens (from first names only)
# LOCATION_KEYWORDS_LOWER: set[str] = set()
# # For pretty output, map lowercase token -> canonical-cased token from first names
# LOCATION_TOKEN_CANONICAL: dict[str, str] = {}

# # Presence cache config
# PRESENCE_CACHE_FILE = os.path.join(
#     os.path.dirname(__file__), "presence_events_cache.json"
# )
# PRESENCE_CACHE_TTL = timedelta(hours=6)

# # In-process cache
# PRESENCE_CACHE: list[dict] | None = None
# PRESENCE_CACHE_EXPIRES_AT: datetime | None = None


# # ─────────────────────────────────────────
# # Generic helpers
# # ─────────────────────────────────────────

# def allowed_file(filename: str) -> bool:
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# def clean_html_to_text(html_text: str) -> str:
#     """Roughly convert Presence HTML descriptions into plain text."""
#     text = unescape(html_text or "")
#     text = re.sub(r"<[^>]+>", "", text)  # strip tags
#     return " ".join(text.split())        # collapse whitespace


# def _tokenize_words(s: str) -> list[str]:
#     """
#     Split a string into alphanumeric tokens.
#     Example: "OSI Granite Pass 163" -> ["OSI", "Granite", "Pass", "163"]
#     """
#     if not s:
#         return []
#     return re.findall(r"[A-Za-z0-9]+", s)


# # ─────────────────────────────────────────
# # Location keyword loading + simple filtering
# # ─────────────────────────────────────────

# def load_location_keywords() -> None:
#     """
#     Load locations.json and build:
#       - LOCATION_KEYWORDS_LOWER: set of lowercase tokens from FIRST name only
#       - LOCATION_TOKEN_CANONICAL: lowercase token -> canonical-cased token

#     We ONLY look at the first element of each "name" array.
#     """
#     global LOCATION_KEYWORDS_LOWER, LOCATION_TOKEN_CANONICAL

#     try:
#         with open(LOCATIONS_JSON_PATH, "r", encoding="utf-8") as f:
#             locations = json.load(f)
#     except OSError as e:
#         logger.warning("Could not load locations.json from %s: %s",
#                        LOCATIONS_JSON_PATH, e)
#         LOCATION_KEYWORDS_LOWER = set()
#         LOCATION_TOKEN_CANONICAL = {}
#         return

#     keywords: set[str] = set()
#     canonical_map: dict[str, str] = {}

#     for loc in locations:
#         names = loc.get("name") or []
#         if not names:
#             continue

#         canonical_name = names[0]  # only first name is considered
#         tokens = _tokenize_words(canonical_name)

#         for tok in tokens:
#             lower = tok.lower()
#             keywords.add(lower)
#             # if token appears in multiple names, first one wins (fine for our purposes)
#             canonical_map.setdefault(lower, tok)

#     LOCATION_KEYWORDS_LOWER = keywords
#     LOCATION_TOKEN_CANONICAL = canonical_map

#     logger.info(
#         "[locations] loaded %d canonical tokens from first names",
#         len(LOCATION_KEYWORDS_LOWER),
#     )


# # Build keyword whitelist at import time
# load_location_keywords()


# def clean_location_from_whitelist(raw_location: str | None) -> str | None:
#     """
#     Given a raw Presence location string, keep ONLY the words/numbers
#     that appear in the first-name tokens from locations.json.

#     Example:
#       locations.json first names include "Granite Pass"
#       raw_location = "OSI Granite Pass 163"
#       -> tokens = ["OSI", "Granite", "Pass", "163"]
#       -> allowed (lowercase) tokens {"granite", "pass"}
#       -> result = "Granite Pass"
#     """
#     if not raw_location:
#         return raw_location

#     tokens = _tokenize_words(raw_location)
#     if not tokens or not LOCATION_KEYWORDS_LOWER:
#         return raw_location.strip()

#     kept: list[str] = []

#     for tok in tokens:
#         lower = tok.lower()
#         if lower in LOCATION_KEYWORDS_LOWER:
#             # use canonical casing if we have it, else original token
#             kept.append(LOCATION_TOKEN_CANONICAL.get(lower, tok))

#     if not kept:
#         # nothing matched our whitelist; better to leave raw value
#         return raw_location.strip()

#     return " ".join(kept)


# def extract_slug_from_guid(guid_url: str) -> str | None:
#     """
#     Example:
#       https://events.ucmerced.edu/event/7571-fsc-general-meeting#7571-1763452800
#     -> "7571-fsc-general-meeting"
#     """
#     parsed = urlparse(guid_url)
#     path = parsed.path  # "/event/7571-fsc-general-meeting"
#     parts = [p for p in path.split("/") if p]

#     try:
#         idx = parts.index("event")
#         if idx + 1 < len(parts):
#             return parts[idx + 1]
#     except ValueError:
#         return None

#     return None


# # ─────────────────────────────────────────
# # Presence events builder + cache
# # ─────────────────────────────────────────

# def build_presence_events() -> list[dict]:
#     """
#     Fetch Presence events, normalize to /get/events shape,
#     and return a list of event dicts (no caching here).
#     """

#     # 1) Campus info (for poster URLs)
#     cdn_base = None
#     campus_api_id = None
#     try:
#         campus_resp = requests.get(PRESENCE_CAMPUS_URL, timeout=10)
#         campus_resp.raise_for_status()
#         campus_info = campus_resp.json()
#         cdn_base = campus_info.get("cdn")
#         campus_api_id = campus_info.get("apiId")
#     except requests.RequestException as e:
#         logger.warning("Failed to fetch Presence campus info: %s", e)
#         # Non-fatal: fallback to placeholder posters

#     # 2) Events
#     resp = requests.get(PRESENCE_EVENTS_URL, timeout=10)
#     resp.raise_for_status()
#     data = resp.json()

#     if not isinstance(data, list):
#         raise RuntimeError("Presence API did not return a list")

#     now_pacific = datetime.now(PACIFIC)
#     normalized_events: list[dict] = []

#     for ev in data:
#         if not isinstance(ev, dict):
#             continue

#         has_ended = ev.get("hasEventEnded", False)
#         start_utc_str = ev.get("startDateTimeUtc")
#         end_utc_str = ev.get("endDateTimeUtc")
#         if not start_utc_str or not end_utc_str:
#             continue

#         try:
#             # Example incoming: "2025-11-05T19:00:00Z"
#             start_utc = datetime.fromisoformat(
#                 start_utc_str.replace("Z", "+00:00"))
#             end_utc = datetime.fromisoformat(
#                 end_utc_str.replace("Z", "+00:00"))
#             start_local = start_utc.astimezone(PACIFIC)
#             end_local = end_utc.astimezone(PACIFIC)
#         except ValueError:
#             continue

#         # Double-check end time vs now
#         if has_ended or end_local <= now_pacific:
#             continue

#         date_str = start_local.date().isoformat()
#         start_at_str = start_local.strftime("%H:%M")
#         end_at_str = end_local.strftime("%H:%M")

#         # Poster URL
#         poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"
#         if cdn_base and campus_api_id and ev.get("hasCoverImage") and ev.get("photoUri"):
#             poster_url = f"{cdn_base}/event-photos/{campus_api_id}/{ev['photoUri']}"

#         description_html = ev.get("description") or ""
#         description_text = clean_html_to_text(description_html)

#         raw_loc = ev.get("location") or ""
#         cleaned_loc = clean_location_from_whitelist(raw_loc)

#         normalized_events.append(
#             {
#                 # Prefix ID to avoid collisions with your Mongo events
#                 "id": f"presence-{ev.get('eventNoSqlId') or ev.get('uri')}",
#                 "_id": ev.get("eventNoSqlId"),  # not a Mongo ObjectId

#                 # cleaned name for matching, raw for display if needed
#                 "location_at": raw_loc,
#                 "location": cleaned_loc,

#                 "date": date_str,
#                 "start_at": start_at_str,
#                 "end_at": end_at_str,
#                 "host": ev.get("organizationName") or ev.get("campusName"),
#                 "title": ev.get("eventName"),
#                 "description": description_text,
#                 "poster_path": None,
#                 "poster_url": poster_url,
#                 "start_dt": start_local.replace(tzinfo=None).isoformat(),
#                 "created_at": None,
#             }
#         )

#     return normalized_events


# def _save_presence_cache_to_file(events: list[dict]) -> None:
#     payload = {
#         "generated_at": datetime.now(timezone.utc).isoformat(),
#         "events": events,
#     }
#     with open(PRESENCE_CACHE_FILE, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False)


# def _load_presence_cache_from_file() -> tuple[list[dict] | None, datetime | None]:
#     try:
#         with open(PRESENCE_CACHE_FILE, "r", encoding="utf-8") as f:
#             payload = json.load(f)
#     except (OSError, json.JSONDecodeError):
#         return None, None

#     events = payload.get("events")
#     gen_str = payload.get("generated_at")

#     if not isinstance(events, list):
#         return None, None

#     gen_at = None
#     if isinstance(gen_str, str):
#         try:
#             gen_at = datetime.fromisoformat(gen_str.replace("Z", "+00:00"))
#         except ValueError:
#             gen_at = None

#     return events, gen_at


# def refresh_presence_cache() -> list[dict]:
#     """
#     Rebuild Presence cache from upstream API,
#     store in memory + disk, and return it.
#     """
#     global PRESENCE_CACHE, PRESENCE_CACHE_EXPIRES_AT

#     events = build_presence_events()
#     PRESENCE_CACHE = events

#     now_utc = datetime.now(timezone.utc)
#     PRESENCE_CACHE_EXPIRES_AT = now_utc + PRESENCE_CACHE_TTL

#     _save_presence_cache_to_file(events)
#     logger.info(
#         "[presence_cache] refreshed %d events, expires at %s",
#         len(events),
#         PRESENCE_CACHE_EXPIRES_AT,
#     )
#     return events


# def get_presence_events_cached() -> list[dict]:
#     """
#     Fast path for /presence_events:

#     1) In-memory cache (if not expired).
#     2) Disk cache JSON (if not expired).
#     3) As a last resort, rebuild now.
#        If rebuild fails, fall back to *any* local data we have.
#     """
#     global PRESENCE_CACHE, PRESENCE_CACHE_EXPIRES_AT

#     now_utc = datetime.now(timezone.utc)

#     # 1) in-memory
#     if PRESENCE_CACHE is not None and PRESENCE_CACHE_EXPIRES_AT is not None:
#         if now_utc < PRESENCE_CACHE_EXPIRES_AT:
#             return PRESENCE_CACHE

#     # 2) disk cache (fresh enough)
#     events_file, gen_at = _load_presence_cache_from_file()
#     if events_file is not None and gen_at is not None:
#         if now_utc - gen_at < PRESENCE_CACHE_TTL:
#             PRESENCE_CACHE = events_file
#             PRESENCE_CACHE_EXPIRES_AT = gen_at + PRESENCE_CACHE_TTL
#             return events_file

#     # 3) rebuild now
#     try:
#         events = refresh_presence_cache()
#         return events
#     except Exception as e:
#         logger.error("Failed to refresh Presence cache: %s", e)
#         # fall back to any local data, even if "stale"
#         if events_file is not None:
#             return events_file
#         if PRESENCE_CACHE is not None:
#             return PRESENCE_CACHE
#         # nothing available
#         raise


# # ─────────────────────────────────────────
# # Routes
# # ─────────────────────────────────────────

# @events_bp.route("/presence_events", methods=["GET"])
# def presence_events():
#     """
#     Return cached Presence events (normalized to /get/events shape).
#     Cache TTL: 6 hours. Backed by disk JSON and remote Presence API.
#     """
#     try:
#         events = get_presence_events_cached()
#         return jsonify(events)
#     except Exception as e:
#         return jsonify(
#             {"error": "Failed to load Presence events", "details": str(e)}
#         ), 502


# @events_bp.route("/rss_events", methods=["GET"])
# def rss_events():
#     """
#     Fetch LiveWhale RSS and return all slugs extracted from <guid> URLs.
#     Example response:
#       {
#         "slugs": [
#           "7571-fsc-general-meeting",
#           "15203-gift-of-life",
#           ...
#         ]
#       }
#     """
#     try:
#         resp = requests.get(FEED_URL, timeout=10)
#         resp.raise_for_status()
#     except requests.RequestException as e:
#         return jsonify({"error": "Failed to fetch RSS feed", "details": str(e)}), 502

#     xml_data = resp.content

#     try:
#         root = ET.fromstring(xml_data)
#     except ET.ParseError as e:
#         return jsonify({"error": "Failed to parse RSS XML", "details": str(e)}), 500

#     channel = root.find("channel")
#     if channel is None:
#         return jsonify({"error": "No <channel> element in RSS"}), 500

#     slugs = []

#     for item in channel.findall("item"):
#         guid_el = item.find("guid")
#         if guid_el is None or not guid_el.text:
#             continue

#         guid_url = guid_el.text.strip()
#         slug = extract_slug_from_guid(guid_url)
#         if slug:
#             slugs.append(slug)

#     return jsonify({"slugs": slugs})


# @events_bp.route("/get/events", methods=["GET"])
# def get_events():
#     events_col = current_app.config["EVENTS_COL"]

#     # "Now" in server local time (assumed PST for your use case)
#     now = datetime.now()

#     docs = list(events_col.find().sort("start_dt", 1))

#     events = []
#     for d in docs:
#         date_str = d.get("date")      # e.g., "2025-11-30"
#         end_at_str = d.get("end_at")  # e.g., "21:00"

#         keep = False
#         if date_str and end_at_str:
#             try:
#                 event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
#                 end_time = datetime.strptime(end_at_str, "%H:%M").time()
#                 end_dt = datetime.combine(event_date, end_time)

#                 if end_dt > now:
#                     keep = True
#             except ValueError:
#                 keep = False

#         if not keep:
#             continue

#         events.append({
#             "id": d.get("id"),
#             "_id": str(d["_id"]),
#             "location_at": d.get("location_at"),
#             "location": d.get("location"),
#             "date": d.get("date"),
#             "start_at": d.get("start_at"),
#             "end_at": d.get("end_at"),
#             "host": d.get("host"),
#             "title": d.get("title"),
#             "description": d.get("description"),
#             "poster_path": d.get("poster_path"),
#             "poster_url": d.get("poster_url"),
#             "start_dt": d.get("start_dt").isoformat() if d.get("start_dt") else None,
#             "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
#         })

#     return jsonify(events)


# @events_bp.route("/add/events", methods=["POST"])
# def add_events():
#     # Text fields come from request.form
#     location_at = request.form.get("location_at")
#     location = request.form.get("location")
#     date_str = request.form.get("date")
#     start_at = request.form.get("start_at")
#     end_at = request.form.get("end_at")
#     host = request.form.get("host")
#     title = request.form.get("title")
#     description = request.form.get("description")

#     poster_file = request.files.get("poster")

#     # Basic validation
#     missing = [k for k, v in {
#         "location_at": location_at,
#         "location": location,
#         "date": date_str,
#         "start_at": start_at,
#         "end_at": end_at,
#         "host": host,
#         "title": title,
#         "description": description,
#         "poster": poster_file.filename if poster_file else None
#     }.items() if not v]
#     if missing:
#         return jsonify({"error": "Missing fields", "fields": missing}), 400

#     # Parse date/time (for validation + start_dt)
#     try:
#         date = datetime.strptime(date_str, "%Y-%m-%d").date()
#     except ValueError:
#         return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

#     try:
#         start_time = datetime.strptime(start_at, "%H:%M").time()
#     except ValueError:
#         return jsonify({"error": "Invalid start_at format. Use HH:MM (24h)."}), 400

#     if poster_file and not allowed_file(poster_file.filename):
#         return jsonify({"error": "Invalid poster type. Allowed: png, jpg, jpeg, gif, webp"}), 400

#     poster_path = None
#     poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"

#     start_dt = datetime.combine(date, start_time)

#     event_doc = {
#         "id": str(uuid.uuid4()),
#         "location_at": location_at,
#         "location": location,
#         "date": date.isoformat(),
#         "start_at": start_at,
#         "end_at": end_at,
#         "host": host,
#         "title": title,
#         "description": description,
#         "poster_path": poster_path,
#         "poster_url": poster_url,
#         "start_dt": start_dt,
#         "created_at": datetime.utcnow(),
#     }

#     events_col = current_app.config["EVENTS_COL"]
#     result = events_col.insert_one(event_doc)

#     response_event = event_doc.copy()
#     response_event["_id"] = str(result.inserted_id)
#     response_event["start_dt"] = start_dt.isoformat()
#     response_event["created_at"] = event_doc["created_at"].isoformat() + "Z"

#     return jsonify({"message": "Event created", "event": response_event}), 201

# # # events.py
# import os
# from datetime import datetime, timezone
# from flask import Blueprint, request, jsonify, current_app, url_for
# from werkzeug.utils import secure_filename
# from collections import defaultdict, deque
# import uuid  # 👈 for a unique event id

# # 👉 NEW imports for RSS + Presence parsing
# import requests
# import xml.etree.ElementTree as ET
# from urllib.parse import urlparse
# from zoneinfo import ZoneInfo
# from html import unescape
# import re


# events_bp = Blueprint('events', __name__)

# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


# def allowed_file(filename: str) -> bool:
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# # 👉 Helper just for this RSS endpoint
# FEED_URL = "https://events.ucmerced.edu/live/rss/events/header/All%20Events"

# events_bp = Blueprint('events', __name__)

# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# PACIFIC = ZoneInfo("America/Los_Angeles")

# # 👉 existing
# FEED_URL = "https://events.ucmerced.edu/live/rss/events/header/All%20Events"

# # 👉 NEW: Presence API endpoints
# PRESENCE_EVENTS_URL = "https://api.presence.io/ucmerced/v1/events"
# PRESENCE_CAMPUS_URL = "https://api.presence.io/ucmerced/v1/app/campus"


# def allowed_file(filename: str) -> bool:
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# def clean_html_to_text(html_text: str) -> str:
#     """Roughly convert Presence HTML descriptions into plain text."""
#     text = unescape(html_text or "")
#     # strip tags
#     text = re.sub(r"<[^>]+>", "", text)
#     # collapse whitespace
#     return " ".join(text.split())


# def extract_slug_from_guid(guid_url: str) -> str | None:
#     """
#     Example:
#       https://events.ucmerced.edu/event/7571-fsc-general-meeting#7571-1763452800
#     -> "7571-fsc-general-meeting"
#     """
#     parsed = urlparse(guid_url)
#     path = parsed.path  # "/event/7571-fsc-general-meeting"
#     parts = [p for p in path.split("/") if p]

#     try:
#         idx = parts.index("event")
#         if idx + 1 < len(parts):
#             return parts[idx + 1]
#     except ValueError:
#         return None

#     return None


# @events_bp.route("/presence_events", methods=["GET"])
# def presence_events():
#     """
#     Fetch Presence events and normalize them into the same shape as /get/events.
#     Only returns events that have not ended yet (in Pacific time).
#     """
#     # ── 1) Fetch campus info (for poster URLs). If this fails, we just fall back to placeholder.
#     cdn_base = None
#     campus_api_id = None

#     try:
#         campus_resp = requests.get(PRESENCE_CAMPUS_URL, timeout=10)
#         campus_resp.raise_for_status()
#         campus_info = campus_resp.json()
#         # e.g. "https://ucmerced-cdn.presence.io"
#         cdn_base = campus_info.get("cdn")
#         campus_api_id = campus_info.get("apiId")   # used in event poster URL
#     except requests.RequestException:
#         # Non-fatal: we can still return events with placeholder posters
#         pass

#     # ── 2) Fetch events list
#     try:
#         resp = requests.get(PRESENCE_EVENTS_URL, timeout=10)
#         resp.raise_for_status()
#     except requests.RequestException as e:
#         return jsonify({"error": "Failed to fetch Presence events", "details": str(e)}), 502

#     try:
#         data = resp.json()
#     except ValueError as e:
#         return jsonify({"error": "Failed to parse Presence JSON", "details": str(e)}), 500

#     if not isinstance(data, list):
#         return jsonify({"error": "Presence API did not return a list"}), 500

#     now_pacific = datetime.now(PACIFIC)
#     normalized_events = []

#     for ev in data:
#         # Skip obviously bad entries
#         if not isinstance(ev, dict):
#             continue

#         # Only keep events that have not ended (both by flag and by time)
#         has_ended = ev.get("hasEventEnded", False)

#         start_utc_str = ev.get("startDateTimeUtc")
#         end_utc_str = ev.get("endDateTimeUtc")

#         if not start_utc_str or not end_utc_str:
#             continue

#         try:
#             # Example incoming: "2025-11-05T19:00:00Z"
#             start_utc = datetime.fromisoformat(
#                 start_utc_str.replace("Z", "+00:00"))
#             end_utc = datetime.fromisoformat(
#                 end_utc_str.replace("Z", "+00:00"))

#             # Convert to Pacific
#             start_local = start_utc.astimezone(PACIFIC)
#             end_local = end_utc.astimezone(PACIFIC)
#         except ValueError:
#             # bad datetime format → skip
#             continue

#         # Double-check end time vs now
#         if has_ended or end_local <= now_pacific:
#             continue

#         # Build date / time strings in the same style as /get/events
#         date_str = start_local.date().isoformat()
#         start_at_str = start_local.strftime("%H:%M")
#         end_at_str = end_local.strftime("%H:%M")

#         # Poster URL
#         poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"
#         if cdn_base and campus_api_id and ev.get("hasCoverImage") and ev.get("photoUri"):
#             # Presence docs: https://[cdn]/event-photos/[campusApiId]/[photoUri]
#             poster_url = f"{cdn_base}/event-photos/{campus_api_id}/{ev['photoUri']}"

#         description_html = ev.get("description") or ""
#         description_text = clean_html_to_text(description_html)

#         # Normalize into the same shape as /get/events
#         normalized_events.append({
#             # Prefix ID to avoid collisions with your Mongo events
#             "id": f"presence-{ev.get('eventNoSqlId') or ev.get('uri')}",
#             # not a Mongo ObjectId, but unique enough for frontend
#             "_id": ev.get("eventNoSqlId"),

#             # you can change this later if you map to your own location IDs
#             "location_at": ev.get("location"),
#             "location": ev.get("location"),

#             "date": date_str,
#             "start_at": start_at_str,
#             "end_at": end_at_str,

#             "host": ev.get("organizationName") or ev.get("campusName"),
#             "title": ev.get("eventName"),
#             "description": description_text,

#             "poster_path": None,
#             "poster_url": poster_url,

#             # Match /get/events style: naive local datetime ISO
#             "start_dt": start_local.replace(tzinfo=None).isoformat(),
#             # Presence doesn't expose this; you can set to start_local if you prefer
#             "created_at": None,
#         })

#     return jsonify(normalized_events)


# @events_bp.route("/rss_events", methods=["GET"])
# def rss_events():
#     """
#     Fetch LiveWhale RSS and return all slugs extracted from <guid> URLs.
#     Example response:
#       {
#         "slugs": [
#           "7571-fsc-general-meeting",
#           "15203-gift-of-life",
#           ...
#         ]
#       }
#     """
#     try:
#         resp = requests.get(FEED_URL, timeout=10)
#         resp.raise_for_status()
#     except requests.RequestException as e:
#         return jsonify({"error": f"Failed to fetch RSS feed", "details": str(e)}), 502

#     xml_data = resp.content

#     try:
#         root = ET.fromstring(xml_data)
#     except ET.ParseError as e:
#         return jsonify({"error": "Failed to parse RSS XML", "details": str(e)}), 500

#     channel = root.find("channel")
#     if channel is None:
#         return jsonify({"error": "No <channel> element in RSS"}), 500

#     slugs = []

#     for item in channel.findall("item"):
#         guid_el = item.find("guid")
#         if guid_el is None or not guid_el.text:
#             continue

#         guid_url = guid_el.text.strip()
#         slug = extract_slug_from_guid(guid_url)
#         if slug:
#             slugs.append(slug)

#     return jsonify({"slugs": slugs})


# @events_bp.route("/get/events", methods=["GET"])
# def get_events():
#     events_col = current_app.config["EVENTS_COL"]

#     # "Now" in PST (we're assuming server is PST; otherwise see note below)
#     now = datetime.now()

#     docs = list(events_col.find().sort("start_dt", 1))

#     events = []
#     for d in docs:
#         date_str = d.get("date")      # e.g., "2025-11-30"
#         end_at_str = d.get("end_at")  # e.g., "21:00"

#         # Default: assume event has ended if we can't parse date/end_at
#         keep = False

#         if date_str and end_at_str:
#             try:
#                 event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
#                 end_time = datetime.strptime(end_at_str, "%H:%M").time()
#                 end_dt = datetime.combine(event_date, end_time)  # naive PST

#                 # Only keep events that have not ended yet
#                 if end_dt > now:
#                     keep = True

#             except ValueError:
#                 # Bad formatting -> treat as ended / drop from list
#                 keep = False

#         if not keep:
#             continue  # skip this event

#         events.append({
#             "id": d.get("id"),
#             "_id": str(d["_id"]),
#             "location_at": d.get("location_at"),
#             "location": d.get("location"),
#             "date": d.get("date"),  # stored as ISO string below
#             "start_at": d.get("start_at"),
#             "end_at": d.get("end_at"),
#             "host": d.get("host"),
#             "title": d.get("title"),
#             "description": d.get("description"),
#             "poster_path": d.get("poster_path"),
#             "poster_url": d.get("poster_url"),
#             "start_dt": d.get("start_dt").isoformat() if d.get("start_dt") else None,
#             "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
#         })

#     return jsonify(events)


# @events_bp.route("/add/events", methods=["POST"])
# def add_events():
#     # Text fields come from request.form
#     location_at = request.form.get("location_at")
#     location = request.form.get("location")
#     date_str = request.form.get("date")        # e.g., "2025-11-30"
#     start_at = request.form.get("start_at")    # e.g., "18:00"
#     end_at = request.form.get("end_at")        # e.g., "21:00"
#     host = request.form.get("host")
#     title = request.form.get("title")
#     description = request.form.get("description")

#     # File comes from request.files
#     poster_file = request.files.get("poster")

#     # Basic validation
#     missing = [k for k, v in {
#         "location_at": location_at,
#         "location": location,
#         "date": date_str,
#         "start_at": start_at,
#         "end_at": end_at,
#         "host": host,
#         "title": title,
#         "description": description,
#         "poster": poster_file.filename if poster_file else None
#     }.items() if not v]
#     if missing:
#         return jsonify({"error": "Missing fields", "fields": missing}), 400

#     # Parse date/time (for validation + start_dt)
#     try:
#         date = datetime.strptime(date_str, "%Y-%m-%d").date()
#     except ValueError:
#         return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

#     try:
#         start_time = datetime.strptime(start_at, "%H:%M").time()
#     except ValueError:
#         return jsonify({"error": "Invalid start_at format. Use HH:MM (24h)."}), 400

#     # Optional: validate poster file type, but DON'T save locally
#     if poster_file and not allowed_file(poster_file.filename):
#         return jsonify({"error": "Invalid poster type. Allowed: png, jpg, jpeg, gif, webp"}), 400

#     # 🚫 No local saving. Just placeholder URL for now.
#     poster_path = None
#     poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"

#     # Build document for MongoDB
#     start_dt = datetime.combine(date, start_time)

#     event_doc = {
#         "id": str(uuid.uuid4()),         # satisfies unique index on "id"
#         "location_at": location_at,
#         "location": location,
#         "date": date.isoformat(),        # store as ISO string
#         "start_at": start_at,
#         "end_at": end_at,
#         "host": host,
#         "title": title,
#         "description": description,
#         "poster_path": poster_path,
#         "poster_url": poster_url,
#         "start_dt": start_dt,            # real datetime for querying/sorting
#         "created_at": datetime.utcnow(),  # audit field
#     }

#     # Insert into MongoDB
#     events_col = current_app.config["EVENTS_COL"]
#     result = events_col.insert_one(event_doc)

#     # ✅ Build a JSON-safe copy (avoid ObjectId from event_doc["_id"])
#     response_event = event_doc.copy()
#     response_event["_id"] = str(result.inserted_id)
#     response_event["start_dt"] = start_dt.isoformat()
#     response_event["created_at"] = event_doc["created_at"].isoformat() + "Z"

#     return jsonify({"message": "Event created", "event": response_event}), 201
