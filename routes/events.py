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

from helper import normalize_location

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


# ─────────────────────────────────────────
# Location Mapping & Normalization
# ─────────────────────────────────────────

# Maps lowercase alias -> Canonical Name
# Example: {'kl': 'Kolligian Library', 'library (kl)': 'Kolligian Library'}
# LOCATION_MAP: dict[str, str] = {}


def load_location_map() -> None:
    """
    Load locations.json and build a map of ALL aliases to the first (canonical) name.
    """

    global LOCATION_MAP

    try:
        with open(LOCATIONS_JSON_PATH, "r", encoding="utf-8") as f:
            locations = json.load(f)
    except OSError as e:
        logger.warning("Could not load locations.json from %s: %s",
                       LOCATIONS_JSON_PATH, e)
        LOCATION_MAP = {}
        return

    new_map = {}

    for loc in locations:
        names = loc.get("name") or []
        if not names:
            continue

        # The first name in the list is the "Official" name we want to return
        canonical_name = names[0]

        # Map every name in the list to the canonical name
        for alias in names:
            # Store as lowercase for case-insensitive matching
            clean_alias = alias.lower().strip()
            new_map[clean_alias] = canonical_name

    LOCATION_MAP = new_map

    logger.info("[locations] loaded %d location aliases", len(LOCATION_MAP))


# Build location map at import time
load_location_map()


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

        # Use the new normalization logic
        cleaned_loc = normalize_location(raw_loc)

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
