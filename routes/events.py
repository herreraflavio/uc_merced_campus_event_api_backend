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

from helper.normalize_location import normalize_event_location

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

# Warm-up URL (often helps obtain cookies / pass basic bot checks)
PRESENCE_WARMUP_URL = "https://ucmerced.presence.io/"

# Browser-like headers
PRESENCE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/143.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9,es-US;q=0.8,es;q=0.7",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

PRESENCE_COOKIE_ENV = "PRESENCE_COOKIE"

# ─── File Paths ───

# Shared locations.json (used for both path resolving and ID lookup)
LOCATIONS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "../helper/locations.json")

# ─── Caching Config ───

PRESENCE_CACHE_TTL = timedelta(hours=6)

# 1. Web Cache
PRESENCE_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "presence_events_cache.json")
PRESENCE_CACHE: list[dict] | None = None
PRESENCE_CACHE_EXPIRES_AT: datetime | None = None

# 2. iOS Cache
PRESENCE_IOS_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "presence_events_ios_cache.json")
PRESENCE_IOS_CACHE: list[dict] | None = None
PRESENCE_IOS_CACHE_EXPIRES_AT: datetime | None = None


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


def _presence_session() -> requests.Session:
    """
    Create a session that mimics a browser more closely than vanilla requests.
    """
    s = requests.Session()
    s.headers.update(PRESENCE_HEADERS)

    cookie = os.getenv(PRESENCE_COOKIE_ENV)
    if cookie:
        s.headers["Cookie"] = cookie

    try:
        s.get(PRESENCE_WARMUP_URL, timeout=15)
    except Exception as e:
        logger.info("[presence] warm-up skipped/failed: %s", e)

    return s


def _get_json_or_raise(session: requests.Session, url: str, timeout: int = 15) -> object:
    headers = {"Referer": PRESENCE_WARMUP_URL}
    r = session.get(url, headers=headers, timeout=timeout)

    if r.status_code != 200:
        body_snip = (r.text or "")[:700]
        raise RuntimeError(
            f"Presence request failed: status={r.status_code} url={url} "
            f"body_snip={body_snip!r}"
        )

    try:
        return r.json()
    except Exception as e:
        body_snip = (r.text or "")[:700]
        raise RuntimeError(
            f"Presence returned non-JSON for url={url}: {e}; body_snip={body_snip!r}"
        )


# ─────────────────────────────────────────
# Location Mapping & Normalization
# ─────────────────────────────────────────

def extract_slug_from_guid(guid_url: str) -> str | None:
    parsed = urlparse(guid_url)
    path = parsed.path
    parts = [p for p in path.split("/") if p]
    try:
        idx = parts.index("event")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        return None
    return None


def load_locations_data() -> list[dict]:
    """Load the locations.json for mapping names to IDs."""
    if not os.path.exists(LOCATIONS_JSON_PATH):
        logger.warning(f"Locations file not found at {LOCATIONS_JSON_PATH}")
        return []

    try:
        with open(LOCATIONS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load locations.json: {e}")
        return []


def find_location_id(normalized_name: str, locations_data: list[dict]) -> int | None:
    """
    Match the normalized location name against the 'name' arrays in locations_data.
    Returns the 'id' if found, else None.
    """
    if not normalized_name:
        return None

    # Case-insensitive comparison
    target = normalized_name.lower().strip()

    for item in locations_data:
        names = item.get("name", [])
        # 'name' can be a list of strings
        if isinstance(names, list):
            for n in names:
                if isinstance(n, str) and n.lower().strip() == target:
                    return item.get("id")

    return None


# ─────────────────────────────────────────
# Presence events builder (WEB)
# ─────────────────────────────────────────

def build_presence_events() -> list[dict]:
    s = _presence_session()

    # 1) Campus info
    cdn_base = None
    campus_api_id = None
    try:
        campus_info = _get_json_or_raise(s, PRESENCE_CAMPUS_URL, timeout=15)
        if isinstance(campus_info, dict):
            cdn_base = campus_info.get("cdn")
            campus_api_id = campus_info.get("apiId")
    except Exception as e:
        logger.warning("Failed to fetch Presence campus info: %s", e)

    # 2) Events
    data = _get_json_or_raise(s, PRESENCE_EVENTS_URL, timeout=20)
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
            start_utc = datetime.fromisoformat(
                start_utc_str.replace("Z", "+00:00"))
            end_utc = datetime.fromisoformat(
                end_utc_str.replace("Z", "+00:00"))
            start_local = start_utc.astimezone(PACIFIC)
            end_local = end_utc.astimezone(PACIFIC)
        except ValueError:
            continue

        if has_ended or end_local <= now_pacific:
            continue

        date_str = start_local.date().isoformat()
        start_at_str = start_local.strftime("%H:%M")
        end_at_str = end_local.strftime("%H:%M")

        poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"
        if cdn_base and campus_api_id and ev.get("hasCoverImage") and ev.get("photoUri"):
            poster_url = f"{cdn_base}/event-photos/{campus_api_id}/{ev['photoUri']}"

        description_text = clean_html_to_text(ev.get("description") or "")
        raw_loc = ev.get("location") or ""
        cleaned_loc = normalize_event_location(raw_loc)

        normalized_events.append({
            "id": f"presence-{ev.get('eventNoSqlId') or ev.get('uri')}",
            "_id": ev.get("eventNoSqlId"),
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
        })

    return normalized_events


# ─────────────────────────────────────────
# Presence events builder (IOS)
# ─────────────────────────────────────────

def build_presence_events_ios() -> list[dict]:
    """
    Fetch Presence events and normalize to the specific iOS/JSON structure
    requested, resolving location_id via locations.json.
    """
    s = _presence_session()

    # Load Metadata for Location ID lookup
    locations_data = load_locations_data()

    # 1) Campus info (for poster URLs)
    cdn_base = None
    campus_api_id = None
    try:
        campus_info = _get_json_or_raise(s, PRESENCE_CAMPUS_URL, timeout=15)
        if isinstance(campus_info, dict):
            cdn_base = campus_info.get("cdn")
            campus_api_id = campus_info.get("apiId")
    except Exception as e:
        logger.warning("Failed to fetch Presence campus info: %s", e)

    # 2) Events
    data = _get_json_or_raise(s, PRESENCE_EVENTS_URL, timeout=20)
    if not isinstance(data, list):
        raise RuntimeError("Presence API did not return a list")

    now_pacific = datetime.now(PACIFIC)
    ios_events: list[dict] = []

    for ev in data:
        if not isinstance(ev, dict):
            continue

        has_ended = ev.get("hasEventEnded", False)
        start_utc_str = ev.get("startDateTimeUtc")
        end_utc_str = ev.get("endDateTimeUtc")
        if not start_utc_str or not end_utc_str:
            continue

        try:
            # Check dates strictly to filter out past events
            start_utc = datetime.fromisoformat(
                start_utc_str.replace("Z", "+00:00"))
            end_utc = datetime.fromisoformat(
                end_utc_str.replace("Z", "+00:00"))

            # We convert to pacific just for the logic check "is event over?"
            end_local = end_utc.astimezone(PACIFIC)
        except ValueError:
            continue

        if has_ended or end_local <= now_pacific:
            continue

        # Prepare Strings
        description_text = clean_html_to_text(ev.get("description") or "")
        raw_loc = ev.get("location") or ""
        cleaned_loc = normalize_event_location(raw_loc)

        # Lookup Location ID
        loc_id = find_location_id(cleaned_loc, locations_data)

        # Prepare Image URLs
        poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"
        if cdn_base and campus_api_id and ev.get("hasCoverImage") and ev.get("photoUri"):
            poster_url = f"{cdn_base}/event-photos/{campus_api_id}/{ev['photoUri']}"

        image_urls = [poster_url]

        # Pin URL (Generic default)
        pin_url = "https://img.icons8.com/color/96/marker.png"

        # Unique ID for iOS
        # Using eventNoSqlId is usually safe, or fall back to uri
        evt_id = str(ev.get('eventNoSqlId') or ev.get('uri') or uuid.uuid4())

        # Construct the iOS Element
        ios_event = {
            "id": evt_id,
            "location_id": loc_id,  # integer or None
            "attributes": {
                "title": ev.get("eventName"),
                "description": description_text,
                "start": start_utc_str,  # Keep original UTC ISO string "2026-02-08T10:00:00Z"
                "end": end_utc_str,
                "location": cleaned_loc,
                "host": ev.get("organizationName") or ev.get("campusName"),
                "source_url": f"https://ucmerced.presence.io/event/{ev.get('urlName')}" if ev.get('urlName') else "https://ucmerced.presence.io",
                "image_urls": image_urls,
                "pin_url": pin_url
            },
            "geometry": {
                "latitude": 37.3655,
                "longitude": -120.4245
            }
        }

        ios_events.append(ios_event)

    return ios_events


# ─────────────────────────────────────────
# Cache Management (WEB)
# ─────────────────────────────────────────

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
    global PRESENCE_CACHE, PRESENCE_CACHE_EXPIRES_AT
    events = build_presence_events()
    PRESENCE_CACHE = events
    now_utc = datetime.now(timezone.utc)
    PRESENCE_CACHE_EXPIRES_AT = now_utc + PRESENCE_CACHE_TTL
    _save_presence_cache_to_file(events)
    logger.info("[presence_cache] refreshed %d events", len(events))
    return events


def get_presence_events_cached() -> list[dict]:
    global PRESENCE_CACHE, PRESENCE_CACHE_EXPIRES_AT
    now_utc = datetime.now(timezone.utc)

    # 1) in-memory
    if PRESENCE_CACHE is not None and PRESENCE_CACHE_EXPIRES_AT is not None:
        if now_utc < PRESENCE_CACHE_EXPIRES_AT:
            return PRESENCE_CACHE

    # 2) disk
    events_file, gen_at = _load_presence_cache_from_file()
    if events_file is not None and gen_at is not None:
        if now_utc - gen_at < PRESENCE_CACHE_TTL:
            PRESENCE_CACHE = events_file
            PRESENCE_CACHE_EXPIRES_AT = gen_at + PRESENCE_CACHE_TTL
            return events_file

    # 3) rebuild
    try:
        return refresh_presence_cache()
    except Exception as e:
        logger.error("Failed to refresh Presence cache: %s", e)
        if events_file is not None:
            return events_file
        if PRESENCE_CACHE is not None:
            return PRESENCE_CACHE
        raise


# ─────────────────────────────────────────
# Cache Management (IOS)
# ─────────────────────────────────────────

def _save_presence_ios_cache_to_file(events: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }
    with open(PRESENCE_IOS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_presence_ios_cache_from_file() -> tuple[list[dict] | None, datetime | None]:
    try:
        with open(PRESENCE_IOS_CACHE_FILE, "r", encoding="utf-8") as f:
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


def refresh_presence_ios_cache() -> list[dict]:
    global PRESENCE_IOS_CACHE, PRESENCE_IOS_CACHE_EXPIRES_AT
    events = build_presence_events_ios()
    PRESENCE_IOS_CACHE = events
    now_utc = datetime.now(timezone.utc)
    PRESENCE_IOS_CACHE_EXPIRES_AT = now_utc + PRESENCE_CACHE_TTL
    _save_presence_ios_cache_to_file(events)
    logger.info("[presence_ios_cache] refreshed %d events", len(events))
    return events


def get_presence_events_ios_cached() -> list[dict]:
    global PRESENCE_IOS_CACHE, PRESENCE_IOS_CACHE_EXPIRES_AT
    now_utc = datetime.now(timezone.utc)

    # 1) in-memory
    if PRESENCE_IOS_CACHE is not None and PRESENCE_IOS_CACHE_EXPIRES_AT is not None:
        if now_utc < PRESENCE_IOS_CACHE_EXPIRES_AT:
            return PRESENCE_IOS_CACHE

    # 2) disk
    events_file, gen_at = _load_presence_ios_cache_from_file()
    if events_file is not None and gen_at is not None:
        if now_utc - gen_at < PRESENCE_CACHE_TTL:
            PRESENCE_IOS_CACHE = events_file
            PRESENCE_IOS_CACHE_EXPIRES_AT = gen_at + PRESENCE_CACHE_TTL
            return events_file

    # 3) rebuild
    try:
        return refresh_presence_ios_cache()
    except Exception as e:
        logger.error("Failed to refresh Presence iOS cache: %s", e)
        if events_file is not None:
            return events_file
        if PRESENCE_IOS_CACHE is not None:
            return PRESENCE_IOS_CACHE
        raise


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@events_bp.route("/presence_events", methods=["GET"])
def presence_events():
    """Web version: Normalized for /get/events shape."""
    try:
        events = get_presence_events_cached()
        return jsonify(events)
    except Exception as e:
        return jsonify({"error": "Failed to load Presence events", "details": str(e)}), 502


@events_bp.route("/presence_events_ios", methods=["GET"])
def presence_events_ios():
    """iOS version: Modified format + locations.json location matching."""
    try:
        events = get_presence_events_ios_cached()
        return jsonify(events)
    except Exception as e:
        return jsonify({"error": "Failed to load Presence iOS events", "details": str(e)}), 502


@events_bp.route("/rss_events", methods=["GET"])
def rss_events():
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
    now = datetime.now()
    docs = list(events_col.find().sort("start_dt", 1))

    events = []
    for d in docs:
        date_str = d.get("date")
        end_at_str = d.get("end_at")
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
    location_at = request.form.get("location_at")
    location = request.form.get("location")
    date_str = request.form.get("date")
    start_at = request.form.get("start_at")
    end_at = request.form.get("end_at")
    host = request.form.get("host")
    title = request.form.get("title")
    description = request.form.get("description")
    poster_file = request.files.get("poster")

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
