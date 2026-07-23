# events.py
import os
import json
import re
import atexit
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import uuid

import requests
import xml.etree.ElementTree as ET

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:  # The API can still run, but scheduled jobs are disabled.
    BackgroundScheduler = None
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from html import unescape
import logging

from helper.normalize_location import normalize_event_location
from helper.location_map import LOCATION_MAP

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

# Warm-up URL
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

# 3. Content API (Pages) Cache
PRESENCE_PAGES_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "presence_pages_cache.json")
PRESENCE_PAGES_CACHE: list[dict] | None = None
PRESENCE_PAGES_CACHE_EXPIRES_AT: datetime | None = None


# ─────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_html_to_text(html_text: str) -> str:
    text = unescape(html_text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split())


def _presence_session() -> requests.Session:
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
    if not normalized_name:
        return None
    target = normalized_name.lower().strip()
    for item in locations_data:
        names = item.get("name", [])
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
    cdn_base = None
    campus_api_id = None
    try:
        campus_info = _get_json_or_raise(s, PRESENCE_CAMPUS_URL, timeout=15)
        if isinstance(campus_info, dict):
            cdn_base = campus_info.get("cdn")
            campus_api_id = campus_info.get("apiId")
    except Exception as e:
        logger.warning("Failed to fetch Presence campus info: %s", e)

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
    s = _presence_session()
    locations_data = load_locations_data()
    cdn_base = None
    campus_api_id = None
    try:
        campus_info = _get_json_or_raise(s, PRESENCE_CAMPUS_URL, timeout=15)
        if isinstance(campus_info, dict):
            cdn_base = campus_info.get("cdn")
            campus_api_id = campus_info.get("apiId")
    except Exception as e:
        logger.warning("Failed to fetch Presence campus info: %s", e)

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
            start_utc = datetime.fromisoformat(
                start_utc_str.replace("Z", "+00:00"))
            end_utc = datetime.fromisoformat(
                end_utc_str.replace("Z", "+00:00"))
            end_local = end_utc.astimezone(PACIFIC)
        except ValueError:
            continue

        if has_ended or end_local <= now_pacific:
            continue

        description_text = clean_html_to_text(ev.get("description") or "")
        raw_loc = ev.get("location") or ""
        cleaned_loc = normalize_event_location(raw_loc)

        loc_id = None
        lat, lon = 37.3655, -120.4245

        if cleaned_loc:
            loc_data = LOCATION_MAP.get(cleaned_loc.lower())
            if loc_data:
                loc_id = loc_data.get("id")
                coords = loc_data.get("coordinates")
                if coords and len(coords) == 2:
                    lon, lat = coords

        poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"
        if cdn_base and campus_api_id and ev.get("hasCoverImage") and ev.get("photoUri"):
            poster_url = f"{cdn_base}/event-photos/{campus_api_id}/{ev['photoUri']}"

        image_urls = [poster_url]
        pin_url = "/event-pin.png"
        evt_id = str(ev.get('eventNoSqlId') or ev.get('uri') or uuid.uuid4())

        ios_event = {
            "id": evt_id,
            "location_id": loc_id,
            "attributes": {
                "title": ev.get("eventName"),
                "description": description_text,
                "start": start_utc_str,
                "end": end_utc_str,
                "location_at": raw_loc,
                "location": cleaned_loc,
                "host": ev.get("organizationName") or ev.get("campusName"),
                "source_url": f"https://ucmerced.presence.io/event/{ev.get('urlName')}" if ev.get('urlName') else "https://ucmerced.presence.io",
                "image_urls": image_urls,
                "pin_url": pin_url
            },
            "geometry": {
                "latitude": lat,
                "longitude": lon
            }
        }

        ios_events.append(ios_event)

    return ios_events

# ─────────────────────────────────────────
# Presence events builder (CONTENT API)
# ─────────────────────────────────────────


def build_content_pages_events() -> list[dict]:
    s = _presence_session()
    locations_data = load_locations_data()
    cdn_base = None
    campus_api_id = None

    try:
        campus_info = _get_json_or_raise(s, PRESENCE_CAMPUS_URL, timeout=15)
        if isinstance(campus_info, dict):
            cdn_base = campus_info.get("cdn")
            campus_api_id = campus_info.get("apiId")
    except Exception as e:
        logger.warning("Failed to fetch Presence campus info: %s", e)

    data = _get_json_or_raise(s, PRESENCE_EVENTS_URL, timeout=20)
    if not isinstance(data, list):
        raise RuntimeError("Presence API did not return a list")

    now_pacific = datetime.now(PACIFIC)
    page_events: list[dict] = []

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
            end_local = end_utc.astimezone(PACIFIC)
        except ValueError:
            continue

        if has_ended or end_local <= now_pacific:
            continue

        description_text = clean_html_to_text(ev.get("description") or "")
        raw_loc = ev.get("location") or ""
        cleaned_loc = normalize_event_location(raw_loc)

        loc_id = None
        lat, lon = 37.3655, -120.4245

        if cleaned_loc:
            loc_data = LOCATION_MAP.get(cleaned_loc.lower())
            if loc_data:
                loc_id = loc_data.get("id")
                coords = loc_data.get("coordinates")
                if coords and len(coords) == 2:
                    lon, lat = coords

        poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"
        if cdn_base and campus_api_id and ev.get("hasCoverImage") and ev.get("photoUri"):
            poster_url = f"{cdn_base}/event-photos/{campus_api_id}/{ev['photoUri']}"

        page_event = {
            "id": str(ev.get('eventNoSqlId') or ev.get('uri') or uuid.uuid4()),
            "type": "event",
            "tags": ["events"],
            "location_id": loc_id,
            "title": ev.get("eventName"),
            "subtitle": raw_loc,
            "description": description_text,
            "image_urls": [poster_url],
            "start": start_utc_str,
            "end": end_utc_str,
            "host": ev.get("organizationName") or ev.get("campusName"),
            "source_url": f"https://ucmerced.presence.io/event/{ev.get('urlName')}" if ev.get('urlName') else "https://ucmerced.presence.io",
            "pin_url": "/event-pin.png",
            "geometry": {
                "type": "point",
                "latitude": lat,
                "longitude": lon
            }
        }

        page_events.append(page_event)

    return page_events

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

    if PRESENCE_CACHE is not None and PRESENCE_CACHE_EXPIRES_AT is not None:
        if now_utc < PRESENCE_CACHE_EXPIRES_AT:
            return PRESENCE_CACHE

    events_file, gen_at = _load_presence_cache_from_file()
    if events_file is not None and gen_at is not None:
        if now_utc - gen_at < PRESENCE_CACHE_TTL:
            PRESENCE_CACHE = events_file
            PRESENCE_CACHE_EXPIRES_AT = gen_at + PRESENCE_CACHE_TTL
            return events_file

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

    if PRESENCE_IOS_CACHE is not None and PRESENCE_IOS_CACHE_EXPIRES_AT is not None:
        if now_utc < PRESENCE_IOS_CACHE_EXPIRES_AT:
            return PRESENCE_IOS_CACHE

    events_file, gen_at = _load_presence_ios_cache_from_file()
    if events_file is not None and gen_at is not None:
        if now_utc - gen_at < PRESENCE_CACHE_TTL:
            PRESENCE_IOS_CACHE = events_file
            PRESENCE_IOS_CACHE_EXPIRES_AT = gen_at + PRESENCE_CACHE_TTL
            return events_file

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
# Cache Management (CONTENT PAGES)
# ─────────────────────────────────────────


def _save_presence_pages_cache_to_file(events: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }
    with open(PRESENCE_PAGES_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_presence_pages_cache_from_file() -> tuple[list[dict] | None, datetime | None]:
    try:
        with open(PRESENCE_PAGES_CACHE_FILE, "r", encoding="utf-8") as f:
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


def refresh_presence_pages_cache() -> list[dict]:
    global PRESENCE_PAGES_CACHE, PRESENCE_PAGES_CACHE_EXPIRES_AT
    events = build_content_pages_events()
    PRESENCE_PAGES_CACHE = events
    PRESENCE_PAGES_CACHE_EXPIRES_AT = datetime.now(
        timezone.utc) + PRESENCE_CACHE_TTL
    _save_presence_pages_cache_to_file(events)
    logger.info("[presence_pages_cache] refreshed %d events", len(events))
    return events


def get_presence_pages_cached() -> list[dict]:
    global PRESENCE_PAGES_CACHE, PRESENCE_PAGES_CACHE_EXPIRES_AT
    now_utc = datetime.now(timezone.utc)

    if PRESENCE_PAGES_CACHE is not None and PRESENCE_PAGES_CACHE_EXPIRES_AT is not None:
        if now_utc < PRESENCE_PAGES_CACHE_EXPIRES_AT:
            return PRESENCE_PAGES_CACHE

    events_file, gen_at = _load_presence_pages_cache_from_file()
    if events_file is not None and gen_at is not None:
        if now_utc - gen_at < PRESENCE_CACHE_TTL:
            PRESENCE_PAGES_CACHE = events_file
            PRESENCE_PAGES_CACHE_EXPIRES_AT = gen_at + PRESENCE_CACHE_TTL
            return events_file

    try:
        return refresh_presence_pages_cache()
    except Exception as e:
        logger.error("Failed to refresh Presence pages cache: %s", e)
        if events_file is not None:
            return events_file
        if PRESENCE_PAGES_CACHE is not None:
            return PRESENCE_PAGES_CACHE
        raise


# ─────────────────────────────────────────
# Scheduled Content Pipeline
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
POLYGONS_JSON_PATH = BASE_DIR / "polygons.json"

MENU_API_URL = "https://widget.api.eagle.bigzpoon.com/menuitems"
MENU_COMP_ID = "61bd7ecd8c760e0011ac0fac"
MENU_DEVICE_ID = "d1f39079-6fac-4eac-bf37-29208df87571"
MENU_REQUEST_DELAY_SECONDS = float(
    os.getenv("MENU_REQUEST_DELAY_SECONDS", "1"))

MENU_USER_PREFERENCES = {
    "allergies": [],
    "lifestyleChoices": [],
    "medicalGoals": [],
    "preferenceApplyStatus": False,
}

MENU_GROUP_IDS = {
    "Sunday": "61bd808b5f2f930010bb6a7a",
    "Monday": "61bd80908b34640010e194b3",
    "Tuesday": "61bd80ab5f2f930010bb6a7d",
    "Wednesday": "61bd80b08b34640010e194b4",
    "Thursday": "61bd80b55f2f930010bb6a7e",
    "Friday": "61bd80ba5f2f930010bb6a7f",
    "Saturday": "61bd80bf8b34640010e194b6",
}

MENU_LOCATION_CONFIG = {
    "YWDC": {
        "location_id": "628672b52903a50010fa751e",
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "meals": [
            ("Lunch", "64b6fe23e615eb39f2b65a5e"),
            ("Dinner", "64b6fe4de615eb39f2b65e9f"),
            ("Late Night", "63320eb6007b6b0010480cad"),
            ("Midori Lunch/Dinner", "6969546c51a912b229ba172d"),
            ("Midori Late Night", "6969548177b727b0edc87615"),
            ("Suannai", "696954f151a912b229ba1890"),
            ("Baked 'n' Grams", "6969537651a912b229ba16f5"),
        ],
    },
    "PAV": {
        "location_id": "61df4a34d5507a00103ee41e",
        "days": [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ],
        "meals": [
            ("Breakfast", "61bd80d68b34640010e194b8"),
            ("Lunch", "61bd80d05f2f930010bb6a81"),
            ("Dinner", "61bd80cc5f2f930010bb6a80"),
            ("Bakery", "62d9c7c26c04ea00104859c7"),
            ("FoG Breakfast", "62daecc36c04ea001048a55d"),
            ("FoG Lunch & Dinner", "61ed885a9be79300147d3898"),
            ("CG Lunch & Dinner", "61ed97b39be79300147d3d06"),
        ],
    },
}

CONTENT_PIPELINE_LOCK = threading.Lock()
CONTENT_SCHEDULER = None
STARTUP_PIPELINE_STARTED = False


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _write_json_atomically(path: Path, payload: Any) -> None:
    """Replace a JSON file without exposing partially written JSON."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _get_polygon_records(polygons_payload: Any) -> list[dict]:
    if isinstance(polygons_payload, list):
        return polygons_payload

    if isinstance(polygons_payload, dict):
        polygons = polygons_payload.get("polygons")
        if isinstance(polygons, list):
            return polygons

    raise ValueError(
        "polygons.json must be a JSON array or an object containing a 'polygons' array"
    )


def _build_menu_fetch_queue() -> list[dict]:
    queue: list[dict] = []

    for source_name, config in MENU_LOCATION_CONFIG.items():
        for day in config["days"]:
            for meal_name, category_id in config["meals"]:
                queue.append({
                    "source": source_name,
                    "day": day,
                    "meal": meal_name,
                    "category_id": category_id,
                    "location_id": config["location_id"],
                    "menu_group_id": MENU_GROUP_IDS[day],
                })

    return queue


def _menu_item_to_section(item: dict) -> dict | None:
    item_name = item.get("name")
    if not isinstance(item_name, str) or not item_name.strip():
        return None

    bullets: list[str] = []
    description = item.get("description") or ""
    if not isinstance(description, str):
        description = str(description)

    if ":" in description:
        station, description = description.split(":", 1)
        station = re.sub(r"[@π]", "", station).strip()
        description = description.strip()
        if station:
            bullets.append(f"Station: {station}")

    if description:
        bullets.append(f"Description: {description}")

    calories = item.get("caloriesInfo")
    if calories:
        bullets.append(f"Calories: {calories}")

    section = {
        "header": item_name.strip(),
        "bullets": bullets,
    }

    image_url = item.get("imageUrl")
    if isinstance(image_url, str) and image_url.strip():
        section["image_urls"] = [image_url.strip()]

    return section


def _build_nested_content(location_data: dict) -> list[dict]:
    days_order = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]

    nested_content: list[dict] = []

    for day in days_order:
        meals = location_data.get(day)
        if not isinstance(meals, dict):
            continue

        tabs = []
        for meal_name, sections in meals.items():
            tabs.append({
                "title": meal_name,
                "sections": sections,
            })

        nested_content.append({
            "title": day,
            "tabs": tabs,
        })

    return nested_content


def generate_food_menu_payload() -> dict:
    """Fetch all Pavilion/YWDC menu endpoints and build nested_content arrays."""
    fetch_queue = _build_menu_fetch_queue()
    parsed_data = {"YWDC": {}, "PAV": {}}
    failures: list[str] = []

    user_preferences = json.dumps(
        MENU_USER_PREFERENCES,
        separators=(",", ":"),
    )

    session = requests.Session()

    logger.info(
        "[food_menu] starting %d menu requests",
        len(fetch_queue),
    )

    for index, task in enumerate(fetch_queue, start=1):
        source = task["source"]
        day = task["day"]
        meal = task["meal"]

        parsed_data[source].setdefault(day, {})
        parsed_data[source][day].setdefault(meal, [])

        params = {
            "categoryId": task["category_id"],
            "locationId": task["location_id"],
            "menuGroupId": task["menu_group_id"],
            "userPreferences": user_preferences,
        }
        headers = {
            "accept": "application/json, text/plain, */*",
            "x-comp-id": MENU_COMP_ID,
            "location-id": task["location_id"],
            "device-id": MENU_DEVICE_ID,
        }

        try:
            response = session.get(
                MENU_API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()

            payload_data = payload.get("data") if isinstance(
                payload, dict) else None
            menu_items = (
                payload_data.get("menuItems", [])
                if isinstance(payload_data, dict)
                else []
            )

            if not isinstance(menu_items, list):
                raise ValueError("response data.menuItems is not an array")

            existing_headers = {
                section.get("header")
                for section in parsed_data[source][day][meal]
                if isinstance(section, dict)
            }

            for item in menu_items:
                if not isinstance(item, dict):
                    continue

                section = _menu_item_to_section(item)
                if section is None or section["header"] in existing_headers:
                    continue

                parsed_data[source][day][meal].append(section)
                existing_headers.add(section["header"])

            logger.info(
                "[food_menu] fetched %d/%d: %s %s - %s",
                index,
                len(fetch_queue),
                source,
                day,
                meal,
            )
        except Exception as exc:
            failure = f"{source} {day} - {meal}: {exc}"
            failures.append(failure)
            logger.error("[food_menu] %s", failure)

        if index < len(fetch_queue) and MENU_REQUEST_DELAY_SECONDS > 0:
            time.sleep(MENU_REQUEST_DELAY_SECONDS)

    if failures:
        raise RuntimeError(
            "Food menu generation failed for "
            f"{len(failures)} endpoint(s): "
            + " | ".join(failures[:5])
        )

    return {
        "YWDC_nested_content": _build_nested_content(parsed_data["YWDC"]),
        "PAV_nested_content": _build_nested_content(parsed_data["PAV"]),
    }


def _replace_food_menu_nested_content(menu_payload: dict) -> dict[str, int]:
    if not POLYGONS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"polygons.json not found: {POLYGONS_JSON_PATH}")

    with POLYGONS_JSON_PATH.open("r", encoding="utf-8") as f:
        polygons_payload = json.load(f)

    polygon_records = _get_polygon_records(polygons_payload)

    pavilion_content = menu_payload.get("PAV_nested_content")
    ywdc_content = menu_payload.get("YWDC_nested_content")

    if not isinstance(pavilion_content, list):
        raise ValueError("PAV_nested_content is missing or invalid")
    if not isinstance(ywdc_content, list):
        raise ValueError("YWDC_nested_content is missing or invalid")

    replacements = {
        "774": {
            "expected_name": "Pavilion",
            "nested_content": pavilion_content,
        },
        "1130": {
            "expected_name": "DC",
            "nested_content": ywdc_content,
        },
    }
    matched = {"774": 0, "1130": 0}

    for record in polygon_records:
        if not isinstance(record, dict):
            continue

        location_id = str(record.get("location_id", ""))
        replacement = replacements.get(location_id)
        if replacement is None:
            continue

        actual_name = record.get("name")
        expected_name = replacement["expected_name"]
        if actual_name and actual_name != expected_name:
            logger.warning(
                "[food_menu] location_id=%s has name=%r; expected %r; updating by ID",
                location_id,
                actual_name,
                expected_name,
            )

        record["nested_content"] = replacement["nested_content"]
        matched[location_id] += 1

    missing_ids = [location_id for location_id,
                   count in matched.items() if count == 0]
    if missing_ids:
        raise ValueError(
            "Missing required location_id value(s) in polygons.json: "
            + ", ".join(missing_ids)
        )

    _write_json_atomically(POLYGONS_JSON_PATH, polygons_payload)

    logger.info(
        "[food_menu] updated polygons.json: Pavilion=%d, DC=%d",
        matched["774"],
        matched["1130"],
    )
    return matched


def generate_food_menus_and_update_polygons() -> dict[str, int]:
    """Public/manual function for a complete weekly menu update."""
    with CONTENT_PIPELINE_LOCK:
        menu_payload = generate_food_menu_payload()
        return _replace_food_menu_nested_content(menu_payload)


def refresh_content_api_presence_cache() -> int:
    """Force-refresh the Presence pages used by /contentAPIURL."""
    with CONTENT_PIPELINE_LOCK:
        events = refresh_presence_pages_cache()
        logger.info(
            "[content_jobs] refreshed Presence pages cache with %d events",
            len(events),
        )
        return len(events)


def run_startup_content_pipeline() -> None:
    """Startup testing sequence: menus -> polygons -> Presence pages cache."""
    with CONTENT_PIPELINE_LOCK:
        menu_payload = generate_food_menu_payload()
        _replace_food_menu_nested_content(menu_payload)
        events = refresh_presence_pages_cache()
        logger.info(
            "[content_jobs] startup pipeline completed with %d Presence events",
            len(events),
        )


def _safe_content_job(job_name: str, function) -> None:
    try:
        function()
    except Exception:
        logger.exception("[content_jobs] %s failed", job_name)


def _shutdown_content_scheduler() -> None:
    global CONTENT_SCHEDULER
    if CONTENT_SCHEDULER is not None and CONTENT_SCHEDULER.running:
        CONTENT_SCHEDULER.shutdown(wait=False)


def init_content_jobs(app):
    """Start the Pacific-time scheduler once for this Flask process."""
    global CONTENT_SCHEDULER, STARTUP_PIPELINE_STARTED

    if not _env_flag("RUN_CONTENT_JOBS", True):
        logger.info("[content_jobs] scheduler disabled by RUN_CONTENT_JOBS")
        return None

    if BackgroundScheduler is None:
        logger.error(
            "[content_jobs] APScheduler is not installed; run: "
            "pip install 'APScheduler>=3.11,<4'"
        )
        return None

    debug_enabled = app.debug or _env_flag("FLASK_DEBUG", False)
    if debug_enabled and os.getenv("WERKZEUG_RUN_MAIN") != "true":
        return None

    if CONTENT_SCHEDULER is not None:
        return CONTENT_SCHEDULER

    scheduler = BackgroundScheduler(
        timezone=PACIFIC,
        daemon=True,
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 3600,
        },
    )

    scheduler.add_job(
        lambda: _safe_content_job(
            "Sunday food-menu generation",
            generate_food_menus_and_update_polygons,
        ),
        trigger="cron",
        id="weekly_food_menu_generation",
        day_of_week="sun",
        hour=7,
        minute=0,
        replace_existing=True,
    )

    scheduler.add_job(
        lambda: _safe_content_job(
            "daily Presence pages cache refresh",
            refresh_content_api_presence_cache,
        ),
        trigger="cron",
        id="daily_presence_pages_cache_refresh",
        hour=7,
        minute=5,
        replace_existing=True,
    )

    scheduler.start()
    CONTENT_SCHEDULER = scheduler
    atexit.register(_shutdown_content_scheduler)

    logger.info(
        "[content_jobs] scheduler started: Sunday 07:00 menus; daily 07:05 cache"
    )

    if (
        _env_flag("RUN_STARTUP_CONTENT_PIPELINE", True)
        and not STARTUP_PIPELINE_STARTED
    ):
        STARTUP_PIPELINE_STARTED = True
        startup_thread = threading.Thread(
            target=lambda: _safe_content_job(
                "startup content pipeline",
                run_startup_content_pipeline,
            ),
            name="startup-content-pipeline",
            daemon=True,
        )
        startup_thread.start()

    return scheduler


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────


@events_bp.route("/contentAPIURL", methods=["GET"])
def content_api_url():
    """
    Unified endpoint that aggregates multiple data sources into a single JSON payload.
    Structured as a pipeline so future data sources can be appended easily to the 'pages' array.
    """
    aggregated_response = {
        "pages": []
    }

    # ─── STEP A: Add Event Pages ───
    try:
        events = get_presence_pages_cached()
        aggregated_response["pages"].extend(events)
    except Exception as e:
        logger.error("ContentAPI Pipeline Error (Events): %s", e)

    # ─── STEP B: Add Migrated Polygons ───
    try:
        # Assuming polygons.json is saved in the same directory as events.py
        polygons_file_path = os.path.join(
            os.path.dirname(__file__), "polygons.json")

        if os.path.exists(polygons_file_path):
            with open(polygons_file_path, "r", encoding="utf-8") as f:
                polygons_data = json.load(f)

                if isinstance(polygons_data, list):
                    aggregated_response["pages"].extend(polygons_data)
                elif isinstance(polygons_data, dict) and "polygons" in polygons_data:
                    aggregated_response["pages"].extend(
                        polygons_data["polygons"])
    except Exception as e:
        logger.error("ContentAPI Pipeline Error (Polygons): %s", e)

    return jsonify(aggregated_response)


@events_bp.route("/presence_events", methods=["GET"])
def presence_events():
    try:
        events = get_presence_events_cached()
        return jsonify(events)
    except Exception as e:
        return jsonify({"error": "Failed to load Presence events", "details": str(e)}), 502


@events_bp.route("/presence_events_ios", methods=["GET"])
def presence_events_ios():
    try:
        events = get_presence_events_ios_cached()
        return jsonify({"events": events})
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


# Start the scheduler automatically when this blueprint is registered.
@events_bp.record_once
def _start_content_jobs_when_blueprint_registers(state) -> None:
    init_content_jobs(state.app)
