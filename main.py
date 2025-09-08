# # -*- coding: utf-8 -*-
# """
# Flask-based proxy and AI interface for UC Merced campus events.

# This application serves two main purposes:
# 1. Acts as a robust, caching proxy for an upstream events API. It handles
#    retries, timeouts, and aggregates data from multiple potential sources.
# 2. Provides an AI-powered conversational endpoint (/ask) to query event
#    information using natural language (routes not shown in this snippet).

# Key features include an in-memory cache, request validation to prevent
# resource exhaustion, and dynamic configuration via environment variables.
# """

# import os
# import json
# import math
# import time
# from collections import defaultdict, deque
# from datetime import datetime, date, timedelta, timezone

# # --- Third-party libraries ---
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# from openai import OpenAI


# # ─────────────────────────────
# # Setup & Configuration
# # ─────────────────────────────
# load_dotenv()

# # --- Core App Setup ---
# app = Flask(__name__)
# CORS(app)

# # --- OpenAI Client ---
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # --- Environment Detection ---
# ON_RENDER = bool(os.getenv("RENDER") or os.getenv("RENDER_EXTERNAL_URL"))

# # --- Upstream API Configuration ---
# # Primary base URL (kept for backward-compat)
# _primary_events_url = os.getenv(
#     "EVENTS_BASE_URL",
#     "https://uc-merced-campus-event-api-backend.onrender.com"
# ).rstrip("/")

# # Default base URLs:
# # - On Render: just the Render backend
# # - Local dev: Render first, then localhost
# _default_urls = [_primary_events_url] if ON_RENDER else [
#     _primary_events_url, "http://localhost:7050"]
# EVENTS_BASE_URLS = [
#     u.strip().rstrip("/")
#     for u in os.getenv("EVENTS_BASE_URLS", ",".join(_default_urls)).split(",")
#     if u.strip()
# ]

# # --- Performance & Caching ---
# CONNECT_TIMEOUT = float(os.getenv("EVENTS_CONNECT_TIMEOUT", "4.5"))
# READ_TIMEOUT = float(os.getenv("EVENTS_READ_TIMEOUT", "20.0"))
# CACHE_TTL_SEC = int(os.getenv("EVENTS_CACHE_TTL_SECONDS", "300"))  # 5 minutes
# MAX_DATE_RANGE_DAYS = 31  # Prevents memory exhaustion from large requests

# # --- In-memory cache: key -> (timestamp, payload, source_url) ---
# EVENTS_CACHE = {}

# # --- AI Chat History & Token Limits ---
# message_history = defaultdict(lambda: deque(maxlen=10))
# MAX_CONTEXT_TOKENS = 3000
# MAX_COMPLETION_TOKENS = 800

# # --- Location Data ---
# LOCATION_LOOKUP = {
#     "UC Merced": {"lat": 37.3660, "lon": -120.4246},
#     "Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
#     "Leo and Dottie Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
#     "Arts and Computational Sciences (ACS)": {"lat": 37.3670, "lon": -120.4255},
#     "Scholars Lane": {"lat": 37.366117, "lon": -120.424205},
# }
# LOCATIONS_PATH = os.path.join(os.getcwd(), "locations.json")
# if os.path.exists(LOCATIONS_PATH):
#     try:
#         with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
#             loaded = json.load(f)
#             if isinstance(loaded, dict):
#                 LOCATION_LOOKUP.update(loaded)
#     except Exception as e:
#         app.logger.warning(f"Could not load locations.json: {e}")


# # ─────────────────────────────
# # Shared Helper Functions
# # ─────────────────────────────

# def first(*vals, default=None):
#     """Returns the first non-None value in a sequence."""
#     for v in vals:
#         if v is not None:
#             return v
#     return default


# def to_bool(v, default=False):
#     """Converts a value to a boolean, handling common string representations."""
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return default
#     return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# def parse_iso(dt_str: str) -> datetime:
#     """Robustly parses an ISO 8601 string into a timezone-aware datetime object."""
#     if not dt_str:
#         return None
#     s = dt_str.strip()
#     if s.endswith("Z"):
#         s = s[:-1] + "+00:00"
#     try:
#         return datetime.fromisoformat(s)
#     except ValueError:
#         # Fallback for formats without timezone info, assuming UTC
#         return datetime.fromisoformat(s + "+00:00")


# def lonlat_to_web_mercator(lon: float, lat: float):
#     """Converts longitude and latitude to Web Mercator coordinates (EPSG:3857)."""
#     # Clamp latitude to the valid range for the Mercator projection
#     lat = max(min(lat, 85.05112878), -85.05112878)
#     R = 6378137.0  # Earth's radius in meters
#     x = R * math.radians(lon)
#     y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
#     return x, y


# # ─────────────────────────────
# # HTTP Session with Retries
# # ─────────────────────────────

# def make_session() -> requests.Session:
#     """Creates a requests.Session with configured retries and timeouts."""
#     s = requests.Session()
#     retries = Retry(
#         total=2,          # Max 3 attempts (1 initial + 2 retries)
#         connect=1,        # Retry once on connection errors
#         read=1,           # Retry once on read timeouts
#         backoff_factor=0.4,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=frozenset(["GET"])
#     )
#     adapter = HTTPAdapter(max_retries=retries,
#                           pool_connections=10, pool_maxsize=10)
#     s.mount("http://", adapter)
#     s.mount("https://", adapter)
#     s.headers.update({
#         "User-Agent": "ucm-events-proxy/1.1",
#         "Accept": "application/json"
#     })
#     return s


# SESSION = make_session()


# # ─────────────────────────────
# # Caching Logic
# # ─────────────────────────────

# def cache_get(key):
#     """Retrieves an item from the cache if it's not expired."""
#     rec = EVENTS_CACHE.get(key)
#     if not rec:
#         return None
#     ts, payload, source = rec
#     if (time.time() - ts) <= CACHE_TTL_SEC:
#         return "hit", payload, source
#     return "stale", payload, source


# def cache_put(key, payload, source):
#     """Adds an item to the cache."""
#     EVENTS_CACHE[key] = (time.time(), payload, source)


# # ─────────────────────────────
# # Core Event Fetching Logic
# # ─────────────────────────────

# def fetch_and_process_events(from_iso: str = None, to_iso: str = None, base_urls=None, nocache=False):
#     """
#     Fetches events from upstream APIs, processes them into a standard format,
#     and utilizes an in-memory cache.
#     """
#     base_urls = base_urls or EVENTS_BASE_URLS

#     # Set default time window (next 7 days) if not provided
#     if not from_iso or not to_iso:
#         start_date = datetime.now(timezone.utc)
#         end_date = start_date + timedelta(days=7)
#         from_iso = start_date.isoformat().replace("+00:00", "Z")
#         to_iso = end_date.isoformat().replace("+00:00", "Z")

#     cache_key = (from_iso, to_iso, tuple(base_urls))
#     if not nocache:
#         cache_result = cache_get(cache_key)
#         if cache_result:
#             state, payload, source = cache_result
#             return payload, source, state  # state: "hit" or "stale"

#     params = {"from": from_iso, "to": to_iso}
#     errors = []

#     for base in base_urls:
#         url = f"{base}/events"
#         try:
#             r = SESSION.get(url, params=params, timeout=(
#                 CONNECT_TIMEOUT, READ_TIMEOUT))
#             r.raise_for_status()
#             raw_data = r.json()
#             events_list = raw_data.get("events") if isinstance(
#                 raw_data, dict) else raw_data
#             if not isinstance(events_list, list):
#                 app.logger.warning(
#                     f"API at {base} returned non-list data: {type(events_list)}")
#                 events_list = []

#             processed = []
#             for e in events_list:
#                 if not isinstance(e, dict):
#                     continue

#                 start_raw = first(e.get("start"), e.get(
#                     "startAt"), e.get("start_time"))
#                 start_dt = parse_iso(start_raw) if isinstance(
#                     start_raw, str) else None
#                 if not start_dt:
#                     continue  # Skip events without a valid start time

#                 end_raw = first(e.get("end"), e.get(
#                     "endAt"), e.get("end_time"))
#                 end_dt = parse_iso(end_raw) if isinstance(
#                     end_raw, str) else None

#                 location_name = first(
#                     e.get("location_name"), e.get("location"), e.get("venue"))
#                 lon = first(e.get("lon"), e.get("lng"), e.get("longitude"))
#                 lat = first(e.get("lat"), e.get("latitude"))

#                 # Geocode known locations if coordinates are missing
#                 if (lon is None or lat is None) and location_name in LOCATION_LOOKUP:
#                     coords = LOCATION_LOOKUP[location_name]
#                     lat, lon = coords.get("lat"), coords.get("lon")

#                 geometry = None
#                 if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
#                     x, y = lonlat_to_web_mercator(float(lon), float(lat))
#                     geometry = {"type": "point", "x": x, "y": y,
#                                 "spatialReference": {"wkid": 3857}}

#                 processed.append({
#                     "id": str(first(e.get("id"), e.get("_id"), e.get("uuid"), default="")),
#                     "event_name": first(e.get("title"), e.get("name"), e.get("event_name"), default=""),
#                     "description": first(e.get("description"), e.get("desc")),
#                     "url": first(e.get("url"), e.get("link")),
#                     "date": start_dt.date().isoformat(),
#                     "startAt": start_dt.strftime("%H:%M"),
#                     "endAt": end_dt.strftime("%H:%M") if end_dt else None,
#                     "locationTag": location_name,
#                     "names": e.get("names") if isinstance(e.get("names"), list) else None,
#                     "geometry": geometry,
#                     "fromUser": to_bool(e.get("fromUser", False)),
#                     # "original": e, # CRITICAL: REMOVED to prevent high memory usage
#                 })

#             cache_put(cache_key, processed, base)
#             return processed, base, "miss"

#         except requests.exceptions.RequestException as ex:
#             errors.append(f"Request failed for {base}: {ex}")
#         except json.JSONDecodeError as ex:
#             errors.append(f"JSON decode error for {base}: {ex}")
#         except Exception as ex:
#             errors.append(f"An unexpected error occurred for {base}: {ex}")
#             app.logger.error(
#                 f"Error processing events from {base}", exc_info=True)

#     # If all upstream sources failed, try to serve stale cache as a last resort
#     stale_result = cache_get(cache_key)
#     if stale_result and stale_result[0] == "stale":
#         _, payload, source = stale_result
#         return payload, source, "stale-fallback"

#     raise RuntimeError(
#         "All upstream event services failed: " + " ; ".join(errors))


# # ─────────────────────────────
# # API Routes
# # ─────────────────────────────

# @app.route("/", methods=["GET"])
# def root():
#     """Provides basic service information."""
#     return jsonify({
#         "status": "ok",
#         "service_name": "ucm-events-proxy",
#         "on_render": ON_RENDER,
#         "configured_event_sources": EVENTS_BASE_URLS,
#         "cache_ttl_seconds": CACHE_TTL_SEC,
#         "max_date_range_days": MAX_DATE_RANGE_DAYS,
#         # Add /ask (POST) if using it
#         "endpoints": ["/health (GET)", "/events (GET)"]
#     })


# @app.route("/health", methods=["GET"])
# def health():
#     """Simple health check endpoint."""
#     return jsonify({"status": "ok"})


# @app.route("/events", methods=["GET"])
# def get_events():
#     """
#     Fetches and returns campus events within a specified date range.

#     Query Params:
#       - from (ISO8601, optional): Start date/time.
#       - to (ISO8601, optional): End date/time.
#       - nocache (1, optional): Bypass the cache for a fresh fetch.
#       - source (render|local, optional): Force use of a specific source type.
#     """
#     try:
#         from_iso = request.args.get("from")
#         to_iso = request.args.get("to")
#         nocache = (request.args.get("nocache") == "1")

#         # --- ROBUSTNESS: Validate date range to prevent memory exhaustion ---
#         try:
#             # Provide sensible defaults if params are missing
#             now_utc = datetime.now(timezone.utc)
#             from_dt = parse_iso(from_iso) if from_iso else now_utc
#             to_dt = parse_iso(to_iso) if to_iso else now_utc + \
#                 timedelta(days=7)

#             if from_dt is None or to_dt is None:
#                 raise ValueError("Invalid 'from' or 'to' date format.")

#             if (to_dt - from_dt).days > MAX_DATE_RANGE_DAYS:
#                 return jsonify({
#                     "error": f"Date range too large. Please request a maximum of {MAX_DATE_RANGE_DAYS} days."
#                 }), 400
#         except (ValueError, TypeError) as e:
#             return jsonify({"error": f"Invalid date format: {e}"}), 400
#         # --- End Validation ---

#         source_pref = (request.args.get("source") or "").strip().lower()
#         if source_pref == "render":
#             bases = [u for u in EVENTS_BASE_URLS if "onrender.com" in u]
#         elif source_pref == "local":
#             bases = [u for u in EVENTS_BASE_URLS if "localhost" in u]
#         else:
#             bases = EVENTS_BASE_URLS

#         if not bases:
#             return jsonify({"error": f"No event sources configured for preference '{source_pref}'"}), 500

#         events, used_source, cache_state = fetch_and_process_events(
#             from_iso, to_iso, bases, nocache=nocache)

#         resp = jsonify(events)
#         resp.headers["X-Events-Source"] = used_source
#         resp.headers["X-Cache-Status"] = cache_state
#         return resp

#     except RuntimeError as e:
#         app.logger.error(f"Failed to fetch events from all sources: {e}")
#         return jsonify({"error": f"Failed to fetch events: {e}"}), 502
#     except Exception as e:
#         app.logger.error(
#             "An unexpected error occurred in /events", exc_info=True)
#         return jsonify({"error": f"An internal server error occurred: {e}"}), 500


# # NOTE: Your /ask and /ask/events routes would go here.
# # They were omitted for brevity as per your original post.


# # ─────────────────────────────
# # Application Entrypoint
# # ─────────────────────────────
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8050))
#     # Use host='0.0.0.0' to be accessible on the network
#     # debug=False is crucial for production
#     app.run(host="0.0.0.0", port=port, debug=False)

# from openai import OpenAI
# import os
# import re
# import base64
# import json
# from collections import defaultdict, deque
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv

# # +++ New/Updated Imports +++
# import requests
# from datetime import datetime, date, timedelta, timezone
# import math

# # ─────────────────────────────
# # Setup
# # ─────────────────────────────
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app = Flask(__name__)
# CORS(app)

# # Where to fetch events from (Render by default; override for local dev)
# EVENTS_BASE_URL = os.getenv(
#     "EVENTS_BASE_URL",
#     "https://uc-merced-campus-event-api-backend.onrender.com"
# ).rstrip("/")

# # ─────────────────────────────
# # Shared helpers
# # ─────────────────────────────


# def extract_json(raw: str) -> dict:
#     """
#     Clean up model output and return the first {...} JSON object inside.
#     Raises ValueError if no valid JSON is found.
#     """
#     # Strip fenced code blocks if present
#     if raw.startswith("```") and raw.endswith("```"):
#         raw = raw.strip("```").strip()
#     start = raw.find("{")
#     if start == -1:
#         raise ValueError("No JSON object found in model output")
#     depth = 0
#     end = None
#     for i, ch in enumerate(raw[start:], start):
#         if ch == "{":
#             depth += 1
#         elif ch == "}":
#             depth -= 1
#             if depth == 0:
#                 end = i
#                 break
#     if end is None:
#         raise ValueError("Unbalanced braces in model output")
#     json_str = raw[start: end + 1]
#     return json.loads(json_str)


# def parse_iso(dt_str: str) -> datetime:
#     """
#     Parse ISO8601 strings, handling 'Z' as UTC.
#     Returns timezone-aware datetime when possible.
#     """
#     if not dt_str:
#         return None
#     s = dt_str.strip()
#     if s.endswith("Z"):
#         s = s[:-1] + "+00:00"
#     try:
#         return datetime.fromisoformat(s)
#     except Exception:
#         try:
#             return datetime.strptime(dt_str.replace("Z", "+00:00"), "%Y-%m-%dT%H:%M:%S%z")
#         except Exception:
#             return None


# def lonlat_to_web_mercator(lon: float, lat: float):
#     """
#     Convert WGS84 lon/lat to Web Mercator (EPSG:3857).
#     Returns (x, y)
#     """
#     lat = max(min(lat, 85.05112878), -85.05112878)
#     R = 6378137.0
#     x = R * math.radians(lon)
#     y = R * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
#     return x, y


# def first(*vals, default=None):
#     for v in vals:
#         if v is not None:
#             return v
#     return default


# def to_bool(v, default=False):
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return default
#     return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# # ─────────────────────────────
# # Events memory / config
# # ─────────────────────────────
# message_history = defaultdict(lambda: deque(maxlen=10))
# MAX_CONTEXT_TOKENS = 3000
# MAX_COMPLETION_TOKENS = 800


# def approximate_token_count(messages):
#     total = 0
#     for msg in messages:
#         content = msg.get("content", "")
#         if not isinstance(content, str):
#             try:
#                 content = json.dumps(content)
#             except Exception:
#                 content = str(content)
#         total += len(content) // 4
#     return total


# EVENTS = []
# EVENTS_PATH = os.path.join(os.getcwd(), "events.json")
# if os.path.exists(EVENTS_PATH):
#     try:
#         with open(EVENTS_PATH, "r", encoding="utf-8") as f:
#             EVENTS = json.load(f)
#     except Exception:
#         EVENTS = []

# # +++ START: Location lookup (user-provided) +++
# LOCATION_LOOKUP = {
#     "UC Merced": {"lat": 37.3660, "lon": -120.4246},
#     "Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
#     "Leo and Dottie Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
#     "Arts and Computational Sciences (ACS)": {"lat": 37.3670, "lon": -120.4255},
#     "Scholars Lane": {"lat": 37.366117, "lon": -120.424205},
# }

# LOCATIONS_PATH = os.path.join(os.getcwd(), "locations.json")
# if os.path.exists(LOCATIONS_PATH):
#     try:
#         with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
#             loaded = json.load(f)
#             if isinstance(loaded, dict):
#                 LOCATION_LOOKUP.update(loaded)
#     except Exception:
#         pass
# # +++ END: Location lookup (user-provided) +++

# # +++ START: Render Events Integration +++


# def fetch_and_process_local_events(from_iso: str = None, to_iso: str = None):
#     """
#     Fetch events from Render backend:
#       {EVENTS_BASE_URL}/events?from=...&to=...

#     Normalize to frontend schema:
#       {
#         id, event_name, description, date, startAt, endAt,
#         locationTag, names, original, geometry, fromUser, url
#       }
#     """
#     # Defaults: 7-day window if not provided
#     if not from_iso or not to_iso:
#         start_date = datetime.now(timezone.utc).date()
#         end_date = start_date + timedelta(days=7)
#         from_iso = datetime.combine(start_date, datetime.min.time(
#         ), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
#         to_iso = datetime.combine(end_date, datetime.max.time(
#         ), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

#     base_url = f"{EVENTS_BASE_URL}/events"
#     params = {"from": from_iso, "to": to_iso}

#     resp = requests.get(base_url, params=params, timeout=12)
#     resp.raise_for_status()

#     raw = resp.json()
#     events_list = raw.get("events") if isinstance(raw, dict) else raw
#     if not isinstance(events_list, list):
#         events_list = []

#     processed = []

#     for e in events_list:
#         # Basic fields (tolerant of different keys)
#         eid = str(first(e.get("id"), e.get("_id"), e.get("uuid"), default=""))
#         title = first(e.get("title"), e.get("name"),
#                       e.get("event_name"), default="")
#         description = first(e.get("description"), e.get("desc"), default=None)
#         url = first(e.get("url"), e.get("link"), default=None)
#         location_name = first(e.get("location_name"), e.get(
#             "location"), e.get("venue"), default=None)
#         names = e.get("names") if isinstance(e.get("names"), list) else None
#         from_user = to_bool(e.get("fromUser", False))

#         # Start/End
#         start_raw = first(e.get("start"), e.get("startAt"),
#                           e.get("start_time"), default=None)
#         end_raw = first(e.get("end"), e.get("endAt"),
#                         e.get("end_time"), default=None)

#         start_dt = parse_iso(start_raw) if isinstance(start_raw, str) else None
#         end_dt = parse_iso(end_raw) if isinstance(end_raw, str) else None

#         if not start_dt:
#             continue

#         # Coordinates: prefer direct numeric fields, else lookup by location string
#         lon = first(e.get("lon"), e.get("lng"), e.get("longitude"))
#         lat = first(e.get("lat"), e.get("latitude"))

#         if (lon is None or lat is None) and location_name and location_name in LOCATION_LOOKUP:
#             coords = LOCATION_LOOKUP[location_name]
#             lat = first(coords.get("lat"))
#             lon = first(coords.get("lon"))

#         geometry = None
#         if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
#             x, y = lonlat_to_web_mercator(float(lon), float(lat))
#             geometry = {
#                 "type": "point",
#                 "x": float(x),
#                 "y": float(y),
#                 "spatialReference": {"wkid": 3857},
#             }

#         processed.append({
#             "id": eid,
#             "event_name": title,
#             "description": description,
#             "date": start_dt.date().isoformat(),
#             "startAt": start_dt.strftime("%H:%M"),
#             "endAt": (end_dt.strftime("%H:%M") if end_dt else None),
#             "locationTag": location_name,
#             "names": names,
#             "original": e,
#             "geometry": geometry,
#             "fromUser": from_user,
#             "url": url,
#         })

#     return processed
# # +++ END: Render Events Integration +++

# # ─────────────────────────────
# # Routes
# # ─────────────────────────────


# @app.route("/", methods=["GET"])
# def root():
#     return jsonify({
#         "ok": True,
#         "events_base_url": EVENTS_BASE_URL,
#         "endpoints": ["/ask (POST)", "/ask/events (POST)", "/events (GET)"]
#     })


# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"ok": True})

# # Proxies to Render service


# @app.route("/events", methods=["GET"])
# def get_events():
#     """
#     Proxy-normalizer for events service at:
#       {EVENTS_BASE_URL}/events?from=...&to=...

#     Query params (optional):
#       - from: ISO8601 string (e.g., 2025-08-31T07:00:00.000Z)
#       - to:   ISO8601 string (e.g., 2025-10-08T06:59:59.999Z)
#     """
#     try:
#         from_iso = request.args.get("from")
#         to_iso = request.args.get("to")
#         events = fetch_and_process_local_events(from_iso, to_iso)
#         return jsonify(events)
#     except requests.exceptions.RequestException as e:
#         return jsonify({"error": f"Failed to fetch from events service: {e}"}), 502
#     except Exception as e:
#         return jsonify({"error": f"An internal error occurred: {e}"}), 500


# @app.route("/ask", methods=["POST"])
# def ask_vision():
#     if "file" not in request.files:
#         return jsonify({"error": "No image file provided (field name should be 'file')"}), 400

#     uploaded = request.files["file"]
#     img_bytes = uploaded.read()
#     if not img_bytes:
#         return jsonify({"error": "Empty file"}), 400

#     mime = uploaded.mimetype or "image/png"
#     b64 = base64.b64encode(img_bytes).decode("utf-8")
#     data_url = f"data:{mime};base64,{b64}"

#     system_message = {
#         "role": "system",
#         "content": (
#             "You are a vision-enabled assistant. "
#             "Extract from the image: date, time, location, names, event name, and a short description of the event. "
#             "Respond *only* with valid JSON matching this schema:\n\n"
#             "{\n"
#             "  \"date\": \"\",\n"
#             "  \"time\": \"\",\n"
#             "  \"location\": \"\",\n"
#             "  \"names\": [],\n"
#             "  \"event_name\": \"\",\n"
#             "  \"description\": \"\"\n"
#             "}\n"
#             "Rules:\n"
#             "- If a field is unknown, use an empty string (or empty array for names).\n"
#             "- Do not add extra keys. Do not include explanations."
#         ),
#     }

#     user_message = {
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": data_url}},
#         ],
#     }

#     try:
#         resp = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[system_message, user_message],
#             temperature=0.0,
#             max_tokens=400,
#         )

#         raw = (resp.choices[0].message.content or "").strip()
#         if not raw:
#             return jsonify({"error": "Empty model response"}), 502

#         try:
#             result = extract_json(raw)
#         except ValueError:
#             return jsonify({"error": "Failed to extract JSON", "raw_response": raw}), 500

#         return jsonify(result)

#     except json.JSONDecodeError:
#         return jsonify({"error": "Model did not return valid JSON"}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/ask/events", methods=["POST"])
# def ask_events():
#     data = request.get_json(silent=True) or {}
#     user_id = data.get("user_id", "default")
#     question = data.get("question", "")
#     tags = data.get("tags", [])

#     if not isinstance(question, str) or not question.strip():
#         return jsonify({"error": "No question provided"}), 400

#     filtered_events = [
#         event for event in EVENTS
#         if not tags or any(tag in event.get("tags", []) for tag in tags)
#     ]

#     system_message = {
#         "role": "system",
#         "content": (
#             "You are a helpful assistant for UC Merced's Bobcat Day. "
#             "You help students find relevant events based on their interests. "
#             "When recommending events, include their IDs at the end in a JSON array like [\"event002\", \"event004\"]."
#         ),
#     }

#     history = message_history[user_id]
#     history.append({"role": "user", "content": question})

#     context_prompt = (
#         f"User asked: \"{question}\"\n\n"
#         f"Here is a list of events:\n{json.dumps(filtered_events, ensure_ascii=False)}"
#     )

#     messages = [system_message] + \
#         list(history) + [{"role": "user", "content": context_prompt}]

#     while approximate_token_count(messages) > MAX_CONTEXT_TOKENS and len(history) > 0:
#         history.pop()
#         messages = [system_message] + \
#             list(history) + [{"role": "user", "content": context_prompt}]

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             temperature=0.4,
#             max_tokens=MAX_COMPLETION_TOKENS,
#         )

#         reply = (response.choices[0].message.content or "").strip()

#         history.append({"role": "assistant", "content": reply})

#         event_ids = re.findall(r"event\d{3}", reply)
#         matched_events = [
#             event for event in EVENTS if event.get("id") in set(event_ids)]

#         return jsonify({"response": reply, "matched_events": matched_events})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ─────────────────────────────
# # Entrypoint
# # ─────────────────────────────
# if __name__ == "__main__":
#     # On Render, PORT is provided as an env var
#     port = int(os.getenv("PORT", "8050"))
#     app.run(host="0.0.0.0", port=port)

# working local down bellow
from openai import OpenAI
import os
import re
import base64
import json
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# +++ New/Updated Imports +++
import requests
from datetime import datetime, date, timedelta, timezone
import math

# ─────────────────────────────
# Setup
# ─────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

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
        raw = raw.strip("```").strip()
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
    return json.loads(json_str)


def parse_iso(dt_str: str) -> datetime:
    """
    Parse ISO8601 strings, handling 'Z' as UTC.
    Returns timezone-aware datetime when possible.
    """
    if not dt_str:
        return None
    s = dt_str.strip()
    # Normalize Z → +00:00 for fromisoformat
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # Last resort: try without microseconds, etc.
        try:
            return datetime.strptime(dt_str.replace("Z", "+00:00"), "%Y-%m-%dT%H:%M:%S%z")
        except Exception:
            return None


def lonlat_to_web_mercator(lon: float, lat: float):
    """
    Convert WGS84 lon/lat to Web Mercator (EPSG:3857).
    Returns (x, y)
    """
    # Clamp latitude to mercator valid range
    lat = max(min(lat, 85.05112878), -85.05112878)
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
    return x, y


def first(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


# ─────────────────────────────
# Events memory / config
# ─────────────────────────────
message_history = defaultdict(lambda: deque(maxlen=10))
MAX_CONTEXT_TOKENS = 3000
MAX_COMPLETION_TOKENS = 800


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


EVENTS = []
EVENTS_PATH = os.path.join(os.getcwd(), "events.json")
if os.path.exists(EVENTS_PATH):
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            EVENTS = json.load(f)
    except Exception:
        EVENTS = []

# +++ START: Location lookup (user-provided) +++
# If a 'locations.json' file is present, use it to override/extend LOCATION_LOOKUP.
# Expected shape: { "Kolligian Library": {"lat": 37.3653, "lon": -120.4243}, ... }
LOCATION_LOOKUP = {
    "UC Merced": {"lat": 37.3660, "lon": -120.4246},
    "Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
    "Leo and Dottie Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
    "Arts and Computational Sciences (ACS)": {"lat": 37.3670, "lon": -120.4255},
    "Scholars Lane": {"lat": 37.366117, "lon": -120.424205},
}

LOCATIONS_PATH = os.path.join(os.getcwd(), "locations.json")
if os.path.exists(LOCATIONS_PATH):
    try:
        with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                # Shallow merge/override
                LOCATION_LOOKUP.update(loaded)
    except Exception:
        # If bad file, just ignore and keep defaults
        pass
# +++ END: Location lookup (user-provided) +++

# +++ START: New Local Events Integration +++


def fetch_and_process_local_events(from_iso: str = None, to_iso: str = None):
    """
    Fetch events from LOCAL endpoint:
      http://localhost:7050/events?from=...&to=...

    Then normalize to the frontend schema:
      {
        id, event_name, description, date, startAt, endAt,
        locationTag, names, original, geometry, fromUser, url
      }

    Coordinates:
      - Prefer event.lat/lon (or latitude/longitude/lng) if present.
      - Else, try LOCATION_LOOKUP by 'location_name' or 'location' string.
      - If lon/lat found, output geometry in Web Mercator (wkid 3857).
    """
    # Defaults: 7-day window if not provided
    if not from_iso or not to_iso:
        start_date = datetime.now(timezone.utc).date()
        end_date = start_date + timedelta(days=7)
        from_iso = datetime.combine(start_date, datetime.min.time(
        ), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        to_iso = datetime.combine(end_date, datetime.max.time(
        ), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    base_url = "https://uc-merced-campus-event-api-backend.onrender.com/events"
    params = {"from": from_iso, "to": to_iso}

    resp = requests.get(base_url, params=params, timeout=12)
    resp.raise_for_status()

    raw = resp.json()
    # Accept either {"events":[...]} or just [...]
    events_list = raw.get("events") if isinstance(raw, dict) else raw
    if not isinstance(events_list, list):
        events_list = []

    processed = []

    for e in events_list:
        # Basic fields (tolerant of different keys)
        eid = str(first(e.get("id"), e.get("_id"), e.get("uuid"), default=""))
        title = first(e.get("title"), e.get("name"),
                      e.get("event_name"), default="")
        description = first(e.get("description"), e.get("desc"), default=None)
        url = first(e.get("url"), e.get("link"), default=None)
        location_name = first(e.get("location_name"), e.get(
            "location"), e.get("venue"), default=None)
        names = e.get("names") if isinstance(e.get("names"), list) else None
        from_user = bool(e.get("fromUser", False))

        # Start/End
        start_raw = first(e.get("start"), e.get("startAt"),
                          e.get("start_time"), default=None)
        end_raw = first(e.get("end"), e.get("endAt"),
                        e.get("end_time"), default=None)

        start_dt = parse_iso(start_raw) if isinstance(start_raw, str) else None
        end_dt = parse_iso(end_raw) if isinstance(end_raw, str) else None

        if not start_dt:
            # Skip events without a valid start time
            continue

        # Coordinates: prefer direct numeric fields, else lookup by location string
        lon = first(e.get("lon"), e.get("lng"), e.get("longitude"))
        lat = first(e.get("lat"), e.get("latitude"))

        if (lon is None or lat is None) and location_name and location_name in LOCATION_LOOKUP:
            coords = LOCATION_LOOKUP[location_name]
            lat = first(coords.get("lat"))
            lon = first(coords.get("lon"))

        geometry = None
        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
            x, y = lonlat_to_web_mercator(float(lon), float(lat))
            geometry = {
                "type": "point",
                "x": float(x),
                "y": float(y),
                "spatialReference": {"wkid": 3857},
            }

        processed.append({
            "id": eid,
            "event_name": title,
            "description": description,
            "date": (start_dt.date().isoformat() if start_dt else None),
            "startAt": (start_dt.strftime("%H:%M") if start_dt else None),
            "endAt": (end_dt.strftime("%H:%M") if end_dt else None),
            "locationTag": location_name,
            "names": names,
            "original": e,
            "geometry": geometry,
            "fromUser": from_user,
            "url": url,
        })

    return processed
# +++ END: New Local Events Integration +++

# ─────────────────────────────
# Routes
# ─────────────────────────────


@app.route("/", methods=["GET"])
def root():
    # Updated to show the new endpoint
    return jsonify({
        "ok": True,
        "endpoints": ["/ask (POST)", "/ask/events (POST)", "/events (GET)"]
    })

# +++ Replaced: /events (GET) proxies to your local service +++


@app.route("/events", methods=["GET"])
def get_events():
    """
    Proxy-normalizer for local events service at:
      http://localhost:7050/events?from=...&to=...

    Query params (optional):
      - from: ISO8601 string (e.g., 2025-08-31T07:00:00.000Z)
      - to:   ISO8601 string (e.g., 2025-10-08T06:59:59.999Z)
    """
    try:
        from_iso = request.args.get("from")
        to_iso = request.args.get("to")
        events = fetch_and_process_local_events(from_iso, to_iso)
        return jsonify(events)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch from local events service: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


@app.route("/ask", methods=["POST"])
def ask_vision():
    # ... (your existing /ask route code is unchanged) ...
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
    # ... (your existing /ask/events route code is unchanged) ...
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


# ─────────────────────────────
# Entrypoint
# ─────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(host="0.0.0.0", port=port)

# from openai import OpenAI
# import os
# import re
# import base64
# import json
# from collections import defaultdict, deque
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv

# # +++ New Imports +++
# import requests
# from datetime import datetime, date, timedelta

# # ─────────────────────────────
# # Setup
# # ─────────────────────────────
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# app = Flask(__name__)
# CORS(app)

# # ─────────────────────────────
# # Shared helpers
# # ─────────────────────────────


# def extract_json(raw: str) -> dict:
#     """
#     Clean up model output and return the first {...} JSON object inside.
#     Raises ValueError if no valid JSON is found.
#     """
#     # Strip fenced code blocks if present
#     if raw.startswith("```") and raw.endswith("```"):
#         raw = raw.strip("```").strip()
#     start = raw.find("{")
#     if start == -1:
#         raise ValueError("No JSON object found in model output")
#     depth = 0
#     end = None
#     for i, ch in enumerate(raw[start:], start):
#         if ch == "{":
#             depth += 1
#         elif ch == "}":
#             depth -= 1
#             if depth == 0:
#                 end = i
#                 break
#     if end is None:
#         raise ValueError("Unbalanced braces in model output")
#     json_str = raw[start: end + 1]
#     return json.loads(json_str)


# # ─────────────────────────────
# # Events memory / config
# # ─────────────────────────────
# message_history = defaultdict(lambda: deque(maxlen=10))
# MAX_CONTEXT_TOKENS = 3000
# MAX_COMPLETION_TOKENS = 800


# def approximate_token_count(messages):
#     total = 0
#     for msg in messages:
#         content = msg.get("content", "")
#         if not isinstance(content, str):
#             try:
#                 content = json.dumps(content)
#             except Exception:
#                 content = str(content)
#         total += len(content) // 4
#     return total


# EVENTS = []
# EVENTS_PATH = os.path.join(os.getcwd(), "events.json")
# if os.path.exists(EVENTS_PATH):
#     try:
#         with open(EVENTS_PATH, "r", encoding="utf-8") as f:
#             EVENTS = json.load(f)
#     except Exception:
#         EVENTS = []

# # +++ START: New UC Merced Integration +++

# # ─────────────────────────────
# # UC Merced Event Configuration
# # ─────────────────────────────

# # Hardcoded lookup table for event locations.
# # The key should match the `location_name` from the API response.
# # Populate this with known campus venues.
# LOCATION_LOOKUP = {
#     "UC Merced": {"lat": 37.3660, "lon": -120.4246},
#     "Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
#     "Leo and Dottie Kolligian Library": {"lat": 37.3653, "lon": -120.4243},
#     "Arts and Computational Sciences (ACS)": {"lat": 37.3670, "lon": -120.4255},
#     "Scholars Lane": {"lat": 37.366117, "lon": -120.424205},
#     # Add more locations here...
#     # "Some Other Building": {"lat": 37.xxxx, "lon": -120.xxxx},
# }


# def fetch_and_process_uc_merced_events():
#     """
#     Fetches events from the UC Merced API, processes them, and formats them
#     for the frontend.
#     """
#     start_date = date.today()
#     end_date = start_date + timedelta(days=7)
#     start_str = start_date.strftime("%Y-%m-%d")
#     end_str = end_date.strftime("%Y-%m-%d")

#     cal_url = f"[https://events.ucmerced.edu/api/2/events?start=](https://events.ucmerced.edu/api/2/events?start=){start_str}&end={end_str}&pp=100"

#     response = requests.get(cal_url, timeout=10)
#     response.raise_for_status()
#     data = response.json()

#     processed_events = []

#     for event_wrapper in data.get("events", []):
#         event = event_wrapper.get("event")
#         if not event:
#             continue

#         place_lat, place_lon = None, None
#         location_name = event.get("location_name")

#         if location_name and location_name in LOCATION_LOOKUP:
#             coords = LOCATION_LOOKUP[location_name]
#             place_lat, place_lon = coords.get("lat"), coords.get("lon")

#         if place_lat is None or place_lon is None:
#             geo = event.get("geo", {}) or {}
#             place_lat = geo.get("latitude")
#             place_lon = geo.get("longitude")

#         inst_list = event.get("event_instances", [])
#         if not inst_list:
#             continue

#         inst = inst_list[0].get("event_instance", {})
#         start_iso = inst.get("start")
#         end_iso = inst.get("end")

#         if not start_iso:
#             continue

#         try:
#             start_dt = datetime.fromisoformat(start_iso)
#             end_dt = datetime.fromisoformat(end_iso) if end_iso else None
#         except (ValueError, TypeError):
#             continue

#         campus_event = {
#             "id": str(event.get("id")),
#             "event_name": event.get("title"),
#             "description": event.get("description_text") or None,
#             "date": start_dt.strftime("%Y-%m-%d"),
#             "startAt": start_dt.strftime("%H:%M"),
#             "endAt": end_dt.strftime("%H:%M") if end_dt else None,
#             "locationTag": location_name,
#             "names": None,
#             "original": event,
#             "geometry": None,
#             "fromUser": False,
#             "url": event.get("localist_url")
#         }

#         if place_lat is not None and place_lon is not None:
#             campus_event["geometry"] = {
#                 "x": float(place_lon),
#                 "y": float(place_lat),
#                 "wkid": 4326
#             }

#         processed_events.append(campus_event)

#     return processed_events

# # +++ END: New UC Merced Integration +++


# # ─────────────────────────────
# # Routes
# # ─────────────────────────────
# @app.route("/", methods=["GET"])
# def root():
#     # Updated to show the new endpoint
#     return jsonify({"ok": True, "endpoints": ["/ask (POST)", "/ask/events (POST)", "/events/ucmerced (GET)"]})


# # +++ New Route for UC Merced Events +++
# @app.route("/events/ucmerced", methods=["GET"])
# def get_uc_merced_events():
#     """
#     Provides a list of upcoming events from the official UC Merced calendar,
#     formatted for the campus map application.
#     """
#     try:
#         events = fetch_and_process_uc_merced_events()
#         return jsonify(events)
#     except requests.exceptions.RequestException as e:
#         # Handle network errors, timeouts, etc.
#         return jsonify({"error": f"Failed to fetch events from source: {e}"}), 502
#     except Exception as e:
#         # Handle other unexpected errors during processing
#         return jsonify({"error": f"An internal error occurred: {e}"}), 500


# @app.route("/ask", methods=["POST"])
# def ask_vision():
#     # ... (your existing /ask route code is unchanged) ...
#     if "file" not in request.files:
#         return jsonify({"error": "No image file provided (field name should be 'file')"}), 400

#     uploaded = request.files["file"]
#     img_bytes = uploaded.read()
#     if not img_bytes:
#         return jsonify({"error": "Empty file"}), 400

#     mime = uploaded.mimetype or "image/png"
#     b64 = base64.b64encode(img_bytes).decode("utf-8")
#     data_url = f"data:{mime};base64,{b64}"

#     system_message = {
#         "role": "system",
#         "content": (
#             "You are a vision-enabled assistant. "
#             "Extract from the image: date, time, location, names, event name, and a short description of the event. "
#             "Respond *only* with valid JSON matching this schema:\n\n"
#             "{\n"
#             "  \"date\": \"\",\n"
#             "  \"time\": \"\",\n"
#             "  \"location\": \"\",\n"
#             "  \"names\": [],\n"
#             "  \"event_name\": \"\",\n"
#             "  \"description\": \"\"\n"
#             "}\n"
#             "Rules:\n"
#             "- If a field is unknown, use an empty string (or empty array for names).\n"
#             "- Do not add extra keys. Do not include explanations."
#         ),
#     }

#     user_message = {
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": data_url}},
#         ],
#     }

#     try:
#         resp = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[system_message, user_message],
#             temperature=0.0,
#             max_tokens=400,
#         )

#         raw = (resp.choices[0].message.content or "").strip()
#         if not raw:
#             return jsonify({"error": "Empty model response"}), 502

#         try:
#             result = extract_json(raw)
#         except ValueError:
#             # Return the raw text to help debugging the prompt/formatting
#             return jsonify({"error": "Failed to extract JSON", "raw_response": raw}), 500

#         # Success
#         return jsonify(result)

#     except json.JSONDecodeError:
#         return jsonify({"error": "Model did not return valid JSON"}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/ask/events", methods=["POST"])
# def ask_events():
#     # ... (your existing /ask/events route code is unchanged) ...
#     data = request.get_json(silent=True) or {}
#     user_id = data.get("user_id", "default")
#     question = data.get("question", "")
#     tags = data.get("tags", [])

#     if not isinstance(question, str) or not question.strip():
#         return jsonify({"error": "No question provided"}), 400

#     filtered_events = [
#         event for event in EVENTS
#         if not tags or any(tag in event.get("tags", []) for tag in tags)
#     ]

#     system_message = {
#         "role": "system",
#         "content": (
#             "You are a helpful assistant for UC Merced's Bobcat Day. "
#             "You help students find relevant events based on their interests. "
#             "When recommending events, include their IDs at the end in a JSON array like [\"event002\", \"event004\"]."
#         ),
#     }

#     history = message_history[user_id]
#     history.append({"role": "user", "content": question})

#     context_prompt = (
#         f"User asked: \"{question}\"\n\n"
#         f"Here is a list of events:\n{json.dumps(filtered_events, ensure_ascii=False)}"
#     )

#     messages = [system_message] + \
#         list(history) + [{"role": "user", "content": context_prompt}]

#     while approximate_token_count(messages) > MAX_CONTEXT_TOKENS and len(history) > 0:
#         history.popleft()
#         messages = [system_message] + \
#             list(history) + [{"role": "user", "content": context_prompt}]

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             temperature=0.4,
#             max_tokens=MAX_COMPLETION_TOKENS,
#         )

#         reply = (response.choices[0].message.content or "").strip()

#         history.append({"role": "assistant", "content": reply})

#         event_ids = re.findall(r"event\d{3}", reply)
#         matched_events = [
#             event for event in EVENTS if event.get("id") in set(event_ids)]

#         return jsonify({"response": reply, "matched_events": matched_events})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ─────────────────────────────
# # Entrypoint
# # ─────────────────────────────
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", "6050"))
#     app.run(host="0.0.0.0", port=port)
