from openai import OpenAI
import os
import re
import base64
import json
import hashlib
from collections import defaultdict, deque
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
import unicodedata
import requests
import threading
import time

# ─────────────────────────────
# Setup
# ─────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

PACIFIC = ZoneInfo("America/Los_Angeles")
UCM_FEED = "https://events.ucmerced.edu/api/2/events"

EVENTS_PATH = os.path.join(os.getcwd(), "events.json")
EVENTS_LOCK = threading.Lock()  # for safe writes from scheduler / requests

# If set (minutes), a background thread will periodically refresh future UCM events.
REFRESH_MINUTES = int(os.getenv("EVENTS_REFRESH_MINUTES", "0"))  # 0 disables
REFRESH_LOOKAHEAD_DAYS = int(os.getenv("EVENTS_REFRESH_LOOKAHEAD_DAYS", "14"))

# ─────────────────────────────
# Shared helpers
# ─────────────────────────────


def extract_json(raw: str) -> dict:
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
    return json.loads(json_str)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def strip_html(html: str | None) -> str | None:
    if not html:
        return None
    return re.sub(r"<[^>]+>", "", html).strip() or None


def hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def today_pacific() -> date:
    return datetime.now(PACIFIC).date()


# ─────────────────────────────
# HARD-CODED location → (lat, lon)
# ─────────────────────────────
HARD_CODED_LOCATIONS = {
    "uc merced": (37.3690, -120.4209),
    "merced station": (37.3027, -120.4811),
    "scholars lane": (37.366117,  -120.424205),
    # add more...
}


def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


_NORM_LOC = {_norm(k): v for k, v in HARD_CODED_LOCATIONS.items()}


def lookup_coords_hardcoded(*candidates: str):
    norm_cands = [(_norm(c or ""), (c or "")) for c in candidates if c]
    for nc, raw in norm_cands:
        if not nc:
            continue
        if nc in _NORM_LOC:
            lat, lon = _NORM_LOC[nc]
            return lat, lon, raw
    for nc, raw in norm_cands:
        if not nc:
            continue
        for key_norm, (lat, lon) in _NORM_LOC.items():
            if key_norm in nc or nc in key_norm:
                return lat, lon, raw
    return None, None, None

# ─────────────────────────────
# Persistent store
# ─────────────────────────────


def load_events_from_disk() -> list[dict]:
    if not os.path.exists(EVENTS_PATH):
        return []
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("events", [])
    except Exception:
        return []


def save_events_to_disk(events: list[dict]):
    tmp = EVENTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    os.replace(tmp, EVENTS_PATH)


# In-memory cache
EVENTS: list[dict] = load_events_from_disk()
LAST_REFRESH_UCM: str | None = None  # ISO8601Z of last UCM pull

# ─────────────────────────────
# Canonical event utils
# ─────────────────────────────


def canon_id_from(title: str, start_iso: str, location: str | None) -> str:
    base = f"{_norm(title)}|{start_iso}|{_norm(location or '')}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"evt_{h}"


def make_canonical_event(
    *,
    title: str,
    start: datetime,
    end: datetime | None,
    description: str | None,
    location: str | None,
    lat: float | None,
    lon: float | None,
    from_user: bool,
    tags: list[str] | None,
    source: dict | None,
    existing_id: str | None = None
) -> dict:
    now = iso_z(datetime.now(timezone.utc))
    start_iso = iso_z(start)
    end_iso = iso_z(end) if end else None
    eid = existing_id or canon_id_from(title, start_iso, location)
    return {
        "id": eid,
        "title": title.strip(),
        "description": (description or None),
        "start": start_iso,
        "end": end_iso,
        "location": (location or None),
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "fromUser": bool(from_user),
        "tags": list(tags or []),
        "source": source or {"name": None, "url": None, "event_id": None, "instance_id": None},
        "created_at": now,
        "updated_at": now,
    }


def events_overlap(a: dict, b: dict, minutes: int = 15) -> bool:
    """Simple time-window overlap used for duplicate checks."""
    sa = parse_iso(a.get("start"))
    sb = parse_iso(b.get("start"))
    if not sa or not sb:
        return False
    return abs((sa - sb).total_seconds()) <= minutes * 60


def is_probable_duplicate(a: dict, b: dict) -> bool:
    """Very lightweight duplicate heuristic (can be upgraded later)."""
    if _norm(a.get("title")) != _norm(b.get("title")):
        return False
    if a.get("location") and b.get("location"):
        if _norm(a["location"]) != _norm(b["location"]):
            return False
    # Start within ±15 min is strong dupe signal
    if not events_overlap(a, b, minutes=15):
        return False
    return True


def upsert_event(ev: dict):
    """Insert or update an event into EVENTS with simple de-duplication."""
    with EVENTS_LOCK:
        # Update if same ID or probable duplicate
        for i, existing in enumerate(EVENTS):
            if existing["id"] == ev["id"] or is_probable_duplicate(existing, ev):
                merged = existing.copy()
                # prefer more complete description/location/coords
                for k in ["description", "location", "lat", "lon"]:
                    if merged.get(k) in (None, "", []):
                        merged[k] = ev.get(k)
                # latest timestamps & source details
                merged["updated_at"] = iso_z(datetime.now(timezone.utc))
                if ev.get("source"):
                    merged["source"] = ev["source"]
                # keep earliest created_at
                merged["fromUser"] = merged.get(
                    "fromUser") or ev.get("fromUser", False)
                merged["tags"] = sorted(
                    set(merged.get("tags", [])) | set(ev.get("tags", [])))
                EVENTS[i] = merged
                save_events_to_disk(EVENTS)
                return merged
        # Insert new
        EVENTS.append(ev)
        save_events_to_disk(EVENTS)
        return ev

# ─────────────────────────────
# UCM feed normalization & refresh
# ─────────────────────────────


def normalize_ucm_instance(ev: dict, inst: dict) -> dict | None:
    title = (ev.get("title") or "").strip()
    if not title:
        return None
    desc_text = ev.get("description_text") or strip_html(ev.get("description"))
    loc_name = ev.get("location_name") or ev.get("location") or None
    room = ev.get("room_number")
    geo = ev.get("geo") or {}
    lat = geo.get("latitude")
    lon = geo.get("longitude")

    # If feed lacks coords, try hard-coded lookup (title and address as fallbacks).
    matched_key = None
    if lat is None or lon is None:
        lat, lon, matched_key = lookup_coords_hardcoded(
            loc_name, ev.get("location"), ev.get("address"), title
        )

    start_s = (inst or {}).get("start")
    end_s = (inst or {}).get("end")
    if not start_s:
        return None
    sdt = parse_iso(start_s)
    edt = parse_iso(end_s) if end_s else None
    if not sdt:
        return None

    source = {
        "name": "ucm",
        "url": ev.get("localist_url"),
        "event_id": ev.get("id"),
        "instance_id": inst.get("id"),
        "coord_source": (
            "feed" if (geo.get("latitude") is not None and geo.get("longitude") is not None)
            else ("lookup" if matched_key else None)
        ),
    }

    return make_canonical_event(
        title=title,
        description=desc_text,
        start=sdt,
        end=edt,
        location=(f"{loc_name} {room}".strip() if room else loc_name),
        lat=(float(lat) if lat is not None else None),
        lon=(float(lon) if lon is not None else None),
        from_user=False,
        tags=[],
        source=source,
        existing_id=None,  # stable id derived from title+start+location
    )


def refresh_ucm_cache(start_local: date | None = None, end_local: date | None = None, per_page: int = 100) -> dict:
    """
    Pulls UCM feed for [start_local, end_local] inclusive-by-day in Pacific time,
    normalizes to canonical, upserts into EVENTS, and updates LAST_REFRESH_UCM.
    """
    global LAST_REFRESH_UCM

    if start_local is None:
        start_local = today_pacific()
    if end_local is None:
        end_local = start_local + timedelta(days=REFRESH_LOOKAHEAD_DAYS)

    start_str = start_local.isoformat()
    end_str = end_local.isoformat()

    page = 1
    max_pages = 10
    normalized_count = 0
    raw_pages = []

    try:
        while page <= max_pages:
            params = {"start": start_str, "end": end_str,
                      "pp": per_page, "page": page}
            r = requests.get(UCM_FEED, params=params, timeout=15)
            r.raise_for_status()
            payload = r.json() or {}
            page_events = payload.get("events") or []
            raw_pages.append(payload)

            if not page_events:
                break

            for wrapper in page_events:
                ev = (wrapper or {}).get("event") or {}
                instances = ev.get("event_instances") or []
                for inst_wrap in instances:
                    inst = (inst_wrap or {}).get("event_instance") or {}
                    ce = normalize_ucm_instance(ev, inst)
                    if ce:
                        upsert_event(ce)
                        normalized_count += 1

            if len(page_events) < per_page:
                break
            page += 1

        LAST_REFRESH_UCM = iso_z(datetime.now(timezone.utc))
        return {"ok": True, "normalized_count": normalized_count, "last_refresh": LAST_REFRESH_UCM}

    except requests.RequestException as e:
        return {"ok": False, "error": str(e), "normalized_count": normalized_count}

# ─────────────────────────────
# Demo events (optional)
# ─────────────────────────────


def build_demo_events() -> list[dict]:
    now = datetime.now(timezone.utc)
    hour = timedelta(hours=1)
    day = timedelta(days=1)
    demo = [
        make_canonical_event(
            title="Campus Tour",
            description="Guided tour for prospective students.",
            start=now + 2 * hour, end=now + 3 * hour,
            location="UC Merced", lat=37.3656, lon=-120.425, from_user=False,
            tags=["demo"], source={"name": "demo", "url": None, "event_id": None, "instance_id": None},
            existing_id="evt_demo_1",
        ),
        make_canonical_event(
            title="Biology Seminar",
            description="Guest lecture on marine ecosystems.",
            start=now - 1 * day, end=now - 1 * day + 2 * hour,
            location="COB 1", lat=37.3637, lon=-120.4245, from_user=False,
            tags=["demo"], source={"name": "demo", "url": None, "event_id": None, "instance_id": None},
            existing_id="evt_demo_2",
        ),
    ]
    return demo


# On first boot, if file is empty, seed with demos (optional)
if not EVENTS:
    for d in build_demo_events():
        upsert_event(d)

# ─────────────────────────────
# Chat memory for /ask/events
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

# ─────────────────────────────
# Routes
# ─────────────────────────────


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "ok": True,
        "endpoints": [
            "/health (GET)",
            "/events (GET)",
            "/events/refresh (GET|POST)",
            "/ask (POST)",
            "/ask/events (POST)"
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "last_refresh_ucm": LAST_REFRESH_UCM})

# GET /events?from=ISO&to=ISO&pp=100&raw=1


@app.route("/events", methods=["GET"])
def get_events():
    """
    Serves from local cache (EVENTS). Maintains existing query params:
    - 'from' inclusive; 'to' exclusive (ISO8601).
    - 'pp' (page size) used only for UCM paging historically; here it's ignored unless raw=1.
    - 'raw=1' returns basic stats + (optional) future: debug info.
    If cache is empty *and* a date window is provided, it attempts a one-time refresh.
    """
    from_param = request.args.get("from")
    to_param = request.args.get("to")
    raw_toggle = request.args.get("raw") == "1"

    from_dt = parse_iso(from_param) if from_param else None
    to_dt = parse_iso(to_param) if to_param else None

    # If no cache and caller provided a window, do a one-time refresh for usability.
    if not EVENTS and (from_dt or to_dt):
        start_local = (from_dt.astimezone(PACIFIC).date()
                       if from_dt else today_pacific())
        # 'to' is exclusive; include previous microsecond to keep same semantics when day-bounding
        end_local = ((to_dt.astimezone(PACIFIC) - timedelta(microseconds=1)).date()
                     if to_dt else today_pacific() + timedelta(days=REFRESH_LOOKAHEAD_DAYS))
        refresh_ucm_cache(start_local, end_local)

    # Filter from cache
    def in_window(ev: dict) -> bool:
        s = parse_iso(ev.get("start"))
        if not s:
            return False
        if from_dt and s < from_dt:
            return False
        if to_dt and s >= to_dt:
            return False
        return True

    with EVENTS_LOCK:
        filtered = [ev for ev in EVENTS if in_window(ev)] if (
            from_dt or to_dt) else list(EVENTS)

    if raw_toggle:
        return jsonify({
            "count": len(filtered),
            "last_refresh_ucm": LAST_REFRESH_UCM,
            "from": from_param, "to": to_param,
            "sample": filtered[:5],
        })

    resp = make_response(jsonify({"events": filtered}))
    resp.headers["Cache-Control"] = "no-store"
    return resp

# GET or POST /events/refresh?from=ISO&to=ISO&pp=100


@app.route("/events/refresh", methods=["GET", "POST"])
def events_refresh_now():
    """Manual trigger to refresh the local cache from UCM feed (decoupled from GET /events)."""
    from_param = request.args.get("from") or (
        request.json.get("from") if request.is_json else None)
    to_param = request.args.get("to") or (
        request.json.get("to") if request.is_json else None)
    per_page = int(request.args.get("pp") or (
        request.json.get("pp") if request.is_json else 100) or 100)

    from_dt = parse_iso(from_param) if from_param else None
    to_dt = parse_iso(to_param) if to_param else None

    start_local = from_dt.astimezone(
        PACIFIC).date() if from_dt else today_pacific()
    end_local = ((to_dt.astimezone(PACIFIC) - timedelta(microseconds=1)).date()
                 if to_dt else (start_local + timedelta(days=REFRESH_LOOKAHEAD_DAYS)))

    result = refresh_ucm_cache(start_local, end_local, per_page)
    code = 200 if result.get("ok") else 502
    return jsonify(result), code


@app.route("/ask", methods=["POST"])
def ask_vision():
    """
    Accepts an image, extracts event info via GPT, normalizes to the canonical event,
    upserts into events.json, and returns the created/updated event.
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

    # Ask the model to emit canonical fields (title/description/start/end/location/names)
    system_message = {
        "role": "system",
        "content": (
            "You are a vision-enabled assistant that extracts event details. "
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            "  \"title\": \"\",\n"
            "  \"description\": \"\",\n"
            "  \"start\": \"YYYY-MM-DDTHH:MM:SSZ\",  // assume America/Los_Angeles if ambiguous\n"
            "  \"end\": \"YYYY-MM-DDTHH:MM:SSZ or empty string if unknown\",\n"
            "  \"location\": \"\",\n"
            "  \"names\": []\n"
            "}\n"
            "Rules:\n"
            "- If a field is unknown, use empty string (or empty array for names).\n"
            "- If only a date and a time window is visible (e.g., 7–9 PM Sep 20, 2025), fill both start and end.\n"
            "- Do not add extra keys. No explanations."
        ),
    }
    user_message = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}}]}
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

        title = (result.get("title") or "").strip()
        description = (result.get("description") or "").strip() or None
        start_iso = result.get("start") or ""
        end_iso = result.get("end") or ""
        location = (result.get("location") or "").strip() or None

        sdt = parse_iso(start_iso)
        edt = parse_iso(end_iso) if end_iso else None
        if not sdt:
            return jsonify({"error": "Model did not return a valid ISO8601 start"}), 400

        # coords (hardcoded lookup; you can swap with a geocoder later)
        lat, lon, _ = lookup_coords_hardcoded(location, title)

        canonical = make_canonical_event(
            title=title or "Untitled event",
            description=description,
            start=sdt,
            end=edt,
            location=location,
            lat=lat, lon=lon,
            from_user=True,
            tags=[],
            source={"name": "user", "url": None,
                    "event_id": None, "instance_id": None},
        )
        saved = upsert_event(canonical)
        return jsonify({"ok": True, "event": saved})
    except json.JSONDecodeError:
        return jsonify({"error": "Model did not return valid JSON"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask/events", methods=["POST"])
def ask_events():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "default")
    question = data.get("question", "")
    tags = data.get("tags", [])
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "No question provided"}), 400

    with EVENTS_LOCK:
        filtered_events = [e for e in EVENTS if not tags or any(
            tag in e.get("tags", []) for tag in tags)]

    system_message = {
        "role": "system",
        "content": (
            "You help students find relevant UC Merced events. "
            "When recommending events, include their IDs at the end in a JSON array like [\"evt_abc123\", \"evt_def456\"]."
        ),
    }
    history = message_history[user_id]
    history.append({"role": "user", "content": question})

    context_prompt = f'User asked: "{question}"\n\nHere is a list of events:\n{json.dumps(filtered_events, ensure_ascii=False)}'
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

        # Match evt_* ids
        evt_ids = set(re.findall(r"evt_[a-f0-9]{6,32}", reply))
        with EVENTS_LOCK:
            matched_events = [e for e in EVENTS if e.get("id") in evt_ids]
        return jsonify({"response": reply, "matched_events": matched_events})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────
# Background scheduler (optional)
# ─────────────────────────────


def _refresh_loop():
    """Runs periodic refresh of future events (today → +REFRESH_LOOKAHEAD_DAYS)."""
    while True:
        try:
            start_local = today_pacific()
            end_local = start_local + timedelta(days=REFRESH_LOOKAHEAD_DAYS)
            refresh_ucm_cache(start_local, end_local, per_page=100)
        except Exception as e:
            print(f"[scheduler] refresh error: {e}")
        # sleep
        minutes = max(REFRESH_MINUTES, 1)
        time.sleep(minutes * 60)


if REFRESH_MINUTES > 0:
    t = threading.Thread(target=_refresh_loop,
                         name="events-refresh", daemon=True)
    t.start()

# ─────────────────────────────
# Entrypoint
# ─────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "6050"))
    app.run(host="0.0.0.0", port=port)
