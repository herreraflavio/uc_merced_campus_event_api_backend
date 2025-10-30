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

PACIFIC = ZoneInfo("America/Los_Angeles")
UCM_FEED = "https://events.ucmerced.edu/api/2/events"

# If set (minutes), a background thread will periodically refresh future UCM events.
REFRESH_MINUTES = int(os.getenv("EVENTS_REFRESH_MINUTES", "0"))  # 0 disables
REFRESH_LOOKAHEAD_DAYS = int(os.getenv("EVENTS_REFRESH_LOOKAHEAD_DAYS", "14"))

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
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.strip("`").strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth, end = 0, None
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
# Canonical event utils (Mongo "model")
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

    title_norm = _norm(title)
    location_norm = _norm(location or "")
    doc = {
        "id": eid,
        "title": title.strip(),
        "title_norm": title_norm,
        "description": (description or None),
        "start": start_iso,
        "end": end_iso,
        # native datetime for queries
        "start_dt": start.astimezone(timezone.utc),
        "end_dt": end.astimezone(timezone.utc) if end else None,
        "location": (location or None),
        "location_norm": location_norm,
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "fromUser": bool(from_user),
        "tags": list(tags or []),
        "source": source or {"name": None, "url": None, "event_id": None, "instance_id": None},
        "created_at": now,
        "updated_at": now,
    }
    return doc


def events_overlap_dt(a_start: datetime, b_start: datetime, minutes: int = 15) -> bool:
    return abs((a_start - b_start).total_seconds()) <= minutes * 60


def _serialize_event(doc: dict) -> dict:
    """Drop Mongo's _id and ensure JSON-friendly output."""
    if not doc:
        return doc
    d = dict(doc)
    d.pop("_id", None)

    # Ensure ISO fields exist (source of truth is start_dt / end_dt)
    if d.get("start_dt") and not d.get("start"):
        d["start"] = iso_z(d["start_dt"])
    if d.get("end_dt") and not d.get("end"):
        d["end"] = iso_z(d["end_dt"])

    # Remove the datetime objects since they're not JSON serializable
    d.pop("start_dt", None)
    d.pop("end_dt", None)

    # Keep created_at/updated_at as strings (already ISO)
    return d


def _merge_preferring_complete(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    # Prefer more complete description/location/coords
    for k in ["description", "location", "lat", "lon"]:
        if not merged.get(k) and incoming.get(k):
            merged[k] = incoming.get(k)
    # Keep earliest created_at, update 'updated_at'
    merged["updated_at"] = iso_z(datetime.now(timezone.utc))
    # If incoming has source, adopt it
    if incoming.get("source"):
        merged["source"] = incoming["source"]
    merged["fromUser"] = bool(merged.get("fromUser")
                              or incoming.get("fromUser", False))
    merged["tags"] = sorted(set(merged.get("tags", []))
                            | set(incoming.get("tags", [])))
    # Keep normalized/title/location values consistent
    for k in ["title", "title_norm", "location_norm", "start", "end", "start_dt", "end_dt"]:
        if incoming.get(k):
            merged[k] = incoming[k]
    return merged


def upsert_event(ev: dict) -> dict:
    """
    Insert or update an event in Mongo with de-duplication:
    1) Probe for fuzzy dupes by (title_norm, location_norm, start_dt in ±15m).
    2) If found, merge into existing document.
    3) Else upsert by deterministic 'id'.
    Returns the saved/merged document (serialized).
    """
    title_norm = ev.get("title_norm", _norm(ev.get("title", "")))
    location_norm = ev.get("location_norm", _norm(ev.get("location", "")))
    sdt: datetime | None = ev.get("start_dt") or parse_iso(ev.get("start"))
    if not sdt:
        raise ValueError("Event missing valid start datetime")

    # 1) Fuzzy duplicate probe
    window_start = sdt - timedelta(minutes=15)
    window_end = sdt + timedelta(minutes=15)
    dupe_query = {
        "title_norm": title_norm,
        "start_dt": {"$gte": window_start, "$lte": window_end},
    }
    if location_norm:
        dupe_query["location_norm"] = location_norm

    existing = events_col.find_one(dupe_query)
    if existing:
        merged = _merge_preferring_complete(existing, ev)
        events_col.update_one({"_id": existing["_id"]}, {"$set": merged})
        return _serialize_event(events_col.find_one({"_id": existing["_id"]}))

    # 2) Deterministic upsert by id
    eid = ev.get("id")
    if not eid:
        eid = canon_id_from(ev.get("title", ""),
                            iso_z(sdt), ev.get("location"))
        ev["id"] = eid

    now_iso = iso_z(datetime.now(timezone.utc))
    ev.setdefault("created_at", now_iso)
    ev["updated_at"] = now_iso

    # FIX: Separate created_at from the update set to avoid conflict
    update_doc = dict(ev)
    created_at = update_doc.pop("created_at", now_iso)

    events_col.update_one(
        {"id": eid},
        {"$setOnInsert": {"created_at": created_at}, "$set": update_doc},
        upsert=True,
    )
    return _serialize_event(events_col.find_one({"id": eid}))
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
        existing_id=None,
    )


def _set_last_refresh(ts_iso: str):
    meta_col.update_one({"_id": "last_refresh_ucm"}, {
                        "$set": {"value": ts_iso}}, upsert=True)


def _get_last_refresh() -> str | None:
    doc = meta_col.find_one({"_id": "last_refresh_ucm"})
    return (doc or {}).get("value")


def refresh_ucm_cache(start_local: date | None = None, end_local: date | None = None, per_page: int = 100) -> dict:
    """
    Pulls UCM feed for [start_local, end_local] inclusive-by-day in Pacific time,
    normalizes to canonical, upserts into Mongo, and updates LAST_REFRESH_UCM meta.
    """
    if start_local is None:
        start_local = today_pacific()
    if end_local is None:
        end_local = start_local + timedelta(days=REFRESH_LOOKAHEAD_DAYS)

    start_str = start_local.isoformat()
    end_str = end_local.isoformat()

    page = 1
    max_pages = 10
    normalized_count = 0

    try:
        while page <= max_pages:
            params = {"start": start_str, "end": end_str,
                      "pp": per_page, "page": page}
            r = requests.get(UCM_FEED, params=params, timeout=15)
            r.raise_for_status()
            payload = r.json() or {}
            page_events = payload.get("events") or []

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

        last_refresh = iso_z(datetime.now(timezone.utc))
        _set_last_refresh(last_refresh)
        return {"ok": True, "normalized_count": normalized_count, "last_refresh": last_refresh}

    except requests.RequestException as e:
        return {"ok": False, "error": str(e), "normalized_count": normalized_count}

# ─────────────────────────────
# Demo seeding (optional, skip if any events exist)
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


if events_col.estimated_document_count() == 0:
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
    return jsonify({"ok": True, "last_refresh_ucm": _get_last_refresh()})

# GET /events?from=ISO&to=ISO&pp=100&raw=1


@app.route("/events", methods=["GET"])
def get_events():
    """
    Serves from Mongo 'events' collection. Maintains existing query params:
    - 'from' inclusive; 'to' exclusive (ISO8601).
    - 'pp' ignored unless raw=1.
    - 'raw=1' returns stats + sample.
    If DB empty *and* a date window is provided, performs a one-time refresh for usability.
    """
    from_param = request.args.get("from")
    to_param = request.args.get("to")
    raw_toggle = request.args.get("raw") == "1"

    from_dt = parse_iso(from_param) if from_param else None
    to_dt = parse_iso(to_param) if to_param else None

    # One-time refresh if empty and window provided
    if events_col.estimated_document_count() == 0 and (from_dt or to_dt):
        start_local = (from_dt.astimezone(PACIFIC).date()
                       if from_dt else today_pacific())
        end_local = ((to_dt.astimezone(PACIFIC) - timedelta(microseconds=1)).date()
                     if to_dt else today_pacific() + timedelta(days=REFRESH_LOOKAHEAD_DAYS))
        refresh_ucm_cache(start_local, end_local)

    q = {}
    if from_dt and to_dt:
        q["start_dt"] = {"$gte": from_dt, "$lt": to_dt}
    elif from_dt:
        q["start_dt"] = {"$gte": from_dt}
    elif to_dt:
        q["start_dt"] = {"$lt": to_dt}

    docs = list(events_col.find(q).sort("start_dt", ASCENDING))
    events = [_serialize_event(d) for d in docs]

    if raw_toggle:
        return jsonify({
            "count": len(events),
            "last_refresh_ucm": _get_last_refresh(),
            "from": from_param, "to": to_param,
            "sample": events[:5],
        })

    resp = make_response(jsonify({"events": events}))
    resp.headers["Cache-Control"] = "no-store"
    return resp

# GET or POST /events/refresh?from=ISO&to=ISO&pp=100


@app.route("/events/refresh", methods=["GET", "POST"])
def events_refresh_now():
    """Manual trigger to refresh the DB from UCM feed (decoupled from GET /events)."""
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
    Accepts an image, extracts event info via GPT, normalizes to canonical,
    upserts into Mongo, and returns the created/updated event.
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

    q = {}
    if tags:
        q["tags"] = {"$in": tags}
    # Pull a bounded set if needed; for now pull all matching
    docs = list(events_col.find(q).sort("start_dt", ASCENDING))
    filtered_events = [_serialize_event(d) for d in docs]

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

        evt_ids = set(re.findall(r"evt_[a-f0-9]{6,32}", reply))
        matched = list(events_col.find({"id": {"$in": list(evt_ids)}}))
        matched_events = [_serialize_event(m) for m in matched]
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
