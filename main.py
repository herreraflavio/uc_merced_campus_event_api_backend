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
from typing import Dict, List, Tuple, Set, Optional
from urllib.parse import unquote_plus  # NEW

# ─────────────────────────────
# New: Mongo
# ─────────────────────────────
from pymongo import MongoClient, ASCENDING, DESCENDING
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

# IMPORTANT: tz_aware=True so datetimes are aware (UTC)
mongo_client = MongoClient(
    MONGODB_URI, serverSelectionTimeoutMS=5000, tz_aware=True)
try:
    mongo_client.admin.command("ping")
except Exception as e:
    raise RuntimeError(f"Failed to connect to MongoDB: {e}")

db = mongo_client.get_default_database()
events_col = db["events"]
meta_col = db["meta"]

# Indexes
events_col.create_index([("id", ASCENDING)], unique=True, name="uniq_id")
events_col.create_index([("start_dt", ASCENDING)], name="start_dt_idx")
events_col.create_index(
    [("title_norm", ASCENDING), ("location_norm", ASCENDING), ("start_dt", ASCENDING)],
    name="dupe_probe_idx",
)
events_col.create_index(
    [("description_norm", ASCENDING)], name="desc_norm_idx")

# ─────────────────────────────
# Categories & stopwords  (UPDATED)
# ─────────────────────────────
CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    # 1) Food & Drinks
    "food_drink": [
        "food", "snack", "snacks", "pizza", "refreshments", "beverage", "beverages",
        "drink", "drinks", "coffee", "tea", "boba", "milk tea", "soda", "juice",
        "cookies", "donuts", "catering", "dessert", "complimentary"
    ],
    # 2) Career & Jobs
    "career_jobs": [
        "career", "jobs", "job", "recruiting", "recruiter", "hiring", "internship", "intern",
        "resume", "cv", "cover letter", "networking", "employer", "career fair", "info session", "interview"
    ],
    # 3) Academics & Research
    "academics_research": [
        "academic", "academics", "seminar", "colloquium", "lecture", "talk", "panel",
        "research", "paper", "thesis", "dissertation", "poster", "workshop", "reading group"
    ],
    # 4) Sports & Fitness
    "sports_fitness": [
        "sports", "sport", "game", "match", "tournament", "intramural", "fitness", "workout",
        "run", "5k", "yoga", "rec", "recreation", "climbing", "hike", "training"
    ],
    # 5) Arts & Culture
    "arts_culture": [
        "arts", "art", "culture", "cultural", "concert", "performance", "dance", "theater",
        "film", "screening", "gallery", "exhibit", "museum", "poetry", "music", "symphony", "choir"
    ],
    # 6) Social & Community
    "social_community": [
        "mixer", "meetup", "social", "hangout", "club", "student org", "organization",
        "community", "volunteer", "volunteering", "service", "fundraiser", "block party"
    ],
    # 7) Tech & Engineering
    "tech_engineering": [
        "tech", "technology", "engineering", "coding", "programming", "developer", "dev",
        "hackathon", "data", "data science", "ai", "ml", "robotics", "cloud", "cybersecurity", "linux", "git"
    ],
    # 8) Wellness & Health
    "wellness_health": [
        "wellness", "health", "mental health", "mindfulness", "meditation", "counseling",
        "nutrition", "health fair", "vaccination", "flu shot", "stress relief", "therapy dogs", "yoga"
    ],
    # 9) Admin & Deadlines
    "admin_deadlines": [
        "registration", "deadline", "due", "drop", "add", "financial aid", "payment",
        "bursar", "application", "orientation", "advising", "office hours", "forms"
    ],
    # 10) Tours & Admissions
    "tours_admissions": [
        "tour", "campus tour", "open house", "admissions", "admitted students",
        "orientation", "info session", "preview day", "visit", "housing tour"
    ],
    # 11) Faith & Spirituality (NEW)
    "faith_spirituality": [
        "religion", "religious", "faith", "spiritual", "spirituality",
        "interfaith", "chaplain", "chaplaincy", "worship", "service",
        "prayer", "pray", "bible", "quran", "torah", "mass", "church",
        "mosque", "synagogue", "temple", "hillel", "fellowship", "ministry"
    ],
}

# Common English stopwords (short list, you can expand later)
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "is", "are", "be", "at", "by", "from", "as", "it", "this", "that",
    "these", "those", "where", "when", "how", "what", "who", "whom",
    "about", "into", "over", "under", "after", "before", "during",
    "up", "down", "out", "off", "than", "then"
}


def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


# Build expansion lookups
TERM_EXPANSIONS: Dict[str, Set[str]] = defaultdict(set)
PHRASE_EXPANSIONS: Dict[str, Set[str]] = defaultdict(set)
for cat, words in CATEGORY_SYNONYMS.items():
    norm_words = [w for w in {_norm(w) for w in words} if w]
    for w in norm_words:
        for other in norm_words:
            if other != w:
                if " " in other:
                    PHRASE_EXPANSIONS[w].add(other)
                else:
                    TERM_EXPANSIONS[w].add(other)

# ─────────────────────────────
# HARD-CODED location → (lat, lon)
# ─────────────────────────────
HARD_CODED_LOCATIONS = {
    "uc merced": (37.3690, -120.4209),
    "merced station": (37.3027, -120.4811),
    "scholars lane": (37.366117,  -120.424205),
}
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
# Canonical event utils
# ─────────────────────────────


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def parse_dateish_to_utc_start(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    dt = parse_iso(s)
    if dt:
        return dt
    try:
        y, m, d = [int(x) for x in s[:10].split("-")]
        pac = datetime(y, m, d, tzinfo=PACIFIC)
        return pac.astimezone(timezone.utc)
    except Exception:
        return None


def parse_dateish_to_utc_end_exclusive(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    dt = parse_iso(s)
    if dt:
        return dt
    try:
        y, m, d = [int(x) for x in s[:10].split("-")]
        pac = datetime(y, m, d, tzinfo=PACIFIC) + timedelta(days=1)
        return pac.astimezone(timezone.utc)
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
    desc_norm = _norm(description or "")
    doc = {
        "id": eid,
        "title": title.strip(),
        "title_norm": title_norm,
        "description": (description or None),
        "description_norm": desc_norm,
        "start": start_iso,
        "end": end_iso,
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
    if not doc:
        return doc
    d = dict(doc)
    d.pop("_id", None)
    if d.get("start_dt") and not d.get("start"):
        d["start"] = iso_z(d["start_dt"])
    if d.get("end_dt") and not d.get("end"):
        d["end"] = iso_z(d["end_dt"])
    d.pop("start_dt", None)
    d.pop("end_dt", None)
    d.pop("title_norm", None)
    d.pop("location_norm", None)
    d.pop("description_norm", None)
    return d


def _merge_preferring_complete(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for k in ["description", "location", "lat", "lon"]:
        if not merged.get(k) and incoming.get(k):
            merged[k] = incoming.get(k)
    merged["updated_at"] = iso_z(datetime.now(timezone.utc))
    if incoming.get("source"):
        merged["source"] = incoming["source"]
    merged["fromUser"] = bool(merged.get("fromUser")
                              or incoming.get("fromUser", False))
    merged["tags"] = sorted(set(merged.get("tags", []))
                            | set(incoming.get("tags", [])))
    for k in ["title", "start", "end", "start_dt", "end_dt", "location", "description"]:
        if incoming.get(k) is not None:
            merged[k] = incoming[k]
    merged["title_norm"] = _norm(merged.get("title", ""))
    merged["location_norm"] = _norm(merged.get("location") or "")
    merged["description_norm"] = _norm(merged.get("description") or "")
    return merged


# ─────────────────────────────
# In-memory cache + inverted index
# ─────────────────────────────
FIELD_ALIASES = {
    "title": "title",
    "description": "description",
    "descriptions": "description",
    "desc": "description",
    "location": "location",
    "loc": "location",
    "tags": "tags",
    "tag": "tags",
    "id": "id",
    "from_user": "fromUser",
    "fromuser": "fromUser",
    "fromUser": "fromUser",
    "source": "source",
    "source.name": "source_name",
    "source.url": "source_url",
    "lat": "lat",
    "lon": "lon",
    "lng": "lon",
    "start": "start",
    "end": "end",
}


def _to_bool(s: str) -> Optional[bool]:
    m = _norm(s)
    if m in {"true", "t", "1", "yes", "y"}:
        return True
    if m in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _as_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _start_key(eid: str) -> datetime:
    sdt = _as_aware_utc(CACHE.by_id[eid].get("start_dt"))
    return sdt or datetime.max.replace(tzinfo=timezone.utc)


class EventCache:
    def __init__(self):
        self.by_id: Dict[str, dict] = {}
        self.norm_fields: Dict[str, Dict[str, str]] = {}
        self.token_ix: Dict[str, Set[str]] = defaultdict(set)
        self.last_build: Optional[str] = None

    def _norm_fields_for_doc(self, d: dict) -> Dict[str, str]:
        tags_txt = " ".join(_norm(t) for t in (d.get("tags") or []))
        source = d.get("source") or {}
        f = {
            "id": d.get("id", ""),
            "title": _norm(d.get("title", "")),
            "description": _norm(d.get("description") or ""),
            "location": _norm(d.get("location") or ""),
            "tags": tags_txt,
            "source_name": _norm(source.get("name") or ""),
            "source_url": _norm(source.get("url") or ""),
            "fromUser": "true" if d.get("fromUser") else "false",
            "lat": _norm(str(d.get("lat") if d.get("lat") is not None else "")),
            "lon": _norm(str(d.get("lon") if d.get("lon") is not None else "")),
            "start": _norm(d.get("start") or ""),
            "end": _norm(d.get("end") or ""),
        }
        f["__all__"] = " ".join([f[k] for k in f.keys()])
        return f

    def _tokenize(self, text: str) -> Set[str]:
        tokens = set(text.split())
        more = set()
        for t in tokens:
            if t.endswith("ies") and len(t) > 3:
                more.add(t[:-3] + "y")
            elif t.endswith("s") and len(t) > 2:
                more.add(t[:-1])
        return tokens.union(more)

    def _index_doc(self, eid: str, f: Dict[str, str]):
        for tok in self._tokenize(f["__all__"]):
            self.token_ix[tok].add(eid)

    def _deindex_doc(self, eid: str):
        for tok, ids in list(self.token_ix.items()):
            if eid in ids:
                ids.discard(eid)
                if not ids:
                    self.token_ix.pop(tok, None)

    def rebuild(self):
        self.by_id.clear()
        self.norm_fields.clear()
        self.token_ix.clear()
        for d in events_col.find({}):
            eid = d.get("id")
            if not eid:
                continue
            d = dict(d)
            d["start_dt"] = _as_aware_utc(d.get("start_dt"))
            d["end_dt"] = _as_aware_utc(d.get("end_dt"))
            self.by_id[eid] = d
            f = self._norm_fields_for_doc(d)
            self.norm_fields[eid] = f
            self._index_doc(eid, f)
        self.last_build = iso_z(datetime.now(timezone.utc))

    def upsert(self, d: dict):
        eid = d.get("id")
        if not eid:
            return
        d = dict(d)
        d["start_dt"] = _as_aware_utc(d.get("start_dt"))
        d["end_dt"] = _as_aware_utc(d.get("end_dt"))
        self.by_id[eid] = d
        f = self._norm_fields_for_doc(d)
        self.norm_fields[eid] = f
        self._deindex_doc(eid)
        self._index_doc(eid, f)

    def delete(self, eid: str):
        if eid in self.by_id:
            self.by_id.pop(eid, None)
        if eid in self.norm_fields:
            self.norm_fields.pop(eid, None)
        self._deindex_doc(eid)

    def stats(self):
        return {
            "events_cached": len(self.by_id),
            "unique_tokens": len(self.token_ix),
            "last_build": self.last_build,
        }


CACHE = EventCache()

# ─────────────────────────────
# Upsert / DB I/O (updates cache)
# ─────────────────────────────


def upsert_event(ev: dict) -> dict:
    title_norm = ev.get("title_norm", _norm(ev.get("title", "")))
    location_norm = ev.get("location_norm", _norm(ev.get("location", "")))
    sdt: Optional[datetime] = ev.get("start_dt") or parse_iso(ev.get("start"))
    if not sdt:
        raise ValueError("Event missing valid start datetime")

    window_start = sdt - timedelta(minutes=15)
    window_end = sdt + timedelta(minutes=15)
    dupe_query = {"title_norm": title_norm, "start_dt": {
        "$gte": window_start, "$lte": window_end}}
    if location_norm:
        dupe_query["location_norm"] = location_norm

    existing = events_col.find_one(dupe_query)
    if existing:
        merged = _merge_preferring_complete(existing, ev)
        events_col.update_one({"_id": existing["_id"]}, {"$set": merged})
        saved = events_col.find_one({"_id": existing["_id"]})
        CACHE.upsert(saved)
        return _serialize_event(saved)

    eid = ev.get("id")
    if not eid:
        eid = canon_id_from(ev.get("title", ""),
                            iso_z(sdt), ev.get("location"))
        ev["id"] = eid

    now_iso = iso_z(datetime.now(timezone.utc))
    ev.setdefault("created_at", now_iso)
    ev["updated_at"] = now_iso

    ev["title_norm"] = _norm(ev.get("title", ""))
    ev["location_norm"] = _norm(ev.get("location") or "")
    ev["description_norm"] = _norm(ev.get("description") or "")

    update_doc = dict(ev)
    created_at = update_doc.pop("created_at", now_iso)

    events_col.update_one(
        {"id": eid},
        {"$setOnInsert": {"created_at": created_at}, "$set": update_doc},
        upsert=True,
    )
    saved = events_col.find_one({"id": eid})
    CACHE.upsert(saved)
    return _serialize_event(saved)

# ─────────────────────────────
# UCM feed normalization & refresh
# ─────────────────────────────


def normalize_ucm_instance(ev: dict, inst: dict) -> Optional[dict]:
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


def _get_last_refresh() -> Optional[str]:
    doc = meta_col.find_one({"_id": "last_refresh_ucm"})
    return (doc or {}).get("value")


def refresh_ucm_cache(start_local: Optional[date] = None, end_local: Optional[date] = None, per_page: int = 100) -> dict:
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
# Demo seeding (optional)
# ─────────────────────────────


def build_demo_events() -> list[dict]:
    now = datetime.now(timezone.utc)
    hour = timedelta(hours=1)
    day = timedelta(days=1)
    demo = [
        make_canonical_event(
            title="Campus Tour",
            description="Guided tour for prospective students with refreshments.",
            start=now + 2 * hour, end=now + 3 * hour,
            location="UC Merced", lat=37.3656, lon=-120.425, from_user=False,
            tags=["demo", "tour"], source={"name": "demo", "url": None, "event_id": None, "instance_id": None},
            existing_id="evt_demo_1",
        ),
        make_canonical_event(
            title="Biology Seminar",
            description="Guest lecture on marine ecosystems. Free coffee & snacks.",
            start=now - 1 * day, end=now - 1 * day + 2 * hour,
            location="COB 1", lat=37.3637, lon=-120.4245, from_user=False,
            tags=["demo", "academics"], source={"name": "demo", "url": None, "event_id": None, "instance_id": None},
            existing_id="evt_demo_2",
        ),
    ]
    return demo


if events_col.estimated_document_count() == 0:
    for d in build_demo_events():
        upsert_event(d)

CACHE.rebuild()

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
# Search parsing helpers  (UPDATED)
# ─────────────────────────────


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _split_csv_allow_quotes(s: str) -> List[str]:
    if not s:
        return []
    parts = re.findall(r'"([^"]+)"|([^,\s]+)', s)
    out = []
    for a, b in parts:
        out.append(a if a else b)
    return [x.strip() for x in out if x and x.strip()]


def _parse_options_string(opts: str) -> dict:
    """
    options:"location:cob2,time:<start>-<end>,tags:free,career,from_user:true"
    """
    opts = (opts or "").strip()
    if not opts:
        return {}
    chunks = []
    i = 0
    cur = []
    in_q = False
    while i < len(opts):
        ch = opts[i]
        if ch == '"':
            in_q = not in_q
            cur.append(ch)
        elif ch == ',' and not in_q:
            chunks.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        chunks.append(''.join(cur).strip())
    out = {}
    for c in chunks:
        if not c or ':' not in c:
            continue
        k, v = c.split(':', 1)
        k = k.strip().lower()
        v = _strip_quotes(v.strip())
        out[k] = v
    return out


def _expand_terms_for_value(value_norm: str) -> Tuple[Set[str], Set[str]]:
    tokens = [t for t in value_norm.split() if t]
    tset: Set[str] = set()
    pset: Set[str] = set()
    for t in tokens:
        tset |= TERM_EXPANSIONS.get(t, set())
        pset |= PHRASE_EXPANSIONS.get(t, set())
    return tset, pset


def _harvest_q_colon_style(field_terms: dict, any_terms: List[str]):
    """
    Supports colon-in-key:
      qdescription:"free food"
      q:"boba"
    """
    raw_qs = request.query_string.decode("utf-8", errors="ignore")
    # q<field>:"..."; field group 1
    pat_field = re.compile(
        r'(?:^|[?&])q([A-Za-z0-9_.]+):(?:"([^"]*)"|\'([^\']*)\'|([^&]+))')
    for m in pat_field.finditer(raw_qs):
        field_key = m.group(1)
        val = m.group(2) or m.group(3) or m.group(4) or ""
        val = unquote_plus(val).strip()
        if not val:
            continue
        canon_field = FIELD_ALIASES.get(field_key.lower(), field_key)
        v_norm = _norm(_strip_quotes(val))
        if not v_norm:
            continue
        tset, pset = _expand_terms_for_value(v_norm)
        values_for_field = {v_norm} | tset | pset
        field_terms[canon_field].extend(sorted(values_for_field))

    # q:"..."; no field specified → general
    pat_any = re.compile(r'(?:^|[?&])q:(?:"([^"]*)"|\'([^\']*)\'|([^&]+))')
    for m in pat_any.finditer(raw_qs):
        val = m.group(1) or m.group(2) or m.group(3) or ""
        val = unquote_plus(val).strip()
        if not val:
            continue
        any_terms.append(_strip_quotes(val))


def _stopword_filter_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t and t not in STOPWORDS]


def _parse_q_fields_and_options(args) -> Tuple[Dict[str, List[str]], dict]:
    """
    Populates:
      - field_terms: dict[field] -> list of normalized values (OR per field)
        Also special field "__any__" for general q searches.
      - options_map: includes:
          * time range (from/to) under 'time_from'/'time_to' (UTC datetimes or None)
          * location filters 'loc_filters' (normalized list)
          * from_user: optional bool
          * tags merged into field_terms["tags"]
          * __any_mode: "any_of" (category) or "all_of" (default)
          * __any_fields: list of fields considered for general search
    """
    field_terms: Dict[str, List[str]] = defaultdict(list)
    options_map: dict = {}

    # A) Field-specific (standard form): ?qdescription="..."
    for key in request.args.keys():
        if not key.lower().startswith("q") or key.lower() == "q":
            continue
        raw_vals = request.args.getlist(key)
        field_key = key[1:]  # strip leading 'q'
        canon_field = FIELD_ALIASES.get(field_key.lower(), field_key)
        for rv in raw_vals:
            v = _strip_quotes(rv)
            if not v:
                continue
            v_norm = _norm(v)
            tset, pset = _expand_terms_for_value(v_norm)
            values_for_field = {v_norm} | tset | pset
            field_terms[canon_field].extend(sorted(values_for_field))

    # B) Harvest colon-in-key variants (including q:"...")
    any_raws: List[str] = []
    _harvest_q_colon_style(field_terms, any_raws)

    # C) General 'q' param (standard form) — may be multiple
    any_raws.extend(request.args.getlist("q"))

    # Build __any__ from general q values
    any_terms_accum: List[str] = []
    category_hit = False
    for raw_q in any_raws:
        val = _strip_quotes(unquote_plus(raw_q or "").strip())
        if not val:
            continue
        val_norm = _norm(val)
        # split into tokens and remove stopwords
        tok_list = _stopword_filter_tokens(val_norm.split())
        if not tok_list and val_norm:
            tok_list = [val_norm]
        # category detection: if any token has expansions defined
        has_cat = any(
            t in TERM_EXPANSIONS or t in PHRASE_EXPANSIONS for t in tok_list)
        category_hit = category_hit or has_cat
        # expansions only for non-stopwords tokens
        expanded: Set[str] = set(tok_list)
        for t in tok_list:
            expanded |= TERM_EXPANSIONS.get(t, set())
            expanded |= PHRASE_EXPANSIONS.get(t, set())
        # keep full phrase as well (e.g., "free food")
        if val_norm:
            expanded.add(val_norm)
        any_terms_accum.extend(sorted(expanded))

    if any_terms_accum:
        field_terms["__any__"] = any_terms_accum
        options_map["__any_mode"] = "any_of" if category_hit else "all_of"

    # D) Top-level filters (canonical), still support options:"..."
    # time range
    from_param = request.args.get("from")
    to_param = request.args.get("to")
    date_from = parse_dateish_to_utc_start(from_param) if from_param else None
    date_to = parse_dateish_to_utc_end_exclusive(
        to_param) if to_param else None

    # location(s)
    loc_param = request.args.get("location", "")
    loc_filters = [_norm(x) for x in _split_csv_allow_quotes(loc_param)]

    # tags (any-of)
    tags_param = request.args.get("tags", "")
    if tags_param:
        field_terms["tags"].extend(
            [_norm(x) for x in _split_csv_allow_quotes(tags_param)])

    # from_user flag
    from_user: Optional[bool] = None
    if "from_user" in request.args:
        from_user = _to_bool(request.args.get("from_user", ""))

    # fields restriction for __any__ searches
    fields_param = request.args.get("fields", "")
    if fields_param:
        fields = [FIELD_ALIASES.get(x.strip().lower(), x.strip())
                  for x in _split_csv_allow_quotes(fields_param)]
        # keep only known
        valid = {"title", "description", "location",
                 "tags", "source_name", "source_url"}
        fields = [f for f in fields if f in valid]
    else:
        fields = ["title", "description", "location",
                  "tags", "source_name", "source_url"]
    options_map["__any_fields"] = fields

    # Merge legacy options:"..."
    options_raw_joined = ",".join(request.args.getlist(
        "options")) or ",".join(request.args.getlist('options"'))
    if options_raw_joined:
        o = _parse_options_string(_strip_quotes(options_raw_joined))
        if "time" in o and not (date_from or date_to):
            tr = o["time"]
            parts = re.split(r'\s*(?:-|\sto\s)\s*', tr, maxsplit=1)
            start_s = parts[0] if parts and parts[0] else ""
            end_s = parts[1] if len(parts) > 1 else ""
            date_from = parse_dateish_to_utc_start(
                start_s) if start_s else date_from
            date_to = parse_dateish_to_utc_end_exclusive(
                end_s) if end_s else date_to
        if "location" in o and not loc_filters:
            loc_filters = [_norm(x) for x in _split_csv_allow_quotes(
                o["location"]) or [o["location"]]]
        if "tags" in o and not field_terms.get("tags"):
            field_terms["tags"].extend(
                [_norm(x) for x in _split_csv_allow_quotes(o["tags"])])
        if "from_user" in o and from_user is None:
            from_user = _to_bool(o["from_user"])

    options_map["time_from"] = date_from
    options_map["time_to"] = date_to
    options_map["loc_filters"] = loc_filters
    options_map["from_user"] = from_user
    return field_terms, options_map

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
            "/events/search (GET)",  # UPDATED
            "/events/<id> (GET)",
            "/events/<id> (DELETE)",
            "/events/<id> (PUT)",
            "/events/refresh (GET|POST)",
            "/cache/stats (GET)",
            "/cache/rebuild (POST)",
            "/ask (POST)",
            "/ask/events (POST)"
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "last_refresh_ucm": _get_last_refresh(), "cache": CACHE.stats()})


@app.route("/events", methods=["GET"])
def get_events():
    from_param = request.args.get("from")
    to_param = request.args.get("to")
    raw_toggle = request.args.get("raw") == "1"

    from_dt = parse_iso(from_param) if from_param else None
    to_dt = parse_iso(to_param) if to_param else None

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


@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    return jsonify({"ok": True, **CACHE.stats()})


@app.route("/cache/rebuild", methods=["POST"])
def cache_rebuild():
    CACHE.rebuild()
    return jsonify({"ok": True, **CACHE.stats()})

# ─────────────────────────────
# NEW/UPDATED: GET /events/search
# ─────────────────────────────


@app.route("/events/search", methods=["GET"])
def events_search():
    """
    Supported:
      - General:  ?q="free food"&from=YYYY-MM-DD&to=YYYY-MM-DD&location=kl,cob2&tags=career,free
      - Field:    ?qdescription:"religion"
      - Field:    ?qtitle="career fair"&options:"time:2025-09-01-2025-10-15,location:kl"
      - General (colon): ?q:"boba"

    Notes:
      * General q does stopword removal.
      * If the query maps to a known category, we do ANY-of across category terms.
      * Otherwise we do ALL-of across remaining tokens.
    """
    sort_mode = (request.args.get("sort") or "relevance").lower()
    limit = min(max(int(request.args.get("limit", "50")), 1), 200)
    offset = max(int(request.args.get("offset", "0")), 0)

    field_terms, opts = _parse_q_fields_and_options(request.args)

    date_from: Optional[datetime] = opts.get("time_from")
    date_to: Optional[datetime] = opts.get("time_to")
    loc_filters: List[str] = opts.get("loc_filters") or []
    from_user_flag: Optional[bool] = opts.get("from_user")
    any_mode: str = opts.get("__any_mode", "all_of")
    any_fields: List[str] = opts.get("__any_fields") or [
        "title", "description", "location", "tags", "source_name", "source_url"]

    # Candidate universe
    candidate_ids: Set[str] = set(CACHE.by_id.keys())

    # Time filter (only if provided)
    def date_ok(d: dict) -> bool:
        sdt = _as_aware_utc(d.get("start_dt"))
        if not isinstance(sdt, datetime):
            return False
        if date_from and sdt < date_from:
            return False
        if date_to and sdt >= date_to:
            return False
        return True

    if date_from or date_to:
        candidate_ids = {
            eid for eid in candidate_ids if date_ok(CACHE.by_id[eid])}

    # Location contains (options)
    if loc_filters:
        def loc_ok(eid: str) -> bool:
            blob = CACHE.norm_fields[eid].get("location", "")
            return all(loc in blob for loc in loc_filters)
        candidate_ids = {eid for eid in candidate_ids if loc_ok(eid)}

    # From user filter
    if from_user_flag is not None:
        cand = set()
        for eid in candidate_ids:
            if CACHE.norm_fields[eid].get("fromUser") == ("true" if from_user_flag else "false"):
                cand.add(eid)
        candidate_ids = cand

    # Tags (any-of) in field_terms
    if "tags" in field_terms and field_terms["tags"]:
        tags_vals = set(field_terms["tags"])
        cand = set()
        for eid in candidate_ids:
            if any(tv in CACHE.norm_fields[eid].get("tags", "") for tv in tags_vals):
                cand.add(eid)
        candidate_ids = cand

    # Field matching logic
    def contains_any_in_fields(eid: str, fields: List[str], values: List[str]) -> bool:
        nf = CACHE.norm_fields[eid]
        for v in values:
            if not v:
                continue
            for f in fields:
                if v in nf.get(f, ""):
                    return True
        return False

    def contains_all_in_fields(eid: str, fields: List[str], values: List[str]) -> bool:
        nf = CACHE.norm_fields[eid]
        for v in values:
            if not v:
                continue
            # This token must appear in at least one of the fields
            if not any(v in nf.get(f, "") for f in fields):
                return False
        return True

    def field_match(eid: str) -> bool:
        nf = CACHE.norm_fields[eid]

        # General ANY/ALL search across selected fields
        if "__any__" in field_terms and field_terms["__any__"]:
            # dedupe, keep order
            vals = list(dict.fromkeys(field_terms["__any__"]))
            if any_mode == "any_of":
                if not contains_any_in_fields(eid, any_fields, vals):
                    return False
            else:
                if not contains_all_in_fields(eid, any_fields, vals):
                    return False

        # Specific fields (AND across fields, OR within a field)
        for field, vals in field_terms.items():
            if field in {"__any__", "tags"}:
                continue  # handled elsewhere
            if field == "source":
                src_blob = nf.get("source_name", "") + " " + \
                    nf.get("source_url", "")
                if not any(v in src_blob for v in vals):
                    return False
                continue
            blob = nf.get(field) or ""
            if not any(v in blob for v in vals):
                return False
        return True

    candidate_ids = {eid for eid in candidate_ids if field_match(eid)}

    # Scoring
    def score_eid(eid: str) -> float:
        d = CACHE.by_id[eid]
        nf = CACHE.norm_fields[eid]
        score = 0.0

        # Weight by matches in important fields
        def add_scores(vals: List[str], fields_weights: List[Tuple[str, float]]):
            nonlocal score
            for v in vals:
                if not v:
                    continue
                for f, w in fields_weights:
                    if v in nf.get(f, ""):
                        score += w

        # General q scoring
        if "__any__" in field_terms and field_terms["__any__"]:
            vals = list(dict.fromkeys(field_terms["__any__"]))
            # title > location > description > tags > source
            add_scores(vals, [("title", 3.0), ("location", 2.0), ("description", 1.4), ("tags", 1.0),
                              ("source_name", 0.6), ("source_url", 0.3)])

        # Field-specific scoring
        for field, vals in field_terms.items():
            if field in {"__any__", "tags"}:
                continue
            weights = [("title", 3.0), ("location", 2.0),
                       ("description", 1.2), ("tags", 1.0)]
            if field in {"title", "location", "description", "tags"}:
                weights = [(field, 3.2)]
            add_scores(vals, weights)

        # Exact equality nudge
        for field, vals in field_terms.items():
            if field == "__any__":
                for v in vals:
                    for f in any_fields:
                        if nf.get(f, "") == v:
                            score += 0.3
            elif field not in {"tags"}:
                for v in vals:
                    if nf.get(field, "") == v:
                        score += 0.3

        # Recency bump
        sdt = _as_aware_utc(d.get("start_dt"))
        if isinstance(sdt, datetime):
            now = datetime.now(timezone.utc)
            days = (sdt - now).total_seconds() / 86400.0
            if -1 <= days <= 7:
                score += 0.7
            elif 0 <= days <= 1:
                score += 0.3

        return score

    ids_list = list(candidate_ids)
    if sort_mode == "time":
        ids_list.sort(key=_start_key)
    else:
        ids_list.sort(key=lambda eid: (-score_eid(eid), _start_key(eid)))

    # Paging
    ids_page = ids_list[offset: offset + limit]
    events = [_serialize_event(CACHE.by_id[eid]) for eid in ids_page]

    return jsonify({
        "ok": True,
        "query": {
            "sort": sort_mode,
            "limit": limit,
            "offset": offset,
            "any_mode": any_mode if "__any__" in field_terms else None,
            "any_fields": any_fields if "__any__" in field_terms else None,
            "fields_used": {k: list(dict.fromkeys(v)) for k, v in field_terms.items()},
            "time_from": iso_z(date_from) if date_from else None,
            "time_to": iso_z(date_to) if date_to else None,
            "locations": loc_filters,
            "from_user": from_user_flag,
        },
        "count": len(events),
        "events": events
    })

# Single event


@app.route("/events/<event_id>", methods=["GET"])
def get_event(event_id):
    event = events_col.find_one({"id": event_id})
    if not event:
        return jsonify({"error": f"Event with id '{event_id}' not found"}), 404
    CACHE.upsert(event)
    return jsonify({"ok": True, "event": _serialize_event(event)})

# Delete event


@app.route("/events/<event_id>", methods=["DELETE"])
def delete_event(event_id):
    event = events_col.find_one({"id": event_id})
    if not event:
        return jsonify({"error": f"Event with id '{event_id}' not found"}), 404
    result = events_col.delete_one({"id": event_id})
    if result.deleted_count > 0:
        CACHE.delete(event_id)
        return jsonify({
            "ok": True,
            "message": f"Event '{event_id}' deleted successfully",
            "deleted_event": _serialize_event(event)
        })
    else:
        return jsonify({"error": "Failed to delete event"}), 500

# Update event


@app.route("/events/<event_id>", methods=["PUT"])
def update_event(event_id):
    existing_event = events_col.find_one({"id": event_id})
    if not existing_event:
        return jsonify({"error": f"Event with id '{event_id}' not found"}), 404

    data = request.get_json(silent=True) or {}
    update_doc = {}

    if "title" in data:
        title = str(data["title"]).strip()
        if not title:
            return jsonify({"error": "Title cannot be empty"}), 400
        update_doc["title"] = title
        update_doc["title_norm"] = _norm(title)

    if "description" in data:
        desc = data["description"]
        desc_val = str(desc).strip() if desc else None
        update_doc["description"] = desc_val
        update_doc["description_norm"] = _norm(desc_val or "")

    if "start" in data:
        start_dt = parse_iso(data["start"])
        if not start_dt:
            return jsonify({"error": "Invalid start datetime format. Use ISO8601"}), 400
        update_doc["start"] = iso_z(start_dt)
        update_doc["start_dt"] = start_dt.astimezone(timezone.utc)

    if "end" in data:
        if data["end"]:
            end_dt = parse_iso(data["end"])
            if not end_dt:
                return jsonify({"error": "Invalid end datetime format. Use ISO8601"}), 400
            update_doc["end"] = iso_z(end_dt)
            update_doc["end_dt"] = end_dt.astimezone(timezone.utc)
        else:
            update_doc["end"] = None
            update_doc["end_dt"] = None

    if "location" in data:
        location = data["location"]
        if location:
            location = str(location).strip()
            update_doc["location"] = location
            update_doc["location_norm"] = _norm(location)
            if "lat" not in data or "lon" not in data:
                lat, lon, _ = lookup_coords_hardcoded(location)
                if lat is not None and lon is not None:
                    update_doc["lat"] = float(lat)
                    update_doc["lon"] = float(lon)
        else:
            update_doc["location"] = None
            update_doc["location_norm"] = ""

    if "lat" in data:
        if data["lat"] is not None:
            try:
                update_doc["lat"] = float(data["lat"])
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid latitude value"}), 400
        else:
            update_doc["lat"] = None

    if "lon" in data:
        if data["lon"] is not None:
            try:
                update_doc["lon"] = float(data["lon"])
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid longitude value"}), 400
        else:
            update_doc["lon"] = None

    if "tags" in data:
        if isinstance(data["tags"], list):
            update_doc["tags"] = [str(tag).strip()
                                  for tag in data["tags"] if tag]
        else:
            return jsonify({"error": "Tags must be an array"}), 400

    if "fromUser" in data:
        update_doc["fromUser"] = bool(data["fromUser"])

    if "source" in data:
        if isinstance(data["source"], dict):
            update_doc["source"] = data["source"]
        else:
            return jsonify({"error": "Source must be an object"}), 400

    if not update_doc:
        return jsonify({"error": "No valid fields provided for update"}), 400

    update_doc["updated_at"] = iso_z(datetime.now(timezone.utc))

    result = events_col.update_one({"id": event_id}, {"$set": update_doc})
    if result.modified_count > 0 or result.matched_count > 0:
        updated_event = events_col.find_one({"id": event_id})
        CACHE.upsert(updated_event)
        return jsonify({
            "ok": True,
            "message": f"Event '{event_id}' updated successfully",
            "event": _serialize_event(updated_event)
        })
    else:
        return jsonify({"error": "Failed to update event"}), 500

# Manual refresh


@app.route("/events/refresh", methods=["GET", "POST"])
def events_refresh_now():
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
    CACHE.rebuild()
    code = 200 if result.get("ok") else 502
    return jsonify(result), code

# Vision


@app.route("/ask", methods=["POST"])
def ask_vision():
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
            "  \"start\": \"YYYY-MM-DDTHH:MM:SSZ\",\n"
            "  \"end\": \"YYYY-MM-DDTHH:MM:SSZ or empty string if unknown\",\n"
            "  \"location\": \"\",\n"
            "  \"names\": []\n"
            "}\n"
            "Rules:\n"
            "- If a field is unknown, use empty string (or empty array for names).\n"
            "- If only a date and a time window is visible, fill both start and end.\n"
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

# Q&A (unchanged)


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
# Background scheduler
# ─────────────────────────────


def _refresh_loop():
    while True:
        try:
            start_local = today_pacific()
            end_local = start_local + timedelta(days=REFRESH_LOOKAHEAD_DAYS)
            refresh_ucm_cache(start_local, end_local, per_page=100)
            CACHE.rebuild()
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
