# events.py
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, url_for
from werkzeug.utils import secure_filename
from collections import defaultdict, deque
import uuid  # 👈 for a unique event id

events_bp = Blueprint('events', __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@events_bp.route("/get/events", methods=["GET"])
def get_events():
    events_col = current_app.config["EVENTS_COL"]

    # "Now" in PST (we're assuming server is PST; otherwise see note below)
    now = datetime.now()

    docs = list(events_col.find().sort("start_dt", 1))

    events = []
    for d in docs:
        date_str = d.get("date")      # e.g., "2025-11-30"
        end_at_str = d.get("end_at")  # e.g., "21:00"

        # Default: assume event has ended if we can't parse date/end_at
        keep = False

        if date_str and end_at_str:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                end_time = datetime.strptime(end_at_str, "%H:%M").time()
                end_dt = datetime.combine(event_date, end_time)  # naive PST

                # Only keep events that have not ended yet
                if end_dt > now:
                    keep = True

            except ValueError:
                # Bad formatting -> treat as ended / drop from list
                keep = False

        if not keep:
            continue  # skip this event

        events.append({
            "id": d.get("id"),
            "_id": str(d["_id"]),
            "location_at": d.get("location_at"),
            "location": d.get("location"),
            "date": d.get("date"),  # stored as ISO string below
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
    date_str = request.form.get("date")        # e.g., "2025-11-30"
    start_at = request.form.get("start_at")    # e.g., "18:00"
    end_at = request.form.get("end_at")        # e.g., "21:00"
    host = request.form.get("host")
    title = request.form.get("title")
    description = request.form.get("description")

    # File comes from request.files
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

    # Optional: validate poster file type, but DON'T save locally
    if poster_file and not allowed_file(poster_file.filename):
        return jsonify({"error": "Invalid poster type. Allowed: png, jpg, jpeg, gif, webp"}), 400

    # 🚫 No local saving. Just placeholder URL for now.
    poster_path = None
    poster_url = "https://via.placeholder.com/600x800.png?text=Event+Poster"

    # Build document for MongoDB
    start_dt = datetime.combine(date, start_time)

    event_doc = {
        "id": str(uuid.uuid4()),         # satisfies unique index on "id"
        "location_at": location_at,
        "location": location,
        "date": date.isoformat(),        # store as ISO string
        "start_at": start_at,
        "end_at": end_at,
        "host": host,
        "title": title,
        "description": description,
        "poster_path": poster_path,
        "poster_url": poster_url,
        "start_dt": start_dt,            # real datetime for querying/sorting
        "created_at": datetime.utcnow(),  # audit field
    }

    # Insert into MongoDB
    events_col = current_app.config["EVENTS_COL"]
    result = events_col.insert_one(event_doc)

    # ✅ Build a JSON-safe copy (avoid ObjectId from event_doc["_id"])
    response_event = event_doc.copy()
    response_event["_id"] = str(result.inserted_id)
    response_event["start_dt"] = start_dt.isoformat()
    response_event["created_at"] = event_doc["created_at"].isoformat() + "Z"

    return jsonify({"message": "Event created", "event": response_event}), 201
