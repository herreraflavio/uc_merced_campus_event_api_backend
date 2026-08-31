from openai import OpenAI
import os
import re
import base64
import json
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from routes import (
    ask_bp,
    events_bp,
    create_autofill_ai_blueprint,
)

# ─────────────────────────────
# New: Mongo
# ─────────────────────────────
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

# ─────────────────────────────
# Setup
# ─────────────────────────────
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = Flask(__name__)
CORS(app)


# ─────────────────────────────
# Health Check
# ─────────────────────────────
@app.route("/health", methods=["GET"])
def health_check():
    """
    Basic health check endpoint.
    Returns a 200 OK status and a JSON payload confirming the service is up.
    """
    return jsonify({
        "status": "healthy",
        "message": "Service is up and running"
    }), 200


# ─────────────────────────────
# Blueprints / Routes
# ─────────────────────────────
app.register_blueprint(
    ask_bp,
    url_prefix="/"
)

app.register_blueprint(
    events_bp,
    url_prefix="/"
)

app.register_blueprint(
    create_autofill_ai_blueprint(client)
)


# ─────────────────────────────
# MongoDB
# ─────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is required")


mongo_client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)

try:
    # Trigger connection check early
    mongo_client.admin.command("ping")
except Exception as e:
    raise RuntimeError(
        f"Failed to connect to MongoDB: {e}"
    )


# If the URI includes '/campusmap', this returns that DB;
# otherwise fallback behavior is handled by PyMongo.
db = mongo_client.get_default_database()

events_col = db["events"]
meta_col = db["meta"]  # Optional for storing last refresh time, etc.


# ─────────────────────────────
# MongoDB Indexes
# ─────────────────────────────

# Unique event ID
events_col.create_index(
    [("id", ASCENDING)],
    unique=True,
    name="uniq_id"
)

# Event start time
events_col.create_index(
    [("start_dt", ASCENDING)],
    name="start_dt_idx"
)

# Duplicate-detection helper
events_col.create_index(
    [
        ("title_norm", ASCENDING),
        ("location_norm", ASCENDING),
        ("start_dt", ASCENDING),
    ],
    name="dupe_probe_idx",
)


# Allow blueprints to access the events collection
app.config["EVENTS_COL"] = events_col


# ─────────────────────────────
# Entrypoint
# ─────────────────────────────
if __name__ == "__main__":
    port = int(
        os.getenv("PORT", "8080")
    )

    app.run(
        host="0.0.0.0",
        port=port
    )