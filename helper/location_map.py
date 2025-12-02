import re
import os
import json

from werkzeug.utils import secure_filename

import logging

logger = logging.getLogger(__name__)

# Location data (your locations.json)
LOCATIONS_JSON_PATH = os.path.join(os.path.dirname(__file__), "locations.json")

LOCATION_MAP: dict[str, str] = {}


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
