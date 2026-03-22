import os
import json
import logging

logger = logging.getLogger(__name__)

LOCATIONS_JSON_PATH = os.path.join(os.path.dirname(__file__), "locations.json")
WALKING_VERTICES_PATH = os.path.join(
    os.path.dirname(__file__), "walking_vertices.json")

# LOCATION_MAP now maps alias -> { "name": str, "id": int, "coordinates": list | None }
LOCATION_MAP: dict[str, dict] = {}


def load_location_map() -> None:
    """
    Load locations.json and walking_vertices.json, building a map of ALL aliases 
    to their canonical name, ID, and coordinates.
    """
    global LOCATION_MAP

    # 1. Load Locations
    try:
        with open(LOCATIONS_JSON_PATH, "r", encoding="utf-8") as f:
            locations = json.load(f)
    except OSError as e:
        logger.warning("Could not load locations.json from %s: %s",
                       LOCATIONS_JSON_PATH, e)
        LOCATION_MAP = {}
        return

    # 2. Load Coordinates (GeoJSON)
    coords_by_id = {}
    try:
        with open(WALKING_VERTICES_PATH, "r", encoding="utf-8") as f:
            vertices_data = json.load(f)
            for feature in vertices_data.get("features", []):
                f_id = feature.get("id")
                coords = feature.get("geometry", {}).get("coordinates")
                if f_id is not None and coords:
                    coords_by_id[f_id] = coords
    except OSError as e:
        logger.warning(
            "Could not load walking_vertices.json from %s: %s", WALKING_VERTICES_PATH, e)

    # 3. Build the combined dictionary
    new_map = {}
    for loc in locations:
        loc_id = loc.get("id")
        names = loc.get("name") or []
        if not names:
            continue

        canonical_name = names[0]
        coordinates = coords_by_id.get(loc_id)

        # Map every name in the list to the full data payload
        for alias in names:
            clean_alias = alias.lower().strip()
            new_map[clean_alias] = {
                "name": canonical_name,
                "id": loc_id,
                "coordinates": coordinates  # Format: [longitude, latitude]
            }

    LOCATION_MAP = new_map
    logger.info(
        "[locations] loaded %d location aliases with coordinates", len(LOCATION_MAP))


# Build location map at import time
load_location_map()
# import re
# import os
# import json

# from werkzeug.utils import secure_filename

# import logging

# logger = logging.getLogger(__name__)

# # Location data (your locations.json)
# LOCATIONS_JSON_PATH = os.path.join(os.path.dirname(__file__), "locations.json")

# LOCATION_MAP: dict[str, str] = {}


# def load_location_map() -> None:
#     """
#     Load locations.json and build a map of ALL aliases to the first (canonical) name.
#     """

#     global LOCATION_MAP

#     try:
#         with open(LOCATIONS_JSON_PATH, "r", encoding="utf-8") as f:
#             locations = json.load(f)
#     except OSError as e:
#         logger.warning("Could not load locations.json from %s: %s",
#                        LOCATIONS_JSON_PATH, e)
#         LOCATION_MAP = {}
#         return

#     new_map = {}

#     for loc in locations:
#         names = loc.get("name") or []
#         if not names:
#             continue

#         # The first name in the list is the "Official" name we want to return
#         canonical_name = names[0]

#         # Map every name in the list to the canonical name
#         for alias in names:
#             # Store as lowercase for case-insensitive matching
#             clean_alias = alias.lower().strip()
#             new_map[clean_alias] = canonical_name

#     LOCATION_MAP = new_map

#     logger.info("[locations] loaded %d location aliases", len(LOCATION_MAP))


# # Build location map at import time
# load_location_map()
